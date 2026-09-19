"""Non-destructive, sample-exact voice reference preparation shared by speech engines."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path


def digest_file(filename):
    digest = hashlib.sha256()
    with Path(filename).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tool(name, runtime):
    result = runtime.get(name + "Path") or shutil.which(name)
    if not result or not Path(result).is_file():
        raise ValueError(f"Configure {name} in Runtime Settings")
    return str(result)


def run_media(args, *, cancelled=None):
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            while True:
                if cancelled and cancelled():
                    raise InterruptedError("Audio preparation cancelled")
                try:
                    out, error = process.communicate(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if process.returncode:
                raise ValueError(
                    "Audio preparation failed: " + error.decode(errors="replace")[-1000:]
                )
            return out
        except BaseException:
            process.kill()
            process.communicate()
            raise


def prepare_reference(reference, *, engine, mode, model_identity, output, runtime, cancelled=None):
    import numpy as np
    from scipy.io import wavfile

    if engine not in {"fishS2Pro", "qwen3TTS"} or mode not in {
        "audioAndTranscript",
        "speakerIdentityOnly",
    }:
        raise ValueError("Choose a supported reference speech mode")
    if engine == "fishS2Pro" and mode == "speakerIdentityOnly":
        raise ValueError("Fish reference speech requires a transcript")
    transcript = reference.get("transcript", "").strip()
    if mode == "audioAndTranscript" and not transcript:
        raise ValueError("Enter the words spoken in the selected reference range")
    start, duration = reference.get("start", 0), reference.get("duration")
    if (
        any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, duration))
        or start < 0
        or not 0 < duration <= 60
    ):
        raise ValueError("Choose a finite reference range up to 60 seconds")
    source = Path(reference.get("path", "")).expanduser().resolve()
    if not source.is_file():
        raise ValueError("Relink the missing reference audio")
    source_hash = digest_file(source)
    expected = reference.get("sourceHash")
    if expected and expected != source_hash:
        raise ValueError("Reference audio changed; review and reselect the sample")
    rate = 44100 if engine == "fishS2Pro" else 24000
    frames = round(duration * rate)
    probe = json.loads(
        run_media(
            [
                tool("ffprobe", runtime),
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=channels,duration:format=duration",
                "-of",
                "json",
                str(source),
            ],
            cancelled=cancelled,
        )
    )
    streams = probe.get("streams", [])
    if not streams:
        raise ValueError("The reference has no audio stream")
    length = float(streams[0].get("duration", probe.get("format", {}).get("duration", 0)))
    if not math.isfinite(length) or start + duration > length + 1 / rate:
        raise ValueError("The selected reference range extends past the source")
    channel = reference.get("channel", "mix")
    if channel not in {"mix", "left", "right"}:
        raise ValueError("Choose mix, left or right reference channel")
    channels = int(streams[0]["channels"])
    if channels not in (1, 2) or (channel == "right" and channels < 2):
        raise ValueError("Use a valid mono or stereo reference channel")
    identity = dict(
        version=1,
        source_sha256=source_hash,
        start=start,
        frames=frames,
        sample_rate=rate,
        channel=channel,
        transcript=transcript,
        engine=engine,
        mode=mode,
        model_identity=model_identity,
    )
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    folder = Path(output).expanduser().resolve() / key
    target, receipt = folder / "reference.wav", folder / "reference.json"
    if target.is_file() and receipt.is_file():
        try:
            result = json.loads(receipt.read_text())
            if result["sha256"] == digest_file(target):
                return result
        except (ValueError, OSError, KeyError):
            pass
    folder.mkdir(parents=True, exist_ok=True)
    pan = (
        "pan=mono|c0=c0"
        if channels == 1 or channel == "left"
        else "pan=mono|c0=c1"
        if channel == "right"
        else "pan=mono|c0=0.5*c0+0.5*c1"
    )
    filters = (f"atrim=start={start}:duration={duration},asetpts=PTS-STARTPTS,"
               f"{pan},aresample={rate},atrim=end_sample={frames}")
    raw = run_media(
        [
            tool("ffmpeg", runtime),
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-af",
            filters,
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-",
        ],
        cancelled=cancelled,
    )
    pcm = np.frombuffer(raw, dtype="<f4").copy()
    if len(pcm) != frames or not np.isfinite(pcm).all() or np.max(np.abs(pcm), initial=0) < 1e-6:
        raise ValueError(
            "Reference must contain finite, audible samples for the complete selected range"
        )
    if digest_file(source) != source_hash:
        raise ValueError("Reference changed while preparing; try again")
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=folder) as tmp:
        pending = Path(tmp) / "reference.wav"
        wavfile.write(pending, rate, pcm)
        result = dict(
            path=str(target),
            key=key,
            frames=frames,
            sample_rate=rate,
            channels=1,
            sha256=digest_file(pending),
            identity=identity,
            transcript=transcript,
            peak=float(np.max(np.abs(pcm))),
        )
        pending.replace(target)
        pending_json = Path(tmp) / "reference.json"
        pending_json.write_text(json.dumps(result, indent=2, allow_nan=False))
        pending_json.replace(receipt)
    return result
