"""Studio/headless music actions; inference remains in WeeTodd's native engine."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _tool(name, runtime):
    candidate = runtime.get(name + "Path") or shutil.which(name)
    if not candidate or not Path(candidate).is_file():
        raise ValueError(f"Configure {name} in Runtime Settings to prepare music excerpts.")
    return str(candidate)


def _digest(source):
    digest = hashlib.sha256()
    with Path(source).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_excerpt(request, output, runtime):
    """Materialize an exact source interval without changing the master or stereo layout."""
    start, duration = request.get("source_in", 0), request.get("duration")
    if (
        any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in (start, duration)
        )
        or start < 0
        or duration <= 0
        or duration > 3600
    ):
        raise ValueError("Choose a finite source range with positive duration up to one hour.")
    source = Path(request.get("path", "")).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("The source music file is unavailable; relink the audio asset.")
    probe = json.loads(
        subprocess.check_output(
            [
                _tool("ffprobe", runtime),
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,channels,duration,duration_ts,time_base",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(source),
            ]
        )
    )
    streams = probe.get("streams", [])
    if not streams:
        raise ValueError("The source has no audio stream.")
    stream = streams[0]
    rate, channels = int(stream["sample_rate"]), int(stream["channels"])
    if channels not in (1, 2) or rate <= 0:
        raise ValueError("Use mono or stereo music for audio-driven video.")
    total = float(stream.get("duration", probe.get("format", {}).get("duration", 0)))
    first, count = round(start * rate), round(duration * rate)
    if count < 1 or first + count > round(total * rate):
        raise ValueError(
            "The requested excerpt extends beyond the song. Shorten it or move its in point."
        )
    source_hash = _digest(source)
    identity = dict(
        source_sha256=source_hash,
        source_start_sample=first,
        frames=count,
        sample_rate=rate,
        channels=channels,
        format="pcm_f32le-v1",
    )
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = Path(output).expanduser().resolve() / key
    target, receipt = folder / "audio.wav", folder / "excerpt.json"
    result = dict(path=str(target), source_path=str(source), duration=count / rate, **identity)
    if target.is_file() and receipt.is_file():
        try:
            saved = json.loads(receipt.read_text())
            if saved["result"] == result and saved["sha256"] == _digest(target):
                return result
        except (KeyError, ValueError, OSError):
            pass
    folder.mkdir(parents=True, exist_ok=True)
    # Temporary output and receipt are published only after a successful, verified decode.
    with tempfile.TemporaryDirectory(prefix="music-excerpt-", dir=folder) as temporary:
        pending = Path(temporary) / "audio.wav"
        subprocess.run(
            [
                _tool("ffmpeg", runtime),
                "-v",
                "error",
                "-nostdin",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-af",
                f"atrim=start_sample={first}:end_sample={first + count},asetpts=PTS-STARTPTS",
                "-c:a",
                "pcm_f32le",
                str(pending),
            ],
            check=True,
            capture_output=True,
        )
        inspected = json.loads(
            subprocess.check_output(
                [
                    _tool("ffprobe", runtime),
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=sample_rate,channels,duration_ts,time_base",
                    "-of",
                    "json",
                    str(pending),
                ]
            )
        )["streams"][0]
        numerator, denominator = map(int, inspected["time_base"].split("/"))
        samples = round(int(inspected["duration_ts"]) * numerator / denominator * rate)
        if (
            samples != count
            or int(inspected["channels"]) != channels
            or int(inspected["sample_rate"]) != rate
        ):
            raise ValueError("Prepared music excerpt failed its sample-accurate audio check.")
        pending_receipt = Path(temporary) / "excerpt.json"
        pending_receipt.write_text(
            json.dumps(dict(result=result, sha256=_digest(pending)), indent=2)
        )
        os.replace(pending, target)
        os.replace(pending_receipt, receipt)
    return result


def driver_excerpt_request(request):
    """Resolve actual model frame duration rather than assuming the timeline's rounded length."""
    if "video_request" not in request:
        return request
    from studio_bridge import ltx_coverage_duration, resolve_clip_generation

    from wee_todd_mlx.task_conditioning import frame_geometry

    video = copy.deepcopy(request["video_request"])
    video["runtime"] = request.get("runtime", video.get("runtime", {}))
    clip = next(c for c in video["project"]["clips"] if c["id"] == video["clipID"])
    clip["generationSelection"] = dict(clip.get("generationSelection") or {}, task="a2v")
    clip["profileID"] = "auto"
    resolved = resolve_clip_generation(video, clip, validate_inputs=False)
    recipe = resolved["recipe"]
    recipe["config"]["duration_seconds"] = clip["duration"]
    if (recipe["engine"] in {"ltx23", "ltx25"}
            and recipe["config"].get("duration_mode", "manual") == "manual"):
        recipe["config"]["duration_seconds"] = ltx_coverage_duration(
            clip["duration"], recipe["config"].get("frame_rate", 24))
    frames, fps = frame_geometry(recipe)
    return {**request, "duration": frames / fps}


def dispatch(command, request, output=None, progress=None, cancelled=None):
    if command == "music-analysis-setup":
        from wee_todd_mlx.audio_analysis.setup import download

        if not isinstance(request.get("directory"), str) or not request["directory"].strip():
            raise ValueError("Choose a library folder for analysis models")
        return download(
            request["directory"],
            progress=progress,
            cancelled=cancelled,
            include_vocals=request.get("include_vocals", False),
        )
    if command == "music-analyze":
        from wee_todd_mlx.audio_analysis.service import analyze
        from wee_todd_mlx.audio_analysis.structure import editing_cues

        if output is None:
            raise ValueError("Choose an analysis output folder")

        def check():
            if cancelled and cancelled():
                raise InterruptedError("Audio analysis cancelled")

        analysis = analyze(
            request["audio_path"],
            expected_sha256=request["audio_sha256"],
            model_directory=request.get("analysis_model_directory", ""),
            lyrics=request.get("lyrics", ""),
            lyrics_status=request.get("lyrics_status", "unknown"),
            mode=request.get("analysis_mode", "neural"),
            vocal_mode=request.get("analysis_vocal_mode", "mixed"),
            cache_directory=Path(output).parent / "music-analysis-cache",
            check=check,
        )
        cues = editing_cues(
            analysis,
            fps=request.get("frame_rate", 24),
            source_start=request.get("source_start_seconds", 0),
            duration=request.get("duration_seconds"),
        )
        return {"analysis": analysis, "editing_cues": cues}
    if command == "music-download":
        from wee_todd_mlx.music_setup import download

        destination = request.get("directory")
        if not isinstance(destination, str) or not destination.strip():
            raise ValueError("Choose a model library folder for the music download.")
        return download(destination, progress=progress)
    if command == "music-excerpt":
        if output is None:
            raise ValueError("Choose an output directory for the music excerpt.")
        return prepare_excerpt(driver_excerpt_request(request), output, request.get("runtime", {}))
    if command in ("music-resynthesize", "music-decode"):
        from yue2_mlx.pipeline import decode_latents, resynthesize

        source = request.get("source_artifacts")
        if not isinstance(source, str) or not source.strip() or output is None:
            raise ValueError("Choose a saved music take and a fresh output directory.")
        if command == "music-decode":
            return decode_latents(source, Path(output), progress=progress, cancelled=cancelled)
        return resynthesize(
            source,
            Path(output),
            steps=request.get("steps"),
            seed=request.get("seed"),
            progress=progress,
            cancelled=cancelled,
        )
    from yue2_mlx.config import validate_request

    normalized = validate_request(request.get("music", {}))
    if command in ("music-generate", "music-plan"):
        if output is None:
            raise ValueError("Choose a new output directory for the music take.")
        from yue2_mlx.pipeline import generate, plan

        operation = plan if command == "music-plan" else generate
        return operation(normalized, Path(output), progress=progress, cancelled=cancelled)
    if command == "music-inspect":
        # Full checkpoint inspection is wired to the native loader, never a foreign runtime.
        from yue2_mlx.checkpoint import inspect_checkpoint

        return inspect_checkpoint(
            normalized["model_path"], normalized.get("vae_path"), normalized["precision"]
        )
    raise ValueError(f"Unknown music command: {command}")
