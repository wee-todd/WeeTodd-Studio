"""Audio preparation and durable results for Studio's native SheetSage2 service."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
import time
from pathlib import Path


def validate_range(request):
    start, duration = request.get("source_in", 0), request.get("duration", 30)
    if (
        any(type(x) not in (int, float) or not math.isfinite(x) for x in (start, duration))
        or start < 0
        or not 1 <= duration <= 300
    ):
        raise ValueError(
            "Choose a finite reference range of 1–300 seconds and a nonnegative in-point."
        )
    return start, duration


def validate_source_range(start, duration, total):
    if not math.isfinite(total) or total <= 0 or start + duration > total + 0.01:
        raise ValueError(
            "The reference range extends beyond the audio. Shorten it or move its in-point."
        )


def digest(path, check):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            check()
            h.update(block)
    return h.hexdigest()


def run(command, check):
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            while True:
                check()
                try:
                    out, _ = process.communicate(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if process.returncode:
                errors.seek(0)
                raise ValueError(errors.read(4096).decode(errors="replace"))
            return out
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def analyze(request, output, progress=None, cancelled=None):
    import numpy as np
    from studio_music import _tool

    from sheetsage2_mlx import inspect_checkpoint, transcribe

    def check():
        if cancelled and cancelled():
            raise InterruptedError("Reference analysis cancelled")

    start, duration = validate_range(request)
    source = Path(request.get("audio_path", "")).expanduser().resolve()
    model = request.get("model_path")
    if not source.is_file() or not isinstance(model, str) or not model.strip():
        raise ValueError("Choose reference audio and an installed SheetSage2 model.")
    if type(request.get("melody_only", True)) is not bool:
        raise ValueError("Melody-only must be a boolean.")
    if output is None:
        raise ValueError("Choose a fresh reference analysis output directory.")
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise FileExistsError("Reference analysis needs a new output directory.")
    inspect_checkpoint(model)
    runtime = request.get("runtime", {})
    probe = json.loads(
        run(
            [
                _tool("ffprobe", runtime),
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=duration:format=duration",
                "-of",
                "json",
                str(source),
            ],
            check,
        )
    )
    streams = probe.get("streams", [])
    if not streams:
        raise ValueError("The selected reference has no audio stream.")
    raw_duration = streams[0].get("duration", probe.get("format", {}).get("duration", 0))
    validate_source_range(start, duration, float(raw_duration))
    original_hash = digest(source, check)
    if progress:
        progress(dict(event="progress", message="Preparing reference audio", fraction=0.01))
    raw = run(
        [
            _tool("ffmpeg", runtime),
            "-v",
            "error",
            "-nostdin",
            "-ss",
            str(start),
            "-i",
            str(source),
            "-t",
            str(duration),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            "24000",
            "-f",
            "f32le",
            "pipe:1",
        ],
        check,
    )
    audio = np.frombuffer(raw, dtype="<f4").copy()
    if len(audio) < 24000 or not np.isfinite(audio).all() or float(np.max(np.abs(audio))) < 1e-6:
        raise ValueError("The reference is too short, silent, or contains invalid samples.")
    if abs(len(audio) / 24000 - duration) > 0.05:
        raise ValueError("Decoded reference did not cover the requested range.")
    check()
    started = time.monotonic()
    result = transcribe(
        audio,
        model_path=model,
        melody_only=request.get("melody_only", True),
        progress=(lambda event: progress(dict(event, event="progress"))) if progress else None,
        cancelled=cancelled,
    )
    check()
    if digest(source, check) != original_hash:
        raise ValueError("Reference audio changed during analysis. Analyze the current file again.")
    if not isinstance(result.get("abc"), str) or not result["abc"].strip():
        raise ValueError("No usable score was extracted. Choose a clearer musical reference.")
    result = dict(
        result,
        source_path=str(source),
        source_sha256=original_hash,
        source_in=start,
        duration=duration,
        melody_only=request.get("melody_only", True),
        elapsed=time.monotonic() - started,
    )
    target.mkdir(parents=True, exist_ok=False)
    (target / "score.abc").write_text(result["abc"])
    (target / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return dict(result, artifacts=str(target))
