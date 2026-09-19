"""Bounded, deterministic movie reference preparation; no models or AI calls.

Derived media and provenance stay visible beside the original. Cache entries are
content addressed and checked before reuse. No video is silently treated as a guide.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


def _digest(source):
    with Path(source).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_reference(request, destination, runtime):
    from studio_bridge import executable, inspect_media, run

    method = request.get("method")
    if method not in {"sheet", "canny"}:
        raise ValueError("Choose a reference sheet or Canny motion guide.")
    duration = request.get("duration", 5)
    if type(duration) not in {int, float} or not math.isfinite(duration) or not 0 < duration <= 60:
        raise ValueError("Reference guide duration must be positive and at most 60 seconds.")
    dimensions = {key: request.get(key, default) for key, default in
                  (("width", 768), ("height", 448), ("fps", 24))}
    for key, value in dimensions.items():
        if type(value) is not int or not (1 <= value <= (60 if key == "fps" else 2048)):
            raise ValueError(f"Invalid reference {key}; use up to 2048 pixels and 60 fps.")
    width, height, fps = (dimensions[k] for k in ("width", "height", "fps"))
    if width % 2 or height % 2:
        raise ValueError("Reference guide dimensions must be even.")
    source = Path(request["path"]).expanduser().resolve()
    media = inspect_media(source, runtime)
    if media["kind"] != "video" or not math.isfinite(media["duration"]) or media["duration"] <= 0:
        raise ValueError("Choose a movie with a known positive duration.")
    source_hash = _digest(source)
    specification = dict(version=1, method=method, sourceSHA256=source_hash,
                         duration=duration, **dimensions)
    identity = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    directory = Path(destination).resolve() / identity
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "reference.json"
    output = directory / ("reference-sheet.png" if method == "sheet" else "canny-guide.mp4")
    if manifest.is_file() and manifest.stat().st_size < 64 * 1024 and output.is_file():
        try:
            cached = json.loads(manifest.read_text())
            if (cached.get("specification") == specification and
                    cached.get("outputSHA256") == _digest(output) and
                    cached["media"]["path"] == str(output)):
                return cached["media"]
        except (ValueError, KeyError, TypeError):
            pass
    ffmpeg = executable("ffmpeg", runtime)
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=directory) as temporary:
        work = Path(temporary)
        prepared = work / output.name
        if method == "sheet":
            from PIL import Image, ImageDraw

            # Six independent seeks avoid decoding an entire long source movie.
            sheet = Image.new("RGB", (1152, 480), "#17191c")
            draw = ImageDraw.Draw(sheet)
            for index in range(6):
                seconds = max(0, media["duration"] - 1 / max(media["fps"], 1)) * index / 5
                frame = work / f"frame-{index}.png"
                run([ffmpeg, "-v", "error", "-nostdin", "-ss", str(seconds), "-i", str(source),
                     "-map", "0:v:0", "-frames:v", "1", "-vf",
                     "scale=384:216:force_original_aspect_ratio=decrease,pad=384:216:(ow-iw)/2:(oh-ih)/2",
                     "-threads", "1", str(frame)])
                with Image.open(frame) as image:
                    sheet.paste(image.convert("RGB"), ((index % 3) * 384, (index // 3) * 240))
                draw.text(((index % 3) * 384 + 8, (index // 3) * 240 + 220),
                          f"{index + 1} · {seconds:.2f}s", fill="white")
            sheet.save(prepared)
        else:
            frames = min(math.floor(min(duration, media["duration"]) * fps), 3600)
            if frames < 9:
                raise ValueError("A motion guide needs nine frames. Choose a longer movie.")
            # The renderer trims to its causal clock. Preserve time; never loop or stretch motion.
            frames = 1 + 8 * ((frames - 1) // 8)
            filters = (f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase,"
                       f"crop={width}:{height},format=gray,"
                       "edgedetect=low=0.1:high=0.4:mode=wires,format=yuv420p")
            run([ffmpeg, "-v", "error", "-nostdin", "-i", str(source), "-map", "0:v:0", "-an",
                 "-vf", filters, "-frames:v", str(frames), "-c:v", "libx264", "-preset", "fast",
                 "-crf", "12", "-threads", "2", str(prepared)])
        if _digest(source) != source_hash:
            raise ValueError("The source movie changed during preparation. Try again.")
        info = inspect_media(prepared, runtime)
        info.update(path=str(output), sourceSHA256=source_hash, referenceMethod=method)
        record = dict(specification=specification, outputSHA256=_digest(prepared), media=info)
        os.replace(prepared, output)
        pending = work / "reference.json"
        pending.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
        os.replace(pending, manifest)
        return info
