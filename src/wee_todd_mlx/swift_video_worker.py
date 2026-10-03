"""Weight-free headless handoff to the versioned Swift H3/LTX workers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import select
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path


def validate_swift_video_recipe(document: object, engine: str, output: Path | None = None) -> bool:
    """Identify the dedicated Ripple shape; Swift owns complete model/media admission."""
    if not isinstance(document, dict) or document.get("engine") != engine:
        raise ValueError("Swift worker recipe format or engine does not match the request")
    if document.get("format") == "weetodd-headless-v2":
        return False
    fields = {
        "version", "engine", "task", "gemma_root", "transformer_root", "connector_checkpoint",
        "video_checkpoint", "audio_checkpoint", "adapter_path", "adapter_strength", "guide_path",
        "first_reference_path", "source_path", "source_sha256", "source_start", "duration",
        "editorial_frames", "width", "height", "frames", "fps", "seed", "prompt",
        "reference_strength", "anchors", "audio_policy", "ffmpeg_path", "output_directory",
    }
    if (engine != "ltx25" or set(document) != fields or type(document["version"]) is not int
            or document["version"] != 1 or document["task"] != "ripple"):
        raise ValueError("Select a headless-v2 recipe or complete version-1 LTX Ripple request")
    paths = fields.intersection({"gemma_root", "transformer_root", "connector_checkpoint",
        "video_checkpoint", "audio_checkpoint", "adapter_path", "guide_path",
        "first_reference_path", "source_path", "ffmpeg_path", "output_directory"})
    if not all(isinstance(document[key], str) and document[key].startswith("/")
               and "\0" not in document[key] for key in paths):
        raise ValueError("Ripple request requires absolute local paths")
    if not all(type(document[key]) is int for key in
               ("editorial_frames", "width", "height", "frames", "seed")):
        raise ValueError("Ripple request requires integer geometry and seed")
    if not all(type(document[key]) in (int, float) and math.isfinite(document[key]) for key in
               ("adapter_strength", "source_start", "duration", "fps", "reference_strength")):
        raise ValueError("Ripple request requires finite controls")
    anchors = document["anchors"]
    if (not isinstance(anchors, list) or len(anchors) > 8 or not all(
        isinstance(anchor, dict) and set(anchor) == {"frame", "path", "strength"}
        and type(anchor["frame"]) is int and isinstance(anchor["path"], str)
        and anchor["path"].startswith("/") and "\0" not in anchor["path"]
        and type(anchor["strength"]) in (int, float) and math.isfinite(anchor["strength"])
        for anchor in anchors) or not isinstance(document["audio_policy"], str)
        or document["audio_policy"] not in {"preserve", "silent"}
        or not isinstance(document["prompt"], str) or not document["prompt"].strip()
        or not isinstance(document["source_sha256"], str)
        or len(document["source_sha256"]) != 64
        or any(value not in "0123456789abcdef" for value in document["source_sha256"])):
        raise ValueError("Ripple request has invalid reference/audio/source controls")
    if output is not None and Path(document["output_directory"]).resolve() != output.resolve():
        raise ValueError("Ripple output directory differs from the worker output")
    return True


def run_swift_video_worker(
    *,
    worker: Path,
    engine: str,
    recipe: Path,
    output: Path,
    mode: str,
    on_progress: Callable[[dict], None] = lambda _: None,
    check_interrupted: Callable[[], None] = lambda: None,
) -> dict:
    """Run one immutable recipe; the Swift worker owns admission and inference."""
    if engine not in {"h3", "ltx25"} or mode not in {"preflight", "render"}:
        raise ValueError("Unsupported Swift video engine or mode")
    worker = Path(worker).expanduser().resolve()
    recipe = Path(recipe).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not worker.is_file() or not os.access(worker, os.X_OK):
        raise FileNotFoundError("Select an executable Swift video worker")
    if not recipe.is_file() or not 0 < recipe.stat().st_size <= 1024 * 1024:
        raise ValueError("Swift worker recipe must be a regular JSON file under 1 MiB")
    data = recipe.read_bytes()
    if len(data) > 1024 * 1024:
        raise ValueError("Swift worker recipe grew during preparation")
    document = json.loads(data)
    ripple = validate_swift_video_recipe(document, engine, output)
    if output.exists():
        raise FileExistsError("Swift worker output already exists")
    if not output.parent.is_dir():
        raise FileNotFoundError("Swift worker output parent does not exist")
    envelope = {
        "version": 1,
        "jobID": str(uuid.uuid4()),
        "engine": engine,
        "recipePath": str(recipe),
        "recipeSHA256": hashlib.sha256(data).hexdigest(),
        "outputDirectory": str(output),
    }
    with tempfile.TemporaryDirectory(prefix="weetodd-swift-worker-", dir=output.parent) as scratch:
        request = Path(scratch) / "request.json"
        request.write_text(json.dumps(envelope, separators=(",", ":")))
        with (Path(scratch) / "stderr.log").open("wb") as stderr:
            process = subprocess.Popen(
                [str(worker), mode, "--request", str(request), "--output", str(output)],
                stdout=subprocess.PIPE,
                stderr=stderr,
            )
            try:
                assert process.stdout is not None
                terminal = None
                pending = b""

                def accept_line(line: bytes) -> None:
                    nonlocal terminal
                    if len(line) > 65536:
                        raise RuntimeError("Oversized Swift worker event")
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise RuntimeError("Invalid Swift worker event")
                    if event.get("event") == "progress":
                        on_progress(event)
                    elif "status" in event:
                        if terminal is not None:
                            raise RuntimeError("Duplicate Swift worker completion")
                        terminal = event

                while True:
                    check_interrupted()
                    ready, _, _ = select.select([process.stdout], [], [], 0.25)
                    if not ready:
                        continue
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:
                        if pending:
                            accept_line(pending)
                        break
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        accept_line(line)
                    if len(pending) > 65536:
                        raise RuntimeError("Oversized Swift worker event")
                code = process.wait()
                if code != 0 or terminal is None or terminal.get("status") != "success":
                    detail = (terminal or {}).get("error") or f"exit code {code}"
                    raise RuntimeError(f"Swift worker failed: {detail}")
                result = terminal.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError("Swift worker returned no result")
                try:
                    returned_id = uuid.UUID(str(result.get("jobID")))
                except (TypeError, ValueError) as error:
                    raise RuntimeError(
                        "Swift worker completion has no valid job identity"
                    ) from error
                if (
                    returned_id != uuid.UUID(envelope["jobID"])
                    or result.get("nativeRuntime") != "swift-mlx"
                ):
                    raise RuntimeError("Swift worker completion identity or runtime differs")
                if mode == "render":
                    video = (result.get("video_path", result.get("path")) if ripple
                             else result.get("video"))
                    published = Path(video).resolve() if isinstance(video, str) else None
                    if (
                        published is None
                        or not published.is_file()
                        or not published.is_relative_to(output)
                    ):
                        raise RuntimeError("Swift worker did not publish a video inside its output")
                    if ripple:
                        for key in ("video_path", "path", "audio", "audio_path", "receipt_path",
                                    "artifacts_directory"):
                            value = result.get(key)
                            if value is None:
                                continue
                            candidate = Path(value).resolve() if isinstance(value, str) else None
                            if (candidate is None or not candidate.exists()
                                    or not candidate.is_relative_to(output)):
                                raise RuntimeError("Ripple path is outside its output")
                            if key in {"video_path", "path"} and candidate != published:
                                raise RuntimeError("Ripple worker movie paths disagree")
                        references = result.get("frozen_references", [])
                        if not isinstance(references, list) or not all(
                            isinstance(reference, dict) and isinstance(reference.get("path"), str)
                            and Path(reference["path"]).resolve().is_relative_to(output)
                            and Path(reference["path"]).is_file() for reference in references
                        ):
                            raise RuntimeError("Ripple worker reference paths escape its output")
                return result
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                raise
            finally:
                process.stdout.close()
