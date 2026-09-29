"""Weight-free headless handoff to the versioned Swift H3/LTX workers."""

from __future__ import annotations

import hashlib
import json
import os
import select
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path


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
    if document.get("format") != "weetodd-headless-v2" or document.get("engine") != engine:
        raise ValueError("Swift worker recipe format or engine does not match the request")
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
                    video = result.get("video")
                    published = Path(video).resolve() if isinstance(video, str) else None
                    if (
                        published is None
                        or not published.is_file()
                        or not published.is_relative_to(output)
                    ):
                        raise RuntimeError("Swift worker did not publish a video inside its output")
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
