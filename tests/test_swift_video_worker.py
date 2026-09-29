"""The headless host must hand an immutable recipe to the Swift video worker."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from wee_todd_mlx.swift_video_worker import run_swift_video_worker


def fake_worker(tmp_path: Path) -> Path:
    worker = tmp_path / "worker"
    worker.write_text(
        f"#!{sys.executable}\n"
        "import hashlib, json, pathlib, sys\n"
        "mode, flag, request, output_flag, output = sys.argv[1:]\n"
        "assert flag == '--request' and output_flag == '--output'\n"
        "envelope = json.loads(pathlib.Path(request).read_text())\n"
        "recipe = pathlib.Path(envelope['recipePath']).read_bytes()\n"
        "assert envelope['recipeSHA256'] == hashlib.sha256(recipe).hexdigest()\n"
        "assert envelope['outputDirectory'] == output\n"
        "print(json.dumps({'event': 'progress', 'fraction': 0.5, "
        "'message': 'sampling'}), flush=True)\n"
        "if mode == 'render':\n"
        "    target = pathlib.Path(output)\n"
        "    target.mkdir()\n"
        "    (target / 'render.mp4').write_bytes(b'media')\n"
        "    result = {'video': str(target / 'render.mp4'), 'nativeRuntime': 'swift-mlx', "
        "'jobID': envelope['jobID']}\n"
        "else:\n"
        "    result = {'task': 't2v', 'nativeRuntime': 'swift-mlx', "
        "'jobID': envelope['jobID']}\n"
        "print(json.dumps({'status': 'success', 'result': result}), flush=True)\n"
    )
    worker.chmod(0o700)
    return worker


def test_headless_swift_worker_preserves_recipe_identity_and_progress(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"format": "weetodd-headless-v2", "engine": "ltx25"}))
    worker = fake_worker(tmp_path)
    output = tmp_path / "take"
    events = []

    preflight = run_swift_video_worker(
        worker=worker, engine="ltx25", recipe=recipe, output=output,
        mode="preflight", on_progress=events.append,
    )
    assert preflight["task"] == "t2v"
    assert not output.exists()

    result = run_swift_video_worker(
        worker=worker, engine="ltx25", recipe=recipe, output=output,
        mode="render", on_progress=events.append,
    )
    assert result["video"] == str(output / "render.mp4")
    assert [event["message"] for event in events] == ["sampling", "sampling"]
    assert not list(tmp_path.glob("*.envelope.json"))


def test_swift_worker_rejects_mismatched_engine_before_launch(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"format": "weetodd-headless-v2", "engine": "h3"}))
    with pytest.raises(ValueError, match="engine"):
        run_swift_video_worker(
            worker=fake_worker(tmp_path), engine="ltx25", recipe=recipe,
            output=tmp_path / "take", mode="preflight",
        )
    assert not (tmp_path / "take").exists()


def test_headless_render_uses_swift_and_keeps_movie_path(tmp_path: Path) -> None:
    from render_headless import execute_native_render

    output = tmp_path / "render.mp4"
    result = execute_native_render(
        {"format": "weetodd-headless-v2", "engine": "ltx25", "prompt": "test"},
        output, swift_worker=fake_worker(tmp_path),
    )
    assert result["video"] == str(output)
    assert output.read_bytes() == b"media"
    assert result["runtime_loaded"] == [False]


def test_headless_cli_can_select_swift_worker_without_python_inference(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"format": "weetodd-headless-v2", "engine": "ltx25",
                                  "candidate": "swift-smoke", "components": {}, "prompt": "test"}))
    output = tmp_path / "job"
    command = [sys.executable, "scripts/render_headless.py", "--recipe", str(recipe),
               "--output-directory", str(output), "--swift-worker", str(fake_worker(tmp_path))]
    run = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    assert (output / "render.mp4").read_bytes() == b"media"
    report = json.loads((output / "result.json").read_text())
    assert report["status"] == "success"
    assert report["native_runtime"] == "swift-mlx"


def test_headless_cli_sigint_cancels_worker_without_traceback_or_take(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"format": "weetodd-headless-v2", "engine": "ltx25",
                                  "candidate": "cancel-smoke", "components": {}, "prompt": "test"}))
    marker = tmp_path / "worker.pid"
    worker = tmp_path / "waiting-worker"
    worker.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, time\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        "print('{\"event\":\"progress\",\"message\":\"sampling\"}', flush=True)\n"
        "time.sleep(30)\n"
    )
    worker.chmod(0o700)
    output = tmp_path / "job"
    process = subprocess.Popen(
        [sys.executable, "scripts/render_headless.py", "--recipe", str(recipe),
         "--output-directory", str(output), "--swift-worker", str(worker)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists(), "Fake worker never reached weighted work"
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 130
        assert "Traceback" not in stderr
        assert json.loads((output / "result.json").read_text())["status"] == "cancelled"
        assert not (output / "render.mp4").exists()
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_swift_worker_rejects_result_that_escapes_output_directory(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"format": "weetodd-headless-v2", "engine": "ltx25"}))
    worker = fake_worker(tmp_path)
    worker.write_text(worker.read_text().replace(
        "str(target / 'render.mp4')", "str(target / '..' / 'escape.mp4')"))
    (tmp_path / "escape.mp4").write_bytes(b"unrelated")
    with pytest.raises(RuntimeError, match="inside its output"):
        run_swift_video_worker(
            worker=worker, engine="ltx25", recipe=recipe,
            output=tmp_path / "take", mode="render",
        )


def test_swift_worker_rejects_wrong_completion_identity(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"format": "weetodd-headless-v2", "engine": "h3"}))
    worker = fake_worker(tmp_path)
    worker.write_text(worker.read_text().replace(
        "'jobID': envelope['jobID']", "'jobID': '00000000-0000-0000-0000-000000000001'"
    ))
    with pytest.raises(RuntimeError, match="identity"):
        run_swift_video_worker(
            worker=worker, engine="h3", recipe=recipe,
            output=tmp_path / "take", mode="preflight",
        )


def test_swift_worker_cancels_even_when_worker_has_not_emitted_progress(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"format": "weetodd-headless-v2", "engine": "h3"}))
    worker = tmp_path / "silent-worker"
    worker.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(30)\n")
    worker.chmod(0o700)
    checks = 0

    def check_interrupted():
        nonlocal checks
        checks += 1
        if checks > 2:
            raise InterruptedError("ComfyUI cancelled")

    with pytest.raises(InterruptedError, match="cancelled"):
        run_swift_video_worker(
            worker=worker, engine="h3", recipe=recipe, output=tmp_path / "take",
            mode="render", check_interrupted=check_interrupted,
        )
    assert checks > 2
    assert not (tmp_path / "take").exists()


def test_exported_job_preflight_forwards_selected_swift_worker(tmp_path: Path, monkeypatch) -> None:
    import studio_job

    worker = fake_worker(tmp_path)
    job = {
        "project": {"clips": [{"id": "clip", "name": "Clip", "engine": "ltx25",
                                 "duration": 1, "sourceIn": 0, "sourcePath": "unused.mp4"}],
                    "assets": [], "audio": [], "titles": [], "settings": {}},
        "runtime": {},
        "recipes": {"clip": {"recipe": {"format": "weetodd-headless-v2",
                                         "engine": "ltx25", "prompt": "test"}}},
        "execution": {},
    }
    commands = []
    monkeypatch.setattr(studio_job.bridge, "preflight_finishing", lambda *_: None)
    monkeypatch.setattr(studio_job.bridge, "run", lambda command: commands.append(command))
    monkeypatch.setattr("wee_todd_mlx.studio_continuity.verify_source", lambda *_: None)
    studio_job.preflight(job, tmp_path / "out", swift_workers={"ltx25": worker})
    assert commands and commands[0][-2:] == ["--swift-worker", str(worker)]


def test_exported_job_swift_selection_cannot_fall_back_to_python(tmp_path: Path) -> None:
    import studio_job

    job = {
        "project": {"clips": [], "assets": [], "audio": [], "titles": [], "settings": {}},
        "runtime": {}, "recipes": {
            "ltx-clip": {"recipe": {"engine": "ltx25"}},
            "h3-clip": {"recipe": {"engine": "h3"}},
        }, "execution": {},
    }
    with pytest.raises(ValueError, match="missing a Swift worker for h3"):
        studio_job.preflight(job, tmp_path / "out",
                             swift_workers={"ltx25": fake_worker(tmp_path)})


def test_bridge_render_forwards_swift_worker_without_python_sampler(
    tmp_path: Path, monkeypatch
) -> None:
    import studio_bridge

    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"format": "weetodd-headless-v2", "engine": "h3"}))
    destination = tmp_path / "render"
    commands = []

    def run(command, **_):
        commands.append(command)
        destination.mkdir()
        (destination / "result.json").write_text(json.dumps({"status": "success", "video": "ok"}))

    monkeypatch.setattr(studio_bridge, "run", run)
    worker = fake_worker(tmp_path)
    studio_bridge.render(str(recipe), destination, swift_worker=worker)
    assert commands[0][-2:] == ["--swift-worker", str(worker)]


def test_exported_job_resume_rejects_changed_swift_backend(tmp_path: Path, monkeypatch) -> None:
    import studio_job

    job = {
        "manifestSHA256": "fixture", "project": {"clips": [], "assets": [],
            "audio": [], "titles": [], "settings": {}}, "runtime": {},
        "recipes": {}, "execution": {},
    }
    monkeypatch.setattr(studio_job, "preflight", lambda *_, **__: {})
    output = tmp_path / "output"
    studio_job.execute(job, output, resume=False)
    with pytest.raises(ValueError, match="Swift worker selection changed"):
        studio_job.execute(job, output, resume=True,
                           swift_workers={"h3": fake_worker(tmp_path)})


def test_exported_job_uses_recorded_studio_swift_selection(tmp_path: Path) -> None:
    import studio_job

    worker = fake_worker(tmp_path)
    job = {"runtime": {"nativeLTX25Enabled": True, "ltx25WorkerPath": str(worker)},
           "recipes": {"clip": {"recipe": {"engine": "ltx25"}}}}
    assert studio_job.selected_swift_workers(job, {}) == {"ltx25": worker}
    job["runtime"]["nativeLTX25Enabled"] = False
    assert studio_job.selected_swift_workers(job, {}) == {}
    assert studio_job.selected_swift_workers(job, {"ltx25": worker}) == {"ltx25": worker}
    job["runtime"]["nativeLTX25Enabled"] = True
    job["runtime"]["ltx25WorkerPath"] = ""
    with pytest.raises(ValueError, match="enabled.*worker path"):
        studio_job.selected_swift_workers(job, {})
