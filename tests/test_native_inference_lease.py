import pytest

from wee_todd_mlx.inference_lease import InferenceLease


def test_waiting_for_another_weighted_job_can_be_cancelled(tmp_path):
    file = tmp_path / "native.lock"
    with InferenceLease(file):
        with pytest.raises(InterruptedError):
            with InferenceLease(file, cancel=lambda: True):
                pytest.fail("Cancelled admission must not enter")
    with InferenceLease(file):
        pass


def test_killed_process_releases_lease(tmp_path):
    import subprocess
    import sys

    lock = tmp_path / "process.lock"
    code = (
        "import sys,time; from wee_todd_mlx.inference_lease import InferenceLease; "
        "lease=InferenceLease(sys.argv[1]); lease.__enter__(); "
        'print("ready",flush=True); time.sleep(60)'
    )
    import os
    from pathlib import Path

    child = subprocess.Popen(
        [sys.executable, "-c", code, str(lock)],
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        child.kill()
        child.wait(timeout=5)
        with InferenceLease(lock):
            pass
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_shared_video_and_text_boundaries_acquire_admission(monkeypatch, tmp_path):
    import render_headless

    from wee_todd_mlx import inference_lease
    from wee_todd_remote import client

    events = []

    class Lease:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            events.append("acquire")

        def __exit__(self, *args):
            events.append("release")

    monkeypatch.setattr(inference_lease, "InferenceLease", Lease)
    monkeypatch.setattr(render_headless, "render_h3", lambda *a: events.append("video") or {})
    render_headless.execute_native_render({"engine": "h3"}, tmp_path / "render.mp4")
    monkeypatch.setattr(client, "_invoke_helper", lambda *a, **k: iter([{"type": "result"}]))
    assert list(
        client.invoke_helper("text", {}, helper=tmp_path / "unused", cancelled=lambda: False)
    )
    assert events == ["acquire", "video", "release", "acquire", "release"]
