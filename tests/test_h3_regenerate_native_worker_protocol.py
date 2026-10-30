"""Compiled Swift H3 regeneration lifecycle checks; no model loading is needed."""

import fcntl
import hashlib
import json
import os
import select
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest


class _RedactedEnvironment(dict):
    def __repr__(self):
        return "<worker environment redacted>"


@pytest.fixture
def native_request(tmp_path):
    binary = os.environ.get("WEETODD_H3_REGENERATE_PROTOCOL_BINARY")
    fixture = os.environ.get("WEETODD_H3_REGENERATE_PROTOCOL_JOB")
    if not binary or not fixture:
        pytest.skip("Set the compiled H3 worker and valid job fixture for process qualification")
    job = json.loads(Path(fixture).read_text())
    # Deliberately absent: a preflight must reject without waiting for the lease;
    # a render must acquire the lease before reaching any component metadata.
    job["qwen"]["tokenizer"]["path"] = str(tmp_path / "absent-tokenizer.json")
    payload = json.dumps(job).encode()
    recipe = tmp_path / "job.json"
    recipe.write_bytes(payload)
    output = tmp_path / "output"
    envelope = tmp_path / "envelope.json"
    envelope.write_text(
        json.dumps(
            dict(
                version=1,
                jobID=str(uuid.uuid4()),
                engine="h3",
                recipePath=str(recipe),
                recipeSHA256=hashlib.sha256(payload).hexdigest(),
                outputDirectory=str(output),
            )
        )
    )
    runtime = tmp_path / "Runtime"
    runtime.mkdir()
    with (runtime / "native-inference.lock").open("w") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = _RedactedEnvironment(os.environ, WEETODD_STUDIO_DATA=str(tmp_path))
        yield binary, envelope, output, env


def command(request, mode="render"):
    binary, envelope, output, _ = request
    return [binary, mode, "--request", str(envelope), "--output", str(output)]


def wait_line(child, timeout=10):
    assert select.select([child.stdout], [], [], timeout)[0], "Worker produced no bounded progress"
    line = child.stdout.readline()
    assert line, "Worker exited before lease waiting"
    return json.loads(line)


def test_render_waits_before_assets_and_sigterm_releases_without_output(native_request):
    child = subprocess.Popen(
        command(native_request),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=native_request[3],
    )
    try:
        assert wait_line(child).get("stage") == "waiting"
        child.send_signal(signal.SIGTERM)
        output, errors = child.communicate(timeout=5)
        assert child.returncode == 130, errors + output
        assert json.loads(output.splitlines()[-1])["status"] == "cancelled"
        assert not native_request[2].exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_preflight_does_not_wait_for_weighted_inference_lock(native_request):
    result = subprocess.run(
        command(native_request, "preflight"),
        capture_output=True,
        text=True,
        env=native_request[3],
        timeout=5,
    )
    assert result.returncode == 1
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events[-1]["status"] == "error"
    assert all(event.get("stage") != "waiting" for event in events)
    assert not native_request[2].exists()


def test_parent_death_cancels_worker_while_lease_waiting(native_request, tmp_path):
    log = tmp_path / "child.jsonl"
    wrapper = (
        "import subprocess,sys,time\n"
        "p=subprocess.Popen(sys.argv[2:],stdout=open(sys.argv[1],'w'),"
        "stderr=subprocess.DEVNULL)\n"
        "print(p.pid,flush=True)\ntime.sleep(30)\n"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", wrapper, str(log), *command(native_request)],
        stdout=subprocess.PIPE,
        text=True,
        env=native_request[3],
    )
    pid = int(parent.stdout.readline())
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not log.read_text():
            time.sleep(0.05)
        assert json.loads(log.read_text().splitlines()[0]).get("stage") == "waiting"
        parent.terminate()
        parent.wait(timeout=5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and '"cancelled"' not in log.read_text():
            time.sleep(0.05)
        assert '"cancelled"' in log.read_text()
        assert not native_request[2].exists()
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
