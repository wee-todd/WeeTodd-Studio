"""Exercise the compiled native worker with real process/pipe/lease ownership."""

import fcntl
import hashlib
import json
import os
import select
import signal
import subprocess
import uuid
from pathlib import Path

import pytest


@pytest.fixture
def binary():
    value = os.environ.get("WEETODD_FLASHVSR_WORKER_BINARY")
    if not value:
        pytest.skip("Set WEETODD_FLASHVSR_WORKER_BINARY for compiled native protocol qualification")
    executable = Path(value)
    assert executable.is_file() and os.access(executable, os.X_OK)
    return str(executable)


def request(root, *, owner=None, lease=None, output=None):
    output = output or root / "owned-output"
    recipe = dict(
        version=1,
        engine="flashvsr",
        sourceID="b" * 64,
        inputRGBPath=str(root / "missing-source.rgb"),
        inputSHA256="a" * 64,
        outputRGBPath=str(output / "restored.rgb"),
        scratchDirectory=str(output / "scratch"),
        modelRoot=str(root / "models"),
        width=64,
        height=64,
        frames=3,
        fpsNumerator=30000,
        fpsDenominator=1001,
        temporalCompression=4,
        variant="turbo",
        decoder="tiny",
        seed=42,
        colorFix="off",
        waveletLevels=5,
        temporalSigma=1,
        maximumActiveBytes=4 * 1024**3,
        ownerPID=owner or os.getpid(),
        leasePath=str(lease) if lease else None,
        sampling=dict(
            version=1,
            sigmas=[1, 0],
            audioSigmas=[],
            prediction="flowVelocity",
            timeBasis="sigmaTimes1000",
            updateConvention="subtractVelocity",
            audioClock="sourcePassthrough",
            guidance=1,
        ),
    )
    recipe_file = root / (str(uuid.uuid4()) + ".recipe.json")
    payload = json.dumps(recipe).encode()
    recipe_file.write_bytes(payload)
    envelope = dict(
        version=1,
        jobID=str(uuid.uuid4()),
        engine="flashvsr",
        recipePath=str(recipe_file),
        recipeSHA256=hashlib.sha256(payload).hexdigest(),
        outputDirectory=str(output),
        ffmpegPath=None,
    )
    envelope_file = root / (str(uuid.uuid4()) + ".envelope.json")
    envelope_file.write_text(json.dumps(envelope))
    return envelope_file, recipe_file, output


def command(binary, mode, envelope, output):
    return [binary, mode, "--request", str(envelope), "--output", str(output)]


def result(process):
    lines = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    assert lines
    return lines[-1]


def test_native_plan_is_metadata_only_and_preserves_frozen_clock(binary, tmp_path):
    envelope, _, output = request(tmp_path)
    process = subprocess.run(
        command(binary, "plan", envelope, output), capture_output=True, text=True, timeout=10
    )
    assert process.returncode == 0, process.stderr + process.stdout
    body = result(process)
    assert body["status"] == "success" and body["result"]["planOnly"] is True
    assert body["result"]["nativeRuntime"] == "swift-mlx"
    assert body["result"]["memory"]["temporaryBytes"] > 0
    assert not output.exists()
    assert not (tmp_path / "missing-source.rgb").exists()
    assert not (tmp_path / "models").exists()


def test_changed_envelope_recipe_and_existing_output_reject_without_mutation(binary, tmp_path):
    envelope, recipe_file, output = request(tmp_path)
    recipe_file.write_bytes(recipe_file.read_bytes() + b" ")
    changed = subprocess.run(
        command(binary, "plan", envelope, output), capture_output=True, text=True, timeout=10
    )
    assert changed.returncode == 1 and result(changed)["status"] == "error"
    assert not output.exists()
    envelope, _, output = request(tmp_path)
    output.mkdir()
    marker = output / "user-work"
    marker.write_bytes(b"preserve")
    exists = subprocess.run(
        command(binary, "plan", envelope, output), capture_output=True, text=True, timeout=10
    )
    assert exists.returncode == 1 and result(exists)["status"] == "error"
    assert marker.read_bytes() == b"preserve"


def test_dead_owner_cancels_before_output_or_model_loading(binary, tmp_path):
    envelope, _, output = request(tmp_path, owner=1 << 30)
    process = subprocess.run(
        command(binary, "render", envelope, output), capture_output=True, text=True, timeout=10
    )
    assert process.returncode == 130, process.stderr + process.stdout
    assert result(process)["status"] == "cancelled"
    assert not output.exists()


def waiting_worker(binary, root, owner=None):
    lease = root / "inference.lock"
    held = lease.open("w+b")
    fcntl.flock(held, fcntl.LOCK_EX)
    envelope, _, output = request(root, owner=owner, lease=lease)
    child = subprocess.Popen(
        command(binary, "render", envelope, output),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert select.select([child.stdout], [], [], 5)[0], "Worker did not report lease wait"
        line = json.loads(child.stdout.readline())
        assert line["stage"] == "waiting"
    except BaseException:
        child.kill()
        child.wait()
        held.close()
        raise
    return child, held, output


def test_sigterm_while_waiting_releases_process_without_loading_models(binary, tmp_path):
    child, held, output = waiting_worker(binary, tmp_path)
    try:
        child.send_signal(signal.SIGTERM)
        stdout, stderr = child.communicate(timeout=5)
        assert child.returncode == 130, stderr + stdout
        assert json.loads(stdout.splitlines()[-1])["status"] == "cancelled"
        assert not output.exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        held.close()


def test_parent_death_while_waiting_cancels_and_does_not_create_output(binary, tmp_path):
    owner = subprocess.Popen(["/bin/sleep", "30"])
    child = held = None
    try:
        child, held, output = waiting_worker(binary, tmp_path, owner=owner.pid)
        owner.terminate()
        owner.wait(timeout=5)
        stdout, stderr = child.communicate(timeout=5)
        assert child.returncode == 130, stderr + stdout
        assert json.loads(stdout.splitlines()[-1])["status"] == "cancelled"
        assert not output.exists()
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait()
        if child and child.poll() is None:
            child.kill()
            child.wait()
        if held:
            held.close()


def test_progress_pipe_loss_while_waiting_unwinds_without_sigpipe(binary, tmp_path):
    child, held, output = waiting_worker(binary, tmp_path)
    try:
        child.stdout.close()
        child.wait(timeout=5)
        assert child.returncode in (1, 130), child.stderr.read()
        assert not output.exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        held.close()
        child.stderr.close()


def test_invalid_large_enum_returns_one_bounded_error_event(binary, tmp_path):
    envelope, recipe_file, output = request(tmp_path)
    recipe = json.loads(recipe_file.read_text())
    recipe["variant"] = "X" * 700000
    payload = json.dumps(recipe).encode()
    recipe_file.write_bytes(payload)
    wrapper = json.loads(envelope.read_text())
    wrapper["recipeSHA256"] = hashlib.sha256(payload).hexdigest()
    envelope.write_text(json.dumps(wrapper))
    process = subprocess.run(
        command(binary, "plan", envelope, output), capture_output=True, text=True, timeout=10
    )
    assert process.returncode == 1
    assert len(process.stdout.encode()) <= 65536
    assert result(process)["status"] == "error"
    assert not output.exists()


def test_sigterm_with_full_progress_pipe_has_bounded_shutdown(binary, tmp_path):
    import time

    held = (tmp_path / "backpressure.lock").open("w+b")
    fcntl.flock(held, fcntl.LOCK_EX)
    envelope, _, output = request(tmp_path, lease=tmp_path / "backpressure.lock")
    reader, writer = os.pipe()
    child = None
    try:
        child = subprocess.Popen(
            command(binary, "render", envelope, output), stdout=writer, stderr=subprocess.PIPE
        )
        assert select.select([reader], [], [], 5)[0]
        waiting = os.read(reader, 4096)
        assert json.loads(waiting)["stage"] == "waiting"
        flags = fcntl.fcntl(writer, fcntl.F_GETFL)
        os.set_blocking(writer, False)
        while True:
            try:
                os.write(writer, b"x" * 4096)
            except BlockingIOError:
                break
        fcntl.fcntl(writer, fcntl.F_SETFL, flags)
        held.close()
        deadline = time.monotonic() + 3
        while not output.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert output.exists(), "Worker did not enter render after its lease was released"
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=5)
        assert child.returncode in (1, 130)
        assert not (output / "restored.rgb").exists()
    finally:
        if child and child.poll() is None:
            child.kill()
            child.wait()
        if child:
            child.stderr.close()
        held.close()
        os.close(reader)
        os.close(writer)
