"""Exercise real pipe/process/lease behavior with a model-free helper fixture."""

import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from wee_todd_mlx.inference_lease import InferenceLease
from wee_todd_remote.assistant_session import AssistantSession


@pytest.fixture
def helper(tmp_path, monkeypatch):
    monkeypatch.setenv("WEETODD_STUDIO_DATA", str(tmp_path / "data"))
    executable = tmp_path / "helper"
    executable.write_text(
        f"#!{sys.executable}\n" + r'''
import json, os, signal, sys, time
def emit(value):
    print(json.dumps(value), flush=True)
mode = os.environ.get("SESSION_FIXTURE", "normal")
if sys.argv[1] == "text":
    request = json.load(sys.stdin)
    emit({"type":"result", "requestID":request["requestID"],
          "value":{"text":request["prompt"],"mode":"legacy"}})
    sys.exit()
if mode == "legacy":
    hello = json.load(sys.stdin)
    emit({"type":"error", "requestID":hello["requestID"],"code":"unsupported_operation"})
    sys.exit(1)
hello = json.loads(sys.stdin.readline())
sid = hello["sessionID"]
def event(kind, item="", **other):
    emit(dict(type=kind, version=1, sessionID=sid, itemID=item, **other))
event("hello", capabilities={"maxActiveItems":1,"maxQueuedItems":32})
for line in sys.stdin:
    record = json.loads(line)
    kind, item = record["type"], record.get("itemID", "")
    if kind == "open":
        event("opened")
    elif kind == "generate":
        prompt = record["value"]["prompt"]
        if prompt == "hold-admission":
            event("progress", item, value={"stage":"holding", "pid":os.getpid()})
            while not os.path.exists(os.environ["SESSION_RELEASE_FILE"]):
                time.sleep(0.02)
            os._exit(0)
        if prompt == "crash": os._exit(9)
        if prompt == "hang":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(30)
        if prompt == "oversize":
            print("x" * (1024 * 1024 + 1), flush=True)
            time.sleep(30)
        if prompt == "mismatch": item = "wrong-item"
        sys.stderr.write("private diagnostics\n" * 10000)
        sys.stderr.flush()
        event("progress", item, value={"stage":"decoding"})
        event("result", item, value={"text":prompt,"pid":os.getpid()})
    elif kind == "close":
        event("closed", value={"resident":False})
        sys.exit()
'''
    )
    executable.chmod(0o700)
    return executable


def request(prompt="first", request_id="one"):
    return {
        "requestID": request_id, "modelPath": "/models/qwen_3.5_4b_i8x.ckpt",
        "prompt": prompt, "systemPrompt": "Describe.", "maxTokens": 20,
    }


def test_reuses_worker_and_releases_it_after_context(helper):
    events = []
    with AssistantSession(helper) as session:
        first = session.generate(request(), progress=events.append)
        second = session.generate(request("second", "two"), dependencies=["one"])
        assert first["text"] == "first" and second["text"] == "second"
        assert first["pid"] == second["pid"]
        assert events[0]["value"]["stage"] == "decoding"
    with pytest.raises(ProcessLookupError):
        os.kill(first["pid"], 0)


def test_partial_success_is_retained_without_retry_after_crash(helper):
    with AssistantSession(helper) as session:
        first = session.generate(request())
        with pytest.raises(RuntimeError, match="exited|closed"):
            session.generate(request("crash", "two"))
        assert first["text"] == "first"
        with pytest.raises(RuntimeError, match="closed"):
            session.generate(request("third", "three"))


def test_legacy_fallback_is_only_before_submission(helper, monkeypatch):
    monkeypatch.setenv("SESSION_FIXTURE", "legacy")
    with AssistantSession(helper, handshake_timeout=0.1) as session:
        assert session.generate(request())["mode"] == "legacy"
        assert session.mode == "legacy"


@pytest.mark.parametrize("prompt, error", [("mismatch", "itemID"), ("oversize", "large")])
def test_rejects_invalid_records_and_shuts_down(helper, prompt, error):
    with AssistantSession(helper) as session:
        with pytest.raises(ValueError, match=error):
            session.generate(request(prompt))


def test_cancel_unresponsive_worker_releases_lease(helper):
    cancelled = threading.Event()
    timer = threading.Timer(0.2, cancelled.set)
    started = time.monotonic()
    timer.start()
    try:
        with AssistantSession(helper, cancelled=cancelled.is_set) as session:
            with pytest.raises(InterruptedError):
                session.generate(request("hang"))
    finally:
        timer.cancel()
    assert time.monotonic() - started < 3
    # A second job must acquire the actual lock after cancellation.
    with AssistantSession(helper) as other:
        assert other.generate(request())["text"] == "first"


def test_rejects_unfinished_dependency_before_submission(helper):
    with AssistantSession(helper) as session:
        with pytest.raises(ValueError, match="dependenc"):
            session.generate(request(), dependencies=["missing"])
        assert session.generate(request())["text"] == "first"


def test_duplicate_id_and_model_switch_are_rejected(helper):
    with AssistantSession(helper) as session:
        session.generate(request())
        with pytest.raises(ValueError, match="itemID"):
            session.generate(request("second"))
        changed = request("second", "two")
        changed["modelPath"] = "/other/qwen_3.5_4b_i8x.ckpt"
        with pytest.raises(ValueError, match="model"):
            session.generate(changed)


def test_9b_keeps_one_shot_compatibility(helper):
    value = request()
    value["modelPath"] = "/models/qwen_3.5_9b_i5x.ckpt"
    with AssistantSession(helper) as session:
        assert session.generate(value)["mode"] == "legacy"


def test_empty_cached_job_does_not_start_helper(tmp_path):
    with AssistantSession(tmp_path / "not-installed") as session:
        assert session.mode == "pending"


def test_job_holds_shared_inference_lock_between_items(helper):
    lock_path = helper.parent / "data/Runtime/native-inference.lock"
    with AssistantSession(helper) as session:
        session.generate(request())
        with lock_path.open("a+b") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    with lock_path.open("a+b") as other:
        fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_worker_holds_admission_after_abrupt_parent_death(helper, monkeypatch):
    release_file = helper.parent / "release-worker"
    monkeypatch.setenv("SESSION_RELEASE_FILE", str(release_file))
    parent = subprocess.run(
        [sys.executable, "-c", r'''
import json, os, sys
sys.path.insert(0, sys.argv[2])
from wee_todd_remote.assistant_session import AssistantSession
def progress(event):
    if event.get("value", {}).get("stage") == "holding":
        print(json.dumps({"pid": event["value"]["pid"]}), flush=True)
        os._exit(0)
with AssistantSession(sys.argv[1]) as session:
    session.generate({"requestID":"one", "modelPath":"/models/qwen_3.5_4b_i8x.ckpt",
                      "prompt":"hold-admission"}, progress=progress)
''', str(helper), str(Path(__file__).resolve().parents[1] / "src")],
        capture_output=True, text=True, timeout=10, check=True,
    )
    pid = json.loads(parent.stdout)["pid"]
    lock_path = helper.parent / "data/Runtime/native-inference.lock"
    try:
        os.kill(pid, 0)
        with lock_path.open("a+b") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        release_file.touch()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            os.kill(pid, 9)
            pytest.fail("Fixture worker did not exit after release")
    with AssistantSession(helper) as other:
        assert other.generate(request())["text"] == "first"


def test_cancel_waiting_for_admission_does_not_launch_worker(helper, monkeypatch):
    cancelled = threading.Event()
    with InferenceLease():
        with monkeypatch.context() as patch:
            def unexpected_launch(*args, **kwargs):
                pytest.fail("A worker was launched before admission")

            patch.setattr(subprocess, "Popen", unexpected_launch)
            with AssistantSession(
                helper, cancelled=cancelled.is_set,
                progress=lambda event: cancelled.set(),
            ) as session:
                with pytest.raises(InterruptedError, match="waiting"):
                    session.generate(request())
                assert session._process is None
                assert session._lease is None
    with AssistantSession(helper) as other:
        assert other.generate(request())["text"] == "first"


def test_item_deadline_cancels_worker_without_replay(helper):
    started = time.monotonic()
    with AssistantSession(helper) as session:
        with pytest.raises(TimeoutError):
            session.generate(request("hang"), timeout=0.15)
    assert time.monotonic() - started < 3


def test_per_call_cancel_callback_is_honored(helper):
    with AssistantSession(helper) as session:
        with pytest.raises(InterruptedError):
            session.generate(request(), cancelled=lambda: True)


def test_item_failure_closes_session_even_if_progress_callback_raises(helper):
    def fail_progress(_):
        raise LookupError("caller error")

    with AssistantSession(helper) as session:
        with pytest.raises(LookupError, match="caller error"):
            session.generate(request(), progress=fail_progress)
        with pytest.raises(RuntimeError, match="closed"):
            session.generate(request("second", "two"))


def test_live_swift_worker_exits_after_parent_loss():
    """Opt-in black-box qualification; no checkpoint or GPU allocation needed."""
    helper = os.environ.get("WEETODD_TEST_SESSION_HELPER")
    if not helper:
        pytest.skip("Set WEETODD_TEST_SESSION_HELPER to qualify the built Swift worker")
    assert Path(helper).is_file()
    parent = subprocess.run(
        [sys.executable, "-c", r'''
import json, os, subprocess, sys
worker = subprocess.Popen([sys.argv[1], "text-session"], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
hello = {"type":"hello", "version":1,"sessionID":"parent-death", "itemID":""}
worker.stdin.write(json.dumps(hello).encode() + b"\n"); worker.stdin.flush()
response = json.loads(worker.stdout.readline())
assert response["type"] == "hello", response
print(json.dumps({"pid":worker.pid}), flush=True)
os._exit(0)
''', helper], capture_output=True, text=True, timeout=15, check=True,
    )
    pid = json.loads(parent.stdout)["pid"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("Orphan assistant worker remained alive after its parent exited")
