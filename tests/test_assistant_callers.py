"""Job lifetime around real extraction and workflow caller boundaries."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from test_character_assist import field, request


class RecordingSession:
    instances = []

    def __init__(self, helper_path, **kwargs):
        self.calls = []
        self.closed = False
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def generate(self, payload, **kwargs):
        assert not self.closed
        assert callable(kwargs["cancelled"])
        self.calls.append(payload)
        prompt = json.loads(payload["prompt"])
        if "collections" in prompt:
            return {
                "text": '{"records":[{"collection":"garments","slot":1,"excerpt":"blue shirt"},'
                '{"collection":"garments","slot":2,"excerpt":"black trousers"}]}'
            }
        return {"text": '{"proposals":[]}'}


@pytest.fixture
def recording(monkeypatch):
    import studio_character_assist as character

    RecordingSession.instances = []
    monkeypatch.setattr(character, "AssistantSession", RecordingSession)
    return RecordingSession.instances


def test_character_inventory_and_attributes_share_one_session_then_close(tmp_path, recording):
    import studio_character_assist as character

    value = request(tmp_path, "text")
    value["sourceText"] = "A blue shirt and black trousers."
    value["fields"] = [field("garments[].type", 6)]
    value["requestedFields"] = ["garments[first].type", "garments[second].type"]
    character.extract_fields(value)
    assert len(recording) == 1
    assert len(recording[0].calls) == 3
    assert recording[0].closed
    character.extract_fields(value)
    assert not recording[-1].calls  # Lazy context does not start inference on a cache hit.
    assert all(session.closed for session in recording)


def test_character_failure_closes_session(tmp_path, recording, monkeypatch):
    import studio_character_assist as character

    def crash(*args, **kwargs):
        raise RuntimeError("worker failed")

    monkeypatch.setattr(RecordingSession, "generate", crash)
    with pytest.raises(RuntimeError, match="worker failed"):
        character.extract_fields(request(tmp_path))
    assert recording[0].closed


def test_oneshot_control_does_not_create_resident_session(tmp_path, recording, monkeypatch):
    import studio_character_assist as character

    value = request(tmp_path)
    value["runtime"]["assistantExecutionMode"] = "oneshot"
    monkeypatch.setattr(character, "assist", lambda *args, **kwargs: {"text": '{"proposals":[]}'})
    character.extract_fields(value)
    assert not recording


def test_workflow_backend_owns_one_session_for_consecutive_calls(tmp_path, monkeypatch):
    import wee_todd_mlx.workflows.backend as module

    monkeypatch.setattr(module, "AssistantSession", RecordingSession)
    model = tmp_path / "qwen_3.5_4b_i8x.ckpt"
    model.write_bytes(b"SQLite format 3\0")
    binding = dict(id="assistant", runtime="drawthings-qwen-local", family="qwen3.5", variant="4b")
    RecordingSession.instances = []
    with module.LocalQwenBackend({"assistant": str(model)}, {}, "/unused/helper") as backend:
        for _ in range(2):
            backend.generate(
                binding, "extract", "{}", [], cancelled=lambda: False, timeout=30, max_tokens=128
            )
        assert not RecordingSession.instances[0].closed
    assert len(RecordingSession.instances) == 1
    assert len(RecordingSession.instances[0].calls) == 2
    assert RecordingSession.instances[0].closed


@pytest.mark.parametrize("operation", ["workflow-run", "workflow-review"])
def test_dispatch_closes_owned_backend_before_returning_review(tmp_path, monkeypatch, operation):
    import wee_todd_mlx.workflows.service as service

    state = {"active": False}

    class Backend:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            state["active"] = True
            return self

        def __exit__(self, *_):
            state["active"] = False

    class Runner:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            assert state["active"]
            return {"status": "awaiting_review"}

        review = run

    monkeypatch.setattr(service, "LocalQwenBackend", Backend)
    monkeypatch.setattr(service, "WorkflowRunner", Runner)
    result = service.dispatch(operation, {"runDirectory": str(tmp_path)})
    assert result["status"] == "awaiting_review"
    assert not state["active"]


def test_workflow_model_switch_unloads_before_next_model(tmp_path, monkeypatch):
    import wee_todd_mlx.workflows.backend as module

    monkeypatch.setattr(module, "AssistantSession", RecordingSession)
    RecordingSession.instances = []
    models = {}
    for variant, filename in [("4b", "qwen_3.5_4b_i8x.ckpt"), ("9b", "qwen_3.5_9b_i5x.ckpt")]:
        model = tmp_path / filename
        model.write_bytes(b"SQLite format 3\0")
        models[variant] = str(model)
    with module.LocalQwenBackend(models, {}, "/unused/helper") as backend:
        for variant in models:
            binding = dict(
                id=variant, runtime="drawthings-qwen-local", family="qwen3.5", variant=variant
            )
            backend.generate(
                binding, "extract", "{}", [], cancelled=lambda: False, timeout=30, max_tokens=128
            )
        assert len(RecordingSession.instances) == 2
        assert RecordingSession.instances[0].closed
    assert all(session.closed for session in RecordingSession.instances)


def test_single_prompt_action_uses_same_session_owner_and_unloads(monkeypatch):
    import studio_prompt_assist as prompt

    RecordingSession.instances = []
    monkeypatch.setattr(prompt, "AssistantSession", RecordingSession, raising=False)

    def no_oneshot(*args, **kwargs):
        pytest.fail("4B prompt action bypassed the scoped session")

    monkeypatch.setattr(prompt, "invoke_helper", no_oneshot)
    result = prompt.assist(
        {
            "runtime": {"drawThingsHelperPath": "/unused/helper"},
            "textRequest": {"modelPath": "/model/qwen_3.5_4b_i8x.ckpt", "prompt": "{}"},
        }
    )
    assert result["text"] == '{"proposals":[]}'
    assert len(RecordingSession.instances) == 1
    assert RecordingSession.instances[0].closed
