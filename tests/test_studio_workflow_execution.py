"""Real runner semantics; only the weighted model boundary is substituted."""

import copy
import json
from pathlib import Path

import pytest

from wee_todd_mlx.workflows import load_document

EXAMPLES = Path(__file__).parents[1] / "examples/studio-workflows"


def api():
    from wee_todd_mlx.workflows.runner import WorkflowRunner

    return WorkflowRunner


class TextBackend:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.prompts = []

    def fingerprint(self, model, images):
        return {"model": model, "images": images, "version": 1}

    def generate(self, model, system, prompt, images, **kwargs):
        self.prompts.append(prompt)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return {"text": reply, "truncated": False, "totalSeconds": 0.01}


def edits():
    return [
        {
            "id": "fox",
            "instruction": "Replace the warrior with a fox",
            "preserve": [],
            "useImages": False,
        },
        {
            "id": "forest",
            "instruction": "Move the scene to a snowy forest",
            "preserve": ["fox"],
            "useImages": False,
        },
    ]


def replies():
    return [
        json.dumps(edits()),
        "A fox in a cathedral.",
        "A fox in a snowy forest.",
        json.dumps({"status": "pass", "items": []}),
    ]


def test_runner_carries_latest_draft_and_resumes_without_model_calls(tmp_path):
    definition = load_document(EXAMPLES / "staged-prompt-editing.json")
    backend = TextBackend(replies())
    runner = api()(definition, tmp_path, backend)
    result = runner.run({})
    assert result["status"] == "completed"
    assert result["outputs"]["prompt"] == "A fox in a snowy forest."
    assert "A fox in a cathedral." in backend.prompts[2]
    assert runner.run({})["outputs"] == result["outputs"]
    assert len(backend.prompts) == 4
    assert (tmp_path / "run.json").stat().st_size < 40_000


def test_next_step_and_changed_instructions_invalidate_descendants(tmp_path):
    definition = load_document(EXAMPLES / "staged-prompt-editing.json")
    backend = TextBackend(replies() + replies())
    runner = api()(definition, tmp_path, backend)
    paused = runner.run({}, max_steps=2)
    assert paused["status"] == "paused"
    assert set(paused["steps"]) == {"describe", "plan"}
    runner.run({})
    result = runner.run({"instructions": "Use a fox in snow."})
    assert result["status"] == "completed"
    assert len(backend.prompts) == 8


def test_cancel_mid_edits_resumes_from_last_successful_edit(tmp_path):
    definition = load_document(EXAMPLES / "staged-prompt-editing.json")
    backend = TextBackend(
        [
            json.dumps(edits()),
            "A fox in a cathedral.",
            InterruptedError("paused"),
            "A fox in a snowy forest.",
            json.dumps({"status": "pass", "items": []}),
        ]
    )
    runner = api()(definition, tmp_path, backend)
    result = runner.run({})
    assert result["status"] == "cancelled"
    assert result["steps"]["edit"]["status"] == "cancelled"
    result = runner.run({})
    assert result["status"] == "completed"
    assert len(backend.prompts) == 5
    assert "A fox in a cathedral." in backend.prompts[3]


def test_invalid_structured_output_stops_and_keeps_previous_steps(tmp_path):
    definition = load_document(EXAMPLES / "staged-prompt-editing.json")
    result = api()(definition, tmp_path, TextBackend(["not JSON", "still not JSON"])).run({})
    assert result["status"] == "failed"
    assert result["steps"]["plan"]["status"] == "failed"
    assert "JSON" in result["error"]
    assert "edit" not in result["steps"]


def test_inputs_bounds_unknown_inputs_and_disk_budget_fail_before_generation(tmp_path):
    definition = load_document(EXAMPLES / "movie-planning-v1.json")
    backend = TextBackend([])
    runner = api()(definition, tmp_path, backend)
    for inputs in ({"frame_rate": 0}, {"brief": 4}, {"unknown": 1}):
        with pytest.raises(ValueError):
            runner.run(inputs)
    definition["limits"]["maxWorkingBytes"] = 100
    with pytest.raises(ValueError, match="budget"):
        api()(definition, tmp_path / "small", backend).run({})
    assert not backend.prompts


def test_movie_exact_allocation_and_continuity_check():
    from wee_todd_mlx.workflows.operations import allocate_clips, check_plan

    inputs = {
        "story": "A fox explores.",
        "duration_seconds": 10.1,
        "target_clip_seconds": 4,
        "frame_rate": 24,
    }
    clips = allocate_clips(inputs, 10)
    assert clips["totalFrames"] == 242
    assert sum(c["frameCount"] for c in clips["clips"]) == 242
    assert len(clips["clips"]) == 3
    endpoints = [
        {"clipID": c["id"], "first": {"description": "start"}, "last": {"description": "end"}}
        for c in clips["clips"]
    ]
    inputs.update(clips=clips, endpoints=endpoints)
    review = check_plan(inputs)
    assert review["status"] == "needs_attention"  # Missing continuity links.
    for i in range(1, len(endpoints)):
        endpoints[i]["first"] = {
            "description": "end",
            "reuseFrom": {"clipID": endpoints[i - 1]["clipID"], "endpoint": "last"},
        }
    review = check_plan(inputs)
    assert not any(i["severity"] == "error" for i in review["items"])
    assert any("round" in i["message"].lower() for i in review["items"])
    endpoints[0]["last"]["reuseFrom"] = {"clipID": endpoints[-1]["clipID"], "endpoint": "first"}
    assert check_plan(inputs)["status"] == "needs_attention"
    with pytest.raises(ValueError, match="maxClips"):
        allocate_clips(inputs, 2)


def test_truncated_output_is_not_accepted_as_complete(tmp_path):
    class Truncated(TextBackend):
        def generate(self, *args, **kwargs):
            return {"text": json.dumps(edits()), "truncated": True}

    definition = load_document(EXAMPLES / "staged-prompt-editing.json")
    result = api()(definition, tmp_path, Truncated([])).run({})
    assert result["status"] == "failed"
    assert "limit" in result["error"].lower()


def test_run_directory_does_not_accept_different_workflow(tmp_path):
    definition = load_document(EXAMPLES / "staged-prompt-editing.json")
    api()(definition, tmp_path, TextBackend(replies())).run({})
    other = copy.deepcopy(definition)
    other["id"] = "custom.other"
    with pytest.raises(ValueError, match="different workflow"):
        api()(other, tmp_path, TextBackend([])).run({})


def test_local_backend_rejects_mismatched_model_and_missing_asset(tmp_path):
    from wee_todd_mlx.workflows.backend import LocalQwenBackend

    model = {
        "id": "assistant",
        "family": "qwen3.5",
        "variant": "4b",
        "runtime": "drawthings-qwen-local",
        "capabilities": ["text", "vision"],
    }
    wrong = tmp_path / "qwen_3.5_9b_i5x.ckpt"
    wrong.write_bytes(b"not a model")
    backend = LocalQwenBackend({"assistant": str(wrong)}, {}, "/bin/echo")
    with pytest.raises(ValueError, match="4b"):
        backend.fingerprint(model, [])
    correct = tmp_path / "qwen_3.5_4b_i8x.ckpt"
    correct.write_bytes(b"SQLite format 3\x00")
    backend = LocalQwenBackend({"assistant": str(correct)}, {}, "/bin/echo")
    with pytest.raises(ValueError, match="asset:missing"):
        backend.fingerprint(model, ["asset:missing"])


def test_packaged_builtins_and_dispatch_support_next_step(tmp_path):
    from wee_todd_mlx.workflows.service import dispatch

    catalog = dispatch("workflow-catalog", {})
    assert {d["id"] for d in catalog["definitions"]} == {
        "weetodd.guided-movie-planning",
        "weetodd.staged-prompt-editing",
        "weetodd.movie-planning",
        "weetodd.subject-inventory-reviewed",
    }
    result = dispatch(
        "workflow-run",
        {
            "builtin": "staged-prompt-editing",
            "runDirectory": str(tmp_path),
            "inputs": {},
            "maxSteps": 1,
        },
        backend=TextBackend([]),
    )
    assert result["status"] == "paused"
    assert result["steps"]["describe"]["status"] == "completed"


def test_movie_endpoint_steps_use_small_text_tasks_and_host_controls_reuse(tmp_path):
    definition = load_document(EXAMPLES / "movie-planning-v1.json")
    backend = TextBackend(
        ["A fox finds shelter.", "A fox in snow.", "A fox at a cabin.", "A fox beside a warm fire."]
    )
    result = api()(definition, tmp_path, backend).run(
        {"duration_seconds": 10, "target_clip_seconds": 5, "frame_rate": 24}
    )
    assert result["status"] == "completed", result.get("error")
    assert result["outputs"]["review"]["status"] == "pass"
    pairs = result["outputs"]["endpoints"]
    assert pairs[1]["first"]["description"] == pairs[0]["last"]["description"]
    assert pairs[1]["first"]["reuseFrom"] == {"clipID": "clip-1", "endpoint": "last"}
    assert (
        len(backend.prompts) == 4
    )  # A continuous clip reuses the first frame without a model call.


def test_vision_descriptions_use_exact_asset_ids_without_model_json(tmp_path):
    definition = load_document(EXAMPLES / "staged-prompt-editing.json")
    result = api()(definition, tmp_path, TextBackend(["A fox.", "A snowy forest."])).run(
        {"images": ["asset:fox", "asset:forest"]}, max_steps=1
    )
    assert result["status"] == "paused", result.get("error")
    observations = result["steps"]["describe"]["outputs"]["observations"]
    assert [o["image"] for o in observations] == ["asset:fox", "asset:forest"]
    assert observations[1]["details"] == "A snowy forest."


def test_retry_receives_format_failure_and_does_not_reuse_invalid_json(tmp_path):
    definition = load_document(EXAMPLES / "staged-prompt-editing.json")
    definition["steps"][1]["retry"]["maxAttempts"] = 2
    backend = TextBackend(["malformed", *replies()])
    result = api()(definition, tmp_path, backend).run({})
    assert result["status"] == "completed", result.get("error")
    assert "Previous attempt" in backend.prompts[1]
    assert result["executions"] == 5


def test_resume_revalidates_changed_model_and_regeneration_invalidates_children(tmp_path):
    definition = load_document(EXAMPLES / "staged-prompt-editing.json")
    backend = TextBackend(replies() + replies()[1:])
    runner = api()(definition, tmp_path, backend)
    runner.run({})
    result = runner.run({}, max_steps=1, regenerate="edit")
    assert result["status"] == "paused"
    assert "check" not in result["steps"]
    assert len(backend.prompts) == 6
    assert runner.run({})["status"] == "completed"


def test_execution_lock_and_cancellation_before_model_call(tmp_path):
    import fcntl

    definition = load_document(EXAMPLES / "staged-prompt-editing.json")
    with (tmp_path / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="already active"):
            api()(definition, tmp_path, TextBackend([])).run({})
    result = api()(definition, tmp_path, TextBackend([]), cancelled=lambda: True).run({})
    assert result["status"] == "cancelled"
    assert result["executions"] == 0


def test_expired_step_does_not_publish_success(tmp_path):
    import time

    definition = load_document(EXAMPLES / "staged-prompt-editing.json")
    definition["steps"][1]["timeoutSeconds"] = 1

    class SlowBackend(TextBackend):
        def generate(self, *args, **kwargs):
            time.sleep(1.05)
            return super().generate(*args, **kwargs)

    result = api()(definition, tmp_path, SlowBackend(replies())).run({})
    assert result["status"] == "failed"
    assert "time limit" in result["error"]
    assert not result["outputs"]


def test_bridge_cli_round_trip_and_signal_cancel_releases_owned_helper(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time

    root = Path(__file__).parents[1]
    helper = tmp_path / "helper"
    helper.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(30)\n")
    helper.chmod(0o755)
    model = tmp_path / "qwen_3.5_4b_i8x.ckpt"
    model.write_bytes(b"SQLite format 3\x00")
    request = {
        "builtin": "staged-prompt-editing",
        "models": {"assistant": str(model)},
        "runtime": {"drawThingsHelperPath": str(helper)},
        "runDirectory": str(tmp_path / "run"),
    }
    request_file = tmp_path / "job.json"
    request_file.write_text(json.dumps(request))
    child = subprocess.Popen(
        [
            sys.executable,
            str(root / "scripts/studio_bridge.py"),
            "workflow-run",
            "--request",
            str(request_file),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        start = time.monotonic()
        while time.monotonic() - start < 4:
            state_file = tmp_path / "run/run.json"
            if state_file.exists() and json.loads(state_file.read_text()).get("executions", 0) >= 1:
                break
            time.sleep(0.02)
        os.kill(child.pid, signal.SIGINT)
        output, _ = child.communicate(timeout=4)
        result = json.loads(output.splitlines()[-1])
        assert child.returncode == 0, output
        assert result["result"]["status"] == "cancelled"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_endpoint_prompt_exposes_time_window_in_seconds(tmp_path):
    definition = load_document(EXAMPLES / "movie-planning-v1.json")
    backend = TextBackend(["Story", "Start", "Middle", "End"])
    api()(definition, tmp_path, backend).run({"duration_seconds": 10, "target_clip_seconds": 5})
    assert '"startSeconds": 0.0' in backend.prompts[1]
    assert '"endSeconds": 5.0' in backend.prompts[2]
    assert '"endSeconds": 10.0' in backend.prompts[3]


def test_image_evidence_is_only_sent_to_edits_that_explicitly_request_it(tmp_path):
    definition = load_document(EXAMPLES / "staged-prompt-editing.json")
    backend = TextBackend(["VISIBLE ARMOR", *replies()])
    result = api()(definition, tmp_path, backend).run({"images": ["asset:reference"]})
    assert result["status"] == "completed"
    assert "VISIBLE ARMOR" not in backend.prompts[2]
    assert "VISIBLE ARMOR" not in backend.prompts[3]
    image_edit = [
        {"id": "image", "instruction": "Describe the reference", "preserve": [], "useImages": True}
    ]
    backend = TextBackend(
        [
            "VISIBLE ARMOR",
            json.dumps(image_edit),
            "A warrior.",
            json.dumps({"status": "pass", "items": []}),
        ]
    )
    result = api()(definition, tmp_path / "images", backend).run({"images": ["asset:reference"]})
    assert result["status"] == "completed"
    assert "VISIBLE ARMOR" in backend.prompts[2]


def test_packaged_builtins_match_shipped_authoring_examples():
    from wee_todd_mlx.workflows.service import BUILTINS, builtin

    for name in BUILTINS:
        assert builtin(name) == load_document(EXAMPLES / f"{name}.json")


def test_empty_draft_starts_from_observed_images(tmp_path):
    definition = load_document(EXAMPLES / "staged-prompt-editing.json")
    backend = TextBackend(
        [
            "A smiling black and white badger.",
            json.dumps(
                [
                    {
                        "id": "describe",
                        "instruction": "Describe the animal",
                        "preserve": [],
                        "useImages": True,
                    }
                ]
            ),
            "A cheerful badger.",
            json.dumps({"status": "pass", "items": []}),
        ]
    )
    result = api()(definition, tmp_path, backend).run({"source": "", "images": ["asset:animal"]})
    assert result["status"] == "completed"
    assert "Latest draft:\nA smiling black and white badger." in backend.prompts[2]
