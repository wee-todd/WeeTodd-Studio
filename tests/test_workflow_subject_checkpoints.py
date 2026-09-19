"""Validated extraction pages survive retries without cross-kind error feedback."""

import json

import pytest
from test_studio_workflow_execution import TextBackend

from wee_todd_mlx.workflows.service import builtin, dispatch


def request(tmp_path):
    definition = builtin("subject-inventory")
    definition["steps"][0]["operation"] = "project.identify_creative_subjects@1"
    definition["limits"]["maxStepExecutions"] = 2048
    return {
        "definition": definition,
        "inputs": {
            "brief": "\n".join(
                [
                    "Ada is a sailor.",
                    "Cup is golden.",
                    *[f"Bay {index:02d} is a place." for index in range(1, 10)],
                ]
            )
        },
        "runDirectory": str(tmp_path),
    }


def selection(name, evidence):
    return {"name": name, "evidenceIDs": [evidence]}


def prefix():
    return [
        json.dumps([selection("Ada", 1)]),
        json.dumps([selection("Cup", 2)]),
        json.dumps([selection(f"Bay {index:02d}", index + 2) for index in range(1, 9)]),
    ]


def finish():
    return [json.dumps([selection("Bay 09", 11)]), *["[]"] * 11]


BAD = json.dumps([selection("Invented Place", 11)])


def test_failed_location_page_retries_without_regenerating_validated_prefix(tmp_path):
    backend = TextBackend([*prefix(), BAD, *finish()])
    state = dispatch("workflow-run", request(tmp_path), backend=backend)
    assert state["status"] == "awaiting_approval", state.get("error")
    subjects = state["steps"]["subjects"]["outputs"]["subjects"]
    assert len(subjects) == 11
    assert subjects[0]["name"] == "Ada" and subjects[0]["kind"] == "character"
    assert subjects[-1]["name"] == "Bay 09" and subjects[-1]["kind"] == "location"
    assert len(backend.prompts) == 16
    assert "Invented Place" in backend.prompts[4]
    assert '"Bay 08"' in backend.prompts[4]
    assert all("Previous attempt failed" not in prompt for prompt in backend.prompts[5:])
    assert not state["steps"]["subjects"].get("approved")


def test_failed_page_exhausts_workflow_retry_cap_and_resumes_saved_prefix(tmp_path):
    req = request(tmp_path)
    backend = TextBackend([*prefix(), BAD, BAD])
    failed = dispatch("workflow-run", req, backend=backend)
    assert failed["status"] == "failed"
    assert len(backend.prompts) == 5
    backend.replies = iter(finish())
    resumed = dispatch("workflow-run", req, backend=backend)
    assert resumed["status"] == "awaiting_approval", resumed.get("error")
    assert len(backend.prompts) == 17
    assert "Invented Place" in backend.prompts[5]
    assert '"Bay 08"' in backend.prompts[5]
    assert len(resumed["steps"]["subjects"]["outputs"]["subjects"]) == 11


def test_interrupted_extraction_page_resumes_without_repeating_saved_prefix(tmp_path):
    req = request(tmp_path)
    backend = TextBackend([*prefix(), InterruptedError("pause")])
    paused = dispatch("workflow-run", req, backend=backend)
    assert paused["status"] == "cancelled"
    backend.replies = iter(finish())
    resumed = dispatch("workflow-run", req, backend=backend)
    assert resumed["status"] == "awaiting_approval", resumed.get("error")
    assert len(backend.prompts) == 16
    assert '"Bay 08"' in backend.prompts[4]
    assert "Previous attempt failed" not in backend.prompts[4]


@pytest.mark.parametrize("change", ["source", "model", "runtime", "settings", "step"])
def test_incomplete_extraction_checkpoint_invalidates_when_parent_identity_changes(
    tmp_path, change
):
    req = request(tmp_path)
    paused = dispatch(
        "workflow-run",
        req,
        backend=TextBackend([*prefix(), InterruptedError("pause")]),
    )
    assert paused["status"] == "cancelled"
    backend = TextBackend([*prefix(), *finish()])
    if change == "source":
        req["inputs"]["brief"] += "\nAdditional source facts."
    elif change == "model":
        req["definition"]["models"]["assistant"]["variant"] = "9b"
    elif change == "runtime":
        backend.fingerprint = lambda model, images: {"model": model, "images": images, "version": 2}
    elif change == "settings":
        req["maxTokens"] = 512
    else:
        req["definition"]["steps"][0]["name"] = "Updated subject extraction"
    resumed = dispatch("workflow-run", req, backend=backend)
    assert resumed["status"] == "awaiting_approval", resumed.get("error")
    assert len(backend.prompts) == 15
    assert "Already identified" not in backend.prompts[0]


def test_pause_after_model_return_does_not_checkpoint_unvalidated_response(tmp_path):
    cancel = {"requested": False}

    class PausingBackend(TextBackend):
        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            if len(self.prompts) == 4:
                cancel["requested"] = True
            return result

    req = request(tmp_path)
    backend = PausingBackend([*prefix(), BAD])
    paused = dispatch(
        "workflow-run",
        req,
        backend=backend,
        cancelled=lambda: cancel["requested"],
    )
    assert paused["status"] == "cancelled"
    cancel["requested"] = False
    backend.replies = iter(finish())
    resumed = dispatch("workflow-run", req, backend=backend)
    assert resumed["status"] == "awaiting_approval", resumed.get("error")
    assert len(backend.prompts) == 16
    assert '"Bay 08"' in backend.prompts[4]
    assert "Previous attempt failed" not in backend.prompts[4]
    assert len(resumed["steps"]["subjects"]["outputs"]["subjects"]) == 11
