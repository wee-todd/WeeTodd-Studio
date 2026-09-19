"""Completed coverage is reusable only with exact artifacts and fresh execution identity."""

import copy
import json

import pytest
from test_studio_workflow_execution import TextBackend
from test_workflow_object_coverage import proposal
from test_workflow_subject_links import empty_proposal, request

from wee_todd_mlx.workflows.service import dispatch


class Backend(TextBackend):
    def __init__(self, replies=(), version=1):
        super().__init__(
            [json.dumps(value) if isinstance(value, dict) else value for value in replies]
        )
        self.version = version

    def fingerprint(self, model, images):
        return {**super().fingerprint(model, images), "version": self.version}


def review(req, state, action, backend=None, **extra):
    return dispatch(
        "workflow-review",
        {
            **req,
            "review": {
                "stepID": "links",
                "expectedRevision": state["revision"],
                "action": action,
                **extra,
            },
        },
        backend=backend or Backend(),
    )


def prepared(tmp_path, *, protected=False, replies=None, manual=True):
    req = request(tmp_path)
    req["definition"]["id"] = "weetodd.music-video-planning"
    links = req["definition"]["steps"][0]
    links["requiresApproval"] = False
    coverage = copy.deepcopy(links)
    coverage.update(
        id="inventory",
        name="Review inventory",
        operation="project.review_object_coverage@1",
        requiresApproval=True,
    )
    coverage["inputs"]["subjects"] = {"step": "links", "output": "subjects"}
    coverage["inputs"]["library"] = {"input": "library"}
    req["definition"]["inputs"]["coverage_brief"] = {
        "type": "text",
        "label": "Coverage brief",
        "default": req["inputs"]["brief"],
    }
    coverage["inputs"]["brief"] = {"input": "coverage_brief"}
    req["definition"]["steps"].append(coverage)
    req["definition"]["outputs"] = {"subjects": {"step": "inventory", "output": "subjects"}}
    state = dispatch(
        "workflow-run",
        {**req, "maxSteps": 1},
        backend=Backend([empty_proposal(), empty_proposal()]),
    )
    assert state["status"] == "paused", state.get("error")
    if not manual:
        return req, state
    if protected:
        state = review(req, state, "approve", itemID="dog")
    state = review(
        req,
        state,
        "review_object_coverage",
        Backend(replies if replies is not None else [proposal(), proposal()]),
    )
    return req, state


def test_manual_coverage_reused_by_direct_inventory_without_calls_or_approval_copy(tmp_path):
    req, state = prepared(tmp_path)
    before = copy.deepcopy(state["steps"]["links"]["outputs"]["subjects"])
    assert before[0]["relationships"]  # The pass changed its own input; reuse its applied output.
    certificate = state["steps"]["links"]["coverageReviewRuns"].get("coverageCertificate")
    assert certificate
    state = review(req, state, "approve")
    backend = Backend()
    resumed = dispatch("workflow-run", req, backend=backend)
    assert resumed["status"] == "awaiting_approval", resumed.get("error")
    assert backend.prompts == []
    step = resumed["steps"]["inventory"]
    assert step["outputs"]["subjects"] == before
    assert step["coverageReuse"]["sourceStepID"] == "links"
    assert step["coverageReuse"]["certificateKey"] == certificate["key"]
    assert not step.get("approved")
    assert all(not item.get("approved") for item in step["items"].values())
    assert resumed["executions"] == state["executions"]


@pytest.mark.parametrize(
    "change",
    [
        "description",
        "relationships",
        "brief",
        "library",
        "runtime",
        "model",
        "settings",
        "implementation",
        "packing",
    ],
)
def test_coverage_reuse_rejects_changed_identity(tmp_path, monkeypatch, change):
    from wee_todd_mlx.workflows import object_coverage

    req, state = prepared(tmp_path)
    version = 1
    if change in {"description", "relationships"}:
        outputs = copy.deepcopy(state["steps"]["links"]["outputs"])
        if change == "description":
            outputs["subjects"][0]["description"] += " A white tail tip."
        else:
            outputs["subjects"][0]["relationships"][0]["placement"] = "new placement"
        state = review(req, state, "edit", outputs=outputs)
    elif change == "brief":
        req["inputs"]["coverage_brief"] = "Dog wears a collar beside a tree."
    elif change == "library":
        req["inputs"]["library"] = [
            {
                "id": "saved",
                "name": "Collar",
                "kind": "clothing",
                "aliases": [],
                "tags": [],
                "description": "Leather",
                "packageID": "pack",
                "version": 1,
                "definitionRevision": "revision",
                "scope": "global",
            }
        ]
    elif change == "runtime":
        version = 2
    elif change == "model":
        req["definition"]["models"]["other"] = copy.deepcopy(
            req["definition"]["models"]["assistant"]
        )
        req["definition"]["steps"][1]["model"] = "other"
    elif change == "settings":
        req["maxTokens"] = 512
    elif change == "implementation":
        monkeypatch.setattr(
            object_coverage,
            "COVERAGE_IMPLEMENTATION_VERSION",
            object_coverage.COVERAGE_IMPLEMENTATION_VERSION + 1,
        )
    else:
        monkeypatch.setattr(object_coverage, "REQUEST_VERSION", object_coverage.REQUEST_VERSION + 1)
    backend = Backend([proposal(), proposal()], version=version)
    resumed = dispatch("workflow-run", req, backend=backend)
    assert resumed["status"] == "awaiting_approval", resumed.get("error")
    assert len(backend.prompts) == 2
    assert "coverageReuse" not in resumed["steps"]["inventory"]


def test_protected_unapplied_proposals_do_not_certify_manual_review(tmp_path):
    req, state = prepared(tmp_path, protected=True)
    step = state["steps"]["links"]
    assert step["coverageReviewReport"]["proposals"]
    assert "coverageCertificate" not in step["coverageReviewRuns"]
    backend = Backend([proposal(), proposal()])
    resumed = dispatch("workflow-run", req, backend=backend)
    assert len(backend.prompts) == 2
    assert resumed["status"] == "awaiting_approval"


def test_failed_advisory_pass_does_not_certify_coverage(tmp_path):
    req, state = prepared(tmp_path, replies=[RuntimeError("Offline")] * 4)
    assert "coverageCertificate" not in state["steps"]["links"]["coverageReviewRuns"]
    backend = Backend([proposal(), proposal()])
    resumed = dispatch("workflow-run", req, backend=backend)
    assert len(backend.prompts) == 2
    assert resumed["status"] == "awaiting_approval"


def test_old_record_without_certificate_cannot_be_inferred(tmp_path):
    req, state = prepared(tmp_path)
    state["steps"]["links"]["coverageReviewRuns"].pop("coverageCertificate", None)
    (tmp_path / "run.json").write_text(json.dumps(state))
    backend = Backend([proposal(), proposal()])
    resumed = dispatch("workflow-run", req, backend=backend)
    assert len(backend.prompts) == 2
    assert resumed["status"] == "awaiting_approval"


def test_incomplete_certificate_owner_is_not_reused(tmp_path):
    req, state = prepared(tmp_path)
    state["steps"]["links"]["coverageReviewRuns"]["items"]["dog"]["status"] = "pending"
    (tmp_path / "run.json").write_text(json.dumps(state))
    backend = Backend([proposal(), proposal()])
    resumed = dispatch("workflow-run", req, backend=backend)
    assert len(backend.prompts) == 2
    assert resumed["status"] == "awaiting_approval"


@pytest.mark.parametrize("runtime_changed", [False, True])
def test_interrupted_manual_pass_resumes_only_its_fresh_runtime(tmp_path, runtime_changed):
    req, state = prepared(tmp_path, manual=False)
    with pytest.raises(InterruptedError):
        review(
            req, state, "review_object_coverage", Backend([proposal(), InterruptedError("Pause")])
        )
    state = json.loads((tmp_path / "run.json").read_text())
    assert "coverageCertificate" not in state["steps"]["links"]["coverageReviewRuns"]
    version = 2 if runtime_changed else 1
    backend = Backend(
        [proposal(), proposal()] if runtime_changed else [proposal()], version=version
    )
    state = review(req, state, "review_object_coverage", backend)
    assert len(backend.prompts) == (2 if runtime_changed else 1)
    assert state["steps"]["links"]["coverageReviewRuns"]["coverageCertificate"]
    continuation = Backend(version=version)
    resumed = dispatch("workflow-run", req, backend=continuation)
    assert resumed["status"] == "awaiting_approval", resumed.get("error")
    assert continuation.prompts == []
    assert resumed["steps"]["inventory"]["coverageReuse"]["sourceStepID"] == "links"


@pytest.mark.parametrize("changed", [False, True])
def test_intervening_design_review_prevents_ancestor_reuse(tmp_path, monkeypatch, changed):
    from wee_todd_mlx.workflows import description_review

    req, _ = prepared(tmp_path)
    inventory = req["definition"]["steps"][1]
    inventory["requiresApproval"] = False
    design = copy.deepcopy(inventory)
    design.update(id="design", operation="project.review_subjects@1")
    design["inputs"].pop("library")
    design["inputs"]["subjects"] = {"step": "inventory", "output": "subjects"}
    final = copy.deepcopy(inventory)
    final.update(id="final", requiresApproval=True)
    final["inputs"]["subjects"] = {"step": "design", "output": "subjects"}
    req["definition"]["steps"] += [design, final]
    req["definition"]["outputs"] = {"subjects": {"step": "final", "output": "subjects"}}

    def reviewed(row, *args, **kwargs):
        result = copy.deepcopy(row)
        if changed:
            result["description"] += " Fine silver details."
            result.pop("descriptionMentions", None)
            result.pop("mentionSourceDescription", None)
        return result

    monkeypatch.setattr(description_review, "review_description", reviewed)
    backend = Backend([proposal(), proposal()])
    state = dispatch("workflow-run", req, backend=backend)
    assert state["status"] == "awaiting_approval", state.get("error")
    assert state["steps"]["inventory"]["coverageReuse"]
    assert "coverageReuse" not in state["steps"]["final"]
    assert len(backend.prompts) == 2


def test_runtime_changed_during_manual_pass_never_gets_a_certificate(tmp_path):
    req, state = prepared(tmp_path, manual=False)

    class ChangingBackend(Backend):
        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            self.version += 1
            return result

    state = review(req, state, "review_object_coverage", ChangingBackend([proposal(), proposal()]))
    assert "coverageCertificate" not in state["steps"]["links"]["coverageReviewRuns"]


def test_direct_automatic_coverage_can_reuse_certified_automatic_output(tmp_path):
    req, _ = prepared(tmp_path, manual=False)
    inventory = req["definition"]["steps"][1]
    inventory["requiresApproval"] = False
    final = copy.deepcopy(inventory)
    final.update(id="final", requiresApproval=True)
    final["inputs"]["subjects"] = {"step": "inventory", "output": "subjects"}
    req["definition"]["steps"].append(final)
    req["definition"]["outputs"] = {"subjects": {"step": "final", "output": "subjects"}}
    backend = Backend([proposal(), proposal()])
    state = dispatch("workflow-run", req, backend=backend)
    assert state["status"] == "awaiting_approval", state.get("error")
    assert len(backend.prompts) == 2
    assert state["steps"]["final"]["coverageReuse"]["sourceRecord"] == "step"
    assert state["steps"]["final"]["outputs"] == state["steps"]["inventory"]["outputs"]


def test_explicit_manual_review_is_not_skipped_by_direct_input_certificate(tmp_path):
    req, _ = prepared(tmp_path)
    state = dispatch("workflow-run", req, backend=Backend())
    backend = Backend([proposal(), proposal()])
    state = review(req, state, "review_object_coverage", backend, stepID="inventory")
    assert len(backend.prompts) == 2
    assert "coverageReuse" not in state["steps"]["inventory"]["coverageReviewRuns"]
