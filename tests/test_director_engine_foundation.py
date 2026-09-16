"""Director reliability contracts; weighted generation is replaced at the boundary."""

import copy
import json

import pytest
from test_studio_workflow_execution import TextBackend
from test_workflow_object_coverage import coverage_request, proposal
from test_workflow_subject_links import inventory
from test_workflow_turns import turns

from wee_todd_mlx.workflows.service import dispatch


def coverage(tmp_path):
    req = coverage_request(tmp_path)
    req["inputs"]["inventory"] = [inventory()[1]]
    return req


@pytest.mark.parametrize("error", [InterruptedError, TimeoutError])
def test_interrupted_coverage_attempts_do_not_become_ready_without_response(tmp_path, error):
    req = coverage(tmp_path)
    for _ in range(2):
        state = dispatch("workflow-run", req, backend=TextBackend([error("Stopped")]))
        assert state["status"] in {"failed", "cancelled"}
    backend = TextBackend([json.dumps(proposal())])
    resumed = dispatch("workflow-run", req, backend=backend)
    assert len(backend.prompts) == 1
    assert (
        resumed["steps"]["links"]["outputs"]["subjects"][0]["coverageReview"]["status"] == "ready"
    )


def test_legacy_exhausted_interrupted_coverage_is_migrated(tmp_path):
    req = coverage(tmp_path)
    state = dispatch("workflow-run", req, backend=TextBackend([InterruptedError("Stopped")]))
    item = state["steps"]["links"]["items"]["collar"]
    item.update(attempts=2, calls=[], status="pending")
    item.pop("attemptStateVersion", None)
    (tmp_path / "run.json").write_text(json.dumps(state))
    backend = TextBackend([json.dumps(proposal())])
    result = dispatch("workflow-run", req, backend=backend)
    assert len(backend.prompts) == 1
    assert result["steps"]["links"]["items"]["collar"]["attemptStateVersion"] == 2


def test_metrics_survive_turn_log_and_resume_cache(tmp_path):
    class Metrics(TextBackend):
        def generate(self, *args, **kwargs):
            return {
                **super().generate(*args, **kwargs),
                "inputTokens": 42,
                "outputTokens": 20,
                "imagesUsed": 0,
                "timing": {"prefillMilliseconds": 12.5},
                "preflight": {"inputBytes": 120, "inputTokens": 42},
            }

    state = dispatch("workflow-run", coverage(tmp_path), backend=Metrics([json.dumps(proposal())]))
    saved = turns(tmp_path)[0]
    assert saved["metrics"]["inputTokens"] == 42
    assert saved["metrics"]["timing"]["prefillMilliseconds"] == 12.5
    call = state["steps"]["links"]["items"]["collar"]["calls"][0]
    assert call["metrics"] == saved["metrics"]


def test_review_decision_and_exact_content_commit_together(tmp_path):
    req = coverage(tmp_path)
    state = dispatch("workflow-run", req, backend=TextBackend([json.dumps(proposal())]))
    before = copy.deepcopy(state["steps"]["links"]["outputs"])
    after = dispatch(
        "workflow-review",
        {
            **req,
            "review": {
                "stepID": "links",
                "expectedRevision": state["revision"],
                "action": "approve",
                "itemID": "collar",
            },
        },
        backend=TextBackend([]),
    )
    decision = after["humanDecisions"][-1]
    assert decision["actor"] == "human" and decision["action"] == "approve"
    assert decision["before"]["outputs"] == decision["after"]["outputs"] == before
    assert decision["before"]["items"]["collar"]["approved"] is False
    assert decision["after"]["items"]["collar"]["approved"] is True
    assert decision["trainingConsent"] == "not_granted"
    assert decision["parentRevision"] == state["revision"]
    assert json.loads((tmp_path / "run.json").read_text()) == after
    with pytest.raises(ValueError, match="revision"):
        dispatch(
            "workflow-review",
            {
                **req,
                "review": {
                    "stepID": "links",
                    "expectedRevision": state["revision"],
                    "action": "unapprove",
                },
            },
            backend=TextBackend([]),
        )
    assert json.loads((tmp_path / "run.json").read_text()) == after


def test_multilingual_brief_stops_before_backend_without_retry(tmp_path):
    from wee_todd_mlx.workflows.service import builtin

    req = {
        "definition": builtin("guided-movie-planning"),
        "runDirectory": str(tmp_path),
        "inputs": {"brief": "猫は町を歩く。" * 1500},
    }
    backend = TextBackend([])
    state = dispatch("workflow-run", req, backend=backend)
    assert state["status"] == "failed"
    assert "UTF-8" in state["error"] and "split" in state["error"]
    assert backend.prompts == []


def test_typed_context_failure_stops_coverage_without_advisory_success(tmp_path):
    from wee_todd_mlx.workflows.context_budget import ContextBudgetError

    backend = TextBackend([ContextBudgetError("split the scene")])
    state = dispatch("workflow-run", coverage(tmp_path), backend=backend)
    assert state["status"] == "failed" and len(backend.prompts) == 1
    assert "outputs" not in state["steps"]["links"]


def test_inventory_context_keeps_all_identities_and_exact_relevant_facts():
    from wee_todd_mlx.workflows.context_budget import inventory_context

    rows = [
        {
            **inventory()[1],
            "id": f"item_{i}",
            "name": f"品物{i}号",
            "description": "猫" * 500,
            "aliases": [],
        }
        for i in range(24)
    ]
    current = {**inventory()[0], "description": "Dog carries 品物7号."}
    context = inventory_context(rows, current, "Dog carries 品物7号.")
    assert [r["id"] for r in context["rows"]] == [r["id"] for r in rows]
    assert context["rows"][7]["description"] == rows[7]["description"]
    assert "description" not in context["rows"][0]
    assert context["omittedAppearanceIDs"] == [r["id"] for r in rows if r["id"] != "item_7"]
    assert len(json.dumps(context, ensure_ascii=False).encode()) < 10000


def test_relevant_source_overflow_is_explicit_and_multilingual_names_match():
    from wee_todd_mlx.workflows.context_budget import ContextBudgetError
    from wee_todd_mlx.workflows.description_review import relevant_passages

    subject = {**inventory()[0], "name": "美咲", "aliases": [], "evidence": []}
    assert relevant_passages(subject, "美咲は猫と歩く。") == {1: "美咲は猫と歩く。"}
    with pytest.raises(ContextBudgetError, match="split"):
        relevant_passages(subject, "\n".join("美咲は" + "猫" * 650 for _ in range(12)))


@pytest.mark.parametrize(
    "code",
    [
        "text_context_too_long",
        "text_input_bytes_exceeded",
        "text_model_unavailable",
        "text_model_invalid_store",
    ],
)
def test_helper_failures_are_actionable_nonretryable(tmp_path, monkeypatch, code):
    from wee_todd_mlx.workflows.backend import LocalQwenBackend
    from wee_todd_mlx.workflows.context_budget import NonRetryableAssistantError

    checkpoint = tmp_path / "qwen_3.5_4b_i8x.ckpt"
    checkpoint.write_bytes(b"SQLite format 3\0")
    backend = LocalQwenBackend({"assistant": str(checkpoint)}, {}, "/bin/echo")
    model = {
        "id": "assistant",
        "runtime": "drawthings-qwen-local",
        "family": "qwen3.5",
        "variant": "4b",
    }

    def fail(*args, **kwargs):
        raise RuntimeError(f"Draw Things request failed ({code})")

    monkeypatch.setattr("wee_todd_mlx.workflows.backend.invoke_helper", fail)
    with pytest.raises(NonRetryableAssistantError) as caught:
        backend.generate(
            model, "Write", "A fox", [], cancelled=lambda: False, timeout=1, max_tokens=64
        )
    assert "split" in str(caught.value) or "model" in str(caught.value)


def test_subject_edit_saves_references_and_lineage_in_one_revision(tmp_path):
    req = coverage(tmp_path)
    state = dispatch("workflow-run", req, backend=TextBackend([json.dumps(proposal())]))
    outputs = copy.deepcopy(state["steps"]["links"]["outputs"])
    outputs["subjects"][0].update(description="A blue collar.", referenceAssets=["reference"])
    outputs["subjects"][0].pop("descriptionMentions", None)
    outputs["subjects"][0].pop("mentionSourceDescription", None)
    image = tmp_path / "reference.png"
    image.write_bytes(b"image placeholder; this action checks binding, not model decoding")
    backend = TextBackend([])
    backend.assets = {"reference": str(image)}
    updated = dispatch(
        "workflow-review",
        {
            **req,
            "review": {
                "stepID": "links",
                "expectedRevision": state["revision"],
                "action": "edit",
                "outputs": outputs,
            },
        },
        backend=backend,
    )
    assert updated["steps"]["links"]["outputs"] == outputs
    assert updated["humanDecisions"][-1]["after"]["outputs"] == outputs
    assert backend.prompts == []


def test_failed_review_save_publishes_neither_decision_nor_artifact(tmp_path, monkeypatch):
    from wee_todd_mlx.workflows.runner import WorkflowRunner

    req = coverage(tmp_path)
    state = dispatch("workflow-run", req, backend=TextBackend([json.dumps(proposal())]))
    before = (tmp_path / "run.json").read_bytes()

    def fail(self):
        raise OSError("Disk unavailable")

    monkeypatch.setattr(WorkflowRunner, "_save", fail)
    with pytest.raises(OSError, match="Disk unavailable"):
        dispatch(
            "workflow-review",
            {
                **req,
                "review": {
                    "stepID": "links",
                    "expectedRevision": state["revision"],
                    "action": "approve",
                },
            },
            backend=TextBackend([]),
        )
    assert (tmp_path / "run.json").read_bytes() == before


def test_reference_edit_rejects_missing_asset_without_any_mutation(tmp_path):
    req = coverage(tmp_path)
    state = dispatch("workflow-run", req, backend=TextBackend([json.dumps(proposal())]))
    outputs = copy.deepcopy(state["steps"]["links"]["outputs"])
    outputs["subjects"][0]["referenceAssets"] = ["missing"]
    before = (tmp_path / "run.json").read_bytes()
    with pytest.raises(ValueError, match="Relink"):
        dispatch(
            "workflow-review",
            {
                **req,
                "review": {
                    "stepID": "links",
                    "expectedRevision": state["revision"],
                    "action": "edit",
                    "outputs": outputs,
                },
            },
            backend=TextBackend([]),
        )
    assert (tmp_path / "run.json").read_bytes() == before


def test_output_truncation_at_hard_limit_does_not_repeat_impossible_request(tmp_path):
    from test_workflow_activity import definition

    from wee_todd_mlx.workflows.runner import WorkflowRunner

    class Truncated(TextBackend):
        def generate(self, *args, **kwargs):
            return {**super().generate(*args, **kwargs), "truncated": True}

    backend = Truncated(["partial JSON", "partial JSON"])
    state = WorkflowRunner(definition(), tmp_path, backend).run({})
    assert state["status"] == "failed" and len(backend.prompts) == 1
    assert "split" in state["error"] and "increase" not in state["error"]


def test_review_lineage_links_exact_execution_and_model_turns(tmp_path):
    req = coverage(tmp_path)
    state = dispatch("workflow-run", req, backend=TextBackend([json.dumps(proposal())]))
    after = dispatch(
        "workflow-review",
        {
            **req,
            "review": {
                "stepID": "links",
                "expectedRevision": state["revision"],
                "action": "approve",
            },
        },
        backend=TextBackend([]),
    )
    decision = after["humanDecisions"][-1]
    assert decision["executionKey"] == state["steps"]["links"]["key"]
    assert decision["modelTurnIDs"] == [turns(tmp_path)[0]["id"]]


def test_model_review_never_publishes_new_artifact_before_decision(tmp_path, monkeypatch):
    from test_workflow_subject_links import empty_proposal, request

    from wee_todd_mlx.workflows.runner import WorkflowRunner

    req = request(tmp_path)
    req["inputs"]["inventory"] = [inventory()[1]]
    state = dispatch("workflow-run", req, backend=TextBackend([json.dumps(empty_proposal())]))
    original_outputs = copy.deepcopy(state["steps"]["links"]["outputs"])
    saved = []
    save = WorkflowRunner._save

    def capture(self):
        save(self)
        saved.append(json.loads((tmp_path / "run.json").read_text()))

    monkeypatch.setattr(WorkflowRunner, "_save", capture)
    result = dispatch(
        "workflow-review",
        {
            **req,
            "review": {
                "stepID": "links",
                "expectedRevision": state["revision"],
                "action": "review_object_coverage",
            },
        },
        backend=TextBackend([json.dumps(proposal())]),
    )
    assert result["steps"]["links"]["outputs"] != original_outputs
    for snapshot in saved:
        if snapshot["steps"]["links"]["outputs"] != original_outputs:
            assert snapshot["humanDecisions"][-1]["action"] == "review_object_coverage"


def test_full_inventory_individual_approvals_keep_bounded_exact_lineage(tmp_path):
    from wee_todd_mlx.workflows.runner import digest

    req = coverage_request(tmp_path)
    req["inputs"]["inventory"] = [
        {
            **inventory()[1],
            "id": f"object_{i}",
            "name": f"Object {i}",
            "aliases": [],
            "description": "A carefully specified object. " * 50,
        }
        for i in range(24)
    ]
    state = dispatch("workflow-run", req, backend=TextBackend([json.dumps(proposal())] * 24))
    assert state["status"] == "awaiting_approval"
    for row in req["inputs"]["inventory"]:
        prior_outputs = copy.deepcopy(state["steps"]["links"]["outputs"])
        prior_revision = state["revision"]
        state = dispatch(
            "workflow-review",
            {
                **req,
                "review": {
                    "stepID": "links",
                    "expectedRevision": prior_revision,
                    "action": "approve",
                    "itemID": row["id"],
                },
            },
            backend=TextBackend([]),
        )
    assert len(state["humanDecisions"]) == 24
    for decision in state["humanDecisions"]:
        assert decision["snapshotScope"] == "changed_records"
        assert [s["id"] for s in decision["before"]["outputs"]["subjects"]] == [decision["itemID"]]
        assert decision["before"]["outputs"] == decision["after"]["outputs"]
        assert (
            decision["beforeContentDigest"]
            == decision["afterContentDigest"]
            == digest(prior_outputs)
        )
    assert state["humanDecisions"][-1]["parentRevision"] == prior_revision
    assert all(item["approved"] for item in state["steps"]["links"]["items"].values())
    assert (tmp_path / "run.json").stat().st_size < 1_000_000
