"""Real review documents may exceed the smaller untrusted definition import limit."""

import copy
import json

import pytest
from test_studio_workflow_execution import TextBackend
from test_workflow_subject_inventory import mutate_inventory, review_inventory

from wee_todd_mlx.workflows.io import load_document
from wee_todd_mlx.workflows.runner import WorkflowRunner
from wee_todd_mlx.workflows.service import dispatch


def test_large_review_checkpoint_preserves_edits_and_resumes_without_model_calls(tmp_path):
    req, state = review_inventory(tmp_path)
    subject = state["steps"]["subjects"]["outputs"]["subjects"][0]
    # Prior whole-step review journals legitimately retained entire inventories.
    snapshot = {"outputs": {"subjects": [
        {**subject, "id": f"character_{index}", "description": "Weathered warrior. " * 100}
        for index in range(24)
    ]}}
    decisions = [{"action": "edit", "before": snapshot, "after": snapshot} for _ in range(30)]
    state["humanDecisions"] = decisions
    runner = WorkflowRunner(state["definition"], tmp_path, TextBackend([]))
    runner.state = state
    runner._save()
    assert (tmp_path / "run.json").stat().st_size > 2 * 1024 * 1024
    outputs = copy.deepcopy(state["steps"]["subjects"]["outputs"])
    outputs["subjects"][0]["description"] = "A weathered dog with one brown eye."
    edited = mutate_inventory(req, state, "edit", outputs=outputs)
    assert edited["humanDecisions"][:30] == decisions
    assert edited["steps"]["subjects"]["outputs"] == outputs
    resumed = dispatch("workflow-run", req, backend=TextBackend([]))
    assert resumed["steps"]["subjects"]["outputs"] == outputs
    assert resumed["humanDecisions"] == edited["humanDecisions"]
    with pytest.raises(ValueError, match="2 MiB"):
        load_document(tmp_path / "run.json")


def test_checkpoint_budget_failure_preserves_previous_file(tmp_path):
    _, state = review_inventory(tmp_path)
    runner = WorkflowRunner(state["definition"], tmp_path, TextBackend([]))
    runner.state = state
    runner.limits = {**runner.limits, "maxWorkingBytes": 1024}
    before = (tmp_path / "run.json").read_bytes()
    with pytest.raises(ValueError, match="budget"):
        runner._save()
    assert (tmp_path / "run.json").read_bytes() == before
    assert not (tmp_path / "run.json.tmp").exists()


def test_checkpoint_reader_rejects_oversize_and_symlink(tmp_path):
    from wee_todd_mlx.workflows.io import load_checkpoint

    source = tmp_path / "run.json"
    source.write_text(json.dumps({"payload": "x" * (16 * 1024 * 1024)}))
    with pytest.raises(ValueError, match="checkpoint"):
        load_checkpoint(source)
    source.write_text('{"format":"weetodd-workflow-run-v1"}')
    link = tmp_path / "linked.json"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        load_checkpoint(link)


def test_checkpoint_read_honors_current_and_saved_working_budgets(tmp_path):
    from wee_todd_mlx.workflows.io import load_checkpoint

    source = tmp_path / "run.json"
    source.write_text(json.dumps({"definition": {"limits": {"maxWorkingBytes": 100}},
                                 "payload": "x" * 1000}))
    with pytest.raises(ValueError, match="budget"):
        load_checkpoint(source)
    source.write_text(json.dumps({"payload": "x" * 1000}))
    with pytest.raises(ValueError, match="budget"):
        load_checkpoint(source, limits={"maxWorkingBytes": 100})


@pytest.mark.parametrize("contents", ['{"a":1,"a":2}', '{"a":NaN}', '[' * 66 + '0' + ']' * 66])
def test_checkpoint_reader_retains_strict_json_checks(tmp_path, contents):
    from wee_todd_mlx.workflows.io import load_checkpoint

    source = tmp_path / "run.json"
    source.write_text(contents)
    with pytest.raises(ValueError):
        load_checkpoint(source)


def test_checkpoint_and_transcript_allocations_cannot_spend_the_same_disk_budget(tmp_path):
    from test_workflow_activity import Backend, answers, definition

    d = definition()
    d["limits"]["maxWorkingBytes"] = 8 * 1024 * 1024
    runner = WorkflowRunner(d, tmp_path, Backend(answers()))
    runner.run({})
    runner.turn_log.start(runner.specs["plan"], "x" * 1_500_000, "input", [], "large")
    runner.state["reviewEvidence"] = "x" * 1_800_000
    runner._save()
    previous = (tmp_path / "run.json").read_bytes()
    runner.state["reviewEvidence"] += "x" * 400_000
    with pytest.raises(ValueError, match="budget"):
        runner._save()
    assert (tmp_path / "run.json").read_bytes() == previous
    # Each file's reserved twin (atomic old/new and database/rollback) fits together.
    assert 2 * sum((tmp_path / name).stat().st_size for name in
                   ("run.json", "model-turns.sqlite")) < 8 * 1024 * 1024


def test_existing_transcript_storage_is_counted_before_checkpoint_replacement(tmp_path):
    _, state = review_inventory(tmp_path)
    runner = WorkflowRunner(state["definition"], tmp_path, TextBackend([]))
    runner.state = state
    runner.limits = {**runner.limits, "maxWorkingBytes": 5 * 1024 * 1024}
    before = (tmp_path / "run.json").read_bytes()
    # A previous definition may have allowed more transcript storage.
    (tmp_path / "model-turns.sqlite").write_bytes(b"x" * (2 * 1024 * 1024))
    with pytest.raises(ValueError, match="budget"):
        runner._save()
    assert (tmp_path / "run.json").read_bytes() == before
