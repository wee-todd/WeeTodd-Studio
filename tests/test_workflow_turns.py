import json
import sqlite3

from test_workflow_activity import Backend, answers, definition

from wee_todd_mlx.workflows.runner import WorkflowRunner


def turns(directory):
    with sqlite3.connect(directory / "model-turns.sqlite") as db:
        return [json.loads(row[0]) for row in db.execute("SELECT detail FROM turns ORDER BY rowid")]


def test_full_turns_keep_rejected_response_and_exact_repair_request(tmp_path):
    runner = WorkflowRunner(definition(), tmp_path, Backend(["bad json", *answers()]))
    result = runner.run({})
    records = turns(tmp_path)
    assert result["status"] == "completed"
    assert len(records) == 4
    assert records[0]["response"] == "bad json"
    assert records[0]["validation"]["status"] == "failed"
    assert "JSON" in records[0]["validation"]["error"]
    assert "Previous attempt failed validation" in records[1]["prompt"]
    assert records[1]["system"].startswith("Split the editing instructions")
    assert records[1]["validation"]["status"] == "passed"
    assert records[1]["settings"]["maxTokens"] == 1024
    assert records[1]["model"]["family"] == "qwen3.5"
    runner.run({})
    assert len(turns(tmp_path)) == 4  # Whole-step reuse makes no model turn.


def test_cancelled_and_reused_turns_are_preserved(tmp_path):
    plan = json.dumps(
        [
            {"id": "one", "instruction": "Use a fox", "preserve": [], "useImages": False},
            {"id": "two", "instruction": "Use snow", "preserve": [], "useImages": False},
        ]
    )
    runner = WorkflowRunner(
        definition(),
        tmp_path,
        Backend([plan, "A fox.", InterruptedError("cancelled"), "A fox in snow.", answers()[-1]]),
    )
    assert runner.run({})["status"] == "cancelled"
    assert turns(tmp_path)[-1]["status"] == "cancelled"
    assert turns(tmp_path)[-1]["response"] is None
    assert runner.run({})["status"] == "completed"
    reused = [r for r in turns(tmp_path) if r["kind"] == "reuse"]
    assert len(reused) == 1 and reused[0]["response"] == "A fox."
    assert reused[0]["prompt"] and reused[0]["system"]


def test_archive_survives_overview_event_trimming(tmp_path):
    runner = WorkflowRunner(definition(), tmp_path, Backend(answers() * 9))
    runner.run({})
    for _ in range(8):
        runner.run({}, regenerate="plan")
    assert len(turns(tmp_path)) == 27


def test_full_log_refuses_new_turn_before_dropping_existing_records(tmp_path):
    import pytest

    from wee_todd_mlx.workflows.io import MAX_DOCUMENT_BYTES

    d = definition()
    d["limits"]["maxWorkingBytes"] = 2 * MAX_DOCUMENT_BYTES + 512 * 1024
    runner = WorkflowRunner(d, tmp_path, Backend(answers()))
    runner.run({})
    before = turns(tmp_path)
    spec = runner.specs["plan"]
    with pytest.raises(ValueError, match="model-turn history"):
        runner.turn_log.start(spec, "x" * 200_000, "input", [], "key")
    assert turns(tmp_path) == before
    assert (tmp_path / "model-turns.sqlite").stat().st_size <= 256 * 1024


def test_interrupted_pending_turn_is_not_left_running_after_resume(tmp_path):
    runner = WorkflowRunner(definition(), tmp_path, Backend(answers()))
    runner.run({})
    first = runner.turn_log.start(runner.specs["plan"], "system", "prompt", [], "unfinished")
    runner.turn_log.start(runner.specs["plan"], "system", "prompt", [], "next")
    saved = next(t for t in turns(tmp_path) if t["id"] == first["id"])
    assert saved["status"] == "interrupted" and saved["response"] is None
