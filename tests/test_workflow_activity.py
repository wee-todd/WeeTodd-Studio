import json

from wee_todd_mlx.workflows.runner import WorkflowRunner
from wee_todd_mlx.workflows.service import builtin


class Backend:
    def __init__(self, replies):
        self.replies = iter(replies)

    def fingerprint(self, model, images):
        return {"fixture": True}

    def generate(self, *args, **kwargs):
        result = next(self.replies)
        if isinstance(result, Exception):
            raise result
        return {"text": result, "totalSeconds": 0.01}


def definition():
    d = builtin("staged-prompt-editing")
    return d


def answers():
    return [
        json.dumps([{"id": "fox", "instruction": "Use a fox", "preserve": [], "useImages": False}]),
        "A fox.",
        json.dumps({"status": "pass", "items": []}),
    ]


def test_history_preserves_retry_and_resume_without_claiming_new_calls(tmp_path):
    runner = WorkflowRunner(definition(), tmp_path, Backend(["bad json", *answers()]))
    first = runner.run({})
    history = first["executionHistory"]
    assert [e["stepID"] for e in history] == ["describe", "plan", "edit", "check"]
    plan = history[1]
    assert plan["modelCalls"] == 2 and plan["retries"] == 1
    assert any(e["kind"] == "retry" and "JSON" in e["message"] for e in plan["events"])
    assert plan["status"] == "completed" and plan["seconds"] >= 0
    resumed = runner.run({})
    assert all(
        e["status"] == "reused" and e["modelCalls"] == 0 for e in resumed["executionHistory"][-4:]
    )
    assert resumed["executionHistory"][1] == plan


def test_failed_calls_record_time_and_cancellation_then_resume(tmp_path):
    runner = WorkflowRunner(
        definition(), tmp_path, Backend([answers()[0], InterruptedError("paused"), *answers()[1:]])
    )
    result = runner.run({})
    failed = result["executionHistory"][-1]
    assert failed["stepID"] == "edit" and failed["status"] == "cancelled"
    assert failed["modelCalls"] == 1 and failed["seconds"] >= 0
    assert failed["events"][-1]["status"] == "cancelled"
    assert result["steps"]["edit"]["seconds"] >= 0
    assert runner.run({})["status"] == "completed"


def test_history_bounds_do_not_grow_forever(tmp_path):
    runner = WorkflowRunner(definition(), tmp_path, Backend(answers()))
    runner.run({})
    for _ in range(25):
        result = runner.run({})
    assert len(result["executionHistory"]) <= 40
    assert result["historyOmitted"] > 0


def test_nested_object_calls_are_counted_and_explicit_reviews_are_separate(tmp_path):
    from test_workflow_object_coverage import coverage_request, proposal
    from test_workflow_subject_links import run

    req = coverage_request(tmp_path)
    state = run(req, [proposal(), proposal()])
    assert state["steps"]["links"]["calls"] == []
    assert state["executionHistory"][-1]["modelCalls"] == 2
    assert len(state["executionHistory"][-1]["events"]) == 2
    from test_studio_workflow_execution import TextBackend

    from wee_todd_mlx.workflows.service import dispatch

    state = dispatch(
        "workflow-review",
        {
            **req,
            "review": {
                "action": "review_object_coverage",
                "stepID": "links",
                "expectedRevision": state["revision"],
                "library": [],
            },
        },
        backend=TextBackend([json.dumps(proposal()), json.dumps(proposal())]),
    )
    assert state["executionHistory"][-1]["reason"] == "review_object_coverage"
