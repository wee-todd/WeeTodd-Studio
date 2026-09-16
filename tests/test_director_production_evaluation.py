"""Objective scoring must detect source loss and collateral shot changes."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "director_production_evaluation", ROOT / "scripts/evaluate_director_production.py"
)


def harness():
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    return module


def test_frozen_suite_rejects_modified_source(tmp_path):
    import pytest

    evaluation = harness()
    suite = ROOT / "examples/director-production-evaluation/sessions.json"
    destination = tmp_path / "frozen"
    digest = evaluation.freeze_suite(suite, destination)
    assert len(evaluation.load_suite(destination / "suite.json", digest)["sessions"]) == 2
    changed = json.loads((destination / "suite.json").read_text())
    changed["sessions"][0]["inputs"]["brief"] += " altered"
    (destination / "suite.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="digest"):
        evaluation.load_suite(destination / "suite.json", digest)


def test_scoring_detects_dialogue_identity_and_timing_loss():
    evaluation = harness()
    case = {
        "id": "probe",
        "inputs": {
            "brief": "Mara says “It’s here.”",
            "frame_rate": 24,
            "duration_seconds": 10,
            "target_clip_seconds": 5,
        },
        "expected": {
            "shots": [
                {"dialogue": ["It’s here."], "actionTerms": ["opens"]},
                {"dialogue": [], "actionTerms": ["closes"]},
            ],
            "subjects": [{"name": "Mara", "kind": "character", "terms": ["human"]}],
        },
    }
    subject = {"id": "mara", "name": "Mara", "kind": "character", "description": "Human."}
    character = {"id": "mara", "description": "Human."}
    clips = [
        {
            "id": "clip-1",
            "startFrame": 0,
            "frameCount": 120,
            "action": "Mara opens. It’s here.",
            "characters": ["mara"],
        },
        {
            "id": "clip-2",
            "startFrame": 120,
            "frameCount": 120,
            "action": "Mara closes.",
            "characters": ["mara"],
        },
    ]
    state = {
        "status": "completed",
        "steps": {
            "creative_brief": {
                "outputs": {
                    "creative_brief": {
                        "sourceText": case["inputs"]["brief"],
                        "facts": [
                            {"text": case["inputs"]["brief"], "evidence": case["inputs"]["brief"]}
                        ],
                    }
                }
            },
            "subjects_coverage": {"outputs": {"subjects": [subject]}},
            "story": {
                "outputs": {
                    "story": {
                        "characters": [character],
                        "beats": ["Mara opens. It’s here.", "Mara closes."],
                    }
                }
            },
            "clips": {
                "outputs": {
                    "clips": {
                        "fps": 24,
                        "totalFrames": 240,
                        "characters": [character],
                        "clips": clips,
                    }
                }
            },
        },
    }
    assert evaluation.score_session(case, state)["violations"] == []
    broken = copy.deepcopy(state)
    plan = broken["steps"]["clips"]["outputs"]["clips"]
    plan["characters"][0]["description"] = "Robot."
    plan["clips"][0]["action"] = "Mara opens. It's here."
    plan["clips"][1]["startFrame"] = 121
    violations = evaluation.score_session(case, broken)["violations"]
    assert "clips.shot-1.dialogue:It’s here." in violations
    assert "clips.approved_characters" in violations
    assert "clips.timing" in violations


def test_dialogue_scoring_rejects_extra_and_duplicate_quotes():
    evaluation = harness()
    expected = ["It’s here."]
    assert evaluation.dialogue_violations("Mara says “It’s here.”", expected) == []
    assert evaluation.dialogue_violations(
        'Mara declares "It\'s here." and says “It’s here.”', expected
    ) == ["unexpected:It's here."]
    assert evaluation.dialogue_violations(
        "Mara says “It’s here.” then repeats “It’s here.”", expected
    ) == ["duplicate:It’s here."]
    assert evaluation.dialogue_violations("Mara's reply is 'It's here.'", expected) == [
        "unexpected:It's here."
    ]
    assert evaluation.dialogue_violations("Mara’s reply is ‘It’s here.’", expected) == []


def test_action_repair_score_detects_collateral_fields_and_unrelated_shots():
    evaluation = harness()
    before = {
        "fps": 24,
        "totalFrames": 240,
        "characters": [],
        "clips": [
            {"id": "clip-1", "action": "Opens", "endState": "Open", "frameCount": 120},
            {"id": "clip-2", "action": "Closes", "endState": "Closed", "frameCount": 120},
        ],
    }
    after = copy.deepcopy(before)
    after["clips"][0]["action"] = "Opens gently"
    assert evaluation.score_repair(before, after, "clip-1", "Opens gently")["violations"] == []
    after["clips"][0]["endState"] = "Different"
    after["clips"][1]["action"] = "Changed"
    violations = evaluation.score_repair(before, after, "clip-1", "Opens gently")["violations"]
    assert "repair.unselected:endState" in violations
    assert "repair.unrelated:clip-2" in violations


def test_default_cli_is_model_free():
    result = subprocess.run([sys.executable, str(SPEC.origin)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["modelCalls"] == 0


def test_driver_records_structural_approval_using_real_runner(tmp_path):
    from types import SimpleNamespace

    from wee_todd_mlx.workflows.service import builtin, dispatch

    evaluation = harness()
    case = evaluation.load_suite(evaluation.DEFAULT_SUITE)["sessions"][0]
    definition = builtin("guided-movie-planning")
    definition["steps"] = definition["steps"][:2]
    definition["outputs"] = {
        "creative_brief": {"step": "creative_brief", "output": "creative_brief"}
    }

    class BoundaryBackend:
        def __init__(self, *args):
            self.assets = {}

        def fingerprint(self, *args):
            return {"fixture": True}

        def generate(self, *args, **kwargs):
            return {"text": '{"questions":[]}', "totalSeconds": 0}

    args = SimpleNamespace(
        output=tmp_path,
        max_calls=4,
        session_seconds=10,
        call_seconds=10,
        max_tokens=1024,
        model=Path("/test-model"),
        helper=Path("/test-helper"),
        repair_scope="action",
    )
    result = evaluation.run_session(case, args, dispatch, lambda name: definition, BoundaryBackend)
    assert result["objective"]["complete"]
    assert result["scriptedApprovals"] == 1
    assert result["weightedCalls"] == result["modelTurns"] == 1
    assert result["humanCreativeAcceptance"] == "not_measured"
    journal = json.loads((tmp_path / case["id"] / "evaluation-decisions.json").read_text())
    assert journal[0]["creativeAcceptance"] is False


@pytest.mark.parametrize("failure", ["exception", "stale_result"])
def test_correction_error_retains_completed_first_run(tmp_path, failure):
    from types import SimpleNamespace

    evaluation = harness()
    case = evaluation.load_suite(evaluation.DEFAULT_SUITE)["sessions"][0]
    plan = {
        "fps": 24,
        "totalFrames": 240,
        "characters": [],
        "clips": [
            {
                "id": "clip-1",
                "startFrame": 0,
                "frameCount": 120,
                "action": "Opens",
                "characters": [],
            },
            {
                "id": "clip-2",
                "startFrame": 120,
                "frameCount": 120,
                "action": case["correction"]["action"],
                "characters": [],
            },
        ],
    }
    state = {
        "revision": "initial",
        "status": "completed",
        "steps": {"clips": {"status": "completed", "outputs": {"clips": plan}}},
    }

    class BoundaryBackend:
        def __init__(self, *args):
            pass

    run_count = 0

    def dispatch(command, request, **kwargs):
        nonlocal run_count
        if command == "workflow-review":
            if request["review"]["action"] == "repair" and failure == "exception":
                raise ValueError("Correction unavailable")
            return {**state, "status": "paused", "revision": "unlocked"}
        run_count += 1
        if run_count > 1:
            return {**state, "status": "failed", "error": "Correction unavailable"}
        return copy.deepcopy(state)

    args = SimpleNamespace(
        output=tmp_path,
        max_calls=4,
        session_seconds=10,
        call_seconds=10,
        max_tokens=1024,
        model=Path("/test-model"),
        helper=Path("/test-helper"),
        repair_scope="action",
    )
    result = evaluation.run_session(case, args, dispatch, lambda name: {}, BoundaryBackend)
    assert result["objective"]["complete"] is True
    expected = "repair.execution_error" if failure == "exception" else "repair.no_completed_result"
    assert result["repair"]["violations"] == [expected]
    assert result["repair"]["error"] == "Correction unavailable"
