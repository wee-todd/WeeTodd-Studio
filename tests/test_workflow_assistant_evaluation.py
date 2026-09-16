"""Evaluation must reject wrong content, not merely valid JSON or a successful process."""

import importlib.util
import json
from pathlib import Path

import pytest

from wee_todd_mlx.assistant_evaluation import evaluate_response, load_suite, summarize


def cli():
    source = Path(__file__).parents[1] / "scripts/evaluate_studio_assistant.py"
    spec = importlib.util.spec_from_file_location("assistant_eval_cli", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_cli_never_calls_model(monkeypatch, capsys):
    module = cli()
    def unexpected(*args, **kwargs):
        pytest.fail("Default validation must not load weights")
    monkeypatch.setattr(module, "invoke_helper", unexpected)
    assert module.main([]) == 0
    assert json.loads(capsys.readouterr().out)["modelCalls"] == 0


def test_shape_prompt_never_uses_expected_answer():
    module = cli()
    item = {**case(), "referenceAnswer": {"dialogue": "secret expected text"}}
    instruction = module.system_instruction(item, "shape")
    assert '"dialogue":"<string>"' in instruction
    assert "secret expected text" not in instruction
    assert "Don't leave" not in instruction


@pytest.mark.parametrize("timeout", ["nan", "inf", "-1"])
def test_nonfinite_or_negative_timeout_is_rejected(timeout):
    with pytest.raises(SystemExit):
        cli().main(["--timeout", timeout])


def case():
    return {
        "id": "dialogue", "category": "dialogue", "group": "harbor", "split": "development",
        "system": "Return JSON.", "prompt": "Keep exact dialogue.",
        "schema": {"type": "object", "required": ["dialogue"],
                   "properties": {"dialogue": {"type": "string"}},
                   "additionalProperties": False},
        "assertions": [{"path": "/dialogue", "equals": "Don't leave."}],
    }


def test_valid_json_with_changed_dialogue_is_not_a_pass():
    result = evaluate_response(case(), {"text": '{"dialogue":"Do not leave."}'})
    assert result["structuralValid"] is True
    assert result["passed"] is False
    assert result["failedAssertions"] == ["/dialogue"]


def test_truncated_but_parseable_response_is_not_accepted():
    result = evaluate_response(case(), {"text": '{"dialogue":"Don\'t leave."}',
                                        "truncated": True})
    assert not result["passed"]
    assert result["error"] == "truncated"


def test_extra_fields_and_markdown_fail_the_structural_contract():
    assert not evaluate_response(case(), {"text": '{"dialogue":"Don\'t leave.","extra":1}'})[
        "structuralValid"]
    assert not evaluate_response(case(), {"text": '```json\n{"dialogue":"Don\'t leave."}\n```'})[
        "structuralValid"]


def test_summary_counts_errors_in_denominator_and_keeps_categories():
    rows = [
        {"id": "a", "category": "dialogue", "passed": True, "structuralValid": True,
         "wallSeconds": 1.0},
        {"id": "b", "category": "dialogue", "passed": False, "structuralValid": False,
         "wallSeconds": 3.0, "error": "runtime"},
    ]
    result = summarize(rows)
    assert result["attempted"] == 2
    assert result["passed"] == 1
    assert result["passRate"] == 0.5
    assert result["wallSeconds"]["p95"] == 3.0
    assert result["categories"]["dialogue"]["passed"] == 1


def test_suite_rejects_group_leakage_and_duplicate_ids(tmp_path):
    first = case()
    second = {**case(), "id": "other", "split": "holdout"}
    source = tmp_path / "suite.json"
    source.write_text(json.dumps({"format": "weetodd-assistant-evaluation-v1",
                                 "cases": [first, second]}))
    with pytest.raises(ValueError, match="group"):
        load_suite(source)
    second.update(split="development", id="dialogue")
    source.write_text(json.dumps({"format": "weetodd-assistant-evaluation-v1",
                                 "cases": [first, second]}))
    with pytest.raises(ValueError, match="Duplicate"):
        load_suite(source)


def test_shipped_cases_have_hand_checked_expectations_and_disjoint_groups():
    suite = load_suite(Path(__file__).parents[1] / "examples/assistant-evaluation/cases.json")
    assert len(suite) == 60
    assert {c["split"] for c in suite} == {"development", "holdout"}
    assert len({c["category"] for c in suite}) == 10
    for item in suite:
        assert item["assertions"]
        assert evaluate_response(item, {"text": json.dumps(item["referenceAnswer"],
                                                           ensure_ascii=False)})["passed"]


def test_exact_relationship_and_continuity_tokens_are_visible_in_case_inputs():
    source = Path(__file__).parents[1] / "examples/assistant-evaluation/cases.json"
    document = json.loads(source.read_text())
    assert document["corpusRevision"] == 2
    for item in document["cases"]:
        if item["category"] == "relationship_direction":
            inventory = json.loads(item["prompt"].split("Inventory JSON: ")[1].split("\n")[0])
            ids = {entry["id"] for entry in inventory}
            answer = item["referenceAnswer"]
            assert answer["source"] in ids and answer["target"] in ids
            assert 'Use only each entry\'s "id" value' in item["system"]
            assert answer["role"] in item["system"].split("Legal role tokens: ")[1]
        if item["category"] == "continuity":
            accepted = json.loads(item["prompt"].split("Accepted fields JSON: ")[1].split("\n")[0])
            proposed = json.loads(item["prompt"].split("Proposed fields JSON: ")[1].split("\n")[0])
            conflicts = [key for key in accepted.keys() & proposed.keys()
                         if accepted[key] != proposed[key]]
            expected = item["referenceAnswer"]["conflict"]
            assert conflicts == ([] if expected == "none" else [expected])
            assert 'lowercase string "none"' in item["system"]
