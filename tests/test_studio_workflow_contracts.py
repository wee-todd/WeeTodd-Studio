"""Portable user workflow contracts: validity never implies generation readiness."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from wee_todd_mlx.workflows import load_document, validate_document, validate_file

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/studio-workflows"


def workflow():
    return load_document(EXAMPLES / "staged-prompt-editing.json")


def adapter():
    return {
        "format": "weetodd-adapter-v1",
        "id": "example.prompt-editor",
        "version": "1.0.0",
        "name": "Prompt editor",
        "description": "Fixture metadata, not trained weights.",
        "purpose": "prompt-editing",
        "license": "Apache-2.0",
        "baseModel": {"family": "qwen3.5", "variant": "4b", "revision": "fixture-revision"},
        "weights": {
            "file": "adapter.safetensors",
            "format": "peft-safetensors",
            "sha256": hashlib.sha256(b"fixture").hexdigest(),
            "sizeBytes": 7,
        },
        "training": {"method": "qlora", "basePrecision": "nf4"},
        "adaptation": {"rank": 16, "alpha": 32, "targetModules": ["q_proj", "v_proj"]},
        "inference": {
            "runtimes": ["drawthings-qwen-local"],
            "defaultStrength": 1,
            "visionSupport": "unverified",
        },
    }


@pytest.mark.parametrize("name", ["staged-prompt-editing", "movie-planning"])
def test_shipped_examples_are_valid_definitions_but_not_executable(name):
    result = validate_file(EXAMPLES / f"{name}.json")
    assert result["valid"], result
    assert result["executionStatus"] == "requires_bindings"
    assert result["topologicalOrder"]


@pytest.mark.parametrize(
    "mutation,code",
    [
        (lambda w: w.update(format="weetodd-workflow-v2"), "schema"),
        (lambda w: w.update(command="python malicious.py"), "schema"),
        (lambda w: w["steps"][0].update(operation="user.shell@1"), "operation_unknown"),
        (lambda w: w["steps"][1].update(id=w["steps"][0]["id"]), "duplicate_id"),
        (lambda w: w["steps"][1]["inputs"].pop("instructions"), "input_ports"),
        (lambda w: w["steps"][1]["inputs"].update(instructions={"input": "missing"}), "reference"),
        (lambda w: w["steps"][1]["inputs"].update(source={"input": "images"}), "type_mismatch"),
        (
            lambda w: w["steps"][1]["inputs"].update(
                source={"step": "describe", "output": "missing"}
            ),
            "reference",
        ),
        (lambda w: w["steps"][2]["parameters"].update(maxEdits=100000), "parameters"),
        (lambda w: w["steps"][0].update(model="missing"), "model_reference"),
        (lambda w: w["models"]["assistant"].update(capabilities=["text"]), "model_capability"),
        (lambda w: w["steps"][0]["retry"].update(maxAttempts=0), "schema"),
        (lambda w: w["inputs"]["images"].update(default="/a.png"), "input_default"),
    ],
)
def test_invalid_contracts_report_actionable_issue(mutation, code):
    value = workflow()
    mutation(value)
    result = validate_document(value)
    assert not result["valid"]
    assert code in {i["code"] for i in result["issues"]}, result
    assert all(i["path"] and i["message"] for i in result["issues"])


def test_cycle_is_rejected_and_forward_references_can_be_sorted():
    value = workflow()
    value["steps"].reverse()
    assert validate_document(value)["valid"]
    value = workflow()
    value["steps"][1]["inputs"]["source"] = {"step": "edit", "output": "draft"}
    result = validate_document(value)
    assert not result["valid"]
    assert "cycle" in {i["code"] for i in result["issues"]}


def test_workflow_outputs_are_checked():
    value = workflow()
    value["outputs"]["prompt"] = {"step": "missing", "output": "draft"}
    assert not validate_document(value)["valid"]


def test_qlora_metadata_does_not_claim_qwen_runtime_support():
    result = validate_document(adapter())
    assert result["valid"], result
    assert result["executionStatus"] == "not_implemented"
    value = workflow()
    value["models"]["assistant"]["revision"] = "fixture-revision"
    value["steps"][1]["adapter"] = "example.prompt-editor"
    result = validate_document(value, adapters=[adapter()])
    assert result["valid"], result
    assert "adapter_runtime_unavailable" in {i["code"] for i in result["warnings"]}


def test_adapter_compatibility_is_exact_and_missing_manifest_is_not_ignored():
    value = workflow()
    value["steps"][1]["adapter"] = "example.prompt-editor"
    assert not validate_document(value)["valid"]
    value["models"]["assistant"]["revision"] = "fixture-revision"
    wrong = adapter()
    wrong["baseModel"]["variant"] = "9b"
    result = validate_document(value, adapters=[wrong])
    assert "adapter_base_mismatch" in {i["code"] for i in result["issues"]}
    wrong = adapter()
    wrong["baseModel"]["revision"] = "other-revision"
    assert not validate_document(value, adapters=[wrong])["valid"]


@pytest.mark.parametrize("filename", ["../a.safetensors", "/a.safetensors", "https://x/a", "a\\b"])
def test_adapter_artifact_is_a_portable_filename(filename):
    value = adapter()
    value["weights"]["file"] = filename
    assert not validate_document(value)["valid"]


def test_validation_does_not_mutate_user_data():
    value = workflow()
    before = copy.deepcopy(value)
    validate_document(value)
    assert value == before


@pytest.mark.parametrize("content", ['{"format":1,"format":2}', '{"x":NaN}', "[" * 300])
def test_import_rejects_ambiguous_or_malformed_json(tmp_path, content):
    source = tmp_path / "workflow.json"
    source.write_text(content)
    with pytest.raises(ValueError):
        load_document(source)


def test_import_is_bounded(tmp_path):
    source = tmp_path / "workflow.json"
    source.write_bytes(b" " * (2 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="2 MiB"):
        load_document(source)


def test_cli_reports_validity_and_fails_if_execution_is_required(tmp_path):
    cmd = [
        sys.executable,
        str(ROOT / "scripts/validate_studio_workflow.py"),
        str(EXAMPLES / "staged-prompt-editing.json"),
        "--json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["valid"]
    result = subprocess.run(
        cmd + ["--require-runnable"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    invalid = tmp_path / "bad.json"
    invalid.write_text('{"format":"future"}')
    result = subprocess.run(cmd[:2] + [str(invalid), "--json"], capture_output=True, text=True)
    assert result.returncode == 1
    assert not json.loads(result.stdout)["valid"]


def test_validator_does_not_import_weighted_runtimes():
    code = """
import sys
from wee_todd_mlx.workflows import validate_file
validate_file(sys.argv[1])
assert not any(n.startswith(('mlx', 'torch', 'comfy')) for n in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(EXAMPLES / "movie-planning.json")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_execution_budget_includes_each_edit_and_retry():
    value = workflow()
    value["limits"]["maxStepExecutions"] = 4
    result = validate_document(value)
    assert "execution_budget" in {i["code"] for i in result["issues"]}


def test_numeric_input_can_have_only_a_negative_maximum():
    value = workflow()
    value["inputs"]["seed"] = {"type": "integer", "label": "Sentinel", "maximum": -1, "default": -1}
    result = validate_document(value)
    assert result["valid"], result


def test_structured_clip_plan_has_exact_contiguous_frame_allocation():
    from wee_todd_mlx.workflows import validate_value

    clips = {
        "fps": 24,
        "totalFrames": 240,
        "clips": [
            {
                "id": "first",
                "startFrame": 0,
                "frameCount": 120,
                "action": "Arrive",
                "continuity": "cut",
            },
            {
                "id": "second",
                "startFrame": 120,
                "frameCount": 120,
                "action": "Leave",
                "continuity": "continue",
            },
        ],
    }
    assert validate_value("clip_plan", clips) == []
    clips["clips"][1]["startFrame"] = 119
    assert validate_value("clip_plan", clips)[0]["code"] == "frame_allocation"
    clips["clips"][1]["startFrame"] = 120
    clips["totalFrames"] = 241
    assert validate_value("clip_plan", clips)[0]["path"] == "$/totalFrames"


def test_registry_port_types_and_schema_resources_are_consistent():
    from wee_todd_mlx.workflows.schema import contracts

    schemas, _, operations = contracts()
    for operation in operations.values():
        for group in ("inputs", "outputs"):
            assert set(operation[group].values()) <= set(schemas["values"]["$defs"])
        assert operation["executionStatus"] == "implemented"


@pytest.mark.parametrize(
    "change,code",
    [
        (lambda a: a["weights"].update(format="drawthings-ckpt"), "weights_format"),
        (lambda a: a["training"].update(basePrecision="bf16"), "training_precision"),
    ],
)
def test_adapter_metadata_does_not_confuse_training_and_weight_formats(change, code):
    value = adapter()
    change(value)
    result = validate_document(value)
    assert code in {i["code"] for i in result["issues"]}


def test_9b_vision_requirement_is_not_advertised_as_supported():
    value = workflow()
    value["models"]["assistant"]["variant"] = "9b"
    result = validate_document(value)
    assert "model_runtime" in {i["code"] for i in result["issues"]}


def test_endpoint_planning_budget_counts_each_clip():
    value = load_document(EXAMPLES / "movie-planning.json")
    value["limits"]["maxStepExecutions"] = 100
    result = validate_document(value)
    assert "execution_budget" in {i["code"] for i in result["issues"]}
