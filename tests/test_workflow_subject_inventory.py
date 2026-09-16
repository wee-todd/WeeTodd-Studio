"""Subject extraction keeps literal evidence and requires review before reuse."""

import copy
import json

import pytest
from test_studio_workflow_execution import TextBackend

from wee_todd_mlx.workflows.service import builtin, dispatch
from wee_todd_mlx.workflows.validation import validate_document


def review_inventory(tmp_path):
    backend = TextBackend([json.dumps([selection()]), "[]", "[]"])
    req = {"builtin": "subject-inventory", "inputs": {"brief": "Dog has one brown eye."},
           "runDirectory": str(tmp_path)}
    state = dispatch("workflow-run", req, backend=backend)
    return req, state


def mutate_inventory(req, state, action, **kwargs):
    return dispatch("workflow-review", {**req, "review": {
        "stepID": "subjects", "expectedRevision": state["revision"], "action": action, **kwargs,
    }})


def test_subject_names_and_descriptions_edit_without_changing_identity(tmp_path):
    req, state = review_inventory(tmp_path)
    outputs = copy.deepcopy(state["steps"]["subjects"]["outputs"])
    original = copy.deepcopy(outputs["subjects"][0])
    outputs["subjects"][0].update(name="Scout", description="A corgi-sized cyborg dog.")
    state = mutate_inventory(req, state, "edit", outputs=outputs)
    edited = state["steps"]["subjects"]["outputs"]["subjects"][0]
    assert edited == {**original, "name": "Scout", "description": "A corgi-sized cyborg dog."}
    state = mutate_inventory(req, state, "approve", itemID=original["id"])
    assert state["steps"]["subjects"]["items"][original["id"]]["approved"]
    before = (tmp_path / "run.json").read_bytes()
    outputs["subjects"][0]["description"] = "Different appearance"
    with pytest.raises(ValueError, match="Unlock"):
        mutate_inventory(req, state, "edit", outputs=outputs)
    assert (tmp_path / "run.json").read_bytes() == before
    state = mutate_inventory(req, state, "unapprove", itemID=original["id"])
    state = mutate_inventory(req, state, "edit", outputs=outputs)
    assert not state["steps"]["subjects"]["items"][original["id"]]["approved"]


@pytest.mark.parametrize("field,value", [
    ("id", "new_id"), ("kind", "prop"), ("evidence", ["Invented source"]),
    ("suggestions", ["Invented suggestion"]), ("aliases", ["Changed alias"]),
])
def test_subject_source_fields_are_read_only(tmp_path, field, value):
    req, state = review_inventory(tmp_path)
    before = (tmp_path / "run.json").read_bytes()
    outputs = copy.deepcopy(state["steps"]["subjects"]["outputs"])
    outputs["subjects"][0][field] = value
    with pytest.raises(ValueError, match="read.only|identity"):
        mutate_inventory(req, state, "edit", outputs=outputs)
    assert (tmp_path / "run.json").read_bytes() == before


def test_editing_another_subject_preserves_individual_approval(tmp_path):
    second = {**selection(), "name": "Pilot"}
    backend = TextBackend([json.dumps([selection(), second]), "[]", "[]"])
    req = {"builtin": "subject-inventory", "inputs": {"brief": "A dog and pilot."},
           "runDirectory": str(tmp_path)}
    state = dispatch("workflow-run", req, backend=backend)
    outputs = copy.deepcopy(state["steps"]["subjects"]["outputs"])
    first_id = outputs["subjects"][0]["id"]
    state = mutate_inventory(req, state, "approve", itemID=first_id)
    outputs["subjects"][1]["description"] = "A pilot in a red jacket."
    state = mutate_inventory(req, state, "edit", outputs=outputs)
    assert state["steps"]["subjects"]["items"][first_id]["approved"]
    assert not state["steps"]["subjects"]["items"][outputs["subjects"][1]["id"]]["approved"]


def subject(kind="character", evidence="Dog has one brown eye."):
    return {
        "id": "dog",
        "kind": kind,
        "name": "Dog",
        "aliases": ["cyborg dog"],
        "description": "One brown eye and one optical eye.",
        "evidence": [evidence],
        "suggestions": [],
    }


def selection(kind="character", evidence_ids=None):
    item = subject(kind)
    item.pop("evidence")
    item["evidenceIDs"] = [1] if evidence_ids is None else evidence_ids
    return item


def test_inventory_is_bounded_and_preserves_source(tmp_path):
    brief = "The cyborg dog has one brown eye and one optical eye."
    backend = TextBackend([json.dumps([selection()]), "[]", "[]"])
    run = dispatch(
        "workflow-run",
        {"builtin": "subject-inventory", "inputs": {"brief": brief}, "runDirectory": str(tmp_path)},
        backend=backend,
    )
    assert run["status"] == "awaiting_approval", run.get("error")
    assert run["inputs"]["brief"] == brief
    result = run["steps"]["subjects"]["outputs"]["subjects"][0]
    assert result["id"].startswith("character_")
    assert {k: v for k, v in result.items() if k != "id"} == {
        k: v for k, v in subject(evidence=brief).items() if k != "id"
    }
    assert len(backend.prompts) == 3
    assert run["steps"]["subjects"].get("approved") is not True
    assert validate_document(builtin("subject-inventory"))["valid"]


def test_invented_evidence_rejected(tmp_path):
    backend = TextBackend([json.dumps([selection(evidence_ids=[999])])] * 2)
    run = dispatch(
        "workflow-run",
        {
            "builtin": "subject-inventory",
            "inputs": {"brief": "Dog has one brown eye."},
            "runDirectory": str(tmp_path),
        },
        backend=backend,
    )
    assert run["status"] == "failed"
    assert "passage" in run["error"]


def test_wrong_kind_cannot_leak_across_extraction_passes(tmp_path):
    backend = TextBackend([json.dumps([selection(kind="prop")])] * 2)
    run = dispatch(
        "workflow-run",
        {
            "builtin": "subject-inventory",
            "inputs": {"brief": "Dog has one brown eye."},
            "runDirectory": str(tmp_path),
        },
        backend=backend,
    )
    assert run["status"] == "failed"
    assert "character" in run["error"]


def test_character_cannot_be_repeated_as_prop(tmp_path):
    duplicate = selection(kind="prop")
    backend = TextBackend([json.dumps([selection()]), json.dumps([duplicate])] * 2)
    run = dispatch(
        "workflow-run",
        {
            "builtin": "subject-inventory",
            "inputs": {"brief": "Dog has one brown eye."},
            "runDirectory": str(tmp_path),
        },
        backend=backend,
    )
    assert run["status"] == "failed"
    assert "already identified" in run["error"]


def test_long_scripts_are_partitioned_before_model_calls():
    from wee_todd_mlx.workflows.subjects import source_passages, source_windows

    brief = "\n".join(["A rainy rooftop with a dog. " * 80] * 8)
    passages = source_passages(brief)
    windows = source_windows(passages)
    assert len(windows) > 1
    assert all(len(window.encode("utf-8")) <= 5000 for window in windows)
    assert all(passage in brief for passage in passages)
    assert f"[{len(passages)}]" in windows[-1]


def test_host_assigns_stable_ids_even_when_model_uses_accents(tmp_path):
    item = selection(kind="prop")
    item.update(id="canapé_tray", name="Canapé tray")
    backend = TextBackend(["[]", json.dumps([item]), "[]"])
    run = dispatch(
        "workflow-run",
        {
            "builtin": "subject-inventory",
            "inputs": {"brief": "Dog has one brown eye."},
            "runDirectory": str(tmp_path),
        },
        backend=backend,
    )
    assert run["status"] == "awaiting_approval", run.get("error")
    result = run["steps"]["subjects"]["outputs"]["subjects"][0]
    assert result["name"] == "Canapé tray"
    assert result["id"].isascii()
    assert result["id"].startswith("prop_")


def test_full_window_warns_that_subjects_may_be_missing(tmp_path):
    found = []
    for index in range(8):
        item = selection()
        item["name"] = f"Character {index}"
        found.append(item)
    backend = TextBackend([json.dumps(found), "[]", "[]"])
    run = dispatch(
        "workflow-run",
        {
            "builtin": "subject-inventory",
            "inputs": {"brief": "Nine named people in a room."},
            "runDirectory": str(tmp_path),
        },
        backend=backend,
    )
    assert run["status"] == "awaiting_approval", run.get("error")
    warnings = run["steps"]["subjects"].get("warnings", [])
    assert any("eight" in message and "missing" in message for message in warnings)
