"""Focused review changes new jobs without weakening saved-job decisions."""

import copy
import json

import pytest
from test_studio_workflow_execution import TextBackend
from test_workflow_creative import subject

from wee_todd_mlx.workflows.service import builtin, dispatch


def test_new_guided_job_has_three_coherent_review_gates():
    definition = builtin("guided-movie-planning")
    assert definition["version"] == "1.1.0"
    assert [s["id"] for s in definition["steps"] if s.get("requiresApproval")] == [
        "creative_brief", "subjects_coverage", "clips",
    ]
    steps = {s["id"]: s for s in definition["steps"]}
    assert steps["story"]["inputs"]["subjects"]["step"] == "subjects_coverage"
    assert steps["prompt_preview"]["inputs"]["clips"]["step"] == "clips"


def test_classification_uses_established_context_instead_of_ambiguous_name():
    from types import SimpleNamespace

    from wee_todd_mlx.workflows.creative import classify_subjects

    row = subject("glasshouse", "location", "Glasshouse is a room inside the greenhouse.")
    row["name"] = "Glasshouse"
    calls = []

    def ask(system, prompt):
        request = json.loads(prompt)
        calls.append(request)
        # The same name could describe a whole building or an interior set. Source context
        # must reach the weighted boundary so the model can make the distinction.
        kind = "set" if request[0].get("description") == row["description"] else "environment"
        return json.dumps([{"slot": 1, "kind": kind}])

    result = classify_subjects([row], row["description"], SimpleNamespace(ask=ask))
    assert result[0]["kind"] == "set"
    assert calls[0][0]["allowedKinds"] == ["environment", "set", "location"]
    assert result[0]["description"] == row["description"]
    assert result[0]["id"] == row["id"]


def test_classification_keeps_source_evidence_when_extracted_description_is_minimal():
    from types import SimpleNamespace

    from wee_todd_mlx.workflows.creative import classify_subjects

    row = subject("lodge", "location", "Lodge")
    row["evidence"] = ["Lodge is the whole timber building. A room is inside it."]

    def ask(system, prompt):
        assignment = json.loads(prompt)[0]
        kind = "environment" if assignment.get("sourceEvidence") == row["evidence"] else "set"
        return json.dumps([{"slot": 1, "kind": kind}])

    result = classify_subjects([row], row["evidence"][0], SimpleNamespace(ask=ask))
    assert result[0]["kind"] == "environment"
    assert result[0]["evidence"] == row["evidence"]


def test_relationship_prompt_exposes_exact_legal_target_ids(tmp_path):
    from test_workflow_subject_links import empty_proposal, proposal, request

    req = request(tmp_path)
    backend = TextBackend([json.dumps(proposal()), json.dumps(empty_proposal())])
    state = dispatch("workflow-run", req, backend=backend)
    assert state["status"] == "awaiting_approval", state.get("error")
    wearer = json.loads(backend.prompts[0])
    collar = json.loads(backend.prompts[1])
    assert wearer["allowedTargetIDsByRole"]["wears"] == ["collar"]
    assert collar["allowedTargetIDsByRole"]["wears"] == []
    for ids in wearer["allowedTargetIDsByRole"].values():
        assert "dog" not in ids


def test_set_parent_cannot_be_another_set_before_subject_approval(tmp_path):
    from test_workflow_subject_links import request

    req = request(tmp_path)
    req["definition"]["id"] = "weetodd.guided-movie-planning"
    req["inputs"]["inventory"] = [
        subject("room", "set", "A tiled room"),
        subject("roof", "set", "A stone roof"),
    ]
    bad = {"description": "A tiled room", "relationships": [
        {"targetID": "roof", "role": "part_of", "placement": "inside"},
    ], "missingObjects": []}
    good = {"description": "A tiled room", "relationships": [], "missingObjects": []}
    roof = {"description": "A stone roof", "relationships": [], "missingObjects": []}
    backend = TextBackend([json.dumps(p) for p in [bad, good, roof]])
    state = dispatch("workflow-run", req, backend=backend)
    assert state["status"] == "awaiting_approval", state.get("error")
    assert state["steps"]["links"]["outputs"]["subjects"][0]["relationships"] == []
    assert len(backend.prompts) == 3


def subject_request(tmp_path, monkeypatch, *, step_id="subjects_coverage"):
    from wee_todd_mlx.workflows import object_coverage

    monkeypatch.setattr(
        object_coverage, "review_object_coverage", lambda rows, *args: {"subjects": rows}
    )
    definition = builtin("guided-movie-planning")
    step = next(s for s in definition["steps"] if s["id"] == step_id)
    step["inputs"] = {
        "subjects": {"input": "subjects"}, "brief": {"input": "brief"},
        "library": {"input": "library"},
    }
    step["requiresApproval"] = True
    definition["steps"] = [step]
    definition["inputs"] = {
        "subjects": {"type": "subject_list", "label": "Subjects", "default": [
            subject("cat"), subject("room", "environment", "A room inside a greenhouse"),
        ]},
        "brief": {"type": "text", "label": "Brief", "default": "Cat enters the room."},
        "library": {"type": "object_catalog", "label": "Library", "default": []},
    }
    definition["outputs"] = {"subjects": {"step": step_id, "output": "subjects"}}
    req = {"definition": definition, "runDirectory": str(tmp_path), "inputs": {}}
    state = dispatch("workflow-run", req, backend=TextBackend([]))
    assert state["awaitingStep"] == step_id, state.get("error")
    return req, state


def mutate(req, state, action, **extra):
    return dispatch("workflow-review", {**req, "review": {
        "stepID": req["definition"]["steps"][0]["id"],
        "expectedRevision": state["revision"], "action": action, **extra,
    }})


def test_final_subject_review_can_correct_kind_then_explicitly_approve_batch(tmp_path, monkeypatch):
    req, state = subject_request(tmp_path, monkeypatch)
    original = state["steps"]["subjects_coverage"]["outputs"]
    outputs = copy.deepcopy(original)
    outputs["subjects"][1]["kind"] = "set"
    state = mutate(req, state, "edit", outputs=outputs)
    step = state["steps"]["subjects_coverage"]
    assert not step.get("approved")
    assert not any(i.get("approved") for i in step["items"].values())
    assert step["outputs"]["subjects"][0] == original["subjects"][0]
    assert step["outputs"]["subjects"][1]["evidence"] == original["subjects"][1]["evidence"]
    state = mutate(req, state, "approve")
    assert state["steps"]["subjects_coverage"]["approved"]
    assert all(i["approved"] for i in state["steps"]["subjects_coverage"]["items"].values())


def test_intermediate_subject_inventory_keeps_existing_kind_edit_contract(tmp_path, monkeypatch):
    req, state = subject_request(tmp_path, monkeypatch, step_id="inventory")
    outputs = copy.deepcopy(state["steps"]["inventory"]["outputs"])
    outputs["subjects"][1]["kind"] = "set"
    with pytest.raises(ValueError, match="read-only"):
        mutate(req, state, "edit", outputs=outputs)


def test_old_guided_job_retains_all_saved_review_gates_on_builtin_resume(tmp_path):
    old = builtin("guided-movie-planning")
    old["version"] = "1.0.0"
    old_gates = {
        "creative_brief", "classify", "inventory", "design", "subjects_coverage", "story",
        "clips", "prompt_preview",
    }
    for step in old["steps"]:
        step["requiresApproval"] = step["id"] in old_gates
    req = {"builtin": "guided-movie-planning", "runDirectory": str(tmp_path),
           "inputs": {"brief": "Cat enters a greenhouse."}}
    state = dispatch("workflow-run", {**req, "definition": old},
                     backend=TextBackend([json.dumps({"questions": []})]))
    assert state["awaitingStep"] == "creative_brief"
    backend = TextBackend([])
    resumed = dispatch("workflow-run", req, backend=backend)
    assert resumed["definition"] == old
    assert {
        s["id"] for s in resumed["definition"]["steps"] if s.get("requiresApproval")
    } == old_gates
    assert (
        resumed["steps"]["creative_brief"]["outputs"]
        == state["steps"]["creative_brief"]["outputs"]
    )
    assert backend.prompts == []
