"""Stable relationship proposals, graph validation and human-owned approvals."""

import copy
import json

import pytest
from test_studio_workflow_execution import TextBackend
from test_workflow_subject_inventory import subject

from wee_todd_mlx.workflows.service import builtin, dispatch
from wee_todd_mlx.workflows.validation import validate_value


def inventory():
    return [
        {**subject(), "id": "dog", "description": "A small dog wearing a red leather collar."},
        {
            **subject("clothing"),
            "id": "collar",
            "name": "Collar",
            "description": "A red leather collar.",
        },
    ]


def proposal(target="collar", description="A small dog."):
    return {
        "description": description,
        "relationships": [{"targetID": target, "role": "wears", "placement": "around the neck"}],
        "missingObjects": [],
    }


def request(tmp_path):
    definition = builtin("subject-inventory-reviewed")
    definition["inputs"]["inventory"] = {"type": "subject_list", "label": "Inventory"}
    definition["steps"] = [
        {
            "id": "links",
            "name": "Link objects",
            "operation": "project.link_subjects@1",
            "inputs": {"brief": {"input": "brief"}, "subjects": {"input": "inventory"}},
            "parameters": {},
            "retry": {"maxAttempts": 1},
            "timeoutSeconds": 1800,
            "model": "assistant",
            "requiresApproval": True,
        }
    ]
    definition["outputs"] = {"subjects": {"step": "links", "output": "subjects"}}
    return {
        "definition": definition,
        "inputs": {"brief": "  Dog wears the collar.\n", "inventory": inventory()},
        "runDirectory": str(tmp_path),
    }


def run(req, values):
    return dispatch(
        "workflow-run",
        req,
        backend=TextBackend([json.dumps(v) if isinstance(v, dict) else v for v in values]),
    )


def empty_proposal():
    return {"description": "A red leather collar.", "relationships": [], "missingObjects": []}


def test_link_proposals_preserve_identity_evidence_and_use_host_ids(tmp_path):
    req = request(tmp_path)
    state = run(req, [proposal(), empty_proposal()])
    assert state["status"] == "awaiting_approval", state.get("error")
    output = state["steps"]["links"]["outputs"]["subjects"]
    assert output[0]["description"] == "A small dog."
    link = output[0]["relationships"][0]
    assert link["id"].startswith("link_")
    assert {k: link[k] for k in ("targetID", "role", "placement")} == proposal()["relationships"][0]
    for old, new in zip(inventory(), output, strict=True):
        assert {
            k: new[k] for k in ("id", "name", "kind", "aliases", "evidence", "suggestions")
        } == {k: old[k] for k in ("id", "name", "kind", "aliases", "evidence", "suggestions")}
    assert state["inputs"]["brief"] == "  Dog wears the collar.\n"
    assert not state["steps"]["links"].get("approved")
    assert json.loads((tmp_path / "run.json").read_text()) == state


@pytest.mark.parametrize("target", ["invented", "dog"])
def test_bad_targets_retry_once_and_remain_advisory_without_invalid_links(tmp_path, target):
    state = run(request(tmp_path), [proposal(target), proposal(target), empty_proposal()])
    assert state["status"] == "awaiting_approval", state.get("error")
    result = state["steps"]["links"]["outputs"]["subjects"][0]
    assert result.get("relationships", []) == []
    assert result["description"] == inventory()[0]["description"]
    assert result["relationshipReview"]["status"] == "needs_attention"
    assert state["executions"] == 3


def test_missing_objects_are_suggestions_without_inventory_mutation(tmp_path):
    value = proposal()
    value["missingObjects"] = ["Consider extracting the distinctive brass name tag."]
    state = run(request(tmp_path), [value, empty_proposal()])
    result = state["steps"]["links"]["outputs"]["subjects"]
    assert [s["id"] for s in result] == ["dog", "collar"]
    assert result[0]["relationshipReview"]["missingObjects"] == value["missingObjects"]


def test_partial_link_run_resumes_without_rewriting_finished_subject(tmp_path):
    req = request(tmp_path)
    state = run(req, [proposal(), RuntimeError("Cancelled")])
    assert state["status"] == "failed"
    saved_link = state["steps"]["links"]["items"]["dog"]["value"]["relationships"][0]
    resumed = run(req, [empty_proposal()])
    assert resumed["status"] == "awaiting_approval", resumed.get("error")
    assert resumed["steps"]["links"]["outputs"]["subjects"][0]["relationships"][0] == saved_link
    assert resumed["executions"] == 3


@pytest.mark.parametrize(
    "kind", ["character", "prop", "location", "environment", "set", "clothing", "outfit"]
)
def test_subject_kinds_support_portable_imported_definitions(kind):
    assert not validate_value(
        "subject_list", [{**subject(kind), "id": "550E8400-E29B-41D4-A716-446655440000"}]
    )


@pytest.mark.parametrize(
    "fault", ["unknown", "self", "duplicate_subject", "duplicate_link", "too_many", "placement"]
)
def test_inventory_validates_relationship_graph_before_use(fault):
    values = inventory()
    values[0]["relationships"] = [{"id": "link_1", **proposal()["relationships"][0]}]
    if fault == "unknown":
        values[0]["relationships"][0]["targetID"] = "missing"
    elif fault == "self":
        values[0]["relationships"][0]["targetID"] = "dog"
    elif fault == "duplicate_subject":
        values[1]["id"] = "dog"
    elif fault == "duplicate_link":
        values[0]["relationships"] *= 2
    elif fault == "too_many":
        values[0]["relationships"] *= 33
    else:
        values[0]["relationships"][0]["placement"] = "x" * 301
    assert validate_value("subject_list", values)


def test_human_relationship_edit_requires_unlock_and_retains_advice(tmp_path):
    req = request(tmp_path)
    state = run(req, [proposal(), empty_proposal()])

    def edit(action, **extra):
        return dispatch(
            "workflow-review",
            {
                **req,
                "review": {
                    "stepID": "links",
                    "expectedRevision": state["revision"],
                    "action": action,
                    "itemID": "dog",
                    **extra,
                },
            },
            backend=TextBackend([]),
        )

    state = edit("approve")
    outputs = copy.deepcopy(state["steps"]["links"]["outputs"])
    report = copy.deepcopy(outputs["subjects"][0]["relationshipReview"])
    outputs["subjects"][0]["relationships"][0]["placement"] = "loosely around the neck"
    with pytest.raises(ValueError, match="Unlock"):
        edit("edit", outputs=outputs)
    state = edit("unapprove")
    state = edit("edit", outputs=outputs)
    assert not state["steps"]["links"]["items"]["dog"]["approved"]
    assert state["steps"]["links"]["outputs"]["subjects"][0]["relationshipReview"] == report
    state = edit("approve")
    assert state["steps"]["links"]["items"]["dog"]["approved"]


def test_link_host_preserves_existing_id_for_exact_placement(tmp_path):
    req = request(tmp_path)
    req["inputs"]["inventory"][0]["relationships"] = [
        {"id": "original_link", **proposal()["relationships"][0]}
    ]
    value = proposal()
    state = run(req, [value, empty_proposal()])
    assert (
        state["steps"]["links"]["outputs"]["subjects"][0]["relationships"][0]["id"]
        == "original_link"
    )


def test_link_repeated_target_appearance_is_retried(tmp_path):
    state = run(
        request(tmp_path),
        [proposal(description="A small dog. A red leather collar."), proposal(), empty_proposal()],
    )
    assert state["steps"]["links"]["outputs"]["subjects"][0]["description"] == "A small dog."
    assert state["executions"] == 3


def test_legacy_saved_builtin_is_reviewed_without_upgrading_its_definition(tmp_path):
    definition = builtin("subject-inventory-reviewed")
    definition["version"] = "1.0.0"
    definition["steps"] = [s for s in definition["steps"]
                           if s["id"] not in {"subjects_links", "subjects_coverage"}]
    definition["steps"][-1]["requiresApproval"] = True
    definition["outputs"]["subjects"]["step"] = "subjects_review"
    definition["steps"][-1]["inputs"]["subjects"]["step"] = "subjects"
    from test_subject_description_review import BRIEF, POOL, draft

    selection = {k: v for k, v in POOL.items() if k != "evidence"}
    selection["evidenceIDs"] = [2]
    req = {"definition": definition, "inputs": {"brief": BRIEF}, "runDirectory": str(tmp_path)}
    state = run(req, ["[]", "[]", json.dumps([selection]), draft(), {"issues": [], "missing": []}])
    resumed_req = {k: v for k, v in req.items() if k != "definition"}
    resumed_req["builtin"] = "subject-inventory-reviewed"
    reviewed = dispatch(
        "workflow-review",
        {
            **resumed_req,
            "review": {
                "stepID": "subjects_review",
                "expectedRevision": state["revision"],
                "action": "approve",
            },
        },
        backend=TextBackend([]),
    )
    assert reviewed["definition"] == definition
    assert reviewed["steps"]["subjects_review"]["approved"]


def test_editing_link_target_invalidates_dependent_approval_without_rewriting_it(tmp_path):
    req = request(tmp_path)
    state = run(req, [proposal(), empty_proposal()])
    state = dispatch(
        "workflow-review",
        {
            **req,
            "review": {
                "stepID": "links",
                "expectedRevision": state["revision"],
                "action": "approve",
                "itemID": "dog",
            },
        },
        backend=TextBackend([]),
    )
    outputs = copy.deepcopy(state["steps"]["links"]["outputs"])
    outputs["subjects"][1]["description"] = "A blue leather collar."
    state = dispatch(
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
    assert not state["steps"]["links"]["items"]["dog"]["approved"]
    assert state["steps"]["links"]["outputs"] == outputs


def test_relationship_review_notes_cannot_be_rewritten_by_output_edit(tmp_path):
    req = request(tmp_path)
    state = run(req, [proposal(), empty_proposal()])
    outputs = copy.deepcopy(state["steps"]["links"]["outputs"])
    outputs["subjects"][0]["relationshipReview"]["issues"] = ["Forged review note"]
    with pytest.raises(ValueError, match="read-only"):
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


def test_review_operation_rejects_missing_targets_before_model_work(tmp_path):
    req = request(tmp_path)
    req["definition"]["steps"][0]["operation"] = "project.review_subjects@1"
    req["inputs"]["inventory"][0]["relationships"] = [
        {"id": "link_1", "targetID": "unknown", "role": "wears", "placement": ""}
    ]
    with pytest.raises(ValueError, match="existing other subject"):
        run(req, [])


def test_same_definition_can_have_multiple_distinct_placements(tmp_path):
    req = request(tmp_path)
    value = proposal()
    value["relationships"] = [
        {"targetID": "collar", "role": "contains", "placement": "left compartment"},
        {"targetID": "collar", "role": "contains", "placement": "right compartment"},
    ]
    state = run(req, [value, empty_proposal()])
    assert state["status"] == "awaiting_approval", state.get("error")
    links = state["steps"]["links"]["outputs"]["subjects"][0].get("relationships", [])
    assert [link["placement"] for link in links] == ["left compartment", "right compartment"]
    assert len({link["id"] for link in links}) == 2
    assert state["executions"] == 2


def test_existing_distinct_placement_ids_survive_link_proposals(tmp_path):
    req = request(tmp_path)
    value = proposal()
    value["relationships"] = [
        {"targetID": "collar", "role": "contains", "placement": "left compartment"},
        {"targetID": "collar", "role": "contains", "placement": "right compartment"},
    ]
    req["inputs"]["inventory"][0]["relationships"] = [
        {"id": "left_link", **value["relationships"][0]},
        {"id": "right_link", **value["relationships"][1]},
    ]
    state = run(req, [value, empty_proposal()])
    assert state["status"] == "awaiting_approval", state.get("error")
    links = state["steps"]["links"]["outputs"]["subjects"][0]["relationships"]
    assert [link["id"] for link in links] == ["left_link", "right_link"]


@pytest.mark.parametrize(
    "source_kind,target_kind",
    [
        ("clothing", "character"),
        ("outfit", "prop"),
        ("environment", "clothing"),
        ("set", "clothing"),
        ("location", "clothing"),
        ("prop", "character"),
    ],
)
def test_clearly_reversed_wears_retries_then_stays_advisory(tmp_path, source_kind, target_kind):
    req = request(tmp_path)
    req["inputs"]["inventory"][0]["kind"] = source_kind
    req["inputs"]["inventory"][1]["kind"] = target_kind
    state = run(req, [proposal(), proposal(), empty_proposal()])
    assert state["status"] == "awaiting_approval", state.get("error")
    result = state["steps"]["links"]["outputs"]["subjects"][0]
    assert result.get("relationships", []) == []
    assert result["description"] == inventory()[0]["description"]
    assert result["relationshipReview"]["status"] == "needs_attention"
    assert "wears" in result["relationshipReview"]["issues"][0]
    assert state["executions"] == 3


@pytest.mark.parametrize("source_kind,role", [("prop", "wears"), ("clothing", "uses")])
def test_direction_checks_preserve_mannequin_clothing_and_generic_uses(tmp_path, source_kind, role):
    req = request(tmp_path)
    req["inputs"]["inventory"][0]["kind"] = source_kind
    value = proposal()
    value["relationships"][0]["role"] = role
    state = run(req, [value, empty_proposal()])
    assert state["status"] == "awaiting_approval", state.get("error")
    result = state["steps"]["links"]["outputs"]["subjects"][0]
    assert result["relationships"][0]["role"] == role
    assert result["relationshipReview"]["status"] == "ready"
    assert state["executions"] == 2
