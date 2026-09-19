"""Guided relationship types agree across choices, acceptance and human review."""

import pytest
from test_workflow_object_coverage import coverage_request
from test_workflow_object_coverage import proposal as coverage_proposal
from test_workflow_subject_inventory import subject
from test_workflow_subject_links import empty_proposal, proposal, request, run

from wee_todd_mlx.workflows.creative import validate_classified_relations
from wee_todd_mlx.workflows.object_coverage import _direction, deterministic
from wee_todd_mlx.workflows.service import dispatch
from wee_todd_mlx.workflows.subject_links import allowed_targets


@pytest.mark.parametrize(
    "source_kind,target_kind,role,valid",
    [
        ("character", "environment", "uses", False),
        ("character", "set", "holds", False),
        ("character", "location", "uses", False),
        ("character", "prop", "located_in", False),
        ("character", "character", "located_in", False),
        ("prop", "environment", "contains", False),
        ("prop", "prop", "uses", False),
        ("set", "set", "part_of", False),
        ("character", "prop", "wears", False),
        ("character", "prop", "holds", True),
        ("character", "character", "holds", True),
        ("character", "clothing", "uses", True),
        ("character", "outfit", "wears", True),
        ("character", "set", "located_in", True),
        ("prop", "location", "located_in", True),
        ("set", "environment", "part_of", True),
        ("environment", "set", "contains", True),
        ("prop", "prop", "contains", True),
        ("character", "character", "part_of", True),
    ],
)
def test_guided_kind_matrix_shared_by_choices_coverage_and_approval(
    source_kind, target_kind, role, valid
):
    source = {**subject(source_kind), "id": "source"}
    target = {**subject(target_kind), "id": "target"}
    source["relationships"] = [{"id": "link", "targetID": "target", "role": role, "placement": ""}]
    assert ("target" in allowed_targets(source, {"target": target}, guided=True)[role]) == valid
    if valid:
        _direction(source, target, role, guided=True)
        validate_classified_relations([source, target])
    else:
        with pytest.raises(ValueError):
            _direction(source, target, role, guided=True)
        with pytest.raises(ValueError):
            validate_classified_relations([source, target])


@pytest.mark.parametrize(
    "definition_id", ["weetodd.guided-movie-planning", "weetodd.music-video-planning"]
)
@pytest.mark.parametrize("role,target_kind", [("uses", "environment"), ("located_in", "prop")])
def test_guided_bad_typed_proposals_never_enter_output(tmp_path, definition_id, role, target_kind):
    req = request(tmp_path)
    req["definition"]["id"] = definition_id
    req["inputs"]["inventory"][1]["kind"] = target_kind
    bad = proposal()
    bad["relationships"][0]["role"] = role
    state = run(req, [bad, bad, empty_proposal()])
    row = state["steps"]["links"]["outputs"]["subjects"][0]
    assert not row.get("relationships")
    assert row["relationshipReview"]["status"] == "needs_attention"
    assert state["executions"] == 3


def test_guided_coverage_preserves_existing_bad_link_for_review_but_blocks_approval(tmp_path):
    req = coverage_request(tmp_path)
    req["definition"]["id"] = "weetodd.music-video-planning"
    source, target = req["inputs"]["inventory"]
    source["description"] = "Dog uses Collar."
    target["kind"] = "environment"
    bad = {"id": "old", "targetID": "collar", "role": "uses", "placement": ""}
    source["relationships"] = [bad]
    state = run(req, [coverage_proposal(), coverage_proposal()])
    row = state["steps"]["links"]["outputs"]["subjects"][0]
    assert row["relationships"] == [bad]
    assert row["coverageReview"]["status"] == "needs_attention"
    with pytest.raises(ValueError, match="holds/uses"):
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
        )
    source.pop("relationships")
    inferred, _ = deterministic(source, [source, target], guided=True)
    assert not inferred["relationships"]


def test_guided_coverage_rejects_new_spatial_use(tmp_path):
    req = coverage_request(tmp_path)
    req["definition"]["id"] = "weetodd.guided-movie-planning"
    req["inputs"]["inventory"][0]["description"] = "A small dog."
    req["inputs"]["inventory"][1]["kind"] = "environment"
    bad = coverage_proposal(relationships=[{"targetID": "collar", "role": "uses", "placement": ""}])
    state = run(req, [bad, bad, coverage_proposal()])
    row = state["steps"]["links"]["outputs"]["subjects"][0]
    assert not row["relationships"]
    assert row["coverageReview"]["status"] == "needs_attention"


@pytest.mark.parametrize("role", ["holds", "uses"])
def test_legacy_spatial_use_contract_is_unchanged(role):
    source = {**subject(), "id": "source"}
    target = {**subject("environment"), "id": "target"}
    assert allowed_targets(source, {"target": target})[role] == ["target"]
    _direction(source, target, role)
