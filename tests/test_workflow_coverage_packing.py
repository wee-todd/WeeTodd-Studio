"""Coverage transport compacts identifiers without changing creative evidence."""

import copy
import json

import pytest
from test_workflow_object_coverage import coverage_request, proposal
from test_workflow_subject_links import run

from wee_todd_mlx.workflows.coverage_packing import compact_request, restore_proposal


def payload():
    return {
        "CURRENT_OBJECT_ONLY": "character_123456789abcdef",
        "currentObject": {
            "id": "character_123456789abcdef",
            "name": "Warrior",
            "kind": "character",
            "relationships": [
                {
                    "id": "saved-link",
                    "targetID": "prop_123456789abcdef",
                    "role": "holds",
                    "placement": "Only after the handover.",
                }
            ],
        },
        "currentDescription": "Unbroken sword; ring stays worn until the handover.",
        "otherInventoryRowsForContextOnly": [
            {
                "id": "prop_123456789abcdef",
                "name": "Ring",
                "kind": "prop",
                "description": "Exact gold surface. No gemstone.",
                "aliases": ["arm-ring"],
            },
            {"id": "r0", "name": "Hall", "kind": "environment"},
        ],
        "contextSelection": {"omittedAppearanceIDs": ["r0"], "selection": "Exact selection."},
        "sourcePassages": {"1": "Preserve this source verbatim: ‘ring’."},
        "libraryCandidatesForCurrentObjectOnly": [{"id": "library-id"}],
        "previousIssues": [],
    }


def test_compaction_preserves_every_description_source_and_placement():
    original = payload()
    saved = copy.deepcopy(original)
    packed, identities = compact_request(original)
    assert original == saved
    current = packed["currentObject"]
    assert current["id"] == packed["CURRENT_OBJECT_ONLY"] == "r1"
    assert current["relationships"][0]["targetID"] == "r2"
    assert current["relationships"][0]["placement"] == "Only after the handover."
    for key in ("currentDescription", "sourcePassages", "libraryCandidatesForCurrentObjectOnly"):
        assert packed[key] == original[key]
    assert (
        packed["otherInventoryRowsForContextOnly"][0]["description"]
        == (original["otherInventoryRowsForContextOnly"][0]["description"])
    )
    assert identities == {
        "r1": "character_123456789abcdef",
        "r2": "prop_123456789abcdef",
        "r0": "r0",
    }
    assert packed["contextSelection"]["omittedAppearanceIDs"] == ["r0"]


def test_restore_validates_both_relationship_and_mention_targets():
    _, identities = compact_request(payload())
    value = proposal(
        relationships=[{"targetID": "r2", "role": "holds", "placement": "hand"}],
        mentions=[{"targetID": "r2", "phrase": "ring", "occurrence": 0}],
    )
    result = restore_proposal(value, identities)
    assert value["relationships"][0]["targetID"] == "r2"
    assert result["relationships"][0]["targetID"] == "prop_123456789abcdef"
    assert result["mentions"][0]["targetID"] == "prop_123456789abcdef"
    for key in ("relationships", "mentions"):
        invalid = copy.deepcopy(value)
        invalid[key][0]["targetID"] = "invented"
        with pytest.raises(ValueError, match="request ID"):
            restore_proposal(invalid, identities)


def test_guided_coverage_restores_persistent_ids_without_changing_evidence(tmp_path):
    req = coverage_request(tmp_path)
    req["definition"]["id"] = "weetodd.music-video-planning"
    rows = req["inputs"]["inventory"]
    rows[0]["id"] = "character_123456789abcdef"
    rows[1]["id"] = "clothing_123456789abcdef"
    rows[0]["description"] = "Dog tugs his neckwear."
    state = run(
        req,
        [
            proposal(
                relationships=[{"targetID": "r1", "role": "wears", "placement": "neck"}],
                mentions=[{"targetID": "r1", "phrase": "his neckwear", "occurrence": 0}],
            ),
            proposal(),
        ],
    )
    assert state["status"] == "awaiting_approval", state.get("error")
    result = state["steps"]["links"]["outputs"]["subjects"][0]
    assert result["description"] == rows[0]["description"]
    assert result["evidence"] == rows[0]["evidence"]
    assert result["relationships"][0]["targetID"] == rows[1]["id"]
    assert result["descriptionMentions"][0]["targetID"] == rows[1]["id"]
    assert json.loads((tmp_path / "run.json").read_text()) == state


@pytest.mark.parametrize(
    "invalid",
    [
        {"targetID": "invented", "role": "wears", "placement": "neck"},
        {"targetID": {}, "role": "wears", "placement": "neck"},
        {"role": "wears", "placement": "neck"},
    ],
)
def test_invalid_request_target_does_not_discard_independent_valid_link(tmp_path, invalid):
    req = coverage_request(tmp_path)
    req["definition"]["id"] = "weetodd.music-video-planning"
    req["inputs"]["inventory"][0]["description"] = "Dog tugs his neckwear."
    first = proposal(
        relationships=[{"targetID": "collar", "role": "wears", "placement": "neck"}, invalid]
    )
    state = run(req, [first, proposal(), proposal()])
    row = state["steps"]["links"]["outputs"]["subjects"][0]
    assert row["relationships"][0]["targetID"] == "collar"
