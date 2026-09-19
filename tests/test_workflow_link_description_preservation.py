"""Guided relationship passes cannot rewrite reviewed object definitions."""

import copy

import pytest
from test_workflow_subject_links import empty_proposal, inventory, proposal, request, run


@pytest.mark.parametrize(
    "definition_id", ["weetodd.guided-movie-planning", "weetodd.music-video-planning"]
)
@pytest.mark.parametrize(
    "reviewed,rewritten",
    [
        ("An intelligent avenger. Nonsexualized design, never a seductive monster.", "A monster."),
        ("An ancient hoard barrow, distinct from the later funeral burial mound.", "A barrow."),
        (
            "A head trophy, distinct from the separate severed arm.\nPreserve this distinction.",
            "A trophy.",
        ),
    ],
)
def test_guided_links_preserve_reviewed_definition_verbatim(
    tmp_path, definition_id, reviewed, rewritten
):
    req = request(tmp_path)
    req["definition"]["id"] = definition_id
    req["inputs"]["inventory"][0]["description"] = reviewed
    before = copy.deepcopy(req["inputs"]["inventory"])
    state = run(req, [proposal(description=rewritten), empty_proposal()])
    assert state["status"] == "awaiting_approval", state.get("error")
    after = state["steps"]["links"]["outputs"]["subjects"]
    for original, result in zip(before, after, strict=True):
        for field in ("id", "name", "kind", "aliases", "description", "evidence", "suggestions"):
            assert result[field] == original[field]
        assert result["relationshipReview"]["reviewedDescription"] == original["description"]
    assert after[0]["relationships"][0]["targetID"] == "collar"
    assert state["executions"] == 2
    assert req["inputs"]["inventory"] == before


def test_guided_link_accepts_repeated_appearance_in_preserved_description(tmp_path):
    req = request(tmp_path)
    req["definition"]["id"] = "weetodd.music-video-planning"
    reviewed = "A small dog. A red leather collar. Keep the collar visible."
    req["inputs"]["inventory"][0]["description"] = reviewed
    state = run(req, [proposal(description=reviewed), empty_proposal()])
    row = state["steps"]["links"]["outputs"]["subjects"][0]
    assert row["description"] == reviewed
    assert row["relationshipReview"]["status"] == "ready"
    assert row["relationships"][0]["targetID"] == "collar"
    assert state["executions"] == 2


def test_guided_link_still_rejects_repeated_appearance_in_placement(tmp_path):
    req = request(tmp_path)
    req["definition"]["id"] = "weetodd.music-video-planning"
    bad = proposal()
    bad["relationships"][0]["placement"] = "A red leather collar."
    state = run(req, [bad, bad, empty_proposal()])
    row = state["steps"]["links"]["outputs"]["subjects"][0]
    assert row["description"] == inventory()[0]["description"]
    assert not row.get("relationships")
    assert row["relationshipReview"]["status"] == "needs_attention"
