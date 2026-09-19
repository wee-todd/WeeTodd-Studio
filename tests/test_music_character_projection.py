"""Music cast is a deterministic projection of explicit, validated visible identities."""

import copy
import json

import pytest
from test_music_shot_json_format import harness, inputs, shot

from wee_todd_mlx.workflows.music import execute_music
from wee_todd_mlx.workflows.music_context import parse_visible_subjects
from wee_todd_mlx.workflows.structured import REPAIR_FIELDS, plan_beats


def without_characters(value):
    return {key: item for key, item in value.items() if key != "characters"}


def test_full_shot_projects_only_known_character_kinds_from_explicit_metadata(monkeypatch):
    value = inputs()
    before = copy.deepcopy(value)
    response = without_characters(shot())
    ctx, calls = harness(monkeypatch, [response, response])
    result = execute_music("music.plan_beats@1", value, {"maxClips": 200}, ctx)
    assert len(calls) == 2
    assert all(clip["characters"] == ["king"] for clip in result["clips"]["clips"])
    assert all("visibleSubjectIDs" not in clip for clip in result["clips"]["clips"])
    assert value == before


def test_projection_preserves_declared_order_without_inventing_characters():
    subjects = inputs()["subjects"] + [{"id": "queen", "kind": "character"}]
    raw = {"visibleSubjectIDs": ["hall", "queen", "king"]}
    parsed, _ = parse_visible_subjects(json.dumps(raw), subjects)
    assert json.loads(parsed)["characters"] == ["queen", "king"]
    parsed, _ = parse_visible_subjects(json.dumps({"visibleSubjectIDs": ["hall"]}), subjects)
    assert json.loads(parsed)["characters"] == []


@pytest.mark.parametrize(
    "characters",
    [[{"id": "king", "name": "King"}], "king", [], ["hall"], ["king", "king"], ["unknown"]],
)
def test_provided_characters_must_be_well_formed_and_match_declaration(characters):
    with pytest.raises(ValueError, match="characters"):
        parse_visible_subjects(
            json.dumps({**shot(), "characters": characters}), inputs()["subjects"]
        )


@pytest.mark.parametrize("declared", [["unknown"], ["king", "king"], [1], "king", None])
def test_projection_never_accepts_unknown_duplicate_or_invalid_metadata(declared):
    with pytest.raises(ValueError, match="visibleSubjectIDs"):
        parse_visible_subjects(json.dumps({"visibleSubjectIDs": declared}), inputs()["subjects"])


@pytest.mark.parametrize("scope", REPAIR_FIELDS)
def test_scoped_projection_only_changes_cast_when_characters_allowed(monkeypatch, scope):
    value = inputs()
    value["subjects"].append(
        {"id": "queen", "name": "Queen", "kind": "character", "description": "A queen."}
    )
    value["story"]["characters"].append({"id": "queen", "description": "A queen."})
    fields = REPAIR_FIELDS[scope]
    patch = {key: item for key, item in shot().items() if key in fields and key != "characters"}
    patch["visibleSubjectIDs"] = ["king", "queen", "hall"]
    good = {**patch, "visibleSubjectIDs": ["king", "hall"]}
    replies = [shot(), shot(), patch] + ([] if "characters" in fields else [good])
    ctx, calls = harness(monkeypatch, replies)
    execute_music("music.plan_beats@1", value, {"maxClips": 200}, ctx)
    completed = copy.deepcopy(ctx.record["items"]["clip-1"])
    record = ctx.record["items"]["clip-2"]
    original = copy.deepcopy(record["value"])
    record.update(
        status="pending",
        repairFieldScope=scope,
        repairBase=original,
        repairInstruction="Use only the requested fields.",
        musicRequiredSubjectIDs=["queen"],
    )
    result = execute_music("music.plan_beats@1", value, {"maxClips": 200}, ctx)
    expected = ["king", "queen"] if "characters" in fields else ["king"]
    assert result["clips"]["clips"][1]["characters"] == expected
    assert len(calls) == (3 if "characters" in fields else 4)
    if "characters" not in fields:
        assert "immutable characters" in calls[-1][1]
    assert ctx.record["items"]["clip-1"] == completed


def test_scoped_action_cannot_override_immutable_cast_even_with_explicit_characters():
    subjects = inputs()["subjects"] + [{"id": "queen", "kind": "character"}]
    raw = {"action": "King waits.", "characters": ["queen"], "visibleSubjectIDs": ["queen", "hall"]}
    with pytest.raises(ValueError, match="immutable characters"):
        parse_visible_subjects(json.dumps(raw), subjects, immutable_characters=["king"])


def test_generic_movie_still_requires_explicit_characters(monkeypatch):
    response = {
        key: item for key, item in shot().items() if key not in {"characters", "visibleSubjectIDs"}
    }
    ctx, calls = harness(monkeypatch, [response, response])
    value = inputs()
    value["_allocation"] = value.pop("music_timing")["allocation"]
    with pytest.raises(ValueError, match="characters.*required"):
        plan_beats(value, {"maxClips": 200}, ctx)
    assert len(calls) == 2
    assert not ctx.record["items"]["clip-1"].get("value")
