"""Short request-local IDs preserve source prose and canonical saved identities."""

import copy
import json

import pytest

from wee_todd_mlx.workflows.music_context import compact_subject_ids, parse_visible_subjects


def test_long_ids_round_trip_without_rewriting_descriptions_or_lyrics():
    hero = "character_123456789abcdef0"
    hall = "location_123456789abcdef0"
    subjects = [
        {"id": hero, "name": "King", "kind": "character", "description": f"Keep {hero} literally."},
        {"id": hall, "name": "Hall", "kind": "environment", "description": "Exact timber."},
    ]
    payload = {"assignment": {
        "subjectIndex": subjects, "characters": [subjects[0]],
        "approvedSubjects": [{**subjects[1], "relationships": [{"targetID": hero}]}],
        "selectedAppearanceIDs": [hero, hall], "omittedAppearanceIDs": [],
        "repairInstruction": f"Use {hero}; exclude {hall}.",
        "musicWindow": {"lines": [{"text": hero}]},
    }, "previous": {"characters": [hero], "endState": hero}}
    original = copy.deepcopy(payload)
    packed, aliases = compact_subject_ids(payload)
    assert payload == original
    assert len(json.dumps(packed)) < len(json.dumps(payload))
    ids = [row["id"] for row in packed["assignment"]["subjectIndex"]]
    assert [aliases[x] for x in ids] == [hero, hall]
    assert packed["assignment"]["characters"][0]["description"] == subjects[0]["description"]
    assert packed["assignment"]["musicWindow"] == original["assignment"]["musicWindow"]
    assert packed["previous"]["endState"] == hero
    assert packed["previous"]["characters"] == [ids[0]]
    assert packed["assignment"]["repairInstruction"] == f"Use {ids[0]}; exclude {ids[1]}."
    assert packed["assignment"]["approvedSubjects"][0]["relationships"][0]["targetID"] == ids[0]
    response = json.dumps({"action": "King waits in Hall.", "visibleSubjectIDs": ids})
    text, declared = parse_visible_subjects(response, subjects, aliases=aliases)
    assert declared == {hero, hall}
    assert json.loads(text)["characters"] == [hero]
    with pytest.raises(ValueError, match="distinct"):
        parse_visible_subjects(
            json.dumps({"visibleSubjectIDs": [hero, ids[0]]}), subjects, aliases=aliases
        )
    with pytest.raises(ValueError, match="known"):
        parse_visible_subjects('{"visibleSubjectIDs":["unknown"]}', subjects, aliases=aliases)


def test_short_source_ids_do_not_collide_with_request_aliases():
    subjects = [{"id": "ref1"}, {"id": "character_123456789abcdef0"}]
    payload = {"assignment": {"subjectIndex": subjects}}
    packed, aliases = compact_subject_ids(payload)
    assert packed["assignment"]["subjectIndex"][0]["id"] == "ref1"
    assert "ref1" not in aliases
    assert list(aliases.values()) == [subjects[1]["id"]]


def test_music_planner_restores_canonical_ids_and_reuses_completed_shots(monkeypatch):
    from test_music_shot_json_format import harness, inputs, shot

    from wee_todd_mlx.workflows.music import execute_music

    hero = "character_123456789abcdef0"
    hall = "location_123456789abcdef0"

    def rename(value):
        if isinstance(value, dict):
            return {k: rename(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rename(v) for v in value]
        return {"king": hero, "hall": hall}.get(value, value) if isinstance(value, str) else value

    value = rename(inputs())
    original = copy.deepcopy(value)
    reply = {**shot(), "characters": ["ref1"], "visibleSubjectIDs": ["ref1", "ref2"]}
    ctx, calls = harness(monkeypatch, [reply, reply])
    result = execute_music("music.plan_beats@1", value, {"maxClips": 200}, ctx)
    assert value == original
    assert all(row["characters"] == [hero] for row in result["clips"]["clips"])
    assert ctx.record["items"]["clip-1"]["base"]["selectedAppearanceIDs"] == [hero, hall]
    saved = copy.deepcopy(ctx.record)
    execute_music("music.plan_beats@1", value, {"maxClips": 200}, ctx)
    assert len(calls) == 2
    assert ctx.record == saved
