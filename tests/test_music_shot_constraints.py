"""Expose existing music shot decisions before inference and retain relevant failures."""

import copy
import json

import pytest
from test_music_shot_json_format import MARKER, harness, inputs, shot

from wee_todd_mlx.workflows.music import execute_music
from wee_todd_mlx.workflows.structured import REPAIR_FIELDS


def location_inputs():
    value = inputs()
    value["subjects"].append(
        {"id": "square", "name": "Square", "kind": "set", "description": "Stone square."}
    )
    value["story"]["beats"][1] = "King waits in Square."
    return value


def at_square(**changes):
    return {
        **shot(),
        "action": "King waits in Square.",
        "startState": "King stands in Square.",
        "endState": "King waits in Square.",
        "location": "Square",
        "visibleSubjectIDs": ["king", "square"],
        **changes,
    }


def assignment(prompt):
    return json.JSONDecoder().raw_decode(prompt)[0]["assignment"]


def test_music_exposes_required_location_and_cut_before_first_attempt(monkeypatch):
    value = location_inputs()
    ctx, calls = harness(monkeypatch, [shot(), at_square()])
    execute_music("music.plan_beats@1", value, {"maxClips": 200}, ctx)
    for i, location in enumerate(["Hall", "Square"]):
        prompt = calls[i][1]
        assert assignment(prompt)["requiredLocation"] == location
        assert assignment(prompt)["requiredContinuity"] == "cut"
        skeleton = json.loads(prompt.split(MARKER)[-1])
        assert skeleton["location"] == location
        assert skeleton["continuity"] == "cut"
        assert "requiredLocation and requiredContinuity are exact host constraints" in prompt
        assert "requiredLocation" not in ctx.record["items"][f"clip-{i + 1}"]["base"]


@pytest.mark.parametrize(
    "beats",
    [
        ["King waits.", "King moves."],
        ["King waits in Hall.", "King moves from Hall toward Square."],
    ],
)
def test_no_unique_location_does_not_invent_required_location(monkeypatch, beats):
    value = location_inputs()
    value["story"]["beats"] = beats
    ctx, calls = harness(monkeypatch, [shot(), shot(), shot()])
    execute_music("music.plan_beats@1", value, {"maxClips": 200}, ctx)
    assert "requiredLocation" not in assignment(calls[-1][1])
    assert "requiredContinuity" not in assignment(calls[-1][1])


def test_same_location_does_not_force_cut(monkeypatch):
    ctx, calls = harness(monkeypatch, [shot(), {**shot(), "continuity": "continue"}])
    result = execute_music("music.plan_beats@1", inputs(), {"maxClips": 200}, ctx)
    assert assignment(calls[1][1])["requiredLocation"] == "Hall"
    assert "requiredContinuity" not in assignment(calls[-1][1])
    assert result["clips"]["clips"][1]["continuity"] == "continue"


def test_location_cut_rejects_copied_previous_start_and_explains_current_scene(monkeypatch):
    copied = at_square(startState=shot()["endState"], visibleSubjectIDs=["king", "square", "hall"])
    ctx, calls = harness(monkeypatch, [shot(), copied, at_square()])
    result = execute_music("music.plan_beats@1", location_inputs(), {"maxClips": 200}, ctx)
    assert len(calls) == 3
    assert "startState must show the new shot" in calls[1][1]
    assert "copied the previous location" in calls[2][1]
    assert result["clips"]["clips"][1]["startState"] == "King stands in Square."


def test_music_shot_json_packing_preserves_complete_assignment(monkeypatch):
    ctx, calls = harness(monkeypatch, [shot(), at_square()])
    execute_music("music.plan_beats@1", location_inputs(), {"maxClips": 200}, ctx)
    prompt = calls[1][1]
    packed, end = json.JSONDecoder().raw_decode(prompt)
    assert prompt[:end] == json.dumps(packed, ensure_ascii=False, separators=(",", ":"))
    expected = copy.deepcopy(ctx.record["items"]["clip-2"]["base"])
    expected.update(requiredLocation="Square", requiredContinuity="cut")
    assert packed["assignment"] == expected
    assert packed["previous"] == {"endState": shot()["endState"], "location": "Hall"}


def test_location_cut_can_repeat_a_location_neutral_pose(monkeypatch):
    pose = "King stands still."
    ctx, calls = harness(monkeypatch, [{**shot(), "endState": pose}, at_square(startState=pose)])
    result = execute_music("music.plan_beats@1", location_inputs(), {"maxClips": 200}, ctx)
    assert len(calls) == 2
    assert result["clips"]["clips"][1]["startState"] == pose


def test_music_resume_retains_latest_relevant_failure_without_replaying_completed_clip(monkeypatch):
    value = location_inputs()
    ctx, calls = harness(
        monkeypatch, [shot(), shot(), at_square(continuity="continue"), at_square()]
    )
    with pytest.raises(ValueError, match="previous location"):
        execute_music("music.plan_beats@1", value, {"maxClips": 200}, ctx)
    completed = copy.deepcopy(ctx.record["items"]["clip-1"])
    assert ctx.record["items"]["clip-2"]["status"] == "failed"
    assert not ctx.record["items"]["clip-2"].get("value")
    execute_music("music.plan_beats@1", value, {"maxClips": 200}, ctx)
    assert len(calls) == 4
    assert "previous location" in calls[3][1]
    assert ctx.record["items"]["clip-1"] == completed
    assert "musicFailureContext" not in ctx.record["items"]["clip-2"]


@pytest.mark.parametrize("changed", ["assignment", "previous", "legacy", "old_identity_rules"])
def test_music_resume_drops_stale_or_unproven_error(monkeypatch, changed):
    value = location_inputs()
    ctx, calls = harness(
        monkeypatch, [shot(), shot(), at_square(continuity="continue"), at_square()]
    )
    with pytest.raises(ValueError):
        execute_music("music.plan_beats@1", value, {"maxClips": 200}, ctx)
    if changed == "assignment":
        value["story"]["beats"][1] = "King pauses in Square."
    elif changed == "previous":
        ctx.record["items"]["clip-1"]["value"]["endState"] = "King sits in Hall."
    elif changed == "old_identity_rules":
        from wee_todd_mlx.workflows.runner import digest

        record = ctx.record["items"]["clip-2"]
        record["musicFailureContext"] = digest({
            "assignment": record["base"],
            "previous": {"endState": shot()["endState"], "location": "Hall"},
        })
    else:
        ctx.record["items"]["clip-2"].pop("musicFailureContext", None)
    execute_music("music.plan_beats@1", value, {"maxClips": 200}, ctx)
    assert "Previous attempt failed validation" not in calls[-1][1]


@pytest.mark.parametrize("scope", REPAIR_FIELDS)
def test_music_constraints_never_add_immutable_fields_to_repair_skeleton(monkeypatch, scope):
    fields = REPAIR_FIELDS[scope]
    patch = {key: val for key, val in at_square().items() if key in fields}
    patch["visibleSubjectIDs"] = ["king", "square"]
    ctx, calls = harness(monkeypatch, [shot(), at_square(), patch])
    value = location_inputs()
    execute_music("music.plan_beats@1", value, {"maxClips": 200}, ctx)
    record = ctx.record["items"]["clip-2"]
    original = copy.deepcopy(record["value"])
    record.update(
        status="pending",
        repairFieldScope=scope,
        repairBase=original,
        repairInstruction="Keep the chosen location.",
    )
    execute_music("music.plan_beats@1", value, {"maxClips": 200}, ctx)
    skeleton = json.loads(calls[-1][1].split(MARKER)[-1])
    assert set(skeleton) == (set(fields) - {"characters"}) | {"visibleSubjectIDs"}
    for key in ["location", "continuity"]:
        if key not in fields:
            assert ctx.record["items"]["clip-2"]["value"][key] == original[key]
