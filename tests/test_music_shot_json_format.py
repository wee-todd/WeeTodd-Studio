"""Music shot wire-format repair stays strict and preserves completed checkpoints."""

import copy
import json
from types import SimpleNamespace

import pytest
from test_music_planning_context import fixture

from wee_todd_mlx.workflows.music import execute_music
from wee_todd_mlx.workflows.music_context import parse_visible_subjects
from wee_todd_mlx.workflows.structured import REPAIR_FIELDS, plan_beats

MARKER = "Flat shot JSON skeleton (replace placeholders; no wrapper):\n"


def inputs():
    value = fixture()
    allocation = value["music_timing"]["allocation"]
    allocation["clips"] = allocation["clips"][:2]
    allocation["totalFrames"] = 200
    value["story"] = {
        "characters": [{"id": "king", "description": "An old king."}],
        "beats": ["King waits in Hall.", "King moves across Hall."],
    }
    value["subjects"].append(
        {"id": "hall", "name": "Hall", "kind": "set", "description": "Timber hall."}
    )
    return value


def shot(action="King waits in Hall."):
    return {
        "action": action,
        "startState": "King stands in Hall.",
        "endState": "King waits in Hall.",
        "location": "Hall",
        "characters": ["king"],
        "continuity": "cut",
        "visibleSubjectIDs": ["king", "hall"],
    }


def harness(monkeypatch, replies):
    from wee_todd_mlx.workflows import runner

    calls = []
    values = iter(replies)

    class Child:
        def __init__(self, *args):
            self.retry_error = None

        def ask(self, system, prompt):
            # Match Context.ask's final suffix, so the skeleton must follow retry guidance.
            if self.retry_error:
                prompt += "\nPrevious attempt failed validation: " + self.retry_error
            calls.append((system, prompt))
            return json.dumps(next(values))

    monkeypatch.setattr(runner, "Context", Child)
    ctx = SimpleNamespace(
        record={"outputs": {"saved": "prior human work"}},
        runner=SimpleNamespace(_save=lambda: None),
        spec={},
        deadline=0,
        check=lambda: None,
        message=lambda _: None,
    )
    return ctx, calls


def assert_format(prompt, fields):
    skeleton = json.loads(prompt.split(MARKER)[-1])
    assert set(skeleton) == (set(fields) - {"characters"}) | {"visibleSubjectIDs"}
    assert isinstance(skeleton["visibleSubjectIDs"][0], str)
    assert "characters" not in skeleton
    assert "host derives characters" in prompt
    assert "never objects with id/name" in prompt
    assert "Do not copy placeholder values" in prompt
    assert "entire resulting shot" in prompt
    return skeleton


@pytest.mark.parametrize("malformed", ["characters", "wrapper"])
def test_music_retries_live_malformed_shapes_with_final_flat_skeleton(monkeypatch, malformed):
    bad = shot()
    if malformed == "characters":
        bad["characters"] = [{"id": "king", "name": "King"}]
    else:
        bad = {"shot": bad}
    ctx, calls = harness(monkeypatch, [bad, shot(), shot("King moves across Hall.")])
    value = inputs()
    before = copy.deepcopy(value)
    result = execute_music("music.plan_beats@1", value, {"maxClips": 200}, ctx)
    assert len(calls) == 3
    for _, prompt in calls:
        assert_format(prompt, REPAIR_FIELDS["all"])
    assert "Previous attempt failed validation" in calls[1][1]
    assert ("strings" if malformed == "characters" else "root") in calls[1][1]
    assert result["clips"]["clips"][0]["characters"] == ["king"]
    assert "visibleSubjectIDs" not in result["clips"]["clips"][0]
    assert value == before


def test_music_failure_keeps_completed_clip_and_outputs_then_resumes(monkeypatch):
    bad_chars = {**shot(), "characters": [{"id": "king", "name": "King"}]}
    ctx, calls = harness(monkeypatch, [shot(), bad_chars, {"shot": shot()}, shot()])
    value = inputs()
    before = copy.deepcopy(value)
    with pytest.raises(ValueError, match="root"):
        execute_music("music.plan_beats@1", value, {"maxClips": 200}, ctx)
    completed = copy.deepcopy(ctx.record["items"]["clip-1"])
    assert completed["status"] == "completed"
    assert ctx.record["items"]["clip-2"]["status"] == "failed"
    assert not ctx.record["items"]["clip-2"].get("value")
    assert ctx.record["outputs"] == {"saved": "prior human work"}
    execute_music("music.plan_beats@1", value, {"maxClips": 200}, ctx)
    assert len(calls) == 4
    assert ctx.record["items"]["clip-1"] == completed
    assert value == before


@pytest.mark.parametrize("scope", REPAIR_FIELDS)
def test_music_scoped_repair_skeleton_contains_only_allowed_fields(monkeypatch, scope):
    fields = REPAIR_FIELDS[scope]
    repair = {key: shot()[key] for key in fields}
    repair["visibleSubjectIDs"] = ["king", "hall"]
    ctx, calls = harness(monkeypatch, [shot(), shot(), {"shot": repair}, repair])
    value = inputs()
    execute_music("music.plan_beats@1", value, {"maxClips": 200}, ctx)
    item = ctx.record["items"]["clip-2"]
    original = copy.deepcopy(item["value"])
    item.update(
        status="pending",
        repairFieldScope=scope,
        repairBase=original,
        repairInstruction="Keep the assigned action and approved identities.",
    )
    result = execute_music("music.plan_beats@1", value, {"maxClips": 200}, ctx)
    for _, prompt in calls[2:]:
        assert_format(prompt, fields)
    repaired = result["clips"]["clips"][1]
    for field in set(REPAIR_FIELDS["all"]) - set(fields):
        assert repaired[field] == original[field]
    assert ctx.record["items"]["clip-2"]["lastRepair"]["before"] == original


@pytest.mark.parametrize("key", ["characters", "visibleSubjectIDs"])
def test_music_unknown_and_placeholder_ids_still_rejected(monkeypatch, key):
    bad = {**shot(), key: ["unknown_placeholder_id"]}
    ctx, calls = harness(monkeypatch, [bad, bad])
    with pytest.raises(ValueError, match="IDs"):
        execute_music("music.plan_beats@1", inputs(), {"maxClips": 200}, ctx)
    assert len(calls) == 2
    assert not ctx.record["items"]["clip-1"].get("value")


def test_wrapped_metadata_is_rejected_without_coercion():
    with pytest.raises(ValueError, match="root"):
        parse_visible_subjects(json.dumps({"shot": shot()}), inputs()["subjects"])


def test_generic_movie_shot_prompt_stays_json_only(monkeypatch):
    response = {key: val for key, val in shot().items() if key != "visibleSubjectIDs"}
    ctx, calls = harness(monkeypatch, [response, response])
    value = inputs()
    value["_allocation"] = value.pop("music_timing")["allocation"]
    plan_beats(value, {"maxClips": 200}, ctx)
    for system, prompt in calls:
        assert "assignment" in json.loads(prompt)
        assert MARKER not in prompt
        assert "visibleSubjectIDs" not in system
