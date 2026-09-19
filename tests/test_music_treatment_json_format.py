"""Bracketed lyric prose must never substitute for a validated JSON action array."""

import copy
import json

import pytest
from test_music_planning_context import Capture, fixture

from wee_todd_mlx.workflows.music import execute_music

MALFORMED = (
    "[Young Beowulf crosses the whale-road under blackened skies to face the thing that fed.]\n"
    "[Old Beowulf raises a giant sword no living man forged to strike the dragon.]"
)


class Replies:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = []
        self.record = {"outputs": {"saved": "previous reviewed work"}}

    def message(self, _):
        pass

    def ask(self, system, prompt):
        self.calls.append((system, prompt))
        return next(self.values)


def inputs():
    value = fixture()
    value["music_timing"]["allocation"]["clips"] = value["music_timing"]["allocation"]["clips"][:8]
    value["music_timing"]["allocation"]["totalFrames"] = 800
    return value


def test_music_malformed_lyric_prose_retry_teaches_exact_json_without_reusing_bad_events():
    value = inputs()
    before = copy.deepcopy(value)
    actions = [
        f"The king completes distinct source action number {i} before returning home."
        for i in range(8)
    ]
    ctx = Replies([MALFORMED, json.dumps(actions)])
    result = execute_music("music.plan_treatment@1", value, {}, ctx)
    assert result["story"]["beats"] == actions
    assert len(ctx.calls) == 2
    first = ctx.calls[0][1]
    retry = ctx.calls[1][1]
    for prompt in (first, retry):
        skeleton = json.loads(
            prompt.split("JSON array skeleton (replace every placeholder):\n")[-1]
        )
        assert len(skeleton) == 8 and all(isinstance(item, str) for item in skeleton)
        assert "double-quoted strings separated by commas" in prompt
        assert "[Verse] and [Chorus] are source data" in prompt
        assert value["music_timing"]["suppliedLyrics"] in prompt
    assert "rejected reply is not source evidence" in retry
    assert "reread the original source" in retry
    assert MALFORMED not in retry
    assert value == before


def test_music_twice_malformed_action_arrays_fail_without_coercion_or_state_changes():
    value = inputs()
    before = copy.deepcopy(value)
    ctx = Replies([MALFORMED, MALFORMED])
    original_record = copy.deepcopy(ctx.record)
    with pytest.raises(ValueError, match="Invalid story_actions JSON"):
        execute_music("music.plan_treatment@1", value, {}, ctx)
    assert len(ctx.calls) == 2
    assert ctx.record == original_record
    assert value == before


def test_music_expansion_uses_its_own_required_array_size():
    ctx = Capture()
    execute_music("music.plan_treatment@1", fixture(), {}, ctx)
    for _, prompt in ctx.calls[1:]:
        assigned = json.loads(prompt.split("\nReturn exactly")[0])
        skeleton = json.loads(
            prompt.split("JSON array skeleton (replace every placeholder):\n")[-1]
        )
        assert len(skeleton) == assigned["actionsRequired"]


def test_generic_movie_format_prompt_is_unchanged():
    from wee_todd_mlx.workflows.creative import execute_creative

    value = inputs()
    value["_allocation"] = value.pop("music_timing")["allocation"]
    actions = [f"King completes distinct action {i}." for i in range(8)]
    ctx = Replies([json.dumps(actions)])
    result = execute_creative("movie.plan_treatment@1", value, {}, ctx)
    assert result["story"]["beats"] == actions
    assert all("JSON array skeleton" not in system + prompt for system, prompt in ctx.calls)
