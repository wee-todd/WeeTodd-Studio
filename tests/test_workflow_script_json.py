"""Recover a missing JSON closer without guessing content; honor authored shot boundaries."""

import json

import pytest
from test_studio_workflow_execution import TextBackend

from wee_todd_mlx.workflows.operations import parse_value
from wee_todd_mlx.workflows.service import dispatch


def test_missing_final_object_closer_preserves_every_field():
    raw = json.dumps(
        {
            "characters": [{"id": "cat_woman", "description": "One natural eye; one optical eye."}],
            "beats": ['She says "No mistakes".', "The dog takes a canapé."],
        }
    )[:-1]
    expected = json.loads(raw + "}")
    assert parse_value(raw, "story_outline") == expected


@pytest.mark.parametrize(
    "raw,kind",
    [
        ('{"status":"pass","items":[]', "review"),
        ('["First action.","Second action."', "story_actions"),
    ],
)
def test_only_one_unambiguous_root_closer_is_recovered(raw, kind):
    value = parse_value(raw, kind)
    assert value


@pytest.mark.parametrize(
    "raw",
    [
        '{"status":"pass","items":[',
        '{"characters":[],"beats":["unterminated',
        '{"characters":[],"beats":["a",',
        '{"characters":[],"beats":["a"] trailing prose',
        '{"characters":[],"beats":["a"}',
        '{"characters":[],"characters":[],"beats":["a"]',
        '{"characters":[],"beats":["a" "b"]',
        '{"characters":[]',
        '{"characters":[],"beats":[NaN]',
    ],
)
def test_recovery_never_invents_strings_fields_separators_or_ignores_invalid_json(raw):
    with pytest.raises(ValueError):
        parse_value(raw, "story_outline")


def test_braces_in_escaped_strings_are_not_treated_as_structure():
    expected = {"characters": [], "beats": ['She says "look at the {sign}".']}
    assert parse_value(json.dumps(expected)[:-1], "story_outline") == expected


def test_six_authored_shots_are_scoped_and_not_replaced_by_four_phases(tmp_path):
    brief = "\n".join(f"[Shot {i}] Source event {i}: {'details ' * 80}" for i in range(1, 7))
    brief += "\noverall_soundscape:\nRain and mechanical clicks."
    outline = {
        "characters": [{"id": "cat", "description": "A feline spy."}],
        "beats": ["Start", "Explore", "Fight", "Escape"],
    }
    backend = TextBackend(
        [json.dumps(outline)[:-1], *[f"Distinct action for shot {i}." for i in range(1, 7)]]
    )
    state = dispatch(
        "workflow-run",
        {
            "builtin": "movie-planning",
            "runDirectory": str(tmp_path),
            "inputs": {"brief": brief, "duration_seconds": 30, "target_clip_seconds": 5},
        },
        backend=backend,
    )
    assert state["status"] == "awaiting_approval", state.get("error")
    assert state["steps"]["story"]["outputs"]["story"]["beats"] == [
        f"Distinct action for shot {i}." for i in range(1, 7)
    ]
    assert state["inputs"]["brief"] == brief
    for i, prompt in enumerate(backend.prompts[1:], 1):
        assert f"Source event {i}:" in prompt
        assert all(f"Source event {j}:" not in prompt for j in range(1, 7) if j != i)
        assert "overall_soundscape" not in prompt


def test_script_frame_times_are_validated_before_weighted_planning(tmp_path):
    backend = TextBackend([])
    brief = "[Shot 1] Start.\n[Shot 2] At 00:20.000, finish."
    state = dispatch(
        "workflow-run",
        {
            "builtin": "movie-planning",
            "runDirectory": str(tmp_path),
            "inputs": {"brief": brief, "duration_seconds": 10, "target_clip_seconds": 5},
        },
        backend=backend,
    )
    assert state["status"] == "failed"
    assert "timing" in state["error"].lower()
    assert not backend.prompts


def test_truncated_model_response_is_never_repaired(tmp_path):
    class Truncated(TextBackend):
        def generate(self, *args, **kwargs):
            return {"text": '{"characters":[],"beats":["A","B","C","D"]', "truncated": True}

    state = dispatch(
        "workflow-run",
        {"builtin": "movie-planning", "runDirectory": str(tmp_path)},
        backend=Truncated([]),
    )
    assert state["status"] == "failed"
    assert "output limit" in state["error"].lower()
    assert not state.get("awaitingStep")


def test_terminal_malformed_response_is_retained_for_diagnosis(tmp_path):
    state = dispatch(
        "workflow-run",
        {"builtin": "movie-planning", "runDirectory": str(tmp_path)},
        backend=TextBackend(["not JSON", "still not JSON"]),
    )
    assert state["status"] == "failed"
    assert state["steps"]["story"]["lastRejectedResponse"]["text"] == "still not JSON"
    assert state["steps"]["story"]["calls"] == []


def test_source_shot_count_mismatch_is_actionable_not_silently_rewritten(tmp_path):
    state = dispatch(
        "workflow-run",
        {
            "builtin": "movie-planning",
            "runDirectory": str(tmp_path),
            "inputs": {
                "brief": "[Shot 1] A.\n[Shot 2] B.",
                "duration_seconds": 30,
                "target_clip_seconds": 5,
            },
        },
        backend=TextBackend([]),
    )
    assert state["status"] == "failed"
    assert "2 labeled shots" in state["error"]
    assert "6 clips" in state["error"]
