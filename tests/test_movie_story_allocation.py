"""Regression for the dungeon plan jumping to shelter during its first clip."""

from wee_todd_mlx.workflows.operations import allocate_clips, check_plan

STORY = """**Title: The Warmth of Stone**

**0:00–0:15**
*Visual:* Close-up on a warrior's face lit by a torch.
The camera reveals him dragging a sword through a dripping dungeon.
He approaches a crumbling archway.
*Audio:* Breathing and dripping water.

**0:15–0:30**
*Visual:* The warrior pushes through the archway.
He discovers a hidden alcove containing dry moss and a fire in a stone bowl.
He sets down his sword beside the fire.
*Audio:* Crackling fire.

**0:30–0:45**
*Visual:* The warrior wraps his cloak around his shoulders.
He touches the warm stone wall. He exhales slowly and closes his eyes.
*Audio:* A gentle hum.

**0:45–0:60**
*Visual:* The warrior sits cross-legged. The camera pulls back from the alcove.
The final image fades to black.
*Audio:* A final fire pop.
"""


def plan(story=STORY):
    return allocate_clips(
        {"story": story, "duration_seconds": 60, "target_clip_seconds": 5, "frame_rate": 24}, 200
    )


def test_twelve_slots_have_distinct_local_actions_and_no_future_ending():
    clips = plan()["clips"]
    assert len({c["action"] for c in clips}) == 12
    assert all(
        "cross-legged" not in c["action"] and "fades to black" not in c["action"] for c in clips[:9]
    )
    assert "torch" in clips[0]["action"]
    assert "dragging a sword" in clips[1]["action"]
    assert "archway" in clips[2]["action"]
    assert "fades to black" in clips[-1]["action"]
    assert all("Title:" not in c["action"] and "*Audio:*" not in c["action"] for c in clips)
    assert sum(c["frameCount"] for c in clips) == 1440


def test_untimed_story_is_distributed_without_whole_outline_repetition():
    clips = plan("A warrior enters a dungeon. He discovers an alcove. He rests beside a fire.")[
        "clips"
    ]
    assert "rests" not in clips[0]["action"]
    assert "enters" not in clips[-1]["action"]
    assert len({c["action"] for c in clips}) == 3  # Do not invent nine extra actions.


def test_sparse_or_uncovered_timed_outline_fails_instead_of_silently_filling():
    import pytest

    with pytest.raises(ValueError, match="cover"):
        plan("**0:00–0:05**\n*Visual:* The warrior enters.")
    with pytest.raises(ValueError, match="overlap"):
        plan("**0:00–0:40**\n*Visual:* Enter.\n**0:30–0:60**\n*Visual:* Rest.")


def test_shelter_in_first_clip_and_repeated_endpoints_need_attention():
    clips = plan()
    ends = []
    for i, clip in enumerate(clips["clips"]):
        first = (
            {"description": "The warrior carries a torch in a dungeon."}
            if i == 0
            else {
                "description": ends[-1]["last"]["description"],
                "reuseFrom": {"clipID": ends[-1]["clipID"], "endpoint": "last"},
            }
        )
        ends.append(
            {
                "clipID": clip["id"],
                "first": first,
                "last": {
                    "description": "The warrior rests cross-legged in the hidden alcove "
                    "by the fire, "
                    "wrapped in his cloak."
                },
            }
        )
    review = check_plan(
        {"clips": clips, "endpoints": ends, "duration_seconds": 60, "frame_rate": 24}
    )
    assert review["status"] == "needs_attention"
    assert any(
        "future" in i["message"].lower() and i.get("clipID") == "clip-1" for i in review["items"]
    )
    assert any("repeat" in i["message"].lower() for i in review["items"])


def test_distinct_progression_passes_and_shared_first_frame_is_not_counted_as_repetition():
    clips = plan()
    ends = []
    for i, clip in enumerate(clips["clips"]):
        first = (
            {"description": "A warrior at the entrance."}
            if i == 0
            else {
                "description": ends[-1]["last"]["description"],
                "reuseFrom": {"clipID": ends[-1]["clipID"], "endpoint": "last"},
            }
        )
        ends.append({"clipID": clip["id"], "first": first, "last": {"description": clip["action"]}})
    review = check_plan(
        {"clips": clips, "endpoints": ends, "duration_seconds": 60, "frame_rate": 24}
    )
    assert review["status"] == "pass", review


def test_camera_fragments_do_not_take_a_clip_or_duplicate_neighbor_actions():
    story = """**0:00-0:15**
*Visual:* Medium shot. The warrior wraps his cloak. He touches the wall.
The camera moves closer. He exhales.
"""
    clips = allocate_clips(
        {"story": story, "duration_seconds": 15, "target_clip_seconds": 5, "frame_rate": 24}, 10
    )["clips"]
    assert "wraps" in clips[0]["action"]
    assert sum("wraps" in c["action"] for c in clips) == 1
    assert sum("touches" in c["action"] for c in clips) == 1
    assert "exhales" in clips[-1]["action"]


def test_generic_torch_lighting_is_not_mistaken_for_the_future_ending():
    clips = plan()
    ends = []
    for i, clip in enumerate(clips["clips"]):
        first = (
            {"description": "A warrior with a torch."}
            if i == 0
            else {
                "description": ends[-1]["last"]["description"],
                "reuseFrom": {"clipID": ends[-1]["clipID"], "endpoint": "last"},
            }
        )
        end = (
            clip["action"]
            if i
            else "Torchlight casts deep dancing shadows across his face in blackness."
        )
        ends.append({"clipID": clip["id"], "first": first, "last": {"description": end}})
    review = check_plan(
        {"clips": clips, "endpoints": ends, "duration_seconds": 60, "frame_rate": 24}
    )
    assert not any(
        i.get("clipID") == "clip-1" and "future" in i["message"] for i in review["items"]
    )


def test_large_repetitive_plan_keeps_review_within_its_output_contract():
    from wee_todd_mlx.workflows import validate_value

    clips = allocate_clips(
        {
            "story": "A warrior waits.",
            "duration_seconds": 1000,
            "target_clip_seconds": 1,
            "frame_rate": 24,
        },
        1000,
    )
    endpoints = []
    for i, clip in enumerate(clips["clips"]):
        first = {"description": "The end of the story fades to black."}
        if i:
            first["reuseFrom"] = {"clipID": endpoints[-1]["clipID"], "endpoint": "last"}
        endpoints.append(
            {
                "clipID": clip["id"],
                "first": first,
                "last": {"description": "The end of the story fades to black."},
            }
        )
    review = check_plan(
        {"clips": clips, "endpoints": endpoints, "duration_seconds": 1000, "frame_rate": 24}
    )
    assert validate_value("review", review) == []
    assert "additional" in review["items"][-1]["message"]
