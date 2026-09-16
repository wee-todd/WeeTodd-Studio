"""Dependable planning: exact timing, narrow repairs, and approvals of exact revisions."""

import copy
import json

import pytest
from test_studio_workflow_execution import TextBackend

from wee_todd_mlx.workflows.service import dispatch


def outline():
    return {
        "characters": [{"id": "warrior", "description": "A warrior in silver armor."}],
        "beats": ["Explore the dungeon", "Discover a shelter", "Rest by the fire"],
    }


def beat(index, *, cut=False):
    return {
        "action": ["Walk to the arch", "Enter the shelter", "Sit by the fire"][index],
        "startState": ["At the entrance", "Beside the arch", "Inside the shelter"][index],
        "endState": ["Beside the arch", "Inside the shelter", "Seated by the fire"][index],
        "location": "Dungeon",
        "characters": ["warrior"],
        "continuity": "cut" if index == 0 or cut else "continue",
    }


def request(tmp_path):
    return {
        "builtin": "movie-planning",
        "runDirectory": str(tmp_path),
        "inputs": {"duration_seconds": 15, "target_clip_seconds": 5},
    }


def review(req, state, step, action="approve", **kwargs):
    return dispatch(
        "workflow-review",
        {
            **req,
            "review": {
                "stepID": step,
                "expectedRevision": state["revision"],
                "action": action,
                **kwargs,
            },
        },
    )


def to_clips(tmp_path, backend=None):
    req = request(tmp_path)
    backend = backend or TextBackend(
        [json.dumps(outline()), *[json.dumps(beat(i)) for i in range(3)]]
    )
    state = dispatch("workflow-run", req, backend=backend)
    assert state["status"] == "awaiting_approval"
    assert state["awaitingStep"] == "story"
    assert len(backend.prompts) == 1
    state = review(req, state, "story")
    state = dispatch("workflow-run", req, backend=backend)
    assert state["awaitingStep"] == "clips", state.get("error")
    return req, state, backend


def test_default_structured_workflow_requires_approval_and_assembles_endpoints(tmp_path):
    req, state, backend = to_clips(tmp_path)
    plan = state["steps"]["clips"]["outputs"]["clips"]
    assert sum(c["frameCount"] for c in plan["clips"]) == 360
    assert len({c["action"] for c in plan["clips"]}) == 3
    review(req, state, "clips")
    done = dispatch("workflow-run", req, backend=backend)
    assert done["status"] == "completed", done.get("error")
    assert len(backend.prompts) == 4  # outline + three states; no extra endpoint model calls
    pairs = done["outputs"]["endpoints"]
    assert pairs[1]["first"]["description"] == pairs[0]["last"]["description"]
    assert done["outputs"]["review"]["structure"] == "valid"
    assert done["outputs"]["review"]["storyReview"] == "human_required"
    assert dispatch("workflow-run", req, backend=backend)["outputs"] == done["outputs"]
    assert len(backend.prompts) == 4


@pytest.mark.parametrize("legacy", [False, True])
def test_runtime_update_preserves_reviewed_outputs_and_approvals(tmp_path, legacy):
    req, state, backend = to_clips(tmp_path)
    outputs = copy.deepcopy(state["steps"]["clips"]["outputs"])
    outputs["clips"]["clips"][0]["action"] = "Carefully walk to the arch"
    state = review(req, state, "clips", "edit", outputs=outputs)
    state = review(req, state, "clips")
    done = dispatch("workflow-run", req, backend=backend)
    if legacy:
        saved = copy.deepcopy(done)
        for step in saved["steps"].values():
            step.pop("contentKey", None)
            step.pop("executionRuntime", None)
            step.pop("executionSettings", None)
        (tmp_path / "run.json").write_text(json.dumps(saved))
    updated = TextBackend([])
    updated.fingerprint = lambda model, images: {"model": model, "images": images, "version": 2}
    resumed = dispatch("workflow-run", {**req, "maxTokens": 512}, backend=updated)
    assert resumed["status"] == "completed", resumed.get("error")
    assert resumed["outputs"] == done["outputs"]
    for sid in ["story", "clips"]:
        assert resumed["steps"][sid]["approved"]
        assert resumed["steps"][sid]["outputs"] == done["steps"][sid]["outputs"]
        assert resumed["steps"][sid]["executionRuntime"]["version"] == 1
    assert updated.prompts == []


def test_changed_reference_content_still_invalidates_completed_review(tmp_path):
    req, _, _ = to_clips(tmp_path)
    updated = TextBackend([json.dumps(outline())])
    updated.fingerprint = lambda model, images: {"model": model, "images": ["changed-content"]}
    resumed = dispatch("workflow-run", req, backend=updated)
    assert resumed["awaitingStep"] == "story"
    assert not resumed["steps"]["story"].get("approved")
    assert len(updated.prompts) == 1


def test_runtime_update_keeps_partially_approved_review_without_rerunning(tmp_path):
    req, state, _ = to_clips(tmp_path)
    item_id = next(iter(state["steps"]["clips"]["items"]))
    state = review(req, state, "clips", itemID=item_id)
    updated = TextBackend([])
    updated.fingerprint = lambda model, images: {"model": model, "images": images, "version": 2}
    resumed = dispatch("workflow-run", req, backend=updated)
    assert resumed["awaitingStep"] == "clips"
    assert resumed["steps"]["clips"]["items"][item_id]["approved"]
    assert resumed["steps"]["clips"]["outputs"] == state["steps"]["clips"]["outputs"]
    assert updated.prompts == []


def test_edit_is_atomic_rejects_stale_revision_and_invalid_values(tmp_path):
    req, state, _ = to_clips(tmp_path)
    before = (tmp_path / "run.json").read_bytes()
    with pytest.raises(ValueError, match="revision"):
        review(req, {**state, "revision": "stale"}, "clips")
    with pytest.raises(ValueError):
        review(req, state, "clips", "edit", outputs={"clips": {}})
    assert (tmp_path / "run.json").read_bytes() == before
    outputs = copy.deepcopy(state["steps"]["clips"]["outputs"])
    outputs["clips"]["clips"][0]["action"] = "Carefully walk to the arch"
    edited = review(req, state, "clips", "edit", outputs=outputs)
    assert edited["steps"]["clips"]["outputs"] == outputs
    assert edited["steps"]["clips"].get("approved") is not True
    assert edited["revision"] != state["revision"]


def test_target_repair_preserves_approved_prefix_and_independent_cut(tmp_path):
    backend = TextBackend(
        [
            json.dumps(outline()),
            json.dumps(beat(0)),
            json.dumps(beat(1)),
            json.dumps(beat(2, cut=True)),
            json.dumps({**beat(1), "action": "Push through the arch"}),
        ]
    )
    req, state, backend = to_clips(tmp_path, backend)
    old = copy.deepcopy(state["steps"]["clips"]["outputs"])
    state = review(req, state, "clips", itemID="clip-1")
    with pytest.raises(ValueError, match="approved"):
        review(req, state, "clips", "repair", itemID="clip-1")
    state = review(req, state, "clips", "repair", itemID="clip-2")
    assert state["steps"]["clips"]["items"]["clip-1"]["approved"] is True
    state = dispatch("workflow-run", req, backend=backend)
    new = state["steps"]["clips"]["outputs"]
    assert len(backend.prompts) == 5
    assert new["clips"]["clips"][0] == old["clips"]["clips"][0]
    assert new["clips"]["clips"][2] == old["clips"]["clips"][2]
    assert new["clips"]["clips"][1]["action"] == "Push through the arch"


def test_manual_endpoint_state_change_marks_continuous_dependents_stale(tmp_path):
    req, state, backend = to_clips(tmp_path)
    state = review(req, state, "clips", itemID="clip-2")
    outputs = copy.deepcopy(state["steps"]["clips"]["outputs"])
    outputs["clips"]["clips"][0]["endState"] = "At a locked door"
    state = review(req, state, "clips", "edit", outputs=outputs)
    assert state["steps"]["clips"]["items"]["clip-2"]["status"] == "stale"
    assert not state["steps"]["clips"]["items"]["clip-2"].get("approved")
    assert state["steps"]["clips"]["items"]["clip-1"]["value"]["endState"] == "At a locked door"


def test_malformed_clip_retry_does_not_repeat_successful_clips(tmp_path):
    backend = TextBackend(
        [
            json.dumps(outline()),
            json.dumps(beat(0)),
            "invalid",
            json.dumps(beat(1)),
            json.dumps(beat(2)),
        ]
    )
    _, state, backend = to_clips(tmp_path, backend)
    assert len(backend.prompts) == 5
    assert "validation" in backend.prompts[3].lower()
    assert state["steps"]["clips"]["items"]["clip-1"]["status"] == "completed"


def test_changed_brief_reopens_approval_and_keeps_previous_revision_readable(tmp_path):
    req, state, backend = to_clips(tmp_path)
    backend.replies = iter([json.dumps(outline())])
    changed = dispatch(
        "workflow-run",
        {**req, "inputs": {**req["inputs"], "brief": "A different story"}},
        backend=backend,
    )
    assert changed["awaitingStep"] == "story"
    assert not changed["steps"]["story"].get("approved")
    assert changed["steps"]["clips"]["status"] == "stale"
    assert changed["steps"]["clips"]["outputs"] == state["steps"]["clips"]["outputs"]


def test_failed_clip_resumes_without_repeating_prefix(tmp_path):
    backend = TextBackend([json.dumps(outline()), json.dumps(beat(0)), "bad", "bad"])
    req = request(tmp_path)
    state = dispatch("workflow-run", req, backend=backend)
    review(req, state, "story")
    state = dispatch("workflow-run", req, backend=backend)
    assert state["status"] == "failed"
    assert state["steps"]["clips"]["items"]["clip-1"]["status"] == "completed"
    backend.replies = iter([json.dumps(beat(1)), json.dumps(beat(2))])
    state = dispatch("workflow-run", req, backend=backend)
    assert state["awaitingStep"] == "clips"
    assert len(backend.prompts) == 6


def test_approved_story_cannot_be_edited_until_unlocked(tmp_path):
    req, state, _ = to_clips(tmp_path)
    with pytest.raises(ValueError, match="approved"):
        review(
            req, state, "story", "edit", outputs={"story": {**outline(), "beats": ["New story"]}}
        )


def test_character_ids_and_requested_movie_time_cannot_be_bypassed_by_edit(tmp_path):
    req, state, _ = to_clips(tmp_path)
    outputs = copy.deepcopy(state["steps"]["clips"]["outputs"])
    outputs["clips"]["clips"][0]["characters"] = ["unknown"]
    with pytest.raises(ValueError, match="character"):
        review(req, state, "clips", "edit", outputs=outputs)
    state = review(req, state, "story", "unapprove")
    bad_story = outline()
    bad_story["characters"].append(bad_story["characters"][0])
    with pytest.raises(ValueError, match="unique"):
        review(req, state, "story", "edit", outputs={"story": bad_story})


def test_cancelled_mid_clip_keeps_completed_beats(tmp_path):
    backend = TextBackend([json.dumps(outline()), json.dumps(beat(0)), InterruptedError("pause")])
    req = request(tmp_path)
    state = dispatch("workflow-run", req, backend=backend)
    review(req, state, "story")
    state = dispatch("workflow-run", req, backend=backend)
    assert state["status"] == "cancelled"
    backend.replies = iter([json.dumps(beat(1)), json.dumps(beat(2))])
    state = dispatch("workflow-run", req, backend=backend)
    assert state["awaitingStep"] == "clips"
    assert len(backend.prompts) == 5


def test_regeneration_of_approved_plan_is_blocked_without_changing_disk(tmp_path):
    req, state, backend = to_clips(tmp_path)
    review(req, state, "clips")
    before = (tmp_path / "run.json").read_bytes()
    with pytest.raises(ValueError, match="approved"):
        dispatch("workflow-run", {**req, "regenerate": "clips"}, backend=backend)
    assert (tmp_path / "run.json").read_bytes() == before


def test_reference_images_are_observed_once_and_not_resent_per_clip(tmp_path):
    backend = TextBackend(
        [
            "Silver armor and red cloak.",
            json.dumps(outline()),
            *[json.dumps(beat(i)) for i in range(3)],
        ]
    )
    req = request(tmp_path)
    req["inputs"]["images"] = ["asset:warrior"]
    state = dispatch("workflow-run", req, backend=backend)
    assert "Silver armor" in backend.prompts[1]
    review(req, state, "story")
    state = dispatch("workflow-run", req, backend=backend)
    assert len(backend.prompts) == 5
    assert state["awaitingStep"] == "clips"


def test_checkpoint_records_exact_model_requests_and_runtime_identity(tmp_path):
    req, state, _ = to_clips(tmp_path)
    assert state["runtimeFingerprints"]["assistant"]["version"] == 1
    call = state["steps"]["story"]["calls"][0]
    assert "characters" in call["system"]
    assert "duration_seconds" in call["prompt"]
    assert call["images"] == []
    assert state["generationSettings"]["maxTokens"] == 1024


def test_story_edit_reuses_unaffected_approved_cut_clips(tmp_path):
    backend = TextBackend(
        [
            json.dumps(outline()),
            *[json.dumps(beat(i, cut=True)) for i in range(3)],
            json.dumps({**beat(2, cut=True), "action": "Warm hands beside the fire"}),
        ]
    )
    req, state, backend = to_clips(tmp_path, backend)
    state = review(req, state, "clips", itemID="clip-1")
    state = review(req, state, "story", "unapprove")
    changed = outline()
    changed["beats"][2] = "Warm hands beside the fire"
    state = review(req, state, "story", "edit", outputs={"story": changed})
    state = review(req, state, "story")
    state = dispatch("workflow-run", req, backend=backend)
    assert state["awaitingStep"] == "clips"
    assert len(backend.prompts) == 5
    assert state["steps"]["clips"]["items"]["clip-1"]["approved"] is True


def test_repair_instruction_reaches_only_target_call(tmp_path):
    backend = TextBackend(
        [
            json.dumps(outline()),
            *[json.dumps(beat(i, cut=True)) for i in range(3)],
            json.dumps({**beat(1, cut=True), "action": "Open the door with both hands"}),
        ]
    )
    req, state, backend = to_clips(tmp_path, backend)
    state = review(
        req,
        state,
        "clips",
        "repair",
        itemID="clip-2",
        instruction="Use both hands to open the door.",
    )
    state = dispatch("workflow-run", req, backend=backend)
    assert len(backend.prompts) == 5
    assert "Use both hands" in backend.prompts[-1]
    assert (
        state["steps"]["clips"]["outputs"]["clips"]["clips"][1]["action"]
        == "Open the door with both hands"
    )


def test_story_must_assign_one_distinct_action_per_clip(tmp_path):
    req = request(tmp_path)
    req["inputs"]["duration_seconds"] = 60
    state = dispatch(
        "workflow-run", req, backend=TextBackend([json.dumps(outline()), json.dumps(outline())])
    )
    assert state["status"] == "failed"
    assert "4" in state["error"]
    assert "clips" not in state["steps"]


def test_duplicate_story_actions_are_rejected_before_clip_planning(tmp_path):
    repeated = outline()
    repeated["beats"] = ["Walk forward."] * 3
    state = dispatch(
        "workflow-run",
        request(tmp_path),
        backend=TextBackend([json.dumps(repeated), json.dumps(repeated)]),
    )
    assert state["status"] == "failed"
    assert "distinct" in state["error"]


def test_long_story_is_written_in_small_exact_batches(tmp_path):
    req = request(tmp_path)
    req["inputs"]["duration_seconds"] = 100
    overview = {**outline(), "beats": ["Enter", "Explore", "Discover", "Rest"]}
    chunks = []
    for phase in range(4):
        chunks.append([f"Inspect chamber {phase * 5 + j + 1} carefully." for j in range(4)])
        chunks.append([f"Inspect chamber {phase * 5 + 5} carefully."])
    backend = TextBackend([json.dumps(overview), *[json.dumps(chunk) for chunk in chunks]])
    state = dispatch("workflow-run", req, backend=backend)
    assert state["awaitingStep"] == "story"
    assert len(state["steps"]["story"]["outputs"]["story"]["beats"]) == 20
    assert len(backend.prompts) == 9
    assert all("exactly" in prompt for prompt in backend.prompts)


def test_reverting_an_edit_revalidates_cached_dependents_without_model_calls(tmp_path):
    req, state, backend = to_clips(tmp_path)
    original = copy.deepcopy(state["steps"]["clips"]["outputs"])
    changed = copy.deepcopy(original)
    changed["clips"]["clips"][0]["endState"] = "At a locked door"
    state = review(req, state, "clips", "edit", outputs=changed)
    state = review(req, state, "clips", "edit", outputs=original)
    assert state["steps"]["clips"]["items"]["clip-2"]["status"] == "completed"
    state = review(req, state, "clips")
    done = dispatch("workflow-run", req, backend=backend)
    assert done["status"] == "completed"
    assert len(backend.prompts) == 4


def test_run_next_cannot_leave_approved_old_step_current_after_input_change(tmp_path):
    from test_studio_workflow_execution import replies

    req = {"builtin": "staged-prompt-editing", "runDirectory": str(tmp_path)}
    backend = TextBackend(replies() + ["New reference appearance."])
    state = dispatch("workflow-run", req, backend=backend)
    state = review(req, state, "plan")
    changed_req = {
        **req,
        "inputs": {"source": "A completely new character", "images": ["asset:new"]},
        "maxSteps": 1,
    }
    state = dispatch("workflow-run", changed_req, backend=backend)
    assert state["steps"]["plan"]["status"] == "stale"
    assert not state["steps"]["plan"].get("approved")
    with pytest.raises(ValueError, match="current inputs"):
        review(changed_req, state, "plan")


def test_fixed_character_appearance_is_not_reported_as_a_premature_story_event():
    from wee_todd_mlx.workflows.structured import endpoints_for, execute_structured

    clips = []
    for i in range(3):
        value = beat(i, cut=True)
        value["action"] = ["Enter the hall", "Notice the tattered cloak and worn armor", "Rest"][i]
        clips.append({"id": f"clip-{i + 1}", "startFrame": i * 120, "frameCount": 120, **value})
    plan = {
        "fps": 24,
        "totalFrames": 360,
        "characters": [{"id": "warrior", "description": "worn armor and tattered cloak"}],
        "clips": clips,
    }
    result = execute_structured(
        "movie.check_plan@2",
        {"clips": plan, "endpoints": endpoints_for(plan), "duration_seconds": 15, "frame_rate": 24},
        {},
        None,
    )
    assert not any("future story" in i["message"] for i in result["review"]["items"])


def test_scoped_action_repair_keeps_other_fields_and_approved_following_shots(tmp_path):
    req, state, backend = to_clips(tmp_path)
    original = copy.deepcopy(state["steps"]["clips"]["outputs"]["clips"])
    state = review(req, state, "clips", itemID="clip-3")
    state = review(
        req, state, "clips", "repair", itemID="clip-2", fieldScope="action",
        instruction="Open the door with both hands.",
    )
    item = state["steps"]["clips"]["items"]["clip-2"]
    assert item["repairBase"] == beat(1)
    backend.replies = iter([json.dumps({
        **beat(1), "action": "Open the door with both hands",
        "startState": "Unrelated opening", "endState": "Unrelated ending",
        "location": "Rooftop", "characters": [], "continuity": "cut",
    })])
    resumed = dispatch("workflow-run", req, backend=backend)
    assert resumed["awaitingStep"] == "clips", resumed.get("error")
    new = resumed["steps"]["clips"]["outputs"]["clips"]
    expected = copy.deepcopy(original)
    expected["clips"][1]["action"] = "Open the door with both hands"
    assert new == expected
    assert resumed["steps"]["clips"]["items"]["clip-3"]["approved"]
    assignment = json.loads(backend.prompts[-1])["assignment"]
    assert assignment["existingShot"] == original["clips"][1]
    assert assignment["allowedFields"] == ["action"]
    assert len(backend.prompts) == 5
    cached = dispatch("workflow-run", req, backend=TextBackend([]))
    assert cached["steps"]["clips"]["outputs"]["clips"] == new


def test_scoped_repair_accepts_only_selected_fields_in_model_response(tmp_path):
    req, state, backend = to_clips(tmp_path)
    state = review(req, state, "clips", "repair", itemID="clip-2", fieldScope="action")
    backend.replies = iter([json.dumps({"action": "Open the door with both hands"})])
    state = dispatch("workflow-run", req, backend=backend)
    assert state["awaitingStep"] == "clips", state.get("error")
    assert state["steps"]["clips"]["items"]["clip-2"]["value"] == {
        **beat(1), "action": "Open the door with both hands",
    }


@pytest.mark.parametrize("scope", ["camera", "", [], None])
def test_unknown_repair_scope_cannot_mutate_saved_plan(tmp_path, scope):
    req, state, _ = to_clips(tmp_path)
    saved = (tmp_path / "run.json").read_bytes()
    with pytest.raises(ValueError, match="scope"):
        review(req, state, "clips", "repair", itemID="clip-2", fieldScope=scope)
    assert (tmp_path / "run.json").read_bytes() == saved


def test_scoped_location_repair_cannot_change_out_of_scope_start_state(tmp_path):
    req, state, backend = to_clips(tmp_path)
    old = copy.deepcopy(state["steps"]["clips"]["outputs"])
    state = review(req, state, "clips", "repair", itemID="clip-2", fieldScope="location")
    invalid = json.dumps({"location": "Rooftop", "continuity": "continue"})
    backend.replies = iter([invalid, invalid])
    state = dispatch("workflow-run", req, backend=backend)
    assert state["status"] == "failed"
    assert "location" in state["error"]
    assert state["steps"]["clips"]["outputs"] == old


def test_scoped_states_repair_cannot_rewrite_a_continuation_start(tmp_path):
    req, state, backend = to_clips(tmp_path)
    old = copy.deepcopy(state["steps"]["clips"]["outputs"])
    state = review(req, state, "clips", "repair", itemID="clip-2", fieldScope="states")
    invalid = json.dumps({"startState": "A different start", "endState": "Inside the shelter"})
    backend.replies = iter([invalid, invalid])
    state = dispatch("workflow-run", req, backend=backend)
    assert state["status"] == "failed"
    assert "starting state" in state["error"]
    assert state["steps"]["clips"]["outputs"] == old


def test_completed_scoped_correction_does_not_constrain_later_continuity_refresh(tmp_path):
    req, state, backend = to_clips(tmp_path)
    state = review(req, state, "clips", "repair", itemID="clip-2", fieldScope="action")
    backend.replies = iter(['{"action":"Open the door with both hands."}'])
    state = dispatch("workflow-run", req, backend=backend)
    edited = copy.deepcopy(state["steps"]["clips"]["outputs"])
    edited["clips"]["clips"][0]["endState"] = "At a locked door"
    state = review(req, state, "clips", "edit", outputs=edited)
    updated = {**beat(1), "action": "Open the door with both hands."}
    backend.replies = iter([json.dumps(updated), json.dumps(updated)])
    state = dispatch("workflow-run", req, backend=backend)
    assert state["status"] == "awaiting_approval", state.get("error")
    value = state["steps"]["clips"]["items"]["clip-2"]["value"]
    assert value == {**updated, "startState": "At a locked door"}
    assignment = json.loads(backend.prompts[-1])["assignment"]
    assert "allowedFields" not in assignment


def test_completed_scoped_correction_retains_history_without_active_scope(tmp_path):
    req, state, backend = to_clips(tmp_path)
    state = review(req, state, "clips", "repair", itemID="clip-2", fieldScope="action")
    backend.replies = iter(['{"action":"Open the door with both hands."}'])
    state = dispatch("workflow-run", req, backend=backend)
    item = state["steps"]["clips"]["items"]["clip-2"]
    assert "repairFieldScope" not in item
    assert item["lastRepair"]["fieldScope"] == "action"
    assert item["lastRepair"]["before"] == beat(1)
    assert item["lastRepair"]["outcome"] == "completed"
    assert dispatch("workflow-run", req, backend=TextBackend([]))["status"] == "awaiting_approval"


def test_manual_edit_supersedes_pending_scoped_correction(tmp_path):
    req, state, backend = to_clips(tmp_path)
    state = review(req, state, "clips", "repair", itemID="clip-2", fieldScope="action")
    edited = copy.deepcopy(state["steps"]["clips"]["outputs"])
    edited["clips"]["clips"][1]["action"] = "My final manual action"
    state = review(req, state, "clips", "edit", outputs=edited)
    assert "repairFieldScope" not in state["steps"]["clips"]["items"]["clip-2"]
    backend.replies = iter(['{"action":"Unwanted correction"}'])
    state = dispatch("workflow-run", req, backend=backend)
    assert state["steps"]["clips"]["outputs"] == edited
    assert len(backend.prompts) == 4


@pytest.mark.parametrize("interruption", [InterruptedError, TimeoutError])
def test_interrupted_scoped_correction_keeps_base_and_resumes_scope(tmp_path, interruption):
    req, state, backend = to_clips(tmp_path)
    state = review(req, state, "clips", "repair", itemID="clip-2", fieldScope="action")
    backend.replies = iter([interruption("Stopped")])
    state = dispatch("workflow-run", req, backend=backend)
    item = state["steps"]["clips"]["items"]["clip-2"]
    assert item["repairFieldScope"] == "action"
    assert item["repairBase"] == beat(1)
    backend.replies = iter(['{"action":"Open the door with both hands."}'])
    state = dispatch("workflow-run", req, backend=backend)
    assert state["status"] == "awaiting_approval", state.get("error")
    assert state["steps"]["clips"]["items"]["clip-2"]["value"] == {
        **beat(1), "action": "Open the door with both hands.",
    }


def test_second_scoped_correction_preserves_intervening_manual_fields(tmp_path):
    req, state, backend = to_clips(tmp_path)
    state = review(req, state, "clips", "repair", itemID="clip-2", fieldScope="action")
    backend.replies = iter(['{"action":"Open the door with both hands."}'])
    state = dispatch("workflow-run", req, backend=backend)
    edited = copy.deepcopy(state["steps"]["clips"]["outputs"])
    edited["clips"]["clips"][1]["endState"] = "Standing just inside the shelter"
    state = review(req, state, "clips", "edit", outputs=edited)
    state = review(req, state, "clips", "repair", itemID="clip-2", fieldScope="characters")
    backend.replies = iter(['{"characters":[]}', json.dumps(beat(2))])
    state = dispatch("workflow-run", req, backend=backend)
    assert state["status"] == "awaiting_approval", state.get("error")
    assert state["steps"]["clips"]["items"]["clip-2"]["value"] == {
        **beat(1), "action": "Open the door with both hands.", "characters": [],
        "endState": "Standing just inside the shelter",
    }


def test_changed_story_cannot_replay_completed_repair_over_manually_saved_fields(tmp_path):
    backend = TextBackend([
        json.dumps(outline()), *[json.dumps(beat(i, cut=True)) for i in range(3)],
    ])
    req, state, backend = to_clips(tmp_path, backend)
    state = review(req, state, "clips", "repair", itemID="clip-2", fieldScope="action")
    backend.replies = iter(['{"action":"Open the door with both hands."}'])
    state = dispatch("workflow-run", req, backend=backend)
    edited = copy.deepcopy(state["steps"]["clips"]["outputs"])
    edited["clips"]["clips"][1].update(location="Shelter", endState="Seated at the doorway")
    state = review(req, state, "clips", "edit", outputs=edited)
    state = review(req, state, "story", "unapprove")
    story = outline()
    story["beats"][1] = "Enter the shelter carefully"
    state = review(req, state, "story", "edit", outputs={"story": story})
    state = review(req, state, "story")
    updated = {
        **beat(1, cut=True), "action": "Enter the shelter carefully",
        "location": "Shelter", "endState": "Seated at the doorway",
    }
    backend.replies = iter([json.dumps(updated)])
    state = dispatch("workflow-run", req, backend=backend)
    assert state["status"] == "awaiting_approval", state.get("error")
    assert state["steps"]["clips"]["items"]["clip-2"]["value"] == updated
