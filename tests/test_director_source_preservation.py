"""Source-authored story facts survive model summaries and shot corrections."""

import json
from types import SimpleNamespace

import pytest
from test_studio_workflow_execution import TextBackend
from test_workflow_creative import prompt_fixture, subject
from test_workflow_structured_review import beat, outline, request, review, to_clips

from wee_todd_mlx.workflows.creative import execute_creative
from wee_todd_mlx.workflows.service import dispatch
from wee_todd_mlx.workflows.validation import validate_value


def test_long_authored_guided_shots_are_summarized_separately_and_keep_speech():
    requests = []
    replies = iter(["The cat lifts the diamond.", "The cat leaves the vault."])

    def ask(system, prompt):
        requests.append((system, prompt))
        if "JSON array" in system:
            return json.dumps(["The cat lifts the diamond.", "The cat leaves the vault."])
        return next(replies)

    source = (
        '[Shot 1] ' + 'Dust covers the shelves. ' * 20
        + 'The cat lifts the diamond. She whispers, “Don’t blink.”\n'
        + '[Shot 2] ' + 'Lights illuminate the corridor. ' * 20
        + 'The cat leaves the vault. She says, "We’re done."'
    )
    ctx = SimpleNamespace(ask=ask, record={}, message=lambda _: None)
    result = execute_creative("movie.plan_treatment@1", {
        "subjects": [subject()], "creative_brief": {"questions": []},
        "brief": source, "duration_seconds": 10, "target_clip_seconds": 5,
        "frame_rate": 24, "observations": [],
    }, {}, ctx)
    assert result["story"] == {
        "characters": [{"id": "cat", "description": "White cat"}],
        "beats": [
            'The cat lifts the diamond.\nShe whispers, “Don’t blink.”',
            'The cat leaves the vault.\nShe says, "We’re done."',
        ],
    }
    assert len(requests) == 2
    assert "We’re done" not in requests[0][1]
    assert "Don’t blink" not in requests[1][1]


def test_approved_appearance_is_not_truncated_at_story_or_clip_boundary():
    appearance = "White fur with charcoal markings. " * 30 + "One amber eye; one blue eye."
    ctx = SimpleNamespace(ask=lambda *_: '["The cat enters the vault."]', record={})
    result = execute_creative("movie.plan_treatment@1", {
        "subjects": [subject(description=appearance)],
        "creative_brief": {"questions": []}, "brief": "The cat enters the vault.",
        "duration_seconds": 5, "target_clip_seconds": 5, "frame_rate": 24,
        "observations": [],
    }, {}, ctx)
    assert result["story"]["characters"] == [{"id": "cat", "description": appearance}]
    assert not validate_value("story_outline", result["story"])
    _, plan, _ = prompt_fixture()
    plan["characters"] = result["story"]["characters"]
    assert not validate_value("structured_clip_plan", plan)


def test_clip_planning_and_action_repair_restore_exact_assigned_dialogue(tmp_path):
    story = outline()
    story["beats"][1] = 'The warrior opens the door. She whispers, “Don’t blink.”'
    backend = TextBackend([json.dumps(story), *[json.dumps(beat(i)) for i in range(3)]])
    req = request(tmp_path)
    req["inputs"]["brief"] = "\n".join(story["beats"])
    state = dispatch("workflow-run", req, backend=backend)
    state = review(req, state, "story")
    state = dispatch("workflow-run", req, backend=backend)
    value = state["steps"]["clips"]["items"]["clip-2"]["value"]
    assert "Don’t blink" in backend.prompts[2]
    assert value["action"] == 'Enter the shelter\nShe whispers, “Don’t blink.”'
    state = review(req, state, "clips", "repair", itemID="clip-2", fieldScope="action")
    backend.replies = iter(['{"action":"Open the door with both hands."}'])
    state = dispatch("workflow-run", req, backend=backend)
    assert state["steps"]["clips"]["items"]["clip-2"]["value"]["action"] == (
        'Open the door with both hands.\nShe whispers, “Don’t blink.”'
    )


def test_invented_dialogue_cannot_enter_a_clip_before_prompt_compilation(tmp_path):
    story = outline()
    backend = TextBackend([json.dumps(story)])
    req = request(tmp_path)
    state = dispatch("workflow-run", req, backend=backend)
    state = review(req, state, "story")
    invented = {**beat(0), "action": 'She says, "Mission complete!"'}
    backend.replies = iter([json.dumps(invented), json.dumps(invented), json.dumps(beat(2))])
    state = dispatch("workflow-run", req, backend=backend)
    assert state["status"] == "failed"
    assert "dialogue" in state["error"].lower()
    assert state["steps"]["clips"]["items"]["clip-1"].get("value") is None


def test_guided_authored_shots_do_not_receive_generated_preference_appendix():
    calls = []
    source = "[Shot 1] " + "Dust covers the shelves. " * 20 + "The cat lifts a diamond."
    ctx = SimpleNamespace(
        ask=lambda system, prompt: calls.append(prompt) or "The cat lifts the diamond.",
        record={}, message=lambda _: None,
    )
    execute_creative("movie.plan_treatment@1", {
        "subjects": [subject()],
        "creative_brief": {"questions": [], "sourceText": source},
        "brief": source + '\n\nApproved creative preferences:\n{"cameraStyle":"Locked off"}',
        "duration_seconds": 5, "target_clip_seconds": 5, "frame_rate": 24,
        "observations": [],
    }, {}, ctx)
    assert len(calls) == 1
    assert "cameraStyle" not in calls[0]


def test_oversized_exact_speech_fails_before_requesting_a_summary():
    from wee_todd_mlx.workflows.script import preserve_dialogue

    source = 'She says, "' + "Very long source dialogue. " * 15 + '"'
    calls = []
    ctx = SimpleNamespace(
        ask=lambda *args: calls.append(args) or "The cat waves.", record={}, message=lambda _: None,
    )
    with pytest.raises(ValueError, match="dialogue"):
        execute_creative("movie.plan_treatment@1", {
            "subjects": [subject()], "creative_brief": {"questions": []},
            "brief": "[Shot 1] " + source, "duration_seconds": 5,
            "target_clip_seconds": 5, "frame_rate": 24, "observations": [],
        }, {}, ctx)
    assert calls == []
    assert preserve_dialogue("A cat waves.", "A cat waves.", 300) == "A cat waves."


def test_repair_request_retains_explicit_field_scope_in_human_provenance(tmp_path):
    req, state, _ = to_clips(tmp_path)
    state = review(req, state, "clips", "repair", itemID="clip-2", fieldScope="action")
    decision = state["humanDecisions"][-1]
    assert decision["action"] == "repair"
    assert decision["fieldScope"] == "action"


@pytest.mark.parametrize("spoken", ['‘I can’t wait.’', "'I can't wait.'"])
def test_single_quoted_dialogue_does_not_end_at_an_in_word_apostrophe(spoken):
    from wee_todd_mlx.workflows.script import preserve_dialogue

    source = "Mara says, " + spoken
    assert preserve_dialogue("Mara waves.", source, 300) == "Mara waves.\n" + source


def test_action_repair_can_change_visible_action_while_retaining_exact_speech():
    from wee_todd_mlx.workflows.script import preserve_dialogue

    source = 'Mara closes the brass case and says “We’ll return at dawn.”'
    edited = 'Mara gently closes the brass case and says “We’ll return at dawn.”'
    assert preserve_dialogue(edited, source, 300) == edited


def test_exact_spoken_words_cannot_silently_change_named_speaker():
    from wee_todd_mlx.workflows.script import preserve_dialogue

    source = 'Mara closes the brass case and says “We’ll return at dawn.”'
    edited = 'Ivo gently closes the brass case and says “We’ll return at dawn.”'
    with pytest.raises(ValueError, match="speaker"):
        preserve_dialogue(edited, source, 300)


def test_unlabeled_treatment_rejects_invented_speech_against_original_source():
    ctx = SimpleNamespace(
        ask=lambda *_: json.dumps(['The cat says, "Mission complete!"']), record={},
    )
    with pytest.raises(ValueError, match="dialogue"):
        execute_creative("movie.plan_treatment@1", {
            "subjects": [subject()], "creative_brief": {"questions": []},
            "brief": "The cat leaves the vault.", "duration_seconds": 5,
            "target_clip_seconds": 5, "frame_rate": 24, "observations": [],
        }, {}, ctx)


def test_unlabeled_treatment_does_not_silently_omit_unassigned_spoken_lines():
    ctx = SimpleNamespace(
        ask=lambda *_: json.dumps(["The cat enters.", "The cat leaves."]), record={},
    )
    with pytest.raises(ValueError, match="shot assignment"):
        execute_creative("movie.plan_treatment@1", {
            "subjects": [subject()], "creative_brief": {"questions": []},
            "brief": 'The cat says, "Stay here." She leaves the vault.', "duration_seconds": 10,
            "target_clip_seconds": 5, "frame_rate": 24, "observations": [],
        }, {}, ctx)


def test_speaker_label_dialogue_cannot_be_reassigned_to_another_label():
    from wee_todd_mlx.workflows.script import preserve_dialogue

    with pytest.raises(ValueError, match="speaker"):
        preserve_dialogue('Ivo: “Same line.”', 'Mara: “Same line.”', 300)


@pytest.mark.parametrize("source, tokens", [
    ('Mara says "Go" and Ivo says "No".', ['"Go"', '"No"']),
    ('Mara: "Go"\nIvo: "No"', ['"Go"', '"No"']),
    ('Inside the observatory, Mara says, "Stay here."', ['"Stay here."']),
])
def test_host_assembled_speech_is_nonduplicating_and_survives_the_next_stage(source, tokens):
    from wee_todd_mlx.workflows.script import preserve_dialogue, require_dialogue_assignment

    action = "Mara walks across the empty observatory and opens the tall brass telescope cabinet"
    story = preserve_dialogue(action, source, 600)
    require_dialogue_assignment([story], source)
    clip = preserve_dialogue("Mara checks the cabinet", story, 600)
    require_dialogue_assignment([clip], source)
    for token in tokens:
        assert story.count(token) == 1
        assert clip.count(token) == 1


@pytest.mark.parametrize("verb", ["utters", "announces", "unknown_speech_verb"])
def test_changed_quoted_tokens_are_rejected_independently_of_speech_verbs(verb):
    from wee_todd_mlx.workflows.script import preserve_dialogue

    with pytest.raises(ValueError, match="dialogue"):
        preserve_dialogue(f'Mara {verb} "Changed words."', 'Mara says, “Exact words.”', 600)


def test_same_single_named_speaker_after_location_prefix_can_keep_exact_speech():
    from wee_todd_mlx.workflows.script import preserve_dialogue

    source = 'Inside the observatory, Mara closes the brass case and says “We’ll return at dawn.”'
    changed = 'Mara gently closes the brass case and says “We’ll return at dawn.”'
    assert preserve_dialogue(changed, source, 600) == changed


def test_approved_clarification_dialogue_is_part_of_treatment_authority():
    source = "The cat greets its friend."
    approved_answer = 'The cat says, "Welcome home."'
    ctx = SimpleNamespace(ask=lambda *_: json.dumps([approved_answer]), record={})
    result = execute_creative("movie.plan_treatment@1", {
        "subjects": [subject()],
        "creative_brief": {
            "sourceText": source,
            "questions": [{"answer": approved_answer, "requiresExplicitChoice": False}],
        },
        "brief": source + "\nApproved clarification: " + approved_answer,
        "duration_seconds": 5, "target_clip_seconds": 5, "frame_rate": 24,
        "observations": [],
    }, {}, ctx)
    assert result["story"]["beats"] == [approved_answer]


def test_short_authored_guided_shots_are_copied_exactly_without_weighted_calls():
    first = 'Mara gently opens the brass case and whispers “I can’t wait.”'
    second = 'At 00:05.000, Mara closes the case and says “We’ll return at dawn.”'
    calls = []
    row = subject("mara", description="An adult woman wearing a blue coat.")
    ctx = SimpleNamespace(
        ask=lambda *args: calls.append(args) or "Changed source action.", record={},
        message=lambda _: None,
    )
    result = execute_creative("movie.plan_treatment@1", {
        "subjects": [row], "creative_brief": {"questions": []},
        "brief": f"[Shot 1] {first}\n[Shot 2] {second}",
        "duration_seconds": 10, "target_clip_seconds": 5, "frame_rate": 24,
        "observations": [],
    }, {}, ctx)
    assert result["story"] == {
        "characters": [{"id": "mara", "description": "An adult woman wearing a blue coat."}],
        "beats": [first, 'Mara closes the case and says “We’ll return at dawn.”'],
    }
    assert calls == []


def test_authored_shots_cannot_silently_omit_approved_clarification_speech():
    source = "[Shot 1] The cat greets its friend."
    approved = 'In shot 1, the cat says, "Welcome home."'
    ctx = SimpleNamespace(ask=lambda *_: "The cat greets its friend.", record={},
                          message=lambda _: None)
    with pytest.raises(ValueError, match="shot assignment"):
        execute_creative("movie.plan_treatment@1", {
            "subjects": [subject()],
            "creative_brief": {
                "sourceText": source,
                "questions": [{"answer": approved, "requiresExplicitChoice": False}],
            },
            "brief": source + "\nApproved clarification: " + approved,
            "duration_seconds": 5, "target_clip_seconds": 5, "frame_rate": 24,
            "observations": [],
        }, {}, ctx)
