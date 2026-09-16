"""Guided intake has immutable evidence, explicit decisions and review-only H3 drafts."""

import copy
import json

import pytest
from test_studio_workflow_execution import TextBackend

from wee_todd_mlx.workflows.service import builtin, dispatch
from wee_todd_mlx.workflows.validation import validate_document


def intake(tmp_path):
    backend = TextBackend(
        [
            json.dumps(
                {
                    "questions": [
                        {
                            "prompt": "Is Cat Woman human or feline?",
                            "options": ["Human", "Feline"],
                            "requiresExplicitChoice": True,
                        }
                    ]
                }
            )
        ]
    )
    req = {
        "builtin": "guided-movie-planning",
        "inputs": {"brief": "Cat Woman enters a vault."},
        "runDirectory": str(tmp_path),
    }
    return req, dispatch("workflow-run", req, backend=backend)


def review(req, state, action, **extra):
    return dispatch(
        "workflow-review",
        {
            **req,
            "review": {
                "stepID": "creative_brief",
                "expectedRevision": state["revision"],
                "action": action,
                **extra,
            },
        },
    )


def test_builtin_and_intake_gate(tmp_path):
    assert validate_document(builtin("guided-movie-planning"))["valid"]
    req, state = intake(tmp_path)
    assert state["awaitingStep"] == "creative_brief", state
    assert "subjects" not in state["steps"]
    brief = state["steps"]["creative_brief"]["outputs"]["creative_brief"]
    assert brief["sourceText"] == req["inputs"]["brief"]
    assert brief["facts"][0]["evidence"] == req["inputs"]["brief"]
    assert brief["questions"][0]["id"] == "question-1"
    with pytest.raises(ValueError, match="Answer"):
        review(req, state, "approve")
    outputs = copy.deepcopy(state["steps"]["creative_brief"]["outputs"])
    outputs["creative_brief"]["questions"][0]["answer"] = "Let the director decide"
    state = review(req, state, "edit", outputs=outputs)
    with pytest.raises(ValueError, match="identity|explicit"):
        review(req, state, "approve")
    outputs["creative_brief"]["questions"][0]["answer"] = "Human with feline features"
    state = review(req, state, "edit", outputs=outputs)
    state = review(req, state, "approve")
    assert state["steps"]["creative_brief"]["approved"]


@pytest.mark.parametrize("field", ["sourceText", "facts", "referenceObservations", "questions"])
def test_intake_cannot_erase_evidence_or_questions(tmp_path, field):
    req, state = intake(tmp_path)
    outputs = copy.deepcopy(state["steps"]["creative_brief"]["outputs"])
    outputs["creative_brief"][field] = (
        "changed"
        if field == "sourceText"
        else [{"image": "asset:fake", "details": "fake", "uncertainties": []}]
        if field == "referenceObservations"
        else []
    )
    with pytest.raises(ValueError):
        review(req, state, "edit", outputs=outputs)


def test_compiler_sound_policy_and_reference_binding():
    from wee_todd_mlx.workflows.creative import compile_prompts

    brief = {
        "sourceText": "A cat enters.",
        "facts": [],
        "questions": [],
        "preferences": {
            "durationSeconds": 5,
            "targetClipSeconds": 5,
            "frameRate": 24,
            "visualStyle": "Stop motion",
            "presentation": "Widescreen",
            "cameraStyle": "Slow pan",
            "audioStyle": "Effects only",
            "designPolicy": "Ask before adding details",
            "constraints": "No logos",
        },
        "referenceObservations": [
            {"image": "asset:cat", "details": "White cat", "uncertainties": []}
        ],
    }
    clips = {
        "fps": 24,
        "totalFrames": 120,
        "characters": [{"id": "cat", "description": "White cat"}],
        "clips": [
            {
                "id": "clip-1",
                "startFrame": 0,
                "frameCount": 120,
                "action": "Cat enters.",
                "startState": "Cat stands outside.",
                "endState": "Cat stands inside.",
                "location": "Door",
                "characters": ["cat"],
                "continuity": "cut",
            }
        ],
    }
    preview = compile_prompts(brief, clips, [{"id": "cat", "description": "White cat"}])
    row = preview["prompts"][0]
    assert preview["status"] == "draft"
    assert row["non_diegetic_music"] == "N/A"
    assert row["referenceAssets"] == ["asset:cat"]
    assert row["subjectIDs"] == ["cat"]
    assert "Stop motion" in row["prompt"] and "Slow pan" in row["prompt"]
    assert row["prompt"].startswith("integrated_multimodal_description: [Shot 1]")
    assert "<d>" not in row["prompt"]


def subject(sid="cat", kind="character", description="White cat"):
    return {
        "id": sid,
        "kind": kind,
        "name": sid,
        "description": description,
        "aliases": [],
        "evidence": [description],
        "suggestions": [],
    }


def test_classification_preserves_identity_and_unresolved_locations():
    from types import SimpleNamespace

    from wee_todd_mlx.workflows.creative import classify_subjects

    rows = [
        subject("cat"),
        subject("vault", "location", "Steel vault"),
        subject("jacket", "prop", "Red jacket"),
    ]
    ctx = SimpleNamespace(
        ask=lambda *args: json.dumps(
            [
                {"slot": 1, "kind": "set", "reason": "Playable vault"},
                {"slot": 2, "kind": "clothing", "reason": "Wearable"},
            ]
        )
    )
    actual = classify_subjects(rows, "A cat in a jacket enters a vault.", ctx)
    assert [r["id"] for r in actual] == [r["id"] for r in rows]
    assert [r["evidence"] for r in actual] == [r["evidence"] for r in rows]
    assert [r["kind"] for r in actual] == ["character", "set", "clothing"]
    assert "none is invented" in actual[1]["suggestions"][-1]
    ctx.ask = lambda *args: json.dumps([{"id": "fake", "kind": "character", "reason": "Fake"}])
    fallback = classify_subjects(rows, "Cat", ctx)
    assert fallback[1]["kind"] == "location"
    assert "unresolved" in fallback[1]["suggestions"][-1]


def test_explicit_library_choice_is_exact_and_reversible():
    from wee_todd_mlx.workflows.creative import apply_library_definition, execute_creative

    row = subject()
    candidate = {
        "id": "library_cat",
        "name": "Cat",
        "kind": "character",
        "description": "Blue cat",
        "aliases": [],
        "tags": [],
        "packageID": "cast",
        "version": 1,
        "definitionRevision": "rev1",
        "scope": "global",
    }
    choice = {
        "objectID": "library_cat",
        "packageID": "cast",
        "version": 1,
        "definitionRevision": "rev1",
        "scope": "global",
    }
    with pytest.raises(ValueError, match="revision"):
        apply_library_definition(row, {**choice, "version": 2}, [candidate])
    apply_library_definition(row, choice, [candidate])
    assert row["id"] == "cat" and row["evidence"] == ["White cat"]
    assert row["description"] == "Blue cat"

    class NoCalls:
        def ask(self, *args, **kwargs):
            raise AssertionError("Reused designs must not be rewritten")

    output = execute_creative(
        "project.review_creative_subjects@1",
        {
            "subjects": [row],
            "brief": "cat",
            "creative_brief": {
                "questions": [],
                "preferences": {"designPolicy": "Propose missing details for approval"},
            },
        },
        {},
        NoCalls(),
    )
    assert output["subjects"][0] == row
    apply_library_definition(row, None, [])
    assert row["description"] == "White cat" and "reusedDefinition" not in row


def test_relations_reject_wrong_parent_and_wears_direction():
    from wee_todd_mlx.workflows.creative import validate_classified_relations

    rows = [subject("vault", "set"), subject("jacket", "clothing")]
    rows[0]["relationships"] = [{"targetID": "jacket", "role": "part_of"}]
    with pytest.raises(ValueError, match="environment"):
        validate_classified_relations(rows)
    rows[0]["relationships"] = []
    rows[1]["relationships"] = [{"targetID": "vault", "role": "wears"}]
    with pytest.raises(ValueError, match="wears"):
        validate_classified_relations(rows)


def test_resolved_preferences_and_questions_are_not_dropped(tmp_path):
    from wee_todd_mlx.workflows.creative import resolve_brief

    req, state = intake(tmp_path)
    brief = state["steps"]["creative_brief"]["outputs"]["creative_brief"]
    brief["questions"][0]["answer"] = "An adult human cat burglar"
    brief["preferences"].update(cameraStyle="Slow pan", durationSeconds=17, frameRate=25)
    resolved = resolve_brief(brief)
    assert resolved["duration_seconds"] == 17 and resolved["frame_rate"] == 25
    assert "An adult human cat burglar" in resolved["brief"]
    assert "Slow pan" in resolved["brief"]
    assert resolved["brief"].startswith(req["inputs"]["brief"])


def test_guided_route_reviews_subjects_before_story_and_reuses_checkpoints():
    workflow = builtin("guided-movie-planning")
    steps = {s["id"]: s for s in workflow["steps"]}
    assert not steps["inventory"]["requiresApproval"]
    assert steps["design"]["inputs"]["subjects"]["step"] == "inventory"
    assert steps["story"]["inputs"]["subjects"]["step"] == "subjects_coverage"
    assert steps["subjects_coverage"]["requiresApproval"]
    assert steps["clips"]["operation"] == "movie.plan_creative_beats@1"
    assert steps["endpoints"]["operation"] == "movie.plan_endpoints@2"
    assert steps["check"]["operation"] == "movie.check_plan@2"
    assert not steps["prompt_preview"]["requiresApproval"]


def test_batched_classification_covers_large_inventory():
    from types import SimpleNamespace

    from wee_todd_mlx.workflows.creative import classify_subjects

    calls = []

    def ask(system, prompt):
        batch = json.loads(prompt)
        calls.append(batch)
        return json.dumps(
            [{"slot": r["slot"], "kind": "prop", "reason": "Explicit prop"} for r in batch]
        )

    rows = [subject(f"prop_{i}", "prop") for i in range(21)]
    assert len(classify_subjects(rows, "Props", SimpleNamespace(ask=ask))) == 21
    assert [len(batch) for batch in calls] == [4, 4, 4, 4, 4, 1]


def test_treatment_cast_is_host_owned(tmp_path):
    from types import SimpleNamespace

    from wee_todd_mlx.workflows.creative import execute_creative

    row = subject(description="White cat " * 180)
    ctx = SimpleNamespace(
        ask=lambda *args: json.dumps(["The cat arrives."]),
        record={},
    )
    result = execute_creative(
        "movie.plan_treatment@1",
        {
            "subjects": [row],
            "creative_brief": {"questions": []},
            "brief": "The cat arrives.",
            "duration_seconds": 5,
            "target_clip_seconds": 5,
            "frame_rate": 24,
            "observations": [],
        },
        {},
        ctx,
    )
    assert result["story"]["characters"] == [{"id": "cat", "description": row["description"]}]


def test_host_does_not_trust_false_identity_flag(tmp_path):
    from types import SimpleNamespace

    from wee_todd_mlx.workflows.creative import prepare_brief

    workflow = builtin("guided-movie-planning")
    inputs = {k: v["default"] for k, v in workflow["inputs"].items()}
    inputs.update(brief="Cat woman enters.", observations=[])
    ctx = SimpleNamespace(
        ask=lambda *args: json.dumps(
            {
                "questions": [
                    {
                        "prompt": "Human or feline?",
                        "options": ["Human", "Feline"],
                        "requiresExplicitChoice": False,
                    }
                ]
            }
        )
    )
    assert prepare_brief(inputs, ctx)["questions"][0]["requiresExplicitChoice"]


def prompt_fixture():
    preferences = {
        v: (
            24
            if v == "frameRate"
            else 5
            if v in {"durationSeconds", "targetClipSeconds"}
            else "Effects only"
            if v == "audioStyle"
            else "Let the director decide"
        )
        for v in __import__(
            "wee_todd_mlx.workflows.creative", fromlist=["PREFERENCES"]
        ).PREFERENCES.values()
    }
    brief = {
        "sourceText": "The cat steals a diamond from the vault.",
        "questions": [],
        "preferences": preferences,
        "referenceObservations": [],
    }
    plan = {
        "fps": 24,
        "totalFrames": 120,
        "characters": [{"id": "cat", "description": "WRONG"}],
        "clips": [
            {
                "id": "clip-1",
                "startFrame": 0,
                "frameCount": 120,
                "action": "The cat lifts the diamond.",
                "location": "Inside the vault",
                "startState": "The cat stands beside the diamond.",
                "endState": "The cat holds it.",
                "characters": ["cat"],
                "continuity": "cut",
            }
        ],
    }
    return (
        brief,
        plan,
        [
            subject(),
            subject("vault", "set", "Steel vault"),
            subject("diamond", "prop", "A blue diamond"),
        ],
    )


def test_compiler_resolves_named_assets_and_uses_approved_definitions():
    from wee_todd_mlx.workflows.creative import compile_prompts

    brief, plan, rows = prompt_fixture()
    rows[-1]["descriptionReview"] = {"referenceAssets": ["asset:diamond"]}
    result = compile_prompts(brief, plan, rows)["prompts"][0]
    assert result["subjectIDs"] == ["cat", "vault", "diamond"]
    assert "Steel vault" in result["prompt"] and "A blue diamond" in result["prompt"]
    assert "WRONG" not in result["prompt"]
    assert result["referenceAssets"] == ["asset:diamond"]


def test_compiler_rejects_invented_quoted_dialogue():
    from wee_todd_mlx.workflows.creative import compile_prompts

    brief, plan, rows = prompt_fixture()
    plan["clips"][0]["action"] = 'The cat lifts the diamond and says, "Mission complete!"'
    with pytest.raises(ValueError, match="dialogue"):
        compile_prompts(brief, plan, rows)
    brief["sourceText"] += ' She says, "Mission complete!"'
    preview = compile_prompts(brief, plan, rows)
    assert any("speaker" in warning for warning in preview["warnings"])


def test_identity_answer_neither_is_an_explicit_choice():
    from wee_todd_mlx.workflows.creative import require_answers

    require_answers(
        {
            "questions": [
                {
                    "answer": "Neither; she is a human in a cat costume",
                    "requiresExplicitChoice": True,
                }
            ]
        }
    )


def test_strict_design_policy_never_adopts_invented_facets():
    from types import SimpleNamespace

    from wee_todd_mlx.workflows.description_review import CRITERIA, review_description

    row = subject()
    facets = [
        {
            "aspect": aspect,
            "detail": "White cat" if index == 0 else "Invented gold armor",
            "basis": "source" if index == 0 else "proposal",
            "evidenceIDs": [1] if index == 0 else [],
        }
        for index, aspect in enumerate(CRITERIA["character"])
    ]
    calls = []

    def ask(system, prompt, images=()):
        calls.append(system)
        return json.dumps({"facets": facets} if len(calls) % 2 else {"issues": [], "missing": []})

    result = review_description(
        row, "White cat", SimpleNamespace(ask=ask, message=lambda _: None), allow_proposals=False
    )
    assert result["description"] == "White cat"
    assert result["descriptionReview"]["proposedDetails"] == []
    assert result["descriptionReview"]["status"] == "needs_attention"
    assert all("ask the user" in text for text in calls[::2])


def test_source_evidence_does_not_include_preference_metadata(tmp_path):
    from wee_todd_mlx.workflows.creative import resolve_brief

    _, state = intake(tmp_path)
    brief = state["steps"]["creative_brief"]["outputs"]["creative_brief"]
    brief["questions"][0]["answer"] = "Human in a cat costume"
    brief["preferences"]["cameraStyle"] = "A helicopter camera tracks a golden crane"
    resolved = resolve_brief(brief)
    assert "helicopter" in resolved["brief"]
    assert "helicopter" not in resolved["source_brief"]
    assert "Human in a cat costume" in resolved["source_brief"]
    steps = {s["id"]: s for s in builtin("guided-movie-planning")["steps"]}
    for sid in ["subjects", "classify", "links", "inventory", "subjects_coverage"]:
        assert steps[sid]["inputs"]["brief"]["output"] == "source_brief"


def test_library_selection_review_is_revision_checked_and_invalidates_design(tmp_path, monkeypatch):
    from wee_todd_mlx.workflows import object_coverage

    monkeypatch.setattr(
        object_coverage, "review_object_coverage", lambda rows, *args: {"subjects": rows}
    )
    definition = builtin("guided-movie-planning")
    inventory_step = next(s for s in definition["steps"] if s["id"] == "inventory")
    inventory_step["inputs"] = {
        "subjects": {"input": "subjects"},
        "brief": {"input": "brief"},
        "library": {"input": "library"},
    }
    definition["steps"] = [inventory_step]
    definition["inputs"] = {
        "subjects": {"type": "subject_list", "default": [subject()]},
        "brief": {"type": "text", "default": "White cat"},
        "library": {"type": "object_catalog", "default": []},
    }
    for value in definition["inputs"].values():
        value["label"] = "Test input"
    definition["outputs"] = {"subjects": {"step": "inventory", "output": "subjects"}}
    req = {"definition": definition, "inputs": {}, "runDirectory": str(tmp_path)}
    state = dispatch("workflow-run", req, backend=TextBackend([]))
    candidate = {
        "id": "library_cat",
        "name": "Cat",
        "kind": "character",
        "description": "Blue cat",
        "aliases": [],
        "tags": [],
        "packageID": "cast",
        "version": 1,
        "definitionRevision": "rev1",
        "scope": "global",
    }
    choice = {
        "objectID": "library_cat",
        "packageID": "cast",
        "version": 1,
        "definitionRevision": "rev1",
        "scope": "global",
    }
    mutation = {
        "stepID": "inventory",
        "expectedRevision": state["revision"],
        "action": "apply_library_definition",
        "itemID": "cat",
        "libraryChoice": choice,
        "library": [candidate],
    }
    state = dispatch("workflow-review", {**req, "review": mutation})
    saved = state["steps"]["inventory"]["outputs"]["subjects"][0]
    assert saved["description"] == "Blue cat" and saved["reusedDefinition"] == choice
    with pytest.raises(ValueError, match="revision"):
        dispatch("workflow-review", {**req, "review": mutation})
    edit = copy.deepcopy(state["steps"]["inventory"]["outputs"])
    edit["subjects"][0]["description"] = "Rewritten"
    with pytest.raises(ValueError, match="Clear"):
        dispatch(
            "workflow-review",
            {
                **req,
                "review": {
                    "stepID": "inventory",
                    "expectedRevision": state["revision"],
                    "action": "edit",
                    "outputs": edit,
                },
            },
        )
    state = dispatch(
        "workflow-review",
        {
            **req,
            "review": {**mutation, "expectedRevision": state["revision"], "libraryChoice": None},
        },
    )
    assert state["steps"]["inventory"]["outputs"]["subjects"][0]["description"] == "White cat"


def test_selected_answer_excludes_unchosen_identity_alternatives():
    from wee_todd_mlx.workflows.creative import resolve_brief

    brief, _, _ = prompt_fixture()
    brief["sourceText"] = "A cybernetic fox explores a spaceship."
    brief["questions"] = [
        {
            "id": "question-1",
            "prompt": "Is 'cybernetic fox' a fully mechanical drone or a biological fox?",
            "options": ["Mechanical drone", "Biological fox"],
            "requiresExplicitChoice": True,
            "answer": "A biological fox with cybernetic enhancements",
        }
    ]
    resolved = resolve_brief(brief)
    for key in ["brief", "source_brief"]:
        assert "drone" not in resolved[key]
        assert "Clarification for cybernetic fox: A biological fox" in resolved[key]


def test_classifier_can_repair_provisional_animal_prop():
    from types import SimpleNamespace

    from wee_todd_mlx.workflows.creative import classify_subjects

    row = subject("fox", "prop", "A cybernetic fox companion")
    result = classify_subjects(
        [row],
        row["description"],
        SimpleNamespace(
            ask=lambda *args: json.dumps(
                [{"slot": 1, "kind": "character", "reason": "An acting animal companion"}]
            )
        ),
    )
    assert result[0]["kind"] == "character"
    assert result[0]["id"] == row["id"] and result[0]["evidence"] == row["evidence"]


@pytest.mark.parametrize("quote", ["'", "‘"])
def test_compiler_rejects_invented_single_quoted_speech(quote):
    from wee_todd_mlx.workflows.creative import compile_prompts

    brief, plan, rows = prompt_fixture()
    closing = "’" if quote == "‘" else quote
    plan["clips"][0]["action"] = f"The cat whispers {quote}Mission complete!{closing}"
    with pytest.raises(ValueError, match="dialogue"):
        compile_prompts(brief, plan, rows)


def test_guided_extraction_prompts_are_specific_to_each_kind():
    from types import SimpleNamespace

    from wee_todd_mlx.workflows.subjects import identify_subjects

    calls = []

    def ask(system, prompt):
        calls.append((system, prompt))
        return "[]"

    identify_subjects(
        "An explorer visits an orbital station and landing pad.",
        SimpleNamespace(ask=ask, record={}, message=lambda _: None),
        guided=True,
    )
    assert len(calls) == 3
    location_system, location_prompt = calls[-1]
    assert "physical places" in location_system
    assert "sapphire" not in location_system and "cybernetic fox" not in location_system
    assert location_prompt.startswith("[1] ")


def test_guided_minimal_selection_copies_source_phrase_without_generated_actions():
    from types import SimpleNamespace

    from wee_todd_mlx.workflows.subjects import identify_subjects

    responses = iter(
        [
            json.dumps([{"name": "cybernetic fox", "evidenceIDs": [1]}]),
            "[]",
            "[]",
            '["cybernetic fox"]',
        ]
    )
    result = identify_subjects(
        "A cybernetic fox escapes after the alarm.",
        SimpleNamespace(ask=lambda *args: next(responses), record={}, message=lambda _: None),
        guided=True,
    )
    assert result["subjects"][0]["description"] == "cybernetic fox"
    assert result["subjects"][0]["evidence"] == ["A cybernetic fox escapes after the alarm."]
    with pytest.raises(ValueError, match="exactly"):
        identify_subjects(
            "A fox escapes.",
            SimpleNamespace(
                ask=lambda *args: json.dumps([{"name": "secret android", "evidenceIDs": [1]}]),
                record={},
                message=lambda _: None,
            ),
            guided=True,
        )


def test_guided_initial_summary_retains_cited_traits_and_user_clarification():
    from types import SimpleNamespace

    from wee_todd_mlx.workflows.subjects import identify_subjects

    responses = iter(
        [
            json.dumps([{"name": "cat woman", "evidenceIDs": [1, 2]}]),
            "[]",
            "[]",
            json.dumps(["glamorous cyberpunk cat woman", "Human with feline features"]),
        ]
    )
    result = identify_subjects(
        "A glamorous cyberpunk cat woman enters.\n"
        "Clarification for cat woman: Human with feline features",
        SimpleNamespace(ask=lambda *args: next(responses), record={}, message=lambda _: None),
        guided=True,
    )
    row = result["subjects"][0]
    assert "glamorous cyberpunk cat woman" in row["description"]
    assert "Human with feline features" in row["description"]
    assert row["id"] == "character_7a56b0421b831ed6"


def test_initial_summary_omits_invented_traits_and_keeps_known_source_details():
    from types import SimpleNamespace

    from wee_todd_mlx.workflows.subjects import describe_identified_subject

    record = {}
    row = {"name": "fox", "kind": "character", "description": "fox", "evidence": []}
    result = describe_identified_subject(
        row,
        "A silver fox sits.",
        SimpleNamespace(
            ask=lambda *args: '["silver fox","red mechanical eyes"]',
            record=record,
            message=lambda _: None,
        ),
    )
    assert result == "silver fox"
    assert record["warnings"]


def test_bad_initial_summary_preserves_existing_description_and_reports_it():
    from types import SimpleNamespace

    from wee_todd_mlx.workflows.subjects import describe_identified_subject

    record = {}
    row = {"name": "fox", "kind": "character", "description": "fox", "evidence": []}
    assert (
        describe_identified_subject(
            row,
            "A fox sits.",
            SimpleNamespace(
                ask=lambda *args: "not JSON",
                record=record,
                message=lambda _: None,
            ),
        )
        == "fox"
    )
    assert record["warnings"]


def test_plan_check_precedes_preview_and_its_warnings_survive():
    from wee_todd_mlx.workflows.creative import execute_creative

    order = validate_document(builtin("guided-movie-planning"))["topologicalOrder"]
    assert order.index("check") < order.index("prompt_preview")
    brief, plan, rows = prompt_fixture()
    inputs = {
        "creative_brief": brief,
        "clips": plan,
        "subjects": rows,
        "planning_review": {
            "structure": "valid",
            "items": [{"severity": "warning", "message": "Review timing"}],
        },
    }
    preview = execute_creative("movie.compile_h3_prompts@1", inputs, {}, None)["h3_prompts"]
    assert "Plan review: Review timing" in preview["warnings"]
    inputs["planning_review"]["structure"] = "invalid"
    with pytest.raises(ValueError, match="structural"):
        execute_creative("movie.compile_h3_prompts@1", inputs, {}, None)


def test_visual_design_prompt_preserves_intrinsic_object_nature():
    from types import SimpleNamespace

    from wee_todd_mlx.workflows.description_review import CRITERIA, review_description

    calls = []
    facets = [
        {"aspect": aspect, "detail": "biscuit", "basis": "source", "evidenceIDs": [1]}
        for aspect in CRITERIA["prop"]
    ]

    def ask(system, prompt, images=()):
        calls.append(system)
        return json.dumps({"facets": facets} if len(calls) % 2 else {"issues": [], "missing": []})

    review_description(
        subject("biscuit", "prop", "biscuit"),
        "biscuit",
        SimpleNamespace(ask=ask, message=lambda _: None),
    )
    assert "Food remains edible" in calls[0]
    assert "living characters remain living" in calls[0]
    assert "Do not invent new abilities, functions or anatomy to explain plot actions" in calls[0]


def test_prompt_inventory_links_do_not_reveal_hidden_props_or_future_sets():
    from wee_todd_mlx.workflows.creative import compile_prompts

    brief, plan, rows = prompt_fixture()
    rows += [
        subject("hidden_food", "prop", "A secret biscuit"),
        subject("future_roof", "set", "A future rooftop"),
    ]
    rows[0]["relationships"] = [
        {"role": "holds", "targetID": "hidden_food"},
        {"role": "located_in", "targetID": "future_roof"},
    ]
    prompt = compile_prompts(brief, plan, rows)["prompts"][0]
    assert "hidden_food" not in prompt["subjectIDs"] and "future_roof" not in prompt["subjectIDs"]
    assert "secret biscuit" not in prompt["prompt"]


def test_guided_shot_receives_approved_assets_and_retries_unknown_locations(monkeypatch):
    from types import SimpleNamespace

    from wee_todd_mlx.workflows import runner
    from wee_todd_mlx.workflows.structured import plan_beats

    brief, plan, rows = prompt_fixture()
    clip = plan["clips"][0]
    value = {
        k: clip[k]
        for k in ["action", "startState", "endState", "location", "characters", "continuity"]
    }
    calls = []

    class Child:
        def __init__(self, *args):
            pass

        def ask(self, system, prompt):
            calls.append(json.loads(prompt)["assignment"])
            return json.dumps({**value, "location": "unknown room" if len(calls) == 1 else "vault"})

    monkeypatch.setattr(runner, "Context", Child)
    ctx = SimpleNamespace(
        record={},
        runner=SimpleNamespace(_save=lambda: None),
        spec={},
        deadline=0,
        check=lambda: None,
        message=lambda _: None,
    )
    inputs = {
        "story": {
            "characters": plan["characters"],
            "beats": ["The cat lifts the diamond in vault."],
        },
        "subjects": rows,
        "creative_brief": brief,
        "duration_seconds": 5,
        "target_clip_seconds": 5,
        "frame_rate": 24,
    }
    output = plan_beats(inputs, {"maxClips": 200}, ctx)
    assert len(calls) == 2 and output["clips"]["clips"][0]["location"] == "vault"
    assert (
        next(r for r in calls[0]["approvedSubjects"] if r["id"] == "diamond")["description"]
        == "A blue diamond"
    )
    assert calls[0]["creativePreferences"] == brief["preferences"]


def test_rooftop_keeps_parent_id_without_showing_its_indoor_furniture():
    from wee_todd_mlx.workflows.creative import compile_prompts

    brief, plan, rows = prompt_fixture()
    roof = subject("roof", "set", "Wet rooftop tiles")
    roof["relationships"] = [{"role": "part_of", "targetID": "vault"}]
    rows[1].update(kind="environment", description="An indoor lounge with a leather sofa")
    rows.append(roof)
    plan["clips"][0].update(
        location="roof",
        action="The cat stands on the roof.",
        startState="The cat is on the roof.",
        endState="The cat waits on the roof.",
    )
    result = compile_prompts(brief, plan, rows)["prompts"][0]
    assert "vault" in result["subjectIDs"] and "roof" in result["subjectIDs"]
    assert "Wet rooftop tiles" in result["prompt"] and "leather sofa" not in result["prompt"]
