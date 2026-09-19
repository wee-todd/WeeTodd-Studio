"""A known compound identity must not imply its namesake is also visible."""

import pytest

from wee_todd_mlx.workflows.music_context import (
    candidate_subject_ids,
    named_subject_ids,
    validate_visible_subjects,
)


def subjects():
    return [
        {"id": "grendel", "name": "Grendel", "kind": "character", "aliases": []},
        {"id": "mother", "name": "Grendel’s mother", "kind": "character",
         "aliases": ["Mother of Grendel"]},
        {"id": "mere", "name": "Black Mere", "kind": "environment", "aliases": []},
    ]


@pytest.mark.parametrize("text", [
    "Grendel’s mother reaches the water.",
    "Grendel's mother reaches the water.",
    "Mother of Grendel reaches the water.",
])
def test_compound_known_identity_does_not_include_namesake(text):
    assert named_subject_ids(subjects(), text) == {"mother"}
    assert named_subject_ids(subjects(), text, positive_only=True) == {"mother"}


@pytest.mark.parametrize("text", [
    "Grendel’s mother stands beside Grendel.",
    "Grendel watches. Grendel's mother stands beside the water.",
    "Mother of Grendel kneels beside Grendel.",
])
def test_a_separate_namesake_mention_is_still_required(text):
    assert named_subject_ids(subjects(), text, positive_only=True) == {"mother", "grendel"}


def test_negated_compound_does_not_reveal_its_namesake_but_separate_mention_survives():
    text = "Without Grendel’s mother. Grendel waits at the Black Mere."
    assert named_subject_ids(subjects(), text, positive_only=True) == {"grendel", "mere"}
    assert named_subject_ids(subjects(), "Without Grendel’s mother.", positive_only=True) == set()


def test_mother_only_shot_validates_without_inventing_son():
    shot = {"action": "Grendel's mother rises through the Black Mere.",
            "startState": "Grendel's mother is below the surface.",
            "endState": "Grendel's mother is above the surface.",
            "location": "Black Mere", "characters": ["mother"]}
    assert candidate_subject_ids(subjects(), shot) == {"mother", "mere"}
    validate_visible_subjects(subjects(), shot, {"mother", "mere"})
    with pytest.raises(ValueError, match=r"Black Mere \(mere\)"):
        validate_visible_subjects(subjects(), shot, {"mother"})


def test_shared_exact_alias_remains_ambiguous():
    rows = subjects()
    rows[0]["aliases"] = ["The creature"]
    rows[1]["aliases"] = ["The creature"]
    assert named_subject_ids(rows, "The creature waits.") == {"mother", "grendel"}


@pytest.mark.parametrize("name,text", [
    ("Beowulf’s final sword", "His sword Hrunting fails against the mother."),
    ("Ceremonial ivory cup", "She drinks from her battered traveling cup."),
    ("Coronation cloak", "He removes his rain-soaked traveling cloak."),
    ("Silver royal shield", "She sets aside a wooden practice shield."),
])
def test_generic_head_never_assigns_a_qualified_unique_object(name, text):
    rows = [{"id": "special", "name": name, "kind": "prop", "aliases": []}]
    candidate = {"action": text, "startState": "", "endState": "", "location": ""}
    assert candidate_subject_ids(rows, candidate) == set()
    validate_visible_subjects(rows, candidate, set())


def test_exact_qualified_name_and_explicit_alias_still_require_declaration():
    rows = [{"id": "special", "name": "Beowulf’s final sword", "kind": "prop",
             "aliases": ["dragon-battle blade"]}]
    for text in ["He raises Beowulf's final sword.", "He raises the dragon-battle blade."]:
        candidate = {"action": text, "startState": "", "endState": "", "location": ""}
        assert candidate_subject_ids(rows, candidate) == {"special"}
        with pytest.raises(ValueError, match="Beowulf’s final sword"):
            validate_visible_subjects(rows, candidate, set())
        validate_visible_subjects(rows, candidate, {"special"})


def test_generic_multiple_variants_need_clarification_without_assigning_all_identities():
    rows = [{"id": "white", "name": "Ivory cup", "kind": "prop"},
            {"id": "gold", "name": "Golden cup", "kind": "prop"}]
    candidate = {"action": "She lifts her cup.", "startState": "", "endState": "", "location": ""}
    assert candidate_subject_ids(rows, candidate) == set()
    with pytest.raises(ValueError, match="Ambiguous object reference.*cup"):
        validate_visible_subjects(rows, candidate, set())
    candidate["action"] = "She lifts the Ivory cup."
    validate_visible_subjects(rows, candidate, {"white"})


def test_declared_qualified_object_must_still_be_named_exactly():
    rows = [{"id": "final", "name": "Final battle sword", "kind": "prop"}]
    candidate = {"action": "She lifts her sword.", "startState": "", "endState": "", "location": ""}
    with pytest.raises(ValueError, match="Name each visible object.*Final battle sword"):
        validate_visible_subjects(rows, candidate, {"final"})


def test_old_identity_diagnostic_does_not_contaminate_resume_or_completed_clip(monkeypatch):
    from test_music_shot_constraints import at_square, location_inputs
    from test_music_shot_json_format import harness, shot

    from wee_todd_mlx.workflows.music import execute_music
    from wee_todd_mlx.workflows.runner import digest

    ctx, calls = harness(monkeypatch, [shot(), shot(), shot(), at_square()])
    value = location_inputs()
    with pytest.raises(ValueError):
        execute_music("music.plan_beats@1", value, {"maxClips": 200}, ctx)
    completed_key = ctx.record["items"]["clip-1"]["key"]
    failed = ctx.record["items"]["clip-2"]
    previous = ctx.record["items"]["clip-1"]["value"]
    failed["musicFailureContext"] = digest({
        "assignment": failed["base"],
        "previous": {"endState": previous["endState"], "location": previous["location"]},
        "identityRules": 2,
    })
    failed["error"] = "Old generic noun incorrectly selected final sword"
    execute_music("music.plan_beats@1", value, {"maxClips": 200}, ctx)
    assert "Old generic noun" not in calls[-1][1]
    assert ctx.record["items"]["clip-1"]["key"] == completed_key
    assert len(calls) == 4
