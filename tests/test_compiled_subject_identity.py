"""Compiled references must respect the reviewed cast and exact visible object mentions."""

import copy

import pytest
from test_workflow_creative import prompt_fixture, subject

from wee_todd_mlx.workflows.creative import compile_prompts


@pytest.mark.parametrize("neutral", [False, True])
def test_compiler_keeps_reviewed_cast_and_excludes_negated_props(neutral):
    brief, plan, rows = prompt_fixture()
    grendel = {**subject("grendel"), "name": "Grendel", "description": "UNSEEN CREATURE"}
    mother = {**subject("mother"), "name": "Grendel's mother", "description": "Drowned avenger"}
    rows.extend([grendel, mother])
    rows[1]["relationships"] = [{"role": "contains", "targetID": "grendel"}]
    plan["characters"] += [{"id": "mother", "description": "Drowned avenger"},
                           {"id": "grendel", "description": "UNSEEN CREATURE"}]
    clip = plan["clips"][0]
    clip.update(action="The cat faces Grendel's mother in the vault without the diamond.",
                startState="The cat faces Grendel's mother.",
                endState="Grendel's mother stands in the vault.", characters=["cat", "mother"])
    original = copy.deepcopy((brief, plan, rows))
    result = compile_prompts(brief, plan, rows, model_neutral=neutral)["prompts"][0]
    assert set(result["subjectIDs"]) == {"cat", "mother", "vault"}
    assert "UNSEEN CREATURE" not in result["prompt"]
    assert "A blue diamond" not in result["prompt"]
    assert (brief, plan, rows) == original


def test_qualified_object_names_resolve_shared_alias_without_inventing_a_choice():
    brief, plan, rows = prompt_fixture()
    rows[2].update(name="Blue diamond", aliases=["gem"])
    rows.append({**subject("ruby", "prop", "Red ruby"), "name": "Red gem", "aliases": ["gem"]})
    clip = plan["clips"][0]
    clip.update(action="The cat lifts the Red gem.", startState="The Red gem lies on stone.",
                endState="The cat holds the Red gem.")
    result = compile_prompts(brief, plan, rows)["prompts"][0]
    assert "ruby" in result["subjectIDs"] and "diamond" not in result["subjectIDs"]
    clip.update(action="The cat lifts the gem.", startState="The gem lies on stone.",
                endState="The cat holds the gem.")
    with pytest.raises(ValueError, match="multiple possible subjects"):
        compile_prompts(brief, plan, rows)
