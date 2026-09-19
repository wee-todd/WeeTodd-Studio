"""Draft reference evidence is preserved independently of renderer input limits."""

import copy

import pytest
from test_workflow_creative import prompt_fixture

from wee_todd_mlx.workflows.creative import compile_prompts
from wee_todd_mlx.workflows.schema import value_schema
from wee_todd_mlx.workflows.validation import validate_value


def inventory(count, references):
    return [
        {
            "id": f"subject_{index}",
            "name": f"Subject {index}",
            "kind": "character" if index < 3 else "prop",
            "description": f"Exact subject {index} description.",
            "referenceAssets": [f"description-ref:{index}-{r}" for r in range(references)],
        }
        for index in range(count)
    ]


def request(rows):
    brief, plan, _ = prompt_fixture()
    characters = [row for row in rows if row["kind"] == "character"]
    plan["characters"] = [{"id": r["id"], "description": r["description"]} for r in characters]
    plan["clips"][0].update(
        characters=[r["id"] for r in characters],
        action="; ".join(row["name"] for row in rows) + " remain visible.",
        startState="The subjects wait.",
        endState="The subjects advance.",
        location="Hall",
    )
    return brief, plan


@pytest.mark.parametrize("music", [False, True])
def test_three_characters_with_three_references_compile_without_losing_any_evidence(music):
    rows = inventory(3, 3)
    brief, plan = request(rows)
    before = copy.deepcopy((brief, plan, rows))
    preview = compile_prompts(brief, plan, rows, model_neutral=music)
    assert preview["prompts"][0]["referenceAssets"] == [
        key for row in rows for key in row["referenceAssets"]
    ]
    assert not validate_value("h3_prompt_preview", preview)
    assert any("unassigned" in note and "selected shot" in note for note in preview["warnings"])
    assert (brief, plan, rows) == before


def test_maximum_draft_evidence_preserves_global_and_subject_references_in_order():
    rows = inventory(24, 8)
    brief, plan = request(rows)
    brief["referenceObservations"] = [{"image": f"asset:global-{i}"} for i in range(8)]
    preview = compile_prompts(brief, plan, rows)
    assets = preview["prompts"][0]["referenceAssets"]
    assert assets == [row["image"] for row in brief["referenceObservations"]] + [
        key for row in rows for key in row["referenceAssets"]
    ]
    assert len(assets) == 200
    assert not validate_value("h3_prompt_preview", preview)
    preview["prompts"][0]["referenceAssets"].append("asset:one-too-many")
    assert validate_value("h3_prompt_preview", preview)


def test_draft_evidence_deduplicates_shared_refs_and_keeps_legacy_review_references():
    rows = inventory(3, 3)
    rows[1]["referenceAssets"][0] = rows[0]["referenceAssets"][0]
    rows[2]["descriptionReview"] = {"referenceAssets": rows[2].pop("referenceAssets")}
    brief, plan = request(rows)
    preview = compile_prompts(brief, plan, rows)
    assert len(preview["prompts"][0]["referenceAssets"]) == 8
    assert (
        rows[2]["descriptionReview"]["referenceAssets"][-1]
        in preview["prompts"][0]["referenceAssets"]
    )


def test_preview_evidence_bound_tracks_source_contracts_without_expanding_image_inputs():
    preview = value_schema("h3_prompt_preview")["properties"]["prompts"]["items"]["properties"]
    subject_refs = value_schema("subject_list")["items"]["properties"]["referenceAssets"]
    bound = (
        preview["subjectIDs"]["maxItems"] * subject_refs["maxItems"]
        + value_schema("observation_list")["maxItems"]
    )
    assert value_schema("draft_reference_evidence_list")["maxItems"] == bound
    assert value_schema("image_list")["maxItems"] == 8
    assert validate_value("image_list", [f"asset:{i}" for i in range(9)])
