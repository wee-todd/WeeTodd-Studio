"""Coverage proposals stay bounded, anchored and subordinate to human approval."""

import copy

import pytest
from test_workflow_subject_links import inventory, request, run

from wee_todd_mlx.workflows.service import builtin, dispatch
from wee_todd_mlx.workflows.validation import validate_document, validate_value


def coverage_request(tmp_path):
    req = request(tmp_path)
    spec = req["definition"]["steps"][0]
    spec["operation"] = "project.review_object_coverage@1"
    spec["inputs"]["library"] = {"input": "library"}
    return req


def proposal(**kwargs):
    return {
        "relationships": [],
        "mentions": [],
        "issues": [],
        "missingObjects": [],
        "libraryMatches": [],
        **kwargs,
    }


def test_coverage_auto_resolves_known_names_and_preserves_source(tmp_path):
    state = run(coverage_request(tmp_path), [proposal(), proposal()])
    assert state["status"] == "awaiting_approval", state.get("error")
    rows = state["steps"]["links"]["outputs"]["subjects"]
    assert rows[0]["relationships"][0]["targetID"] == "collar"
    assert rows[0]["relationships"][0]["role"] == "wears"
    assert rows[0]["descriptionMentions"][0]["phrase"] == "collar"
    assert rows[0]["mentionSourceDescription"] == inventory()[0]["description"]
    for original, result in zip(inventory(), rows, strict=True):
        for field in ("id", "name", "kind", "description", "aliases", "evidence", "suggestions"):
            assert result[field] == original[field]
    assert not validate_value("subject_list", rows)


def test_semantic_pronoun_anchor_and_host_owned_ids(tmp_path):
    req = coverage_request(tmp_path)
    req["inputs"]["inventory"][0]["description"] = "Dog tugs at his neckwear."
    value = proposal(
        relationships=[{"targetID": "collar", "role": "wears", "placement": "neck"}],
        mentions=[{"targetID": "collar", "phrase": "his neckwear", "occurrence": 0}],
    )
    state = run(req, [value, proposal()])
    row = state["steps"]["links"]["outputs"]["subjects"][0]
    assert row["descriptionMentions"][0]["id"].startswith("mention_")
    assert row["descriptionMentions"][0]["targetID"] == "collar"


@pytest.mark.parametrize("fault", ["target", "direction", "occurrence", "duplicate", "unlinked"])
def test_invalid_proposals_do_not_override_safe_deterministic_links(tmp_path, fault):
    value = proposal(
        relationships=[{"targetID": "collar", "role": "wears", "placement": "neck"}],
        mentions=[{"targetID": "collar", "phrase": "collar", "occurrence": 0}],
    )
    if fault == "target":
        value["relationships"][0]["targetID"] = "invented"
    if fault == "direction":
        value["relationships"][0].update(targetID="dog")
    if fault == "occurrence":
        value["mentions"][0]["occurrence"] = 1
    if fault == "duplicate":
        value["mentions"] *= 2
    if fault == "unlinked":
        value["relationships"] = []
        value["mentions"][0]["targetID"] = "dog"
    state = run(coverage_request(tmp_path), [value, value, proposal()])
    row = state["steps"]["links"]["outputs"]["subjects"][0]
    assert row["coverageReview"]["status"] == "needs_attention"
    assert all(link["targetID"] == "collar" for link in row["relationships"])
    assert state["executions"] == 3


def test_library_selection_requires_exact_host_metadata(tmp_path):
    req = coverage_request(tmp_path)
    candidate = {
        "id": "saved_collar",
        "name": "Collar",
        "kind": "clothing",
        "aliases": [],
        "tags": [],
        "description": "Red leather collar",
        "packageID": "pack",
        "version": 2,
        "definitionRevision": "abc",
        "scope": "project",
    }
    req["inputs"]["library"] = [candidate]
    state = run(
        req,
        [
            proposal(),
            proposal(
                libraryMatches=[
                    {
                        "objectID": "saved_collar",
                        "packageID": "pack",
                        "version": 2,
                        "reason": "Same collar",
                    }
                ]
            ),
        ],
    )
    match = state["steps"]["links"]["outputs"]["subjects"][1]["coverageReview"]["libraryMatches"][0]
    assert match["definitionRevision"] == "abc"
    assert match["scope"] == "project"
    assert [r["id"] for r in state["steps"]["links"]["outputs"]["subjects"]] == ["dog", "collar"]


def test_stale_or_unlinked_mentions_rejected(tmp_path):
    state = run(coverage_request(tmp_path), [proposal(), proposal()])
    rows = state["steps"]["links"]["outputs"]["subjects"]
    rows[0]["description"] += " Changed."
    assert validate_value("subject_list", rows)


def test_review_action_preserves_approved_rows_and_saves_proposals(tmp_path):
    req = request(tmp_path)
    from test_workflow_subject_links import empty_proposal
    from test_workflow_subject_links import proposal as link_proposal

    state = run(req, [link_proposal(), empty_proposal()])

    def review(action, state, **extra):
        import json

        from test_studio_workflow_execution import TextBackend

        return dispatch(
            "workflow-review",
            {
                **req,
                "review": {
                    "stepID": "links",
                    "expectedRevision": state["revision"],
                    "action": action,
                    **extra,
                },
            },
            backend=TextBackend([json.dumps(proposal()), json.dumps(proposal())]),
        )

    state = review("approve", state, itemID="dog")
    original = copy.deepcopy(state["steps"]["links"]["outputs"]["subjects"][0])
    state = review("review_object_coverage", state)
    step = state["steps"]["links"]
    assert step["outputs"]["subjects"][0] == original
    assert step["items"]["dog"]["approved"] is True
    assert step["coverageReviewReport"]["proposals"][0]["id"] == "dog"
    assert len(step["coverageReviewReport"]["proposals"]) == 2
    assert "coverageReview" not in step["outputs"]["subjects"][1]


def test_builtin_final_coverage_gate_and_call_bound():
    definition = builtin("subject-inventory-reviewed")
    assert definition["version"] == "1.2.0"
    assert definition["steps"][-1]["id"] == "subjects_coverage"
    assert [s["id"] for s in definition["steps"] if s["requiresApproval"]] == ["subjects_coverage"]
    assert validate_document(definition)["valid"]


def test_reviewed_failures_are_advisory_and_calls_cannot_exceed_48(tmp_path):
    req = coverage_request(tmp_path)
    req["inputs"]["inventory"] = [
        {**inventory()[1], "id": f"item_{i}", "name": f"Item {i}", "aliases": []} for i in range(24)
    ]
    state = run(req, [RuntimeError("Offline")] * 48)
    assert state["status"] == "awaiting_approval"
    assert state["executions"] == 48
    assert all(
        row["coverageReview"]["status"] == "needs_attention"
        for row in state["steps"]["links"]["outputs"]["subjects"]
    )
    resumed = run(req, [])
    assert resumed["executions"] == 48


def test_cancellation_resume_keeps_completed_rows_and_attempt_budget(tmp_path):
    req = coverage_request(tmp_path)
    state = run(req, [proposal(), InterruptedError("Paused")])
    assert state["status"] == "cancelled"
    first = copy.deepcopy(state["steps"]["links"]["items"]["dog"]["value"])
    state = run(req, [proposal()])
    assert state["steps"]["links"]["outputs"]["subjects"][0] == first
    assert state["executions"] == 3


def test_ambiguous_name_does_not_autolink_and_unknown_library_rejected(tmp_path):
    req = coverage_request(tmp_path)
    req["inputs"]["inventory"].append({**inventory()[1], "id": "other_collar"})
    invalid = proposal(
        libraryMatches=[
            {"objectID": "made_up", "packageID": "pack", "version": 1, "reason": "invented"}
        ]
    )
    state = run(req, [invalid, invalid, proposal(), proposal()])
    row = state["steps"]["links"]["outputs"]["subjects"][0]
    assert row["relationships"] == []
    assert row["descriptionMentions"] == []
    assert row["coverageReview"]["libraryMatches"] == []
    assert "Ambiguous" in row["coverageReview"]["issues"][0]


def test_human_text_edit_drops_old_anchors_without_altering_report(tmp_path):
    req = coverage_request(tmp_path)
    state = run(req, [proposal(), proposal()])
    outputs = copy.deepcopy(state["steps"]["links"]["outputs"])
    report = outputs["subjects"][0]["coverageReview"]
    outputs["subjects"][0]["description"] = "A small dog."
    state = dispatch(
        "workflow-review",
        {
            **req,
            "review": {
                "action": "edit",
                "stepID": "links",
                "expectedRevision": state["revision"],
                "outputs": outputs,
            },
        },
    )
    row = state["steps"]["links"]["outputs"]["subjects"][0]
    assert row.get("descriptionMentions", []) == []
    assert row["coverageReview"] == report


def test_semantic_possessive_phrase_replaces_short_deterministic_anchor(tmp_path):
    req = coverage_request(tmp_path)
    req["inputs"]["inventory"][0]["description"] = "Dog wears his collar."
    value = proposal(
        relationships=[{"targetID": "collar", "role": "wears", "placement": "neck"}],
        mentions=[{"targetID": "collar", "phrase": "his collar", "occurrence": 0}],
    )
    state = run(req, [value, proposal()])
    row = state["steps"]["links"]["outputs"]["subjects"][0]
    assert row["coverageReview"]["status"] == "ready"
    assert row["descriptionMentions"][0]["phrase"] == "his collar"
    assert len(row["relationships"]) == 1


def test_distinct_same_named_targets_are_not_silently_resolved(tmp_path):
    req = coverage_request(tmp_path)
    req["inputs"]["inventory"].append({**inventory()[1], "id": "other_collar"})
    value = proposal(
        relationships=[{"targetID": "collar", "role": "wears", "placement": "neck"}],
        mentions=[{"targetID": "collar", "phrase": "collar", "occurrence": 0}],
    )
    state = run(req, [value, value, proposal(), proposal()])
    row = state["steps"]["links"]["outputs"]["subjects"][0]
    assert row["relationships"] == []
    assert row["coverageReview"]["status"] == "needs_attention"


def test_missing_objects_are_structured_proposals_with_literal_evidence(tmp_path):
    req = coverage_request(tmp_path)
    req["inputs"]["inventory"][0]["description"] += " It has a brass tag."
    missing = {
        "name": "Brass tag",
        "kind": "prop",
        "description": "A small brass tag.",
        "evidence": ["a brass tag"],
    }
    state = run(req, [proposal(missingObjects=[missing]), proposal()])
    rows = state["steps"]["links"]["outputs"]["subjects"]
    assert len(rows) == 2
    assert rows[0]["coverageReview"]["missingObjects"] == [missing]
    assert rows[0]["coverageReview"]["status"] == "needs_attention"


def test_mixed_valid_components_survive_bad_links_anchors_and_failed_retry(tmp_path):
    req = coverage_request(tmp_path)
    req["inputs"]["inventory"][0]["description"] = "Dog tugs his collar beside a brass tag."
    good_link = {"targetID": "collar", "role": "wears", "placement": "neck"}
    missing = {
        "name": "Brass tag",
        "kind": "prop",
        "description": "A brass tag.",
        "evidence": ["a brass tag"],
    }
    value = proposal(
        relationships=[good_link, {"targetID": "invented", "role": "uses", "placement": ""}],
        mentions=[
            {"targetID": "collar", "phrase": "his collar", "occurrence": 0},
            {"targetID": "collar", "phrase": "COLLAR", "occurrence": 4},
        ],
        missingObjects=[missing],
    )
    state = run(req, [value, RuntimeError("Offline"), proposal()])
    row = state["steps"]["links"]["outputs"]["subjects"][0]
    assert row["relationships"][0]["targetID"] == "collar"
    assert row["descriptionMentions"][0]["phrase"] == "his collar"
    assert row["coverageReview"]["missingObjects"] == [missing]
    assert row["coverageReview"]["status"] == "needs_attention"
    assert any("relationship" in issue.lower() for issue in row["coverageReview"]["issues"])
    assert state["executions"] == 3


def test_library_match_survives_reversed_link_and_invalid_component_shape(tmp_path):
    req = coverage_request(tmp_path)
    candidate = {
        "id": "saved_collar",
        "name": "Collar",
        "kind": "clothing",
        "aliases": [],
        "tags": [],
        "description": "Red leather collar",
        "packageID": "pack",
        "version": 2,
        "definitionRevision": "abc",
        "scope": "global",
    }
    req["inputs"]["library"] = [candidate]
    value = proposal(
        relationships=[{"targetID": "dog", "role": "wears", "placement": ""}, {"targetID": "dog"}],
        libraryMatches=[
            {"objectID": "saved_collar", "packageID": "pack", "version": 2, "reason": "Same collar"}
        ],
    )
    state = run(req, [proposal(), value, value])
    row = state["steps"]["links"]["outputs"]["subjects"][1]
    assert row["relationships"] == []
    assert row["coverageReview"]["libraryMatches"][0]["objectID"] == "saved_collar"
    assert row["coverageReview"]["status"] == "needs_attention"


def test_partial_components_checkpoint_before_cancel_and_survive_resume(tmp_path):
    req = coverage_request(tmp_path)
    value = proposal(
        relationships=[
            {"targetID": "collar", "role": "wears", "placement": "neck"},
            {"targetID": "invented", "role": "holds", "placement": ""},
        ]
    )
    state = run(req, [value, InterruptedError("Paused")])
    assert state["status"] == "cancelled"
    # The interrupted review must finish before moving to the next subject.
    state = run(req, [proposal(), proposal()])
    assert (
        state["steps"]["links"]["outputs"]["subjects"][0]["relationships"][0]["placement"] == "neck"
    )
    assert state["executions"] == 4


def test_prompt_scopes_owner_and_library_relevance_with_eight_candidate_bound(tmp_path):
    import json

    from test_studio_workflow_execution import TextBackend

    req = coverage_request(tmp_path)
    catalog = [
        {
            "id": f"saved_{index}",
            "name": f"Coat {index}",
            "kind": "clothing",
            "aliases": [],
            "tags": [],
            "description": "Long design. " * 100,
            "packageID": f"package_{index}",
            "version": 1,
            "definitionRevision": "revision",
        }
        for index in range(16)
    ]
    catalog[-1]["name"] = "Collar"
    req["inputs"]["library"] = catalog
    backend = TextBackend([json.dumps(proposal()), json.dumps(proposal())])
    state = dispatch("workflow-run", req, backend=backend)
    assert state["status"] == "awaiting_approval"
    owner = json.loads(backend.prompts[0])
    assert owner["CURRENT_OBJECT_ONLY"] == "dog"
    assert owner["currentDescription"] == inventory()[0]["description"]
    assert owner["libraryCandidatesForCurrentObjectOnly"] == []
    context = json.loads(backend.prompts[1])
    assert context["CURRENT_OBJECT_ONLY"] == "collar"
    assert "wears" not in context["legalOutgoingRoles"]
    candidates = context["libraryCandidatesForCurrentObjectOnly"]
    assert len(candidates) == 8
    assert candidates[0]["name"] == "Collar"
    assert all(len(candidate["description"]) <= 500 for candidate in candidates)
    assert all(row["id"] != "collar" for row in context["otherInventoryRowsForContextOnly"])


def test_exact_library_alias_match_is_host_pinned_even_when_model_fails(tmp_path):
    req = coverage_request(tmp_path)
    candidate = {
        "id": "saved_collar",
        "name": "Red neckwear",
        "kind": "clothing",
        "aliases": ["  COLLAR  "],
        "tags": [],
        "description": "Red leather collar.",
        "packageID": "saved_pack",
        "version": 3,
        "scope": "project",
        "definitionRevision": "pinned_revision",
    }
    req["inputs"]["library"] = [candidate, {**candidate, "id": "actor", "kind": "character"}]
    state = run(req, [proposal(), RuntimeError("Offline"), RuntimeError("Offline")])
    rows = state["steps"]["links"]["outputs"]["subjects"]
    assert rows[0]["coverageReview"]["libraryMatches"] == []
    matches = rows[1]["coverageReview"]["libraryMatches"]
    assert len(matches) == 1
    assert matches[0]["objectID"] == "saved_collar"
    assert matches[0]["packageID"] == "saved_pack"
    assert matches[0]["version"] == 3
    assert matches[0]["definitionRevision"] == "pinned_revision"
    assert matches[0]["scope"] == "project"
    assert "verify" in matches[0]["reason"].lower()
    assert [row["id"] for row in rows] == ["dog", "collar"]


def test_unrelated_missing_object_from_shared_source_not_copied_to_other_rows(tmp_path):
    req = coverage_request(tmp_path)
    req["inputs"]["brief"] = "Dog wears Collar. A brass tag lies beside Dog."
    req["inputs"]["inventory"][0]["description"] += " A brass tag lies beside Dog."
    missing = {
        "name": "Brass tag",
        "kind": "prop",
        "description": "A small brass tag.",
        "evidence": [req["inputs"]["brief"]],
    }
    value = proposal(missingObjects=[missing])
    state = run(req, [value, value, value])
    rows = state["steps"]["links"]["outputs"]["subjects"]
    assert rows[0]["coverageReview"]["missingObjects"] == [missing]
    assert rows[1]["coverageReview"]["missingObjects"] == []
    assert any("current object" in issue.lower() for issue in rows[1]["coverageReview"]["issues"])


def test_source_owned_missing_object_supported_without_description_mention(tmp_path):
    req = coverage_request(tmp_path)
    req["inputs"]["brief"] = "Dog holds an ornate helmet."
    missing = {
        "name": "Ornate helmet",
        "kind": "prop",
        "description": "An ornate helmet.",
        "evidence": [req["inputs"]["brief"]],
    }
    state = run(req, [proposal(missingObjects=[missing]), proposal()])
    assert state["steps"]["links"]["outputs"]["subjects"][0]["coverageReview"][
        "missingObjects"
    ] == [missing]
