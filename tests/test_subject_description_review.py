"""Visual coverage, source provenance and bounded critique for subject descriptions."""

import copy
import json

import pytest

from wee_todd_mlx.workflows.description_review import relevant_passages, review_description

POOL = {
    "id": "pool",
    "kind": "location",
    "name": "indoor pool",
    "aliases": [],
    "description": "Where the bark ends with a splash.",
    "evidence": ["The bark ends with a splash into the indoor pool."],
    "suggestions": [],
}
BRIEF = (
    "A gallery with black floors.\nA drone drops into a shallow decorative indoor pool.\n"
    + POOL["evidence"][0]
)


def draft():
    return {
        "facets": [
            {
                "aspect": "identity",
                "detail": "A shallow decorative indoor pool.",
                "basis": "source",
                "evidenceIDs": [2],
            },
            {
                "aspect": "layout_scale",
                "detail": "A compact rectangular basin along one wall.",
                "basis": "proposal",
                "evidenceIDs": [],
            },
            {
                "aspect": "materials",
                "detail": "Polished black stone surrounds the water.",
                "basis": "proposal",
                "evidenceIDs": [],
            },
            {
                "aspect": "palette_lighting",
                "detail": "Cool blue reflections illuminate the dark water.",
                "basis": "proposal",
                "evidenceIDs": [],
            },
            {
                "aspect": "features_context",
                "detail": "The pool is recessed flush with the gallery floor.",
                "basis": "proposal",
                "evidenceIDs": [],
            },
        ]
    }


class Context:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = []
        self.messages = []
        self.images = []

    def ask(self, system, prompt, images=()):
        self.images.append(list(images))
        self.calls.append((system, prompt))
        return json.dumps(next(self.values))

    def message(self, text):
        self.messages.append(text)


def test_retrieval_finds_visual_detail_missing_from_extracted_evidence():
    passages = relevant_passages(POOL, BRIEF)
    assert passages[2] == "A drone drops into a shallow decorative indoor pool."
    assert len(json.dumps(passages).encode()) <= 6500


def test_review_separates_proposals_and_preserves_identity():
    ctx = Context([draft(), {"issues": [], "missing": []}])
    result = review_description(POOL, BRIEF, ctx)
    assert result["id"] == "pool"
    assert "shallow decorative" in result["description"]
    report = result["descriptionReview"]
    assert report["status"] == "ready"
    assert len(report["proposedDetails"]) == 4
    assert report["reviewedDescription"] == result["description"]
    assert "A drone drops into a shallow decorative indoor pool." in result["evidence"]
    assert len(ctx.calls) == 2
    assert POOL["description"] == "Where the bark ends with a splash."


def test_image_claim_without_images_becomes_explicit_unverified_proposal():
    value = draft()
    value["facets"][1].update(basis="reference", evidenceIDs=[0])
    ctx = Context([value, {"issues": [], "missing": []}, value])
    result = review_description(POOL, BRIEF, ctx)
    report = result["descriptionReview"]
    assert "compact rectangular basin" in result["description"]
    assert report["status"] == "ready"
    assert report["referenceDetails"] == []
    assert any("compact rectangular basin" in text for text in report["proposedDetails"])
    assert "No images are attached" in ctx.calls[0][0]
    assert "the app handles their labels" in ctx.calls[0][0]


@pytest.mark.parametrize("basis,images", [("source", []), ("reference", ["pool-image"])])
def test_zero_citation_is_never_rebased_to_a_real_source_or_image(basis, images):
    value = draft()
    value["facets"][0].update(basis=basis, evidenceIDs=[0])
    result = review_description(POOL, BRIEF, Context([value, value]), images)
    assert result["descriptionReview"]["status"] == "needs_attention"
    assert result["description"] == POOL["description"]


def test_critic_failure_gets_one_revision_then_needs_attention():
    ctx = Context(
        [draft(), {"issues": ["Material detail contradicts the script"], "missing": []}] * 2
    )
    result = review_description(POOL, BRIEF, ctx)
    assert result["descriptionReview"]["status"] == "needs_attention"
    assert len(ctx.calls) == 4
    assert "contradicts" in ctx.calls[2][1]


def test_reference_image_reaches_both_writer_and_critic():
    value = draft()
    value["facets"][2].update(basis="reference", evidenceIDs=[1])
    ctx = Context([value, {"issues": [], "missing": []}])
    result = review_description(POOL, BRIEF, ctx, ["image:pool"])
    assert ctx.images == [["image:pool"], ["image:pool"]]
    assert result["descriptionReview"]["referenceAssets"] == ["image:pool"]
    assert len(result["descriptionReview"]["proposedDetails"]) == 3


@pytest.mark.parametrize("fault", ["citation", "missing_facet", "unsupported_basis"])
def test_invalid_provenance_or_coverage_cannot_pass(fault):
    value = copy.deepcopy(draft())
    if fault == "citation":
        value["facets"][0]["evidenceIDs"] = [999]
    elif fault == "missing_facet":
        value["facets"].pop()
    else:
        value["facets"][0]["basis"] = "verified"
    ctx = Context([value, value])
    result = review_description(POOL, BRIEF, ctx)
    assert result["descriptionReview"]["status"] == "needs_attention"
    assert result["description"] == POOL["description"]


def test_retrieval_ignores_generic_modifier_match():
    brief = BRIEF + "\nIndoors, ventilation hums and metallic footsteps echo."
    assert all("ventilation" not in text for text in relevant_passages(POOL, brief).values())


def test_image_observations_are_labeled_separately_from_design_proposals():
    value = draft()
    value["facets"][2].update(basis="reference", evidenceIDs=[1])
    result = review_description(
        POOL, BRIEF, Context([value, {"issues": [], "missing": []}]), ["pool-image"]
    )
    assert result["descriptionReview"]["referenceDetails"] == [
        "Image 1 · materials: Polished black stone surrounds the water."
    ]


def inventory(tmp_path):
    from test_studio_workflow_execution import TextBackend

    from wee_todd_mlx.workflows.service import dispatch

    selection = {k: v for k, v in POOL.items() if k != "evidence"}
    selection["evidenceIDs"] = [2]
    req = {
        "builtin": "subject-inventory",
        "inputs": {"brief": BRIEF},
        "runDirectory": str(tmp_path),
    }
    state = dispatch(
        "workflow-run", req, backend=TextBackend(["[]", "[]", json.dumps([selection])])
    )
    assert state["status"] == "awaiting_approval", state.get("error")
    assert len(state["steps"]["subjects"]["outputs"]["subjects"]) == 1
    return req, state


def update(req, state, action, backend=None, **kwargs):
    from wee_todd_mlx.workflows.service import dispatch

    subject = state["steps"]["subjects"]["outputs"]["subjects"][0]
    return dispatch(
        "workflow-review",
        {
            **req,
            "review": {
                "stepID": "subjects",
                "expectedRevision": state["revision"],
                "itemID": subject["id"],
                "action": action,
                **kwargs,
            },
        },
        backend=backend,
    )


def test_review_is_persisted_unapproved_and_approved_subject_is_protected(tmp_path):
    from test_studio_workflow_execution import TextBackend

    req, state = inventory(tmp_path)
    state = update(
        req,
        state,
        "review_description",
        TextBackend([json.dumps(draft()), '{"issues":[],"missing":[]}']),
    )
    subject = state["steps"]["subjects"]["outputs"]["subjects"][0]
    assert subject["descriptionReview"]["status"] == "ready"
    assert state["steps"]["subjects"]["items"][subject["id"]]["approved"] is False
    state = update(req, state, "approve")
    before = (tmp_path / "run.json").read_bytes()
    with pytest.raises(ValueError, match="Unlock"):
        update(req, state, "review_description", TextBackend([]))
    assert (tmp_path / "run.json").read_bytes() == before


def test_human_approval_accepts_corrections_and_preserves_agent_advice(tmp_path):
    from test_studio_workflow_execution import TextBackend

    req, state = inventory(tmp_path)
    state = update(
        req,
        state,
        "review_description",
        TextBackend([json.dumps(draft()), '{"issues":["Wrong material"],"missing":[]}'] * 2),
    )
    subject = state["steps"]["subjects"]["outputs"]["subjects"][0]
    report = copy.deepcopy(subject["descriptionReview"])
    assert report["status"] == "needs_attention"
    executions = state["executions"]
    state = update(req, state, "approve", TextBackend([]))
    assert state["steps"]["subjects"]["items"][subject["id"]]["approved"]
    assert state["executions"] == executions  # human approval must not call the model
    state = update(req, state, "unapprove")
    outputs = copy.deepcopy(state["steps"]["subjects"]["outputs"])
    outputs["subjects"][0]["description"] = "A shallow rectangular basin with pale stone edges."
    state = update(req, state, "edit", outputs=outputs)
    state = update(req, state, "approve", TextBackend([]))
    assert state["steps"]["subjects"]["items"][subject["id"]]["approved"]
    assert state["steps"]["subjects"]["outputs"]["subjects"][0]["descriptionReview"] == report
    assert state["executions"] == executions
    state = update(req, state, "unapprove")
    outputs["subjects"][0]["descriptionReview"]["reviewedDescription"] = "Changed after review"
    with pytest.raises(ValueError, match="read-only"):
        update(req, state, "edit", outputs=outputs)


def test_stale_successful_agent_review_does_not_block_human_approval(tmp_path):
    from test_studio_workflow_execution import TextBackend

    req, state = inventory(tmp_path)
    state = update(
        req,
        state,
        "review_description",
        TextBackend([json.dumps(draft()), '{"issues":[],"missing":[]}']),
    )
    outputs = copy.deepcopy(state["steps"]["subjects"]["outputs"])
    outputs["subjects"][0]["description"] = "My corrected pool description."
    state = update(req, state, "edit", outputs=outputs)
    approved = update(req, state, "approve", TextBackend([]))
    sid = outputs["subjects"][0]["id"]
    assert approved["steps"]["subjects"]["items"][sid]["approved"]
    assert approved["steps"]["subjects"]["outputs"] == outputs
    assert json.loads((tmp_path / "run.json").read_text()) == approved


def test_review_failure_saves_revision_and_can_retry(tmp_path):
    from test_studio_workflow_execution import TextBackend

    req, state = inventory(tmp_path)
    with pytest.raises(RuntimeError, match="Cancelled"):
        update(req, state, "review_description", TextBackend([RuntimeError("Cancelled")]))
    saved = json.loads((tmp_path / "run.json").read_text())
    assert saved["revision"] != state["revision"]
    assert saved["steps"]["subjects"]["outputs"] == state["steps"]["subjects"]["outputs"]
    retried = update(
        req,
        saved,
        "review_description",
        TextBackend([json.dumps(draft()), '{"issues":[],"missing":[]}']),
    )
    assert (
        retried["steps"]["subjects"]["outputs"]["subjects"][0]["descriptionReview"]["status"]
        == "ready"
    )


def test_critic_checks_only_current_candidate_not_previous_feedback():
    initial = draft()
    initial["facets"][1]["detail"] = "OUTDATED pool layout."
    ctx = Context(
        [
            initial,
            {"issues": ["Fix OUTDATED layout"], "missing": []},
            draft(),
            {"issues": [], "missing": []},
        ]
    )
    result = review_description(POOL, BRIEF, ctx)
    assert result["descriptionReview"]["status"] == "ready"
    assert "OUTDATED" in ctx.calls[2][1]  # writer must see what to fix
    assert "OUTDATED" not in ctx.calls[3][1]  # critic must not judge the old candidate


def test_new_builtin_reviews_before_requesting_approval(tmp_path):
    from test_studio_workflow_execution import TextBackend

    from wee_todd_mlx.workflows.service import dispatch

    selection = {k: v for k, v in POOL.items() if k != "evidence"}
    selection["evidenceIDs"] = [2]
    req = {
        "builtin": "subject-inventory-reviewed",
        "inputs": {"brief": BRIEF},
        "runDirectory": str(tmp_path),
    }
    backend = TextBackend(
        [
            "[]",
            "[]",
            json.dumps([selection]),
            json.dumps(
                {"description": POOL["description"], "relationships": [], "missingObjects": []}
            ),
            json.dumps(draft()),
            '{"issues":[],"missing":[]}',
            '{"relationships":[],"mentions":[],"issues":[],"missingObjects":[],"libraryMatches":[]}',
        ]
    )
    state = dispatch("workflow-run", req, backend=backend)
    assert state["status"] == "awaiting_approval", state.get("error")
    assert state["awaitingStep"] == "subjects_coverage"
    assert (
        state["steps"]["subjects_review"]["outputs"]["subjects"][0]["descriptionReview"]["status"]
        == "ready"
    )
    assert len(backend.prompts) == 7


def test_malformed_critique_has_actionable_message_not_json_dump():
    class BrokenCritic(Context):
        def ask(self, system, prompt, images=()):
            if "Review a visual DESIGN" in system:
                return '{"issues":["unfinished'
            return super().ask(system, prompt, images)

    result = review_description(POOL, BRIEF, BrokenCritic([draft(), draft()]))
    issues = result["descriptionReview"]["issues"]
    assert result["descriptionReview"]["status"] == "needs_attention"
    assert "review again" in issues[0].lower()
    assert "JSON" not in issues[0]


def test_existing_source_id_does_not_certify_invented_visual_details():
    value = draft()
    value["facets"][0]["detail"] = "A shallow decorative indoor pool lined in solid gold."
    ctx = Context([value, {"issues": [], "missing": []}])
    result = review_description(POOL, BRIEF, ctx)
    report = result["descriptionReview"]
    assert any("solid gold" in text for text in report["proposedDetails"])
    assert "solid gold" not in " ".join(result["evidence"])
    assert '"basis": "proposal"' in ctx.calls[1][1]
    assert len(ctx.calls) == 2


@pytest.mark.parametrize("kind", ["environment", "set", "clothing", "outfit"])
def test_imported_object_kinds_get_complete_visual_coverage(kind):
    from wee_todd_mlx.workflows.description_review import CRITERIA

    value = draft()
    aspects = (
        ("identity", "layout_scale", "materials", "palette_lighting", "features_context")
        if kind in {"environment", "set"}
        else ("identity", "shape_scale", "materials", "palette_finish", "distinctive_features")
    )
    for facet, aspect in zip(value["facets"], aspects, strict=True):
        facet["aspect"] = aspect
    result = review_description(
        {**POOL, "kind": kind}, BRIEF, Context([value, {"issues": [], "missing": []}])
    )
    assert result["descriptionReview"]["status"] == "ready"
    assert result["descriptionReview"]["criteria"] == list(CRITERIA[kind])


def test_writer_and_critic_receive_owned_relationships_with_target_identity():
    linked = {
        **POOL,
        "relationships": [
            {
                "id": "link_a",
                "targetID": "fountain",
                "role": "contains",
                "placement": "in the center",
            }
        ],
    }
    ctx = Context([draft(), {"issues": [], "missing": []}])
    result = review_description(
        linked, BRIEF, ctx, inventory=[{**POOL, "id": "fountain", "name": "Fountain"}]
    )
    assert result["relationships"] == linked["relationships"]
    assert all(
        '"targetID": "fountain"' in prompt and '"name": "Fountain"' in prompt
        for _, prompt in ctx.calls
    )
    assert all("linked" in system.lower() for system, _ in ctx.calls)


def test_plain_visual_design_is_labeled_as_proposals_by_host():
    values = {row["aspect"]: row["detail"] for row in draft()["facets"]}
    ctx = Context([values, {"issues": [], "missing": []}])
    result = review_description(POOL, BRIEF, ctx)
    assert result["descriptionReview"]["status"] == "ready"
    assert len(result["descriptionReview"]["proposedDetails"]) == 5
    assert result["descriptionReview"]["referenceDetails"] == []
    assert result["evidence"] == POOL["evidence"]


def test_another_inventory_object_cannot_silently_become_body_detail():
    values = {row["aspect"]: row["detail"] for row in draft()["facets"]}
    values["features_context"] = "A glowing data crystal is embedded in the pool wall."
    crystal = {"id": "crystal", "name": "glowing data crystal", "kind": "prop"}
    ctx = Context([values, {"issues": [], "missing": []}] * 2)
    result = review_description(POOL, BRIEF, ctx, inventory=[POOL, crystal])
    report = result["descriptionReview"]
    assert report["status"] == "needs_attention"
    assert any("crystal" in issue and "ID" in issue for issue in report["issues"])


def test_separate_inventory_object_referenced_by_id_is_allowed():
    values = {row["aspect"]: row["detail"] for row in draft()["facets"]}
    values["features_context"] = "Contains [crystal]."
    crystal = {"id": "crystal", "name": "crystal", "kind": "prop"}
    ctx = Context([values, {"issues": [], "missing": []}])
    result = review_description(POOL, BRIEF, ctx, inventory=[POOL, crystal])
    assert result["descriptionReview"]["status"] == "ready"


def test_reference_attachment_needs_no_model_and_preserves_description(tmp_path):
    from test_studio_workflow_execution import TextBackend

    req, state = inventory(tmp_path)
    state = update(req, state, "approve")
    old = copy.deepcopy(state["steps"]["subjects"]["outputs"]["subjects"][0])
    image = tmp_path / "generated.png"
    image.write_bytes(b"fixture image")
    backend = TextBackend([])
    backend.assets = {"sheet:one": str(image)}
    state = update(req, state, "set_reference_assets", backend, referenceAssets=["sheet:one"])
    step = state["steps"]["subjects"]
    row = step["outputs"]["subjects"][0]
    assert row["description"] == old["description"]
    assert row["evidence"] == old["evidence"]
    assert row["referenceAssets"] == ["sheet:one"]
    assert not step["items"][row["id"]]["approved"]
    assert not step.get("approved")
    assert backend.prompts == []
    with pytest.raises(ValueError, match="Relink"):
        update(req, state, "set_reference_assets", backend, referenceAssets=["missing"])
    with pytest.raises(ValueError, match="revision"):
        update(
            req, {**state, "revision": "old"}, "set_reference_assets", backend, referenceAssets=[]
        )
    reopened = json.loads((tmp_path / "run.json").read_text())
    assert reopened["steps"]["subjects"]["outputs"]["subjects"][0]["referenceAssets"] == [
        "sheet:one"
    ]
