"""Guided visual design respects curated identity and rejects unresolved candidates."""

import copy
import json

import pytest
from test_studio_workflow_execution import TextBackend
from test_subject_description_review import Context
from test_workflow_subject_links import request as base_request

from wee_todd_mlx.workflows.creative import execute_creative
from wee_todd_mlx.workflows.service import builtin, dispatch

BASELINE = (
    "The same face as Young Beowulf, now lined and scarred; powerful, never frail. "
    "Brown hair and beard threaded with gray. Dark mail, a weathered cloak, "
    "an intact sword at his hip and a gold arm-ring worn on his arm."
)
BRIEF = (
    "Old Beowulf is a powerful gray-haired king.\n"
    "In the dragon fight his sword breaks and his shield blackens.\n"
    "The dying king gives his gold arm-ring to Wiglaf."
)
ROW = dict(
    id="old_beowulf",
    name="Old Beowulf",
    kind="character",
    aliases=[],
    description=BASELINE,
    evidence=["Old Beowulf is a powerful gray-haired king."],
    suggestions=[],
)
CLEAN = dict(
    identity="The same face as Young Beowulf, now lined and scarred; powerful, never frail.",
    build_features="Broad shoulders and a powerful upright stance.",
    face_hair="Brown hair and beard threaded with gray.",
    palette_materials="Dark iron mail over a coarse charcoal wool cloak.",
    wardrobe_equipment="An intact sword at his hip and a gold arm-ring worn on his arm.",
)
BAD = {
    **CLEAN,
    "wardrobe_equipment": "A broken sword, blackened shield and gold arm-ring held out for Wiglaf.",
}
CONFLICT = {
    "issues": [
        "Later battle damage and the final handover must not become his reusable base wardrobe."
    ],
    "missing": [],
}
OK = {"issues": [], "missing": []}


def design(values, images=()):
    ctx = Context(values)
    row = {**ROW, "referenceAssets": list(images)}
    result = execute_creative(
        "project.review_creative_subjects@1",
        {
            "subjects": [row],
            "brief": BRIEF,
            "creative_brief": {
                "questions": [],
                "preferences": {"designPolicy": "Propose missing details for approval"},
            },
        },
        {},
        ctx,
    )
    return result["subjects"][0], ctx


def facets(values):
    return {
        "facets": [
            dict(aspect=k, detail=v, basis="proposal", evidenceIDs=[]) for k, v in values.items()
        ]
    }


@pytest.mark.parametrize("images", [[], ["ref:old"]])
def test_guided_writer_and_critic_receive_current_identity_and_conditional_state_rules(images):
    result, ctx = design([facets(CLEAN) if images else CLEAN, OK], images)
    assert result["descriptionReview"]["status"] == "ready"
    assert len(ctx.calls) == 2
    for system, prompt in ctx.calls:
        assert BASELINE in prompt
        assert "Old Beowulf (character)" in prompt
        assert "authoritative" in system.lower() + prompt.lower()
        assert "distinctions from other subjects" in system
        assert "never contradict or silently drop" in system
        assert "Unreviewed draft" not in prompt
        assert "conditional" in system.lower()
        assert "reference" in system.lower() and "temporary" in system.lower()


def test_guided_other_subject_identity_proposal_cannot_replace_source_row():
    original = {
        **ROW,
        "name": "Hrothgar",
        "id": "hrothgar",
        "description": "The elderly Danish king, regal and broad, never his lean young retainer.",
    }
    retainer = {
        **ROW,
        "name": "Retainer",
        "id": "retainer",
        "description": "A lean young retainer, distinct from the elderly king.",
    }
    swapped = {**CLEAN, "identity": "A lean young retainer."}
    conflict = {"issues": ["This makes the king into his young retainer."], "missing": []}
    ctx = Context([swapped, conflict, swapped, conflict, CLEAN, OK])
    result = execute_creative(
        "project.review_creative_subjects@1",
        {
            "subjects": [original, retainer],
            "brief": BRIEF,
            "creative_brief": {
                "questions": [],
                "preferences": {"designPolicy": "Propose missing details for approval"},
            },
        },
        {},
        ctx,
    )["subjects"][0]
    assert result["id"] == original["id"]
    assert result["name"] == original["name"]
    assert result["description"] == original["description"]
    assert result["descriptionReview"]["status"] == "needs_attention"
    for system, prompt in ctx.calls[:4]:
        assert original["description"] in prompt
        assert "Hrothgar (character)" in prompt
        assert "distinctions from other subjects" in system


def test_guided_clean_visual_expansion_is_retained_for_human_review():
    result, _ = design([CLEAN, OK])
    assert "coarse charcoal wool" in result["description"]
    assert "never frail" in result["description"]
    assert result["description"] != BASELINE
    assert result["descriptionReview"]["proposedDetails"]
    assert ROW["description"] == BASELINE


def test_guided_temporal_conflict_preserves_original_and_exposes_proposals():
    result, _ = design([BAD, CONFLICT, BAD, CONFLICT])
    assert result["description"] == BASELINE
    assert result["evidence"] == ROW["evidence"]
    report = result["descriptionReview"]
    assert report["status"] == "needs_attention"
    assert report["reviewedDescription"] == BASELINE
    assert any("broken sword" in detail for detail in report["proposedDetails"])
    assert any("handover" in issue for issue in report["issues"])


def test_guided_bad_final_response_cannot_leak_prior_rejected_candidate():
    result, _ = design([BAD, CONFLICT, {"invalid": "response"}])
    assert result["description"] == BASELINE
    assert result["descriptionReview"]["status"] == "needs_attention"
    assert any(
        "broken sword" in detail for detail in result["descriptionReview"]["proposedDetails"]
    )


def test_guided_rejected_source_facet_does_not_expand_canonical_evidence():
    from wee_todd_mlx.workflows.description_review import review_description

    event = "Old Beowulf holds a broken sword after the dragon fight."
    candidate = facets(BAD)
    candidate["facets"][-1].update(detail=event, basis="source", evidenceIDs=[2])
    result = review_description(
        ROW,
        ROW["evidence"][0] + "\n" + event,
        Context([candidate, CONFLICT, candidate, CONFLICT]),
        ["ref:old"],
        preserve_current_identity=True,
    )
    assert result["description"] == BASELINE
    assert result["evidence"] == ROW["evidence"]
    assert result["descriptionReview"]["status"] == "needs_attention"


def test_manual_review_still_replaces_unreviewed_description():
    from wee_todd_mlx.workflows.description_review import review_description

    result = review_description(ROW, BRIEF, Context([CLEAN, OK]))
    assert "coarse charcoal wool" in result["description"]
    assert result["description"] != BASELINE


def workflow_request(tmp_path):
    req = base_request(tmp_path)
    guided = builtin("guided-movie-planning")
    req["definition"]["models"] = guided["models"]
    step = req["definition"]["steps"][0]
    step["operation"] = "project.review_creative_subjects@1"
    step["inputs"]["creative_brief"] = {"step": "creative_brief", "output": "creative_brief"}
    req["definition"]["inputs"].update(guided["inputs"])
    prepare = copy.deepcopy(next(s for s in guided["steps"] if s["id"] == "creative_brief"))
    prepare["requiresApproval"] = False
    describe = copy.deepcopy(next(s for s in guided["steps"] if s["id"] == "describe"))
    req["definition"]["steps"] = [describe, prepare, step]
    req["inputs"].update(inventory=[copy.deepcopy(ROW)], brief=BRIEF)
    return req


@pytest.mark.parametrize("approved", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_guided_contract_change_rechecks_only_unapproved_saved_design(
    tmp_path, monkeypatch, approved, legacy
):
    from wee_todd_mlx.workflows import description_review

    req = workflow_request(tmp_path)
    first = dispatch(
        "workflow-run",
        req,
        backend=TextBackend(['{"questions":[]}', json.dumps(CLEAN), json.dumps(OK)]),
    )
    assert first["status"] == "awaiting_approval", first.get("error")
    if approved:
        first = dispatch(
            "workflow-review",
            {
                **req,
                "review": {
                    "stepID": "links",
                    "action": "approve",
                    "expectedRevision": first["revision"],
                },
            },
            backend=TextBackend([]),
        )
    saved = copy.deepcopy(first)
    if legacy:
        # An old execution has neither the request revision nor its qualification marker.
        saved["steps"]["links"]["key"] = "old-guided-design-execution"
        saved["steps"]["links"].pop("guidedDesignVersion", None)
    else:
        monkeypatch.setattr(description_review, "GUIDED_DESIGN_VERSION", 2)
    (tmp_path / "run.json").write_text(json.dumps(saved))
    backend = TextBackend([] if approved else [json.dumps(CLEAN), json.dumps(OK)])
    resumed = dispatch("workflow-run", req, backend=backend)
    assert resumed["status"] == ("completed" if approved else "awaiting_approval"), resumed.get(
        "error"
    )
    if approved:
        assert resumed["steps"]["links"]["approved"]
        assert resumed["steps"]["links"]["outputs"] == first["steps"]["links"]["outputs"]
        assert resumed["steps"]["links"].get("guidedDesignVersion") == (None if legacy else 1)
        assert not backend.prompts
    else:
        assert len(backend.prompts) == 2
        assert resumed["steps"]["links"].get("guidedDesignVersion") == (1 if legacy else 2)


@pytest.mark.parametrize("human_work", ["approve", "edit", "references"])
def test_guided_contract_change_preserves_downstream_reviewed_inventory(tmp_path, human_work):
    req = workflow_request(tmp_path)
    design_step = req["definition"]["steps"][-1]
    design_step["requiresApproval"] = False
    inventory = copy.deepcopy(design_step)
    inventory.update(id="inventory", name="Review inventory", requiresApproval=True)
    inventory["inputs"]["subjects"] = {"step": "links", "output": "subjects"}
    req["definition"]["steps"].append(inventory)
    req["definition"]["outputs"] = {"subjects": {"step": "inventory", "output": "subjects"}}
    state = dispatch(
        "workflow-run",
        req,
        backend=TextBackend([json.dumps(v) for v in [{"questions": []}, CLEAN, OK, CLEAN, OK]]),
    )
    assert state["status"] == "awaiting_approval", state.get("error")
    if human_work in {"approve", "edit"}:
        mutation = {
            "stepID": "inventory",
            "action": human_work,
            "expectedRevision": state["revision"],
        }
        if human_work == "edit":
            mutation["outputs"] = copy.deepcopy(state["steps"]["inventory"]["outputs"])
            mutation["outputs"]["subjects"][0]["description"] += (
                " A human-authored identity detail."
            )
        state = dispatch("workflow-review", {**req, "review": mutation}, backend=TextBackend([]))
    else:
        # Legacy attachments may predate persisted human-decision records.
        state["steps"]["inventory"]["outputs"]["subjects"][0]["referenceAssets"] = ["ref:old"]
    expected = copy.deepcopy(state["steps"]["inventory"]["outputs"])
    state["steps"]["links"].update(key="old-guided-design-execution")
    state["steps"]["links"].pop("guidedDesignVersion", None)
    (tmp_path / "run.json").write_text(json.dumps(state))
    backend = TextBackend([])
    resumed = dispatch("workflow-run", req, backend=backend)
    assert resumed["status"] == ("completed" if human_work == "approve" else "awaiting_approval"), (
        resumed.get("error")
    )
    assert resumed["steps"]["inventory"]["outputs"] == expected
    assert "guidedDesignVersion" not in resumed["steps"]["links"]
    assert not backend.prompts
