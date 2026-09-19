"""Link transport preserves evidence and restores host identities after compact requests."""

import copy
import json

import pytest
from test_studio_workflow_execution import TextBackend
from test_workflow_subject_links import inventory, proposal, request

from wee_todd_mlx.workflows.service import dispatch


def test_link_request_compacts_24_object_ids_and_deduplicates_exact_evidence(tmp_path):
    req = request(tmp_path)
    rows = []
    for index in range(24):
        row = copy.deepcopy(inventory()[0])
        row.update(
            id=f"character_{index:016x}",
            name=f"Warrior {index:02d}",
            description=f"Warrior {index:02d} wears mail.",
            evidence=["Warrior 00 carries a shield.", "Exact evidence outside the brief."],
        )
        rows.append(row)
    req["inputs"]["inventory"] = rows
    req["inputs"]["brief"] = "Warrior 00 carries a shield."
    original = copy.deepcopy(rows)

    class CaptureBackend(TextBackend):
        def generate(self, model, system, prompt, images, **kwargs):
            self.prompts.append(prompt)
            payload = json.loads(prompt)
            current = payload["subject"]
            return {
                "text": json.dumps(
                    {
                        "description": current["description"],
                        "relationships": [],
                        "missingObjects": [],
                    }
                ),
                "truncated": False,
            }

    backend = CaptureBackend([])
    state = dispatch("workflow-run", req, backend=backend)
    assert state["status"] == "awaiting_approval", state.get("error")
    payload = json.loads(backend.prompts[0])
    assert all(row["id"] not in backend.prompts[0] for row in rows)
    assert len(payload["inventory"]) == 23
    assert len({row["id"] for row in payload["inventory"]}) == 23
    evidence = [
        payload["sourcePassages"][str(ref["passageID"])] if "passageID" in ref else ref["text"]
        for ref in payload["subject"]["evidence"]
    ]
    assert evidence == original[0]["evidence"]
    assert rows == original
    output = state["steps"]["links"]["outputs"]["subjects"]
    assert [row["id"] for row in output] == [row["id"] for row in original]
    assert [row["evidence"] for row in output] == [row["evidence"] for row in original]


@pytest.mark.parametrize("unknown", [False, True])
def test_link_response_restores_compact_target_or_rejects_unknown(tmp_path, unknown):
    req = request(tmp_path)
    req["inputs"]["inventory"][0]["id"] = "character_1111111111111111"
    req["inputs"]["inventory"][1]["id"] = "prop_2222222222222222"

    class AliasBackend(TextBackend):
        def generate(self, model, system, prompt, images, **kwargs):
            self.prompts.append(prompt)
            payload = json.loads(prompt)
            if payload["subject"]["name"] == "Dog":
                target = payload["allowedTargetIDsByRole"]["wears"][0]
                assert len(target) <= 6
                value = proposal("not_supplied" if unknown else target)
            else:
                value = {
                    "description": "A red leather collar.",
                    "relationships": [],
                    "missingObjects": [],
                }
            return {"text": json.dumps(value), "truncated": False}

    backend = AliasBackend([])
    state = dispatch("workflow-run", req, backend=backend)
    assert state["status"] == "awaiting_approval", state.get("error")
    current = state["steps"]["links"]["outputs"]["subjects"][0]
    if unknown:
        assert current.get("relationships", []) == []
        assert current["relationshipReview"]["status"] == "needs_attention"
    else:
        assert current["relationships"][0]["targetID"] == "prop_2222222222222222"
        assert current["id"] == "character_1111111111111111"
        assert current["relationshipReview"]["status"] == "ready"


def test_music_video_uses_guided_relationship_kind_rules(tmp_path):
    req = request(tmp_path)
    req["definition"]["id"] = "weetodd.music-video-planning"
    req["inputs"]["inventory"][1]["kind"] = "prop"
    backend = TextBackend(
        [
            json.dumps(proposal()),
            json.dumps(proposal()),
            json.dumps(
                {
                    "description": "A red leather collar.",
                    "relationships": [],
                    "missingObjects": [],
                }
            ),
        ]
    )
    state = dispatch("workflow-run", req, backend=backend)
    assert state["status"] == "awaiting_approval", state.get("error")
    assert json.loads(backend.prompts[0])["allowedTargetIDsByRole"]["wears"] == []
    assert state["steps"]["links"]["outputs"]["subjects"][0].get("relationships", []) == []


def test_compact_ids_cannot_collide_with_existing_short_ids():
    from wee_todd_mlx.workflows.context_budget import inventory_context
    from wee_todd_mlx.workflows.subject_links import allowed_targets, compact_link_request

    rows = [copy.deepcopy(inventory()[0]) for _ in range(3)]
    for row, saved_id in zip(rows, ["r0", "character_1111111111111111", "r1"], strict=True):
        row["id"] = saved_id
    selected = inventory_context(rows, rows[0], "")
    packed, originals = compact_link_request(
        rows[0],
        selected,
        allowed_targets(rows[0], {row["id"]: row for row in rows}),
        {},
        [],
    )
    assert originals == {"r0": "r0", "r2": "character_1111111111111111", "r1": "r1"}
    assert {row["id"] for row in packed["inventory"]} == {"r1", "r2"}


def test_existing_link_target_is_compact_in_request_and_restored_with_stable_link_id(tmp_path):
    req = request(tmp_path)
    dog, collar = req["inputs"]["inventory"]
    dog["id"], collar["id"] = "character_1111111111111111", "prop_2222222222222222"
    dog["relationships"] = [
        {
            "id": "existing_link",
            "targetID": collar["id"],
            "role": "wears",
            "placement": "around the neck",
        }
    ]

    class ExistingLinkBackend(TextBackend):
        def generate(self, model, system, prompt, images, **kwargs):
            payload = json.loads(prompt)
            current = payload["subject"]
            relationships = current.get("relationships", [])
            if relationships:
                target = relationships[0]["targetID"]
                assert target in payload["allowedTargetIDsByRole"]["wears"]
                assert target != collar["id"]
                value = proposal(target)
            else:
                value = {
                    "description": "A red leather collar.",
                    "relationships": [],
                    "missingObjects": [],
                }
            return {"text": json.dumps(value), "truncated": False}

    state = dispatch("workflow-run", req, backend=ExistingLinkBackend([]))
    assert state["status"] == "awaiting_approval", state.get("error")
    linked = state["steps"]["links"]["outputs"]["subjects"][0]
    assert linked["relationships"] == dog["relationships"]


def test_illegal_role_retry_lists_active_request_ids_and_restores_corrected_target(tmp_path):
    req = request(tmp_path)
    dog, collar = req["inputs"]["inventory"]
    dog["id"], collar["id"] = "character_1111111111111111", "clothing_2222222222222222"
    bag = {**copy.deepcopy(collar), "id": "prop_3333333333333333", "name": "Bag", "kind": "prop"}
    req["inputs"]["inventory"].append(bag)
    req["definition"]["id"] = "weetodd.music-video-planning"

    class RepairBackend(TextBackend):
        def generate(self, model, system, prompt, images, **kwargs):
            self.prompts.append(prompt)
            payload = json.loads(prompt)
            if payload["subject"]["name"] == "Dog":
                if not payload["previousIssues"]:
                    target = next(row["id"] for row in payload["inventory"] if row["name"] == "Bag")
                else:
                    target = payload["allowedTargetIDsByRole"]["wears"][0]
                    issue = payload["previousIssues"][0]
                    assert target in issue
                    assert collar["id"] not in issue
                value = proposal(target)
            else:
                value = {
                    "description": payload["subject"]["description"],
                    "relationships": [],
                    "missingObjects": [],
                }
            return {"text": json.dumps(value), "truncated": False}

    backend = RepairBackend([])
    state = dispatch("workflow-run", req, backend=backend)
    assert state["status"] == "awaiting_approval", state.get("error")
    linked = state["steps"]["links"]["outputs"]["subjects"][0]
    assert linked["relationships"][0]["targetID"] == collar["id"]
    assert len(backend.prompts) == 4
