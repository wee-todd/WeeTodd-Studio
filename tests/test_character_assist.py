import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def field(identifier, section, kind="text", *, authored=False, max_length=240):
    return {
        "id": identifier,
        "label": identifier,
        "section": section,
        "kind": kind,
        "authoredOnly": authored,
        "maxLength": max_length,
    }


CATALOG = [
    field("identity.species", 2),
    field("identity.authoredAge", 2, authored=True),
    field("body.build", 3),
    field("eyes.color", 4),
    field("hair.color", 5),
    field("garments.coat.color", 6),
    field("features.scar.side", 7),
    field("surfaces.coat.texture", 8),
    field("style.presetID", 9, "choice"),
    field("camera.projection", 10),
    field("lighting.key", 11),
]


def test_inventory_inference_enforces_deadline(tmp_path, monkeypatch):
    import studio_character_assist as service

    value = request(tmp_path, "text")
    value["sourceText"] = "A jacket and trousers."
    value["fields"] = [field("garments[].type", 6)]
    value["requestedFields"] = ["garments[first].type", "garments[second].type"]
    now = [0.0]
    stopped = []
    monkeypatch.setattr(service.time, "monotonic", lambda: now[0])

    def fake_assist(payload, *, cancelled, **_):
        now[0] = service.CALL_TIMEOUT_SECONDS + 1
        stopped.append(cancelled())
        return {"text": '{"records":[]}', "truncated": False}

    monkeypatch.setattr(service, "assist", fake_assist)
    with pytest.raises(TimeoutError, match="timed out"):
        service.extract_fields(value)
    assert stopped == [True]


def test_inventory_preserves_single_slot_collections(tmp_path, monkeypatch):
    import studio_character_assist as service

    value = request(tmp_path, "text")
    value["sourceText"] = "A jacket, trousers, and a bracelet."
    value["fields"] = [field("garments[].type", 6), field("accessories[].type", 7)]
    value["requestedFields"] = [
        "garments[first].type",
        "garments[second].type",
        "accessories[only].type",
    ]

    def fake_assist(payload, **_):
        prompt = json.loads(payload["textRequest"]["prompt"])
        if "collections" in prompt:
            return {
                "text": json.dumps(
                    {
                        "records": [
                            {"collection": "garments", "slot": 1, "excerpt": "jacket"},
                            {"collection": "garments", "slot": 2, "excerpt": "trousers"},
                        ]
                    }
                ),
                "truncated": False,
            }
        alias = prompt["allowedFields"][0]
        return completion(
            alias,
            {"garment1.type": "jacket", "garment2.type": "trousers", "accessory1.type": "bracelet"}[
                alias
            ],
        )

    monkeypatch.setattr(service, "assist", fake_assist)
    result = service.extract_fields(value)
    assert {item["field"] for item in result["proposals"]} == set(value["requestedFields"])


def test_conflicting_fields_after_repair_are_all_omitted_and_cache_stays_clean(
    tmp_path, monkeypatch
):
    import studio_character_assist as service

    value = request(tmp_path)
    value["fields"] = [field("hair.color", 5), field("hair.length", 5)]
    value["requestedFields"] = ["hair.color", "hair.length"]
    calls = []

    def fake_assist(payload, **_):
        calls.append(payload)
        proposals = [
            json.loads(completion("hair.color", color)["text"])["proposals"][0]
            for color in ("red", "blue")
        ]
        proposals += json.loads(completion("hair.length", "short")["text"])["proposals"]
        return {"text": json.dumps({"proposals": proposals}), "truncated": False}

    monkeypatch.setattr(service, "assist", fake_assist)
    first = service.extract_fields(value)
    assert [item["field"] for item in first["proposals"]] == ["hair.length"]
    assert first["diagnostics"]
    assert {item["field"] for item in first["diagnostics"]} == {"hair.color"}
    second = service.extract_fields(value)
    assert second["proposals"] == first["proposals"]
    assert len(calls) == 2


def request(tmp_path, role="character"):
    image = tmp_path / "source.png"
    image.write_bytes(b"source")
    model = tmp_path / "qwen_3.5_4b_i8x.ckpt"
    model.write_bytes(b"model")
    return {
        "runtime": {"drawThingsHelperPath": "/unused/helper"},
        "modelPath": str(model),
        "modelFingerprint": "qwen-test-model-v1",
        "role": role,
        "sourceImage": {
            "path": str(image),
            "label": "Reference",
            "sha256": hashlib.sha256(b"source").hexdigest(),
            "crop": {"x": 0, "y": 0, "width": 1, "height": 1},
            "orientation": 1,
        },
        "fields": CATALOG,
        "capturedTarget": {"documentID": "doc-1", "revision": 7},
        "cacheDirectory": str(tmp_path / "cache"),
    }


def completion(field_id, value="brown"):
    return {
        "text": json.dumps(
            {
                "proposals": [
                    {
                        "field": field_id,
                        "value": value,
                        "evidence": "visible on the source",
                        "uncertainty": "",
                    }
                ]
            }
        ),
        "truncated": False,
    }


def test_head_detail_is_hashed_and_only_sent_to_face_hair_batch(tmp_path, monkeypatch):
    import studio_character_assist as service

    value = request(tmp_path)
    detail = tmp_path / "head.jpg"
    detail.write_bytes(b"bounded head detail")
    value["sourceDetailImages"] = [
        {
            "path": str(detail),
            "label": "primary head and face detail",
            "sha256": hashlib.sha256(b"bounded head detail").hexdigest(),
        }
    ]
    value["fields"] = [
        field("body.build", 3),
        field("face.skinTone", 4, "color"),
        field("face.skinTexture", 4),
        field("hair.scalpCoverage", 5),
        field("hair.length", 5),
    ]
    value["requestedFields"] = [item["id"] for item in value["fields"]]
    calls = []

    def fake_assist(payload, **_):
        calls.append(payload["textRequest"])
        return {"text": '{"proposals":[]}', "truncated": False}

    monkeypatch.setattr(service, "assist", fake_assist)
    service.extract_fields(value)
    assert len(calls) == 3
    body = calls[0]
    skin = next(
        call for call in calls if "face.skinTone" in json.loads(call["prompt"])["allowedFields"]
    )
    head = next(
        call
        for call in calls
        if "hair.scalpCoverage" in json.loads(call["prompt"])["allowedFields"]
    )
    assert len(body["images"]) == 1
    assert len(skin["images"]) == 2
    assert [image["path"] for image in head["images"]] == [
        value["sourceImage"]["path"],
        str(detail),
    ]
    prompt = json.loads(head["prompt"])
    skin_prompt = json.loads(skin["prompt"])
    assert "bald or sparse crown" in prompt["fieldRules"]["hair.scalpCoverage"]["meaning"]
    assert "never infer ancestry" in skin_prompt["fieldRules"]["face.skinTone"]["meaning"]
    assert "never describe smoothing" in skin_prompt["fieldRules"]["face.skinTexture"]["meaning"]
    assert (
        "never call a bald or sparse crown short" in prompt["fieldRules"]["hair.length"]["meaning"]
    )
    assert "do not beautify" in head["systemPrompt"]


def test_skin_forensics_are_a_standalone_head_detail_batch(tmp_path, monkeypatch):
    import studio_character_assist as service

    value = request(tmp_path)
    detail = tmp_path / "head.jpg"
    detail.write_bytes(b"head")
    value["sourceDetailImages"] = [
        {
            "path": str(detail),
            "label": "primary head and face detail",
            "sha256": hashlib.sha256(b"head").hexdigest(),
        }
    ]
    fields = [
        "face.shape",
        "face.covering",
        "face.skinTone",
        "face.skinTexture",
        "eyes.color",
        "hair.scalpCoverage",
        "hair.style",
    ]
    value["fields"] = [
        field(
            item, 5 if item.startswith("hair.") else 4, "color" if item.endswith("Tone") else "text"
        )
        for item in fields
    ]
    value["requestedFields"] = fields
    calls = []

    def fake_assist(payload, **_):
        calls.append(payload["textRequest"])
        return {"text": '{"proposals":[]}', "truncated": False}

    monkeypatch.setattr(service, "assist", fake_assist)
    service.extract_fields(value)
    allowed = [json.loads(call["prompt"])["allowedFields"] for call in calls]
    assert [item for item in allowed if set(item) & service.FORENSIC_SKIN_FIELDS] == [
        ["face.covering", "face.skinTone", "face.skinTexture"]
    ]
    skin_call = calls[allowed.index(["face.covering", "face.skinTone", "face.skinTexture"])]
    assert len(skin_call["images"]) == 2
    assert all(field_name not in skin_call["prompt"] for field_name in ("face.shape", "eyes.color"))


def test_hair_location_and_musculature_rules_are_explicit(tmp_path):
    import studio_character_assist as service

    value = request(tmp_path)
    value["fields"] = [field("hair.style", 5), field("body.musculature", 3)]
    rules = service._field_rules(value, ["hair.style", "body.musculature"])
    assert "value itself must name its location" in rules["hair.style"]["meaning"]
    assert "soft tissue" in rules["body.musculature"]["meaning"]


def test_head_detail_contract_and_hash_are_strict(tmp_path):
    import studio_character_assist as service

    value = request(tmp_path)
    detail = tmp_path / "head.jpg"
    detail.write_bytes(b"detail")
    value["sourceDetailImages"] = [
        {"path": str(detail), "label": "Head", "sha256": hashlib.sha256(b"detail").hexdigest()}
    ]
    with pytest.raises(ValueError, match="hash does not match"):
        service.extract_fields(value)


def test_head_detail_participates_in_cache_and_mutation_identity(tmp_path, monkeypatch):
    import studio_character_assist as service

    value = request(tmp_path)
    detail = tmp_path / "head.jpg"
    detail.write_bytes(b"first")
    value["sourceDetailImages"] = [
        {
            "path": str(detail),
            "label": "primary head and face detail",
            "sha256": hashlib.sha256(b"first").hexdigest(),
        }
    ]

    def mutate_detail(*_args, **_kwargs):
        detail.write_bytes(b"changed")
        return {"text": '{"proposals":[]}', "truncated": False}

    monkeypatch.setattr(service, "assist", mutate_detail)
    with pytest.raises(ValueError, match="source detail changed"):
        service.extract_fields(value)


@pytest.mark.parametrize("face_field_count", [0, 12])
def test_character_hair_forensics_are_one_atomic_batch(tmp_path, monkeypatch, face_field_count):
    import studio_character_assist as service

    hair = [
        "hair.length",
        "hair.color",
        "hair.scalpCoverage",
        "hair.facialHair",
        "hair.style",
        "hair.texture",
        "hair.hairline",
    ]
    face = [f"face.fact{index}" for index in range(face_field_count)]
    value = request(tmp_path)
    value["fields"] = [field(item, 4) for item in face] + [field(item, 5) for item in hair]
    value["requestedFields"] = face + hair
    calls = []

    def fake_assist(payload, **_):
        calls.append(json.loads(payload["textRequest"]["prompt"])["allowedFields"])
        return {"text": '{"proposals":[]}', "truncated": False}

    monkeypatch.setattr(service, "assist", fake_assist)
    service.extract_fields(value)
    hair_calls = [allowed for allowed in calls if set(allowed) & set(hair)]
    assert hair_calls == [hair]


def test_character_hair_atomic_batch_fails_if_output_budget_cannot_hold_it(tmp_path, monkeypatch):
    import studio_character_assist as service

    hair = sorted(service.FORENSIC_HAIR_FIELDS)
    value = request(tmp_path)
    value["fields"] = [field("face.shape", 4)] + [field(item, 5) for item in hair]
    value["requestedFields"] = ["face.shape", *hair]
    monkeypatch.setattr(service, "MAX_FIELDS_PER_BATCH", len(hair) - 1)
    with pytest.raises(ValueError, match="record cannot fit"):
        service.extract_fields(value)


def test_model_json_fence_is_unwrapped_without_repair(tmp_path, monkeypatch):
    import studio_character_assist as service

    calls = []

    def fenced(payload, **_):
        calls.append(payload)
        result = completion("style.presetID", "photograph")
        result["text"] = "```json\n" + result["text"] + "\n```"
        return result

    monkeypatch.setattr(service, "assist", fenced)
    result = service.extract_fields(request(tmp_path, "style"))
    assert result["proposals"][0]["value"] == "photograph"
    assert len(calls) == 1


def test_character_groups_face_and_hair_and_never_offers_authored_fields(tmp_path, monkeypatch):
    import studio_character_assist as service

    seen = []

    def fake_assist(value, **_):
        seen.append(value["textRequest"])
        allowed = json.loads(value["textRequest"]["prompt"])["allowedFields"]
        return completion(allowed[0])

    monkeypatch.setattr(service, "assist", fake_assist)
    result = service.extract_fields(request(tmp_path), progress=lambda _: None)

    assert len(seen) == 3
    offered = {item for call in seen for item in json.loads(call["prompt"])["allowedFields"]}
    assert "identity.authoredAge" not in offered
    assert {p["field"] for p in result["proposals"]} <= {
        "identity.species",
        "body.build",
        "eyes.color",
        "hair.color",
        "garments.coat.color",
        "surfaces.coat.texture",
        "features.scar.side",
    }
    assert result["capturedTarget"] == {"documentID": "doc-1", "revision": 7}
    assert result["sourceHash"] == hashlib.sha256(b"source").hexdigest()
    assert result["role"] == "character"


def test_style_pass_cannot_return_character_fields(tmp_path, monkeypatch):
    import studio_character_assist as service

    monkeypatch.setattr(service, "assist", lambda *_args, **_kwargs: completion("eyes.color"))
    result = service.extract_fields(request(tmp_path, "style"))
    assert result["proposals"] == []
    assert result["diagnostics"][0]["field"] == "eyes.color"


def test_malformed_response_gets_one_repair_then_stops(tmp_path, monkeypatch):
    import studio_character_assist as service

    calls = []

    def fake_assist(value, **_):
        calls.append(value)
        return {"text": "not json", "truncated": False}

    monkeypatch.setattr(service, "assist", fake_assist)
    value = request(tmp_path, "style")
    with pytest.raises(ValueError, match="JSON"):
        service.extract_fields(value)
    assert len(calls) == 2
    assert "repair" in calls[1]["textRequest"]["systemPrompt"].lower()


def test_truncated_response_is_repaired_once(tmp_path, monkeypatch):
    import studio_character_assist as service

    answers = [
        {"text": '{"proposals": [', "truncated": True},
        completion("style.presetID", "photograph"),
    ]
    monkeypatch.setattr(service, "assist", lambda *_args, **_kwargs: answers.pop(0))
    result = service.extract_fields(request(tmp_path, "style"))
    assert result["proposals"][0]["value"] == "photograph"


def test_validated_cache_reuses_content_but_returns_current_target(tmp_path, monkeypatch):
    import studio_character_assist as service

    calls = 0

    def fake_assist(value, **_):
        nonlocal calls
        calls += 1
        allowed = json.loads(value["textRequest"]["prompt"])["allowedFields"]
        return completion(allowed[0])

    monkeypatch.setattr(service, "assist", fake_assist)
    first = request(tmp_path, "style")
    assert service.extract_fields(first)["capturedTarget"]["revision"] == 7
    second = request(tmp_path, "style")
    second["capturedTarget"] = {"documentID": "doc-2", "revision": 99}
    assert service.extract_fields(second)["capturedTarget"] == {
        "documentID": "doc-2",
        "revision": 99,
    }
    assert calls == 1


def test_cancel_and_budget_fail_before_provider_call(tmp_path, monkeypatch):
    import studio_character_assist as service

    monkeypatch.setattr(
        service,
        "assist",
        lambda *_args, **_kwargs: pytest.fail("provider must not be called"),
    )
    with pytest.raises(InterruptedError):
        service.extract_fields(request(tmp_path, "style"), cancelled=lambda: True)
    oversized = request(tmp_path, "text")
    oversized.pop("sourceImage")
    oversized["sourceText"] = "x" * 24_001
    with pytest.raises(ValueError, match="24,000"):
        service.extract_fields(oversized)


def test_rejects_wrong_hash_and_missing_model_before_provider(tmp_path, monkeypatch):
    import studio_character_assist as service

    monkeypatch.setattr(service, "assist", lambda *_args, **_kwargs: pytest.fail("provider called"))
    value = request(tmp_path)
    value["sourceImage"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        service.extract_fields(value)
    value = request(tmp_path)
    Path(value["modelPath"]).unlink()
    with pytest.raises(ValueError, match="model"):
        service.extract_fields(value)


def test_progress_has_bounded_pass_sequence(tmp_path, monkeypatch):
    import studio_character_assist as service

    def fake_assist(value, **_):
        allowed = json.loads(value["textRequest"]["prompt"])["allowedFields"]
        return completion(allowed[0])

    monkeypatch.setattr(service, "assist", fake_assist)
    messages = []
    service.extract_fields(request(tmp_path), progress=messages.append)
    assert messages[0] == "Preparing image"
    assert sum(message.startswith("Reading character") for message in messages) == 3
    assert messages[-1] == "Ready for review"


def test_validator_rejects_authored_image_fact_and_wrong_typed_value():
    from wee_todd_mlx.character_fields import validate_proposals

    payload = {
        "proposals": [
            {
                "field": "identity.authoredAge",
                "value": "34",
                "evidence": "looks adult",
                "uncertainty": "",
            }
        ]
    }
    with pytest.raises(ValueError, match="Authored-only"):
        validate_proposals(
            payload,
            role="character",
            allowed_fields={"identity.authoredAge"},
            catalog=[field("identity.authoredAge", 2, authored=True)],
        )
    payload["proposals"][0].update(field="body.height", value="tall")
    with pytest.raises(ValueError, match="text"):
        validate_proposals(
            {"proposals": [{**payload["proposals"][0], "value": ["tall"]}]},
            role="text",
            allowed_fields={"body.height"},
            catalog=[field("body.height", 3, kind="measurement")],
        )


def test_cache_evicts_oldest_validated_entries(tmp_path, monkeypatch):
    import studio_character_assist as service

    monkeypatch.setattr(service, "CACHE_LIMIT_BYTES", 500)
    root = tmp_path / "cache"
    for index in range(5):
        service._write_cache(
            root,
            str(index),
            {"proposals": [{"id": str(index), "field": "style.finish", "value": "x" * 180}]},
        )
    assert sum(item.stat().st_size for item in root.glob("*.json")) <= 500
    assert not (root / "0.json").exists()


def test_text_mapping_covers_all_appearance_fields_and_exact_host_repeat_paths(
    tmp_path, monkeypatch
):
    import studio_character_assist as service

    value = request(tmp_path, "text")
    value.pop("sourceImage")
    value["sourceText"] = "A brown-eyed figure in a red coat."
    value["fields"] = CATALOG + [field("garments[].color", 6)]
    value["requestedFields"] = ["eyes.color", "garments[coat-uuid].color"]
    seen = []

    def fake_assist(payload, **_):
        allowed = json.loads(payload["textRequest"]["prompt"])["allowedFields"]
        seen.extend(allowed)
        return completion(allowed[0])

    monkeypatch.setattr(service, "assist", fake_assist)
    result = service.extract_fields(value)
    assert seen == ["eyes.color", "garment1.color"]
    assert [item["field"] for item in result["proposals"]] == [
        "eyes.color",
        "garments[coat-uuid].color",
    ]
    assert result["proposals"][0]["state"] == "value"


def test_real_catalog_shape_allows_text_role_even_when_extraction_roles_are_character(
    tmp_path, monkeypatch
):
    import studio_character_assist as service

    value = request(tmp_path, "text")
    value.pop("sourceImage")
    value["sourceText"] = "A 170 cm tall person with blue eyes."
    value["schemaVersion"] = 1
    value["fields"] = [
        {
            "key": "body.height",
            "section": 3,
            "label": "Height",
            "valueKind": "measurement",
            "suggestions": [],
            "applicability": [],
            "defaultRequired": False,
            "canBeAbsent": True,
            "extractionRoles": ["character"],
            "maxLength": 240,
        },
        {
            "key": "eyes.color",
            "section": 4,
            "label": "Eye color",
            "valueKind": "color",
            "suggestions": ["Blue"],
            "applicability": [],
            "defaultRequired": False,
            "canBeAbsent": True,
            "extractionRoles": ["character"],
            "maxLength": 240,
        },
    ]
    value["requestedFields"] = ["body.height", "eyes.color"]

    def fake_assist(*_args, **_kwargs):
        return {
            "text": json.dumps(
                {
                    "proposals": [
                        {
                            "field": "body.height",
                            "value": "170 cm",
                            "evidence": "explicit text",
                            "uncertainty": "",
                        },
                        {
                            "field": "eyes.color",
                            "value": "blue",
                            "evidence": "explicit text",
                            "uncertainty": "",
                        },
                    ]
                }
            ),
            "truncated": False,
        }

    monkeypatch.setattr(service, "assist", fake_assist)
    result = service.extract_fields(value)
    assert [item["field"] for item in result["proposals"]] == [
        "body.height",
        "eyes.color",
    ]
    assert result["metadata"] == {
        "schemaVersion": 1,
        "extractionVersion": 1,
        "promptVersion": 12,
        "modelFingerprint": "qwen-test-model-v1",
    }


def test_rejects_unresolved_repeat_template(tmp_path, monkeypatch):
    import studio_character_assist as service

    monkeypatch.setattr(service, "assist", lambda *_args, **_kwargs: pytest.fail("provider called"))
    value = request(tmp_path)
    value["fields"] = CATALOG + [field("garments[].type", 6)]
    value["requestedFields"] = ["garments[].type"]
    with pytest.raises(ValueError, match="record path"):
        service.extract_fields(value)


def test_source_mutation_during_call_rejects_result_and_cache(tmp_path, monkeypatch):
    import studio_character_assist as service

    value = request(tmp_path, "style")
    source = Path(value["sourceImage"]["path"])

    def mutate_source(*_args, **_kwargs):
        source.write_bytes(b"changed")
        return completion("style.presetID", "photograph")

    monkeypatch.setattr(service, "assist", mutate_source)
    with pytest.raises(ValueError, match="changed during extraction"):
        service.extract_fields(value)
    assert not list(Path(value["cacheDirectory"]).glob("*.json"))


def test_requested_source_transform_mutation_rejects_result(tmp_path, monkeypatch):
    import studio_character_assist as service

    value = request(tmp_path, "style")

    def mutate_request(*_args, **_kwargs):
        value["sourceImage"]["crop"]["width"] = 0.5
        return completion("style.presetID", "photograph")

    monkeypatch.setattr(service, "assist", mutate_request)
    with pytest.raises(ValueError, match="source changed"):
        service.extract_fields(value)


@pytest.mark.parametrize(
    ("kind", "value"),
    [("measurement", "tall"), ("measurement", "170"), ("color", "#00ff00")],
)
def test_catalog_kinds_reject_unconvertible_display_values(kind, value):
    from wee_todd_mlx.character_fields import validate_proposals

    with pytest.raises(ValueError, match="value"):
        validate_proposals(
            {
                "proposals": [
                    {
                        "field": "test.field",
                        "value": value,
                        "evidence": "supplied",
                        "uncertainty": "",
                    }
                ]
            },
            role="text",
            allowed_fields={"test.field"},
            catalog=[{"key": "test.field", "valueKind": kind, "maxLength": 240}],
        )


def test_full_swift_catalog_and_uuid_pools_are_split_into_bounded_text_batches(
    tmp_path, monkeypatch
):
    import studio_character_assist as service

    scalar_sections = {
        2: [
            "identity.species",
            "identity.type",
            "identity.authoredAge",
            "identity.authoredSexGender",
            "identity.authoredAncestry",
            "identity.designClass",
        ],
        3: [
            "body.plan",
            "body.height",
            "body.scale",
            "body.build",
            "body.shoulders",
            "body.torso",
            "body.arms",
            "body.legs",
            "body.musculature",
            "body.massDistribution",
            "body.posture",
            "body.pose",
        ],
        4: [
            "head.shape",
            "face.shape",
            "face.jaw",
            "face.cheekbones",
            "face.noseMuzzleBeak",
            "face.mouth",
            "face.lips",
            "eyes.shape",
            "eyes.color",
            "eyes.spacing",
            "brows.shape",
            "ears.shape",
            "face.covering",
            "face.expression",
        ],
        5: [
            "hair.color",
            "hair.length",
            "hair.texture",
            "hair.style",
            "hair.hairline",
            "hair.facialHair",
            "covering.pattern",
            "covering.growthDirection",
        ],
    }
    repeat_sections = {
        6: (
            "garments",
            [
                "type",
                "bodyRegion",
                "layer",
                "fit",
                "cut",
                "color",
                "closures",
                "seams",
                "trim",
                "condition",
                "placement",
            ],
            3,
        ),
        7: (
            "accessories",
            ["type", "color", "placement", "side", "orientation", "shape", "condition"],
            2,
        ),
        17: (
            "features",
            ["type", "shape", "color", "placement", "side", "orientation", "asymmetry"],
            2,
        ),
        8: (
            "surfaces",
            ["target", "material", "finish", "texture", "pattern", "wear", "detailScale"],
            3,
        ),
    }
    catalog = []
    requested = []
    for section, keys in scalar_sections.items():
        for key in keys:
            catalog.append(
                {
                    "key": key,
                    "section": section,
                    "valueKind": "text",
                    "extractionRoles": ["character", "text"],
                    "maxLength": 240,
                }
            )
            requested.append(key)
    for section_key, (collection, names, count) in repeat_sections.items():
        section = 7 if section_key == 17 else section_key
        for name in names:
            catalog.append(
                {
                    "key": f"{collection}[].{name}",
                    "section": section,
                    "valueKind": "text",
                    "extractionRoles": ["character", "text"],
                    "maxLength": 240,
                }
            )
        for index in range(count):
            record_id = f"00000000-0000-0000-0000-{section:02d}{index:010d}"
            requested.extend(f"{collection}[{record_id}].{name}" for name in names)

    value = request(tmp_path, "text")
    value.pop("sourceImage")
    value["sourceText"] = "A person has short black hair. They wear a dark blue coat."
    value["fields"] = catalog
    value["requestedFields"] = requested
    calls = []

    def fake_assist(payload, **_):
        text_request = payload["textRequest"]
        calls.append(text_request)
        assert (
            len(text_request["systemPrompt"].encode()) + len(text_request["prompt"].encode())
            <= service.MAX_ASSIST_INPUT_BYTES
        )
        prompt = json.loads(text_request["prompt"])
        if "collections" in prompt:
            return {
                "text": '{"records":[{"collection":"garments","slot":1,'
                '"excerpt":"dark blue coat"}]}',
                "truncated": False,
            }
        return {"text": '{"proposals":[]}', "truncated": False}

    monkeypatch.setattr(service, "assist", fake_assist)
    result = service.extract_fields(value)
    detail_calls = [call for call in calls if "allowedFields" in json.loads(call["prompt"])]
    sent = [
        field_id
        for call in detail_calls
        for field_id in json.loads(call["prompt"])["allowedFields"]
    ]
    assert len(calls) > 1
    assert all("00000000-" not in path for path in sent)
    assert "garment1.type" in sent
    assert {"garment2.type", "garment3.type"}.isdisjoint(sent)
    assert all(
        len(json.loads(call["prompt"])["allowedFields"]) <= service.MAX_FIELDS_PER_BATCH
        for call in detail_calls
    )
    assert all(requested[0] not in call["systemPrompt"] for call in detail_calls)
    rules = {
        path: rule
        for call in detail_calls
        for path, rule in json.loads(call["prompt"])["fieldRules"].items()
    }
    assert {rule["recordSlot"] for path, rule in rules.items() if path.startswith("garment")} == {
        "garments#1"
    }
    for slot in ("garment1",):
        containing = [
            call
            for call in detail_calls
            if any(
                path.startswith(slot + ".") for path in json.loads(call["prompt"])["allowedFields"]
            )
        ]
        assert len(containing) == 1
        assert (
            sum(
                path.startswith(slot + ".")
                for path in json.loads(containing[0]["prompt"])["allowedFields"]
            )
            == 11
        )
    assert result["metadata"]["promptVersion"] == 12


def test_repeat_aliases_map_back_to_exact_host_uuid_paths(tmp_path, monkeypatch):
    import studio_character_assist as service

    first = "11111111-1111-1111-1111-111111111111"
    second = "22222222-2222-2222-2222-222222222222"
    value = request(tmp_path, "text")
    value.pop("sourceImage")
    value["sourceText"] = "A blue jacket and gray trousers."
    value["fields"] = [
        {"key": "garments[].type", "section": 6, "valueKind": "text", "maxLength": 240},
        {"key": "garments[].color", "section": 6, "valueKind": "color", "maxLength": 240},
    ]
    value["requestedFields"] = [
        f"garments[{first}].type",
        f"garments[{first}].color",
        f"garments[{second}].type",
        f"garments[{second}].color",
    ]

    def fake_assist(payload, **_):
        prompt = json.loads(payload["textRequest"]["prompt"])
        if "collections" in prompt:
            return {
                "text": json.dumps(
                    {
                        "records": [
                            {"collection": "garments", "slot": 1, "excerpt": "blue jacket"},
                            {
                                "collection": "garments",
                                "slot": 2,
                                "excerpt": "gray trousers",
                            },
                        ]
                    }
                ),
                "truncated": False,
            }
        allowed = prompt["allowedFields"]
        slot = "garment1" if "garment1.type" in allowed else "garment2"
        scope = "garments#1" if slot == "garment1" else "garments#2"
        assert "earlier or later item" in prompt["recordScope"][scope]
        assert prompt["sourceText"] == ("blue jacket" if slot == "garment1" else "gray trousers")
        return {
            "text": json.dumps(
                {
                    "proposals": [
                        {
                            "field": f"{slot}.type",
                            "value": "jacket" if slot == "garment1" else "trousers",
                            "evidence": prompt["sourceText"],
                            "uncertainty": "",
                        },
                    ]
                }
            ),
            "truncated": False,
        }

    monkeypatch.setattr(service, "assist", fake_assist)
    result = service.extract_fields(value)
    assert [item["field"] for item in result["proposals"]] == [
        f"garments[{first}].type",
        f"garments[{second}].type",
    ]


def test_measurement_batch_declares_type_and_repair_receives_validation_error(
    tmp_path, monkeypatch
):
    import studio_character_assist as service

    value = request(tmp_path, "text")
    value.pop("sourceImage")
    value["sourceText"] = "The character is 170 cm tall."
    value["fields"] = [
        {
            "key": "body.height",
            "section": 3,
            "valueKind": "measurement",
            "suggestions": [],
            "extractionRoles": ["character", "text"],
            "maxLength": 240,
        }
    ]
    value["requestedFields"] = ["body.height"]
    prompts = []
    systems = []

    def fake_assist(payload, **_):
        prompt = json.loads(payload["textRequest"]["prompt"])
        prompts.append(prompt)
        systems.append(payload["textRequest"]["systemPrompt"])
        if len(prompts) == 1:
            return completion("body.height", "tall")
        return completion("body.height", "170 cm")

    monkeypatch.setattr(service, "assist", fake_assist)
    result = service.extract_fields(value)
    assert prompts[0]["fieldRules"]["body.height"] == {
        "kind": "measurement",
        "format": "positive number + mm|cm|m|in|ft",
        "omitUnless": "source explicitly states both number and unit",
    }
    assert "number and unit" in prompts[1]["validationError"]
    assert "Remove any proposal" in systems[1]
    assert result["proposals"][0]["value"] == "170 cm"


def test_invalid_value_after_repair_is_omitted_with_diagnostic(tmp_path, monkeypatch):
    import studio_character_assist as service

    value = request(tmp_path, "text")
    value.pop("sourceImage")
    value["sourceText"] = "The character is tall and has short hair."
    value["fields"] = [
        {"key": "body.height", "section": 3, "valueKind": "measurement", "maxLength": 240},
        {"key": "hair.length", "section": 5, "valueKind": "text", "maxLength": 240},
    ]
    value["requestedFields"] = ["body.height", "hair.length"]
    bad = {
        "text": json.dumps(
            {
                "proposals": [
                    {
                        "field": "body.height",
                        "value": "tall",
                        "evidence": "word tall",
                        "uncertainty": "",
                    },
                    {
                        "field": "hair.length",
                        "value": "short",
                        "evidence": "explicit",
                        "uncertainty": "",
                    },
                ]
            }
        ),
        "truncated": False,
    }
    monkeypatch.setattr(service, "assist", lambda *_args, **_kwargs: bad)
    result = service.extract_fields(value)
    assert [item["field"] for item in result["proposals"]] == ["hair.length"]
    assert result["diagnostics"] == [
        {
            "code": "proposal.invalid",
            "field": "body.height",
            "message": "body.height: Proposal measurement value must include a number and unit",
            "index": 0,
        }
    ]


def test_unknown_placeholders_are_omitted_with_field_diagnostic(tmp_path, monkeypatch):
    import studio_character_assist as service

    value = request(tmp_path, "text")
    value.pop("sourceImage")
    value["sourceText"] = "An adult character."
    value["fields"] = [{"key": "body.arms", "section": 3, "valueKind": "text", "maxLength": 240}]
    value["requestedFields"] = ["body.arms"]
    answer = completion("body.arms", "unknown")
    monkeypatch.setattr(service, "assist", lambda *_args, **_kwargs: answer)
    result = service.extract_fields(value)
    assert result["proposals"] == []
    assert result["diagnostics"][0]["field"] == "body.arms"
    assert "omitted" in result["diagnostics"][0]["message"]


def test_default_model_fingerprint_is_stable_string_and_changes_with_stat(tmp_path):
    import studio_character_assist as service

    model = tmp_path / "model.ckpt"
    model.write_bytes(b"one")
    value = {"modelPath": str(model)}
    first = service._model_fingerprint(value, model)
    second = service._model_fingerprint(value, model)
    assert first == second
    assert isinstance(first, str) and len(first) == 64
    model.write_bytes(b"changed-size")
    assert service._model_fingerprint(value, model) != first


def test_cache_validates_more_than_sixty_four_proposals_by_original_batches(tmp_path, monkeypatch):
    import studio_character_assist as service

    value = request(tmp_path, "text")
    value.pop("sourceImage")
    value["sourceText"] = "Explicit catalog facts."
    value["fields"] = [
        {"key": f"body.fact{index}", "section": 3, "valueKind": "text", "maxLength": 240}
        for index in range(70)
    ]
    value["requestedFields"] = [item["key"] for item in value["fields"]]
    calls = 0

    def fake_assist(payload, **_):
        nonlocal calls
        calls += 1
        allowed = json.loads(payload["textRequest"]["prompt"])["allowedFields"]
        return {
            "text": json.dumps(
                {
                    "proposals": [
                        {"field": path, "value": "explicit", "evidence": "text", "uncertainty": ""}
                        for path in allowed
                    ]
                }
            ),
            "truncated": False,
        }

    monkeypatch.setattr(service, "assist", fake_assist)
    assert len(service.extract_fields(value)["proposals"]) == 70
    first_calls = calls
    assert len(service.extract_fields(value)["proposals"]) == 70
    assert calls == first_calls


def test_clothing_detail_only_reaches_garment_accessory_and_surface_calls(tmp_path, monkeypatch):
    import studio_character_assist as service

    value = request(tmp_path)
    details = []
    for name, label in [
        ("head", "primary head and face detail"),
        ("clothing", "primary torso and lap detail"),
    ]:
        image = tmp_path / f"{name}.jpg"
        image.write_bytes(name.encode())
        details.append(
            {
                "path": str(image),
                "label": label,
                "sha256": hashlib.sha256(name.encode()).hexdigest(),
            }
        )
    value["sourceDetailImages"] = details
    value["fields"] = [
        field("hair.length", 5),
        field("garments[].type", 6),
        field("accessories[].type", 7),
        field("features[].type", 7),
        field("surfaces[].material", 8),
    ]
    value["requestedFields"] = [
        "hair.length",
        "garments[shirt].type",
        "accessories[watch].type",
        "features[scar].type",
        "surfaces[denim].material",
    ]
    calls = []

    def fake_assist(payload, **_):
        calls.append(payload["textRequest"])
        return {"text": '{"proposals":[]}', "truncated": False}

    monkeypatch.setattr(service, "assist", fake_assist)
    service.extract_fields(value)
    for call in calls:
        fields = json.loads(call["prompt"])["allowedFields"]
        labels = [item["label"] for item in call["images"]]
        if any(key.startswith(("garment", "accessory", "surface")) for key in fields):
            assert labels[-1] == "primary torso and lap detail"
            assert "primary head and face detail" not in labels
        elif "hair.length" in fields:
            assert labels[-1] == "primary head and face detail"
        else:
            assert len(labels) == 1


@pytest.mark.parametrize("role,expected", [("character", []), ("text", ["worn"])])
def test_image_condition_cannot_generalize_fading_into_wear_but_authored_wear_survives(
    tmp_path, monkeypatch, role, expected
):
    import studio_character_assist as service

    value = request(tmp_path, role)
    if role == "text":
        value["sourceText"] = "A worn shirt."
    value["fields"] = [field("garments[].condition", 6)]
    value["requestedFields"] = ["garments[shirt].condition"]
    monkeypatch.setattr(
        service,
        "assist",
        lambda *_args, **_kwargs: {
            "text": json.dumps(
                {
                    "proposals": [
                        {
                            "field": "garment1.condition",
                            "value": "worn",
                            "evidence": "The shirt is slightly faded and creased.",
                            "uncertainty": "",
                        }
                    ]
                }
            ),
            "truncated": False,
        },
    )
    result = service.extract_fields(value)
    assert [item["value"] for item in result["proposals"]] == expected
    if role == "character":
        assert any("literal" in item["message"] for item in result["diagnostics"])


def test_accepted_garment_identity_is_carried_to_material_call(tmp_path, monkeypatch):
    import studio_character_assist as service

    value = request(tmp_path)
    value["fields"] = [field("garments[].type", 6), field("surfaces[].material", 8)]
    value["requestedFields"] = ["garments[jeans].type", "surfaces[denim].material"]
    calls = []

    def fake_assist(payload, **_):
        prompt = json.loads(payload["textRequest"]["prompt"])
        calls.append(prompt)
        return {
            "text": json.dumps(
                {
                    "proposals": (
                        [
                            {
                                "field": "garment1.type",
                                "value": "jeans",
                                "evidence": "Blue trousers on the primary person's lap",
                                "uncertainty": "",
                            }
                        ]
                        if "garment1.type" in prompt["allowedFields"]
                        else []
                    )
                }
            ),
            "truncated": False,
        }

    monkeypatch.setattr(service, "assist", fake_assist)
    service.extract_fields(value)
    assert calls[1]["observedWardrobe"] == {"garment1.type": "jeans"}


def test_visual_inventory_binds_records_before_details_and_omits_unused_slots(
    tmp_path, monkeypatch
):
    import studio_character_assist as service

    value = request(tmp_path)
    value["fields"] = [field("garments[].type", 6), field("surfaces[].material", 8)]
    value["requestedFields"] = [
        "garments[shirt].type",
        "garments[jeans].type",
        "garments[unused].type",
        "surfaces[shirt].material",
        "surfaces[jeans].material",
    ]
    calls = []

    def fake_assist(payload, **_):
        text = payload["textRequest"]
        prompt = json.loads(text["prompt"])
        calls.append(prompt)
        if "collections" in prompt:
            assert len(text["images"]) == 1
            return {
                "text": json.dumps(
                    {
                        "records": [
                            {
                                "collection": "garments",
                                "slot": 1,
                                "excerpt": "brown crew-neck T-shirt on torso",
                            },
                            {
                                "collection": "garments",
                                "slot": 2,
                                "excerpt": "blue jeans on primary person's lap",
                            },
                            {
                                "collection": "surfaces",
                                "slot": 1,
                                "excerpt": "brown crew-neck T-shirt on torso",
                            },
                            {
                                "collection": "surfaces",
                                "slot": 2,
                                "excerpt": "blue jeans on primary person's lap",
                            },
                        ]
                    }
                ),
                "truncated": False,
            }
        return {"text": '{"proposals":[]}', "truncated": False}

    monkeypatch.setattr(service, "assist", fake_assist)
    service.extract_fields(value)
    assert "collections" in calls[0], (
        "Identify distinct visible items before independent record calls."
    )
    assert all("garment3.type" not in call.get("allowedFields", []) for call in calls[1:])
    jeans = next(call for call in calls[1:] if "surface2.material" in call["allowedFields"])
    assert (
        "blue jeans on primary person's lap"
        in jeans["fieldRules"]["surface2.material"]["recordAnchor"]
    )


def test_long_prior_wardrobe_context_keeps_repair_within_input_budget(tmp_path):
    import studio_character_assist as service

    value = request(tmp_path)
    names = [
        "type",
        "bodyRegion",
        "layer",
        "fit",
        "cut",
        "color",
        "closures",
        "seams",
        "trim",
        "condition",
        "placement",
    ]
    value["fields"] = [field(f"garments[].{name}", 6) for name in names]
    value["requestedFields"] = [f"garments[shirt].{name}" for name in names]
    value["_observedWardrobe"] = {f"garment{index}.type": "á" * 100 for index in range(12)}
    payload = service._payload(
        value,
        "character",
        Path(value["modelPath"]),
        value["requestedFields"],
        repair_text="x" * 2000,
        repair_error="x" * 400,
    )
    text_request = payload["textRequest"]
    assert len(text_request["systemPrompt"].encode()) + len(text_request["prompt"].encode()) <= 8000
    assert json.loads(text_request["prompt"])["observedWardrobe"]


@pytest.mark.parametrize(
    "candidate,evidence,accepted",
    [
        ("slightly faded and creased", "The shirt is slightly faded and creased.", True),
        ("worn", {"summary": "The shirt is slightly faded and creased."}, False),
        ("torn", "The shirt is not torn or stained.", False),
        ("stained", "The shirt is not torn or stained.", False),
        ("worn", "The shirt is unworn.", False),
        ("torn", {"summary": "The shirt has a torn sleeve."}, True),
        ("faded", "The shirt has no holes but is faded.", True),
    ],
)
def test_literal_visible_condition_is_retained(
    tmp_path, monkeypatch, candidate, evidence, accepted
):
    import studio_character_assist as service

    value = request(tmp_path)
    value["fields"] = [field("garments[].condition", 6)]
    value["requestedFields"] = ["garments[shirt].condition"]
    monkeypatch.setattr(
        service,
        "assist",
        lambda *_args, **_kwargs: {
            "text": json.dumps(
                {
                    "proposals": [
                        {
                            "field": "garment1.condition",
                            "value": candidate,
                            "evidence": evidence,
                            "uncertainty": "",
                        }
                    ]
                }
            ),
            "truncated": False,
        },
    )
    result = service.extract_fields(value)
    assert [item["value"] for item in result["proposals"]] == ([candidate] if accepted else [])
    if not accepted:
        assert any(item["code"] == "proposal.invalid" for item in result["diagnostics"])


def test_visual_inventory_rejects_truncated_partial_records(tmp_path):
    import studio_character_assist as service

    response = {
        "text": '{"records":[{"collection":"garments","slot":1,"excerpt":"shirt"}]}',
        "truncated": True,
    }
    with pytest.raises(ValueError, match="truncated"):
        service._parse_inventory(response, request(tmp_path), {"garments": 3})


def test_visual_surface_slots_reuse_observed_garment_targets_when_inventory_omits_them(tmp_path):
    import studio_character_assist as service

    response = {
        "text": json.dumps(
            {
                "records": [
                    {"collection": "garments", "slot": 1, "excerpt": "brown T-shirt on torso"},
                    {"collection": "garments", "slot": 2, "excerpt": "blue jeans on lap"},
                ]
            }
        ),
        "truncated": False,
    }
    inventory = service._parse_inventory(
        response, request(tmp_path), {"garments": 3, "surfaces": 3}
    )
    assert inventory["surfaces#1"] == "brown T-shirt on torso"
    assert inventory["surfaces#2"] == "blue jeans on lap"
    assert "surfaces#3" not in inventory
