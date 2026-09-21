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


def test_character_uses_three_visible_only_passes_and_never_offers_authored_fields(
    tmp_path, monkeypatch
):
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
    with pytest.raises(ValueError, match="field"):
        service.extract_fields(request(tmp_path, "style"))


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
    assert seen == ["eyes.color", "garments[coat-uuid].color"]
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
        "promptVersion": 1,
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
