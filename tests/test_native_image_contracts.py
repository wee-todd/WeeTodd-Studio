import copy
import hashlib

import pytest
from PIL import Image

from wee_todd_mlx.image_contracts import validate_image_request


@pytest.fixture
def request_factory(tmp_path):
    def make(count=1):
        image = tmp_path / "reference.png"
        Image.new("RGBA", (64, 32), (20, 50, 90, 128)).save(image)
        return {
            "schema": "weetodd-native-image-request-v1",
            "requestID": "test",
            "engine": "qwen_image21",
            "modelID": "Qwen/Qwen-Image-2.1",
            "modelManifestPath": str(tmp_path / "manifest.json"),
            "prompt": "Edit <image1>",
            "configuration": {
                "width": 1024,
                "height": 1024,
                "steps": 40,
                "seed": 42,
                "guidance": 1.0,
                "scheduler": "flow_euler_dynamic",
                "referenceResolution": 1024,
                "memoryMode": "automatic",
                "livePreview": True,
            },
            "inputs": [
                {
                    "id": str(i),
                    "role": "moodboard",
                    "path": str(image),
                    "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    "imageIndex": i + 1,
                }
                for i in range(count)
            ],
        }

    return make


def test_ten_images_keep_distinct_slots_and_content_identity(request_factory):
    result = validate_image_request(request_factory(10))
    assert [x["imageIndex"] for x in result["inputs"]] == list(range(1, 11))
    assert len({x["id"] for x in result["inputs"]}) == 10
    assert result["configuration"]["seed"] == 42
    assert result["inputs"][0]["processedWidth"] == 1440
    assert result["inputs"][0]["processedHeight"] == 736


def test_overflow_and_stale_content_rejected(request_factory):
    with pytest.raises(ValueError, match="10"):
        validate_image_request(request_factory(11))
    request = request_factory()
    request["inputs"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed"):
        validate_image_request(request)


@pytest.mark.parametrize(
    "field,value",
    [
        ("width", 1000),
        ("steps", 0),
        ("guidance", 2),
        ("seed", True),
        ("sampler", 17),
        ("referenceResolution", 0),
    ],
)
def test_invalid_or_unsupported_settings_rejected(request_factory, field, value):
    request = request_factory()
    request["configuration"][field] = value
    with pytest.raises(ValueError):
        validate_image_request(request)


def test_fingerprint_tracks_order_but_not_request_id(request_factory):
    request = request_factory(2)
    first = validate_image_request(request)["fingerprint"]
    request["requestID"] = "another"
    assert validate_image_request(request)["fingerprint"] == first
    request["inputs"].reverse()
    for i, item in enumerate(request["inputs"]):
        item["imageIndex"] = i + 1
    assert validate_image_request(request)["fingerprint"] != first


def test_validation_does_not_mutate_request_and_zero_images_is_valid(request_factory):
    request = request_factory(0)
    original = copy.deepcopy(request)
    assert validate_image_request(request)["inputs"] == []
    assert request == original
