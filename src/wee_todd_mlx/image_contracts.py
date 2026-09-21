"""Weight-free native image requests. No model imports or implicit backend fallback."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path

SCHEMA = "weetodd-native-image-request-v1"
MODEL = "Qwen/Qwen-Image-2.1"
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def image_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resized_dimensions(width, height, resolution):
    scale = resolution / math.sqrt(width * height)
    return max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32)


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")


def validate_image_request(value: dict) -> dict:
    from PIL import Image

    allowed = {
        "schema",
        "requestID",
        "engine",
        "modelID",
        "modelManifestPath",
        "prompt",
        "configuration",
        "inputs",
    }
    if not isinstance(value, dict) or set(value) != allowed:
        raise ValueError("Invalid native image request fields")
    request = copy.deepcopy(value)
    if (
        request["schema"] != SCHEMA
        or request["engine"] != "qwen_image21"
        or request["modelID"] != MODEL
    ):
        raise ValueError("Choose the Qwen-Image-2.1 native image model")
    if not isinstance(request["requestID"], str) or not request["requestID"]:
        raise ValueError("A request ID is required")
    if not isinstance(request["prompt"], str) or len(request["prompt"]) > 32000:
        raise ValueError("The prompt must be text of at most 32000 characters")
    manifest = request["modelManifestPath"]
    if not isinstance(manifest, str) or not Path(manifest).is_absolute():
        raise ValueError("Choose an absolute local model manifest path")
    config = request["configuration"]
    keys = {
        "width",
        "height",
        "steps",
        "seed",
        "guidance",
        "scheduler",
        "referenceResolution",
        "memoryMode",
        "livePreview",
    }
    if not isinstance(config, dict) or set(config) != keys:
        raise ValueError("Unsupported or missing Qwen image settings")
    for name in ("width", "height"):
        _integer(config[name], name, 32, 4096)
        if config[name] % 32:
            raise ValueError("Qwen image dimensions must be multiples of 32")
    if config["width"] * config["height"] > 8_388_608:
        raise ValueError("Output exceeds the supported eight-megapixel pixel budget")
    _integer(config["steps"], "steps", 1, 1000)
    _integer(config["seed"], "seed", -1, 2**32 - 1)
    _integer(config["referenceResolution"], "referenceResolution", 256, 2048)
    if config["referenceResolution"] % 32:
        raise ValueError("Reference resolution must be a multiple of 32")
    if (
        type(config["guidance"]) not in (int, float)
        or config["guidance"] != 1
        or config["scheduler"] != "flow_euler_dynamic"
        or config["memoryMode"] not in ("automatic", "lower_memory")
        or type(config["livePreview"]) is not bool
    ):
        raise ValueError("Qwen requires CFG 1, Euler dynamic shift and a supported memory mode")
    inputs = request["inputs"]
    if not isinstance(inputs, list) or len(inputs) > 10:
        raise ValueError("Qwen accepts at most 10 active images including the canvas")
    ids = set()
    for index, item in enumerate(inputs):
        if not isinstance(item, dict) or set(item) != {
            "id",
            "role",
            "imageIndex",
            "path",
            "sha256",
        }:
            raise ValueError("Invalid native image input")
        if not isinstance(item["id"], str) or not item["id"] or item["id"] in ids:
            raise ValueError("Image input IDs must be unique")
        ids.add(item["id"])
        if (
            type(item["imageIndex"]) is not int
            or item["imageIndex"] != index + 1
            or item["role"] not in ("canvas", "moodboard")
            or (item["role"] == "canvas" and index != 0)
        ):
            raise ValueError("Inputs must use ordered image indices with canvas first")
        if not isinstance(item["path"], str) or not Path(item["path"]).is_absolute():
            raise ValueError("Image input paths must be absolute")
        path = Path(item["path"])
        if not path.is_file() or not 0 < path.stat().st_size <= MAX_IMAGE_BYTES:
            raise ValueError("Relink a readable image of at most 64 MiB")
        if (
            not isinstance(item["sha256"], str)
            or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"])
            or image_digest(path) != item["sha256"]
        ):
            raise ValueError("An input image changed; check settings again")
        with Image.open(path) as image:
            width, height = image.size
            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError("Image exceeds the 40-megapixel decode limit")
            if image.getexif().get(274) in (5, 6, 7, 8):
                width, height = height, width
        # PNG EXIF inspection may consume the stream; verify a fresh parser.
        with Image.open(path) as image:
            image.verify()
        width, height = resized_dimensions(width, height, config["referenceResolution"])
        if max(width, height) > 8192:
            raise ValueError("Reference aspect ratio is too extreme; crop it before generation")
        item.update(processedWidth=width, processedHeight=height)
    identity = {k: v for k, v in request.items() if k != "requestID"}
    request["fingerprint"] = digest(identity)
    return request
