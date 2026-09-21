"""Strict local component manifests and bounded checkpoint IO."""

from __future__ import annotations

import json
from pathlib import Path

from wee_todd_mlx.image_contracts import MODEL, digest, image_digest

FORMAT = "weetodd-qwen-image21-v1"
SOURCE_REVISION = "b3179ad355be050328e483a9dfdd9e60cd62adfa"


def read_json(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"Missing or oversized model metadata: {path.name}")
    return json.loads(path.read_text())


def inspect_manifest(
    path: Path, *, verify_hashes=False, require_preview=True, cancel=lambda: False
) -> dict:
    path = Path(path).resolve()
    manifest = read_json(path)
    if manifest.get("format") != FORMAT or manifest.get("modelID") != MODEL:
        raise ValueError("Choose a prepared Qwen-Image-2.1 model manifest")
    components = manifest.get("components")
    if not isinstance(components, dict) or set(components) != {
        "text_encoder",
        "transformer",
        "vae",
    }:
        raise ValueError("The Qwen manifest requires encoder, transformer and VAE components")
    if manifest.get("sourceRevision") != SOURCE_REVISION:
        raise ValueError("This model revision has not been qualified by this runtime")
    if manifest.get("bits") != 8 or manifest.get("groupSize") != 64:
        raise ValueError("This runtime requires the prepared 8-bit, group-64 checkpoint")
    for name, component in components.items():
        folder = (path.parent / name).resolve()
        files = component.get("files", [])
        if not files:
            raise ValueError(f"Missing {name} checkpoint shards")
        total = 0
        for item in files:
            if cancel():
                raise InterruptedError("Model verification cancelled")
            file = (folder / item["name"]).resolve()
            if file.parent != folder or not file.is_file() or file.stat().st_size != item["bytes"]:
                raise ValueError(f"Missing or changed {name} shard: {file.name}")
            if verify_hashes and image_digest(file) != item["sha256"]:
                raise ValueError(f"Corrupt {name} shard: {file.name}")
            total += item["bytes"]
        component["weightBytes"] = total
        component["directory"] = str(folder)
        component["config"] = read_json(folder / "config.json")
    c = components["transformer"]["config"]
    if (c.get("num_layers"), c.get("in_channels"), c.get("context_in_dim")) != (32, 64, 4096):
        raise ValueError("Incompatible Qwen transformer architecture")
    v = components["vae"]["config"]
    if (v.get("in_channels"), v.get("out_channels"), v.get("z_dim")) != (4, 4, 64):
        raise ValueError("Qwen requires its four-channel RGBA VAE")
    processor = path.parent / "processor"
    if (
        not (processor / "tokenizer.json").is_file()
        or not (processor / "preprocessor_config.json").is_file()
    ):
        raise ValueError("Missing Qwen3-VL tokenizer or vision processor")
    for item in manifest.get("metadata", []):
        file = (path.parent / item["name"]).resolve()
        if (
            not file.is_relative_to(path.parent)
            or not file.is_file()
            or file.stat().st_size != item["bytes"]
            or image_digest(file) != item["sha256"]
        ):
            raise ValueError(f"Changed model metadata: {item['name']}")
    manifest["processorPath"] = str(processor)
    manifest["scheduler"] = read_json(path.parent / "scheduler" / "scheduler_config.json")
    preview = manifest.get("preview", {})
    if require_preview:
        file = (path.parent / preview.get("name", "missing-preview")).resolve()
        if (
            file.parent != path.parent
            or not file.is_file()
            or file.stat().st_size != preview.get("bytes")
            or image_digest(file) != preview.get("sha256")
        ):
            raise ValueError(
                "Live-preview calibration is missing or changed; prepare the model again"
            )
        manifest["previewPath"] = str(file)
    manifest["manifestFingerprint"] = digest(read_json(path))
    return manifest


def load_weights(component):
    import mlx.core as mx

    for shard in component["files"]:
        tensors = mx.load(str(Path(component["directory"]) / shard["name"]))
        yield from tensors.items()
        del tensors


def load_component(module, component, *, transpose_convolutions=False, cancel=lambda: False):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    # Prepared checkpoints explicitly identify quantized module paths.
    quantized = set(component.get("quantizedModules", []))
    if quantized:
        nn.quantize(module, group_size=64, bits=8, class_predicate=lambda p, m: p in quantized)
    expected = {key: value.shape for key, value in tree_flatten(module.parameters())}
    seen = set()
    for key, value in load_weights(component):
        if cancel():
            raise InterruptedError("Model loading cancelled")
        if key not in expected:
            raise ValueError(f"Unexpected checkpoint tensor: {key}")
        if transpose_convolutions and value.ndim == 4 and key.endswith("weight"):
            value = value.transpose(0, 2, 3, 1)
        if key.endswith(".gamma"):
            value = value.reshape(-1)
        if value.shape != expected[key]:
            raise ValueError(f"Wrong shape for {key}: {value.shape}, expected {expected[key]}")
        module.load_weights([(key, value)], strict=False)
        mx.eval(value)
        seen.add(key)
    missing = expected.keys() - seen
    if missing:
        raise ValueError(f"Missing checkpoint tensors: {sorted(missing)[:5]}")
    module.eval()
    return module
