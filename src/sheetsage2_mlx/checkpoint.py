"""Bounded safetensors header inspection without allocating model tensors."""

import hashlib
import json
import math
from pathlib import Path


def _identity(path):
    value = Path(path).stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def unchanged(path, identity):
    if _identity(path) != identity:
        raise ValueError("SheetSage2 checkpoint changed during inference; run analysis again")


def checkpoint_digest(path, check=lambda: None):
    identity = _identity(path)
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while True:
            check()
            block = source.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    unchanged(path, identity)
    return digest.hexdigest(), identity


def expected_shapes():
    shapes = {
        "encoder.feature_extractor.mel_mean": (128,),
        "encoder.feature_extractor.mel_std": (128,),
        "encoder.feature_extractor.mel_scale.fb": (1025, 128),
        "encoder.feature_extractor.spectrogram.window": (2048,),
        "token_embedding.weight": (31678, 512),
        "decoder.embed_positions.weight": (5122, 512),
        "layer_weight": (25,),
    }

    def linear(name, out, inp, bias=True):
        shapes[name + ".weight"] = (out, inp)
        if bias:
            shapes[name + ".bias"] = (out,)

    def norm(name, width):
        shapes[name + ".weight"] = shapes[name + ".bias"] = (width,)

    linear("encoder_projection", 512, 1024)
    norm("decoder.layernorm_embedding", 512)
    for stage, (width, depth) in enumerate(((128, 3), (512, 4), (1024, 5))):
        prefix = f"encoder.subsampling_module.{stage}."
        if stage:
            previous = (128, 512)[stage - 1]
            norm(prefix + "resampling_layer.0", previous)
            shapes[prefix + "resampling_layer.2.weight"] = (width, previous, 2)
            shapes[prefix + "resampling_layer.2.bias"] = (width,)
        for block in range(depth):
            name = prefix + f"convnext_layers.{block}."
            shapes[name + "depthwise_block.1.weight"] = (width, 1, 7)
            shapes[name + "depthwise_block.1.bias"] = (width,)
            norm(name + "pointwise_block.0", width)
            linear(name + "pointwise_block.1", width * 4, width)
            shapes[name + "pointwise_block.3.weight"] = (1, 1, width * 4)
            shapes[name + "pointwise_block.3.bias"] = (1, 1, width * 4)
            linear(name + "pointwise_block.4", width, width * 4)
    for layer in range(24):
        p = f"encoder.layers.{layer}."
        for name in (
            "ffn1_layer_norm",
            "ffn2_layer_norm",
            "attn_layer_norm",
            "final_layer_norm",
            "conv_module.layer_norm",
            "conv_module.conv_block.4.1",
        ):
            norm(p + name, 1024)
        for ffn in ("ffn1", "ffn2"):
            linear(p + ffn + ".w_1", 4096, 1024)
            linear(p + ffn + ".w_2", 1024, 4096)
        for projection in ("query_proj", "key_proj", "value_proj", "out_proj"):
            linear(p + "attn." + projection, 1024, 1024)
        shapes[p + "conv_module.conv_block.1.weight"] = (2048, 1024, 1)
        shapes[p + "conv_module.conv_block.3.weight"] = (1024, 1, 31)
        shapes[p + "conv_module.conv_block.6.weight"] = (1024, 1024, 1)
    for layer in range(6):
        p = f"decoder.layers.{layer}."
        for name in ("self_attn", "encoder_attn"):
            for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
                linear(p + name + "." + projection, 512, 512)
            norm(p + name + "_layer_norm", 512)
        linear(p + "fc1", 2048, 512)
        linear(p + "fc2", 512, 2048)
        norm(p + "final_layer_norm", 512)
    return shapes


def inspect_checkpoint(path):
    target = Path(path).expanduser()
    if target.is_dir():
        candidates = [
            target / "sheetsage2_bf16.safetensors",
            target / "audio_encoders/sheetsage2_bf16.safetensors",
            target / "model.safetensors",
        ]
        target = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    if not target.is_file():
        raise ValueError("Choose the installed merged SheetSage2 safetensors checkpoint")
    size = target.stat().st_size
    with target.open("rb") as stream:
        prefix = stream.read(8)
        length = int.from_bytes(prefix, "little")
        if len(prefix) != 8 or not 2 <= length <= 8 * 1024 * 1024 or length + 8 > size:
            raise ValueError("Invalid or truncated SheetSage2 checkpoint header")
        try:
            header = json.loads(stream.read(length))
        except (ValueError, UnicodeError) as exc:
            raise ValueError("Invalid SheetSage2 checkpoint header") from exc
    if not isinstance(header, dict):
        raise ValueError("Invalid SheetSage2 checkpoint header")
    required = expected_shapes()
    aliases = {"output_projection.weight", "decoder.embed_tokens.weight"}
    unexpected = set(header) - set(required) - aliases - {"__metadata__"}
    if unexpected:
        raise ValueError(
            f"Unsupported SheetSage2 merged checkpoint: unexpected tensor {min(unexpected)}"
        )
    for alias in aliases.intersection(header):
        required[alias] = (31678, 512)
    for name, shape in required.items():
        value = header.get(name, {})
        if not isinstance(value, dict) or value.get("shape") != list(shape):
            raise ValueError(
                f"SheetSage2 needs a complete merged checkpoint; missing/incompatible {name}"
            )
    extent = 0
    intervals = []
    dtype_bytes = {"F32": 4, "F16": 2, "BF16": 2}
    for name, value in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(value, dict) or value.get("dtype") not in dtype_bytes:
            raise ValueError(f"Unsupported SheetSage2 checkpoint tensor: {name}")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        if (
            not isinstance(shape, list)
            or any(type(d) is not int or d < 0 for d in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(n) is not int or n < 0 for n in offsets)
            or offsets[1] - offsets[0] != math.prod(shape) * dtype_bytes[value["dtype"]]
        ):
            raise ValueError(f"Invalid SheetSage2 tensor storage: {name}")
        intervals.append(tuple(offsets))
        extent = max(extent, offsets[1])
    previous = 0
    for start, end in sorted(intervals):
        if start != previous:
            raise ValueError("SheetSage2 tensor data is overlapping or incomplete")
        previous = end
    if extent + length + 8 != size:
        raise ValueError("SheetSage2 checkpoint is truncated or has unexpected trailing data")
    return {
        "model_path": str(target.resolve()),
        "model": "SheetSage2",
        "format": "merged",
        "sampling_rate": 24000,
        "max_seconds": 300,
        "vocab_size": 31678,
        "size_bytes": size,
        "tensor_count": len(intervals),
        "dtype": header["token_embedding.weight"]["dtype"],
        "license": (header.get("__metadata__") or {}).get("license", "CC-BY-NC-4.0"),
    }
