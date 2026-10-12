"""Strict, optional instrumental AR adapter; no numerical imports before execution.

The supported format is the unfused rank-64 YuE2-3B AR pair inventory. The
independent implementation evaluates W(x) + strength * B(A(x)); packed base
weights remain packed. NAR projections and normalization weights are untouched.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from .checkpoint import tensor_header
from .config import number


def adapter_shapes():
    shapes = {}
    projections = {
        "self_attn.q_proj": (2048, 2048),
        "self_attn.k_proj": (1024, 2048),
        "self_attn.v_proj": (1024, 2048),
        "self_attn.o_proj": (2048, 2048),
        "mlp.gate_proj": (6144, 2048),
        "mlp.up_proj": (6144, 2048),
        "mlp.down_proj": (2048, 6144),
    }
    for layer in range(28):
        for projection, (outputs, inputs) in projections.items():
            root = f"layers.{layer}.{projection}"
            shapes[root + ".lora_A"] = (64, inputs)
            shapes[root + ".lora_B"] = (outputs, 64)
    return shapes


def _snapshot(filename):
    stat = Path(filename).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def assert_unchanged(inspection):
    if _snapshot(inspection["path"]) != inspection["snapshot"]:
        raise ValueError("AR LoRA changed after preflight; retry with a stable adapter file")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate AR LoRA header key: {key}")
        result[key] = value
    return result


def inspect_lora(filename, config=None):
    filename = Path(filename).expanduser().resolve()
    if not filename.is_file():
        raise ValueError(f"AR LoRA must identify a local safetensors file: {filename}")
    before = _snapshot(filename)
    # Reject ambiguous JSON, non-object entries and metadata explicitly. The
    # shared parser then checks every payload extent without loading tensors.
    with filename.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Invalid AR LoRA safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= min(64 * 1024 * 1024, before[2] - 8):
            raise ValueError("Invalid AR LoRA safetensors header size")
        header = json.loads(stream.read(length), object_pairs_hook=_unique_object)
    if not isinstance(header, dict) or any(not isinstance(v, dict) for v in header.values()):
        raise ValueError("AR LoRA header entries must be objects")
    for name, entry in header.items():
        if name != "__metadata__" and not isinstance(entry.get("dtype"), str):
            raise ValueError(f"Invalid AR LoRA tensor dtype: {name}")
    metadata = header.get("__metadata__", {})
    if any(k in metadata for k in ("alpha", "lora_alpha")):
        raise ValueError("AR LoRA alpha scaling is unsupported; require unscaled B @ A")
    for key, expected in (("rank", "64"), ("lora_scale", "1.0")):
        if key in metadata and str(metadata[key]) != expected:
            raise ValueError(f"Unsupported AR LoRA metadata {key}")
    header = tensor_header(filename)
    shapes = adapter_shapes()
    if set(header) != set(shapes):
        raise ValueError(
            "AR LoRA requires the complete 392-tensor unfused AR inventory; "
            "fused/NAR adapters are unsupported"
        )
    dtypes = {entry["dtype"] for entry in header.values()}
    if len(dtypes) != 1 or not dtypes <= {"BF16", "F32"}:
        raise ValueError("AR LoRA requires uniformly BF16 or F32 tensors")
    for key, shape in shapes.items():
        if tuple(header[key]["shape"]) != shape:
            raise ValueError(f"Incompatible AR LoRA tensor shape: {key}")
    if config is not None:
        expected = dict(
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
        )
        if any(config.get(k) != v for k, v in expected.items()):
            raise ValueError("Instrumental AR LoRA requires the YuE2-3B architecture")
    result = dict(
        path=str(filename),
        snapshot=before,
        bytes=before[2],
        rank=64,
        tensors=len(header),
        dtype=next(iter(dtypes)),
    )
    assert_unchanged(result)
    return result


def fingerprint(inspection, strength, cancelled=None):
    from .artifacts import digest

    strength = number(strength, "ar_lora_strength", 0, 2)
    assert_unchanged(inspection)
    sha256 = digest(inspection["path"], cancelled)
    assert_unchanged(inspection)
    return dict(
        path=inspection["path"],
        bytes=inspection["bytes"],
        sha256=sha256,
        rank=64,
        tensors=392,
        dtype=inspection["dtype"],
        strength=strength,
        branch="ar",
        scaling="B@A",
        application="runtime-residual",
        active=strength != 0,
    )


def residual_linear(base, a, b, strength):
    """Return a dtype-preserving residual layer, with exact zero-strength bypass."""
    import mlx.core as mx
    import mlx.nn as nn

    strength = number(strength, "ar_lora_strength", 0, 2)
    if strength == 0:
        return base

    class ResidualLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.base, self.lora_a, self.lora_b = base, a, b
            self.strength = strength

        def __call__(self, x):
            original = self.base(x)
            # Accumulate the small residual in float32, then round once to the
            # base output dtype. This works with BF16 and affine quantized bases.
            delta = (x.astype(mx.float32) @ self.lora_a.astype(mx.float32).T) @ self.lora_b.astype(
                mx.float32
            ).T
            return (original.astype(mx.float32) + self.strength * delta).astype(original.dtype)

    return ResidualLinear()


def attach_lora(model, inspection, identity, cancelled=None):
    """Validate all tensors and targets before making a single model mutation."""
    import mlx.core as mx
    import mlx.nn as nn

    def cancel():
        if cancelled and cancelled():
            raise InterruptedError("AR LoRA loading cancelled")

    cancel()
    assert_unchanged(inspection)
    if identity != fingerprint(inspection, identity["strength"], cancelled):
        raise ValueError("AR LoRA content provenance mismatch before loading")
    if identity["strength"] == 0:
        return model
    tensors = mx.load(inspection["path"])
    shapes = adapter_shapes()
    if set(tensors) != set(shapes):
        raise ValueError("AR LoRA tensor inventory changed during loading")
    pending = []
    for key, shape in shapes.items():
        cancel()
        value = tensors[key]
        if tuple(value.shape) != shape or value.dtype not in (mx.bfloat16, mx.float32):
            raise ValueError(f"Incompatible AR LoRA payload: {key}")
        if not bool(mx.all(mx.isfinite(value)).item()):
            raise ValueError(f"Non-finite AR LoRA tensor: {key}")
        if not key.endswith(".lora_A"):
            continue
        parts = key.split(".")
        layer = model.model.layers[int(parts[1])]
        parent = getattr(layer, parts[2])
        base = getattr(parent, parts[3])
        if not isinstance(base, (nn.Linear, nn.QuantizedLinear)):
            raise ValueError(f"AR LoRA target is not an unadapted linear: {key}")
        b = tensors[key[:-1] + "B"]
        inputs = base.weight.shape[1]
        if isinstance(base, nn.QuantizedLinear):
            inputs *= 32 // base.bits
        if (base.weight.shape[0], inputs) != (b.shape[0], value.shape[1]):
            raise ValueError(f"AR LoRA target shape mismatch: {key}")
        pending.append((parent, parts[3], residual_linear(base, value, b, identity["strength"])))
    mx.eval(tensors)
    cancel()
    assert_unchanged(inspection)
    if identity != fingerprint(inspection, identity["strength"], cancelled):
        raise ValueError("AR LoRA content changed while loading")
    # No cancellation points or file reads after the commit begins.
    for parent, name, replacement in pending:
        setattr(parent, name, replacement)
    return model
