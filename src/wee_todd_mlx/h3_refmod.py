"""Inspect and load external visual H3 RefMods without a ComfyUI dependency."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from safetensors import safe_open

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_VISUAL_TOKENS = 8192


def inspect_visual_refmod(path, *, expected_sha256=None) -> dict:
    """Validate a standalone normalized H3 visual latent and report its true row cost."""
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".safetensors" or not source.is_file():
        raise ValueError("Choose an existing H3 RefMod .safetensors file.")
    size = source.stat().st_size
    if not 64 < size <= MAX_FILE_BYTES:
        raise ValueError("H3 RefMod file size is outside the supported range.")
    with safe_open(str(source), framework="np", device="cpu") as handle:
        raw = (handle.metadata() or {}).get("refmod_meta")
        if not isinstance(raw, str):
            raise ValueError("The selected file has no H3 RefMod metadata.")
        try:
            meta = json.loads(raw)
        except (TypeError, ValueError) as error:
            raise ValueError("H3 RefMod metadata is invalid JSON.") from error
        if not isinstance(meta, dict):
            raise ValueError("H3 RefMod metadata must be an object.")
        kind = meta.get("kind")
        if kind == "bundle":
            raise ValueError(
                "H3 RefMod bundles are not supported yet; select a standalone visual file."
            )
        if kind not in {"image", "video"}:
            raise ValueError("Only visual H3 RefMods are supported in Studio.")
        if "latent" not in handle.keys():
            raise ValueError("H3 RefMod has no latent tensor.")
        tensor = handle.get_slice("latent")
        shape = tuple(tensor.get_shape())
        if tensor.get_dtype() not in {"F16", "F32"}:
            raise ValueError("H3 RefMod visual latent must use FP16 or FP32.")
    if (len(shape) != 5 or shape[0:2] != (1, 24)
            or min(shape[2:]) < 1 or shape[3] % 2 or shape[4] % 2
            or shape[3] < 2 or shape[4] < 2
            or (kind == "image" and shape[2] != 1)):
        raise ValueError("H3 RefMod visual latent geometry is invalid.")
    if any(meta.get(key, value) != value for key, value in
           (("latent_t", shape[2]), ("latent_h", shape[3]), ("latent_w", shape[4]))):
        raise ValueError("H3 RefMod metadata geometry disagrees with its latent tensor.")
    tokens = shape[2] * (shape[3] // 2) * (shape[4] // 2)
    if tokens > MAX_VISUAL_TOKENS:
        raise ValueError(
            f"H3 RefMod needs {tokens} tokens; Studio supports at most {MAX_VISUAL_TOKENS}."
        )
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            "H3 RefMod file changed since import; inspect or relink it before rendering."
        )
    return {
        "kind": "refmod", "path": str(source), "referenceKind": kind,
        "tokenCount": tokens, "sha256": digest,
        "name": str(meta.get("name") or source.stem), "latentShape": list(shape),
    }


def load_visual_refmod(path, *, expected_sha256=None):
    """Return existing normalized VAE rows and matching native Ref2VA geometry."""
    info = inspect_visual_refmod(path, expected_sha256=expected_sha256)
    import mlx.core as mx
    from safetensors.numpy import load_file

    from minimax_h3_mlx.packing import patchify_video_latents
    from minimax_h3_mlx.ref2va import PreparedReference

    latent = load_file(info["path"])["latent"]
    if not np.isfinite(latent).all():
        raise ValueError("H3 RefMod visual latent contains nonfinite values.")
    reference = PreparedReference(
        kind=info["referenceKind"], num_latent_frames=info["latentShape"][2],
        latent_height=info["latentShape"][3], latent_width=info["latentShape"][4],
        latent_only=True,
    )
    rows = patchify_video_latents(mx.array(latent), (1, 2, 2))
    mx.eval(rows)
    return reference, rows


def attach_visual_refmod(conditioning, selection: dict):
    """Add one saved visual reference to an already encoded text-only H3 request."""
    if conditioning.task != "ref2va" or conditioning.load_vision or conditioning.references:
        raise ValueError("A visual RefMod requires unused text-only Ref2VA conditioning.")
    reference, rows = load_visual_refmod(
        selection["path"], expected_sha256=selection.get("sha256")
    )
    return replace(conditioning, condition_video_rows=rows, references=(reference,))
