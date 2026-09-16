"""Payload-free LoRA catalog inspection and Studio's clip model boundary.

Training provenance filters the library, never replaces the renderer's complete
projection, shape, scaling, and specialized-pipeline checks.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from wee_todd_mlx.adapter_contract import inspect_adapter
from wee_todd_mlx.studio_h3_turbo import h3_lora_metadata


def inspect_lora(source, model_hint=None):
    if model_hint is not None and (
        not isinstance(model_hint, str) or model_hint not in {"h3", "ltx23", "ltx25"}
    ):
        raise ValueError("Choose a supported trained model for this LoRA.")
    source = Path(source).expanduser().resolve()
    if source.suffix.lower() != ".safetensors":
        raise ValueError("Choose a SafeTensors LoRA file.")
    report = inspect_adapter(source)
    metadata = report["metadata"]
    models = set()
    for key in (
        "model_version",
        "base_model",
        "base_model_name_or_path",
        "ss_base_model_version",
        "modelspec.architecture",
        "converted_layout",
    ):
        value = str(metadata.get(key, "")).lower()
        if "minimax" in value and "h3" in value:
            models.add("h3")
        if key == "model_version" or "ltx" in value:
            version = re.search(r"(?:^|[^0-9])(2)[._-]([0-9]+)(?:[^0-9]|$)", value)
            if version:
                minor = version.group(2)
                if minor not in {"3", "5"}:
                    raise ValueError(f"Unsupported LoRA training version: {value}.")
                models.add("ltx2" + minor)
    if len(models) > 1:
        raise ValueError("LoRA has conflicting training-model metadata.")
    model = next(iter(models), model_hint)
    h3_fields = h3_lora_metadata(report) if model == "h3" else {}
    if h3_fields.get("loraProfile") == "turbo":
        for field in (
            "inference_steps",
            "num_inference_steps",
            "steps",
            "transformer_evaluations",
            "schedule_points",
        ):
            expected = 5 if field == "schedule_points" else 4
            if field in metadata and int(metadata[field]) != expected:
                raise ValueError(
                    "This H3 Turbo adapter declares a different sampling count. "
                    "Use a dedicated model/task recipe; Studio's Turbo library flow "
                    "supports 4 evaluations (5 schedule points)."
                )
    role = metadata.get("adapter_role", "standard").strip().lower()
    specialized_profile = model != "h3" and any(
        metadata.get(key, "").strip().lower() not in {"", "standard", "base", "quality"}
        for key in ("adapter_profile", "profile", "distillation_profile")
    )
    if (
        specialized_profile
        or any(key.startswith("reference_") for key in metadata)
        or role
        not in {
            "standard",
            "transformer_lora",
            "style",
            "character",
            *(("turbo",) if model == "h3" else ()),
        }
    ):
        raise ValueError(
            "This specialized adapter belongs in a model/task recipe, not a style LoRA group."
        )
    return {"kind": "lora", "path": str(source), "loraModel": model, **h3_fields}


def clip_lora(asset, attachment, engine):
    if asset.get("kind") != "lora":
        raise ValueError("A LoRA attachment must reference a SafeTensors adapter.")
    model = asset.get("loraModel")
    if model not in {"h3", "ltx23", "ltx25"} or not (
        model == engine or (model == "ltx23" and engine == "ltx25")
    ):
        raise ValueError("Choose a compatible trained model for this LoRA in the LoRA library.")
    strength = attachment.get("strength", 1)
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(strength)
        or not 0 <= strength <= 2
    ):
        raise ValueError("LoRA strength must be a finite number from 0 to 2.")
    inspected = inspect_lora(asset["path"], model_hint=model)
    if inspected["loraModel"] and inspected["loraModel"] != model:
        raise ValueError(
            "The LoRA file declares a different trained model. Reimport it in the LoRA library."
        )
    result = {"path": inspected["path"], "strength": strength}
    if engine == "h3":
        # Explicit choices allow metadata-poor adapters, but cannot contradict declarations.
        for field, native, allowed in (
            ("loraProfile", "profile", {"standard", "turbo"}),
            ("loraLayout", "qkv_layout", {"auto", "native_interleaved", "contiguous_qkv"}),
        ):
            selected = asset.get(field)
            if selected is not None and (not isinstance(selected, str) or selected not in allowed):
                raise ValueError(f"Invalid H3 LoRA {field} selection.")
            declared = inspected.get(field)
            if selected not in (None, "auto") and declared and selected != declared:
                raise ValueError(f"The LoRA file declares a different {field}. Reimport it.")
            resolved = declared if selected in (None, "auto") else selected
            if resolved is not None:
                result[native] = resolved
        grid = asset.get("loraAdalnInputGrid")
        if grid is not None:
            if not isinstance(grid, str) or not grid.strip():
                raise ValueError("H3 LoRA AdaLN input grid must be a nonempty path.")
            result["adaln_input_grid"] = str(Path(grid).expanduser().resolve())
    elif any(
        asset.get(field) is not None
        for field in ("loraProfile", "loraLayout", "loraAdalnInputGrid")
    ):
        raise ValueError("H3 LoRA profile, layout, and AdaLN grid require the H3 engine.")
    return result
