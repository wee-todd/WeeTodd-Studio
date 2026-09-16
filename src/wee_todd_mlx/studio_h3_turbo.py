"""Header-only Studio H3 Turbo selection; native sampling remains in the renderer."""

from __future__ import annotations

import re

_STEP_FIELDS = ("inference_steps", "num_inference_steps", "steps", "transformer_evaluations")


def h3_lora_metadata(report: dict) -> dict:
    """Return only explicit declarations; ambiguous metadata is never a Turbo default."""
    metadata = report["metadata"]
    profiles = set()
    for key in ("adapter_profile", "profile", "distillation_profile"):
        declared = metadata.get(key, "").strip().lower()
        if not declared:
            continue
        if declared in {"standard", "base", "quality"}:
            profiles.add("standard")
        elif declared == "turbo":
            profiles.add("turbo")
        else:
            raise ValueError("This specialized LoRA profile belongs in a model/task recipe.")
    if metadata.get("adapter_role", "").strip().lower() == "turbo":
        profiles.add("turbo")
    counts = set()
    for key in (*_STEP_FIELDS, "schedule_points"):
        if key not in metadata:
            continue
        value = metadata[key].strip()
        if not re.fullmatch(r"[1-9][0-9]*", value):
            raise ValueError(f"Invalid H3 LoRA sampling metadata: {key}.")
        counts.add(int(value) - (key == "schedule_points"))
    if len(counts) > 1:
        raise ValueError("LoRA has conflicting sampling-count metadata.")
    if counts and 1 <= next(iter(counts)) <= 8:
        profiles.add("turbo")
    if len(profiles) > 1:
        raise ValueError("LoRA has conflicting profile metadata.")
    layouts = set()
    for key in ("qkv_layout", "qkv_fusion", "conversion"):
        declared = metadata.get(key, "").strip().lower()
        if not declared:
            continue
        # Export descriptions are useful only when they explicitly describe QKV.
        if key == "conversion" and "qkv" not in declared:
            continue
        recognized = False
        if any(
            marker in declared
            for marker in ("contiguous", "concat", "block diagonal", "block-diag")
        ):
            layouts.add("contiguous_qkv")
            recognized = True
        if any(marker in declared for marker in ("interleaved", "per-head")):
            layouts.add("native_interleaved")
            recognized = True
        if key == "qkv_layout" and not recognized and declared != "auto":
            raise ValueError("Unsupported H3 LoRA QKV layout metadata.")
    if len(layouts) > 1:
        raise ValueError("LoRA has conflicting QKV layout metadata.")
    result = {
        "loraRequiresAdalnGrid": any("adaln_proj" in pair["target"] for pair in report["pairs"])
    }
    if profiles:
        result["loraProfile"] = next(iter(profiles))
    if layouts:
        result["loraLayout"] = next(iter(layouts))
    return result


def _validate_turbo_targets(report: dict, grid: str | None) -> None:
    """Qualify the released H3 fused projection schema without importing MLX."""
    from minimax_h3_mlx.config import DiTConfig

    from .model_library import inspect_safetensors_header

    config = DiTConfig()
    shapes = {
        "attn.qkv_proj": (3 * config.inner_dim, config.hidden_size),
        "attn.out_proj": (config.hidden_size, config.inner_dim),
        "mlp.fc1": (2 * config.ffn_hidden_size, config.hidden_size),
        "mlp.fc2": (config.hidden_size, config.ffn_hidden_size),
        "adaln_proj.linear": (config.adaln_out_features, config.time_embed_dim),
    }
    needs_grid = False
    normalized = set()
    for pair in report["pairs"]:
        target = pair["target"]
        for prefix in (
            "base_model.model.model.diffusion_model.",
            "base_model.model.diffusion_model.",
            "base_model.model.transformer.",
            "base_model.model.",
            "model.diffusion_model.",
            "diffusion_model.",
            "transformer.",
            "model.",
        ):
            if target.startswith(prefix):
                target = target[len(prefix) :]
                break
        match = re.fullmatch(r"(token_refiner\.)?blocks\.([0-9]+)\.(.+)", target)
        if target == "final_layer.adaln_proj.linear":
            shape = (config.final_adaln_out_features, config.time_embed_dim)
        elif match and int(match[2]) < (
            config.token_refiner_num_layers if match[1] else config.num_layers
        ):
            shape = shapes.get(match[3])
            if match[1] and match[3] == "adaln_proj.linear":
                shape = None
        else:
            shape = None
        if shape is None or target in normalized:
            raise ValueError(f"Unsupported or duplicate Studio H3 Turbo target: {target}.")
        normalized.add(target)
        if tuple(pair["logical_shape"]) != shape:
            raise ValueError(f"H3 Turbo target shape mismatch for {target}: expected {shape}.")
        needs_grid |= "adaln_proj" in target
    if needs_grid and not grid:
        raise ValueError(
            "This H3 Turbo adapter requires an AdaLN input grid for native generation."
        )
    if grid:
        header = inspect_safetensors_header(grid, include_tensors=True)
        tensors = list(header["tensors"].values())
        if len(tensors) != 1:
            raise ValueError("The H3 Turbo AdaLN input grid must contain exactly one tensor.")
        tensor = tensors[0]
        shape = tensor["shape"]
        if (
            len(shape) != 2
            or shape[0] < 2
            or shape[1] != config.time_embed_dim
            or tensor["dtype"] not in {"F16", "BF16", "F32", "F64"}
        ):
            raise ValueError(
                "The H3 Turbo AdaLN input grid must have shape "
                f"(at least 2, {config.time_embed_dim}) and floating-point values."
            )


def resolve_studio_h3_turbo(
    recipe: dict, attached_loras: list[dict], task: str
) -> tuple[dict, list[str]]:
    """Apply the bounded four-evaluation Turbo choice to a copy of an ordinary recipe.

    ``attached_loras`` contains native H3LoRASpec fields from Studio's clip boundary.
    Optional ``enabled`` flags are accepted here and disabled entries are skipped.
    The caller owns stack assembly: this function never appends or changes adapters.
    With no attached Turbo, even legacy custom embedded Turbo recipes pass unchanged.
    """
    import copy
    import math

    from wee_todd_nodes.lora import H3LoRAStack

    from .adapter_contract import inspect_adapter
    from .generation_selection import special_h3_sampling

    result = copy.deepcopy(recipe)
    active, reports, turbo_entries = [], [], []
    for item in attached_loras:
        if not isinstance(item, dict):
            raise ValueError("Studio LoRA adapter entries must be objects.")
        if type(item.get("enabled", True)) is not bool:
            raise ValueError("Studio LoRA adapter enabled state must be a boolean.")
        if not item.get("enabled", True):
            continue
        entry = {key: value for key, value in item.items() if key != "enabled"}
        profile = entry.get("profile", "auto")
        if not isinstance(profile, str) or profile not in {"auto", "standard", "turbo"}:
            raise ValueError("Invalid Studio H3 LoRA profile.")
        report = inspect_adapter(entry["path"])
        fields = h3_lora_metadata(report)
        declared = fields.get("loraProfile")
        if profile != "auto" and declared and profile != declared:
            raise ValueError("LoRA profile conflicts with its declared checkpoint metadata.")
        layout = entry.get("qkv_layout", "auto")
        if not isinstance(layout, str) or layout not in {
            "auto",
            "native_interleaved",
            "contiguous_qkv",
        }:
            raise ValueError("Invalid Studio H3 LoRA QKV layout.")
        if layout != "auto" and fields.get("loraLayout") not in (None, layout):
            raise ValueError("LoRA QKV layout conflicts with its declared checkpoint metadata.")
        active.append(entry)
        reports.append(report)
        if (declared if profile == "auto" else profile) == "turbo":
            turbo_entries.append(len(active) - 1)
    if not turbo_entries:
        return result, []
    if len(turbo_entries) != 1:
        raise ValueError("Studio supports only one enabled H3 Turbo adapter in a stack.")
    if (
        recipe.get("engine") != "h3"
        or task not in {"t2v", "i2v", "fflf"}
        or recipe.get("components", {}).get("task") not in {"t2va", "fl2va"}
    ):
        raise ValueError(
            "Studio H3 Turbo supports only native H3 text, image, and first/last-frame tasks."
        )
    config = recipe.get("config", {})
    if (
        special_h3_sampling(recipe)
        or recipe.get("components", {}).get("loras")
        or config.get("sampling_method", "euler") != "euler"
        or any(config.get(key) for key in ("sigmas", "custom_sigmas", "turbo", "pipeline_mode"))
    ):
        raise ValueError(
            "H3 Turbo requires an ordinary Euler recipe without another accelerated, "
            "distilled, VDN, or embedded Turbo configuration."
        )
    index = turbo_entries[0]
    entry, report = active[index], reports[index]
    strength = entry.get("strength", 1.0)
    if type(strength) not in {int, float} or not math.isfinite(strength) or not 0 < strength <= 2:
        raise ValueError("Enabled H3 Turbo strength must be greater than 0 and at most 2.")
    metadata = report["metadata"]
    if (
        metadata.get("adapter_role", "standard").strip().lower()
        not in {"standard", "transformer_lora", "style", "character", "turbo"}
        or any(key.startswith("reference_") for key in metadata)
        or metadata.get("partial_conversion", "false").lower() != "false"
        or metadata.get("format", "").lower().startswith("fastvideo")
    ):
        raise ValueError(
            "This specialized or partial H3 adapter requires its own model/task recipe."
        )
    for field in (*_STEP_FIELDS, "schedule_points"):
        expected = 5 if field == "schedule_points" else 4
        if field in metadata and int(metadata[field]) != expected:
            raise ValueError(
                "Studio H3 Turbo requires metadata compatible with "
                "4 evaluations (5 schedule points)."
            )
    if entry.get("start_after_evaluations", 0) != 0:
        raise ValueError("Studio H3 Turbo must be active for all four evaluations.")
    stack = H3LoRAStack.from_recipe({"loras": {"adapters": active}})
    stack.validate_for_steps(5)
    _validate_turbo_targets(report, stack.adapters[index].adaln_input_grid)
    result.setdefault("config", {})["steps"] = 5
    return result, [
        "H3 Turbo uses 4 transformer evaluations (5 schedule points). "
        "Saved standard Steps are preserved while Turbo is enabled."
    ]
