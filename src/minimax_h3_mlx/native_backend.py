"""Payload-free qualification and discovery for the optional native H3 core."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def validate_native_settings(
    config, task, adapters, *, block_residency="checkpoint_default", features=None
):
    if config.transformer_backend != "nnc_experimental":
        return
    validate_native_rows(config)
    valid = (
        task == "ref2va"
        and config.steps == 5
        and config.drop_adaln
        and config.sampling_method == "euler"
        and config.inference_optimization == "off"
        and config.projection_backend in {"auto", "mlx"}
        and config.paging_cache_gb == 0
        and block_residency == "checkpoint_default"
        and config.attention_chunk_size == "automatic"
        and config.attention_head_chunk_size in {"automatic", "disabled"}
        and config.ffn_row_chunk_size in {"automatic", "disabled"}
        and not any(value is not None and value is not False for value in (features or {}).values())
    )
    if len(adapters) != 1:
        valid = False
    else:
        adapter = adapters[0]
        valid = (
            valid
            and adapter.strength == 1
            and adapter.resolved_profile == "turbo"
            and adapter.resolved_qkv_layout == "contiguous_qkv"
            and adapter.start_after_evaluations == 0
            and adapter.adaln_input_grid is None
        )
    if not valid:
        raise ValueError(
            "Native NNC currently requires Ref2VA, one contiguous-QKV Turbo LoRA at strength 1, "
            "5 schedule points (4 Euler evaluations), Drop AdaLN, checkpoint-default residency, "
            "automatic chunk controls and auto/MLX pre/post projections. Cache, approximation, "
            "control, continuation and refinement experiments are not qualified."
        )


def validate_native_rows(config, *, prompt_rows=1, video_rows=0, audio_rows=0):
    from wee_todd_nodes.preflight import H3PreflightRequest, estimate_h3_token_budget

    rows = estimate_h3_token_budget(
        H3PreflightRequest(
            duration_seconds=config.duration_seconds,
            steps=config.steps,
            width=config.width,
            height=config.height,
            prompt_tokens=prompt_rows,
        ),
        condition_video_rows=video_rows,
        condition_audio_rows=audio_rows,
    )["packed_rows"]
    if rows > 40000:
        raise ValueError(
            f"Native NNC supports at most 40000 packed rows; this request needs {rows}"
        )
    return rows


def validate_native_adapter_header(tensors):
    targets = {
        "attn.qkv_proj": (21504, 5376),
        "attn.out_proj": (5376, 7168),
        "mlp.fc1": (28672, 5376),
        "mlp.fc2": (5376, 14336),
    }
    allowed = set()
    ranks = {}
    for index in range(50):
        for target, (output_size, input_size) in targets.items():
            prefix = f"diffusion_model.blocks.{index}.{target}"
            names = [prefix + suffix for suffix in (".lora_A.weight", ".lora_B.weight", ".alpha")]
            allowed.update(names)
            a, b, alpha = [tensors.get(name, {}) for name in names]
            shape = a.get("shape", [])
            rank = shape[0] if len(shape) == 2 else 0
            ranks.setdefault(target, rank)
            if (
                rank <= 0
                or rank != ranks[target]
                or shape != [rank, input_size]
                or b.get("shape") != [output_size, rank]
                or a.get("dtype") != "BF16"
                or b.get("dtype") != "BF16"
                or alpha.get("shape") not in ([], [1])
                or alpha.get("dtype") != "F32"
            ):
                raise ValueError(f"Native NNC requires uniform complete BF16 Turbo pairs: {prefix}")
    # AdaLN is computed by MLX; all other main-block deltas must be understood by the worker.
    for name in tensors:
        if not name.startswith("diffusion_model.") or name.startswith(
            "diffusion_model.transformer_blocks."
        ):
            raise ValueError(f"Native NNC requires exact ComfyUI adapter target names: {name}")
        if name.startswith("diffusion_model.blocks.") and name not in allowed:
            if ".adaln_proj." not in name:
                raise ValueError(f"Native NNC does not support this core adapter target: {name}")


def find_native_worker(*, root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    explicit = os.environ.get("WEETODD_H3_NATIVE_WORKER")
    candidates = (
        [Path(explicit).expanduser()]
        if explicit
        else [
            root.parent.parent / "MacOS/WeeToddH3Worker",
            root / "integrations/h3-native-worker/.build/release/WeeToddH3Worker",
        ]
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise FileNotFoundError(
        "Native H3 worker unavailable; install a bundled Studio build or run "
        "scripts/build_h3_worker.py in the checkout"
    )


def preflight_native_backend(config, task, checkpoint, adapters, **kwargs):
    validate_native_settings(config, task, adapters, **kwargs)
    if config.transformer_backend != "nnc_experimental":
        return None
    from wee_todd_mlx.model_library import inspect_safetensors_header

    from .comfy_h3_checkpoint import describe_comfy_h3

    describe_comfy_h3(checkpoint)  # Header/quantization markers only; no weight allocation.
    adapter = adapters[0]
    adapter.validate()
    tensors = inspect_safetensors_header(adapter.path, include_tensors=True)["tensors"]
    validate_native_adapter_header(tensors)
    binary = find_native_worker()
    capabilities = json.loads(
        subprocess.check_output([str(binary), "--capabilities"], timeout=10, text=True)
    )
    if capabilities != dict(
        protocol=1, backend="nnc_experimental", blocks=50, max_rows=40000, parent_watchdog=True
    ):
        raise ValueError("Native NNC worker has an incompatible protocol")
    return {
        "backend": "nnc_experimental",
        "protocol": 1,
        "worker": str(binary),
        "precision": "FP16 projections; FP32 residuals and attention",
        "max_rows": 40000,
        "residency": "one GPU block plus one CPU prefetch",
    }


def preflight_native_recipe(recipe):
    if recipe.get("engine") != "h3":
        return None
    from wee_todd_nodes.lora import H3LoRAStack
    from wee_todd_nodes.runtime import H3GenerationConfig

    config = H3GenerationConfig.from_recipe_fields(recipe["config"])
    if config.transformer_backend != "nnc_experimental":
        return None
    if recipe.get("conditioning", {}).get("task", "ref2va") != "ref2va":
        raise ValueError("Native NNC currently requires the Ref2VA conditioning task")
    components = recipe["components"]
    return preflight_native_backend(
        config,
        components.get("task", "t2va"),
        components.get("transformer") or str(Path(components["checkpoint"]) / "transformer"),
        H3LoRAStack.from_recipe(recipe).adapters,
        block_residency=recipe.get("block_residency", "checkpoint_default"),
        features={
            key: recipe.get(key)
            for key in (
                "easycache",
                "blockcache",
                "trajectory_forecast",
                "sol_attention",
                "fastvideo",
                "vdn",
                "attention",
                "continuation",
                "refinement",
                "fun_control",
            )
        },
    )
