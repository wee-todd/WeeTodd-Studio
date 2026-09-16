"""Studio Turbo validation uses sparse fixtures and reads only SafeTensors headers."""

import copy
import json
import math
import struct

import pytest

from wee_todd_mlx import studio_h3_turbo as turbo


def tensors_file(tmp_path, tensors, metadata=None, name="adapter.safetensors"):
    header = {"__metadata__": metadata or {}}
    offset = 0
    for key, shape in tensors.items():
        stop = offset + math.prod(shape) * 4
        header[key] = dict(dtype="F32", shape=shape, data_offsets=[offset, stop])
        offset = stop
    encoded = json.dumps(header).encode()
    source = tmp_path / name
    with source.open("wb") as file:
        file.write(struct.pack("<Q", len(encoded)) + encoded)
        file.truncate(8 + len(encoded) + offset)
    return source


def adapter(tmp_path, *, metadata=None, adaln=False, name="adapter.safetensors", target=None):
    targets = {target or "diffusion_model.blocks.0.attn.qkv_proj": (21504, 5376)}
    if adaln:
        targets["blocks.0.adaln_proj.linear"] = (96768, 2688)
    tensors = {}
    for key, (out_width, in_width) in targets.items():
        tensors[key + ".lora_A.weight"] = [1, in_width]
        tensors[key + ".lora_B.weight"] = [out_width, 1]
    return tensors_file(tmp_path, tensors, metadata, name)


def ordinary_recipe():
    return {
        "engine": "h3",
        "components": {"task": "fl2va"},
        "config": {"steps": 16, "sampling_method": "euler", "seed": 41},
    }


def attached(tmp_path, **kwargs):
    return {
        "path": str(adapter(tmp_path, **kwargs)),
        "profile": "turbo",
        "strength": 1.0,
        "qkv_layout": "contiguous_qkv",
    }


@pytest.mark.parametrize("task", ["t2v", "i2v", "fflf"])
def test_turbo_resolves_four_evaluations_without_mutating_or_appending(tmp_path, task):
    recipe = ordinary_recipe()
    adapters = [attached(tmp_path)]
    before = copy.deepcopy((recipe, adapters))
    result, warnings = turbo.resolve_studio_h3_turbo(recipe, adapters, task)
    assert result["config"]["steps"] == 5
    assert result["config"]["sampling_method"] == "euler"
    assert result["config"]["seed"] == 41
    assert "loras" not in result
    assert any("4" in warning and "evaluations" in warning for warning in warnings)
    assert (recipe, adapters) == before


def test_disabled_turbo_preserves_standard_steps_and_does_not_require_file(tmp_path):
    entry = {"path": str(tmp_path / "offline.safetensors"), "profile": "turbo", "enabled": False}
    recipe = ordinary_recipe()
    result, warnings = turbo.resolve_studio_h3_turbo(recipe, [entry], "t2v")
    assert result == recipe and result is not recipe
    result["config"]["steps"] = 99
    assert recipe["config"]["steps"] == 16
    assert warnings == []


def test_existing_embedded_custom_turbo_schedule_is_preserved_without_attached_turbo():
    recipe = ordinary_recipe()
    recipe["config"]["steps"] = 9
    recipe["loras"] = {"adapters": [{"path": "offline.safetensors", "profile": "turbo"}]}
    result, warnings = turbo.resolve_studio_h3_turbo(recipe, [], "t2v")
    assert result == recipe
    assert warnings == []


@pytest.mark.parametrize("task", ["ref2va", "a2v", "control", "extension"])
def test_unsupported_turbo_tasks_fail_before_generation(tmp_path, task):
    with pytest.raises(ValueError, match="task"):
        turbo.resolve_studio_h3_turbo(ordinary_recipe(), [attached(tmp_path)], task)


@pytest.mark.parametrize(
    "patch",
    [
        {"engine": "ltx25"},
        {"components": {"task": "ref2va"}},
        {"vdn": {"path": "vdn.safetensors"}},
        {"fastvideo": {"profile": "dense"}},
        {"attention": {"mode": "vsa"}},
        {"config": {"steps": 16, "sampling_method": "heun"}},
        {"config": {"steps": 16, "custom_sigmas": [1, 0]}},
        {"config": {"steps": 16, "turbo": True}},
        {"config": {"steps": 16, "pipeline_mode": "distilled"}},
        {"loras": {"adapters": [{"path": "embedded.safetensors", "profile": "turbo"}]}},
    ],
)
def test_conflicting_recipe_sampling_is_rejected(tmp_path, patch):
    recipe = ordinary_recipe()
    recipe.update(patch)
    with pytest.raises(ValueError, match="Turbo"):
        turbo.resolve_studio_h3_turbo(recipe, [attached(tmp_path)], "t2v")


def test_declared_checkpoint_acceleration_blocks_attached_turbo(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "model_identity.json").write_text(
        json.dumps({"source": "FastVideo/FastH3-Dense", "sampling": {"transformer_evaluations": 4}})
    )
    recipe = ordinary_recipe()
    recipe["components"]["transformer"] = str(root)
    with pytest.raises(ValueError, match="Turbo"):
        turbo.resolve_studio_h3_turbo(recipe, [attached(tmp_path)], "t2v")


def test_multiple_active_turbo_adapters_and_zero_strength_are_rejected(tmp_path):
    first = attached(tmp_path)
    second = attached(tmp_path, name="second.safetensors")
    with pytest.raises(ValueError, match="one"):
        turbo.resolve_studio_h3_turbo(ordinary_recipe(), [first, second], "t2v")
    first["strength"] = 0
    with pytest.raises(ValueError, match="strength"):
        turbo.resolve_studio_h3_turbo(ordinary_recipe(), [first], "t2v")


@pytest.mark.parametrize(
    "metadata",
    [
        {"inference_steps": "8"},
        {"num_inference_steps": "3"},
        {"steps": "8"},
        {"schedule_points": "9"},
        {"transformer_evaluations": "6"},
        {"inference_steps": "4.0"},
        {"adapter_profile": "distilled"},
        {"adapter_role": "control"},
        {"partial_conversion": "true"},
    ],
)
def test_incompatible_declared_adapter_schedule_or_role_is_rejected(tmp_path, metadata):
    with pytest.raises(ValueError, match="(Turbo|metadata|recipe)"):
        turbo.resolve_studio_h3_turbo(
            ordinary_recipe(), [attached(tmp_path, metadata=metadata)], "t2v"
        )


def test_declared_turbo_is_resolved_when_profile_is_auto(tmp_path):
    entry = attached(tmp_path, metadata={"adapter_profile": "turbo", "inference_steps": "4"})
    entry["profile"] = "auto"
    result, _ = turbo.resolve_studio_h3_turbo(ordinary_recipe(), [entry], "t2v")
    assert result["config"]["steps"] == 5


def test_turbo_requires_compatible_auxiliary_grid_and_keeps_its_request(tmp_path):
    entry = attached(tmp_path, adaln=True)
    with pytest.raises(ValueError, match="AdaLN input grid"):
        turbo.resolve_studio_h3_turbo(ordinary_recipe(), [entry], "fflf")
    wrong = tensors_file(tmp_path, {"silu_t_emb_grid": [2, 8]}, name="wrong.safetensors")
    entry["adaln_input_grid"] = str(wrong)
    with pytest.raises(ValueError, match="AdaLN input grid"):
        turbo.resolve_studio_h3_turbo(ordinary_recipe(), [entry], "fflf")
    good = tensors_file(tmp_path, {"silu_t_emb_grid": [1025, 2688]}, name="grid.safetensors")
    entry["adaln_input_grid"] = str(good)
    result, warnings = turbo.resolve_studio_h3_turbo(ordinary_recipe(), [entry], "fflf")
    assert result["config"]["steps"] == 5
    assert entry["adaln_input_grid"] == str(good)
    assert warnings


def test_unsupported_native_targets_fail_without_tensor_loading(tmp_path):
    entry = attached(tmp_path, target="foreign_model.projection")
    with pytest.raises(ValueError, match="target"):
        turbo.resolve_studio_h3_turbo(ordinary_recipe(), [entry], "t2v")


def test_native_target_shape_mismatch_is_rejected(tmp_path):
    source = tensors_file(
        tmp_path,
        {
            "blocks.0.attn.qkv_proj.lora_A.weight": [1, 3],
            "blocks.0.attn.qkv_proj.lora_B.weight": [6, 1],
        },
    )
    with pytest.raises(ValueError, match="shape"):
        turbo.resolve_studio_h3_turbo(
            ordinary_recipe(), [{"path": str(source), "profile": "turbo"}], "t2v"
        )


@pytest.mark.parametrize(
    "fields",
    [
        {"profile": True},
        {"qkv_layout": True},
        {"qkv_layout": "guess"},
        {"strength": True},
        {"strength": float("nan")},
        {"adaln_input_grid": 8},
        {"enabled": "no"},
        {"start_after_evaluations": 1},
    ],
)
def test_turbo_rejects_invalid_request_fields(tmp_path, fields):
    entry = attached(tmp_path)
    entry.update(fields)
    with pytest.raises(ValueError, match="(LoRA|Turbo|adapter)"):
        turbo.resolve_studio_h3_turbo(ordinary_recipe(), [entry], "t2v")
