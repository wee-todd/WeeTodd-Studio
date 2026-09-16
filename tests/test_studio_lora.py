"""Studio catalog provenance and renderer transport, without allocating model tensors."""

import importlib
import json
import math
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
lora = importlib.import_module("studio_lora")


def adapter(tmp_path, metadata, name="renamed.safetensors"):
    header = {"__metadata__": metadata}
    offset = 0
    for key, shape in {"block.lora_A.weight": [1, 2], "block.lora_B.weight": [2, 1]}.items():
        stop = offset + math.prod(shape) * 4
        header[key] = dict(dtype="F32", shape=shape, data_offsets=[offset, stop])
        offset = stop
    encoded = json.dumps(header).encode()
    source = tmp_path / name
    source.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))
    return source


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({"model_version": "2.3.0"}, "ltx23"),
        ({"base_model": "Lightricks/LTX-2.5"}, "ltx25"),
        ({"base_model": "MiniMax-H3"}, "h3"),
        ({}, None),
    ],
)
def test_inspection_uses_metadata_not_filename(tmp_path, metadata, expected):
    result = lora.inspect_lora(adapter(tmp_path, metadata, "H3-LTX-2.5.safetensors"))
    assert result.get("loraModel") == expected
    assert result["kind"] == "lora"


def test_import_rejects_non_adapters_and_conflicting_metadata(tmp_path):
    broken = tmp_path / "model.safetensors"
    broken.write_bytes(b"not a checkpoint")
    with pytest.raises((ValueError, OSError)):
        lora.inspect_lora(broken)
    with pytest.raises(ValueError, match="conflicting"):
        lora.inspect_lora(adapter(tmp_path, {"base_model": "MiniMax-H3", "model_version": "2.5.0"}))


@pytest.mark.parametrize(
    "engine,model,accepted",
    [
        ("h3", "h3", True),
        ("h3", "ltx23", False),
        ("ltx23", "ltx25", False),
        ("ltx25", "ltx23", True),
        ("ltx25", "ltx25", True),
        ("movie", "ltx25", False),
    ],
)
def test_model_filter_rechecked_at_recipe_boundary(tmp_path, engine, model, accepted):
    source = adapter(tmp_path, {})
    asset = dict(kind="lora", path=str(source), loraModel=model)
    if accepted:
        assert lora.clip_lora(asset, {"strength": 0.7}, engine)["strength"] == 0.7
    else:
        with pytest.raises(ValueError, match="compatible"):
            lora.clip_lora(asset, {}, engine)


def test_declaration_cannot_override_checkpoint_metadata(tmp_path):
    source = adapter(tmp_path, {"model_version": "2.5.0"})
    with pytest.raises(ValueError, match="declares"):
        lora.clip_lora(dict(kind="lora", path=str(source), loraModel="h3"), {}, "h3")


@pytest.mark.parametrize("strength", [float("nan"), float("inf"), -1, 2.1, True, "no"])
def test_reject_invalid_strengths(tmp_path, strength):
    asset = dict(kind="lora", path=str(adapter(tmp_path, {})), loraModel="ltx25")
    with pytest.raises(ValueError, match="strength"):
        lora.clip_lora(asset, {"strength": strength}, "ltx25")


def recipe_request(tmp_path, engine="ltx25", embedded=False):
    from studio_bridge import compose_recipe

    first = adapter(
        tmp_path, {"model_version": "2.3.0"} if engine != "h3" else {"base_model": "MiniMax-H3"}
    )
    second = adapter(tmp_path, {}, "second.safetensors")
    base = dict(
        format="weetodd-headless-v2", engine=engine, components={}, config={"frame_rate": 24}
    )
    if engine == "h3":
        base["components"]["task"] = "t2va"
    if embedded:
        base["components"]["loras"] = [{"path": "/tmp/profile-lora.safetensors", "strength": 0.4}]
    profile = tmp_path / "recipe.json"
    profile.write_text(json.dumps(base))
    model = "h3" if engine == "h3" else "ltx23"
    clip = dict(
        id="clip",
        name="Group shot",
        engine=engine,
        profileID=str(profile),
        prompt="A cyclist rides through a sunlit street.",
        generationWidth=768,
        generationHeight=512,
        seed=42,
        duration=5,
        attachments=[
            dict(
                id="a",
                assetID="one",
                role="lora",
                strength=0.65,
                loraGroupName="Mixed",
                loraGroupID="group",
            ),
            dict(
                id="b",
                assetID="two",
                role="lora",
                strength=1.2,
                loraGroupName="Mixed",
                loraGroupID="group",
            ),
        ],
    )
    assets = [
        dict(id="one", kind="lora", path=str(first), loraModel=model),
        dict(
            id="two",
            kind="lora",
            path=str(second),
            loraModel="ltx25" if engine == "ltx25" else model,
        ),
    ]
    request = dict(
        clipID="clip",
        project=dict(settings={}, clips=[clip], assets=assets),
        runtime=dict(
            profilesDirectory=str(tmp_path), ffmpegPath="/usr/bin/true", ffprobePath="/usr/bin/true"
        ),
    )
    return request, compose_recipe


@pytest.mark.parametrize(
    "engine,embedded", [("h3", False), ("ltx23", False), ("ltx23", True), ("ltx25", False)]
)
def test_composed_groups_use_existing_renderer_transport(tmp_path, engine, embedded):
    request, compose = recipe_request(tmp_path, engine, embedded)
    result, report = compose(request)
    stack = (
        result["components"]["loras"]
        if engine == "ltx25" or embedded
        else result["loras"]["adapters"]
    )
    strengths = (
        [item[1] for item in stack] if engine == "ltx25" else [item["strength"] for item in stack]
    )
    assert strengths == ([0.4] if embedded else []) + [0.65, 1.2]
    assert report["task"] == "t2v"
    assert result["conditioning"]["inputs"] == []
    if embedded:
        assert "loras" not in result


def test_duplicate_clip_files_rejected_before_render(tmp_path):
    request, compose = recipe_request(tmp_path)
    request["project"]["assets"][1].update(request["project"]["assets"][0], id="two")
    with pytest.raises(ValueError, match="only once"):
        compose(request)


@pytest.mark.parametrize("clip_only", [False, True])
def test_exported_jobs_embed_group_strengths_without_library_dependency(tmp_path, clip_only):
    import studio_job

    request, _ = recipe_request(tmp_path)
    request.update(generateIDs=["clip"], clipOnly=clip_only)
    target = tmp_path / "export.weetodd-job.json"
    studio_job.export_job(request, target)
    job = json.loads(target.read_text())
    assert job["scope"] == ("clip" if clip_only else "movie")
    assert [entry[1] for entry in job["recipes"]["clip"]["recipe"]["components"]["loras"]] == [
        0.65,
        1.2,
    ]
    request["project"]["clips"][0]["attachments"][0]["strength"] = 1.9
    assert job["project"]["clips"][0]["attachments"][0]["strength"] == 0.65
    assert "loraGroups" not in job


def test_h3_turbo_metadata_enters_library_and_passes_to_native_spec(tmp_path):
    source = adapter(
        tmp_path,
        {
            "base_model": "MiniMax-H3",
            "adapter_profile": "turbo",
            "adapter_role": "turbo",
            "qkv_layout": "contiguous",
            "inference_steps": "4",
        },
    )
    inspected = lora.inspect_lora(source)
    assert inspected["loraProfile"] == "turbo"
    assert inspected["loraLayout"] == "contiguous_qkv"
    assert inspected["loraRequiresAdalnGrid"] is False
    request = lora.clip_lora(inspected, {"strength": 0.8}, "h3")
    assert request == {
        "path": str(source),
        "strength": 0.8,
        "profile": "turbo",
        "qkv_layout": "contiguous_qkv",
    }


def test_explicit_turbo_can_classify_metadata_poor_h3_without_filename_inference(tmp_path):
    source = adapter(tmp_path, {"base_model": "MiniMax-H3"}, "turbo-8step.safetensors")
    inspected = lora.inspect_lora(source)
    assert inspected.get("loraProfile") is None
    inspected.update(loraProfile="turbo", loraLayout="native_interleaved")
    request = lora.clip_lora(inspected, {}, "h3")
    assert request["profile"] == "turbo"
    assert request["qkv_layout"] == "native_interleaved"


def test_conversion_metadata_recognizes_h3_and_qkv_without_inventing_turbo(tmp_path):
    source = adapter(
        tmp_path,
        {
            "converted_layout": "comfyui_minimax_h3",
            "conversion": "qkv block-diag fused (per-projection ranks); "
            "mlp.fc1 swiglu halves swapped",
        },
    )
    inspected = lora.inspect_lora(source)
    assert inspected["loraModel"] == "h3"
    assert inspected["loraLayout"] == "contiguous_qkv"
    assert inspected.get("loraProfile") is None


@pytest.mark.parametrize(
    "metadata",
    [
        {"adapter_profile": "turbo", "profile": "standard"},
        {"adapter_profile": "standard", "inference_steps": "4"},
        {"adapter_profile": "turbo", "inference_steps": "4", "steps": "8"},
        {"qkv_layout": "interleaved", "qkv_fusion": "concat A; block diagonal B"},
    ],
)
def test_reject_conflicting_h3_metadata(tmp_path, metadata):
    source = adapter(tmp_path, {"base_model": "MiniMax-H3", **metadata})
    with pytest.raises(ValueError, match="conflicting"):
        lora.inspect_lora(source)


@pytest.mark.parametrize(
    "metadata",
    [
        {"adapter_role": "reference"},
        {"adapter_role": "control"},
        {"adapter_profile": "distilled"},
        {"adapter_profile": "dmd"},
        {"reference_type": "character"},
    ],
)
def test_other_specialized_adapters_remain_recipe_owned(tmp_path, metadata):
    with pytest.raises(ValueError, match="recipe"):
        lora.inspect_lora(adapter(tmp_path, {"base_model": "MiniMax-H3", **metadata}))


@pytest.mark.parametrize(
    "fields",
    [
        {"loraProfile": True},
        {"loraProfile": "dmd"},
        {"loraLayout": 3},
        {"loraLayout": "contiguous"},
        {"loraAdalnInputGrid": 6},
        {"loraAdalnInputGrid": ""},
    ],
)
def test_clip_rejects_invalid_h3_metadata_types(tmp_path, fields):
    asset = dict(kind="lora", path=str(adapter(tmp_path, {})), loraModel="h3", **fields)
    with pytest.raises(ValueError, match="LoRA"):
        lora.clip_lora(asset, {}, "h3")


@pytest.mark.parametrize(
    "fields",
    [
        {"loraProfile": "standard"},
        {"loraLayout": "native_interleaved"},
    ],
)
def test_asset_cannot_override_declared_h3_profile_or_layout(tmp_path, fields):
    source = adapter(
        tmp_path,
        {"base_model": "MiniMax-H3", "adapter_profile": "turbo", "qkv_layout": "contiguous_qkv"},
    )
    asset = dict(kind="lora", path=str(source), loraModel="h3", **fields)
    with pytest.raises(ValueError, match="declares"):
        lora.clip_lora(asset, {}, "h3")


@pytest.mark.parametrize(
    "metadata",
    [
        {"model_version": "2.5.0", "adapter_profile": "turbo"},
        {"model_version": "2.3.0", "distillation_profile": "distilled"},
        {"profile": "control"},
    ],
)
def test_non_h3_specialized_profiles_remain_recipe_owned(tmp_path, metadata):
    with pytest.raises(ValueError, match="recipe"):
        lora.inspect_lora(adapter(tmp_path, metadata))


def test_clip_transports_h3_adaln_grid_path(tmp_path):
    source = adapter(tmp_path, {"base_model": "MiniMax-H3"})
    asset = dict(
        kind="lora",
        path=str(source),
        loraModel="h3",
        loraProfile="turbo",
        loraAdalnInputGrid=str(tmp_path / "grid.safetensors"),
    )
    result = lora.clip_lora(asset, {"strength": 0.6}, "h3")
    assert result["adaln_input_grid"] == str(tmp_path / "grid.safetensors")
    from wee_todd_nodes.lora import H3LoRASpec

    assert H3LoRASpec(**result).adaln_input_grid == str(tmp_path / "grid.safetensors")


def test_h3_model_hint_allows_profile_only_turbo_import_and_execution(tmp_path):
    source = adapter(tmp_path, {"adapter_profile": "turbo", "inference_steps": "4"})
    inspected = lora.inspect_lora(source, model_hint="h3")
    assert inspected["loraModel"] == "h3"
    assert inspected["loraProfile"] == "turbo"
    request = lora.clip_lora(dict(kind="lora", path=str(source), loraModel="h3"), {}, "h3")
    assert request["profile"] == "turbo"


def test_profile_only_turbo_import_without_model_hint_remains_specialized(tmp_path):
    source = adapter(tmp_path, {"adapter_profile": "turbo", "inference_steps": "4"})
    with pytest.raises(ValueError, match="recipe"):
        lora.inspect_lora(source)


def test_model_hint_does_not_override_declared_training_model(tmp_path):
    source = adapter(tmp_path, {"model_version": "2.5.0"})
    assert lora.inspect_lora(source, model_hint="h3")["loraModel"] == "ltx25"
    with pytest.raises(ValueError, match="declares"):
        lora.clip_lora(dict(kind="lora", path=str(source), loraModel="h3"), {}, "h3")


def test_model_hint_cannot_resolve_conflicting_training_metadata(tmp_path):
    source = adapter(tmp_path, {"base_model": "MiniMax-H3", "model_version": "2.5.0"})
    with pytest.raises(ValueError, match="conflicting"):
        lora.inspect_lora(source, model_hint="h3")


@pytest.mark.parametrize("model_hint", [True, 3, {}, [], "", "movie", "ltx24"])
def test_invalid_import_model_hint_is_rejected(tmp_path, model_hint):
    with pytest.raises(ValueError, match="model"):
        lora.inspect_lora(adapter(tmp_path, {}), model_hint=model_hint)


@pytest.mark.parametrize(
    "schedule",
    [
        {"inference_steps": "8"},
        {"num_inference_steps": "8"},
        {"steps": "8"},
        {"transformer_evaluations": "8"},
        {"schedule_points": "9"},
    ],
)
def test_declared_eight_step_turbo_import_requires_dedicated_recipe(tmp_path, schedule):
    source = adapter(tmp_path, {"base_model": "MiniMax-H3", "adapter_profile": "turbo", **schedule})
    with pytest.raises(ValueError, match="dedicated.*recipe"):
        lora.inspect_lora(source)
