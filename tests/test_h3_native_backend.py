from dataclasses import replace
from types import SimpleNamespace

import pytest

from minimax_h3_mlx.native_backend import find_native_worker, validate_native_settings
from wee_todd_nodes.runtime import H3GenerationConfig


def adapter(**changes):
    values = dict(
        strength=1.0,
        resolved_profile="turbo",
        resolved_qkv_layout="contiguous_qkv",
        start_after_evaluations=0,
        adaln_input_grid=None,
    )
    return SimpleNamespace(**(values | changes))


def settings(**changes):
    return H3GenerationConfig(steps=5, transformer_backend="nnc_experimental", **changes)


def test_old_config_keeps_mlx_and_new_backend_validates():
    assert H3GenerationConfig().transformer_backend == "mlx"
    settings().validate()
    with pytest.raises(ValueError, match="transformer_backend"):
        replace(settings(), transformer_backend="typo").validate()


def test_qualified_settings_accept_four_actual_evaluations():
    validate_native_settings(settings(), "ref2va", [adapter()])


@pytest.mark.parametrize(
    "changes",
    [
        dict(steps=4),
        dict(drop_adaln=False),
        dict(sampling_method="res_multistep"),
        dict(inference_optimization="transient_q8"),
        dict(paging_cache_gb=1),
        dict(projection_backend="mpp_experimental"),
    ],
)
def test_unsupported_settings_rejected(changes):
    with pytest.raises(ValueError, match="Native NNC"):
        validate_native_settings(replace(settings(), **changes), "ref2va", [adapter()])


@pytest.mark.parametrize(
    "changes",
    [
        dict(strength=0.8),
        dict(start_after_evaluations=1),
        dict(resolved_profile="standard"),
        dict(resolved_qkv_layout="native_interleaved"),
        dict(adaln_input_grid="grid"),
    ],
)
def test_unqualified_adapter_rejected(changes):
    with pytest.raises(ValueError, match="Native NNC"):
        validate_native_settings(settings(), "ref2va", [adapter(**changes)])


def test_controls_and_resident_blocks_rejected():
    for kwargs in [dict(features={"blockcache": object()}), dict(block_residency="resident")]:
        with pytest.raises(ValueError, match="Native NNC"):
            validate_native_settings(settings(), "ref2va", [adapter()], **kwargs)


def test_discovery_prefers_bundled_worker_without_checkout(tmp_path, monkeypatch):
    root = tmp_path / "Studio.app/Contents/Resources/RendererSource"
    worker = root.parent.parent / "MacOS/WeeToddH3Worker"
    worker.parent.mkdir(parents=True)
    worker.write_text("binary")
    worker.chmod(0o755)
    monkeypatch.delenv("WEETODD_H3_NATIVE_WORKER", raising=False)
    assert find_native_worker(root=root) == worker.resolve()


def test_bad_explicit_worker_does_not_silently_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("WEETODD_H3_NATIVE_WORKER", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="worker"):
        find_native_worker(root=tmp_path)


def native_header():
    tensors = {}
    targets = {
        "attn.qkv_proj": (21504, 5376),
        "attn.out_proj": (5376, 7168),
        "mlp.fc1": (28672, 5376),
        "mlp.fc2": (5376, 14336),
    }
    for index in range(50):
        for target, (output_size, input_size) in targets.items():
            prefix = f"diffusion_model.blocks.{index}.{target}"
            tensors[prefix + ".lora_A.weight"] = dict(dtype="BF16", shape=[1, input_size])
            tensors[prefix + ".lora_B.weight"] = dict(dtype="BF16", shape=[output_size, 1])
            tensors[prefix + ".alpha"] = dict(dtype="F32", shape=[])
    return tensors


def test_every_native_block_adapter_validated_before_load():
    from minimax_h3_mlx.native_backend import validate_native_adapter_header

    tensors = native_header()
    validate_native_adapter_header(tensors)
    tensors["diffusion_model.blocks.49.mlp.fc2.lora_A.weight"]["dtype"] = "F16"
    with pytest.raises(ValueError, match="BF16"):
        validate_native_adapter_header(tensors)
    tensors = native_header()
    tensors["diffusion_model.blocks.30.norm.lora_A.weight"] = dict(dtype="BF16", shape=[1, 5376])
    with pytest.raises(ValueError, match="target"):
        validate_native_adapter_header(tensors)


def test_oversized_native_request_fails_before_weights_and_exact_conditioning_checked():
    from minimax_h3_mlx.native_backend import validate_native_rows

    with pytest.raises(ValueError, match="40000"):
        validate_native_settings(settings(width=1920, height=1088), "ref2va", [adapter()])
    with pytest.raises(ValueError, match="40000"):
        validate_native_rows(settings(), prompt_rows=100, video_rows=40000)


@pytest.mark.parametrize(
    "prefix",
    [
        "blocks.",
        "transformer.blocks.",
        "model.diffusion_model.blocks.",
        "diffusion_model.transformer_blocks.",
    ],
)
def test_native_rejects_extra_alias_targets(prefix):
    from minimax_h3_mlx.native_backend import validate_native_adapter_header

    tensors = native_header()
    tensors[prefix + "0.attn.to_q.lora_A.weight"] = dict(dtype="BF16", shape=[1, 5376])
    with pytest.raises(ValueError, match="target"):
        validate_native_adapter_header(tensors)


def test_warm_native_reports_do_not_mutate_previous_take():
    from minimax_h3_mlx.native_blocks import NativeH3Blocks

    owner = NativeH3Blocks("unused", "unused", "unused")
    previous = owner.report
    previous["calls"].append({"step_index": 0})
    owner.begin_run()
    owner.report["calls"].append({"step_index": 1})
    owner.close()
    assert previous["calls"] == [{"step_index": 0}]
