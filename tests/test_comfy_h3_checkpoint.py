import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from minimax_h3_mlx import comfy_h3_checkpoint as checkpoint
from minimax_h3_mlx.config import DiTConfig
from minimax_h3_mlx.dit import MiniMaxH3DiT


def regular_hadamard(group):
    base = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float32)
    result = np.ones((1, 1), dtype=np.float32)
    while len(result) < group:
        result = np.kron(result, base)
    return result / np.sqrt(group)


@pytest.mark.parametrize("group", [4, 16, 64, 256])
def test_inverse_convrot_matches_regular_hadamard_and_is_self_inverse(group):
    values = np.random.default_rng(7).normal(size=(3, group * 2)).astype(np.float32)
    expected = (values.reshape(-1, group) @ regular_hadamard(group)).reshape(values.shape)
    actual = checkpoint.inverse_convrot(mx.array(values), group)
    np.testing.assert_allclose(np.array(actual), expected, atol=2e-6)
    np.testing.assert_allclose(
        np.array(checkpoint.inverse_convrot(actual, group)), values, atol=2e-6
    )


def fixture(tmp_path, monkeypatch):
    config = DiTConfig(
        hidden_size=64,
        num_layers=2,
        token_refiner_num_layers=1,
        num_attention_heads=4,
        attention_head_dim=16,
        ffn_hidden_size=32,
        latents_dim=4,
        audio_latents_dim=8,
        text_dim=32,
        timestep_input_dim=16,
        time_embed_hidden_size=64,
        time_embed_dim=32,
        adaln_out_features=1152,
        final_adaln_out_features=128,
        rope_inv_freq_len=2,
    )
    monkeypatch.setattr(checkpoint, "DiTConfig", lambda: config)
    monkeypatch.setattr(checkpoint, "ROW_CHUNK", 32)
    model = MiniMaxH3DiT(config)
    values = dict(tree_flatten(model.parameters()))
    quant_key = "blocks.0.attn.qkv_proj.weight"
    shape = values[quant_key].shape
    q = np.random.default_rng(3).integers(-127, 128, size=shape, dtype=np.int8)
    scale = np.linspace(0.0005, 0.005, shape[0], dtype=np.float32).reshape(-1, 1)
    values[quant_key] = mx.array(q)
    prefix = quant_key.removesuffix(".weight")
    values[prefix + ".weight_scale"] = mx.array(scale)
    marker = json.dumps(
        {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 64}
    ).encode()
    values[prefix + ".comfy_quant"] = mx.array(list(marker), dtype=mx.uint8)
    values = {"model.diffusion_model." + k: v for k, v in values.items()}
    filename = tmp_path / "finetune.safetensors"
    mx.save_safetensors(str(filename), values)
    return filename, config, q, scale


def test_direct_loader_preserves_source_and_decodes_only_active_block(tmp_path, monkeypatch):
    filename, config, q, scale = fixture(tmp_path, monkeypatch)
    before = filename.stat()
    model = checkpoint.load_comfy_h3_dit(filename)
    pager = model.paged_blocks
    try:
        assert model.blocks == []
        assert pager.store.source.quantized_tensors_read == 0
        with pager.window(0) as blocks:
            weight = blocks[0].attn.qkv_proj.weight
            expected = (
                (q.astype(np.float32) * scale).reshape(-1, 64) @ regular_hadamard(64)
            ).reshape(q.shape)
            expected = expected.reshape(3, 4, 16, 64).transpose(1, 0, 2, 3).reshape(q.shape)
            np.testing.assert_allclose(
                np.array(weight.astype(mx.float32)), expected, rtol=0.008, atol=1e-7
            )
        assert pager.store.source.quantized_tensors_read == 1
        assert pager.store.active_page is None
        assert model.video_patch_proj.weight.dtype == mx.float32
        with pytest.raises(RuntimeError, match="cancel"):
            with pager.window(0):
                raise RuntimeError("cancel")
        assert pager.store.active_page is None
        assert sorted(p.name for p in tmp_path.iterdir()) == [filename.name]
        assert filename.stat().st_mtime_ns == before.st_mtime_ns
    finally:
        pager.close()
    assert pager.store.source.closed


def test_header_validation_rejects_incomplete_checkpoint(tmp_path, monkeypatch):
    filename, *_ = fixture(tmp_path, monkeypatch)
    assert checkpoint.is_comfy_h3_checkpoint(filename)
    report = checkpoint.describe_comfy_h3(filename)
    assert report["window_bytes"] > 0
    with filename.open("r+b") as handle:
        handle.truncate(filename.stat().st_size - 1)
    with pytest.raises(ValueError):
        checkpoint.describe_comfy_h3(filename)


def test_fixed_refiner_qkv_reorders_contiguous_source_heads(tmp_path, monkeypatch):
    filename, config, *_ = fixture(tmp_path, monkeypatch)
    key = "token_refiner.blocks.0.attn.qkv_proj.weight"
    original = mx.load(str(filename))[checkpoint.PREFIX + key]
    # Compare projections semantically: native [head, QKV, channel] must equal
    # the source's separate contiguous Q, K and V projection slices.
    source = checkpoint.ComfyH3TensorSource(checkpoint._inspect(filename))
    try:
        native = source.read(key)
        x = mx.arange(config.hidden_size, dtype=mx.float32) / config.hidden_size
        actual = (native.astype(mx.float32) @ x).reshape(4, 3, 16)
        expected = (original.astype(mx.bfloat16).astype(mx.float32) @ x).reshape(3, 4, 16)
        np.testing.assert_allclose(np.array(actual.transpose(1, 0, 2)), np.array(expected))
    finally:
        source.close()


def test_source_mutation_and_skip_adaln_are_guarded(tmp_path, monkeypatch):
    filename, *_ = fixture(tmp_path, monkeypatch)
    model = checkpoint.load_comfy_h3_dit(filename)
    pager = model.paged_blocks
    try:
        values = pager.store.load_blocks((0,), skip_adaln=True)
        assert not any(".adaln_proj." in key for key in values)
        values.clear()
        pager.store.release()
        filename.touch()
        with pytest.raises(ValueError, match="changed"):
            pager.store.load_block(0)
    finally:
        pager.close()


def test_direct_cache_counts_decoded_weights_and_releases_on_close(tmp_path, monkeypatch):
    filename, *_ = fixture(tmp_path, monkeypatch)
    pager = checkpoint.load_comfy_h3_dit(filename).paged_blocks
    store = pager.store
    try:
        values = store.load_block(0)
        decoded_bytes = sum(value.nbytes for value in values.values())
        values.clear()
        store.release()
        store.configure_cache(decoded_bytes)
        store.begin_cache()
        values = store.load_block(0)
        values.clear()
        store.release()
        assert store.retained_bytes == decoded_bytes
        reads = store.source.payload_bytes_read
        values = store.load_block(0)
        assert store.source.payload_bytes_read == reads
        assert store.raw_cache_hits == 1
        values.clear()
        store.release()
    finally:
        pager.close()
    assert store.retained_bytes == 0
    assert store.source.closed


@pytest.mark.parametrize("invalid", ["format", "rotation", "missing_scale"])
def test_unsupported_quantization_rejected_before_loading(tmp_path, monkeypatch, invalid):
    filename, *_ = fixture(tmp_path, monkeypatch)
    tensors = mx.load(str(filename))
    mx.eval(tensors)
    base = checkpoint.PREFIX + "blocks.0.attn.qkv_proj"
    if invalid == "missing_scale":
        del tensors[base + ".weight_scale"]
    else:
        marker = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 64}
        marker["format" if invalid == "format" else "convrot_groupsize"] = (
            "unknown" if invalid == "format" else 128
        )
        tensors[base + ".comfy_quant"] = mx.array(list(json.dumps(marker).encode()), dtype=mx.uint8)
    mx.save_safetensors(str(filename), tensors)
    with pytest.raises(ValueError, match="Unsupported H3|Invalid INT8"):
        checkpoint.describe_comfy_h3(filename)
