"""Small signal/tensor and notation contracts; no checkpoints are downloaded."""

import importlib.util
import json

import numpy as np
import pytest


def test_native_package_exists():
    assert importlib.util.find_spec("sheetsage2_mlx") is not None


def test_checkpoint_rejects_adapter_and_truncated_header(tmp_path):
    from sheetsage2_mlx import inspect_checkpoint

    target = tmp_path / "model.safetensors"
    header = json.dumps(
        {
            "adapter.layers.0.attn.query_proj.lora_A.weight": {
                "dtype": "BF16",
                "shape": [64, 1024],
                "data_offsets": [0, 131072],
            }
        }
    ).encode()
    target.write_bytes(len(header).to_bytes(8, "little") + header)
    with pytest.raises(ValueError, match="merged|incomplete|truncated"):
        inspect_checkpoint(target)
    target.write_bytes((1000).to_bytes(8, "little") + b"{}")
    with pytest.raises(ValueError, match="header|truncated"):
        inspect_checkpoint(target)


def test_audio_contract_rejects_long_stereo_nonfinite_before_model_loading():
    from sheetsage2_mlx import transcribe

    for audio, message in [
        (np.zeros((2, 24000), np.float32), "mono"),
        (np.zeros(7200001, np.float32), "300"),
        (np.full(24000, np.nan, np.float32), "finite"),
        (np.zeros(100, np.float32), "1025"),
    ]:
        with pytest.raises(ValueError, match=message):
            transcribe(audio, "/no/checkpoint")


def test_cancellation_before_checkpoint_loading():
    from sheetsage2_mlx import transcribe

    with pytest.raises(InterruptedError, match="cancel"):
        transcribe(np.zeros(24000, np.float32), "/no/checkpoint", cancelled=lambda: True)


def test_mel_frontend_matches_direct_dft_and_uses_checkpoint_normalization():
    import mlx.core as mx

    from sheetsage2_mlx._model import mel_features

    rng = np.random.default_rng(40)
    audio = rng.normal(0, 0.1, 4096).astype(np.float32)
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(2048) / 2048)).astype(np.float32)
    bank = rng.uniform(0, 0.05, (1025, 4)).astype(np.float32)
    mean = np.array([1, 2, 3, 4], np.float32)
    std = np.array([2, 3, 4, 5], np.float32)
    values = {
        "spectrogram.window": mx.array(window),
        "mel_scale.fb": mx.array(bank),
        "mel_mean": mx.array(mean),
        "mel_std": mx.array(std),
    }
    actual = np.array(mel_features(mx.array(audio), values))
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(audio, 1024, mode="reflect"), 2048)[
        ::240
    ][:-1]
    power = abs(np.fft.rfft(frames * window)) ** 2
    expected = (10 * np.log10(np.maximum(power @ bank, 1e-10)) - mean) / std
    np.testing.assert_allclose(actual[0], expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("groups", [1, 4])
def test_conv1d_matches_explicit_cross_correlation(groups):
    import mlx.core as mx

    from sheetsage2_mlx._model import conv1d

    rng = np.random.default_rng(2)
    x = rng.normal(size=(1, 9, 4)).astype(np.float32)
    w = rng.normal(size=(4, 4 // groups, 3)).astype(np.float32)
    b = rng.normal(size=(4,)).astype(np.float32)
    actual = np.array(conv1d(mx.array(x), mx.array(w), mx.array(b), padding=1, groups=groups))
    padded = np.pad(x, ((0, 0), (1, 1), (0, 0)))
    expected = np.empty_like(x)
    for t in range(9):
        for out in range(4):
            first = (out // (4 // groups)) * (4 // groups)
            expected[0, t, out] = (
                padded[0, t : t + 3, first : first + 4 // groups].T * w[out]
            ).sum() + b[out]
    np.testing.assert_allclose(actual, expected, atol=2e-6)


def _decoder_weights(seed=9):
    import mlx.core as mx

    rng = np.random.default_rng(seed)
    values = {}

    def linear(name, out=8, inp=8):
        values[name + ".weight"] = mx.array(rng.normal(0, 0.12, (out, inp)).astype(np.float32))
        values[name + ".bias"] = mx.array(rng.normal(0, 0.02, out).astype(np.float32))

    def norm(name):
        values[name + ".weight"] = mx.ones((8,))
        values[name + ".bias"] = mx.zeros((8,))

    values["token_embedding.weight"] = mx.array(rng.normal(0, 0.2, (13, 8)).astype(np.float32))
    values["decoder.embed_positions.weight"] = mx.array(
        rng.normal(0, 0.2, (32, 8)).astype(np.float32)
    )
    norm("decoder.layernorm_embedding")
    for i in range(2):
        p = f"decoder.layers.{i}."
        for kind in ("self_attn", "encoder_attn"):
            for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
                linear(p + kind + "." + proj)
            norm(p + kind + "_layer_norm")
        linear(p + "fc1", 16, 8)
        linear(p + "fc2", 8, 16)
        norm(p + "final_layer_norm")
    return values


def test_bart_cache_matches_full_causal_decode_and_future_does_not_leak():
    import mlx.core as mx

    from sheetsage2_mlx._model import Decoder

    memory = mx.array(np.random.default_rng(8).normal(size=(1, 7, 8)).astype(np.float32))
    decoder = Decoder(_decoder_weights(), memory, heads=2)
    whole = np.array(decoder.step([1, 4, 5, 3]))
    decoder.reset()
    first = np.array(decoder.step([1, 4]))
    third = np.array(decoder.step([5]))
    last = np.array(decoder.step([3]))
    np.testing.assert_allclose(np.concatenate([first, third, last], axis=1), whole, atol=2e-6)
    decoder.reset()
    changed = np.array(decoder.step([1, 4, 12, 12]))
    np.testing.assert_allclose(changed[:, :2], whole[:, :2], atol=2e-6)
    decoder.close()
    assert not decoder.weights and not decoder.cache


def test_bart_matches_independent_transformers_reference_when_available():
    torch = pytest.importorskip("torch")
    import mlx.core as mx
    from transformers import BartConfig
    from transformers.models.bart.modeling_bart import BartDecoder

    from sheetsage2_mlx._model import Decoder

    weights = _decoder_weights()
    config = BartConfig(
        vocab_size=13,
        d_model=8,
        decoder_layers=2,
        decoder_attention_heads=2,
        decoder_ffn_dim=16,
        max_position_embeddings=30,
        pad_token_id=0,
        dropout=0,
        attention_dropout=0,
        activation_dropout=0,
        activation_function="gelu",
    )
    config._attn_implementation = "sdpa"
    reference = BartDecoder(config).eval()
    state = {
        k.removeprefix("decoder."): torch.tensor(np.array(v))
        for k, v in weights.items()
        if k.startswith("decoder.")
    }
    state["embed_tokens.weight"] = torch.tensor(np.array(weights["token_embedding.weight"]))
    reference.load_state_dict(state)
    memory = np.random.default_rng(8).normal(size=(1, 7, 8)).astype(np.float32)
    tokens = [1, 4, 5, 3]
    with torch.no_grad():
        hidden = reference(
            input_ids=torch.tensor([tokens]),
            encoder_hidden_states=torch.tensor(memory),
            use_cache=False,
        ).last_hidden_state
        expected = (hidden @ state["embed_tokens.weight"].T).numpy()
    native = Decoder(weights, mx.array(memory), heads=2)
    np.testing.assert_allclose(np.array(native.step(tokens)), expected, rtol=2e-5, atol=2e-5)
    native.close()


def test_symbolic_vocabulary_and_required_duration_grammar():
    from sheetsage2_mlx.symbolic import Grammar, Vocabulary

    v = Vocabulary()
    assert v.size == 31678
    assert v.prefix == [1, 4, 5, 6, 7, 9, 11, 3]
    assert v.ranges["time"] == (517, 30517)
    assert v.ranges["pitch"] == (31398, 31654)
    grammar = Grammar(v)
    assert grammar.allowed()[260]
    assert not grammar.allowed()[2]
    assert not grammar.allowed()[v.ranges["time"][0]]
    assert not grammar.allowed()[v.ranges["pitch"][0]]
    grammar.accept(260)
    grammar.accept(v.ranges["meter"][0] + 20)
    assert grammar.allowed()[v.ranges["eighth"][0]]
    assert not grammar.allowed()[v.ranges["key"][0]]
    assert not grammar.allowed()[2]
    assert not grammar.allowed()[260]


def test_symbolic_events_have_observed_timing_and_abc_preserves_notes():
    from sheetsage2_mlx.symbolic import Vocabulary, decode, to_abc

    v = Vocabulary()
    r = v.ranges
    tokens = v.prefix + [
        260,
        r["time"][0] + 100,
        r["meter"][0] + 20,
        r["eighth"][0],
        r["structure"][0] + 1,
        r["key"][0],
        r["pitch"][0] + 60,
        r["duration"][0] + 3,
        268,
        r["time"][0] + 200,
        r["pitch"][0] + 64,
        r["duration"][0] + 3,
        2,
    ]
    result = decode(tokens, duration=3)
    assert result["events"][0]["time"] == 1
    assert result["events"][1]["time"] == 2
    note = result["events"][0]["values"]["melody"][0]
    assert note["pitch"] == 60 and note["duration_steps"] == 4
    assert note["startSeconds"] == 1 and note["endSeconds"] == 1.5
    abc = to_abc(result["events"], melody_only=True)
    assert "X:1" in abc and "C" in abc and "E" in abc


def test_no_timestamp_evidence_does_not_fabricate_a_score():
    from sheetsage2_mlx.symbolic import Vocabulary, decode

    v = Vocabulary()
    with pytest.raises(ValueError, match="timestamp|timing"):
        decode(v.prefix + [260, v.ranges["pitch"][0] + 60, 2], duration=2)


def test_checkpoint_digest_detects_replacement_and_checks_cancellation(tmp_path):
    from sheetsage2_mlx.checkpoint import checkpoint_digest, unchanged

    target = tmp_path / "weights"
    target.write_bytes(b"weights")
    digest, identity = checkpoint_digest(target)
    assert len(digest) == 64
    unchanged(target, identity)
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        unchanged(target, identity)

    def stop():
        raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError, match="cancelled"):
        checkpoint_digest(target, stop)


def test_checkpoint_inventory_rejects_unexpected_decoder_layers(tmp_path):
    from sheetsage2_mlx.checkpoint import expected_shapes, inspect_checkpoint

    shapes = expected_shapes()
    shapes["decoder.layers.6.fc1.weight"] = (2048, 512)
    offset = 0
    header = {}
    for name, shape in shapes.items():
        size = int(np.prod(shape)) * 2
        header[name] = dict(shape=list(shape), dtype="BF16", data_offsets=[offset, offset + size])
        offset += size
    encoded = json.dumps(header).encode()
    target = tmp_path / "weights.safetensors"
    # Sparse file: exercise the real header contract without allocating model weights.
    with target.open("wb") as f:
        f.write(len(encoded).to_bytes(8, "little") + encoded)
        f.truncate(8 + len(encoded) + offset)
    with pytest.raises(ValueError, match="unexpected|Unsupported"):
        inspect_checkpoint(target)
