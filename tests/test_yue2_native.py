"""Independent protocol and numerical checks for the owned YuE2 implementation."""

import importlib.util
import math
import subprocess
import sys

import numpy as np
import pytest

from yue2_mlx.config import validate_request
from yue2_mlx.protocol import chunk_ranges, negative_tokens, prefix_tokens


def request(**changes):
    return dict(model_path="/tmp/yue2", style="piano", lyrics="hello", **changes)


def test_validation_defaults_and_unknowns():
    value = validate_request(request())
    assert value["steps"] == 32 and value["seed"] == 831001
    assert value["abc_sampling"]["temperature"] == 0.7
    assert value["semantic_sampling"]["max_tokens"] == 9000
    assert value["cfg_scale"] == 1
    with pytest.raises(ValueError, match="unknown"):
        validate_request(request(typo=1))


@pytest.mark.parametrize(
    "changes",
    [
        dict(steps=True),
        dict(steps=1.5),
        dict(seed=-1),
        dict(cfg_scale=float("nan")),
        dict(semantic_sampling={"temperature": float("inf")}),
        dict(abc="score", cot="off"),
        dict(precision="fp16"),
        dict(memory_mode="banana"),
        dict(semantic_sampling={"max_tokens": 10}),
        dict(abc_sampling={"oops": 1}),
    ],
)
def test_validation_rejects_invalid(changes):
    with pytest.raises(ValueError):
        validate_request(request(**changes))


class TextTokenizer:
    def encode(self, text):
        self.last = text
        return [11, 12]


def test_protocol_exact_boundaries_and_conditioning():
    tok = TextTokenizer()
    req = validate_request(request(cot="full"))
    assert prefix_tokens(req, tok) == [151643, 11, 12, 151847]
    assert tok.last == (
        "Generate a chord-annotated ABC transcription, then generate music with codec tokens "
        "from the given conditions.\n[Tags]\npiano\n[Lyrics]\nhello\n"
    )
    assert prefix_tokens(req, tok, [21]) == [151643, 11, 12, 151847, 21, 151848, 151851]
    assert negative_tokens(req, tok, [21]) == [151643, 11, 12, 151847, 21, 151848, 151851]
    off = validate_request(request(cot="off"))
    assert off["cfg_scale"] == 1.01
    assert prefix_tokens(off, tok) == [151643, 11, 12, 151847, 151848, 151851]
    assert negative_tokens(off, tok, []) == [151643, 11, 12, 151851]


def test_context_chunks_account_for_both_streams_and_boundaries():
    assert chunk_ranges(10, 3, context=16) == [(0, 5), (5, 10)]
    with pytest.raises(ValueError):
        chunk_ranges(1, 14, context=16)


@pytest.mark.parametrize("score", [None, [21, 22]])
def test_context_error_reports_a_usable_music_budget(score):
    import re

    from yue2_mlx.pipeline import validate_context

    req = validate_request(
        request(abc_sampling={"max_tokens": 32}, semantic_sampling={"max_tokens": 1000})
    )
    with pytest.raises(ValueError) as caught:
        validate_context(req, TextTokenizer(), score, 1000)
    match = re.search(r"at most (\d+) music tokens \(([\d.]+) seconds\)", str(caught.value))
    assert match, "Context error must explain the usable length budget"
    limit = int(match[1])
    assert limit == (962 if score is None else 992)
    assert float(match[2]) == (38.48 if score is None else 39.68)
    req["semantic_sampling"]["max_tokens"] = limit
    validate_context(req, TextTokenizer(), score, 1000)
    req["semantic_sampling"]["max_tokens"] += 1
    with pytest.raises(ValueError):
        validate_context(req, TextTokenizer(), score, 1000)


def test_import_is_light():
    code = 'import sys; import yue2_mlx.pipeline; assert "mlx.core" not in sys.modules'
    subprocess.run([sys.executable, "-c", code], check=True)


mlx = pytest.mark.skipif(importlib.util.find_spec("mlx") is None, reason="MLX unavailable")


@mlx
def test_bf16_norm_matches_explicit_rounding():
    import mlx.core as mx

    from yue2_mlx.model import RMSNorm

    x = mx.array([[0.13, 12, -3.14, 2.77]], dtype=mx.bfloat16)
    norm = RMSNorm(4, 1e-6)
    norm.weight = mx.array([1.1, 0.9, 0.6, 2.2], dtype=mx.bfloat16)
    scale = mx.rsqrt(mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True) + 1e-6).astype(
        x.dtype
    )
    expected = ((x * scale).astype(x.dtype) * norm.weight).astype(x.dtype)
    np.testing.assert_array_equal(
        np.array(norm(x).astype(mx.float32)), np.array(expected.astype(mx.float32))
    )


def tiny_config():
    return dict(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=32,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        max_position_embeddings=64,
        latent_dim=4,
        max_latent_frames=64,
        timestep_shift=1.0,
    )


@mlx
def test_cached_ar_matches_full_causal_and_cache_bounds():
    import mlx.core as mx

    from yue2_mlx.model import KVCache, YuE2Model

    mx.random.seed(3)
    model = YuE2Model(tiny_config())
    all_tokens = mx.array([[1, 2, 3, 4]])
    full = model.ar(all_tokens, model.make_cache(8), all_positions=True)
    cache = model.make_cache(8)
    model.ar(all_tokens[:, :2], cache)
    incremental = model.ar(all_tokens[:, 2:], cache, all_positions=True)
    np.testing.assert_allclose(np.array(incremental), np.array(full[:, 2:]), rtol=1e-5, atol=1e-5)
    bound = KVCache(1)
    k = mx.ones((1, 1, 2, 4))
    with pytest.raises(ValueError, match="capacity"):
        bound.append(k, k)


@mlx
def test_sampling_frequency_penalty_and_masking():
    import mlx.core as mx

    from yue2_mlx.sampling import filter_logits

    scores = mx.full((184704,), -10.0)
    scores[151853] = 8.0
    scores[151854] = -2.0
    scores[151852] = 100.0
    settings = dict(
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        repetition_penalty=2.0,
        penalty_window=10,
        min_tokens=1,
        max_tokens=10,
    )
    result = np.array(filter_logits(scores, settings, [151853, 151853, 151854], 0, "semantic"))
    assert result[151853] == 2 and result[151854] == -4
    assert math.isinf(result[151852]) and math.isinf(result[12])


@mlx
def test_midpoint_constant_velocity_and_cancellation():
    import mlx.core as mx

    from yue2_mlx.acoustic import midpoint

    initial = mx.ones((2, 4), dtype=mx.float32) * 3
    out = midpoint(lambda x, t: mx.ones_like(x) * 2, initial, 7)
    np.testing.assert_allclose(np.array(out), 1.0, atol=1e-6)
    with pytest.raises(InterruptedError):
        midpoint(lambda x, t: x, initial, 3, cancelled=lambda: True)


@mlx
def test_transpose_convolution_matches_impulse_definition():
    import mlx.core as mx

    from yue2_mlx.vae import Decoder

    cfg = dict(channels=1, c_mults=[1], strides=[3], latent_dim=1, out_channels=2)
    model = Decoder(cfg)
    conv = model.layers[1].layers[1]
    conv.weight = (
        mx.array(
            [
                [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]],
                [[-1.0], [-2.0], [-3.0], [-4.0], [-5.0], [-6.0]],
            ]
        )
        if conv.weight.shape[0] == 2
        else mx.array([[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]])
    )
    conv.bias = mx.zeros_like(conv.bias)
    actual = np.array(conv(mx.array([[[1.0], [2.0]]])))[0, :, 0]
    # Correlation transpose: sum each input-scaled kernel starting at i*stride-padding.
    expected = np.zeros(5)
    for i, v in enumerate([1.0, 2.0]):
        for j, w in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
            p = i * 3 + j - 2
            if 0 <= p < 5:
                expected[p] += v * w
    np.testing.assert_array_equal(actual, expected)


@mlx
def test_vae_tiled_decode_matches_whole_with_halo():
    import mlx.core as mx

    from yue2_mlx.vae import Decoder

    cfg = dict(
        channels=1,
        c_mults=[1, 1, 1, 1, 1, 1],
        strides=[2, 2, 4, 4, 5, 6],
        latent_dim=2,
        out_channels=2,
    )
    decoder = Decoder(cfg)
    z = mx.random.normal((36, 2)) * 0.01
    whole = np.array(decoder(z[None])[0])
    tiled = np.array(decoder.decode_tiled(z, core=4, halo=16))
    assert whole.shape == (36 * 1920 - 64, 2)
    np.testing.assert_allclose(tiled, whole, atol=1e-5, rtol=1e-4)


def test_safetensors_rejects_truncated_payload(tmp_path):
    import json
    import struct

    from yue2_mlx.checkpoint import tensor_header

    metadata = json.dumps({"x": dict(dtype="F32", shape=[2], data_offsets=[0, 8])}).encode()
    filename = tmp_path / "broken.safetensors"
    filename.write_bytes(struct.pack("<Q", len(metadata)) + metadata + b"1234")
    with pytest.raises(ValueError, match="payload"):
        tensor_header(filename)


def test_pcm32_master_length_and_channels(tmp_path):
    import wave

    from yue2_mlx.pipeline import write_master

    audio = np.array([[0.5, -0.5], [1.0, -1.0], [0.0, 0.25]], dtype=np.float32)
    filename = tmp_path / "master.wav"
    write_master(filename, audio)
    with wave.open(str(filename), "rb") as stream:
        assert (
            stream.getframerate(),
            stream.getnchannels(),
            stream.getsampwidth(),
            stream.getnframes(),
        ) == (48000, 2, 4, 3)
        pcm = np.frombuffer(stream.readframes(3), dtype="<i4").reshape(-1, 2)
    np.testing.assert_allclose(pcm.astype(np.float64) / 2147483647, audio, atol=1e-9)
    with pytest.raises(FloatingPointError):
        write_master(tmp_path / "bad.wav", np.array([[float("nan"), 0.0]], dtype=np.float32))


def test_pipeline_cancel_before_loading_and_existing_output(tmp_path):
    from yue2_mlx.pipeline import generate

    with pytest.raises(InterruptedError):
        generate(request(), tmp_path / "new", cancelled=lambda: True)
    with pytest.raises(FileExistsError):
        generate(request(), tmp_path)


@mlx
def test_cached_acoustic_matches_full_hybrid_attention():
    """Recompute the full mixed stream, including blocked AR→NAR edges."""
    import mlx.core as mx

    from yue2_mlx.model import YuE2Model

    mx.random.seed(90)
    model = YuE2Model(tiny_config())
    tokens = [1, 2, 3]
    state = mx.array([[0.1, -0.2, 0.3, 0.4], [0.5, 0.4, -0.3, 0.2]])
    cache = model.acoustic_prefix(tokens)
    cached = model.velocity(state, 0.0, cache, len(tokens))
    count = len(state) + 2
    acoustic = model.vae2llm(mx.pad(state, [(1, 1), (0, 0)])[None])
    acoustic = acoustic + model.time_embedder(mx.full((count,), 0.5), mx.float32)[None]
    acoustic = acoustic + model.latent_pos_embed.pe[mx.arange(count)][None]
    x = mx.concatenate([model.model.embed_tokens(mx.array([tokens])), acoustic], axis=1)
    total = x.shape[1]
    allowed = np.zeros((total, total), dtype=bool)
    allowed[:3, :3] = np.tril(np.ones((3, 3), dtype=bool))
    allowed[3:, :] = True
    for layer in model.model.layers:
        aq, ak, av = layer.self_attn.project(layer.input_layernorm(x), 0)
        nq, nk, nv = layer.nar_self_attn.project(layer.nar_input_layernorm(x), 0)
        q = mx.concatenate([aq[:, :, :3], nq[:, :, 3:]], axis=2)
        k = mx.concatenate([ak[:, :, :3], nk[:, :, 3:]], axis=2)
        v = mx.concatenate([av[:, :, :3], nv[:, :, 3:]], axis=2)
        # NumPy attention is an independent unfused oracle with explicit GQA.
        qn, kn, vn = map(np.array, (q, k, v))
        kn = np.repeat(kn, 2, axis=1)
        vn = np.repeat(vn, 2, axis=1)
        logits = qn @ kn.transpose(0, 1, 3, 2) / 2
        logits = np.where(allowed, logits, -np.inf)
        weights = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        context = mx.array((weights @ vn).transpose(0, 2, 1, 3).reshape(1, total, 8))
        x = x + mx.concatenate(
            [layer.self_attn.o_proj(context[:, :3]), layer.nar_self_attn.o_proj(context[:, 3:])],
            axis=1,
        )
        x = x + mx.concatenate(
            [
                layer.mlp(layer.post_attention_layernorm(x[:, :3])),
                layer.nar_mlp(layer.nar_pre_mlp_layernorm(x[:, 3:])),
            ],
            axis=1,
        )
    expected = model.llm2vae(model.model.norm(x))[0, 4:-1]
    np.testing.assert_allclose(np.array(cached), np.array(expected), atol=2e-5, rtol=2e-5)


@mlx
def test_guidance_rounds_each_bf16_stage():
    import mlx.core as mx

    from yue2_mlx.sampling import guided_logits, key_for_seed

    positive = mx.array([0.03125, 11.1, -7.3], dtype=mx.bfloat16)
    negative = mx.array([0.021, 8.9, -2.2], dtype=mx.bfloat16)
    scale = 1.01
    delta = (positive - negative).astype(mx.bfloat16)
    scaled = (delta.astype(mx.float32) * scale).astype(mx.bfloat16)
    expected = (negative + scaled).astype(mx.bfloat16)
    np.testing.assert_array_equal(
        np.array(guided_logits(positive, negative, scale).astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
    )
    assert not np.array_equal(np.array(key_for_seed(1)), np.array(key_for_seed(2**32 + 1)))


@mlx
def test_guidance_requires_negative_prefix_before_allocating_cache():
    from yue2_mlx.model import YuE2Model
    from yue2_mlx.sampling import generate_tokens

    model = YuE2Model(tiny_config())
    settings = validate_request(request())["semantic_sampling"] | {"max_tokens": 1, "min_tokens": 0}
    with pytest.raises(ValueError, match="negative"):
        generate_tokens(model, [1], settings, 1, "semantic", guidance=2.0)


def test_explicit_unload_rejects_active_generation():
    from yue2_mlx import pipeline

    pipeline._LOCK.acquire()
    try:
        with pytest.raises(RuntimeError, match="active"):
            pipeline.unload()
    finally:
        pipeline._LOCK.release()
