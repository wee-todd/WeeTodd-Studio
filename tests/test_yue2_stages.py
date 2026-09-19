"""Artifact stage contracts and lifetime regressions; no real checkpoints required."""

import gc
import json
import weakref

import numpy as np
import pytest

from yue2_mlx import pipeline


def request(**kwargs):
    return dict(model_path="/unused", style="piano", lyrics="hello", **kwargs)


class Tokenizer:
    def __init__(self, *args):
        pass

    def encode(self, text):
        return [11, 12]

    def decode(self, tokens):
        return "score"


@pytest.fixture
def fake_inputs(monkeypatch):
    from yue2_mlx import tokenizer

    monkeypatch.setattr(tokenizer, "Tokenizer", Tokenizer)
    info = dict(
        files={"tokenizer": "unused"},
        config={"max_position_embeddings": 24576},
        precision="8bit",
        layout="merged-mlx",
        identities={},
        generation_config={},
    )
    monkeypatch.setattr(pipeline, "inspect_checkpoint", lambda *args: info)
    from yue2_mlx import artifacts

    monkeypatch.setattr(artifacts, "inspect_checkpoint", lambda *args: info)
    return info


@pytest.mark.parametrize("error_type", [InterruptedError, RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("stage", ["load", "ar", "decode"])
def test_retained_exception_does_not_retain_weighted_stage(
    monkeypatch, tmp_path, fake_inputs, error_type, stage
):
    import mlx.core as mx

    from yue2_mlx import acoustic, sampling

    references = []

    class Weighted:
        def __init__(self):
            references.append(weakref.ref(self))

        def decode_tiled(self, *args, **kwargs):
            cache = Weighted()
            assert cache is not None
            try:
                raise ValueError("inner")
            except ValueError as error:
                raise error_type("decoder failure") from error

    def load(*args):
        model = Weighted()
        if stage == "load":
            raise error_type("load failure")
        return model

    def tokens(model, *args, **kwargs):
        if stage == "ar":
            cache = Weighted()
            assert cache is not None
            raise error_type("AR failure")
        return [151853], True

    monkeypatch.setattr(pipeline, "load_model", load)
    monkeypatch.setattr(pipeline, "load_vae", load)
    monkeypatch.setattr(sampling, "generate_tokens", tokens)
    monkeypatch.setattr(acoustic, "synthesize", lambda *args, **kwargs: mx.zeros((1, 64)))
    saved = None
    try:
        pipeline.generate(request(cot="off"), tmp_path / "failed")
    except error_type as error:
        saved = error
    assert saved is not None
    gc.collect()
    assert references and all(ref() is None for ref in references)


def test_generated_score_reserves_semantic_budget_before_load(monkeypatch, tmp_path, fake_inputs):
    called = []
    monkeypatch.setattr(pipeline, "load_model", lambda *args: called.append(True))
    with pytest.raises(ValueError, match="context|budget"):
        pipeline.generate(request(semantic_sampling={"max_tokens": 24576}), tmp_path / "bad")
    assert not called


def test_pcm_headroom_is_global_and_only_when_needed(tmp_path):
    import wave

    audio = np.array([[2.0, -1.0], [0.5, 0.25]], dtype=np.float32)
    result = pipeline.write_master(tmp_path / "scaled.wav", audio)
    assert result["gain"] == 0.5 and result["peak"] == 2 and result["output_peak"] == 1
    with wave.open(str(tmp_path / "scaled.wav"), "rb") as stream:
        values = np.frombuffer(stream.readframes(2), dtype="<i4").reshape(2, 2) / 2147483647
    np.testing.assert_allclose(values, audio * 0.5, atol=1e-9)
    unchanged = pipeline.write_master(tmp_path / "unchanged.wav", audio * 0.25)
    assert unchanged["gain"] == 1


def artifact_bundle(tmp_path):
    from yue2_mlx.artifacts import seal
    from yue2_mlx.config import validate_request
    from yue2_mlx.protocol import prefix_tokens

    folder = tmp_path / "source"
    folder.mkdir()
    req = validate_request(request(cot="off"))
    tokens = dict(
        abc=[],
        prefix=prefix_tokens(req, Tokenizer()),
        negative_prefix=[151643, 11, 12, 151851],
        codec=[0, 1],
        semantic=[151853, 151854],
        truncated={"abc": False, "semantic": True},
    )
    for filename, value in [
        ("request.json", req),
        ("tokens.json", tokens),
        (
            "effective.json",
            dict(
                models={},
                protocol="yue2-native-v1",
                precision="8bit",
                layout="merged-mlx",
                steps=req["steps"],
                guidance=req["cfg_scale"],
                memory_mode=req["memory_mode"],
                sampling={"abc": req["abc_sampling"], "semantic": req["semantic_sampling"]},
            ),
        ),
        ("result.json", dict(stage="generate", request=req, truncated=tokens["truncated"])),
    ]:
        (folder / filename).write_text(json.dumps(value))
    (folder / "score.abc").write_text("score")
    np.save(folder / "noise.npy", np.zeros((2, 64), dtype=np.float32))
    np.save(folder / "latents.npy", np.ones((2, 64), dtype=np.float32))
    seal(folder, "generate")
    return folder


def test_artifact_hash_rejects_tampering_before_replay(tmp_path, monkeypatch, fake_inputs):
    from yue2_mlx.artifacts import verify

    folder = artifact_bundle(tmp_path)
    (folder / "latents.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        verify(folder)


def test_artifact_semantics_rejects_resealed_mismatch(tmp_path, monkeypatch, fake_inputs):
    from yue2_mlx.artifacts import seal, verify

    folder = artifact_bundle(tmp_path)
    tokens = json.loads((folder / "tokens.json").read_text())
    tokens["codec"] = [9, 9]
    (folder / "tokens.json").write_text(json.dumps(tokens))
    seal(folder, "generate")
    with pytest.raises(ValueError, match="semantic|codec"):
        verify(folder)


def test_plan_only_returns_text_without_loading_decoder(tmp_path, monkeypatch, fake_inputs):
    from yue2_mlx import sampling

    monkeypatch.setattr(pipeline, "load_model", lambda *_: object())
    monkeypatch.setattr(pipeline, "load_vae", lambda *_: pytest.fail("plan loaded VAE"))
    monkeypatch.setattr(sampling, "generate_tokens", lambda *args, **kwargs: ([11, 12], False))
    result = pipeline.plan(request(), tmp_path / "plan")
    assert result["stage"] == "plan" and result["abc"] == "score" and "audio" not in result
    assert (tmp_path / "plan" / "manifest.json").is_file()


def test_redecode_reuses_latents_without_transformer(tmp_path, monkeypatch, fake_inputs):
    import mlx.core as mx

    source = artifact_bundle(tmp_path)
    monkeypatch.setattr(pipeline, "load_model", lambda *_: pytest.fail("decode loaded transformer"))

    class Decoder:
        def decode_tiled(self, latents, **kwargs):
            np.testing.assert_array_equal(np.array(latents), 1.0)
            return mx.ones((2 * 1920 - 64, 2)) * 0.5

    monkeypatch.setattr(pipeline, "load_vae", lambda *_: Decoder())
    result = pipeline.decode_latents(source, tmp_path / "decoded")
    assert result["duration"] == (2 * 1920 - 64) / 48000 and result["gain"] == 1
    assert result["stage"] == "decode_latents"


def test_resynthesize_default_reuses_noise_and_steps_override(tmp_path, monkeypatch, fake_inputs):
    import mlx.core as mx

    from yue2_mlx import acoustic

    source = artifact_bundle(tmp_path)
    monkeypatch.setattr(pipeline, "load_model", lambda *_: object())

    def synthesize(model, prefix, codec, noise, steps, **kwargs):
        assert steps == 3
        np.testing.assert_array_equal(np.array(noise), 0.0)
        return mx.ones((2, 64))

    monkeypatch.setattr(acoustic, "synthesize", synthesize)

    class Decoder:
        def decode_tiled(self, latents, **kwargs):
            return mx.ones((2 * 1920 - 64, 2)) * 0.5

    monkeypatch.setattr(pipeline, "load_vae", lambda *_: Decoder())
    result = pipeline.resynthesize(source, tmp_path / "replayed", steps=3)
    assert result["request"]["steps"] == 3 and result["stage"] == "resynthesize"
    np.testing.assert_array_equal(np.load(tmp_path / "replayed" / "noise.npy"), 0.0)


def test_checkpoint_content_mismatch_prevents_loading(tmp_path, monkeypatch, fake_inputs):
    from yue2_mlx import artifacts

    source = artifact_bundle(tmp_path)
    monkeypatch.setattr(
        artifacts,
        "checkpoint_fingerprints",
        lambda *args: {"model": {"sha256": "a" * 64, "bytes": 7}},
    )
    monkeypatch.setattr(
        pipeline, "load_model", lambda *_: pytest.fail("loaded unverified checkpoint")
    )
    with pytest.raises(ValueError, match="provenance|mismatch"):
        pipeline.resynthesize(source, tmp_path / "bad-model")


def test_copy_verifies_source_did_not_change_after_verification(tmp_path, monkeypatch, fake_inputs):
    from yue2_mlx.artifacts import copy_verified, verify

    source = artifact_bundle(tmp_path)
    checked = verify(source)
    np.save(source / "latents.npy", np.zeros((2, 64), dtype=np.float32))
    with pytest.raises(ValueError, match="integrity"):
        copy_verified(checked, "latents.npy", tmp_path / "copied.npy")


def test_resynthesize_seed_override_records_original_semantic_request(
    tmp_path, monkeypatch, fake_inputs
):
    import mlx.core as mx

    from yue2_mlx import acoustic

    source = artifact_bundle(tmp_path)
    monkeypatch.setattr(pipeline, "load_model", lambda *_: object())

    def synthesize(model, prefix, codec, noise, steps, **kwargs):
        assert np.any(np.array(noise) != 0)
        return mx.zeros((2, 64))

    monkeypatch.setattr(acoustic, "synthesize", synthesize)

    class Decoder:
        def decode_tiled(self, latents, **kwargs):
            return mx.zeros((2 * 1920 - 64, 2))

    monkeypatch.setattr(pipeline, "load_vae", lambda *_: Decoder())
    result = pipeline.resynthesize(source, tmp_path / "seed-override", seed=17)
    assert result["request"]["seed"] == 17
    effective = json.loads((tmp_path / "seed-override" / "effective.json").read_text())
    assert effective["semantic_request"]["seed"] == 831001


@pytest.mark.parametrize(
    "filename,field,value",
    [
        ("result.json", "stage", "plan"),
        ("result.json", "truncated", {"abc": True, "semantic": False}),
        ("effective.json", "precision", "bf16"),
        ("effective.json", "sampling", {}),
    ],
)
def test_resealed_metadata_mismatch_is_rejected(
    tmp_path, monkeypatch, fake_inputs, filename, field, value
):
    from yue2_mlx.artifacts import seal, verify

    source = artifact_bundle(tmp_path)
    payload = json.loads((source / filename).read_text())
    payload[field] = value
    (source / filename).write_text(json.dumps(payload))
    seal(source, "generate")
    with pytest.raises(ValueError, match="metadata|sampling|precision|stage|truncation"):
        verify(source)
