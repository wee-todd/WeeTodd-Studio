"""Native acoustic model contracts and independent numerical operator checks."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).parents[1] / "src/wee_todd_mlx/audio_analysis/ctc_model.py"


def adapter():
    assert MODULE.exists(), "native acoustic adapter must exist"
    spec = importlib.util.spec_from_file_location("ctc_under_test", MODULE)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_chunk_plan_covers_every_frame_once_on_global_grid():
    model = adapter()
    for size in (400, 16000, 480001, 960000, 2100123):
        chunks = model.chunk_plan(size)
        frames = []
        for start, end, keep_start, keep_end in chunks:
            assert start % 320 == 0
            assert end - start <= 480000
            assert 0 <= keep_start < keep_end <= model.output_frames(end - start)
            frames.extend(range(start // 320 + keep_start, start // 320 + keep_end))
        assert frames == list(range(model.output_frames(size)))


def test_input_invalid_before_model_loading(tmp_path):
    model = adapter()
    for values in (np.zeros((2, 500)), np.array([np.nan] * 500), np.zeros(399)):
        with pytest.raises(ValueError):
            model.ctc_emissions(values, tmp_path)


def test_normalization_uses_population_variance_and_finite_silence():
    model = adapter()
    x = np.arange(10, dtype=np.float32)
    np.testing.assert_allclose(model.normalize_audio(x), (x - x.mean()) / np.sqrt(x.var() + 1e-7))
    np.testing.assert_array_equal(model.normalize_audio(np.zeros(400)), np.zeros(400))


def test_mlx_operators_match_numpy_math():
    mx = pytest.importorskip("mlx.core")
    model = adapter()
    rng = np.random.default_rng(123)
    x = rng.normal(size=(1, 9, 4)).astype(np.float32)
    scale = rng.normal(size=4).astype(np.float32)
    bias = rng.normal(size=4).astype(np.float32)
    expected = (x - x.mean(-1, keepdims=True)) / np.sqrt(
        x.var(-1, keepdims=True) + 1e-5
    ) * scale + bias
    actual = model.layer_norm(mx.array(x), mx.array(scale), mx.array(bias), 1e-5)
    np.testing.assert_allclose(np.array(actual), expected, atol=2e-6)
    q, k, v = [rng.normal(size=(1, 2, 5, 3)).astype(np.float32) for _ in range(3)]
    scores = q @ k.swapaxes(-1, -2) / np.sqrt(3)
    scores = np.exp(scores - scores.max(-1, keepdims=True))
    scores /= scores.sum(-1, keepdims=True)
    actual = model.attention(mx.array(q), mx.array(k), mx.array(v))
    np.testing.assert_allclose(np.array(actual), scores @ v, atol=2e-6)


def test_config_rejects_other_architectures():
    model = adapter()
    with pytest.raises(ValueError, match="wav2vec2-base-960h"):
        model.validate_config({"model_type": "hubert"})


def test_checkpoint_integrity_rejects_unpinned_assets_before_mlx(tmp_path):
    model = adapter()
    for name in ("config.json", "vocab.json", "model.safetensors"):
        (tmp_path / name).write_bytes(b"wrong model")
    with pytest.raises(ValueError, match="integrity"):
        model.verify_assets(tmp_path)


def test_operation_releases_model_on_success_and_cancellation(tmp_path, monkeypatch):
    import json

    pytest.importorskip("mlx.core")
    model = adapter()
    (tmp_path / "config.json").write_text("{}")
    vocabulary = {"<pad>": 0, **{str(i): i for i in range(1, 32)}}
    (tmp_path / "vocab.json").write_text(json.dumps(vocabulary))
    monkeypatch.setattr(model, "validate_config", lambda config: None)
    monkeypatch.setattr(model, "verify_assets", lambda directory, check=None: None)
    closed = []

    class FakeModel:
        def __init__(self, directory, config):
            pass

        def __call__(self, samples, check):
            check()
            return np.full((model.output_frames(len(samples)), 32), -np.log(32), dtype=np.float32)

        def close(self):
            closed.append(True)

    monkeypatch.setattr(model, "_AcousticModel", FakeModel)
    result = model.ctc_emissions(np.zeros(490000), tmp_path)
    assert result["log_probs"].shape == (model.output_frames(490000), 32)
    assert result["log_probs"].dtype == np.float32
    assert result["frame_seconds"] == 0.02
    assert result["offset_seconds"] == 0.0125
    assert closed == [True]
    calls = []

    def cancel_inside_inference():
        calls.append(True)
        if len(calls) == 3:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        model.ctc_emissions(np.zeros(16000), tmp_path, cancel_inside_inference)
    assert closed == [True, True]


def test_failure_drops_tensor_locals_even_while_caller_holds_traceback(tmp_path, monkeypatch):
    import json

    mx = pytest.importorskip("mlx.core")
    model = adapter()
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "vocab.json").write_text(
        json.dumps({"<pad>": 0, **{str(i): i for i in range(1, 32)}})
    )
    monkeypatch.setattr(model, "validate_config", lambda config: None)
    monkeypatch.setattr(model, "verify_assets", lambda directory, check=None: None)

    class FailedModel:
        def __init__(self, *args):
            pass

        def __call__(self, *args):
            activation = mx.ones((1000000,))
            mx.eval(activation)
            raise RuntimeError("model failure")

        def close(self):
            pass

    monkeypatch.setattr(model, "_AcousticModel", FailedModel)
    before = mx.get_active_memory()
    try:
        model.ctc_emissions(np.zeros(16000), tmp_path)
    except RuntimeError as error:
        assert str(error) == "model failure"
        assert mx.get_active_memory() <= before + 65536
    else:
        pytest.fail("expected failure")
