"""Reverb contracts and numerical properties, without model weights."""

import importlib
import importlib.util

import numpy as np
import pytest
from scipy.signal import fftconvolve


def module():
    assert importlib.util.find_spec("wee_todd_mlx.audio_mix.reverb") is not None
    return importlib.import_module("wee_todd_mlx.audio_mix.reverb")


def test_validation_and_exact_bypass():
    r = module()
    assert r.validate_reverb(None) is None
    assert r.validate_reverb({"enabled": False}) is None
    assert r.validate_reverb({"mix": 0}) is None
    assert r.validate_reverb({}) == dict(
        enabled=True, preset="room", mix=0.18, decay=0.7, tone=0.45, preDelay=0.012
    )
    for preset, (decay, delay, tone) in r.PRESETS.items():
        assert r.validate_reverb({"preset": preset})["decay"] == decay
        assert r.validate_reverb({"preset": preset})["preDelay"] == delay
        assert r.validate_reverb({"preset": preset})["tone"] == tone
    source = np.random.default_rng(4).normal(size=(90000, 2)).astype("float32")
    target = np.zeros_like(source)
    r.add_reverb(source, target, None)
    np.testing.assert_array_equal(source, target)


@pytest.mark.parametrize(
    "value",
    [
        [],
        True,
        {"preset": "bath"},
        {"enabled": 1},
        {"mix": True},
        {"mix": "0.2"},
        {"mix": float("nan")},
        {"mix": float("inf")},
        {"decay": 0.19},
        {"decay": 6.1},
        {"preDelay": -0.01},
        {"preDelay": 0.101},
        {"tone": 1.1},
        {"enabled": False, "tone": "bad"},
        {"mix": 0, "preset": 8},
    ],
)
def test_invalid_settings(value):
    with pytest.raises(ValueError):
        module().validate_reverb(value)


def test_additive_nonmutating_deterministic_and_silence():
    r = module()
    source = np.zeros((20000, 2), dtype="float32")
    source[0, 0] = 1
    original = source.copy()
    a, b = np.zeros_like(source), np.ones_like(source)
    r.add_reverb(source, a, {}, sample_rate=8000)
    r.add_reverb(source, b, {}, sample_rate=8000)
    np.testing.assert_allclose(b, a + 1, atol=1e-7)
    np.testing.assert_array_equal(source, original)
    c = np.zeros_like(source)
    r.add_reverb(source, c, {}, sample_rate=8000)
    np.testing.assert_array_equal(a, c)
    assert a[0, 0] == np.float32(0.82)
    assert np.max(np.abs(a[:, 1])) > 0.001
    z = np.zeros_like(source)
    r.add_reverb(z, z.copy(), {}, sample_rate=8000)
    out = np.zeros_like(source)
    r.add_reverb(z, out, {}, sample_rate=8000)
    assert not out.any()


def test_block_boundaries_match_full_convolution_and_tail():
    r = module()
    rate = 8000
    source = np.zeros((150000, 2), dtype="float32")
    source[65530:65560] = np.random.default_rng(11).normal(size=(30, 2))
    settings = {"preset": "hall", "mix": 0.4, "decay": 1.2}
    ir = r.impulse_response(settings, sample_rate=rate)
    expected = source * 0.6
    for out in range(2):
        for inp in range(2):
            expected[:, out] += 0.4 * fftconvolve(source[:, inp], ir[:, out, inp])[: len(source)]
    actual = np.zeros_like(source)
    r.add_reverb(source, actual, settings, sample_rate=rate)
    np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=2e-5)
    assert np.linalg.norm(actual[66000:69000]) > 0.01


def test_decay_tone_energy_and_antiphase():
    r = module()
    rate = 8000

    def ir(**kwargs):
        return r.impulse_response(dict(mix=1, preDelay=0, **kwargs), sample_rate=rate)

    short, long = ir(decay=0.4), ir(decay=2)
    assert np.sum(long[4000:] ** 2) > np.sum(short[2000:] ** 2) * 10
    dark, bright = ir(tone=0), ir(tone=1)

    def high_ratio(x):
        spectrum = np.abs(np.fft.rfft(x[:, 0, 0])) ** 2
        return spectrum[len(spectrum) // 2 :].sum() / spectrum.sum()

    assert high_ratio(bright) > 2 * high_ratio(dark)
    for preset in r.PRESETS:
        extreme = r.impulse_response(
            {"preset": preset, "decay": 6, "tone": 0, "preDelay": 0.1}, sample_rate=rate
        )
        assert np.isfinite(extreme).all()
        np.testing.assert_allclose(
            np.sum(extreme.astype("float64") ** 2, axis=(0, 1)), 1, atol=2e-5
        )
    source = np.zeros((20000, 2), dtype="float32")
    source[0] = [1, -1]
    out = np.zeros_like(source)
    r.add_reverb(source, out, {"mix": 1}, sample_rate=rate)
    assert np.linalg.norm(out) > 1
    assert np.linalg.norm(out[:, 0] - out[:, 1]) > 1


def test_cancel_ir_and_blocks():
    r = module()
    source = np.zeros((150000, 2), dtype="float32")
    with pytest.raises(InterruptedError):
        r.add_reverb(source, source.copy(), {}, cancelled=lambda: True)
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        return calls == 3

    with pytest.raises(InterruptedError):
        r.impulse_response({}, sample_rate=8000, cancelled=cancel)
    calls = 0
    with pytest.raises(InterruptedError):
        r.add_reverb(source, source.copy(), None, cancelled=cancel)


def test_validation_import_stays_lightweight():
    import subprocess
    import sys

    script = """
import sys
from wee_todd_mlx.audio_mix.reverb import validate_reverb
validate_reverb({})
assert 'numpy' not in sys.modules
assert 'scipy' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_memmap_bus_and_input_separation(tmp_path):
    r = module()
    source = np.memmap(tmp_path / "source", dtype="float32", mode="w+", shape=(80000, 2))
    destination = np.memmap(tmp_path / "dest", dtype="float32", mode="w+", shape=source.shape)
    source[:] = 0
    source[65535, 0] = 1
    destination[:] = 0.25
    r.add_reverb(source, destination, {}, sample_rate=8000)
    assert source[65535, 0] == 1 and np.count_nonzero(source) == 1
    assert np.max(np.abs(destination[66000:, 1] - 0.25)) > 0.00001
    with pytest.raises(ValueError, match="separate"):
        r.add_reverb(source, source, {})


def test_frequency_dependent_decay_and_dc_rejection():
    r = module()
    from scipy.signal import butter, sosfilt

    rate = 16000
    response = r.impulse_response({"decay": 2, "tone": 0.35, "preDelay": 0}, sample_rate=rate)[
        :, 0, 0
    ].astype("float64")
    low = sosfilt(butter(6, [150, 700], fs=rate, btype="bandpass", output="sos"), response)
    high = sosfilt(butter(6, 3500, fs=rate, btype="highpass", output="sos"), response)
    early, late = slice(1000, 5000), slice(12000, 18000)
    assert np.sum(high[late] ** 2) / np.sum(high[early] ** 2) < 0.15 * (
        np.sum(low[late] ** 2) / np.sum(low[early] ** 2)
    )
    spectrum = np.abs(np.fft.rfft(response)) ** 2
    assert spectrum[0] < np.mean(spectrum[100:1000]) * 0.001
