import copy

import numpy as np
import pytest
from scipy.io import wavfile

from wee_todd_mlx.audio_mix import compile_mix, render_mix


def project(tmp_path):
    source = tmp_path / "pulse.wav"
    audio = np.zeros(4800, dtype="float32")
    audio[0] = 0.1
    wavfile.write(source, 48000, audio)
    return dict(
        audioMixPolicy="studio-v1",
        clips=[dict(id="c", duration=2, sourcePath="")],
        audioTracks=[dict(id="v", role="voice", reverb=dict(preset="hall", mix=0.3, decay=1))],
        audio=[dict(id="a", trackID="v", path=str(source), duration=0.1, volume=1, fade=0)],
    )


def test_reverb_survives_region_end_and_matches_driver_crop(tmp_path):
    p = project(tmp_path)
    full = render_mix(compile_mix(p), tmp_path / "cache", {})
    _, audio = wavfile.read(full["path"])
    assert np.max(abs(audio[7000:20000])) > 1e-6
    part = render_mix(
        compile_mix(p, start=0.2, duration=0.7, purpose="driver", selection={"mode": "voice"}),
        tmp_path / "cache",
        {},
    )
    np.testing.assert_allclose(wavfile.read(part["path"])[1], audio[9600:43200], atol=1e-7)
    export = render_mix(compile_mix(p, purpose="export"), tmp_path / "cache", {})
    assert full["mix_key"] == export["mix_key"]


def test_disabled_or_zero_amount_is_exact_dry_bypass(tmp_path):
    p = project(tmp_path)
    del p["audioTracks"][0]["reverb"]
    dry = compile_mix(p)
    for effect in [dict(enabled=False, preset="hall"), dict(mix=0)]:
        p["audioTracks"][0]["reverb"] = effect
        assert compile_mix(p) == dry
    p["audioTracks"][0]["reverb"]["mix"] = 0.2
    assert compile_mix(p) != dry


def test_reverb_is_applied_once_to_track_bus_and_keeps_split_pcm(tmp_path):
    p = project(tmp_path)
    source = p["audio"][0]["path"]
    rng = np.random.default_rng(4)
    wavfile.write(source, 48000, rng.normal(0, 0.015, 48000).astype("float32"))
    p["audio"][0]["duration"] = 1
    full = render_mix(compile_mix(p), tmp_path / "cache", {})
    first = p["audio"][0]
    first["duration"] = 0.4
    second = dict(first, id="b", start=0.4, sourceIn=0.4, duration=0.6)
    p["audio"].append(second)
    split = render_mix(compile_mix(p), tmp_path / "cache", {})
    np.testing.assert_allclose(
        wavfile.read(full["path"])[1], wavfile.read(split["path"])[1], atol=2e-7
    )


def test_reverb_validation_and_muting(tmp_path):
    p = project(tmp_path)
    for change in [
        dict(decay=float("nan")),
        dict(decay=7),
        dict(mix=-1),
        dict(preDelay=0.2),
        dict(tone=2),
        dict(preset="unknown"),
    ]:
        invalid = copy.deepcopy(p)
        invalid["audioTracks"][0]["reverb"].update(change)
        with pytest.raises(ValueError):
            compile_mix(invalid)
    p["audioTracks"][0]["muted"] = True
    assert compile_mix(p)["inputs"] == []


def test_music_ducking_reduces_existing_reverb_tail(tmp_path):
    p = project(tmp_path)
    p["audioTracks"][0]["role"] = "music"
    p["audioTracks"][0]["reverb"].update(mix=1, decay=2)
    p["audioTracks"].append(dict(id="voice", role="voice"))
    voice_source = tmp_path / "voice.wav"
    wavfile.write(voice_source, 48000, np.full(24000, 0.3, dtype="float32"))
    # Opposite pan isolates the music tail in the left channel; sidechain is pre-pan.
    p["audioTracks"][1]["pan"] = 1
    p["audio"].append(
        dict(
            id="voice-region",
            trackID="voice",
            path=str(voice_source),
            start=0.2,
            duration=0.5,
            volume=1,
            fade=0,
        )
    )
    plain = render_mix(compile_mix(p), tmp_path / "cache", {})
    p["audioTracks"][0]["ducking"] = dict(amountDb=18, thresholdDb=-40, attack=0.001, release=0.1)
    ducked = render_mix(compile_mix(p), tmp_path / "cache", {})
    dry = wavfile.read(plain["path"])[1][16000:24000, 0]
    wet = wavfile.read(ducked["path"])[1][16000:24000, 0]
    assert np.linalg.norm(wet) < np.linalg.norm(dry) * 0.2


def test_final_pcm_verification_stays_bounded(tmp_path, monkeypatch):
    p = project(tmp_path)
    del p["audioTracks"][0]["reverb"]
    finite = np.isfinite
    lengths = []

    def checked_finite(value, *args, **kwargs):
        if isinstance(value, np.ndarray) and value.ndim == 2:
            lengths.append(len(value))
        return finite(value, *args, **kwargs)

    monkeypatch.setattr(np, "isfinite", checked_finite)
    render_mix(compile_mix(p), tmp_path / "cache", {})
    assert lengths and max(lengths) <= 48000
