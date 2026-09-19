import copy

import numpy as np
import pytest
from scipy.io import wavfile


def fixture(tmp_path):
    voice = tmp_path / "voice.wav"
    music = tmp_path / "music.wav"
    t = np.arange(24000) / 48000
    wavfile.write(voice, 48000, (0.1 * np.sin(2 * np.pi * 240 * t)).astype("float32"))
    wavfile.write(
        music,
        48000,
        np.column_stack(
            [0.1 * np.sin(2 * np.pi * 400 * t), 0.1 * np.sin(2 * np.pi * 600 * t)]
        ).astype("float32"),
    )
    return dict(
        audioMixPolicy="studio-v1",
        clips=[dict(id="c", duration=0.5, sourcePath="", volume=1)],
        audioTracks=[dict(id="v", role="voice"), dict(id="m", role="music")],
        audio=[
            dict(
                id="v1",
                trackID="v",
                path=str(voice),
                start=0,
                sourceIn=0,
                duration=0.5,
                volume=1,
                fade=0,
            ),
            dict(
                id="m1",
                trackID="m",
                path=str(music),
                start=0,
                sourceIn=0,
                duration=0.5,
                volume=1,
                fade=0,
            ),
        ],
    )


def test_driver_modes_gain_pan_and_exact_frames(tmp_path):
    from wee_todd_mlx.audio_mix import compile_mix, render_mix

    p = fixture(tmp_path)
    p["audioTracks"][0].update(gainDb=-6, pan=-1)
    plan = compile_mix(p, start=0, duration=0.5, purpose="driver", selection={"mode": "voice"})
    r = render_mix(plan, tmp_path / "cache", {})
    rate, a = wavfile.read(r["path"])
    assert rate == 48000 and a.shape == (24000, 2)
    assert np.max(abs(a[:, 1])) < 1e-6
    assert np.max(abs(a[:, 0])) == pytest.approx(0.1 * 10 ** (-6 / 20), abs=1e-4)
    both = render_mix(
        compile_mix(
            p, start=0, duration=0.5, purpose="driver", selection={"mode": "voiceAndMusic"}
        ),
        tmp_path / "cache",
        {},
    )
    assert both["mix_key"] != r["mix_key"]


def test_muted_missing_file_not_opened_and_solo_does_not_change_driver(tmp_path):
    from wee_todd_mlx.audio_mix import compile_mix, render_mix

    p = fixture(tmp_path)
    p["audioTracks"][1].update(muted=True, solo=True)
    p["audio"][1]["path"] = "/missing"
    r = render_mix(
        compile_mix(p, start=0, duration=0.5, purpose="driver", selection={"mode": "voice"}),
        tmp_path / "cache",
        {},
    )
    assert r["peak"] > 0
    with pytest.raises(ValueError, match="silent|audible|selected"):
        render_mix(
            compile_mix(p, start=0, duration=0.5, purpose="driver", selection={"mode": "music"}),
            tmp_path / "cache",
            {},
        )


def test_cache_corruption_repair_and_anchor(tmp_path):
    from wee_todd_mlx.audio_mix import compile_mix, render_mix

    p = fixture(tmp_path)
    p["clips"].insert(0, dict(id="first", duration=0.25, sourcePath="", volume=1))
    p["audio"][0]["anchor"] = {"clipID": "c", "offsetSeconds": 0}
    plan = compile_mix(p, start=0.25, duration=0.5, purpose="driver", selection={"mode": "voice"})
    assert plan["inputs"][0]["start"] == 0.25
    r = render_mix(plan, tmp_path / "cache", {})
    expected = open(r["path"], "rb").read()
    open(r["path"], "wb").write(b"broken")
    r2 = render_mix(plan, tmp_path / "cache", {})
    assert open(r2["path"], "rb").read() == expected
    p2 = copy.deepcopy(p)
    p2["audioTracks"][0]["pan"] = float("nan")
    with pytest.raises(ValueError):
        compile_mix(p2, start=0, duration=0.5, purpose="preview")


def test_preview_export_pcm_identity_and_ducking(tmp_path):
    from wee_todd_mlx.audio_mix import compile_mix, render_mix

    p = fixture(tmp_path)
    preview = render_mix(compile_mix(p, purpose="preview"), tmp_path / "cache", {})
    export = render_mix(compile_mix(p, purpose="export"), tmp_path / "cache", {})
    assert preview["mix_key"] == export["mix_key"]
    p["audioTracks"][1]["ducking"] = dict(amountDb=12, thresholdDb=-36, attack=0.02, release=0.25)
    duck = render_mix(compile_mix(p, purpose="export"), tmp_path / "cache", {})
    # Compare music difference using a voice-only bus; ducking must lower the music energy.
    voice = render_mix(
        compile_mix(p, purpose="driver", selection={"mode": "voice"}), tmp_path / "cache", {}
    )
    _, a = wavfile.read(export["path"])
    _, b = wavfile.read(duck["path"])
    _, v = wavfile.read(voice["path"])
    assert np.mean((b[10000:] - v[10000:]) ** 2) < 0.2 * np.mean((a[10000:] - v[10000:]) ** 2)
    music = render_mix(
        compile_mix(p, purpose="driver", selection={"mode": "music"}), tmp_path / "cache", {}
    )
    p["audioTracks"][1]["ducking"] = None
    dry = render_mix(
        compile_mix(p, purpose="driver", selection={"mode": "music"}), tmp_path / "cache", {}
    )
    np.testing.assert_array_equal(wavfile.read(music["path"])[1], wavfile.read(dry["path"])[1])


def test_driver_subrange_matches_full_mix_with_limiter_history(tmp_path):
    from wee_todd_mlx.audio_mix import compile_mix, render_mix

    p = fixture(tmp_path)
    full = render_mix(
        compile_mix(p, purpose="driver", selection={"mode": "voiceAndMusic"}),
        tmp_path / "cache",
        {},
    )
    part = render_mix(
        compile_mix(
            p, start=0.25, duration=0.25, purpose="driver", selection={"mode": "voiceAndMusic"}
        ),
        tmp_path / "cache",
        {},
    )
    np.testing.assert_allclose(
        wavfile.read(part["path"])[1], wavfile.read(full["path"])[1][12000:], atol=1e-6
    )


def test_gain_edit_reuses_decoded_audio(tmp_path, monkeypatch):
    import wee_todd_mlx.audio_mix.render as module
    from wee_todd_mlx.audio_mix import compile_mix, render_mix

    p = fixture(tmp_path)
    original = module.run_media
    decodes = []

    def record(args, **kwargs):
        if "-map" in args and "0:a:0" in args:
            decodes.append(args)
        return original(args, **kwargs)

    monkeypatch.setattr(module, "run_media", record)
    render_mix(compile_mix(p), tmp_path / "cache", {})
    first = len(decodes)
    p["audioTracks"][0]["gainDb"] = -3
    render_mix(compile_mix(p), tmp_path / "cache", {})
    assert first == 2
    assert len(decodes) == first


def test_limiter_future_peak_matches_continuous_mix(tmp_path):
    from wee_todd_mlx.audio_mix import compile_mix, render_mix

    p = fixture(tmp_path)
    x = np.full(24000, 0.2, np.float32)
    x[12000:13000] = 2
    wavfile.write(p["audio"][0]["path"], 48000, x)
    p["audioTracks"][0]["pan"] = -1
    full = render_mix(
        compile_mix(p, purpose="driver", selection={"mode": "voice"}), tmp_path / "cache", {}
    )
    part = render_mix(
        compile_mix(p, duration=0.25, purpose="driver", selection={"mode": "voice"}),
        tmp_path / "cache",
        {},
    )
    np.testing.assert_allclose(
        wavfile.read(part["path"])[1], wavfile.read(full["path"])[1][:12000], atol=1e-6
    )


def test_legacy_policy_retains_limiter_makeup_and_fade_cap(tmp_path):
    from wee_todd_mlx.audio_mix import compile_mix, render_mix

    p = fixture(tmp_path)
    p.pop("audioMixPolicy", None)
    p["audio"][0]["fade"] = 0.4
    plan = compile_mix(p)
    assert plan["policy"] == "legacy-v1"
    assert plan["inputs"][0]["fade_in"] == 0.25
    p["audio"][0]["fade"] = 0
    old = render_mix(compile_mix(p), tmp_path / "cache", {})
    p["audioMixPolicy"] = "studio-v1"
    new = render_mix(compile_mix(p), tmp_path / "cache", {})
    np.testing.assert_allclose(
        wavfile.read(old["path"])[1], wavfile.read(new["path"])[1] / 0.95, atol=1e-6
    )


def test_legacy_limiter_uses_extra_region_bus_not_source_clip_count(tmp_path):
    from wee_todd_mlx.audio_mix import compile_mix, render_mix

    p = fixture(tmp_path)
    p.pop("audioMixPolicy")
    p["audio"] = p["audio"][:1]
    old = render_mix(compile_mix(p), tmp_path / "cache", {})
    p["audioMixPolicy"] = "studio-v1"
    new = render_mix(compile_mix(p), tmp_path / "cache", {})
    np.testing.assert_allclose(
        wavfile.read(old["path"])[1], wavfile.read(new["path"])[1] / 0.95, atol=1e-6
    )
    p.pop("audioMixPolicy")
    p["clips"] = [
        dict(id=str(i), duration=0.25, sourcePath=p["audio"][0]["path"]) for i in range(2)
    ]
    p["audio"] = []
    old = render_mix(compile_mix(p), tmp_path / "cache", {})
    assert old["peak"] == pytest.approx(0.1 / np.sqrt(2), abs=1e-5)


@pytest.mark.parametrize("split", [0.05, 0.45])
def test_split_region_preserves_envelope_phase(tmp_path, split):
    from wee_todd_mlx.audio_mix import compile_mix, render_mix

    p = fixture(tmp_path)
    p["audio"] = p["audio"][:1]
    p["audio"][0].update(fadeIn=0.2, fadeOut=0.2, fadeCurve="equalPower")
    original = render_mix(compile_mix(p), tmp_path / "cache", {})
    first = copy.deepcopy(p["audio"][0])
    tail = copy.deepcopy(first)
    window = dict(offset=0, duration=0.5, fadeIn=0.2, fadeOut=0.2, curve="equalPower")
    first.update(duration=split, fadeOut=0, envelope=window)
    tail.update(
        id="tail",
        sourceIn=split,
        start=split,
        duration=0.5 - split,
        fadeIn=0,
        envelope=dict(window, offset=split),
    )
    p["audio"] = [first, tail]
    sliced = render_mix(compile_mix(p), tmp_path / "cache", {})
    np.testing.assert_allclose(
        wavfile.read(original["path"])[1], wavfile.read(sliced["path"])[1], atol=1e-6
    )


def test_preview_eviction_keeps_leased_audio(tmp_path):
    import fcntl
    import os

    from wee_todd_mlx.audio_mix.cache import prune_previews

    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    for folder in [first, second]:
        (folder / "mix.wav").write_bytes(b"1234")
        (folder / "mix.json").write_text("{}")
        os.utime(folder / "mix.wav", (1, 1))
    with (first / "lease.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_SH)
        prune_previews(tmp_path, budget=0, grace=0)
        assert (first / "mix.wav").exists()
        assert not (second / "mix.wav").exists()
    prune_previews(tmp_path, budget=0, grace=0)
    assert not (first / "mix.wav").exists()
