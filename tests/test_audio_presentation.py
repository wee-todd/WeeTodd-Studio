import numpy as np
import pytest


def test_features_are_silent_for_silence_and_frequency_selective():
    from wee_todd_mlx.audio_presentation import audio_features

    quiet = audio_features(np.zeros(16000, dtype=np.float32), 16000)
    assert not any(any(row) for row in quiet["spectrum"])
    t = np.arange(16000) / 16000
    low = audio_features(np.sin(2 * np.pi * 200 * t).astype("float32") * 0.2, 16000)
    high = audio_features(np.sin(2 * np.pi * 3000 * t).astype("float32") * 0.2, 16000)
    assert np.argmax(low["spectrum"][4]) < np.argmax(high["spectrum"][4])
    assert len(low["waveform"]) == 11


def test_caption_cues_keep_only_acoustically_timed_words():
    from wee_todd_mlx.audio_presentation import caption_cues

    words = [
        dict(text="Hello", startSeconds=0.2, endSeconds=0.5),
        dict(text="missing", startSeconds=None, endSeconds=None),
        dict(text="world", startSeconds=0.6, endSeconds=0.9),
    ]
    cues = caption_cues(words, 1)
    assert cues == [dict(start=0.2, end=0.9, text="Hello world")]


def test_mix_extent_includes_anchored_and_unanchored_audio():
    from wee_todd_mlx.audio_presentation import presentation_duration

    p = dict(clips=[dict(id="c", duration=5)], audio=[dict(start=0, duration=79)])
    assert presentation_duration(p) == 79
    p["audio"] = [dict(anchor=dict(clipID="c", offsetSeconds=3), duration=10)]
    assert presentation_duration(p) == 13
    p["audio"][0]["duration"] = float("nan")
    with pytest.raises(ValueError):
        presentation_duration(p)


def test_feature_cache_tracks_volume_and_silent_audio(tmp_path):
    import json

    from scipy.io import wavfile
    from studio_presentation import features

    audio = tmp_path / "tone.wav"
    wavfile.write(audio, 48000, (np.sin(np.arange(24000) * 0.05) * 0.2).astype("float32"))
    project = dict(
        clips=[],
        audioMixPolicy="studio-v1",
        audioTracks=[dict(id="v", role="voice")],
        audio=[
            dict(
                id="a",
                trackID="v",
                path=str(audio),
                start=0,
                duration=0.5,
                sourceIn=0,
                volume=1,
                fade=0,
            )
        ],
        audioPresentation=dict(source="mix"),
    )
    loud = features(dict(project=project), tmp_path / "features")
    project["audioTracks"][0]["muted"] = True
    quiet = features(dict(project=project), tmp_path / "features")
    assert loud["path"] != quiet["path"]
    assert np.max(json.loads(open(quiet["path"]).read())["spectrum"]) == 0
