import numpy as np

from wee_todd_mlx.audio_analysis.structure import editing_cues, structure_analysis


def test_bars_keep_meter_and_silent_audio_has_no_chorus():
    beats = [dict(timeSeconds=i * 0.5, confidence=0.9) for i in range(15)]
    down = [beats[i] for i in (0, 3, 7, 11)]
    result = structure_analysis(np.zeros(8 * 22050), beats=beats, downbeats=down)
    assert [bar["beatCount"] for bar in result["bars"]] == [3, 4, 4]
    assert not result["sections"]


def test_cues_snap_with_source_offset_and_prefer_downbeats():
    result = editing_cues(
        dict(
            beats=[dict(timeSeconds=2.11, confidence=0.8)],
            downbeats=[dict(timeSeconds=2.11, confidence=0.9)],
            sections=[],
            words=[],
        ),
        fps=24,
        source_start=2,
        duration=2,
    )
    assert len(result) == 1
    assert result[0]["kind"] == "downbeat"
    assert result[0]["frame"] == 3
    assert result[0]["timeSeconds"] == 2.11
    assert abs(result[0]["snapErrorSeconds"] - 0.015) < 0.00001


def test_unaligned_lyrics_never_imply_instrumental_break():
    result = structure_analysis(
        np.zeros(8 * 22050),
        beats=[],
        downbeats=[],
        words=[dict(startSeconds=None, endSeconds=None)],
    )
    assert result["vocalGaps"] == []


def test_instrumental_analysis_has_no_alignment_object():
    assert editing_cues(dict(alignment=None, beats=[], downbeats=[]), fps=24, duration=4) == []


def test_silence_inside_song_does_not_create_repeated_sections():
    t = np.arange(80 * 22050) / 22050
    audio = 0.1 * np.sin(2 * np.pi * 220 * t)
    audio[30 * 22050 : 50 * 22050] = 0
    result = structure_analysis(audio, beats=[], downbeats=[])
    assert not any(34 <= s["startSeconds"] <= 46 for s in result["sections"])


def test_planner_avoids_supported_word_interior_but_respects_manual_lock():
    from wee_todd_mlx.music_video import optimize_timing

    args = dict(
        total_frames=96,
        fps=24,
        minimum_frames=24,
        maximum_frames=72,
        preferred_frames=48,
        protected_ranges=[(43, 55)],
    )
    result = optimize_timing(**args)
    assert not 43 < result["clips"][1]["startFrame"] < 55
    locked = optimize_timing(**args, locked_frames=[48])
    assert locked["clips"][1]["startFrame"] == 48


def test_quick_preview_exposes_onset_editing_cues():
    result = editing_cues(
        {"cues": [dict(timeSeconds=1.5, kind="onset", strength=0.8)]}, fps=24, duration=4
    )
    assert result[0]["kind"] == "onset"
    assert result[0]["frame"] == 36
