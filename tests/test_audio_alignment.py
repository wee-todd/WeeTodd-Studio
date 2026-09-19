import numpy as np
import pytest

from wee_todd_mlx.audio_analysis.alignment import align_lyrics, forced_token_spans

VOCAB = {"<pad>": 0, "|": 1, "A": 2, "B": 3, "C": 4}


def emissions(tokens):
    scores = np.full((len(tokens), len(VOCAB)), -12.0, dtype=np.float32)
    scores[np.arange(len(tokens)), tokens] = -0.001
    return scores


def test_missing_lyric_has_no_fabricated_timestamp_and_repeat_is_flagged():
    result = align_lyrics(emissions([0, 2, 0, 1, 3, 0, 1, 3, 0]), VOCAB, "[Chorus]\nA C B")
    assert result["words"][1]["startSeconds"] is None
    assert "possible_omission" in result["words"][1]["flags"]
    assert result["lines"][0]["sectionLabel"] == "Chorus"
    assert any("possible_repeat" in word["flags"] for word in result["extraWords"])
    assert result["words"][0]["endSeconds"] <= result["words"][2]["startSeconds"]


def test_unknown_lyrics_return_recognized_word_times_and_pauses():
    tokens = [2, 0, 1] + [0] * 30 + [3, 0]
    result = align_lyrics(emissions(tokens), VOCAB, "")
    assert [w["text"] for w in result["words"]] == ["A", "B"]
    assert result["pauses"][0]["durationSeconds"] > 0.5
    assert result["status"] == "transcribed_unreviewed"


def test_unsupported_lyric_does_not_silently_transliterate():
    result = align_lyrics(emissions([2, 0]), VOCAB, "你好")
    assert result["words"][0]["startSeconds"] is None
    assert "unsupported_characters" in result["words"][0]["flags"]


def test_ctc_repeated_token_requires_blank():
    spans = forced_token_spans(emissions([2, 0, 2]), [2, 2])
    assert spans == [(0, 1), (2, 3)]
    with pytest.raises(ValueError, match="align"):
        forced_token_spans(emissions([2]), [2, 2])


def test_silence_and_cancel_do_not_make_lyric_times():
    result = align_lyrics(emissions([0] * 20), VOCAB, "A B")
    assert all(w["startSeconds"] is None for w in result["words"])

    def cancel():
        raise RuntimeError("cancel")

    with pytest.raises(RuntimeError, match="cancel"):
        align_lyrics(emissions([2]), VOCAB, "A", check=cancel)


def word_emissions(text):
    vocabulary = {"<pad>": 0, "|": 1}
    vocabulary.update({c: i + 2 for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")})
    ids = [item for c in text for item in (vocabulary[c], 0)]
    scores = np.full((len(ids), len(vocabulary)), -12.0, dtype=np.float32)
    scores[np.arange(len(ids)), ids] = -0.001
    return scores, vocabulary


def test_plausible_word_mismatch_uses_supported_characters():
    scores, vocabulary = word_emissions("CATS")
    result = align_lyrics(scores, vocabulary, "CAT")
    assert result["words"][0]["startSeconds"] is not None
    assert "text_mismatch" in result["words"][0]["flags"]


@pytest.mark.parametrize(
    ("observed", "supplied"),
    [("SUNSHINES", "SUN SHINES"), ("SUN|SHINES", "SUNSHINES"), ("ONETWOTHREE", "ONE TWO THREE")],
)
def test_word_boundary_mismatch_realigns_supported_joint_window(observed, supplied):
    scores, vocabulary = word_emissions(observed)
    result = align_lyrics(scores, vocabulary, supplied)
    assert all(word["startSeconds"] is not None for word in result["words"])
    assert all("word_boundary_mismatch" in word["flags"] for word in result["words"])
    assert not result["extraWords"]
    for first, second in zip(result["words"], result["words"][1:], strict=False):
        assert first["endSeconds"] <= second["startSeconds"]
    assert result["words"][-1]["endSeconds"] <= len(scores) * 0.02 + 0.0125


def test_plausible_spelling_match_cannot_fabricate_missing_acoustic_character():
    scores, vocabulary = word_emissions("CAT")
    result = align_lyrics(scores, vocabulary, "CATS")
    assert result["words"][0]["startSeconds"] is None


def test_long_word_cannot_hide_one_unsupported_character_in_average():
    scores, vocabulary = word_emissions("CARTWHEEL")
    result = align_lyrics(scores, vocabulary, "CARTWHEELS")
    assert result["words"][0]["startSeconds"] is None


def test_missing_line_and_repeat_across_lines_preserve_lyric_identity():
    scores, vocabulary = word_emissions("SUN|SHINES|SUN|SHINES")
    result = align_lyrics(scores, vocabulary, "[Verse]\nSUN SHINES\nBLUE SKY\n[Chorus]\nSUN SHINES")
    assert [word["lineIndex"] for word in result["words"]] == [0, 0, 1, 1, 2, 2]
    assert all(word["startSeconds"] is None for word in result["words"][2:4])
    assert result["lines"][1]["startSeconds"] is None
    assert result["lines"][2]["sectionLabel"] == "Chorus"
    assert result["lines"][0]["endSeconds"] <= result["lines"][2]["startSeconds"]
    assert not result["extraWords"]


@pytest.mark.parametrize(
    "observed,supplied",
    [
        ("SUNSH|INES", "SUN SHINES"),
        ("SUNSH|INESON", "SUN SHINES ON"),
        ("SUNSH|IN|ES", "SUN SHINES"),
    ],
)
def test_lyrics_repair_shifted_phrase_boundaries_without_replacing_raw_recognition(
    observed, supplied
):
    scores, vocabulary = word_emissions(observed)
    result = align_lyrics(scores, vocabulary, supplied)
    assert all(word["startSeconds"] is not None for word in result["words"])
    assert result["recognizedText"] == observed.replace("|", " ")
    assert result["lyricAssistedText"] == supplied
    assert all(word["verification"] == "lyric_assisted" for word in result["words"])
    assert not result["extraWords"]
    for first, second in zip(result["words"], result["words"][1:], strict=False):
        assert first["endSeconds"] <= second["startSeconds"]


def test_phrase_context_recovers_misheard_word_only_with_acoustic_alternatives():
    scores, vocabulary = word_emissions("THEY|KEN|REMEMBER")
    # C and A are plausible acoustic alternatives to the greedy K and E.
    scores[10, vocabulary["K"]] = np.log(0.55)
    scores[10, vocabulary["C"]] = np.log(0.4)
    scores[12, vocabulary["E"]] = np.log(0.55)
    scores[12, vocabulary["A"]] = np.log(0.4)
    result = align_lyrics(scores, vocabulary, "THEY CAN REMEMBER")
    assert result["recognizedText"] == "THEY KEN REMEMBER"
    assert result["lyricAssistedText"] == "THEY CAN REMEMBER"
    assert result["words"][1]["verification"] == "lyric_assisted"
    assert result["words"][1]["observedText"] == "KEN"
    assert result["words"][1]["startSeconds"] == pytest.approx(0.2025)
    assert result["words"][0]["verification"] == "matched"
    assert result["words"][2]["verification"] == "matched"


def test_matching_context_cannot_verify_an_unsupported_middle_word():
    scores, vocabulary = word_emissions("THEY|KEN|REMEMBER")
    result = align_lyrics(scores, vocabulary, "THEY CAN REMEMBER")
    assert result["words"][1]["startSeconds"] is None
    assert result["words"][1]["verification"] == "unresolved"
    assert result["lyricAssistedText"] == "THEY KEN REMEMBER"
    assert [w["text"] for w in result["extraWords"]] == ["KEN"]
    assert result["words"][0]["startSeconds"] is not None
    assert result["words"][2]["startSeconds"] is not None


def test_assisted_transcript_retains_extra_words_and_does_not_insert_missing_lyrics():
    scores, vocabulary = word_emissions("SUNSH|INES|AGAIN")
    result = align_lyrics(scores, vocabulary, "SUN SHINES BLUE")
    assert result["recognizedText"] == "SUNSH INES AGAIN"
    assert result["lyricAssistedText"] == "SUN SHINES AGAIN"
    assert result["words"][-1]["verification"] == "unresolved"
    assert result["extraWords"][0]["text"] == "AGAIN"


def test_phrase_correction_does_not_cross_supplied_line_boundary():
    scores, vocabulary = word_emissions("THEY|KEN|REMEMBER")
    scores[10, vocabulary["K"]] = np.log(0.55)
    scores[10, vocabulary["C"]] = np.log(0.4)
    scores[12, vocabulary["E"]] = np.log(0.55)
    scores[12, vocabulary["A"]] = np.log(0.4)
    result = align_lyrics(scores, vocabulary, "THEY\nCAN REMEMBER")
    assert result["words"][1]["startSeconds"] is None
    assert result["lyricAssistedText"] == "THEY KEN REMEMBER"


def test_repeated_context_does_not_move_a_supported_word_to_another_occurrence():
    scores, vocabulary = word_emissions("THEY|KEN|REMEMBER|THEY|CAN|REMEMBER")
    result = align_lyrics(scores, vocabulary, "THEY CAN REMEMBER\nTHEY CAN REMEMBER")
    assert result["words"][1]["startSeconds"] is None
    assert result["words"][4]["startSeconds"] is not None
    assert result["words"][4]["startSeconds"] > result["words"][2]["endSeconds"]
    assert result["recognizedText"] == result["lyricAssistedText"]


def test_repeated_refrains_reuse_acoustic_windows(monkeypatch):
    from wee_todd_mlx.audio_analysis import alignment

    repeats = 20
    scores, vocabulary = word_emissions("|".join(["THEY|KEN|REMEMBER"] * repeats))
    calls = 0
    original = alignment.forced_token_spans

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(alignment, "forced_token_spans", counted)
    result = align_lyrics(scores, vocabulary, "\n".join(["THEY CAN REMEMBER"] * repeats))
    assert sum(w["startSeconds"] is not None for w in result["words"]) == 2 * repeats
    # Repeating the same text must not repeat each window's acoustic solve
    # for every other occurrence of that lyric (quadratic inference work).
    assert calls <= 6 * repeats
