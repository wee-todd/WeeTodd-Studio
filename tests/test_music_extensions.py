import pytest


def test_extension_catalog_is_pinned_and_excludes_tokenizer():
    from wee_todd_mlx.model_downloads import DownloadFile
    from wee_todd_mlx.music_extensions import catalog

    entries = catalog()
    assert {e["kind"] for e in entries} == {"instrumental", "sheetsage"}
    for entry in entries:
        for spec in entry["files"]:
            item = DownloadFile(**spec)
            assert item.size > 0
            assert "tokenizer" not in item.filename


def test_reference_range_requires_finite_positive_bounded_duration():
    from studio_music_reference import validate_range

    assert validate_range({"source_in": 3, "duration": 20}) == (3, 20)
    for start, duration in [(-1, 2), (0, 0), (0, 301), (float("nan"), 2), (0, True)]:
        with pytest.raises(ValueError):
            validate_range({"source_in": start, "duration": duration})


def test_reference_range_does_not_silently_truncate_source():
    from studio_music_reference import validate_source_range

    with pytest.raises(ValueError, match="beyond"):
        validate_source_range(8, 5, 10)
    validate_source_range(8, 2, 10)


def test_reference_service_prepares_exact_range_and_publishes_reviewable_score(
    tmp_path, monkeypatch
):
    import json
    import shutil
    import wave

    import numpy as np
    from studio_music_reference import analyze

    import sheetsage2_mlx

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    rate = 24000
    source = tmp_path / "reference.wav"
    waveform = np.sin(np.arange(rate * 3) * 2 * np.pi * 220 / rate) * 0.3
    with wave.open(str(source), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes((waveform * 32767).astype("<i2").tobytes())
    # Stub only weighted inference; exercise real decoding, source identity and publication.
    monkeypatch.setattr(sheetsage2_mlx, "inspect_checkpoint", lambda _: {})

    def infer(audio, **kwargs):
        assert audio.shape == (24000,)
        assert kwargs["melody_only"] is True
        assert float(np.max(audio)) > 0.29
        return dict(abc="X:1\nL:1/8\nK:C\nCDEF|", events=[{"time": 0}], model_sha256="model-hash")

    monkeypatch.setattr(sheetsage2_mlx, "transcribe", infer)
    request = dict(audio_path=str(source), model_path="/model", source_in=1, duration=1)
    result = analyze(request, tmp_path / "analysis")
    assert result["source_in"] == 1 and len(result["source_sha256"]) == 64
    assert (tmp_path / "analysis/score.abc").read_text() == result["abc"]
    assert (
        json.loads((tmp_path / "analysis/analysis.json").read_text())["model_sha256"]
        == "model-hash"
    )
    with pytest.raises(FileExistsError):
        analyze(request, tmp_path / "analysis")


def test_reference_service_cancellation_creates_no_result(tmp_path, monkeypatch):
    from studio_music_reference import analyze

    import sheetsage2_mlx

    source = tmp_path / "sample.wav"
    source.write_bytes(b"pending")
    monkeypatch.setattr(sheetsage2_mlx, "inspect_checkpoint", lambda _: {})
    with pytest.raises(InterruptedError):
        analyze(
            dict(audio_path=str(source), model_path="/model", duration=1),
            tmp_path / "result",
            cancelled=lambda: True,
        )
    assert not (tmp_path / "result").exists()
