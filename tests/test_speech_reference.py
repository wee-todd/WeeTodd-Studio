import hashlib
import wave
from pathlib import Path

import numpy as np
import pytest


def sample(tmp_path):
    filename = tmp_path / "reference.wav"
    samples = (np.sin(np.arange(24000) * 0.08) * 12000).astype("<i2")
    with wave.open(str(filename), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(samples.tobytes())
    return dict(path=str(filename), start=0.2, duration=0.5, transcript="A test.", channel="mix")


def test_reference_resamples_and_preserves_source(tmp_path):
    from wee_todd_mlx.speech_reference import prepare_reference

    request = sample(tmp_path)
    before = hashlib.sha256(Path(request["path"]).read_bytes()).hexdigest()
    result = prepare_reference(
        request,
        engine="fishS2Pro",
        mode="audioAndTranscript",
        model_identity={},
        output=tmp_path / "out",
        runtime={},
    )
    assert result["sample_rate"] == 44100
    assert result["frames"] == 22050
    assert hashlib.sha256(Path(request["path"]).read_bytes()).hexdigest() == before
    assert result["path"] != request["path"]


def test_reference_rejects_silent_fallback_and_out_of_range(tmp_path):
    from wee_todd_mlx.speech_reference import prepare_reference

    request = sample(tmp_path)
    for change in [dict(transcript=""), dict(start=0.9), dict(duration=float("nan"))]:
        with pytest.raises(ValueError):
            prepare_reference(
                {**request, **change},
                engine="fishS2Pro",
                mode="audioAndTranscript",
                model_identity={},
                output=tmp_path / "out",
                runtime={},
            )


def test_reference_cache_is_content_and_transcript_sensitive(tmp_path):
    from wee_todd_mlx.speech_reference import prepare_reference

    request = sample(tmp_path)
    kwargs = dict(
        engine="qwen3TTS",
        mode="audioAndTranscript",
        model_identity={"revision": "a"},
        output=tmp_path / "out",
        runtime={},
    )
    a = prepare_reference(request, **kwargs)
    b = prepare_reference({**request, "transcript": "Another test."}, **kwargs)
    assert a["key"] != b["key"]
    Path(a["path"]).write_bytes(b"corrupt")
    assert Path(prepare_reference(request, **kwargs)["path"]).stat().st_size > 100
