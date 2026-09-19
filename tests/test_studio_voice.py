import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))


def test_voice_request_refuses_unsupported_modes_and_nonfinite_sampling():
    validate = importlib.import_module("studio_voice").validate_request
    base = dict(engine="fishS2Pro", model_path="/weights", text="Hello", reference_mode="synthetic")
    assert validate(base)["sampling"]["max_tokens"] == 512
    for patch in [
        dict(reference_mode="speakerIdentityOnly"),
        dict(engine="qwen3TTS"),
        dict(sampling={"temperature": float("nan")}),
        dict(seed=-1),
    ]:
        with pytest.raises(ValueError):
            validate(dict(base, **patch))


def test_studio_voice_failure_never_publishes_completed_take(tmp_path, monkeypatch):
    import fish_speech_mlx.pipeline
    from wee_todd_mlx import speech_checkpoint

    voice = importlib.import_module("studio_voice")
    monkeypatch.setattr(
        speech_checkpoint, "inspect", lambda *a, **k: dict(identity="test", precision="8bit")
    )
    monkeypatch.setattr(
        fish_speech_mlx.pipeline,
        "generate",
        lambda *a, **k: (_ for _ in ()).throw(InterruptedError("cancelled")),
    )
    request = dict(
        voice=dict(
            engine="fishS2Pro", model_path="/weights", text="Hello", reference_mode="synthetic"
        )
    )
    with pytest.raises(InterruptedError):
        voice.dispatch("voice-generate", request, tmp_path / "job")
    assert json.loads((tmp_path / "job/failure.json").read_text())["state"] == "cancelled"
    assert not (tmp_path / "job/receipt.json").exists()


@pytest.mark.parametrize(
    "kind,engine,size,name",
    [
        ("fish_qwen3_omni", "fishS2Pro", None, "Fish S2 Pro · 8-bit"),
        ("qwen3_tts", "qwen3TTS", "1b7", "Qwen3-TTS Base · 1.7B · 8-bit"),
        ("qwen3_tts", "qwen3TTS", "0b6", "Qwen3-TTS Base · 0.6B · 8-bit"),
    ],
)
def test_model_setup_detects_engine_and_names_validated_checkpoint(
    tmp_path, monkeypatch, kind, engine, size, name
):
    from wee_todd_mlx import speech_checkpoint

    config = dict(model_type=kind, tts_model_size=size)
    (tmp_path / "config.json").write_text(json.dumps(config))
    calls = []

    def inspect(root, **kwargs):
        calls.append(kwargs)
        return dict(identity="checked", precision="8bit", config=config, root=str(tmp_path))

    monkeypatch.setattr(speech_checkpoint, "inspect", inspect)
    result = importlib.import_module("studio_voice").dispatch(
        "voice-inspect", {"voice": {"model_path": str(tmp_path)}}
    )
    assert calls[0]["engine"] == engine
    assert result["model"] == dict(
        path=str(tmp_path),
        engine=engine,
        name=name,
        kind="base" if engine == "qwen3TTS" else "fish",
    )


def test_model_setup_does_not_guess_an_unknown_model_engine(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"unrelated"}')
    with pytest.raises(ValueError, match="Fish S2 Pro or Qwen3-TTS Base/CustomVoice"):
        importlib.import_module("studio_voice").dispatch(
            "voice-inspect", {"voice": {"model_path": str(tmp_path)}}
        )


def test_custom_voice_controls_require_explicit_mode_and_no_audio_reference():
    validate = importlib.import_module("studio_voice").validate_request
    request = dict(
        engine="qwen3TTS",
        model_path="/model",
        text="Hello",
        reference_mode="customVoice",
        speaker="ryan",
        instruct="Happy",
    )
    assert validate(request)["instruct"] == "Happy"
    for change in [
        dict(reference={"path": "/ref"}),
        dict(instruct="x" * 2001),
        dict(instruct="<|im_start|>"),
        dict(speaker=""),
        dict(reference_mode="speakerIdentityOnly", reference={"path": "/ref"}),
    ]:
        with pytest.raises(ValueError):
            validate(dict(request, **change))
