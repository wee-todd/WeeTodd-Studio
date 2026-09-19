import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")


def test_speaker_mel_shape_and_silence_floor():
    from qwen3_tts_mlx.speaker import mel_spectrogram

    mel = mel_spectrogram(np.zeros(24000, np.float32))
    assert mel.shape == (1, 93, 128)
    assert np.isfinite(np.array(mel)).all()
    assert np.max(np.array(mel)) < 0


def test_qwen_causal_downsample_uses_right_padding_for_incomplete_frame():
    from qwen3_tts_mlx.codec import QwenCodec

    codec = QwenCodec({"c.weight": mx.ones((1, 1, 4))}, {})
    x = mx.ones((1, 5, 1))
    np.testing.assert_allclose(np.array(codec.conv(x, "c", stride=2)).ravel(), [2, 4, 3])


def test_mrope_with_equal_text_positions_matches_standard_rope():
    from qwen3_tts_mlx.model import text_rope
    from wee_todd_mlx.speech_ops import rope

    x = mx.array(np.random.default_rng(42).normal(size=(1, 2, 4, 128)).astype("float32"))
    np.testing.assert_allclose(
        np.array(text_rope(x, 9, 1000000)), np.array(rope(x, 9, 1000000)), atol=1e-6
    )


def test_custom_voice_prompt_uses_codec_speaker_and_preceding_instruction(monkeypatch):
    from qwen3_tts_mlx.model import QwenTalker

    model = QwenTalker(
        {},
        {
            "talker_config": {
                "spk_id": {"eric": 2875},
                "spk_is_dialect": {"eric": "sichuan_dialect"},
            }
        },
    )
    calls = []

    class Tokenizer:
        def encode(self, text, **kwargs):
            calls.append(text)
            return [10, 11]

    monkeypatch.setattr(model, "codec", lambda ids: mx.array(ids).reshape(1, -1, 1))
    monkeypatch.setattr(model, "text", lambda ids: mx.array(ids).reshape(1, -1, 1))

    def base_prompt(tokenizer, text, transcript, speaker, codes, language):
        assert int(speaker.item()) == 2875
        assert codes is None and transcript == "" and language == "sichuan_dialect"
        return mx.array([[[20]]]), mx.array([[[30]]]), mx.array([[[40]]])

    monkeypatch.setattr(model, "prompt", base_prompt)
    prompt, trailing, pad = model.custom_voice_prompt(
        Tokenizer(), "Hello", "Eric", "Sound happy.", "auto"
    )
    assert calls == ["<|im_start|>user\nSound happy.<|im_end|>\n"]
    np.testing.assert_array_equal(np.array(prompt).ravel(), [10, 11, 20])
    assert int(trailing.item()) == 30 and int(pad.item()) == 40
    neutral, _, _ = model.custom_voice_prompt(Tokenizer(), "Hello", "eric", "", "chinese")
    assert neutral.shape == (1, 1, 1)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("speaker", "unknown", "speaker"),
        ("language", "elvish", "language"),
        ("instruct", "x" * 2001, "2000"),
        ("instruct", 7, "2000"),
        ("reference", {"audio_path": "ref.wav"}, "reference audio"),
        ("reference_mode", "audioAndTranscript", "customVoice"),
        ("text", "[happy] Hello", "inline delivery"),
    ],
)
def test_custom_voice_controls_fail_before_tensor_load(field, value, message):
    from qwen3_tts_mlx.pipeline import validate_request

    config = {
        "tts_model_type": "custom_voice",
        "tts_model_size": "1b7",
        "talker_config": {"codec_language_id": {"english": 2050}, "spk_id": {"ryan": 3061}},
    }
    request = dict(text="Hello", reference_mode="customVoice", speaker="ryan")
    assert validate_request(request, config)
    request[field] = value
    with pytest.raises(ValueError, match=message):
        validate_request(request, config)


@pytest.mark.parametrize(
    "controls",
    [dict(instruct="Sound happy."), dict(speaker="ryan"), dict(reference_mode="customVoice")],
)
def test_base_rejects_custom_voice_controls(controls):
    from qwen3_tts_mlx.pipeline import validate_request

    config = {"tts_model_type": "base", "talker_config": {"codec_language_id": {}}}
    with pytest.raises(ValueError, match="Base"):
        validate_request(dict(text="Hello", **controls), config, reference_path="ref.wav")


def test_base_retains_both_reference_modes():
    from qwen3_tts_mlx.pipeline import validate_request

    config = {"tts_model_type": "base", "talker_config": {"codec_language_id": {}}}
    for mode in ("audioAndTranscript", "speakerIdentityOnly"):
        assert not validate_request(
            dict(text="Hello", reference_mode=mode), config, reference_path="ref.wav"
        )
