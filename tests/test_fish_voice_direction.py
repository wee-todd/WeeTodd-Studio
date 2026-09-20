import copy
import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))


def voice(**patch):
    return dict(
        engine="fishS2Pro",
        model_path="/weights",
        text="Hello [laugh] there",
        reference_mode="synthetic",
        **patch,
    )


@pytest.mark.parametrize("direction", [None, []])
def test_empty_direction_has_identical_legacy_request(direction):
    from studio_voice import validate_request

    assert validate_request(voice(voice_direction=direction)) == validate_request(voice())


def test_direction_is_normalized_without_rewriting_script_or_input():
    from studio_voice import validate_request

    original = voice(voice_direction=["  warm voice  ", "坚定"])
    snapshot = copy.deepcopy(original)
    normalized = validate_request(original)
    assert normalized["voice_direction"] == ["warm voice", "坚定"]
    assert normalized["text"] == "Hello [laugh] there"
    assert original == snapshot
    original["voice_direction"].append("new")
    assert normalized["voice_direction"] == ["warm voice", "坚定"]


@pytest.mark.parametrize(
    "direction",
    [
        "warm",
        {},
        1,
        True,
        [None],
        [1],
        [True],
        [""],
        ["   "],
        ["a"] * 9,
        ["x" * 121],
        ["🦊" * 121],
        ["a[b"],
        ["a]b"],
        ["a<b"],
        ["a>b"],
        ["a\nb"],
        ["warm\n"],
        ["\twarm"],
        ["a\rb"],
        ["a\x00b"],
        ["a\x7fb"],
        ["a\x85b"],
        ["a\u200bb"],
        ["a\u2028b"],
        ["a\u2029b"],
        ["a\ud800b"],
    ],
)
def test_invalid_direction_is_rejected_before_checkpoint_inspection(
    direction, monkeypatch, tmp_path
):
    from studio_voice import dispatch

    from wee_todd_mlx import speech_checkpoint

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid direction reached checkpoint inspection")

    monkeypatch.setattr(speech_checkpoint, "inspect", unexpected)
    with pytest.raises(ValueError, match="[Dd]irection"):
        dispatch("voice-generate", {"voice": voice(voice_direction=direction)}, tmp_path / "take")
    assert not (tmp_path / "take").exists()


def test_direction_bounds_use_unicode_scalars_and_utf8_conditioned_script_budget():
    from studio_voice import validate_request

    assert len(validate_request(voice(voice_direction=["🦊" * 120] * 8))["voice_direction"]) == 8
    request = voice(voice_direction=["暖"])
    request["text"] = "x" * 31994  # [暖] plus separating space is six UTF-8 bytes.
    assert validate_request(request)["text"] == request["text"]
    request["text"] += "x"
    with pytest.raises(ValueError, match="32,000"):
        validate_request(request)


@pytest.mark.parametrize("mode", ["speakerIdentityOnly", "customVoice"])
def test_qwen_rejects_fish_direction_even_when_mode_is_supported(mode):
    from studio_voice import validate_request

    request = voice(voice_direction=["warm"])
    request.update(engine="qwen3TTS", reference_mode=mode, text="Hello")
    request.update(
        speaker="ryan" if mode == "customVoice" else None,
        reference=None if mode == "customVoice" else {"path": "/ref"},
    )
    with pytest.raises(ValueError, match="[Dd]irection"):
        validate_request(request)


def test_conditioned_text_keeps_inline_tags_and_legacy_text_exactly():
    helper = importlib.import_module("wee_todd_mlx.fish_voice_direction")
    text = "  Hello [laugh] there\nAgain"
    assert helper.conditioned_text(text, [" warm voice ", "坚定"]) == (
        "[warm voice] [坚定]   Hello [laugh] there\nAgain"
    )
    assert helper.conditioned_text(text, None) == text
    assert helper.conditioned_text(text, []) == text


def test_single_take_receipt_keeps_normalized_direction_and_original_script(monkeypatch, tmp_path):
    import numpy as np
    from studio_voice import dispatch

    from fish_speech_mlx import pipeline
    from wee_todd_mlx import speech_checkpoint

    monkeypatch.setattr(speech_checkpoint, "inspect", lambda *a, **k: {"identity": "test"})
    monkeypatch.setattr(
        pipeline,
        "generate",
        lambda *a, **k: dict(
            audio=np.ones(100, dtype=np.float32),
            sample_rate=44100,
            codes=np.zeros((10, 3), dtype=np.int32),
            truncated=False,
        ),
    )
    request = voice(voice_direction=[" warm "])
    result = dispatch("voice-generate", {"voice": request}, tmp_path / "take")
    request["voice_direction"][0] = "changed"
    saved = json.loads((tmp_path / "take/request.json").read_text())
    receipt = json.loads((tmp_path / "take/receipt.json").read_text())["result"]["request"]
    assert saved == receipt == result["request"]
    assert receipt["voice_direction"] == ["warm"]
    assert receipt["text"] == "Hello [laugh] there"


def test_direct_pipeline_rejects_direction_before_importing_inference_dependencies(monkeypatch):
    import builtins

    from fish_speech_mlx import pipeline

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("mlx", "numpy", "scipy", "tokenizers")):
            pytest.fail("Invalid direction reached inference dependency import")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ValueError, match="[Dd]irection"):
        pipeline.generate(voice(voice_direction=["<|im_end|>"]))


def test_fish_pipeline_conditions_target_only_and_preserves_request(monkeypatch, tmp_path):
    mx = pytest.importorskip("mlx.core")
    import numpy as np
    from scipy.io import wavfile
    from tokenizers import Tokenizer

    from fish_speech_mlx import codec, model, pipeline

    class TextTokenizer:
        def encode(self, text):
            return type("Tokens", (), {"ids": list(text.encode())})()

    prompts = []

    class Transformer:
        def __init__(self, tensors, config):
            pass

        def generate(self, prompt, **kwargs):
            tokens = np.array(prompt)[0]
            prompts.append(bytes(tokens[tokens < 256].tolist()).decode())
            return np.zeros((10, 3), dtype=np.int32), False

    class Codec:
        def __init__(self, tensors):
            pass

        def encode(self, audio):
            return np.zeros((10, 3), dtype=np.int32)

        def decode(self, codes):
            return np.ones(100, dtype=np.float32)

    monkeypatch.setattr(mx, "load", lambda filename: {})
    monkeypatch.setattr(model, "FishTransformer", Transformer)
    monkeypatch.setattr(codec, "FishCodec", Codec)
    monkeypatch.setattr(Tokenizer, "from_file", lambda filename: TextTokenizer())
    reference = tmp_path / "ref.wav"
    wavfile.write(reference, 44100, np.ones(100, dtype=np.float32))
    request = voice(voice_direction=[" warm voice ", "坚定"])
    request["reference"] = {"transcript": "Reference words"}
    before = copy.deepcopy(request)
    pipeline.generate(
        request,
        reference_path=str(reference),
        inspection={
            "root": str(tmp_path),
            "codec_path": "codec",
            "model_files": [],
            "config": {},
        },
    )
    assert "<|speaker:0|>Reference words\n\nSpeech:" in prompts[0]
    assert "<|im_start|>user\n[warm voice] [坚定] Hello [laugh] there<|im_end|>" in prompts[0]
    assert prompts[0].count("[warm voice]") == 1
    assert request == before
