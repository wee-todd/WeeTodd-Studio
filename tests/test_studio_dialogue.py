import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))


def request(engine="fishS2Pro"):
    return {
        "voice": dict(
            engine=engine,
            model_path="/weights",
            seed=2**32 - 1,
            turns=[
                dict(
                    id="a",
                    speaker_id="alice",
                    speaker_name="Alice",
                    text="Hello",
                    reference_mode="audioAndTranscript",
                    reference=dict(path="alice.wav", transcript="A"),
                    gap_after=0.01,
                ),
                dict(
                    id="b",
                    speaker_id="bob",
                    speaker_name="Bob",
                    text="Hi",
                    reference_mode="audioAndTranscript",
                    reference=dict(path="bob.wav", transcript="B"),
                    gap_after=0.02,
                ),
            ],
        )
    }


@pytest.fixture
def backend(monkeypatch):
    module = importlib.import_module("studio_dialogue")
    from wee_todd_mlx import speech_checkpoint

    events = []
    monkeypatch.setattr(
        speech_checkpoint,
        "inspect",
        lambda *a, **k: (
            events.append("inspect")
            or {
                "identity": "model",
                "config": {
                    "tts_model_type": "base",
                    "talker_config": {"codec_language_id": {"english": 1}},
                },
            }
        ),
    )

    def prepare(ref, **kwargs):
        events.append(("prepare", ref["path"]))
        return {"path": ref["path"], "identity": kwargs["model_identity"]}

    monkeypatch.setattr(module, "prepare_reference", prepare)
    for engine in ("fish_speech_mlx", "qwen3_tts_mlx"):
        pipeline = importlib.import_module(engine + ".pipeline")

        def generate(value, *, reference_path, inspection, _engine=engine, **kwargs):
            events.append(("render", value, reference_path, inspection))
            return dict(
                audio=np.full(100, 0.25, dtype=np.float32),
                codes=np.zeros((2, 3), dtype=np.int32),
                sample_rate=44100 if _engine == "fish_speech_mlx" else 24000,
                truncated=value["text"] == "Hi",
            )

        monkeypatch.setattr(pipeline, "generate", generate)
    return module, events


@pytest.mark.parametrize("engine,rate", [("fishS2Pro", 44100), ("qwen3TTS", 24000)])
def test_dialogue_preflights_then_renders_separate_speakers_and_assembles(
    backend, tmp_path, engine, rate
):
    module, events = backend
    original = request(engine)
    result = importlib.import_module("studio_voice").dispatch(
        "voice-dialogue", original, tmp_path / "take"
    )
    assert events[:3] == ["inspect", ("prepare", "alice.wav"), ("prepare", "bob.wav")]
    assert [e[1]["seed"] for e in events[3:]] == [2**32 - 1, 0]
    assert [e[1]["text"] for e in events[3:]] == ["Hello", "Hi"]
    assert [e[2] for e in events[3:]] == ["alice.wav", "bob.wav"]
    assert all("speaker_name" not in e[1] and "turns" not in e[1] for e in events[3:])
    assert original["voice"]["turns"][1].get("seed") is None
    assert result["truncated"] is True
    assert [(t["start_frame"], t["end_frame"], t["gap_frames"]) for t in result["turns"]] == [
        (0, 100, round(rate * 0.01)),
        (100 + round(rate * 0.01), 200 + round(rate * 0.01), round(rate * 0.02)),
    ]
    actual_rate, pcm = wavfile.read(result["audio"])
    assert actual_rate == rate
    assert len(pcm) == 200 + round(rate * 0.01) + round(rate * 0.02) == result["frames"]
    assert np.all(pcm[100 : 100 + round(rate * 0.01)] == 0)
    assert json.loads((tmp_path / "take/receipt.json").read_text())["state"] == "complete"
    for turn in result["turns"]:
        assert Path(turn["audio"]).is_file()
        assert (Path(turn["artifacts"]) / "codes.npy").is_file()
        assert (Path(turn["artifacts"]) / "receipt.json").is_file()


@pytest.mark.parametrize(
    "patch",
    [
        dict(id="a"),
        dict(speaker_id=""),
        dict(speaker_name=""),
        dict(text=""),
        dict(gap_after=float("nan")),
        dict(gap_after=11),
        dict(reference_mode="speakerIdentityOnly"),
    ],
)
def test_all_turn_parameters_validated_before_inference(backend, tmp_path, patch):
    module, events = backend
    payload = request()
    payload["voice"]["turns"][1].update(patch)
    with pytest.raises(ValueError):
        module.dispatch(payload, tmp_path / "take")
    assert not any(isinstance(e, tuple) and e[0] == "render" for e in events)
    assert not (tmp_path / "take/receipt.json").exists()


def test_later_invalid_reference_prevents_all_inference(backend, tmp_path, monkeypatch):
    module, events = backend

    def prepare(ref, **kwargs):
        if ref["path"] == "bob.wav":
            raise ValueError("missing reference")
        return {"path": ref["path"]}

    monkeypatch.setattr(module, "prepare_reference", prepare)
    with pytest.raises(ValueError, match="missing reference"):
        module.dispatch(request(), tmp_path / "take")
    assert events == ["inspect"]
    assert not (tmp_path / "take/receipt.json").exists()


@pytest.mark.parametrize("failure", [ValueError("failed"), InterruptedError("cancelled")])
def test_failed_turn_retains_prior_take_but_no_dialogue_receipt(
    backend, tmp_path, monkeypatch, failure
):
    module, events = backend
    pipeline = importlib.import_module("fish_speech_mlx.pipeline")
    original = pipeline.generate

    def generate(value, **kwargs):
        if value["text"] == "Hi":
            raise failure
        return original(value, **kwargs)

    monkeypatch.setattr(pipeline, "generate", generate)
    with pytest.raises(type(failure)):
        module.dispatch(request(), tmp_path / "take")
    assert not (tmp_path / "take/receipt.json").exists()
    assert not (tmp_path / "take/take.wav").exists()
    assert len(list((tmp_path / "take").glob("turns/*/receipt.json"))) == 1


def test_cancel_before_preparation_never_inspects_or_renders(backend, tmp_path):
    module, events = backend
    with pytest.raises(InterruptedError):
        module.dispatch(request(), tmp_path / "take", cancelled=lambda: True)
    assert events == []


@pytest.mark.parametrize("audio", [np.zeros(10), np.array([np.nan]), np.ones((2, 2))])
def test_bad_audio_never_publishes_dialogue(backend, tmp_path, monkeypatch, audio):
    module, events = backend
    monkeypatch.setattr(
        importlib.import_module("fish_speech_mlx.pipeline"),
        "generate",
        lambda *a, **k: dict(audio=audio, sample_rate=44100, codes=np.zeros(2), truncated=False),
    )
    with pytest.raises(ValueError, match="audible"):
        module.dispatch(request(), tmp_path / "take")
    assert not (tmp_path / "take/receipt.json").exists()


@pytest.mark.parametrize("turns", [[], [None], [{}] * 65])
def test_dialogue_turn_count_and_type_are_bounded(backend, tmp_path, turns):
    module, events = backend
    payload = request()
    payload["voice"]["turns"] = turns
    with pytest.raises(ValueError):
        module.dispatch(payload, tmp_path / "take")
    assert events == []


def test_synthetic_fish_needs_no_reference_and_default_gap_is_zero(backend, tmp_path):
    module, events = backend
    payload = request()
    payload["voice"]["turns"] = [
        dict(
            id="a",
            speaker_id="a",
            speaker_name="Narrator",
            text="Hello",
            reference_mode="synthetic",
        )
    ]
    result = module.dispatch(payload, tmp_path / "take")
    assert len(events) == 2
    assert events[1][2] is None
    assert result["frames"] == 100
    assert result["turns"][0]["gap_frames"] == 0


def test_cancel_during_reference_preparation_prevents_all_inference(backend, tmp_path, monkeypatch):
    module, events = backend
    stopped = False

    def prepare(ref, **kwargs):
        nonlocal stopped
        stopped = True
        return {"path": ref["path"]}

    monkeypatch.setattr(module, "prepare_reference", prepare)
    with pytest.raises(InterruptedError):
        module.dispatch(request(), tmp_path / "take", cancelled=lambda: stopped)
    assert events == ["inspect"]
    assert not (tmp_path / "take/receipt.json").exists()


def test_cancellation_after_final_audio_write_prevents_completed_receipt(
    backend, tmp_path, monkeypatch
):
    module, events = backend
    stopped = False
    original = module.os.replace

    def replace(source, target):
        nonlocal stopped
        original(source, target)
        if Path(target) == tmp_path / "take/take.wav":
            stopped = True

    monkeypatch.setattr(module.os, "replace", replace)
    with pytest.raises(InterruptedError):
        module.dispatch(request(), tmp_path / "take", cancelled=lambda: stopped)
    assert not (tmp_path / "take/receipt.json").exists()
    assert json.loads((tmp_path / "take/failure.json").read_text())["state"] == "cancelled"


def test_qwen_language_checked_before_reference_or_model_work(backend, tmp_path, monkeypatch):
    module, events = backend
    from wee_todd_mlx import speech_checkpoint

    monkeypatch.setattr(
        speech_checkpoint,
        "inspect",
        lambda *a, **k: dict(
            identity="model",
            config=dict(
                tts_model_type="base", talker_config=dict(codec_language_id={"english": 1})
            ),
        ),
    )
    payload = request("qwen3TTS")
    payload["voice"]["language"] = "invalid"
    with pytest.raises(ValueError, match="language"):
        module.dispatch(payload, tmp_path / "take")
    assert events == []


def test_qwen_speaker_identity_mode_preserved(backend, tmp_path):
    module, events = backend
    payload = request("qwen3TTS")
    payload["voice"]["turns"][0]["reference_mode"] = "speakerIdentityOnly"
    result = module.dispatch(payload, tmp_path / "take")
    assert events[3][1]["reference_mode"] == "speakerIdentityOnly"
    assert result["turns"][0]["request"]["reference_mode"] == "speakerIdentityOnly"


def test_reference_file_validation_really_precedes_all_weighted_work(
    backend, tmp_path, monkeypatch
):
    module, events = backend
    from wee_todd_mlx.speech_reference import prepare_reference

    monkeypatch.setattr(module, "prepare_reference", prepare_reference)
    payload = request()
    for turn in payload["voice"]["turns"]:
        turn["reference"].update(start=0, duration=1, path=str(tmp_path / "missing.wav"))
    with pytest.raises(ValueError, match="missing reference"):
        module.dispatch(payload, tmp_path / "take")
    assert events == ["inspect"]


def test_qwen_inline_delivery_tags_rejected_before_checkpoint_or_inference(backend, tmp_path):
    module, events = backend
    payload = request("qwen3TTS")
    payload["voice"]["turns"][1]["text"] = "[whispering] Hi"
    with pytest.raises(ValueError, match="Fish.*tags"):
        module.dispatch(payload, tmp_path / "take")
    assert events == []


@pytest.mark.parametrize("text", ["[happy] Hello", "Hello [voice soft]", "[laughing]", "[pause]"])
def test_qwen_single_voice_rejects_fish_tags(text):
    from studio_voice import validate_request

    with pytest.raises(ValueError, match="Fish.*tags"):
        validate_request(
            dict(
                engine="qwen3TTS",
                model_path="/weights",
                text=text,
                reference_mode="speakerIdentityOnly",
                reference={},
            )
        )


def test_fish_tags_remain_supported():
    from studio_voice import validate_request

    assert (
        validate_request(
            dict(
                engine="fishS2Pro",
                model_path="/weights",
                text="[happy] Hello",
                reference_mode="synthetic",
            )
        )["text"]
        == "[happy] Hello"
    )


def test_custom_voice_dialogue_preserves_per_speaker_delivery_without_references(
    backend, tmp_path, monkeypatch
):
    module, events = backend
    from wee_todd_mlx import speech_checkpoint

    monkeypatch.setattr(
        speech_checkpoint,
        "inspect",
        lambda *a, **k: dict(
            identity="model",
            config=dict(
                tts_model_type="custom_voice",
                tts_model_size="1b7",
                talker_config=dict(
                    codec_language_id={"english": 1}, spk_id={"ryan": 1, "aiden": 2}
                ),
            ),
        ),
    )
    payload = request("qwen3TTS")
    for turn, speaker, instruct in zip(
        payload["voice"]["turns"], ["ryan", "aiden"], ["Happy", "Sad"], strict=True
    ):
        turn.update(reference_mode="customVoice", speaker=speaker, instruct=instruct)
        del turn["reference"]
    result = module.dispatch(payload, tmp_path / "custom")
    renders = [e for e in events if e[0] == "render"]
    assert len(renders) == 2
    assert [e[1]["instruct"] for e in renders] == ["Happy", "Sad"]
    assert [e[1]["speaker"] for e in renders] == ["ryan", "aiden"]
    assert all(e[2] is None for e in renders)
    assert not any(e[0] == "prepare" for e in events)
    assert result["request"]["turns"][1]["instruct"] == "Sad"


def test_custom_voice_model_mismatch_rejected_before_reference_preparation(backend, tmp_path):
    module, events = backend
    payload = request("qwen3TTS")
    payload["voice"]["turns"][1].update(
        reference_mode="customVoice", reference=None, speaker="ryan", instruct="Sad"
    )
    with pytest.raises(ValueError, match="Base does not support"):
        module.dispatch(payload, tmp_path / "mismatch")
    assert events == ["inspect"]
