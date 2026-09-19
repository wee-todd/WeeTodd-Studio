"""App-owned staged Qwen Base and named CustomVoice inference."""

import gc
import json
import re

from wee_todd_mlx.speech_lifecycle import serialized

_active = None


@serialized
def unload():
    _release()


def _release():
    global _active
    _active = None
    gc.collect()
    import mlx.core as mx

    mx.clear_cache()


def validate_request(request, config, *, reference_path=None):
    """Reject unsupported controls before allocating model tensors."""
    custom = config.get("tts_model_type") == "custom_voice"
    if config.get("tts_model_type") not in {"base", "custom_voice"}:
        raise ValueError("Unsupported Qwen speech model kind")
    language = request.get("language", "auto")
    if not isinstance(language, str) or language.lower() not in {
        "auto",
        *config["talker_config"]["codec_language_id"],
    }:
        raise ValueError("Unsupported Qwen speech language")
    text = request.get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Qwen requires speech text")
    if re.search(r"\[[^\[\]\r\n]+\]", text):
        raise ValueError("Qwen does not support Fish inline delivery tags")
    instruct = request.get("instruct", "")
    if not isinstance(instruct, str) or len(instruct) > 2000:
        raise ValueError("Qwen instructions must be text of at most 2000 characters")
    if custom:
        if config.get("tts_model_size") != "1b7":
            raise ValueError("Emotion instructions require Qwen CustomVoice 1.7B")
        if request.get("reference_mode") != "customVoice":
            raise ValueError("Qwen CustomVoice requires customVoice mode")
        if reference_path or request.get("reference"):
            raise ValueError("Qwen CustomVoice uses a named speaker, not reference audio")
        speaker = request.get("speaker", "")
        if not isinstance(speaker, str) or speaker.lower() not in config["talker_config"].get(
            "spk_id", {}
        ):
            raise ValueError("Unsupported Qwen CustomVoice speaker")
    else:
        if instruct or request.get("speaker"):
            raise ValueError("Qwen Base does not support speaker presets or emotion instructions")
        if request.get("reference_mode", "audioAndTranscript") not in {
            "audioAndTranscript",
            "speakerIdentityOnly",
        }:
            raise ValueError("Qwen Base requires a reference audio mode")
        if not reference_path:
            raise ValueError("Qwen Base needs reference audio")
    return custom


@serialized
def generate(
    request, *, reference_path=None, inspection=None, cancelled=lambda: False, progress=None
):
    import mlx.core as mx
    import numpy as np
    from scipy.io import wavfile
    from transformers import Qwen2TokenizerFast

    from wee_todd_mlx.speech_checkpoint import inspect

    from .codec import QwenCodec
    from .model import QwenTalker
    from .speaker import SpeakerEncoder

    global _active
    spec = inspection or inspect(
        request["model_path"], engine="qwen3TTS", precision=request.get("precision", "auto")
    )
    custom = validate_request(request, spec["config"], reference_path=reference_path)
    codec_config = json.loads(open(spec["root"] + "/speech_tokenizer/config.json").read())

    def stage(message):
        if cancelled():
            raise InterruptedError("Speech generation cancelled")
        if progress:
            progress(message)

    try:
        reference_codes = None
        if not custom:
            rate, audio = wavfile.read(reference_path)
            if rate != 24000 or audio.ndim != 1 or audio.dtype.kind != "f":
                raise ValueError("Qwen requires a prepared mono 24,000 Hz float reference")
            stage("Extracting Qwen speaker identity")
            # MLX loads tensors lazily. Retain only this stage before evaluating it.
            weights = {}
            for filename in spec["model_files"]:
                weights.update(
                    {k: v for k, v in mx.load(filename).items() if k.startswith("speaker_encoder.")}
                )
            _active = SpeakerEncoder(weights)
            del weights
            speaker = np.array(_active(audio))
            _release()
            if request.get("reference_mode", "audioAndTranscript") == "audioAndTranscript":
                stage("Encoding Qwen reference speech")
                _active = QwenCodec(mx.load(spec["codec_path"]), codec_config)
                reference_codes = np.array(_active.encode(audio))
                _release()
        stage("Loading Qwen CustomVoice talker" if custom else "Loading Qwen Base talker")
        weights = {}
        for filename in spec["model_files"]:
            weights.update({k: v for k, v in mx.load(filename).items() if k.startswith("talker.")})
        _active = QwenTalker(weights, spec["config"])
        del weights
        tokenizer = Qwen2TokenizerFast.from_pretrained(spec["root"], local_files_only=True)
        if custom:
            prompt, trailing, pad = _active.custom_voice_prompt(
                tokenizer,
                request["text"],
                request["speaker"],
                request.get("instruct", ""),
                request.get("language", "auto"),
            )
        else:
            prompt, trailing, pad = _active.prompt(
                tokenizer,
                request["text"],
                (request.get("reference") or {}).get("transcript", ""),
                mx.array(speaker),
                reference_codes,
                request.get("language", "auto"),
            )
        sampling = request.get("sampling", {})
        mx.random.seed(int(request.get("seed", 0)))
        stage("Generating Qwen speech")
        codes, truncated = _active.generate(
            prompt,
            trailing,
            pad,
            max_tokens=int(sampling.get("max_tokens", 512)),
            temperature=float(sampling.get("temperature", 0.9)),
            top_p=float(sampling.get("top_p", 1.0)),
            top_k=int(sampling.get("top_k", 50)),
            cancelled=cancelled,
            progress=lambda n, total: stage(f"Qwen speech: {n} / {total} frames"),
        )
        # The arrays retain weights through lazy graphs unless evaluated before unloading.
        del prompt, trailing, pad
        _release()
        stage("Decoding Qwen speech")
        _active = QwenCodec(mx.load(spec["codec_path"]), codec_config)
        full = (
            codes if reference_codes is None else np.concatenate([reference_codes, codes], axis=1)
        )
        audio = np.array(_active.decode(mx.array(full)), dtype=np.float32)
        if reference_codes is not None:
            audio = audio[reference_codes.shape[1] * 1920 :]
        if not np.all(np.isfinite(audio)):
            raise ValueError("Qwen decoded non-finite audio")
        return dict(audio=audio, sample_rate=24000, codes=codes, truncated=truncated)
    finally:
        _release()
