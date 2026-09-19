"""Studio speech actions and immutable take receipts; native inference stays in adapters."""

from __future__ import annotations

import copy
import json
import os
import re
import time
from pathlib import Path

from wee_todd_mlx.audio_mix.plan import number
from wee_todd_mlx.speech_reference import digest_file, prepare_reference


def validate_request(value):
    request = copy.deepcopy(value)
    engine = request.get("engine")
    mode = request.get("reference_mode", "audioAndTranscript")
    if engine not in {"fishS2Pro", "qwen3TTS"}:
        raise ValueError("Choose Fish S2 Pro or Qwen3-TTS")
    if mode not in (
        {"audioAndTranscript", "synthetic"}
        if engine == "fishS2Pro"
        else {"audioAndTranscript", "speakerIdentityOnly", "customVoice"}
    ):
        raise ValueError("This speech engine does not support the selected reference mode")
    if (
        not isinstance(request.get("text"), str)
        or not request["text"].strip()
        or len(request["text"].encode()) > 32000
    ):
        raise ValueError("Enter a speech script of at most 32,000 UTF-8 bytes")
    if engine == "qwen3TTS" and re.search(r"\[[^\[\]\r\n]+\]", request["text"]):
        raise ValueError("Qwen3-TTS Base does not support Fish inline delivery tags")
    if not isinstance(request.get("model_path"), str) or not request["model_path"].strip():
        raise ValueError("Choose a local speech checkpoint")
    if request.get("precision", "auto") not in {"auto", "8bit", "bf16"}:
        raise ValueError("Unsupported speech precision")
    seed = request.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be a 32-bit nonnegative integer")
    request["seed"] = seed
    request["reference_mode"] = mode
    sampling = request.setdefault("sampling", {})
    for k, default, low, high in [
        ("temperature", 0.9, 0, 2),
        ("top_p", 1.0, 0.0001, 1),
        ("top_k", 50, 1, 1000),
        ("max_tokens", 512, 1, 4096),
    ]:
        v = sampling.setdefault(k, default)
        number(v, k, low, high)
        if k in {"top_k", "max_tokens"} and type(v) is not int:
            raise ValueError("Speech token budgets must be integers")
    if mode not in {"synthetic", "customVoice"} and not isinstance(request.get("reference"), dict):
        raise ValueError("Add a voice reference")
    if engine == "fishS2Pro" and request.get("language", "auto") != "auto":
        raise ValueError("Fish infers language from the script; use Auto")
    if mode == "customVoice":
        if request.get("reference"):
            raise ValueError(
                "Qwen CustomVoice uses built-in voices; choose Base for reference cloning"
            )
        if not isinstance(request.get("speaker"), str) or not request["speaker"].strip():
            raise ValueError("Choose a Qwen preset speaker")
        instruct = request.get("instruct", "")
        if not isinstance(instruct, str) or len(instruct) > 2000 or "<|" in instruct:
            raise ValueError(
                "Use a delivery instruction of at most 2000 characters without chat markers"
            )
    elif request.get("instruct") or request.get("speaker"):
        raise ValueError("Direct delivery instructions require a Qwen CustomVoice model")
    return request


def validate_checkpoint_request(value, spec):
    if value["engine"] == "qwen3TTS":
        from qwen3_tts_mlx.pipeline import validate_request as validate_qwen

        validate_qwen(
            value, spec["config"], reference_path=(value.get("reference") or {}).get("path")
        )


def dispatch(command, request, output=None, progress=None, cancelled=None):
    if command == "voice-dialogue":
        from studio_dialogue import dispatch as dialogue_dispatch

        return dialogue_dispatch(request, output, progress, cancelled)
    cancelled = cancelled or (lambda: False)

    def report(message):
        if progress:
            progress(dict(event="progress", message=message))

    if command == "voice-catalog":
        from wee_todd_mlx.speech_setup import catalog

        return {"models": [{k: v for k, v in r.items() if k != "files"} for r in catalog()]}
    if command == "voice-download":
        from wee_todd_mlx.speech_setup import download

        if not request.get("directory"):
            raise ValueError("Choose a speech model library folder")
        return download(
            request["model_id"], request["directory"], progress=progress, cancelled=cancelled
        )
    from wee_todd_mlx.speech_checkpoint import inspect

    value = request.get("voice", {})
    if command == "voice-generate":
        value = validate_request(value)
    elif command == "voice-inspect" and not value.get("engine"):
        root = Path(value.get("model_path", "")).expanduser().resolve()
        config = json.loads((root / "config.json").read_text())
        engine = {"fish_qwen3_omni": "fishS2Pro", "qwen3_tts": "qwen3TTS"}.get(
            config.get("model_type")
        )
        if engine is None:
            raise ValueError("Choose an installed Fish S2 Pro or Qwen3-TTS Base/CustomVoice model")
        value = dict(value, engine=engine)
    report("Checking local speech checkpoint")
    spec = inspect(
        value.get("model_path", ""),
        engine=value.get("engine", ""),
        precision=value.get("precision", "auto"),
    )
    if command == "voice-inspect":
        name = "Fish S2 Pro"
        if value["engine"] == "qwen3TTS":
            size = {"1b7": "1.7B", "0b6": "0.6B"}[spec["config"]["tts_model_size"]]
            kind = (
                "CustomVoice" if spec["config"].get("tts_model_type") == "custom_voice" else "Base"
            )
            name = f"Qwen3-TTS {kind} · {size}"
        name += " · " + ("8-bit" if spec["precision"] == "8bit" else "BF16")
        return dict(
            description=f"{name} · "
            + (
                "ready for preset voices and delivery instructions"
                if spec["config"].get("tts_model_type") == "custom_voice"
                else "ready for reference speech"
            ),
            identity=spec["identity"],
            model=dict(
                path=spec["root"],
                engine=value["engine"],
                name=name,
                kind=spec["config"].get("tts_model_type", "base")
                if value["engine"] == "qwen3TTS"
                else "fish",
            ),
        )
    if command != "voice-generate" or output is None:
        raise ValueError("Choose a new folder for the speech take")
    validate_checkpoint_request(value, spec)
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        (root / "request.json").write_text(json.dumps(value, indent=2))
        reference = None
        if value["reference_mode"] not in {"synthetic", "customVoice"}:
            report("Preparing sampled voice reference")
            reference = prepare_reference(
                value["reference"],
                engine=value["engine"],
                mode=value["reference_mode"],
                model_identity=spec["identity"],
                output=root.parent / "reference-cache",
                runtime=request.get("runtime", {}),
                cancelled=cancelled,
            )
        if cancelled():
            raise InterruptedError("Speech generation cancelled")
        if value["engine"] == "fishS2Pro":
            from fish_speech_mlx.pipeline import generate
        else:
            from qwen3_tts_mlx.pipeline import generate
        result = generate(
            value,
            reference_path=reference["path"] if reference else None,
            inspection=spec,
            cancelled=cancelled,
            progress=report,
        )
        import numpy as np
        from scipy.io import wavfile

        audio = np.asarray(result["audio"], dtype=np.float32)
        if (
            audio.ndim != 1
            or not len(audio)
            or not np.isfinite(audio).all()
            or np.max(abs(audio)) < 1e-7
        ):
            raise ValueError("Speech did not produce a finite audible take")
        if cancelled():
            raise InterruptedError("Speech generation cancelled")
        pending = root / "take.pending.wav"
        wavfile.write(pending, result["sample_rate"], audio)
        np.save(root / "codes.npy", result["codes"], allow_pickle=False)
        response = dict(
            audio=str(root / "take.wav"),
            duration=len(audio) / result["sample_rate"],
            frames=len(audio),
            sample_rate=result["sample_rate"],
            channels=1,
            artifacts=str(root),
            truncated=result["truncated"],
            request=value,
        )
        receipt = dict(
            version=1,
            state="complete",
            result=response,
            model_identity=spec["identity"],
            reference=reference,
            audio_sha256=digest_file(pending),
            codes_sha256=digest_file(root / "codes.npy"),
            elapsed=time.monotonic() - started,
        )
        (root / "receipt.pending.json").write_text(json.dumps(receipt, indent=2))
        os.replace(pending, root / "take.wav")
        os.replace(root / "receipt.pending.json", root / "receipt.json")
        return response
    except BaseException as error:
        (root / "failure.json").write_text(
            json.dumps(
                dict(
                    state="cancelled" if isinstance(error, InterruptedError) else "failed",
                    error=str(error),
                )
            )
        )
        raise
