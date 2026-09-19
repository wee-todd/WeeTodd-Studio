"""Staged process-local Fish inference. No server or external inference wrapper."""

from __future__ import annotations

import gc

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


@serialized
def generate(
    request, *, reference_path=None, inspection=None, cancelled=lambda: False, progress=None
):
    import mlx.core as mx
    import numpy as np
    from scipy.io import wavfile
    from tokenizers import Tokenizer

    from wee_todd_mlx.speech_checkpoint import inspect

    from .codec import FishCodec
    from .model import FishTransformer, build_prompt

    global _active
    spec = inspection or inspect(
        request["model_path"], engine="fishS2Pro", precision=request.get("precision", "auto")
    )

    def stage(message):
        if cancelled():
            raise InterruptedError("Speech generation cancelled")
        if progress:
            progress(message)

    try:
        reference_codes = None
        if reference_path:
            stage("Encoding Fish voice reference")
            rate, audio = wavfile.read(reference_path)
            if rate != 44100 or audio.ndim != 1 or audio.dtype.kind != "f":
                raise ValueError("Fish requires a prepared mono 44,100 Hz float reference")
            _active = FishCodec(mx.load(spec["codec_path"]))
            reference_codes = np.array(_active.encode(audio))
            _release()
        stage("Loading Fish S2 Pro transformer")
        tensors = {}
        for filename in spec["model_files"]:
            tensors.update(mx.load(filename))
        _active = FishTransformer(tensors, spec["config"])
        del tensors
        tokenizer = Tokenizer.from_file(spec["root"] + "/tokenizer.json")
        prompt = build_prompt(
            tokenizer,
            request["text"],
            reference_codes,
            (request.get("reference") or {}).get("transcript", ""),
        )
        sampling = request.get("sampling", {})
        mx.random.seed(int(request.get("seed", 0)))
        stage("Generating Fish speech")
        codes, truncated = _active.generate(
            prompt,
            max_tokens=int(sampling.get("max_tokens", 512)),
            temperature=float(sampling.get("temperature", 0.9)),
            top_p=float(sampling.get("top_p", 1)),
            top_k=int(sampling.get("top_k", 50)),
            cancelled=cancelled,
            progress=lambda count, limit: stage(f"Fish speech: {count} / {limit} frames"),
        )
        _release()
        stage("Decoding Fish speech")
        _active = FishCodec(mx.load(spec["codec_path"]))
        audio = np.array(_active.decode(mx.array(codes)), dtype=np.float32)
        if not np.all(np.isfinite(audio)):
            raise ValueError("Fish decoded non-finite audio")
        return dict(audio=audio, sample_rate=44100, codes=codes, truncated=truncated)
    finally:
        _release()
