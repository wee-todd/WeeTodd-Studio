"""Process-local, explicitly released SheetSage2 inference stages."""

import gc
import time
import traceback

import mlx.core as mx

from ._model import Decoder, encode
from .checkpoint import checkpoint_digest, unchanged
from .symbolic import PROMPTS, Grammar, Vocabulary, decode, to_abc


def run(audio, info, melody_only, progress, cancelled):
    started = time.monotonic()
    weights = {}
    decoder = None
    memory = None
    timings = {}

    def check():
        if cancelled is not None and cancelled():
            raise InterruptedError("SheetSage2 transcription cancelled")

    def report(stage, message, fraction):
        check()
        if progress:
            progress(
                {"event": "progress", "stage": stage, "message": message, "fraction": fraction}
            )

    try:
        report("loading", "Verifying installed SheetSage2 weights", 0.005)
        model_sha256, identity = checkpoint_digest(info["model_path"], check)
        timings["checkpoint_hash_seconds"] = time.monotonic() - started
        report("loading", "Loading installed SheetSage2 weights", 0.01)
        load_started = time.monotonic()
        weights = mx.load(info["model_path"])
        unchanged(info["model_path"], identity)
        for alias in ("output_projection.weight", "decoder.embed_tokens.weight"):
            if alias in weights and not bool(
                mx.all(weights[alias] == weights["token_embedding.weight"]).item()
            ):
                raise ValueError(
                    f"SheetSage2 tied embedding differs from {alias}; "
                    "choose the supported merged checkpoint"
                )
        check()
        timings["load_seconds"] = time.monotonic() - load_started
        begin = time.monotonic()
        memory = encode(audio, weights, check, report)
        timings["encode_seconds"] = time.monotonic() - begin
        # Encoder graphs were materialized at each stage. Only the projected memory
        # is retained; decoder weights are still lazy until its stage begins.
        for key in list(weights):
            if (
                key.startswith("encoder.")
                or key.startswith("encoder_projection.")
                or key == "layer_weight"
            ):
                del weights[key]
        gc.collect()
        mx.clear_cache()
        report("decoding", "Reading melody, harmony, rhythm and structure", 0.76)
        begin = time.monotonic()
        decoder = Decoder(weights, memory)
        memory = None
        vocabulary = Vocabulary()
        grammar = Grammar(vocabulary)
        tokens = list(vocabulary.prefix)
        pending = tokens
        truncated = True
        while len(tokens) < 5120:
            check()
            logits = decoder.step(pending)[0, -1].astype(mx.float32)
            if not bool(mx.all(mx.isfinite(logits)).item()):
                raise RuntimeError(
                    "SheetSage2 produced nonfinite logits; no transcription was saved"
                )
            scores = mx.where(mx.array(grammar.allowed()), logits, -float("inf"))
            token = int(mx.argmax(scores).item())
            tokens.append(token)
            if grammar.accept(token):
                truncated = False
                break
            pending = [token]
            if len(tokens) % 32 == 0:
                report(
                    "decoding",
                    f"Reading score: {len(tokens)} tokens",
                    0.76 + 0.2 * len(tokens) / 5120,
                )
        if truncated:
            raise RuntimeError(
                "SheetSage2 reached its 5,120-token score limit. Choose a shorter reference "
                "excerpt; the score was not silently truncated"
            )
        timings["decode_seconds"] = time.monotonic() - begin
        decoder.close()
        decoder = None
        weights.clear()
        gc.collect()
        mx.clear_cache()
        report("notation", "Preparing editable timed events and melody notation", 0.98)
        decoded = decode(tokens, duration=len(audio) / 24000)
        abc = to_abc(decoded["events"], melody_only=melody_only)
        unchanged(info["model_path"], identity)
        timings["total_seconds"] = time.monotonic() - started
        check()
        return {
            "abc": abc,
            "events": decoded["events"],
            "tokens": tokens,
            "duration_seconds": len(audio) / 24000,
            "sampling_rate": 24000,
            "model": "SheetSage2",
            "backend": "mlx",
            "melody_only": melody_only,
            "prompts": list(PROMPTS),
            "timing": timings,
            "schema_version": "v1",
            "warnings": decoded["warnings"],
            "model_resident": False,
            "model_sha256": model_sha256,
            "peak_memory_bytes": mx.get_peak_memory(),
            "needs_review": True,
        }
    except BaseException as exc:
        # Tracebacks otherwise retain large encoder locals after cancellation.
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        if decoder is not None:
            decoder.close()
        memory = None
        weights.clear()
        gc.collect()
        mx.clear_cache()
