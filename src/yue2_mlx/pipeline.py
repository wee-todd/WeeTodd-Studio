"""Native score → semantic → acoustic → stereo master pipeline.

Public imports remain light. Request-local state is serialized inside this
process; all weights are released on success, failure, and BaseException.
"""

from __future__ import annotations

import gc
import json
import sys
import threading
import time
import traceback
import wave
from pathlib import Path

from .checkpoint import inspect_checkpoint, load_model, load_vae
from .config import validate_request
from .protocol import CODEC_OFFSET, negative_tokens, prefix_tokens

_LOCK = threading.Lock()
_STATE = {"model": None, "vae": None}


def _release():
    """Release components while the caller owns the workload lock."""
    _STATE["model"] = None
    _STATE["vae"] = None
    gc.collect()
    runtime = sys.modules.get("mlx.core")
    if runtime is not None:
        runtime.clear_cache()


def unload():
    """Explicitly release process-local components outside an active workload."""
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("Cannot unload while a YuE2 generation is active")
    try:
        _release()
    finally:
        _LOCK.release()


def _json(filename, value):
    filename.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def _cancel(callback):
    if callback is not None and callback():
        raise InterruptedError("YuE2 generation cancelled")


def write_master(filename, audio):
    """Write verified 32-bit PCM, preserving stereo and the exact sample count."""
    import numpy as np

    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[1] != 2 or len(samples) < 1:
        raise ValueError("YuE2 master must contain nonempty stereo audio")
    if not np.isfinite(samples).all():
        raise FloatingPointError("Decoder produced non-finite audio")
    peak = float(np.max(np.abs(samples)))
    gain = 1.0 / peak if peak > 1.0 else 1.0
    normalized = samples.astype(np.float64) * gain
    pcm = np.rint(normalized * 2147483647).astype("<i4")
    with wave.open(str(filename), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(4)
        stream.setframerate(48000)
        stream.writeframes(pcm.tobytes())
    with wave.open(str(filename), "rb") as stream:
        if (stream.getnframes(), stream.getnchannels(), stream.getframerate()) != (
            len(samples),
            2,
            48000,
        ):
            raise RuntimeError("Written master WAV failed verification")
    return dict(
        peak=peak,
        gain=gain,
        output_peak=peak * gain,
        clipped_samples=0,
        samples_over_full_scale=int(np.count_nonzero(np.abs(samples) > 1)),
    )


def _generate(request: dict, output: Path, progress=None, cancelled=None) -> dict:
    normalized = validate_request(request)
    for callback, name in ((progress, "progress"), (cancelled, "cancelled")):
        if callback is not None and not callable(callback):
            raise ValueError(f"{name} must be callable")
    _cancel(cancelled)
    directory = Path(output).expanduser().resolve()
    if directory.exists():
        raise FileExistsError(f"YuE2 requires a fresh artifact directory: {directory}")
    started = time.perf_counter()
    timings = {}
    last_update = 0.0

    def emit(message, fraction, force=False):
        nonlocal last_update
        now = time.monotonic()
        if progress and (force or now - last_update >= 0.5):
            progress(dict(event="progress", message=message, fraction=float(fraction)))
            last_update = now

    emit("Validating YuE2 model and decoder", 0, True)
    inspection = inspect_checkpoint(
        normalized["model_path"], normalized["vae_path"], normalized["precision"]
    )
    from .tokenizer import Tokenizer

    tokenizer = Tokenizer(inspection["files"]["tokenizer"])
    # Validate prompt budgets before loading the transformer.
    score_ids = tokenizer.encode(normalized["abc"]) if normalized["abc"] is not None else None
    initial_prefix = prefix_tokens(normalized, tokenizer, score_ids)
    validate_context(
        normalized, tokenizer, score_ids, inspection["config"]["max_position_embeddings"]
    )
    _cancel(cancelled)
    directory.mkdir(parents=True, exist_ok=False)
    _json(directory / "request.json", normalized)
    effective = _effective(normalized, inspection, timings, cancelled)
    _json(directory / "effective.json", effective)
    import mlx.core as mx
    import numpy as np

    from .acoustic import initial_noise, synthesize
    from .sampling import generate_tokens

    checkpoint_start = time.perf_counter()
    emit("Loading native YuE2 transformer", 0.02, True)
    _STATE["model"] = load_model(inspection)
    timings["load_transformer"] = time.perf_counter() - checkpoint_start
    _cancel(cancelled)
    truncated = {"abc": False, "semantic": False}
    stage = time.perf_counter()
    if normalized["cot"] != "off" and score_ids is None:
        emit("Composing the ABC score", 0.05, True)
        score_ids, truncated["abc"] = generate_tokens(
            _STATE["model"],
            initial_prefix,
            normalized["abc_sampling"],
            normalized["seed"],
            "abc",
            cancelled=cancelled,
            progress=lambda n, total: emit(f"ABC score: {n} tokens", 0.05 + 0.2 * n / total),
        )
    score_ids = score_ids or []
    score = tokenizer.decode(score_ids)
    (directory / "score.abc").write_text(score, encoding="utf-8")
    timings["abc"] = time.perf_counter() - stage
    prefix = prefix_tokens(normalized, tokenizer, score_ids)
    negative = (
        negative_tokens(normalized, tokenizer, score_ids) if normalized["cfg_scale"] != 1 else None
    )
    validate_context(
        normalized, tokenizer, score_ids, inspection["config"]["max_position_embeddings"]
    )
    stage = time.perf_counter()
    emit("Generating semantic music tokens", 0.25, True)
    semantic, truncated["semantic"] = generate_tokens(
        _STATE["model"],
        prefix,
        normalized["semantic_sampling"],
        normalized["seed"],
        "semantic",
        negative=negative,
        guidance=normalized["cfg_scale"],
        legacy_off=normalized["cot"] == "off",
        cancelled=cancelled,
        progress=lambda n, total: emit(f"Semantic music: {n} frames", 0.25 + 0.3 * n / total),
    )
    timings["semantic"] = time.perf_counter() - stage
    codec = [token - CODEC_OFFSET for token in semantic]
    _json(
        directory / "tokens.json",
        dict(
            abc=score_ids,
            prefix=prefix,
            negative_prefix=negative,
            semantic=semantic,
            codec=codec,
            truncated=truncated,
        ),
    )
    if not codec:
        raise RuntimeError("Semantic stage produced no music frames")
    _cancel(cancelled)
    noise = initial_noise(len(codec), 64, normalized["seed"])
    mx.eval(noise)
    np.save(directory / "noise.npy", np.array(noise))
    stage = time.perf_counter()
    emit("Synthesizing acoustic latents", 0.55, True)
    latents = synthesize(
        _STATE["model"],
        prefix,
        codec,
        noise,
        normalized["steps"],
        cancelled=cancelled,
        progress=lambda n, total: emit(
            f"Acoustic midpoint step {n}/{total}", 0.55 + 0.35 * n / total
        ),
    )
    latent_array = np.array(latents, dtype=np.float32)
    if latent_array.shape != (len(codec), 64) or not np.isfinite(latent_array).all():
        raise FloatingPointError("Acoustic stage produced invalid latents")
    np.save(directory / "latents.npy", latent_array)
    timings["acoustic"] = time.perf_counter() - stage
    if normalized["memory_mode"] == "staged":
        _STATE["model"] = None
        gc.collect()
        mx.clear_cache()
    _cancel(cancelled)
    stage = time.perf_counter()
    emit("Loading stereo audio decoder", 0.9, True)
    _STATE["vae"] = load_vae(inspection)
    timings["load_vae"] = time.perf_counter() - stage
    stage = time.perf_counter()
    audio = _STATE["vae"].decode_tiled(
        latents,
        cancelled=cancelled,
        progress=lambda n, total: emit(
            f"Decoding audio: {n}/{total} frames", 0.9 + 0.09 * n / total
        ),
    )
    samples = np.array(audio, dtype=np.float32)
    if samples.shape != (len(codec) * 1920 - 64, 2):
        raise RuntimeError(f"Unexpected natural audio shape {samples.shape}")
    _cancel(cancelled)
    master = directory / "master.wav"
    master_metadata = write_master(master, samples)
    timings["decode"] = time.perf_counter() - stage
    timings["total"] = time.perf_counter() - started
    result = dict(
        audio=str(master),
        duration=len(samples) / 48000,
        sample_rate=48000,
        channels=2,
        artifacts=str(directory),
        truncated=truncated,
        timings=timings,
        request=normalized,
        **master_metadata,
        precision=inspection["precision"],
    )
    result["stage"] = "generate"
    _finish(directory, result, "generate", started, cancelled)
    emit("Stereo master ready", 1, True)
    return result


def validate_context(request, tokenizer, score_ids, context):
    """Reserve all requested score and semantic tokens before loading weights."""
    initial = prefix_tokens(request, tokenizer, score_ids)
    semantic = request["semantic_sampling"]["max_tokens"]
    if request["cot"] != "off" and score_ids is None:
        score_budget = request["abc_sampling"]["max_tokens"]
        needed = len(initial) + score_budget + 2 + semantic
        negative_length = len(negative_tokens(request, tokenizer, [])) + score_budget
    else:
        needed = len(initial) + semantic
        negative_length = len(negative_tokens(request, tokenizer, score_ids or []))
    if needed > context or (request["cfg_scale"] != 1 and negative_length + semantic > context):
        reserved = max(needed - semantic, negative_length if request["cfg_scale"] != 1 else 0)
        available = max(0, context - reserved)
        raise ValueError(
            f"Requested score and semantic budgets exceed the YuE2 context ({context} tokens). "
            f"With these lyrics, style and score settings, allow at most {available} music tokens "
            f"({available / 25:.2f} seconds). Lower the Length budget or score token budget. "
            "This is a maximum allowance; songs can finish earlier."
        )


def _clear_exception_frames(error):
    """Clear completed frames in the complete exception chain, preserving locations."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        pending.extend([current.__cause__, current.__context__])
        pending.extend(getattr(current, "exceptions", ()))


def _execute(operation, output, progress=None, cancelled=None):
    for callback, name in ((progress, "progress"), (cancelled, "cancelled")):
        if callback is not None and not callable(callback):
            raise ValueError(f"{name} must be callable")
    _cancel(cancelled)
    directory = Path(output).expanduser().resolve()
    if directory.exists():
        raise FileExistsError(f"YuE2 requires a fresh artifact directory: {directory}")
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("A YuE2 model workload is already active in this process")
    try:
        return operation(directory)
    except BaseException as error:
        # operation's weighted frames have now unwound; clearing only _STATE
        # would leave their model/cache locals owned by a retained exception.
        _clear_exception_frames(error)
        if directory.is_dir():
            try:
                _json(
                    directory / "failure.json", dict(error=type(error).__name__, message=str(error))
                )
            except OSError:
                pass
        raise
    finally:
        _release()
        _LOCK.release()


def generate(request: dict, output: Path, progress=None, cancelled=None) -> dict:
    normalized = validate_request(request)
    return _execute(
        lambda directory: _generate(normalized, directory, progress, cancelled),
        output,
        progress,
        cancelled,
    )


def _effective(request, inspection, timings, cancelled=None, *, models=None):
    from .artifacts import checkpoint_fingerprints, engine_identity

    start = time.perf_counter()
    fingerprints = checkpoint_fingerprints(inspection, cancelled) if models is None else models
    timings["checkpoint_hash"] = time.perf_counter() - start
    identity = engine_identity()
    return dict(
        engine="weetodd-native-mlx-yue2",
        protocol="yue2-native-v1",
        engine_revision=identity["source_sha256"],
        runtime=identity["runtime"],
        precision=inspection["precision"],
        layout=inspection["layout"],
        models=fingerprints,
        sampling={"abc": request["abc_sampling"], "semantic": request["semantic_sampling"]},
        steps=request["steps"],
        solver="midpoint",
        guidance=request["cfg_scale"],
        memory_mode=request["memory_mode"],
        sample_rate=48000,
        channels=2,
        noise_rng="mlx-threefry-request-local",
        cache="bounded-block-256",
        checkpoint_generation_config=inspection["generation_config"],
    )


def _finish(directory, result, stage, started, cancelled):
    from .artifacts import digest, seal

    _json(directory / "result.json", result)
    before = time.perf_counter()
    manifest = seal(directory, stage, cancelled)
    result["timings"]["artifact_hash"] = time.perf_counter() - before
    result["timings"]["total"] = time.perf_counter() - started
    _json(directory / "result.json", result)
    manifest["files"]["result.json"] = dict(
        sha256=digest(directory / "result.json"), bytes=(directory / "result.json").stat().st_size
    )
    _json(directory / "manifest.json", manifest)


def _emitter(progress):
    last = 0.0

    def emit(message, fraction, force=False):
        nonlocal last
        now = time.monotonic()
        if progress and (force or now - last >= 0.5):
            progress(dict(event="progress", message=message, fraction=float(fraction)))
            last = now

    return emit


def _plan(request, directory, progress, cancelled):
    from .tokenizer import Tokenizer

    started = time.perf_counter()
    timings = {}
    emit = _emitter(progress)
    emit("Validating score plan", 0, True)
    inspection = inspect_checkpoint(
        request["model_path"], request["vae_path"], request["precision"]
    )
    tokenizer = Tokenizer(inspection["files"]["tokenizer"])
    score_ids = tokenizer.encode(request["abc"]) if request["abc"] is not None else None
    validate_context(request, tokenizer, score_ids, inspection["config"]["max_position_embeddings"])
    directory.mkdir(parents=True, exist_ok=False)
    _json(directory / "request.json", request)
    _json(directory / "effective.json", _effective(request, inspection, timings, cancelled))
    truncated = {"abc": False, "semantic": False}
    if score_ids is None:
        from .sampling import generate_tokens

        _cancel(cancelled)
        mark = time.perf_counter()
        emit("Loading native score transformer", 0.05, True)
        _STATE["model"] = load_model(inspection)
        timings["load_transformer"] = time.perf_counter() - mark
        mark = time.perf_counter()
        score_ids, truncated["abc"] = generate_tokens(
            _STATE["model"],
            prefix_tokens(request, tokenizer),
            request["abc_sampling"],
            request["seed"],
            "abc",
            cancelled=cancelled,
            progress=lambda n, total: emit(f"ABC score: {n} tokens", 0.1 + 0.85 * n / total),
        )
        timings["abc"] = time.perf_counter() - mark
    score = tokenizer.decode(score_ids)
    (directory / "score.abc").write_text(score, encoding="utf-8")
    tokens = dict(
        abc=score_ids,
        prefix=prefix_tokens(request, tokenizer, score_ids),
        negative_prefix=negative_tokens(request, tokenizer, score_ids)
        if request["cfg_scale"] != 1
        else None,
        semantic=[],
        codec=[],
        truncated=truncated,
    )
    _json(directory / "tokens.json", tokens)
    result = dict(
        stage="plan",
        abc=score,
        score=str(directory / "score.abc"),
        artifacts=str(directory),
        request=request,
        timings=timings,
        truncated=truncated,
        precision=inspection["precision"],
    )
    _finish(directory, result, "plan", started, cancelled)
    emit("Score plan ready", 1, True)
    return result


def plan(request: dict, output: Path, progress=None, cancelled=None) -> dict:
    """Compose and preserve an ABC plan without semantic/acoustic/VAE work."""
    normalized = validate_request(request)
    if normalized["cot"] == "off":
        raise ValueError("Score planning requires cot full or melody")
    return _execute(
        lambda directory: _plan(normalized, directory, progress, cancelled),
        output,
        progress,
        cancelled,
    )


def _replay(source, directory, progress, cancelled, *, steps=None, seed=None, decode_only=False):
    from .artifacts import copy_verified, verify

    started = time.perf_counter()
    timings = {}
    emit = _emitter(progress)
    emit("Verifying source artifacts and checkpoint contents", 0, True)
    checked = verify(source, cancelled)
    if checked["manifest"]["stage"] == "plan":
        raise ValueError(
            "Acoustic replay requires semantic tokens and saved noise, not a score plan"
        )
    request = checked["request"].copy()
    if steps is not None:
        request["steps"] = steps
    if seed is not None:
        request["seed"] = seed
    request = validate_request(request)
    inspection = checked["inspection"]
    tokens = checked["tokens"]
    source_dir = checked["directory"]
    timings["source_verification"] = checked["verification_seconds"]
    effective = _effective(
        request, inspection, timings, cancelled, models=checked["effective"]["models"]
    )
    effective["semantic_request"] = checked["effective"].get("semantic_request", checked["request"])
    effective["source"] = dict(
        artifacts=str(source_dir),
        manifest_sha256=checked["manifest_sha256"],
        operation="decode_latents" if decode_only else "resynthesize",
        noise="saved" if seed is None else "regenerated",
    )
    directory.mkdir(parents=True, exist_ok=False)
    _json(directory / "request.json", request)
    _json(directory / "effective.json", effective)
    _json(directory / "tokens.json", tokens)
    copy_verified(checked, "score.abc", directory / "score.abc", cancelled)
    import mlx.core as mx
    import numpy as np

    from .acoustic import initial_noise, synthesize

    frames = len(tokens["codec"])
    _cancel(cancelled)
    if seed is None:
        copy_verified(checked, "noise.npy", directory / "noise.npy", cancelled)
        noise = mx.array(np.load(directory / "noise.npy", allow_pickle=False))
    else:
        noise = initial_noise(frames, 64, request["seed"])
        np.save(directory / "noise.npy", np.array(noise))
    if decode_only:
        copy_verified(checked, "latents.npy", directory / "latents.npy", cancelled)
        latents = mx.array(np.load(directory / "latents.npy", allow_pickle=False))
    else:
        emit("Loading native acoustic transformer", 0.1, True)
        mark = time.perf_counter()
        _STATE["model"] = load_model(inspection)
        timings["load_transformer"] = time.perf_counter() - mark
        mark = time.perf_counter()
        latents = synthesize(
            _STATE["model"],
            tokens["prefix"],
            tokens["codec"],
            noise,
            request["steps"],
            cancelled=cancelled,
            progress=lambda n, total: emit(
                f"Acoustic midpoint step {n}/{total}", 0.15 + 0.7 * n / total
            ),
        )
        values = np.array(latents, dtype=np.float32)
        if values.shape != (frames, 64) or not np.isfinite(values).all():
            raise FloatingPointError("Replayed acoustic stage produced invalid latents")
        np.save(directory / "latents.npy", values)
        timings["acoustic"] = time.perf_counter() - mark
        if request["memory_mode"] == "staged":
            _STATE["model"] = None
            gc.collect()
            mx.clear_cache()
    _cancel(cancelled)
    emit("Decoding verified acoustic latents", 0.85, True)
    mark = time.perf_counter()
    _STATE["vae"] = load_vae(inspection)
    timings["load_vae"] = time.perf_counter() - mark
    mark = time.perf_counter()
    samples = np.array(
        _STATE["vae"].decode_tiled(
            latents,
            cancelled=cancelled,
            progress=lambda n, total: emit(
                f"Decoding audio: {n}/{total} frames", 0.85 + 0.14 * n / total
            ),
        ),
        dtype=np.float32,
    )
    if samples.shape != (frames * 1920 - 64, 2):
        raise RuntimeError("Replayed decoder produced an unexpected natural sample count")
    _cancel(cancelled)
    metadata = write_master(directory / "master.wav", samples)
    timings["decode"] = time.perf_counter() - mark
    stage = "decode_latents" if decode_only else "resynthesize"
    result = dict(
        stage=stage,
        audio=str(directory / "master.wav"),
        duration=len(samples) / 48000,
        sample_rate=48000,
        channels=2,
        artifacts=str(directory),
        truncated=tokens["truncated"],
        timings=timings,
        request=request,
        precision=inspection["precision"],
        **metadata,
    )
    _finish(directory, result, stage, started, cancelled)
    emit("Stereo master ready", 1, True)
    return result


def resynthesize(
    source_artifacts, output: Path, *, steps=None, seed=None, progress=None, cancelled=None
) -> dict:
    """Replay saved semantic tokens; reuse exact saved noise unless seed is supplied."""
    from .config import integer

    if steps is not None:
        integer(steps, "steps", 1, 4096)
    if seed is not None:
        integer(seed, "seed", 0, 2**63 - 1)
    return _execute(
        lambda directory: _replay(
            source_artifacts, directory, progress, cancelled, steps=steps, seed=seed
        ),
        output,
        progress,
        cancelled,
    )


def decode_latents(source_artifacts, output: Path, progress=None, cancelled=None) -> dict:
    """Re-decode integrity-checked latents without loading transformer weights."""
    return _execute(
        lambda directory: _replay(
            source_artifacts, directory, progress, cancelled, decode_only=True
        ),
        output,
        progress,
        cancelled,
    )
