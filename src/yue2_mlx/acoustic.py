"""Cached acoustic conditioning and the checkpoint's reverse-time midpoint ODE."""

import math

import mlx.core as mx

from .protocol import CODEC_OFFSET, MUSIC_END, chunk_ranges
from .sampling import key_for_seed


def check_cancelled(callback):
    if callback is not None and callback():
        raise InterruptedError("YuE2 generation cancelled")


def logit_time(t):
    return (
        max(-20.0, min(20.0, math.log(t / (1 - t)))) if 0 < t < 1 else (20.0 if t >= 1 else -20.0)
    )


def midpoint(velocity, initial, steps, *, cancelled=None, progress=None):
    if type(steps) is not int or steps < 1:
        raise ValueError("Midpoint requires positive integer steps")
    state = initial
    dt = 1.0 / steps
    for i in range(steps):
        check_cancelled(cancelled)
        first = velocity(state, logit_time(1 - i * dt))
        half = state - (first.astype(mx.float32) * (dt / 2)).astype(state.dtype)
        check_cancelled(cancelled)
        second = velocity(half, logit_time(1 - (i + 0.5) * dt))
        state = state - (second.astype(mx.float32) * dt).astype(state.dtype)
        mx.eval(state)
        if progress:
            progress(i + 1, steps)
    return state


def initial_noise(frames, dimensions, seed):
    return mx.random.normal((frames, dimensions), dtype=mx.float32, key=key_for_seed(seed))


def synthesize(model, prefix, codec, noise, steps, *, cancelled=None, progress=None):
    ranges = chunk_ranges(len(codec), len(prefix), model.config["max_position_embeddings"])
    outputs = []
    for index, (start, end) in enumerate(ranges):
        check_cancelled(cancelled)
        tokens = prefix + [CODEC_OFFSET + i for i in codec[start:end]] + [MUSIC_END]
        cache = model.acoustic_prefix(tokens, cancelled)
        length = len(tokens)
        try:

            def velocity(state, t, cache=cache, length=length):
                return model.velocity(state, t, cache, length, cancelled)

            def update(i, n, index=index):
                if progress:
                    progress(index * n + i, len(ranges) * n)

            state = midpoint(
                velocity,
                noise[start:end].astype(mx.bfloat16),
                steps,
                cancelled=cancelled,
                progress=update,
            )
            outputs.append(state.astype(mx.float32))
            mx.eval(outputs[-1])
        finally:
            cache.clear()
    return mx.concatenate(outputs, axis=0)
