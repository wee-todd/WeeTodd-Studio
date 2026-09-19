"""Request-local random sampling and the native frequency-powered penalties."""

import mlx.core as mx

from .protocol import ABC_END, CODEC_OFFSET, CODEC_SIZE, EOD, MUSIC_END


def key_for_seed(seed):
    return mx.array([seed >> 32, seed & 0xFFFFFFFF], dtype=mx.uint32)


def guided_logits(conditional, unconditional, scale):
    if unconditional is None:
        return conditional
    dtype = conditional.dtype
    delta = (conditional - unconditional).astype(dtype)
    return unconditional + (delta.astype(mx.float32) * mx.array(scale, mx.float32)).astype(dtype)


def filter_logits(logits, settings, history, step, phase, legacy_off=False):
    if phase not in ("abc", "semantic"):
        raise ValueError("Unknown token generation phase")
    x = logits if legacy_off else logits.astype(mx.float32)
    ids = mx.arange(x.shape[-1])
    eos = ABC_END if phase == "abc" else MUSIC_END
    allowed = (
        ids < EOD if phase == "abc" else (ids >= CODEC_OFFSET) & (ids < CODEC_OFFSET + CODEC_SIZE)
    )
    allowed = allowed | ((ids == eos) & (step >= settings["min_tokens"]))
    x = mx.where(allowed, x, mx.array(-mx.inf, x.dtype))
    recent = history[-settings["penalty_window"] :]
    if recent and settings["repetition_penalty"] != 1:
        counts = mx.zeros(x.shape, x.dtype).at[mx.array(recent, dtype=mx.int32)].add(1)
        multiplier = mx.power(
            mx.array(settings["repetition_penalty"], mx.float32), counts.astype(mx.float32)
        ).astype(x.dtype)
        x = mx.where(x < 0, x * multiplier, x / multiplier)
    if settings["temperature"] == 0:
        return x
    x = (x.astype(mx.float32) / settings["temperature"]).astype(x.dtype)
    k = min(settings["top_k"], x.shape[-1])
    threshold = mx.min(mx.topk(x, k))
    x = mx.where(x >= threshold, x, mx.array(-mx.inf, x.dtype))
    if settings["top_p"] < 1:
        order = mx.argsort(-x)
        sorted_x = x[order]
        p = mx.softmax(sorted_x, precise=True)
        mass = (mx.cumsum(p) - p).astype(x.dtype)
        remove = (mass.astype(mx.float32) > settings["top_p"]) & (
            mx.arange(x.shape[-1]) >= (3 if legacy_off else 1)
        )
        filtered = mx.where(remove, mx.array(-mx.inf, x.dtype), sorted_x)
        x = mx.put_along_axis(x, order, filtered, axis=0)
    return x


def generate_tokens(
    model,
    prefix,
    settings,
    seed,
    phase,
    *,
    negative=None,
    guidance=1.0,
    legacy_off=False,
    progress=None,
    cancelled=None,
):
    from .acoustic import check_cancelled

    if guidance != 1 and negative is None:
        raise ValueError("Guidance requires a negative prompt")
    capacity = len(prefix) + settings["max_tokens"]
    caches = model.make_cache(capacity)
    other_cache = (
        model.make_cache(len(negative) + settings["max_tokens"]) if guidance != 1 else None
    )
    check_cancelled(cancelled)
    positive = model.ar(mx.array([prefix], dtype=mx.int32), caches)[0]
    negative_logits = (
        model.ar(mx.array([negative], dtype=mx.int32), other_cache)[0] if other_cache else None
    )
    key = key_for_seed(seed)
    history = []
    eos = ABC_END if phase == "abc" else MUSIC_END
    for index in range(settings["max_tokens"]):
        check_cancelled(cancelled)
        scores = filter_logits(
            guided_logits(positive, negative_logits, guidance),
            settings,
            history,
            index,
            phase,
            legacy_off,
        )
        if settings["temperature"] == 0:
            selected = mx.argmax(scores)
        else:
            key, draw = mx.random.split(key)
            probabilities = mx.softmax(scores, precise=True)
            selected = mx.random.categorical(mx.log(probabilities.astype(mx.float32)), key=draw)
        token = int(selected.item())
        if token == eos:
            return history, False
        history.append(token)
        if progress and (index % 32 == 0 or index + 1 == settings["max_tokens"]):
            progress(index + 1, settings["max_tokens"])
        if index + 1 < settings["max_tokens"]:
            token_array = selected.reshape(1, 1)
            positive = model.ar(token_array, caches)[0]
            if other_cache:
                negative_logits = model.ar(token_array, other_cache)[0]
    return history, True
