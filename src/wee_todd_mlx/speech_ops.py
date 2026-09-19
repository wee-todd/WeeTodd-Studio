"""Small tensor primitives shared by the independently owned speech engines.

Imported only inside execution; model and codec architecture stay in their adapters.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class Weights:
    def __init__(self, tensors):
        self.w = tensors

    def linear(self, x, name):
        if name + ".scales" in self.w:
            y = mx.quantized_matmul(
                x,
                self.w[name + ".weight"],
                self.w[name + ".scales"],
                self.w[name + ".biases"],
                bits=8,
                group_size=64,
            )
        else:
            y = x @ self.w[name + ".weight"].T
        bias = self.w.get(name + ".bias")
        return y if bias is None else y + bias

    def embedding(self, tokens, name):
        if name + ".scales" in self.w:
            shape = tokens.shape
            indices = tokens.reshape(-1)
            result = mx.dequantize(
                self.w[name + ".weight"][indices],
                self.w[name + ".scales"][indices],
                self.w[name + ".biases"][indices],
                bits=8,
                group_size=64,
            )
            return result.reshape(*shape, result.shape[-1])
        return self.w[name + ".weight"][tokens]

    def norm(self, x, name, eps=1e-6):
        return mx.fast.rms_norm(x, self.w[name + ".weight"], eps)

    def layer_norm(self, x, name, eps=1e-6):
        return mx.fast.layer_norm(x, self.w[name + ".weight"], self.w.get(name + ".bias"), eps)


def rope(x, offset=0, base=1000000.0, *, adjacent=False, rounded=False):
    if not rounded:
        return mx.fast.rope(
            x, dims=x.shape[-1], traditional=adjacent, base=base, scale=1.0, offset=offset
        )
    # Fish stores adjacent-pair trigonometric tables in BF16, even with FP32 input.
    theta = mx.arange(offset, offset + x.shape[-2], dtype=mx.float32)[:, None] * (
        base ** (-mx.arange(0, x.shape[-1], 2, dtype=mx.float32) / x.shape[-1])
    )
    c, s = (
        mx.cos(theta).astype(mx.bfloat16).astype(mx.float32),
        mx.sin(theta).astype(mx.bfloat16).astype(mx.float32),
    )
    pairs = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    return mx.stack([a * c - b * s, a * s + b * c], axis=-1).reshape(x.shape).astype(x.dtype)


def attention(q, k, v, cache=None, mask=None):
    offset = 0 if not cache else cache[0].shape[-2]
    if cache:
        k, v = mx.concatenate([cache[0], k], axis=-2), mx.concatenate([cache[1], v], axis=-2)
    if cache is not None:
        cache[:] = [k, v]
    if mask is None and q.shape[-2] > 1:
        mask = mx.arange(k.shape[-2])[None, :] <= mx.arange(offset, offset + q.shape[-2])[:, None]
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask=mask)


def sample(logits, *, temperature=0.9, top_p=1.0, top_k=50):
    logits = logits.reshape(-1).astype(mx.float32)
    if temperature == 0:
        return int(mx.argmax(logits).item())
    order = mx.argsort(-logits)
    sorted_logits = logits[order] / temperature
    if top_k > 0:
        sorted_logits = mx.where(mx.arange(logits.size) < top_k, sorted_logits, -mx.inf)
    if top_p < 1:
        probabilities = mx.softmax(sorted_logits)
        sorted_logits = mx.where(
            mx.cumsum(probabilities) - probabilities < top_p, sorted_logits, -mx.inf
        )
    return int(order[mx.random.categorical(sorted_logits)].item())


def gated_mlp(weights, x, prefix, names=("w1", "w3", "w2")):
    gate, up, down = names
    return weights.linear(
        nn.silu(weights.linear(x, prefix + "." + gate)) * weights.linear(x, prefix + "." + up),
        prefix + "." + down,
    )
