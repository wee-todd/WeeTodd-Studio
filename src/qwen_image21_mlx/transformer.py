"""Native MLX single-stream Qwen 2.1 blocks and bounded block-causal attention.

The checkpoint's public tensor/configuration contract is the reference. The
implementation uses channels-last MLX operations, request-local caches and
query chunking; it has no dependency on the reference PyTorch runtime.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def token_layout(image_mask, shapes):
    mask = np.asarray(image_mask, dtype=bool)
    image_positions = np.flatnonzero(mask)
    if sum(h * w for h, w in shapes) != len(image_positions):
        raise ValueError("Image token count does not match latent grids")
    ids = np.full(len(mask), -1, dtype=np.int32)
    positions = np.zeros((len(mask), 3), dtype=np.float32)
    cursor = position = offset = 0
    for block, (height, width) in enumerate(shapes):
        count = height * width
        start = int(image_positions[offset])
        if not np.array_equal(
            image_positions[offset : offset + count], np.arange(start, start + count)
        ):
            raise ValueError("An image latent grid must occupy contiguous tokens")
        text = start - cursor
        positions[cursor:start] = np.arange(position, position + text)[:, None]
        position += text
        positions[start : start + count, 0] = position
        positions[start : start + count, 1] = np.repeat(
            np.arange(height) - (height - height // 2), width
        )
        positions[start : start + count, 2] = np.tile(
            np.arange(width) - (width - width // 2), height
        )
        ids[start : start + count] = block
        cursor = start + count
        offset += count
        position += max(height, width)
    positions[cursor:] = np.arange(position, position + len(mask) - cursor)[:, None]
    return ids, positions


def block_attention(q, k, v, image_ids, *, query_offset=0, chunk_size=1024):
    """Exact segment attention; image queries need no mask, text uses causal SDPA.

    Each image attends its complete block and all preceding tokens. Contiguous
    text attends preceding tokens plus its causal prefix. This representation
    never constructs a dense sequence-square mask, including during prefill.
    """
    ids = np.asarray(image_ids)
    outputs = []
    cursor = query_offset
    stop = query_offset + q.shape[2]
    while cursor < stop:
        block_id = ids[cursor]
        end = cursor + 1
        while end < len(ids) and ids[end] == block_id:
            end += 1
        for start in range(cursor, min(end, stop), chunk_size):
            query_end = min(start + chunk_size, end, stop)
            key_end = end if block_id >= 0 else query_end
            output = mx.fast.scaled_dot_product_attention(
                q[:, :, start - query_offset : query_end - query_offset],
                k[:, :, :key_end],
                v[:, :, :key_end],
                scale=q.shape[-1] ** -0.5,
                mask=None if block_id >= 0 else "causal",
            )
            mx.eval(output)
            outputs.append(output)
        cursor = end
    return mx.concatenate(outputs, axis=2)


def rotary(x, angles):
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    cos, sin = mx.cos(angles)[None, None], mx.sin(angles)[None, None]
    a, b = pairs[..., 0].astype(mx.float32), pairs[..., 1].astype(mx.float32)
    return (
        mx.stack([a * cos - b * sin, a * sin + b * cos], axis=-1).reshape(x.shape).astype(x.dtype)
    )


class ZeroCenteredRMS(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x):
        value = x.astype(mx.float32)
        return (
            value
            * mx.rsqrt(mx.mean(value * value, axis=-1, keepdims=True) + self.eps)
            * (self.weight.astype(mx.float32) + 1)
        ).astype(x.dtype)


class TextProjection(nn.Module):
    def __init__(self, source, dim, eps):
        super().__init__()
        self.text_norm = ZeroCenteredRMS(source, eps)
        self.in_layer = nn.Linear(source, dim, bias=False)
        self.out_layer = nn.Linear(dim, dim, bias=False)

    def __call__(self, x):
        return self.out_layer(nn.gelu_approx(self.in_layer(self.text_norm(x))))


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.timestep_embedder = {
            "linear_1": nn.Linear(256, dim, bias=False),
            "linear_2": nn.Linear(dim, dim, bias=False),
        }

    def __call__(self, times, dtype):
        angles = (
            times[:, None].astype(mx.float32)
            * 1000
            * mx.exp(-math.log(10000) * mx.arange(128) / 128)
        )
        value = mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1).astype(dtype)
        return self.timestep_embedder["linear_2"](
            nn.silu(self.timestep_embedder["linear_1"](value))
        )


class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, eps):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = [nn.Linear(dim, dim, bias=False)]
        self.norm_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps)

    def __call__(self, x, angles, ids, prefix, previous):
        shape = (x.shape[0], x.shape[1], self.heads, self.head_dim)
        q = rotary(self.norm_q(self.to_q(x).reshape(shape)).transpose(0, 2, 1, 3), angles)
        k = rotary(self.norm_k(self.to_k(x).reshape(shape)).transpose(0, 2, 1, 3), angles)
        v = self.to_v(x).reshape(shape).transpose(0, 2, 1, 3)
        if previous is not None:
            k = mx.concatenate([previous[0], k], axis=2)
            v = mx.concatenate([previous[1], v], axis=2)
        output = block_attention(q, k, v, ids, query_offset=prefix if previous is not None else 0)
        saved = (mx.contiguous(k[:, :, :prefix]), mx.contiguous(v[:, :, :prefix]))
        return self.to_out[0](output.transpose(0, 2, 1, 3).reshape(x.shape)), saved


class FeedForward(nn.Module):
    def __init__(self, dim, ratio):
        super().__init__()
        self.proj = nn.Linear(dim, dim * ratio, bias=False)
        self.gate_layer = nn.Linear(dim, dim * ratio, bias=False)
        self.out = nn.Linear(dim * ratio, dim, bias=False)

    def __call__(self, x):
        return self.out(nn.silu(self.gate_layer(x)) * self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, heads, head_dim, ratio, eps):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.img_norm2 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.attn = Attention(dim, heads, head_dim, eps)
        self.img_mlp = FeedForward(dim, ratio)

    def __call__(self, x, modulation, angles, ids, prefix, previous):
        s1, g1, s2, g2 = mx.split(modulation, 4, axis=-1)
        attended, saved = self.attn(self.img_norm1(x) * (1 + s1), angles, ids, prefix, previous)
        x = x + mx.tanh(g1) * attended
        x = x + mx.tanh(g2) * self.img_mlp(self.img_norm2(x) * (1 + s2))
        if x.dtype == mx.float16:
            x = mx.clip(x, -65504, 65504)
        return x, saved


class OutputNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=eps, affine=False)

    def __call__(self, x, time):
        return self.norm(x) * (1 + self.linear(nn.silu(time)))


class QwenImage21Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        heads, head_dim = config["num_attention_heads"], config["attention_head_dim"]
        dim = heads * head_dim
        eps = config.get("eps", 1e-6)
        self.axes = config["axes_dims_rope"]
        self.img_in = nn.Linear(config["in_channels"], dim, bias=False)
        self.txt_in = TextProjection(config["context_in_dim"], dim, eps)
        self.time_text_embed = TimeEmbedding(dim)
        self.modulation = [nn.Identity(), nn.Linear(dim, 4 * dim, bias=False)]
        self.transformer_blocks = [
            Block(dim, heads, head_dim, config.get("mlp_ratio", 3), eps)
            for _ in range(config["num_layers"])
        ]
        self.norm_out = OutputNorm(dim, eps)
        self.proj_out = nn.Linear(
            dim, config.get("out_channels", config["in_channels"]), bias=False
        )

    def __call__(
        self,
        latents,
        conditioning,
        timestep,
        *,
        cache=None,
        cancel=lambda: False,
        progress=lambda completed, total: None,
    ):
        features = conditioning["features"]
        target_tokens = math.prod(conditioning["shapes"][-1])
        mask = np.concatenate([conditioning["imageMask"], np.ones(target_tokens // 4, dtype=bool)])
        repeats = np.where(mask, 4, 1)
        expanded = np.repeat(mask, repeats)
        ids, positions = token_layout(expanded, conditioning["shapes"])
        prefix = len(ids) - target_tokens
        cached = cache is not None and bool(cache)
        angles = mx.array(
            np.concatenate(
                [
                    positions[:, i : i + 1]
                    * np.power(10000.0, -np.arange(0, dim, 2, dtype=np.float32) / dim)[None]
                    for i, dim in enumerate(self.axes)
                ],
                axis=-1,
            )
        )
        if cached:
            x = self.img_in(latents[:, -target_tokens:])
            angles = angles[prefix:]
            row = mx.zeros(target_tokens, dtype=mx.int32)
        else:
            encoded = self.txt_in(features)
            source = mx.concatenate(
                [encoded, mx.zeros((1, target_tokens // 4, encoded.shape[-1]), encoded.dtype)],
                axis=1,
            )
            x = source[:, mx.array(np.repeat(np.arange(len(mask)), repeats))]
            x[:, mx.array(np.flatnonzero(expanded))] = self.img_in(latents)
            row = mx.array(np.where(np.arange(len(ids)) >= prefix, 0, 1))
        time = self.time_text_embed(mx.array([timestep, 0], dtype=x.dtype), x.dtype)
        modulation = self.modulation[1](nn.silu(time))[row][None]
        for i, block in enumerate(self.transformer_blocks):
            if cancel():
                raise InterruptedError("Qwen sampling cancelled")
            x, saved = block(x, modulation, angles, ids, prefix, cache.get(i) if cached else None)
            mx.eval(x)
            progress(i + 1, len(self.transformer_blocks))
            if cache is not None and not cached:
                cache[i] = saved
                mx.eval(*saved)
        return self.proj_out(self.norm_out(x[:, -target_tokens:], time[0:1, None]))
