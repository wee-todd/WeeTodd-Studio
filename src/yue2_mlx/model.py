"""Native dual-expert transformer. Only this execution module imports MLX.

The parameter names form the merged MLX checkpoint ABI. Each layer has a causal
text/semantic expert and a bidirectional acoustic expert. Cached acoustic prefix
keys are invariant throughout the flow solve, so the causal expert runs once.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def silu(x):
    return (x.astype(mx.float32) * mx.sigmoid(x.astype(mx.float32))).astype(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = mx.ones((size,))
        self.eps = eps

    def __call__(self, x):
        inverse = mx.rsqrt(
            mx.mean(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True) + self.eps
        ).astype(x.dtype)
        return (x * inverse).astype(x.dtype) * self.weight


def rotary(x, offset, base):
    """Split-half rotation, with source-dtype rounding at tensor operations."""
    half = x.shape[-1] // 2
    positions = mx.arange(offset, offset + x.shape[-2], dtype=mx.float32)
    frequency = mx.power(mx.array(base, mx.float32), -mx.arange(half, dtype=mx.float32) / half)
    angles = positions[:, None] * frequency[None, :]
    cosine, sine = mx.cos(angles).astype(x.dtype), mx.sin(angles).astype(x.dtype)
    a, b = x[..., :half], x[..., half:]
    return mx.concatenate(
        [
            (a * cosine).astype(x.dtype) - (b * sine).astype(x.dtype),
            (b * cosine).astype(x.dtype) + (a * sine).astype(x.dtype),
        ],
        axis=-1,
    )


class KVCache:
    """Block-grown cache with a hard request-local token capacity."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.offset = 0
        self.keys = self.values = None

    def append(self, keys, values):
        end = self.offset + keys.shape[2]
        if end > self.capacity:
            raise ValueError("KV cache capacity exceeded")
        if self.keys is None or end > self.keys.shape[2]:
            size = min(self.capacity, ((end + 255) // 256) * 256)
            shape = (*keys.shape[:2], size, keys.shape[-1])
            new_keys, new_values = mx.zeros(shape, keys.dtype), mx.zeros(shape, values.dtype)
            if self.offset:
                new_keys[:, :, : self.offset] = self.keys[:, :, : self.offset]
                new_values[:, :, : self.offset] = self.values[:, :, : self.offset]
            self.keys, self.values = new_keys, new_values
        self.keys[:, :, self.offset : end] = keys
        self.values[:, :, self.offset : end] = values
        self.offset = end
        return self.keys[:, :, :end], self.values[:, :, :end]


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.width = config["head_dim"]
        self.base = config["rope_theta"]
        hidden = config["hidden_size"]
        self.q_proj = nn.Linear(hidden, self.heads * self.width, bias=False)
        self.k_proj = nn.Linear(hidden, self.kv_heads * self.width, bias=False)
        self.v_proj = nn.Linear(hidden, self.kv_heads * self.width, bias=False)
        self.o_proj = nn.Linear(self.heads * self.width, hidden, bias=False)
        self.q_norm = RMSNorm(self.width, config["rms_norm_eps"])
        self.k_norm = RMSNorm(self.width, config["rms_norm_eps"])

    def project(self, x, offset):
        b, t, _ = x.shape
        query = self.q_norm(self.q_proj(x).reshape(b, t, self.heads, self.width))
        key = self.k_norm(self.k_proj(x).reshape(b, t, self.kv_heads, self.width))
        value = self.v_proj(x).reshape(b, t, self.kv_heads, self.width).transpose(0, 2, 1, 3)
        return (
            rotary(query.transpose(0, 2, 1, 3), offset, self.base),
            rotary(key.transpose(0, 2, 1, 3), offset, self.base),
            value,
        )

    def attend(self, query, keys, values, mask=None):
        result = mx.fast.scaled_dot_product_attention(
            query, keys, values, scale=self.width**-0.5, mask=mask
        )
        b, _, t, _ = result.shape
        return self.o_proj(result.transpose(0, 2, 1, 3).reshape(b, t, -1))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden, intermediate = config["hidden_size"], config["intermediate_size"]
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x))


class ExpertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        h, e = config["hidden_size"], config["rms_norm_eps"]
        self.input_layernorm = RMSNorm(h, e)
        self.post_attention_layernorm = RMSNorm(h, e)
        self.nar_input_layernorm = RMSNorm(h, e)
        self.nar_pre_mlp_layernorm = RMSNorm(h, e)
        self.self_attn, self.nar_self_attn = Attention(config), Attention(config)
        self.mlp, self.nar_mlp = FeedForward(config), FeedForward(config)

    def causal(self, x, cache):
        offset = cache.offset
        query, keys, values = self.self_attn.project(self.input_layernorm(x), offset)
        keys, values = cache.append(keys, values)
        mask = None
        if x.shape[1] > 1:
            # Avoid allocating a quadratic prefill mask. Cached continuation
            # still needs absolute positions when more than one token is added.
            mask = (
                "causal"
                if offset == 0
                else (
                    mx.arange(keys.shape[2])[None, :]
                    <= mx.arange(offset, offset + x.shape[1])[:, None]
                )
            )
        x = x + self.self_attn.attend(query, keys, values, mask)
        return x + self.mlp(self.post_attention_layernorm(x))

    def acoustic(self, x, prefix, offset):
        q, k, v = self.nar_self_attn.project(self.nar_input_layernorm(x), offset)
        k = mx.concatenate([prefix[0], k], axis=2)
        v = mx.concatenate([prefix[1], v], axis=2)
        x = x + self.nar_self_attn.attend(q, k, v)
        return x + self.nar_mlp(self.nar_pre_mlp_layernorm(x))


class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = [ExpertLayer(config) for _ in range(config["num_hidden_layers"])]
        self.norm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])


class TimeEmbedding(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(256, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def __call__(self, t, dtype):
        frequency = mx.exp(mx.arange(128, dtype=mx.float32) * (-math.log(10000) / 128))
        phase = t.astype(mx.float32).reshape(-1, 1) * frequency
        sinusoid = mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1).astype(dtype)
        return self.fc2(silu(self.fc1(sinusoid)))


class Positions(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pe = mx.zeros((config["max_latent_frames"], config["hidden_size"]))


class YuE2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        object.__setattr__(self, "config", dict(config))
        self.model = Backbone(config)
        h, d = config["hidden_size"], config["latent_dim"]
        self.lm_head = nn.Linear(h, config["vocab_size"], bias=False)
        self.vae2llm = nn.Linear(d, h)
        self.llm2vae = nn.Linear(h, d)
        self.time_embedder = TimeEmbedding(h)
        self.latent_pos_embed = Positions(config)

    def make_cache(self, capacity):
        if not 1 <= capacity <= self.config["max_position_embeddings"]:
            raise ValueError("Invalid transformer cache capacity")
        return [KVCache(capacity) for _ in self.model.layers]

    def ar(self, tokens, caches, all_positions=False):
        x = self.model.embed_tokens(tokens)
        for layer, cache in zip(self.model.layers, caches, strict=True):
            x = layer.causal(x, cache)
        return (
            self.lm_head(self.model.norm(x if all_positions else x[:, -1:]))
            if all_positions
            else self.lm_head(self.model.norm(x[:, -1]))
        )

    def acoustic_prefix(self, tokens, cancelled=None):
        from .acoustic import check_cancelled

        cache = self.make_cache(len(tokens))
        x = self.model.embed_tokens(mx.array([tokens], dtype=mx.int32))
        result = []
        for layer, slot in zip(self.model.layers, cache, strict=True):
            check_cancelled(cancelled)
            x = layer.causal(x, slot)
            keys = mx.contiguous(slot.keys[:, :, : slot.offset])
            values = mx.contiguous(slot.values[:, :, : slot.offset])
            mx.eval(keys, values, x)
            result.append((keys, values))
        return result

    def velocity(self, state, raw_time, prefix, prefix_length, cancelled=None):
        from .acoustic import check_cancelled

        dtype = state.dtype
        shift = self.config["timestep_shift"]
        t = mx.sigmoid(mx.array(raw_time, dtype=dtype).astype(mx.float32)).astype(dtype)
        numerator = (t.astype(mx.float32) * shift).astype(dtype)
        denominator = mx.array(1.0, dtype) + (t.astype(mx.float32) * (shift - 1)).astype(dtype)
        t = numerator / denominator
        count = state.shape[0] + 2
        x = self.vae2llm(mx.pad(state, [(1, 1), (0, 0)])[None])
        x = x + self.time_embedder(mx.broadcast_to(t, (count,)), dtype)[None]
        positions = mx.minimum(mx.arange(count), self.config["max_latent_frames"] - 1)
        x = x + self.latent_pos_embed.pe[positions][None]
        for layer, kv in zip(self.model.layers, prefix, strict=True):
            check_cancelled(cancelled)
            x = layer.acoustic(x, kv, prefix_length)
        return self.llm2vae(self.model.norm(x))[0, 1:-1]
