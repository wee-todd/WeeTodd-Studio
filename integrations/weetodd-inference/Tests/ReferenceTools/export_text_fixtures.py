"""Test-only MLX equation references; never imported by native inference."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--gemma-fixed", type=Path)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7511)


def random(shape):
    return mx.array(rng.normal(0, 0.15, shape).astype(np.float32))


def rms(x, w=None, eps=1e-6):
    return mx.fast.rms_norm(x, w, eps)


def normweights(d):
    return mx.array(rng.uniform(0.8, 1.2, d).astype(np.float32))


def save(name, weights, x, y):
    mx.save_safetensors(
        str(args.output / (name + ".safetensors")), dict(weights, input=x, expected=y)
    )


def gemma(full):
    D, H, heads, kv, hd, T = 8, 12, 4, 1 if full else 2, 4, 3
    w = {}
    for key in [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ]:
        w[key + ".weight"] = normweights(D)
    for key in ["q_norm", "k_norm"]:
        w["self_attn." + key + ".weight"] = normweights(hd)
    for key, shape in [
        ("q_proj", (heads * hd, D)),
        ("k_proj", (kv * hd, D)),
        ("o_proj", (D, heads * hd)),
    ]:
        w["self_attn." + key + ".weight"] = random(shape)
    if not full:
        w["self_attn.v_proj.weight"] = random((kv * hd, D))
    for key, shape in [("gate_proj", (H, D)), ("up_proj", (H, D)), ("down_proj", (D, H))]:
        w["mlp." + key + ".weight"] = random(shape)
    w["layer_scalar"] = mx.array([0.83])
    x = random((T, D))

    def linear(name, v):
        return v @ w[name + ".weight"].T

    h = rms(x, w["input_layernorm.weight"])
    q = linear("self_attn.q_proj", h).reshape(T, heads, hd)
    k = linear("self_attn.k_proj", h).reshape(T, kv, hd)
    v = k if full else linear("self_attn.v_proj", h).reshape(T, kv, hd)
    q = rms(q, w["self_attn.q_norm.weight"])
    k = rms(k, w["self_attn.k_norm.weight"])
    v = rms(v)

    def rope(z):
        freq = np.zeros(hd // 2, np.float32)
        n = hd // 4 if full else hd // 2
        freq[:n] = 1 / np.power(1e6 if full else 1e4, 2 * np.arange(n) / hd)
        a = mx.array(np.arange(T)[:, None, None] * freq[None, None, :], dtype=mx.float32)
        a = mx.broadcast_to(a, z.shape[:-1] + (hd // 2,))
        first, last = mx.split(z, 2, axis=-1)
        return mx.concatenate(
            [first * mx.cos(a) - last * mx.sin(a), last * mx.cos(a) + first * mx.sin(a)], axis=-1
        )

    q, k = rope(q), rope(k)
    k = mx.repeat(k, heads // kv, axis=1)
    v = mx.repeat(v, heads // kv, axis=1)
    q, k, v = [z.transpose(1, 0, 2) for z in (q, k, v)]
    mask = np.arange(T)[None, :] <= np.arange(T)[:, None]
    if not full:
        mask &= np.arange(T)[None, :] > np.arange(T)[:, None] - 2
    scores = q @ k.transpose(0, 2, 1)
    scores = mx.where(mx.array(mask), scores, -1e30)
    attn = (mx.softmax(scores, axis=-1) @ v).transpose(1, 0, 2).reshape(T, -1)
    h = x + rms(linear("self_attn.o_proj", attn), w["post_attention_layernorm.weight"])
    n = rms(h, w["pre_feedforward_layernorm.weight"])
    y = (
        h
        + rms(
            linear(
                "mlp.down_proj",
                nn.gelu_approx(linear("mlp.gate_proj", n)) * linear("mlp.up_proj", n),
            ),
            w["post_feedforward_layernorm.weight"],
        )
    ) * w["layer_scalar"]
    save("gemma-full" if full else "gemma-sliding", w, x, y)


def connector():
    D, heads, T = 8, 2, 3
    hd = D // heads
    w = {}
    for key, out in [
        ("to_q", D),
        ("to_k", D),
        ("to_v", D),
        ("to_out.0", D),
        ("to_gate_logits", heads),
    ]:
        w["attn1." + key + ".weight"] = random((out, D))
        w["attn1." + key + ".bias"] = random((out,))
    for key in ["q_norm", "k_norm"]:
        w["attn1." + key + ".weight"] = normweights(D)
    for key, inn, out in [("ff.net.0.proj", D, 4 * D), ("ff.net.2", 4 * D, D)]:
        w[key + ".weight"] = random((out, inn))
        w[key + ".bias"] = random((out,))

    def lin(k, x):
        return x @ w[k + ".weight"].T + w[k + ".bias"]

    x = random((T, D))
    n = rms(x)
    q = rms(lin("attn1.to_q", n), w["attn1.q_norm.weight"], 1e-5)
    k = rms(lin("attn1.to_k", n), w["attn1.k_norm.weight"], 1e-5)
    v = lin("attn1.to_v", n)
    freq = mx.array((np.power(10000, np.linspace(0, 1, D // 2)) * np.pi / 2).astype(np.float32))
    angles = ((mx.arange(T, dtype=mx.float32) / 4096 * 2 - 1)[:, None] * freq[None, :]).reshape(
        T, heads, hd // 2
    )

    def rope(z):
        a, b = mx.split(z.reshape(T, heads, hd), 2, axis=-1)
        return mx.concatenate(
            [a * mx.cos(angles) - b * mx.sin(angles), a * mx.sin(angles) + b * mx.cos(angles)],
            axis=-1,
        ).transpose(1, 0, 2)

    q, k = rope(q), rope(k)
    v = v.reshape(T, heads, hd).transpose(1, 0, 2)
    attn = (mx.softmax((q @ k.transpose(0, 2, 1)) * hd**-0.5, axis=-1) @ v).transpose(1, 0, 2)
    gate = 2 * mx.sigmoid(lin("attn1.to_gate_logits", n))
    attn = (attn * gate[:, :, None]).reshape(T, D)
    y = x + lin("attn1.to_out.0", attn)
    y = y + lin("ff.net.2", nn.gelu_approx(lin("ff.net.0.proj", rms(y))))
    save("connector", w, x, y)


gemma(False)
gemma(True)
connector()
if args.gemma_fixed:
    from safetensors import safe_open
    from tokenizers import Tokenizer

    with safe_open(args.gemma_fixed, framework="numpy") as f:
        t = Tokenizer.from_buffer(f.get_tensor("tokenizer_json").tobytes())
    prompts = [
        "",
        "  A red fox jumps.  ",
        "Café café e\u0301 — 你好 👋🏽",
        "One\n\nTwo\tthree",
        "<bos>hello<eos>",
        "A  B   C",
        "\u001c foo \u001f",
        "👨‍👩‍👧‍👦🧪\u0378",
        "<|turn>user\nhello<turn|>",
        "là la\u0300 có co\u0301 \u09dfে",
        "\ufeff\ufefftest",
    ]
    records = []
    for prompt in prompts:
        ids = t.encode(prompt.strip()).ids
        if not ids or ids[0] != 2:
            ids = [2] + ids
        records.append(dict(prompt=prompt, ids=ids))
    (args.output / "tokenizer-reference.json").write_text(json.dumps(records, ensure_ascii=False))
