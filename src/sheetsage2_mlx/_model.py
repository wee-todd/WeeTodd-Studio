"""Native inference equations. Weight names are the checkpoint interchange schema."""

import math

import mlx.core as mx


def linear(x, weights, name):
    y = x @ weights[name + ".weight"].T
    bias = weights.get(name + ".bias")
    return y if bias is None else y + bias


def norm(x, weights, name, eps=1e-5):
    return mx.fast.layer_norm(x, weights[name + ".weight"], weights[name + ".bias"], eps)


def gelu(x):
    return 0.5 * x * (1 + mx.erf(x / math.sqrt(2)))


def conv1d(x, weight, bias=None, stride=1, padding=0, groups=1):
    result = mx.conv1d(x, weight.transpose(0, 2, 1), stride=stride, padding=padding, groups=groups)
    return result if bias is None else result + bias


def conv(x, weights, name, **kwargs):
    return conv1d(x, weights[name + ".weight"], weights.get(name + ".bias"), **kwargs)


def mel_features(audio, buffers, check=lambda: None):
    """Centered periodic-Hann power STFT, checkpoint mel bank, unbounded power dB."""
    # Torch's centered STFT uses reflect padding; drop the final frame, as trained.
    waveform = mx.concatenate([audio[1:1025][::-1], audio, audio[-1025:-1][::-1]])
    count = audio.size // 240
    parts = []
    for begin in range(0, count, 512):
        check()
        batch = min(512, count - begin)
        segment = waveform[begin * 240 : begin * 240 + (batch - 1) * 240 + 2048]
        windows = mx.as_strided(segment, shape=(batch, 2048), strides=(240, 1))
        spectrum = mx.fft.rfft(windows * buffers["spectrogram.window"], axis=-1)
        power = mx.square(mx.abs(spectrum))
        mel = power @ buffers["mel_scale.fb"]
        db = 10 * mx.log10(mx.maximum(mel, 1e-10))
        normalized = (db - buffers["mel_mean"]) / mx.maximum(buffers["mel_std"], 1e-5)
        mx.eval(normalized)
        parts.append(normalized)
    return mx.concatenate(parts, axis=0)[None]


def convnext(x, weights, prefix):
    residual = x
    x = conv(x, weights, prefix + ".depthwise_block.1", padding=3, groups=x.shape[-1])
    x = norm(x, weights, prefix + ".pointwise_block.0", eps=1e-6)
    x = gelu(linear(x, weights, prefix + ".pointwise_block.1"))
    # Global response normalization accumulates along all time steps, including padding.
    magnitude = mx.sqrt(mx.sum(mx.square(x.astype(mx.float32)), axis=1, keepdims=True)).astype(
        x.dtype
    )
    relative = magnitude / (mx.mean(magnitude, axis=-1, keepdims=True) + 1e-6)
    x = (
        x
        + weights[prefix + ".pointwise_block.3.weight"] * x * relative
        + weights[prefix + ".pointwise_block.3.bias"]
    )
    return residual + linear(x, weights, prefix + ".pointwise_block.4")


def rotary(x):
    width = x.shape[-1]
    angle = (
        mx.arange(x.shape[-2], dtype=mx.float32)[:, None]
        * (10000 ** (-mx.arange(0, width, 2, dtype=mx.float32) / width))[None]
    )
    angle = mx.concatenate([angle, angle], axis=-1)
    cosine, sine = mx.cos(angle).astype(x.dtype), mx.sin(angle).astype(x.dtype)
    first, second = mx.split(x, 2, axis=-1)
    return x * cosine + mx.concatenate([-second, first], axis=-1) * sine


def heads(x, count):
    return x.reshape(x.shape[0], x.shape[1], count, -1).transpose(0, 2, 1, 3)


def conformer(x, weights, prefix, head_count=16):
    def ff(value, which):
        value = norm(value, weights, prefix + "." + which + "_layer_norm")
        return linear(
            gelu(linear(value, weights, prefix + "." + which + ".w_1")),
            weights,
            prefix + "." + which + ".w_2",
        )

    x = x + 0.5 * ff(x, "ffn1")
    a = norm(x, weights, prefix + ".attn_layer_norm")
    q = rotary(heads(linear(a, weights, prefix + ".attn.query_proj"), head_count))
    k = rotary(heads(linear(a, weights, prefix + ".attn.key_proj"), head_count))
    v = heads(linear(a, weights, prefix + ".attn.value_proj"), head_count)
    attended = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5)
    attended = attended.transpose(0, 2, 1, 3).reshape(x.shape)
    x = x + linear(attended, weights, prefix + ".attn.out_proj")
    c = norm(x, weights, prefix + ".conv_module.layer_norm")
    c = conv(c, weights, prefix + ".conv_module.conv_block.1")
    left, right = mx.split(c, 2, axis=-1)
    c = left * mx.sigmoid(right)
    c = conv(c, weights, prefix + ".conv_module.conv_block.3", padding=15, groups=x.shape[-1])
    c = gelu(norm(c, weights, prefix + ".conv_module.conv_block.4.1"))
    x = x + conv(c, weights, prefix + ".conv_module.conv_block.6")
    x = x + 0.5 * ff(x, "ffn2")
    return norm(x, weights, prefix + ".final_layer_norm")


def encode(audio, weights, check, report):
    buffers = {
        key.removeprefix("encoder.feature_extractor."): value
        for key, value in weights.items()
        if key.startswith("encoder.feature_extractor.")
    }
    audio = mx.pad(mx.array(audio), [(0, 7_200_000 - len(audio))])
    x = mel_features(audio, buffers, check)
    dtype = weights["encoder.subsampling_module.0.convnext_layers.0.depthwise_block.1.weight"].dtype
    x = x.astype(dtype)
    for stage, depth in enumerate((3, 4, 5)):
        check()
        p = f"encoder.subsampling_module.{stage}"
        if stage:
            x = conv(
                norm(x, weights, p + ".resampling_layer.0", eps=1e-6),
                weights,
                p + ".resampling_layer.2",
                stride=2,
            )
        for block in range(depth):
            check()
            x = convnext(x, weights, p + f".convnext_layers.{block}")
            mx.eval(x)
        report("encoding", f"Audio features: subsampling stage {stage + 1}/3", 0.05 + stage * 0.04)
    coefficients = mx.softmax(weights["layer_weight"], axis=0)
    mixed = x * coefficients[0]
    for layer in range(24):
        check()
        x = conformer(x, weights, f"encoder.layers.{layer}")
        mixed = mixed + x * coefficients[layer + 1]
        mx.eval(x, mixed)
        report(
            "encoding", f"Listening: encoder layer {layer + 1}/24", 0.18 + 0.56 * (layer + 1) / 24
        )
    memory = linear(mixed, weights, "encoder_projection")
    mx.eval(memory)
    return memory


class Decoder:
    """Post-normalized BART decoder with persistent cross-attention and causal KV caches."""

    def __init__(self, weights, memory, heads=8):
        self.weights = {
            key: value
            for key, value in weights.items()
            if key.startswith("decoder.") or key == "token_embedding.weight"
        }
        self.head_count = heads
        self.layer_count = sum(key.endswith(".fc1.weight") for key in self.weights)
        self.cross = []
        for layer in range(self.layer_count):
            p = f"decoder.layers.{layer}.encoder_attn"
            self.cross.append(
                (
                    self._split(linear(memory, self.weights, p + ".k_proj")),
                    self._split(linear(memory, self.weights, p + ".v_proj")),
                )
            )
        mx.eval(self.cross)
        self.reset()

    def _split(self, x):
        return heads(x, self.head_count)

    def reset(self):
        self.cache = [None] * self.layer_count
        self.position = 0

    def step(self, tokens):
        ids = mx.array(tokens, dtype=mx.int32)[None]
        length = len(tokens)
        positions = mx.arange(self.position + 2, self.position + length + 2)
        w = self.weights
        x = w["token_embedding.weight"][ids] + w["decoder.embed_positions.weight"][positions]
        x = norm(x, w, "decoder.layernorm_embedding")
        mask = (
            mx.arange(self.position + length)[None, :]
            <= (self.position + mx.arange(length))[:, None]
        )
        for layer in range(self.layer_count):
            p = f"decoder.layers.{layer}"
            q = self._split(linear(x, w, p + ".self_attn.q_proj"))
            k = self._split(linear(x, w, p + ".self_attn.k_proj"))
            v = self._split(linear(x, w, p + ".self_attn.v_proj"))
            if self.cache[layer] is not None:
                old_k, old_v = self.cache[layer]
                k, v = mx.concatenate([old_k, k], axis=2), mx.concatenate([old_v, v], axis=2)
            self.cache[layer] = (k, v)
            a = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=q.shape[-1] ** -0.5, mask=mask if length > 1 else None
            )
            a = a.transpose(0, 2, 1, 3).reshape(x.shape)
            x = norm(x + linear(a, w, p + ".self_attn.out_proj"), w, p + ".self_attn_layer_norm")
            q = self._split(linear(x, w, p + ".encoder_attn.q_proj"))
            k, v = self.cross[layer]
            a = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5)
            a = a.transpose(0, 2, 1, 3).reshape(x.shape)
            x = norm(
                x + linear(a, w, p + ".encoder_attn.out_proj"), w, p + ".encoder_attn_layer_norm"
            )
            x = norm(
                x + linear(gelu(linear(x, w, p + ".fc1")), w, p + ".fc2"),
                w,
                p + ".final_layer_norm",
            )
        logits = x @ w["token_embedding.weight"].T
        mx.eval(logits, self.cache)
        self.position += length
        return logits

    def close(self):
        self.cache.clear()
        self.cross.clear()
        self.weights.clear()
