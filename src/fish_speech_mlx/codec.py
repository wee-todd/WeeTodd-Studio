"""Native causal DAC encoder/decoder for Fish S2 Pro (44,100 Hz).

Architecture follows the published checkpoint contract. Convolutions use NTC internally;
source OIK/IOK tensors are converted once when used, with weight norm folded explicitly.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from wee_todd_mlx.speech_ops import Weights, attention, gated_mlp, rope


class FishCodec(Weights):
    frame_length = 2048
    sample_rate = 44100

    def __init__(self, tensors):
        converted = {}
        wn = {
            k.split(".conv.parametrizations")[0] for k in tensors if ".conv.parametrizations" in k
        }
        for key, value in tensors.items():
            key = key.replace(".conv.parametrizations.weight.original0", ".weight_g")
            key = key.replace(".conv.parametrizations.weight.original1", ".weight_v")
            key = key.replace(".parametrizations.weight.original0", ".weight_g")
            key = key.replace(".parametrizations.weight.original1", ".weight_v")
            if key.endswith(".conv.bias") and key[:-10] in wn:
                key = key[:-10] + ".bias"
            converted[key] = value
        super().__init__(converted)
        self._conv = {}

    def conv(self, x, prefix, *, stride=1, dilation=1, transpose=False, groups=1, causal=True):
        cache_key = (prefix, transpose)
        if cache_key not in self._conv:
            if prefix + ".weight_v" in self.w:
                value = self.w[prefix + ".weight_v"]
                value = (
                    value
                    * self.w[prefix + ".weight_g"]
                    / mx.sqrt(mx.sum(value * value, axis=(1, 2), keepdims=True))
                )
                bias = self.w.get(prefix + ".bias")
            else:
                name = prefix if prefix + ".weight" in self.w else prefix + ".conv"
                value, bias = self.w[name + ".weight"], self.w.get(name + ".bias")
            kernel = value.shape[-1]
            value = value.transpose(1, 2, 0) if transpose else value.transpose(0, 2, 1)
            self._conv[cache_key] = (value, bias, kernel)
        value, bias, kernel = self._conv[cache_key]
        if transpose:
            result = mx.conv_transpose1d(x, value, stride=stride)
            crop = kernel - stride if causal else (kernel - stride) // 2
            if crop:
                result = result[:, :-crop] if causal else result[:, crop:-crop]
        else:
            span = (kernel - 1) * dilation + 1
            if causal:
                left = span - stride
                extra = (-x.shape[1]) % stride
                x = mx.pad(x, [(0, 0), (left, extra), (0, 0)])
            result = mx.conv1d(x, value, stride=stride, dilation=dilation, groups=groups)
        return result if bias is None else result + bias

    def snake(self, x, prefix):
        a = self.w[prefix + ".alpha"].transpose(0, 2, 1)
        return x + mx.sin(a * x) ** 2 / (a + 1e-9)

    def residual(self, x, prefix, dilation):
        y = self.snake(x, prefix + ".block.0")
        y = self.conv(y, prefix + ".block.1", dilation=dilation)
        y = self.snake(y, prefix + ".block.2")
        return x + self.conv(y, prefix + ".block.3")

    def transformer(self, x, prefix, layers, window):
        n = x.shape[1]
        mask = (mx.arange(n)[None, :] <= mx.arange(n)[:, None]) & (
            mx.arange(n)[None, :] > mx.arange(n)[:, None] - window
        )
        for i in range(layers):
            p = f"{prefix}.layers.{i}"
            normed = self.norm(x, p + ".attention_norm", 1e-5)
            qkv = self.linear(normed, p + ".attention.wqkv")
            q, k, v = [
                a.reshape(1, n, -1, 64).transpose(0, 2, 1, 3) for a in mx.split(qkv, 3, axis=-1)
            ]
            q, k = rope(q, base=10000, adjacent=True), rope(k, base=10000, adjacent=True)
            y = attention(q, k, v, mask=mask).transpose(0, 2, 1, 3).reshape(x.shape)
            x = x + self.linear(y, p + ".attention.wo") * self.w[p + ".attention_layer_scale.gamma"]
            x = (
                x
                + gated_mlp(self, self.norm(x, p + ".ffn_norm", 1e-5), p + ".feed_forward")
                * self.w[p + ".ffn_layer_scale.gamma"]
            )
        return self.norm(x, prefix + ".norm", 1e-5)

    def convnext(self, x, p):
        y = self.conv(x, p + ".dwconv", groups=x.shape[-1])
        y = self.layer_norm(y, p + ".norm")
        y = self.linear(nn.gelu(self.linear(y, p + ".pwconv1")), p + ".pwconv2")
        return x + y * self.w[p + ".gamma"]

    def quantizer_prefix(self, i):
        return (
            "quantizer.semantic_quantizer.quantizers.0"
            if i == 0
            else f"quantizer.quantizer.quantizers.{i - 1}"
        )

    def encode(self, audio):
        x = mx.array(audio, dtype=mx.float32).reshape(1, -1, 1)
        x = mx.pad(x, [(0, 0), (0, (-x.shape[1]) % self.frame_length), (0, 0)])
        x = self.conv(x, "encoder.block.0")
        for i, stride in enumerate((2, 4, 8, 8), 1):
            p = f"encoder.block.{i}.block"
            for j, dilation in enumerate((1, 3, 9)):
                x = self.residual(x, f"{p}.{j}", dilation)
            x = self.conv(self.snake(x, p + ".3"), p + ".4", stride=stride)
            if i == 4:
                x = self.transformer(x, p + ".5", 4, 512)
        x = self.conv(self.snake(x, "encoder.block.5"), "encoder.block.6")
        for i in range(2):
            x = self.conv(x, f"quantizer.downsample.{i}.0", stride=2)
            x = self.convnext(x, f"quantizer.downsample.{i}.1")
        x = self.transformer(x, "quantizer.pre_module", 8, 128)
        codes = []
        for i in range(10):
            p = self.quantizer_prefix(i)
            z = self.conv(x, p + ".in_proj", causal=False)
            table = self.w[p + ".codebook.weight"]
            normalized = z / mx.maximum(mx.sqrt(mx.sum(z * z, axis=-1, keepdims=True)), 1e-12)
            vectors = table / mx.maximum(
                mx.sqrt(mx.sum(table * table, axis=-1, keepdims=True)), 1e-12
            )
            indices = mx.argmax(normalized @ vectors.T, axis=-1)
            codes.append(indices[0])
            x = x - self.conv(table[indices], p + ".out_proj", causal=False)
        return mx.stack(codes)

    def decode(self, codes):
        z = 0
        for i in range(10):
            p = self.quantizer_prefix(i)
            table = self.w[p + ".codebook.weight"]
            z = z + self.conv(
                table[mx.clip(codes[i], 0, table.shape[0] - 1)][None], p + ".out_proj", causal=False
            )
        x = self.transformer(z, "quantizer.post_module", 8, 128)
        for i in range(2):
            x = self.conv(x, f"quantizer.upsample.{i}.0", stride=2, transpose=True)
            x = self.convnext(x, f"quantizer.upsample.{i}.1")
        x = self.conv(x, "decoder.model.0")
        for i, stride in enumerate((8, 8, 4, 2), 1):
            p = f"decoder.model.{i}.block"
            x = self.conv(self.snake(x, p + ".0"), p + ".1", stride=stride, transpose=True)
            for j, dilation in enumerate((1, 3, 9), 2):
                x = self.residual(x, f"{p}.{j}", dilation)
        return mx.tanh(self.conv(self.snake(x, "decoder.model.5"), "decoder.model.6"))[0, :, 0]
