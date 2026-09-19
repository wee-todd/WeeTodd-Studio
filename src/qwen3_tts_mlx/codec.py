"""Native Qwen 12Hz speech tokenizer. Actual codec stride is 1,920 samples."""

import mlx.core as mx
import mlx.nn as nn

from wee_todd_mlx.speech_ops import Weights

from .model import transformer


class QwenCodec(Weights):
    sample_rate = 24000
    frame_length = 1920

    def __init__(self, tensors, config):
        super().__init__(tensors)
        self.config = config
        self.tables = {}

    def conv(self, x, p, stride=1, dilation=1, transpose=False, groups=1):
        p = p if p + ".weight" in self.w else p + ".conv"
        w = self.w[p + ".weight"]
        kernel = w.shape[-1]
        if transpose:
            y = mx.conv_transpose1d(x, w.transpose(1, 2, 0), stride=stride)
            trim = kernel - stride
            if trim:
                y = y[:, :-trim]
        else:
            padding = (kernel - 1) * dilation + 1 - stride
            x = mx.pad(x, [(0, 0), (padding, (-x.shape[1]) % stride), (0, 0)])
            y = mx.conv1d(x, w.transpose(0, 2, 1), stride=stride, dilation=dilation, groups=groups)
        bias = self.w.get(p + ".bias")
        return y if bias is None else y + bias

    def table(self, p, encoder=False):
        if p not in self.tables:
            sum_name = ".embed_sum" if encoder else ".embedding_sum"
            total = self.w[p + sum_name]
            count = self.w[p + ".cluster_usage"]
            self.tables[p] = total / mx.maximum(count[:, None], 1e-5)
        return self.tables[p]

    def encode(self, audio):
        x = mx.array(audio, dtype=mx.float32).reshape(1, -1, 1)
        p = "encoder.encoder.layers"
        x = self.conv(x, p + ".0")
        for i, stride in enumerate((4, 5, 6, 8)):
            r = f"{p}.{1 + 3 * i}.block"
            y = self.conv(nn.elu(x), r + ".1")
            x = x + self.conv(nn.elu(y), r + ".3")
            x = self.conv(nn.elu(x), f"{p}.{3 + 3 * i}", stride=stride)
        x = self.conv(nn.elu(x), p + ".14")
        x = transformer(
            self,
            x,
            "encoder.encoder_transformer",
            self.config["encoder_config"],
            norm_kind="layer",
            scales=True,
            gelu=True,
            final_norm=False,
            window=self.config["encoder_config"]["sliding_window"],
        )
        x = self.conv(x, "encoder.downsample", stride=2)
        codes = []
        for kind, count in [("semantic", 1), ("acoustic", 15)]:
            p = f"encoder.quantizer.{kind}_residual_vector_quantizer"
            z = self.conv(x, p + ".input_proj")
            for j in range(count):
                table = self.table(f"{p}.layers.{j}.codebook", encoder=True)
                distance = (
                    mx.sum(z * z, axis=-1, keepdims=True)
                    - 2 * (z @ table.T)
                    + mx.sum(table * table, axis=-1)
                )
                indices = mx.argmin(distance, axis=-1)
                codes.append(indices[0])
                z = z - table[indices]
        return mx.stack(codes)

    def snake(self, x, p):
        a, b = mx.exp(self.w[p + ".alpha"]), mx.exp(self.w[p + ".beta"])
        return x + mx.sin(a * x) ** 2 / (b + 1e-9)

    def decode(self, codes):
        if codes.shape[0] != 16:
            raise ValueError("Qwen decoding requires all sixteen code groups")
        x = 0
        for kind, offset, count in [("first", 0, 1), ("rest", 1, 15)]:
            p = f"decoder.quantizer.rvq_{kind}"
            z = sum(
                self.table(f"{p}.vq.layers.{j}._codebook")[codes[offset + j]][None]
                for j in range(count)
            )
            x = x + self.conv(z, p + ".output_proj")
        x = self.conv(x, "decoder.pre_conv")
        p = "decoder.pre_transformer"
        x = self.linear(x, p + ".input_proj")
        x = transformer(self, x, p, self.config["decoder_config"], scales=True)
        x = self.linear(x, p + ".output_proj")
        for i in range(2):
            p = f"decoder.upsample.{i}"
            x = self.conv(x, p + ".0", stride=2, transpose=True)
            y = self.conv(x, p + ".1.dwconv", groups=x.shape[-1])
            y = self.linear(
                nn.gelu(self.linear(self.layer_norm(y, p + ".1.norm"), p + ".1.pwconv1")),
                p + ".1.pwconv2",
            )
            x = x + y * self.w[p + ".1.gamma"]
        x = self.conv(x, "decoder.decoder.0")
        for i, stride in enumerate((8, 5, 4, 3), 1):
            p = f"decoder.decoder.{i}.block"
            x = self.conv(self.snake(x, p + ".0"), p + ".1", stride=stride, transpose=True)
            for j, dilation in enumerate((1, 3, 9), 2):
                r = f"{p}.{j}"
                y = self.conv(self.snake(x, r + ".act1"), r + ".conv1", dilation=dilation)
                x = x + self.conv(self.snake(y, r + ".act2"), r + ".conv2")
        return mx.clip(
            self.conv(self.snake(x, "decoder.decoder.5"), "decoder.decoder.6")[0, :, 0], -1.0, 1.0
        )
