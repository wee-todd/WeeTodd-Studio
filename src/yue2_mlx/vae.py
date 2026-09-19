"""Channels-last Oobleck stereo decoder with exact halo-cropped tiling."""

import math

import mlx.core as mx
import mlx.nn as nn


class Snake(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x):
        return x + mx.square(mx.sin(x * mx.exp(self.alpha))) / (mx.exp(self.beta) + 1e-9)


class Residual(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.layers = [
            Snake(channels),
            nn.Conv1d(channels, channels, 7, padding=3 * dilation, dilation=dilation),
            Snake(channels),
            nn.Conv1d(channels, channels, 1),
        ]

    def __call__(self, x):
        transformed = x
        for layer in self.layers:
            transformed = layer(transformed)
        return x + transformed


class Upsample(nn.Module):
    def __init__(self, source, target, stride):
        super().__init__()
        self.layers = [
            Snake(source),
            nn.ConvTranspose1d(
                source, target, 2 * stride, stride=stride, padding=math.ceil(stride / 2)
            ),
            *[Residual(target, dilation) for dilation in (1, 3, 9)],
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = [config["channels"] * i for i in [1, *config["c_mults"]]]
        strides = config["strides"]
        object.__setattr__(self, "strides", tuple(reversed(strides)))
        self.layers = [nn.Conv1d(config["latent_dim"], channels[-1], 7, padding=3)]
        for i in reversed(range(len(strides))):
            self.layers.append(Upsample(channels[i + 1], channels[i], strides[i]))
        self.layers.extend(
            [
                Snake(channels[0]),
                nn.Conv1d(channels[0], config["out_channels"], 7, padding=3, bias=False),
            ]
        )

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def output_length(self, frames):
        for stride in self.strides:
            frames = (frames - 1) * stride - 2 * math.ceil(stride / 2) + 2 * stride
        return frames

    def decode_tiled(self, latents, core=1024, halo=16, *, cancelled=None, progress=None):
        from .acoustic import check_cancelled

        if type(core) is not int or core < 1 or type(halo) is not int or halo < 16:
            raise ValueError("Decoder requires a positive core and halo >= 16 latent frames")
        frames = latents.shape[0]
        if frames < 1:
            raise ValueError("Cannot decode empty latents")
        hop = math.prod(self.strides)
        length = self.output_length(frames)
        pieces = []
        for start in range(0, frames, core):
            check_cancelled(cancelled)
            end = min(frames, start + core)
            left, right = max(0, start - halo), min(frames, end + halo)
            tile = self(latents[None, left:right])[0]
            offset = (start - left) * hop
            count = min(end * hop, length) - start * hop
            piece = tile[offset : offset + count]
            mx.eval(piece)
            pieces.append(piece)
            if progress:
                progress(end, frames)
        return mx.concatenate(pieces, axis=0)
