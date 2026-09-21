"""Single-frame RGBA VAE in native MLX NHWC layout.

Temporal resampling is specialized to the first image frame. Parameter-free
residual shortcuts retain their temporal zero-padding/duplication semantics.
"""

from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn


def average_shortcut(x, out_channels, *, temporal, spatial):
    batch, height, width, channels = x.shape
    value = x.transpose(0, 3, 1, 2).reshape(
        batch, channels, height // spatial, spatial, width // spatial, spatial
    )
    value = value.transpose(0, 1, 3, 5, 2, 4)
    value = value[:, :, None]
    if temporal == 2:
        value = mx.concatenate([mx.zeros_like(value), value], axis=2)
    group = channels * temporal * spatial * spatial // out_channels
    value = value.reshape(batch, out_channels, group, height // spatial, width // spatial)
    return mx.mean(value, axis=2).transpose(0, 2, 3, 1)


def duplicate_shortcut(x, out_channels, *, temporal):
    batch, height, width, channels = x.shape
    repeats = out_channels * temporal * 4 // channels
    value = mx.repeat(x.transpose(0, 3, 1, 2), repeats, axis=1)
    value = value.reshape(batch, out_channels, temporal, 2, 2, height, width)
    value = value[:, :, temporal - 1]
    return value.transpose(0, 4, 2, 5, 3, 1).reshape(batch, height * 2, width * 2, out_channels)


class ChannelNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = mx.ones((dim,))

    def __call__(self, x):
        value = x.astype(mx.float32)
        length = mx.sqrt(mx.sum(value * value, axis=-1, keepdims=True))
        return (
            (value / mx.maximum(length, 1e-12)).astype(x.dtype) * (x.shape[-1] ** 0.5) * self.gamma
        )


class Residual(nn.Module):
    def __init__(self, source, target):
        super().__init__()
        self.norm1 = ChannelNorm(source)
        self.norm2 = ChannelNorm(target)
        self.conv1 = nn.Conv2d(source, target, 3, padding=1)
        self.conv2 = nn.Conv2d(target, target, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(source, target, 1) if source != target else nn.Identity()

    def __call__(self, x):
        residual = self.conv_shortcut(x)
        x = self.conv1(nn.silu(self.norm1(x)))
        return residual + self.conv2(nn.silu(self.norm2(x)))


class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = ChannelNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def __call__(self, x):
        q, k, v = mx.split(
            self.to_qkv(self.norm(x)).reshape(x.shape[0], 1, -1, 3 * x.shape[-1]), 3, axis=-1
        )
        parts = []
        for start in range(0, q.shape[2], 128):
            out = mx.fast.scaled_dot_product_attention(
                q[:, :, start : start + 128], k, v, scale=x.shape[-1] ** -0.5
            )
            mx.eval(out)
            parts.append(out)
        value = mx.concatenate(parts, axis=2).reshape(x.shape)
        return x + self.proj(value)


class Middle(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.resnets = [Residual(dim, dim), Residual(dim, dim)]
        self.attentions = [SpatialAttention(dim)]

    def __call__(self, x):
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class Resample(nn.Module):
    def __init__(self, dim, up):
        super().__init__()
        self.up = up
        self.resample = [
            nn.Identity(),
            nn.Conv2d(dim, dim, 3, stride=1 if up else 2, padding=1 if up else 0),
        ]

    def __call__(self, x):
        if self.up:
            x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)
        else:
            x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        return self.resample[1](x)


class DownBlock(nn.Module):
    def __init__(self, source, target, blocks, temporal, down):
        super().__init__()
        self.target = target
        self.temporal = temporal
        self.down = down
        self.resnets = [Residual(source if i == 0 else target, target) for i in range(blocks)]
        self.downsampler = Resample(target, False) if down else None

    def __call__(self, x):
        skip = average_shortcut(
            x, self.target, temporal=self.temporal, spatial=2 if self.down else 1
        )
        for layer in self.resnets:
            x = layer(x)
            mx.eval(x)
        if self.downsampler is not None:
            x = self.downsampler(x)
        return x + skip


class UpBlock(nn.Module):
    def __init__(self, source, target, blocks, temporal, up):
        super().__init__()
        self.target = target
        self.temporal = temporal
        self.up = up
        self.resnets = [Residual(source if i == 0 else target, target) for i in range(blocks + 1)]
        self.upsampler = Resample(target, True) if up else None

    def __call__(self, x):
        skip = duplicate_shortcut(x, self.target, temporal=self.temporal) if self.up else None
        for layer in self.resnets:
            x = layer(x)
            mx.eval(x)
        if self.upsampler is not None:
            x = self.upsampler(x) + skip
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config["base_dim"]
        multipliers = config["dim_mult"]
        dims = [dim] + [dim * m for m in multipliers]
        self.conv_in = nn.Conv2d(4, dim, 3, padding=1)
        self.down_blocks = [
            DownBlock(
                a,
                b,
                config["num_res_blocks"],
                2 if i < len(multipliers) - 1 and config["temperal_downsample"][i] else 1,
                i < len(multipliers) - 1,
            )
            for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True))
        ]
        self.mid_block = Middle(dims[-1])
        self.norm_out = ChannelNorm(dims[-1])
        self.conv_out = nn.Conv2d(dims[-1], config["z_dim"] * 2, 3, padding=1)

    def __call__(self, x, cancel):
        x = self.conv_in(x)
        for block in self.down_blocks:
            if cancel():
                raise InterruptedError("Reference encoding cancelled")
            x = block(x)
            mx.eval(x)
        return self.conv_out(nn.silu(self.norm_out(self.mid_block(x))))


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config["decoder_base_dim"]
        multipliers = config["dim_mult"]
        dims = [dim * multipliers[-1]] + [dim * m for m in reversed(multipliers)]
        temporal = list(reversed(config["temperal_downsample"]))
        self.conv_in = nn.Conv2d(config["z_dim"], dims[0], 3, padding=1)
        self.mid_block = Middle(dims[0])
        self.up_blocks = [
            UpBlock(
                a,
                b,
                config["num_res_blocks"],
                2 if i < len(temporal) and temporal[i] else 1,
                i < len(multipliers) - 1,
            )
            for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True))
        ]
        self.norm_out = ChannelNorm(dims[-1])
        self.conv_out = nn.Conv2d(dims[-1], 4, 3, padding=1)

    def __call__(self, x, cancel):
        x = self.mid_block(self.conv_in(x))
        for block in self.up_blocks:
            if cancel():
                raise InterruptedError("Image decode cancelled")
            x = block(x)
            mx.eval(x)
        return self.conv_out(nn.silu(self.norm_out(x)))


@contextmanager
def bounded_vae_memory():
    # Resolution-changing VAE stages otherwise retain large, differently shaped free buffers.
    previous = mx.set_cache_limit(0)
    try:
        yield
    finally:
        mx.synchronize()
        mx.clear_cache()
        mx.set_cache_limit(previous)


class ImageVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.quant_conv = nn.Conv2d(2 * config["z_dim"], 2 * config["z_dim"], 1)
        self.post_quant_conv = nn.Conv2d(config["z_dim"], config["z_dim"], 1)

    def encode(self, pixels, cancel=lambda: False):
        with bounded_vae_memory():
            mean = mx.split(self.quant_conv(self.encoder(pixels, cancel)), 2, axis=-1)[0]
            result = (mean - mx.array(self.config["latents_mean"])) / mx.array(
                self.config["latents_std"]
            )
            mx.eval(result)
            return result

    def decode(self, latents, cancel=lambda: False):
        with bounded_vae_memory():
            value = latents * mx.array(self.config["latents_std"]) + mx.array(
                self.config["latents_mean"]
            )
            result = mx.clip(self.decoder(self.post_quant_conv(value), cancel), -1, 1)
            mx.eval(result)
            return result
