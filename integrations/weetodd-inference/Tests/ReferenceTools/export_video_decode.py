"""Installed-weight FP32 MLX oracle for Swift's convolutional decoder.

Writes local validation output only. No model weights or runtime Python bridge.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--frames", type=int, default=2)
parser.add_argument("--height", type=int, default=2)
parser.add_argument("--width", type=int, default=2)
args = parser.parse_args()
shape = [1, 128, args.frames, args.height, args.width]
x = (
    mx.sin(mx.arange(128 * args.frames * args.height * args.width, dtype=mx.float32) * 0.017) * 0.15
).reshape(shape)
model = VideoDecoder(causal=False, spatial_padding_mode="zeros")
source = mx.load(str(args.checkpoint))
weights = []
for name, value in source.items():
    if name.startswith("decoder."):
        name = name.removeprefix("decoder.")
        if value.ndim == 5:
            value = value.transpose(0, 2, 3, 4, 1)
        weights.append((name, value.astype(mx.float32)))
weights.extend(
    [
        (
            "per_channel_statistics.mean",
            source["per_channel_statistics.mean-of-means"].astype(mx.float32),
        ),
        (
            "per_channel_statistics.std",
            source["per_channel_statistics.std-of-means"].astype(mx.float32),
        ),
    ]
)
model.load_weights(weights, strict=True)
mx.eval(model.parameters())
y = model.decode(x, _materialize_stages=True).transpose(0, 2, 3, 4, 1)
mx.eval(y)
args.output.parent.mkdir(parents=True, exist_ok=True)
if args.output.suffix == ".safetensors":
    mx.save_safetensors(str(args.output), {"latent": x, "output": y[0]})
else:
    args.output.write_text(
        json.dumps(
            dict(shape=shape, latent=x.flatten().tolist(), output=y.flatten().tolist()),
            separators=(",", ":"),
        )
        + "\n"
    )
print(
    json.dumps(
        dict(
            shape=shape,
            output_shape=list(y.shape),
            output=str(args.output),
            peak_memory=mx.get_peak_memory(),
        )
    )
)
