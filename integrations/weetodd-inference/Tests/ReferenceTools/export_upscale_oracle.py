#!/usr/bin/env python3
"""Developer-only Float32 oracle; Swift inference never launches Python."""

import argparse
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--vae", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--height", type=int, default=4)
    parser.add_argument("--width", type=int, default=8)
    args = parser.parse_args()
    if not (1 <= args.frames <= 9 and 1 <= args.height <= 8 and 1 <= args.width <= 16):
        parser.error("Qualification dimensions exceed the bounded oracle scope")
    sys.path.insert(0, str(args.project / "src"))
    from ltx25_mlx.components import LTX25LatentNormalizer, load_ltx25_spatial_upsampler

    model = load_ltx25_spatial_upsampler(args.checkpoint)
    model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))
    normalizer = LTX25LatentNormalizer(args.vae)
    normalizer.mean = normalizer.mean.astype(mx.float32)
    normalizer.std = normalizer.std.astype(mx.float32)
    shape = (args.frames, args.height, args.width, 128)
    x = (
        mx.sin(mx.arange(args.frames * args.height * args.width * 128).astype(mx.float32) * 0.173)
        * 0.1
    ).reshape(shape)
    denormalized = normalizer.denormalize_latent(x[None]).transpose(0, 4, 1, 2, 3)
    y = model(denormalized).transpose(0, 2, 3, 4, 1)
    y = normalizer.normalize_latent(y)[0]
    mx.eval(x, y)
    mx.save_safetensors(str(args.output), {"input": x, "output": y})
    print({"shape": list(y.shape), "peak": float(mx.max(mx.abs(y)))})


if __name__ == "__main__":
    main()
