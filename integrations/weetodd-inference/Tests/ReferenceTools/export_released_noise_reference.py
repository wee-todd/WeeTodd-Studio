"""Export production-helper noise/blend fixtures; development only, never inference."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
from ltx_pipelines_mlx.utils.helpers import create_noised_state


def state(tokens, seed, sigma=1.0, initial=None, legacy=True):
    return create_noised_state(
        (1, tokens, 128),
        [],
        (2, 1, 2),
        mx.zeros((1, tokens, 1)),
        seed,
        sigma,
        initial,
        legacy_scalar_blend=legacy,
    ).latent


v = state(4, 42)
a = state(10, 43)
k = mx.random.key(10042)


def draw(shape):
    global k
    k, d = mx.random.split(k, 2)
    return mx.random.normal(shape, key=d)


vn = draw(v.shape)
an = draw(a.shape)
firstv = (v * 0.25 + vn * 0.01).astype(mx.bfloat16)
firsta = (a * 0.25 + an * 0.01).astype(mx.bfloat16)
high = mx.ones((1, 16, 128)) * firstv[0, 0, 0].astype(mx.float32)
rv = state(16, 44, 0.909375, high)
ra = state(10, 44, 0.909375, firsta, False)
values = {
    name: x.astype(mx.float32).flatten().tolist()
    for name, x in [
        ("initial_video", v),
        ("initial_audio", a),
        ("ancestral_video", vn),
        ("ancestral_audio", an),
        ("refined_video", rv),
        ("refined_audio", ra),
    ]
}
parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.write_text(json.dumps(values, indent=2) + "\n")
