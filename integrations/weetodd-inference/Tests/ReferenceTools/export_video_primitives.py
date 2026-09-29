"""Developer-only MLX numerical oracle; never imported by the Swift runtime."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

out = Path(__file__).resolve().parents[1] / "LTX25VideoTests/Fixtures/primitives.json"
x = mx.sin(mx.arange(3 * 3 * 4 * 2, dtype=mx.float32) * 0.13).reshape(1, 3, 3, 4, 2)
w = mx.cos(mx.arange(3 * 2 * 27, dtype=mx.float32) * 0.07).reshape(3, 2, 3, 3, 3) * 0.03
b = mx.array([0.1, -0.2, 0.3])
results = {}
for causal in (False, True):
    pad = (2, 0) if causal else (1, 1)
    z = mx.concatenate(
        [mx.repeat(x[:, :1], pad[0], axis=1), x, mx.repeat(x[:, -1:], pad[1], axis=1)], axis=1
    )
    z = mx.pad(z, [(0, 0), (0, 0), (1, 1), (1, 1), (0, 0)])
    y = mx.conv3d(z, w.transpose(0, 2, 3, 4, 1)) + b
    results["causal" if causal else "symmetric"] = y.flatten().tolist()
results.update(
    input=x.flatten().tolist(),
    weight=w.flatten().tolist(),
    bias=b.tolist(),
    norm=nn.silu(mx.fast.rms_norm(x, weight=None, eps=1e-8)).flatten().tolist(),
)
out.write_text(json.dumps(results, separators=(",", ":")) + "\n")
print(out)
