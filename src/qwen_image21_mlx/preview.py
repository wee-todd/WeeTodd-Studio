"""Bounded approximate previews from a calibrated, model-specific latent projection."""

import time
from pathlib import Path

import numpy as np
from PIL import Image


class LivePreview:
    def __init__(self, directory, projection, progress, *, clock=time.monotonic):
        self.path = Path(directory) / "live-preview.png"
        self.partial = Path(directory) / "live-preview.partial.png"
        with np.load(projection, allow_pickle=False) as pack:
            self.weights = pack["weights"].copy()
        if self.weights.shape != (65, 4) or not np.isfinite(self.weights).all():
            raise ValueError("The Qwen live-preview projection is invalid")
        self.progress = progress
        self.clock = clock
        self.last = None
        self.revision = 0

    def update(self, latents, *, width, height, step, total):
        now = self.clock()
        if self.last is not None and now - self.last < 2 and step != total:
            return
        pixels = np.asarray(latents, dtype=np.float32).reshape(height // 16, width // 16, 64)
        rgba = pixels @ self.weights[:-1] + self.weights[-1]
        image = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8))
        image.thumbnail((256, 256), Image.Resampling.BILINEAR)
        image.save(self.partial, format="PNG")
        self.partial.replace(self.path)
        self.revision += 1
        self.last = now
        self.progress(
            {
                "event": "progress",
                "stage": "sampling",
                "message": f"Sampling {step}/{total} · live preview",
                "fraction": 0.3 + 0.55 * step / total,
                "previewPath": str(self.path),
                "previewRevision": self.revision,
            }
        )

    def close(self):
        self.path.unlink(missing_ok=True)
        self.partial.unlink(missing_ok=True)


def calibrate(vae, output, *, cancel, progress):
    """Fit a small color projection using procedural calibration images."""
    import mlx.core as mx

    rng = np.random.default_rng(21)
    features, colors = [], []
    for i in range(8):
        if cancel():
            raise InterruptedError("Preview calibration cancelled")
        # Smooth fields exercise chroma and alpha; inference previews remain explicitly approximate.
        field = Image.fromarray(rng.integers(0, 256, (8, 8, 4), dtype=np.uint8)).resize(
            (256, 256), Image.Resampling.BILINEAR
        )
        pixels = np.asarray(field, dtype=np.float32) / 255
        latent = vae.encode(mx.array(pixels[None] * 2 - 1), cancel=cancel)
        mx.eval(latent)
        features.append(np.asarray(latent).reshape(-1, 64))
        colors.append(pixels.reshape(16, 16, 16, 16, 4).mean(axis=(1, 3)).reshape(-1, 4))
        progress(
            {
                "event": "progress",
                "stage": "preview-calibration",
                "fraction": (i + 1) / 8,
                "message": f"Preparing approximate live previews {i + 1}/8",
            }
        )
    x = np.concatenate(features).astype(np.float64)
    x = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    y = np.concatenate(colors)
    weights = np.linalg.solve(x.T @ x + np.eye(65) * 0.01, x.T @ y).astype(np.float32)
    np.savez(output, weights=weights, calibrationVersion=np.array(1))
