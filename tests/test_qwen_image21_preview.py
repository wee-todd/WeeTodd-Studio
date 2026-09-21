import numpy as np
from PIL import Image

from qwen_image21_mlx.preview import LivePreview


def test_live_preview_is_throttled_atomic_revisioned_and_does_not_mutate_latents(tmp_path):
    weights = np.zeros((65, 4), dtype=np.float32)
    weights[-1] = [0.2, 0.4, 0.8, 1]
    projection = tmp_path / "projection.npz"
    np.savez(projection, weights=weights)
    events = []
    clock = [0.0]
    preview = LivePreview(tmp_path, projection, events.append, clock=lambda: clock[0])
    latents = np.zeros((1, 16, 64), dtype=np.float32)
    original = latents.copy()
    preview.update(latents, width=64, height=64, step=1, total=10)
    preview.update(latents, width=64, height=64, step=2, total=10)
    assert len(events) == 1
    assert events[0]["previewRevision"] == 1
    assert events[0]["previewPath"] == str(tmp_path / "live-preview.png")
    assert Image.open(events[0]["previewPath"]).mode == "RGBA"
    clock[0] = 3.0
    preview.update(latents, width=64, height=64, step=3, total=10)
    assert events[-1]["previewRevision"] == 2
    np.testing.assert_array_equal(latents, original)
    preview.close()
    assert not (tmp_path / "live-preview.png").exists()
    assert not list(tmp_path.glob("*.partial*"))
