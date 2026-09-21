import mlx.core as mx
import pytest

from qwen_image21_mlx.pipeline import render_prepared


def test_render_stages_unload_and_report_progress_without_model_weights(tmp_path):
    events, stages = [], []

    class VAE:
        def decode(self, value, cancel):
            return mx.zeros((1, 32, 32, 4))

    class Transformer:
        def __call__(self, latents, conditioning, timestep, **kwargs):
            return mx.zeros_like(latents)

    class Runtime:
        def load(self, name):
            stages.append("load:" + name)
            return VAE() if name == "vae" else Transformer() if name == "transformer" else object()

        def release(self, name, module):
            stages.append("release:" + name)

        def encode(self, model, prompt, images, cancel, progress):
            return {"features": mx.zeros((1, 1, 8)), "imageMask": [False]}

    request = {
        "prompt": "test",
        "inputs": [],
        "fingerprint": "fixture",
        "configuration": {"width": 32, "height": 32, "steps": 2, "seed": 42, "livePreview": False},
        "cacheMode": "recompute",
    }
    result = render_prepared(
        request,
        {"scheduler": {}},
        tmp_path,
        runtime=Runtime(),
        cancel=lambda: False,
        progress=events.append,
    )
    assert stages == [
        "load:text_encoder",
        "release:text_encoder",
        "load:transformer",
        "release:transformer",
        "load:vae",
        "release:vae",
    ]
    assert result["asset"]["path"].endswith("image.png")
    assert result["normalizedRequest"]["configuration"]["seed"] == 42
    assert [e["step"] for e in events if "step" in e] == [1, 2]
    assert events[-1]["fraction"] == 1


@pytest.mark.parametrize("stop_stage", ["encoding", "prefill", "sampling", "decoding", "saving"])
def test_cancellation_releases_active_stage_and_never_returns_success(tmp_path, stop_stage):
    resident = set()
    stopped = False

    class Model:
        def __call__(self, latents, *args, **kwargs):
            return mx.zeros_like(latents)

        def decode(self, value, **kwargs):
            return mx.zeros((1, 32, 32, 4))

    class Runtime:
        def load(self, name):
            assert not resident
            resident.add(name)
            return Model()

        def release(self, name, module):
            resident.remove(name)

        def encode(self, model, prompt, images, cancel, progress):
            progress({"stage": "encoding"})
            if cancel():
                raise InterruptedError("cancelled")
            return {"features": mx.zeros((1, 1, 8)), "imageMask": [False]}

    def progress(event):
        nonlocal stopped
        if event.get("stage") == stop_stage:
            stopped = True

    request = {
        "prompt": "test",
        "inputs": [],
        "fingerprint": "test",
        "cacheMode": "recompute",
        "configuration": {"width": 32, "height": 32, "steps": 2, "seed": 42, "livePreview": False},
    }
    with pytest.raises(InterruptedError):
        render_prepared(
            request,
            {"scheduler": {}},
            tmp_path,
            cancel=lambda: stopped,
            progress=progress,
            runtime=Runtime(),
        )
    assert not resident
    assert not (tmp_path / "generation.json").exists()
    assert not list(tmp_path.glob("*.partial.*"))


def test_cancel_after_publication_does_not_revoke_committed_success(tmp_path, monkeypatch):
    from pathlib import Path

    stopped = False
    original = Path.replace

    def replace(file, target):
        nonlocal stopped
        value = original(file, target)
        if Path(target).name == "generation.json":
            stopped = True
        return value

    monkeypatch.setattr(Path, "replace", replace)

    class Model:
        def __call__(self, x, *a, **k):
            return mx.zeros_like(x)

        def decode(self, x, **k):
            return mx.zeros((1, 32, 32, 4))

    class Runtime:
        def load(self, name):
            return Model()

        def release(self, *args):
            pass

        def encode(self, *args):
            return {"features": mx.zeros((1, 1, 8)), "imageMask": [False]}

    request = {
        "prompt": "test",
        "inputs": [],
        "fingerprint": "test",
        "cacheMode": "recompute",
        "configuration": {"width": 32, "height": 32, "steps": 1, "seed": 42, "livePreview": False},
    }
    result = render_prepared(
        request,
        {"scheduler": {}},
        tmp_path,
        cancel=lambda: stopped,
        progress=lambda e: None,
        runtime=Runtime(),
    )
    assert Path(result["asset"]["path"]).is_file()
    assert (tmp_path / "generation.json").is_file()
