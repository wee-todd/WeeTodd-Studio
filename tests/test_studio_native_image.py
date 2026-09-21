from pathlib import Path

import pytest
from test_native_image_contracts import request_factory as request_factory

from wee_todd_mlx import image_service


def test_native_preflight_has_no_remote_requirements(request_factory, monkeypatch):
    manifest = {
        "manifestFingerprint": "model",
        "components": {
            name: {"weightBytes": 100} for name in ("text_encoder", "transformer", "vae")
        },
    }
    monkeypatch.setattr(image_service, "inspect_manifest", lambda *a, **k: manifest)
    report = image_service.prepare_image(request_factory(10))
    assert report["eligibility"] == "allowed"
    assert len(report["normalizedRequest"]["inputs"]) == 10
    assert "CU" not in report["memorySummary"]


def test_generate_rechecks_before_loading(request_factory, monkeypatch, tmp_path):
    request = request_factory(1)
    Path(request["inputs"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        image_service.generate_image(request, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_missing_model_returns_actionable_preflight(request_factory):
    report = image_service.prepare_image(request_factory(0))
    assert report["eligibility"] == "blocked"
    assert report["issues"]


def test_generation_rejects_changed_preflight_identity(request_factory, monkeypatch, tmp_path):
    manifest = {
        "manifestFingerprint": "model",
        "components": {
            name: {"weightBytes": 100} for name in ("text_encoder", "transformer", "vae")
        },
    }
    monkeypatch.setattr(image_service, "inspect_manifest", lambda *a, **k: manifest)
    request = request_factory(0)
    fingerprint = image_service.prepare_image(request)["fingerprint"]
    request["prompt"] = "Different prompt"
    with pytest.raises(ValueError, match="changed since"):
        image_service.generate_image(request, tmp_path / "out", expected_fingerprint=fingerprint)


def test_memory_is_admitted_only_after_waiting_and_is_not_preflight_identity(
    request_factory, monkeypatch, tmp_path
):
    from qwen_image21_mlx import pipeline

    manifest = {
        "manifestFingerprint": "model",
        "components": {
            name: {"weightBytes": 100} for name in ("text_encoder", "transformer", "vae")
        },
    }
    monkeypatch.setattr(image_service, "inspect_manifest", lambda *a, **k: manifest)
    request = request_factory(0)
    fingerprint = image_service.prepare_image(request)["fingerprint"]
    acquired = False

    class Lease:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            nonlocal acquired
            acquired = True

        def __exit__(self, *args):
            pass

    def memory(*args):
        assert acquired, "Memory was checked while another job still held the lease"
        return {"fits": True, "cacheMode": "recompute"}

    monkeypatch.setattr(image_service, "InferenceLease", Lease)
    monkeypatch.setattr(image_service, "estimate_memory", memory)
    monkeypatch.setattr(pipeline, "render_prepared", lambda request, *a, **k: request)
    result = image_service.generate_image(
        request, tmp_path / "out", expected_fingerprint=fingerprint
    )
    assert result["cacheMode"] == "recompute"
