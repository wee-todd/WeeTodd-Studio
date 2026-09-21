import json
from pathlib import Path
from types import SimpleNamespace


def test_setup_resumes_below_original_full_space_requirement_and_repairs_receipt(
    tmp_path, monkeypatch
):
    import shutil

    import huggingface_hub
    import studio_image

    from qwen_image21_mlx import checkpoint, convert
    from wee_todd_mlx import inference_lease

    target = tmp_path / "Qwen-Image-2.1-8bit"
    target.mkdir()
    (target / "manifest.json").write_text("{broken")
    monkeypatch.setattr(checkpoint, "read_json", lambda p: {"files": {"LICENSE": {"bytes": 4}}})
    monkeypatch.setattr(shutil, "disk_usage", lambda p: SimpleNamespace(free=30 * 2**30))
    downloaded = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: downloaded.append(a))

    class Lease:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(inference_lease, "InferenceLease", Lease)

    def conversion(source, destination, **kwargs):
        assert not (destination / "manifest.json").exists()
        assert list(destination.glob("manifest.invalid-*.json"))
        (destination / "manifest.json").write_text(json.dumps({"repaired": True}))
        return destination / "manifest.json"

    monkeypatch.setattr(convert, "convert", conversion)
    result = studio_image.prepare_model(tmp_path, cancelled=lambda: False, progress=lambda e: None)
    assert Path(result["manifestPath"]).is_file()
    assert downloaded
