import importlib
import subprocess
import sys

import pytest


def test_speech_imports_do_not_load_weights_or_frameworks():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import fish_speech_mlx,qwen3_tts_mlx,sys; "
            "assert 'mlx.core' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_catalog_has_pinned_independent_model_components():
    setup = importlib.import_module("wee_todd_mlx.speech_setup")
    records = setup.catalog()
    assert len(records) == 5
    for record in records:
        assert len(record["revision"]) == 40
        assert all(len(f["sha256"]) == 64 for f in record["files"])
        assert any(
            "codec" in f["filename"] or "speech_tokenizer/" in f["filename"]
            for f in record["files"]
        )


def test_checkpoint_missing_config_fails_before_mlx_import(tmp_path):
    from fish_speech_mlx import inspect_checkpoint

    with pytest.raises((ValueError, FileNotFoundError)):
        inspect_checkpoint(tmp_path)


def test_speech_tensor_layout_rejects_unknown_shapes_before_allocation():
    import pytest

    from wee_todd_mlx.speech_checkpoint import validate_layout

    with pytest.raises(ValueError, match="tensor layout"):
        validate_layout({}, {"bad.weight": dict(shape=[3, 5], dtype="F32")}, {})


def test_speech_mask_dtype_does_not_weaken_default_checkpoint_reader(tmp_path):
    import json
    import struct

    import pytest

    from yue2_mlx.checkpoint import tensor_header

    raw = json.dumps({"mask": dict(shape=[1], dtype="BOOL", data_offsets=[0, 1])}).encode()
    p = tmp_path / "mask.safetensors"
    p.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x01")
    with pytest.raises(ValueError, match="dtype"):
        tensor_header(p)
    assert tensor_header(p, additional_widths={"BOOL": 1})["mask"]["shape"] == [1]


@pytest.mark.parametrize("existing", [True, False])
def test_setup_only_reserves_missing_verified_bytes(tmp_path, monkeypatch, existing):
    import hashlib
    from types import SimpleNamespace

    from wee_todd_mlx import speech_setup as setup

    size = 1024
    payload = b"x" * size
    f = dict(
        repo="test/model",
        revision="1" * 40,
        filename="model.bin",
        target="model.bin",
        size=size,
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(
        setup, "catalog", lambda: [dict(id="model", engine="fishS2Pro", license="test", files=[f])]
    )
    root = tmp_path / "model"
    root.mkdir()
    if existing:
        (root / "model.bin").write_bytes(payload)
    monkeypatch.setattr(
        setup.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=128 * 1024**2 + (0 if existing else size)),
    )
    monkeypatch.setattr(setup, "download_file", lambda *a, **k: None)
    assert setup.download("model", tmp_path)["model_path"] == str(root)


def test_speech_architecture_preserves_named_voice_and_language_maps():
    from wee_todd_mlx.speech_checkpoint import architecture

    config = {
        "talker_config": {
            "spk_id": {"ryan": 3061},
            "spk_is_dialect": {"eric": "sichuan_dialect"},
            "codec_language_id": {"english": 2050},
        }
    }
    assert architecture(config) == config
