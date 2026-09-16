"""Durable scene artifacts must be complete, bounded and tied to their prefix."""

import importlib
import json

import mlx.core as mx
import pytest


def store(tmp_path):
    return importlib.import_module("ltx25_mlx.chain_checkpoints").ChainCheckpointStore(tmp_path)


def test_roundtrip_original_latents_and_continuation(tmp_path):
    checkpoint = store(tmp_path)
    arrays = {
        "video": mx.ones((1, 128, 5, 2, 2), dtype=mx.bfloat16),
        "audio": mx.ones((1, 8, 34, 16)),
        "stage1": mx.ones((1, 4, 128)),
        "stage2": mx.ones((1, 16, 128)),
        "audio_tail": mx.ones((1, 26, 128)),
    }
    shapes = {key: value.shape for key, value in arrays.items()}
    checkpoint.save(0, "a" * 64, arrays, shapes)
    loaded = checkpoint.load(0, "a" * 64, shapes)
    assert loaded is not None
    assert set(loaded) == set(arrays)
    for key in arrays:
        assert mx.array_equal(loaded[key], arrays[key])
        assert loaded[key].dtype == arrays[key].dtype
    assert checkpoint.load(0, "b" * 64, shapes) is None


def test_corruption_incomplete_and_mismatched_shape_are_not_reused(tmp_path):
    checkpoint = store(tmp_path)
    arrays = {"video": mx.ones((1, 128, 5, 2, 2))}
    shapes = {"video": arrays["video"].shape}
    assert checkpoint.load(0, "a" * 64, shapes) is None
    checkpoint.save(0, "a" * 64, arrays, shapes)
    assert checkpoint.load(0, "a" * 64, {"video": (1, 128, 6, 2, 2)}) is None
    payload = next(tmp_path.glob("*.safetensors"))
    data = bytearray(payload.read_bytes())
    data[-1] ^= 0xFF
    payload.write_bytes(data)
    assert checkpoint.load(0, "a" * 64, shapes) is None
    checkpoint.save(0, "a" * 64, arrays, shapes)
    assert checkpoint.load(0, "a" * 64, shapes) is not None


def test_header_shape_bomb_rejected_before_mlx_load(tmp_path, monkeypatch):
    checkpoint = store(tmp_path)
    arrays = {"video": mx.ones((1, 128, 5, 2, 2))}
    shapes = {"video": arrays["video"].shape}
    checkpoint.save(0, "a" * 64, arrays, shapes)
    manifest_path = next(tmp_path.glob("window-*.json"))
    manifest = json.loads(manifest_path.read_text())
    payload = tmp_path / manifest["payload"]
    data = payload.read_bytes()
    header_size = int.from_bytes(data[:8], "little")
    header = json.loads(data[8 : 8 + header_size])
    header["video"]["shape"] = [2**60]
    new_header = json.dumps(header).encode()
    payload.write_bytes(
        len(new_header).to_bytes(8, "little") + new_header + data[8 + header_size :]
    )
    monkeypatch.setattr(mx, "load", lambda *_a, **_k: pytest.fail("unbounded tensor loader called"))
    assert checkpoint.load(0, "a" * 64, shapes) is None


def test_prefix_invalidates_all_descendants_but_keeps_unchanged_ancestors():
    module = importlib.import_module("ltx25_mlx.chain_checkpoints")
    a = module.prefix_identity("shared", {"prompt": "a"})
    b = module.prefix_identity(a, {"prompt": "b"})
    changed_b = module.prefix_identity(a, {"prompt": "new b"})
    assert a == module.prefix_identity("shared", {"prompt": "a"})
    assert b != changed_b
    assert module.prefix_identity(b, {"prompt": "c"}) != module.prefix_identity(
        changed_b, {"prompt": "c"}
    )


def test_file_identity_changes_with_content_even_same_size(tmp_path):
    module = importlib.import_module("ltx25_mlx.chain_checkpoints")
    filename = tmp_path / "model.safetensors"
    filename.write_bytes(b"a" * 10)
    first = module.content_identity(filename)
    filename.write_bytes(b"b" * 10)
    assert first != module.content_identity(filename)


def test_manifest_traversal_and_truncated_payload_are_ignored(tmp_path):
    checkpoint = store(tmp_path)
    arrays = {"video": mx.ones((1, 128, 5, 2, 2))}
    shapes = {"video": arrays["video"].shape}
    checkpoint.save(0, "a" * 64, arrays, shapes)
    manifest_path = next(tmp_path.glob("window-*.json"))
    manifest = json.loads(manifest_path.read_text())
    manifest["payload"] = "../outside.safetensors"
    manifest_path.write_text(json.dumps(manifest))
    assert checkpoint.load(0, "a" * 64, shapes) is None
    checkpoint.save(0, "a" * 64, arrays, shapes)
    payload = next(tmp_path.glob("*.safetensors"))
    payload.write_bytes(payload.read_bytes()[:20])
    assert checkpoint.load(0, "a" * 64, shapes) is None


def test_model_identities_use_durable_cache_but_images_reverify_content(tmp_path, monkeypatch):
    module = importlib.import_module("ltx25_mlx.chain_checkpoints")
    monkeypatch.setattr(
        "wee_todd_mlx.model_hash_cache._database", lambda: tmp_path / "hashes.sqlite3"
    )
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"weights")
    hashes = []
    original = module._sha256

    def hash_file(filename):
        hashes.append(filename)
        return original(filename)

    monkeypatch.setattr(module, "_sha256", hash_file)
    first = module.content_identity(source, model_file=True)
    assert module.content_identity(source, model_file=True) == first
    assert len(hashes) == 1
    assert module.content_identity(source) == first
    assert module.content_identity(source) == first
    assert len(hashes) == 3
