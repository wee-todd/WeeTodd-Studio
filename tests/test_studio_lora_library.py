"""Folder discovery must stay bounded, payload-free and independent of clip state."""

import importlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
lora = importlib.import_module("studio_lora")


def write_adapter(source, metadata):
    header = json.dumps({
        "__metadata__": metadata,
        "block.lora_A.weight": {"dtype": "F32", "shape": [1, 2], "data_offsets": [0, 8]},
        "block.lora_B.weight": {"dtype": "F32", "shape": [2, 1], "data_offsets": [8, 16]},
    }).encode()
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(struct.pack("<Q", len(header)) + header + bytes(16))
    return source


def test_folder_scan_deduplicates_nested_roots_and_survives_missing_folder(tmp_path):
    source = write_adapter(tmp_path / "library/sub/style.safetensors", {"model_version": "2.5.0"})
    alias = tmp_path / "alias"
    alias.symlink_to(source.parent, target_is_directory=True)
    result = lora.scan_lora_folders([
        {"path": str(tmp_path / "library")}, {"path": str(alias)},
        {"path": str(tmp_path / "missing")},
        {"path": str(tmp_path / "disabled"), "enabled": False},
    ], tmp_path / "cache.json")
    assert len(result["entries"]) == 1
    assert result["entries"][0]["path"] == str(source.resolve())
    assert result["entries"][0]["loraModel"] == "ltx25"
    assert result["entries"][0]["status"] == "ready"
    assert any("missing" in warning for warning in result["warnings"])
    assert not any("disabled" in warning for warning in result["warnings"])


def test_scan_cache_invalidates_changed_file_and_removes_deleted_entries(tmp_path, monkeypatch):
    source = write_adapter(tmp_path / "library/style.safetensors", {"model_version": "2.3.0"})
    folders = [{"path": str(source.parent)}]
    cache = tmp_path / "cache.json"
    first = lora.scan_lora_folders(folders, cache)
    original = lora.inspect_lora

    def no_reinspection(*args, **kwargs):
        raise AssertionError("An unchanged file should reuse its metadata inspection")

    monkeypatch.setattr(lora, "inspect_lora", no_reinspection)
    assert lora.scan_lora_folders(folders, cache)["entries"] == first["entries"]
    monkeypatch.setattr(lora, "inspect_lora", original)
    write_adapter(source, {"model_version": "2.5.0"})
    assert lora.scan_lora_folders(folders, cache)["entries"][0]["loraModel"] == "ltx25"
    source.unlink()
    assert lora.scan_lora_folders(folders, cache)["entries"] == []


def test_folder_scan_explains_unknown_and_specialized_adapters_and_skips_dt_weights(tmp_path):
    write_adapter(tmp_path / "unknown.safetensors", {})
    write_adapter(tmp_path / "control.safetensors", {"adapter_role": "control"})
    (tmp_path / "drawthings.ckpt").write_bytes(b"opaque DT store")
    (tmp_path / "broken.safetensors").write_bytes(b"bad header")
    result = lora.scan_lora_folders([{"path": str(tmp_path)}], tmp_path / "cache.json")
    entries = {entry["name"]: entry for entry in result["entries"]}
    assert entries["unknown"]["status"] == "needsModel"
    assert entries["control"]["status"] == "specialized"
    assert entries["broken"]["status"] == "unsupported"
    assert "drawthings" not in entries
    assert any("Draw Things" in warning for warning in result["warnings"])


def test_folder_model_hint_is_fallback_only_and_nonrecursive_scan_is_respected(tmp_path):
    write_adapter(tmp_path / "native.safetensors", {"model_version": "2.5.0"})
    write_adapter(tmp_path / "unknown.safetensors", {})
    write_adapter(tmp_path / "sub/hidden.safetensors", {})
    result = lora.scan_lora_folders([
        {"path": str(tmp_path), "modelHint": "h3", "recursive": False}
    ], tmp_path / "cache.json")
    assert {entry["name"]: entry["loraModel"] for entry in result["entries"]} == {
        "native": "ltx25", "unknown": "h3"
    }


def test_scan_handles_symlink_cycles_and_reports_scan_limit(tmp_path, monkeypatch):
    (tmp_path / "cycle").symlink_to(tmp_path, target_is_directory=True)
    for index in range(3):
        write_adapter(tmp_path / f"{index}.safetensors", {})
    monkeypatch.setattr(lora, "MAX_LIBRARY_FILES", 2, raising=False)
    result = lora.scan_lora_folders([{"path": str(tmp_path)}], tmp_path / "cache.json")
    assert len(result["entries"]) == 2
    assert any("limit" in warning.lower() for warning in result["warnings"])


def test_malformed_cache_is_rebuilt_from_headers(tmp_path):
    source = write_adapter(tmp_path / "style.safetensors", {"model_version": "2.5.0"})
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"version": 1, "files": {str(source): ["broken"]}}))
    result = lora.scan_lora_folders([{"path": str(tmp_path)}], cache)
    assert result["entries"][0]["status"] == "ready"
    assert result["entries"][0]["loraModel"] == "ltx25"
