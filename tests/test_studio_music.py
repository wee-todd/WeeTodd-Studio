import importlib
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))


def module():
    return importlib.import_module("studio_music")


def test_excerpt_rejects_invalid_range_before_opening_media(tmp_path):
    for start, duration in [(float("nan"), 1), (-1, 1), (0, 0), (0, float("inf"))]:
        with pytest.raises(ValueError, match="range"):
            module().prepare_excerpt(
                {"path": "missing.wav", "source_in": start, "duration": duration}, tmp_path, {}
            )


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_excerpt_is_exact_stereo_interval_and_cache_repairs_corruption(tmp_path):
    source = tmp_path / "music.wav"
    rate = 48000
    # Different signed channels and different seconds prove source offset and stereo survive.
    samples = np.empty((rate * 3, 2), dtype=np.int16)
    for i in range(3):
        samples[i * rate : (i + 1) * rate] = [1000 * (i + 1), -2000 * (i + 1)]
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(samples.tobytes())
    request = {"path": str(source), "source_in": 1.25, "duration": 0.5}
    result = module().prepare_excerpt(request, tmp_path / "excerpts", {})
    raw = subprocess.check_output(
        [shutil.which("ffmpeg"), "-v", "error", "-i", result["path"], "-f", "f32le", "-"]
    )
    actual = np.frombuffer(raw, dtype="<f4").reshape(-1, 2)
    assert actual.shape == (24000, 2)
    np.testing.assert_allclose(actual[0], [2000 / 32768, -4000 / 32768], atol=1e-6)
    assert result["source_start_sample"] == 60000
    stamp = Path(result["path"]).stat().st_mtime_ns
    assert module().prepare_excerpt(request, tmp_path / "excerpts", {}) == result
    assert Path(result["path"]).stat().st_mtime_ns == stamp
    Path(result["path"]).write_bytes(b"bad")
    repaired = module().prepare_excerpt(request, tmp_path / "excerpts", {})
    assert repaired == result
    assert Path(result["path"]).stat().st_size > 24000
    with pytest.raises(ValueError, match="beyond"):
        module().prepare_excerpt({**request, "source_in": 2.9}, tmp_path / "excerpts", {})
    assert source.stat().st_size > 280000


def test_bridge_declares_music_commands():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "scripts/studio_bridge.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "music-generate" in result.stdout
    assert "music-inspect" in result.stdout
    assert "music-excerpt" in result.stdout


@pytest.mark.parametrize("duration", [4.8, 5.0])
def test_driver_excerpt_uses_native_frame_geometry(tmp_path, duration):
    from test_studio_lora import recipe_request

    request, _ = recipe_request(tmp_path)
    runtime = request.pop("runtime")  # Bridge adds runtime to the outer envelope only.
    clip = request["project"]["clips"][0]
    clip["duration"] = duration
    settings = module().driver_excerpt_request(
        {
            "path": "/song.wav",
            "source_in": 42,
            "duration": duration,
            "runtime": runtime,
            "video_request": request,
        }
    )
    assert settings["duration"] == pytest.approx(121 / 24)
    assert settings["source_in"] == 42
    assert clip.get("generationSelection", {}).get("task") != "a2v"


def test_music_download_catalog_is_complete_and_does_not_include_executable_code():
    from wee_todd_mlx.music_setup import catalog

    record = catalog()
    assert record["license"] == "cc-by-nc-4.0"
    assert {item["target"] for item in record["files"]} == {
        "config.json",
        "model.safetensors",
        "qwen.tiktoken",
        "vae.safetensors",
        "vae_config.json",
        "yue2_generation_config.json",
    }
    assert all(len(item["sha256"]) == 64 for item in record["files"])


def test_stage_bridge_forwards_only_supported_acoustic_overrides(tmp_path, monkeypatch):
    import yue2_mlx.pipeline as native

    received = []
    monkeypatch.setattr(
        native, "resynthesize", lambda *args, **kwargs: received.append((args, kwargs))
    )

    def cancelled():
        return False

    module().dispatch(
        "music-resynthesize",
        {"source_artifacts": "/saved/take", "steps": 8, "seed": 123},
        tmp_path / "new",
        cancelled=cancelled,
    )
    assert received[0][0] == ("/saved/take", tmp_path / "new")
    assert received[0][1]["steps"] == 8
    assert received[0][1]["seed"] == 123
    assert received[0][1]["cancelled"] is cancelled
    with pytest.raises(ValueError, match="saved music take"):
        module().dispatch("music-decode", {}, tmp_path / "new")


def test_optional_vocal_setup_and_analysis_bridge_forward_selection(tmp_path, monkeypatch):
    from wee_todd_mlx.audio_analysis import service, setup, structure

    calls = []
    monkeypatch.setattr(setup, "download", lambda *args, **kwargs: calls.append(kwargs) or {})
    module().dispatch("music-analysis-setup", {"directory": str(tmp_path), "include_vocals": True})
    assert calls[-1]["include_vocals"] is True
    monkeypatch.setattr(service, "analyze", lambda *args, **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(structure, "editing_cues", lambda *args, **kwargs: [])
    module().dispatch(
        "music-analyze",
        {"audio_path": "song.wav", "audio_sha256": "a" * 64, "analysis_vocal_mode": "isolated"},
        output=tmp_path,
    )
    assert calls[-1]["vocal_mode"] == "isolated"
