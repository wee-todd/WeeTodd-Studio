"""Canonical song intervals pass through every supported native single-clip path."""

import wave
from pathlib import Path

import numpy as np
import pytest

from wee_todd_mlx.conditioning_media import inspect_media, read_media
from wee_todd_mlx.task_conditioning import validate_conditioning


def source_recipe(tmp_path, engine, *, duration=3):
    source = tmp_path / "song.wav"
    samples = np.arange(40 * 16000, dtype=np.int16)
    with wave.open(str(source), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(samples.tobytes())
    item = dict(
        id="song",
        kind="audio",
        role="audio_driver",
        path=str(source),
        source_start_seconds=12,
        source_duration_seconds=duration,
    )
    return dict(
        engine=engine,
        components=(
            {"task": "ref2va"}
            if engine == "h3"
            else {"model_dir": str(tmp_path), "gemma_model": str(tmp_path)}
        ),
        config={"pipeline_mode": "two_stage", "duration_seconds": 3, "frame_rate": 24},
        prompt="A singer sways to music.",
        conditioning=dict(version=1, task="a2v", inputs=[item]),
    )


@pytest.mark.parametrize("engine,policy", [("h3", "generated"), ("ltx23", "source")])
def test_native_single_clip_accepts_source_interval_and_keeps_audio_policy(
    tmp_path, engine, policy
):
    recipe = source_recipe(tmp_path, engine)
    contract = validate_conditioning(recipe)["contract"]
    assert contract["audio_policy"] == policy
    reports = inspect_media(recipe, contract)
    audio = read_media(recipe, contract["inputs"][0], reports[0])
    assert audio["sample_rate"] == (32000 if engine == "h3" else 16000)
    assert audio["waveform"].shape[-1] == audio["sample_rate"] * 3
    assert audio["waveform"].shape[1] == (2 if engine == "h3" else 1)


@pytest.mark.parametrize("engine,duration", [("h3", 15.01), ("ltx23", 30.01)])
def test_native_interval_limits_reject_before_weights(tmp_path, engine, duration):
    with pytest.raises(ValueError, match="source_duration_seconds"):
        validate_conditioning(source_recipe(tmp_path, engine, duration=duration))


@pytest.mark.parametrize("failure", [False, True])
def test_ltx23_headless_receives_only_selected_audio_and_cleans_temporary(
    tmp_path, monkeypatch, failure
):
    from test_studio_scene_headless import renderer

    from ltx23_mlx.runtime import RUNTIME

    recipe = source_recipe(tmp_path, "ltx23")
    paths = []

    def generate(*args, **kwargs):
        selected = Path(kwargs["audio_path"])
        paths.append(selected)
        assert selected != Path(recipe["conditioning"]["inputs"][0]["path"])
        item = dict(id="prepared", kind="audio", role="audio_driver", path=str(selected))
        report = inspect_media(recipe, {"inputs": [item]})[0]
        assert report["duration_seconds"] == 3
        audio = read_media(recipe, item, report)
        original = read_media(
            recipe,
            recipe["conditioning"]["inputs"][0],
            inspect_media(recipe, recipe["conditioning"])[0],
        )
        np.testing.assert_allclose(audio["waveform"][0, 0], original["waveform"][0, 0], atol=1e-7)
        if failure:
            raise RuntimeError("controlled failure")
        return {"video_path": str(tmp_path / "result.mp4")}

    monkeypatch.setattr(RUNTIME, "generate_to_file", generate)
    monkeypatch.setattr(RUNTIME, "unload", lambda: None)
    if failure:
        with pytest.raises(RuntimeError, match="controlled failure"):
            renderer.render_ltx(recipe, tmp_path / "result.mp4")
    else:
        renderer.render_ltx(recipe, tmp_path / "result.mp4")
    assert len(paths) == 1 and not paths[0].exists()
