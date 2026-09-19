"""Studio covers editorial trims without changing the native nearest-grid policy."""

import copy
import json
from pathlib import Path

import pytest
from test_studio_lora import recipe_request

from wee_todd_mlx.task_conditioning import frame_geometry


@pytest.mark.parametrize("engine", ["ltx23", "ltx25"])
@pytest.mark.parametrize("duration", [4.8, 5.0, 5.01, 1.1, 30.0])
def test_manual_ltx_recipe_covers_editorial_duration_without_music_metadata(
    tmp_path, monkeypatch, engine, duration,
):
    request, compose = recipe_request(tmp_path, engine=engine)
    clip = request["project"]["clips"][0]
    clip["duration"] = duration
    before = copy.deepcopy(request)
    monkeypatch.setattr("wee_todd_mlx.task_conditioning.validate_conditioning", lambda _: {})
    recipe, report = compose(request)
    frames, fps = frame_geometry(recipe)
    assert frames / fps >= duration
    assert report["preserveEditorialDuration"] is True
    assert request == before
    if duration == 4.8:
        assert recipe["config"]["duration_seconds"] == 5.0
        assert frames == 121


def test_automatic_ltx_duration_remains_model_owned(tmp_path, monkeypatch):
    request, compose = recipe_request(tmp_path)
    clip = request["project"]["clips"][0]
    clip["duration"] = 4.8
    profile = Path(clip["profileID"])
    recipe = json.loads(profile.read_text())
    recipe["config"]["duration_mode"] = "automatic"
    profile.write_text(json.dumps(recipe))
    monkeypatch.setattr("wee_todd_mlx.task_conditioning.validate_conditioning", lambda _: {})
    recipe, report = compose(request)
    assert recipe["config"]["duration_seconds"] == 4.8
    assert report["preserveEditorialDuration"] is False


def test_one_frame_i2v_rounds_up_and_keeps_first_anchor(tmp_path, monkeypatch):
    request, compose = recipe_request(tmp_path)
    clip = request["project"]["clips"][0]
    clip.update(duration=4.8, attachments=[dict(id="first", assetID="image", role="first")])
    request["project"]["assets"].append(dict(id="image", kind="image", path="/tmp/first.png"))
    monkeypatch.setattr("wee_todd_mlx.task_conditioning.validate_conditioning", lambda _: {})
    recipe, _ = compose(request)
    assert frame_geometry(recipe)[0] == 121
    assert recipe["conditioning"]["inputs"][0]["frame_index"] == 0
