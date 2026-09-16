import copy
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
bridge = importlib.import_module("studio_bridge")


def request_for(tmp_path, engine="ltx25", mode="frame"):
    source = tmp_path / "source.mp4"
    if not source.exists():
        source.write_bytes(b"movie")
    return {
        "clipID": "b",
        "runtime": {},
        "project": {
            "assets": [],
            "settings": {},
            "clips": [
                {
                    "id": "a",
                    "engine": engine,
                    "name": "Source",
                    "sourcePath": str(source),
                    "sourceIn": 1.0,
                    "duration": 3.0,
                    "versions": [{"id": "take-a", "path": str(source), "usableDuration": 5.0}],
                },
                {
                    "id": "b",
                    "engine": engine,
                    "duration": 5.0,
                    "attachments": [],
                    "generationSelection": {"task": "t2v", "preset": "balanced"},
                    "continuity": {"mode": mode},
                },
            ],
        },
    }


def test_source_selection_is_backward_only_and_does_not_mutate(tmp_path):
    from wee_todd_mlx.studio_continuity import continuity_state

    request = request_for(tmp_path)
    original = copy.deepcopy(request)
    state = continuity_state(request)
    assert state["sourceClipID"] == "a"
    assert state["sourceTakeID"] == "take-a"
    assert state["sourceIn"] == 1.0 and state["duration"] == 3.0
    assert request == original
    request["project"]["clips"][1]["continuity"]["sourceClipID"] = "b"
    with pytest.raises(ValueError, match="earlier"):
        continuity_state(request)


def test_h3_motion_prompt_aligns_pictures_with_shifted_sample_anchors():
    from wee_todd_mlx.studio_continuity import configure_recipe

    body = ("integrated_multimodal_description: [Shot 1] A robot takes a step.\n"
            "overall_soundscape: Footsteps.\nnon_diegetic_music: N/A")
    recipe = {
        "engine": "h3", "components": {"task": "fl2va"},
        "config": {"duration_seconds": 4.96},
        "conditioning": {"task": "fflf", "inputs": [
            {"frame_index": "last"}, {"frame_index": 0}]},
        "prompt": ("Picture 1 is fully referenced at 0.00 seconds. "
                   "Picture 2 is fully referenced at 5.125 seconds.\n" + body),
    }
    state = {"mode": "motion", "saveContext": True, "artifact": {
        "manifest": "/context/manifest.json", "manifest_sha256": "a" * 64}}
    configure_recipe(recipe, state)
    assert recipe["prompt"] == (
        "Picture 1 is fully referenced at 0.916667 seconds. "
        "Picture 2 is fully referenced at 5.833333 seconds.\n" + body)
    assert state["usableDuration"] == 119 / 24


def test_h3_motion_refuses_missing_or_trimmed_context(tmp_path):
    from wee_todd_mlx.studio_continuity import continuity_state

    request = request_for(tmp_path, "h3", "motion")
    with pytest.raises(ValueError, match="Save motion context"):
        continuity_state(request)
    take = request["project"]["clips"][0]["versions"][0]
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    take["continuationArtifact"] = {
        "manifest": str(manifest),
        "manifest_sha256": "a" * 64,
        "payload_sha256": "b" * 64,
    }
    with pytest.raises(ValueError, match="trimmed"):
        continuity_state(request)


def test_ltx_motion_preserves_and_rejects_conflicting_media(tmp_path):
    from wee_todd_mlx.studio_continuity import continuity_state

    request = request_for(tmp_path, mode="motion")
    request["project"]["clips"][1]["attachments"] = [{"role": "last", "assetID": "end"}]
    original = copy.deepcopy(request)
    with pytest.raises(ValueError, match="Match previous frame"):
        continuity_state(request)
    assert request == original


def test_h3_recording_is_inferred_from_direct_dependent(tmp_path):
    from wee_todd_mlx.studio_continuity import continuity_state

    request = request_for(tmp_path, "h3", "motion")
    request["clipID"] = "a"
    state = continuity_state(request)
    assert state["saveContext"] is True
    assert state["mode"] == "independent"


def test_prepared_snapshot_refuses_replaced_source(tmp_path):
    from wee_todd_mlx.studio_continuity import continuity_state, verify_source

    request = request_for(tmp_path)
    state = continuity_state(request)
    verify_source(state)
    Path(state["sourcePath"]).write_bytes(b"changed movie")
    with pytest.raises(ValueError, match="changed"):
        verify_source(state)


@pytest.fixture
def source_movie(tmp_path):
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if not Path(ffmpeg).is_file():
        pytest.skip("ffmpeg unavailable")
    target = tmp_path / "source.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=128x128:rate=24:duration=5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=5",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-y",
            str(target),
        ],
        check=True,
    )
    return target


def test_frame_materializes_last_visible_frame_without_replacing_saved_first(
    tmp_path, source_movie
):
    from wee_todd_mlx.studio_continuity import prepare_request

    request = request_for(tmp_path)
    request["project"]["clips"][1]["attachments"] = [
        {"id": "first", "role": "first", "assetID": "old"},
        {"id": "last", "role": "last", "assetID": "end"},
    ]
    original = copy.deepcopy(request)
    updated, state = prepare_request(request, tmp_path / "prepared", bridge)
    clip = updated["project"]["clips"][1]
    assert request == original
    assert [a["role"] for a in clip["attachments"]] == ["last", "first"]
    asset = updated["project"]["assets"][-1]
    assert bridge.inspect_media(asset["path"], {})["kind"] == "image"
    assert state["sourceTimeEnd"] == 4.0  # visible [1s,4s)
    assert clip["generationSelection"]["task"] == "fflf"


def test_ltx_motion_extracts_visible_audio_video_tail_on_native_grid(tmp_path, source_movie):
    from wee_todd_mlx.studio_continuity import configure_recipe, prepare_request

    request = request_for(tmp_path)
    request["project"]["clips"][1]["continuity"]["mode"] = "motion"
    updated, state = prepare_request(request, tmp_path / "prepared", bridge, fps=24)
    clip = updated["project"]["clips"][1]
    media = bridge.inspect_media(clip["extensionSource"], {})
    assert media["hasAudio"] and media["fps"] == 24
    assert media["duration"] == pytest.approx(49 / 24, abs=0.03)
    recipe = {
        "engine": "ltx25",
        "config": {"duration_seconds": 5, "frame_rate": 24},
        "conditioning": {"extension": {"additional_frames": 120, "context_frames": 25}},
    }
    configure_recipe(recipe, state)
    from wee_todd_mlx.task_conditioning import frame_geometry

    assert frame_geometry(recipe)[0] == 145  # 25 overlap+120 new
    assert state["usableSourceIn"] == 49 / 24


def test_bridge_render_returns_new_span_and_checks_continuity_sidecar(tmp_path, monkeypatch):
    from wee_todd_mlx.studio_continuity import continuity_state

    request = request_for(tmp_path)
    state = continuity_state(request)
    state.update(usableSourceIn=49 / 24, usableDuration=5)
    recipe = tmp_path / "recipe.json"
    recipe.write_text("{}")
    recipe.with_name("continuity.json").write_text(json.dumps(state))
    output = tmp_path / "output"
    output.mkdir()
    (output / "result.json").write_text(json.dumps({"status": "success", "video": "result.mp4"}))
    monkeypatch.setattr(bridge, "run", lambda *args, **kwargs: None)
    result = bridge.render(str(recipe), output)
    assert result["usable_source_in"] == 49 / 24
    assert result["usable_duration"] == 5
    Path(state["sourcePath"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        bridge.render(str(recipe), output)


def test_clip_job_resolves_source_before_narrowing_project(tmp_path, monkeypatch):
    jobs = importlib.import_module("studio_job")
    request = request_for(tmp_path)
    request.update(clipOnly=True, generateIDs=["b"])
    request["project"].update(name="Movie", titles=[], audio=[])
    for clip in request["project"]["clips"]:
        clip.update(name=clip["id"], transition="cut")

    def compose(current):
        from wee_todd_mlx.studio_continuity import continuity_state

        return {"prompt": "test"}, {"continuity": continuity_state(current)}

    monkeypatch.setattr(bridge, "compose_recipe", compose)
    target = tmp_path / "job.json"
    jobs.export_job(request, target)
    job = json.loads(target.read_text())
    assert len(job["project"]["clips"]) == 1
    assert job["recipes"]["b"]["report"]["continuity"]["sourceClipID"] == "a"


def test_movie_job_does_not_silently_use_old_take_when_predecessor_is_queued(tmp_path):
    jobs = importlib.import_module("studio_job")
    request = request_for(tmp_path)
    request.update(generateIDs=["a", "b"])
    with pytest.raises(ValueError, match="accept.*source|source.*accept"):
        jobs.export_job(request, tmp_path / "job.json")


def test_vfr_source_uses_last_visible_timestamp_not_average_fps(tmp_path):
    from PIL import Image

    from wee_todd_mlx.studio_continuity import prepare_request

    for color in ("red", "green", "blue"):
        Image.new("RGB", (128, 128), color).save(tmp_path / f"{color}.png")
    concat = tmp_path / "frames.txt"
    concat.write_text("file 'red.png'\nduration 1\nfile 'green.png'\nduration 3\n"
                      "file 'blue.png'\nduration 2\nfile 'blue.png'\n")
    subprocess.run([bridge.executable("ffmpeg", {}), "-v", "error", "-f", "concat", "-i",
                    str(concat), "-fps_mode", "vfr", "-pix_fmt", "yuv420p",
                    str(tmp_path / "source.mp4")], check=True)
    request = request_for(tmp_path)
    request["project"]["clips"][0].update(sourceIn=0, duration=3.5)
    updated, _ = prepare_request(request, tmp_path / "prepared", bridge)
    with Image.open(updated["project"]["assets"][-1]["path"]) as image:
        red, green, blue = image.convert("RGB").getpixel((64, 64))
    assert green > red + 80 and green > blue + 80


def test_failed_export_cleans_only_its_own_inputs_and_can_retry(tmp_path, monkeypatch):
    jobs = importlib.import_module("studio_job")
    request = request_for(tmp_path)
    request.update(clipOnly=True, generateIDs=["b"])
    request["project"].update(name="Movie", titles=[], audio=[])
    for clip in request["project"]["clips"]:
        clip.update(name=clip["id"], transition="cut")
    created = []

    def fail_after_materializing(current):
        directory = Path(current["_continuityDirectory"])
        directory.mkdir(parents=True)
        (directory / "previous-frame.png").write_bytes(b"frame")
        created.append(directory)
        raise ValueError("later recipe invalid")

    monkeypatch.setattr(bridge, "compose_recipe", fail_after_materializing)
    for _ in range(2):
        with pytest.raises(ValueError, match="later recipe invalid"):
            jobs.export_job(request, tmp_path / "job.json")
    assert all(not directory.exists() for directory in created)


def test_h3_save_preference_remains_inactive_after_switch_to_ltx(tmp_path):
    from wee_todd_mlx.studio_continuity import continuity_state

    request = request_for(tmp_path, mode="independent")
    request["project"]["clips"][1]["continuity"]["saveContext"] = True
    assert continuity_state(request)["saveContext"] is False


def test_ltx_tail_keeps_49_frames_with_rounded_visible_endpoint(tmp_path, source_movie):
    from wee_todd_mlx.studio_continuity import prepare_request

    request = request_for(tmp_path, mode="motion")
    request["project"]["clips"][0].update(sourceIn=0, duration=5.003)
    updated, _ = prepare_request(request, tmp_path / "prepared", bridge)
    tail = updated["project"]["clips"][1]["extensionSource"]
    probe = json.loads(bridge.run([bridge.executable("ffprobe", {}), "-v", "error",
                                  "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
                                  "-of", "json", tail], capture=True))
    assert int(probe["streams"][0]["nb_frames"]) == 49


def test_freeze_continuity_frame_keeps_explicit_source_visible_trim(tmp_path, monkeypatch):
    from PIL import Image

    for color in ('red', 'green', 'blue'):
        Image.new('RGB', (64, 64), color).save(tmp_path / f'{color}.png')
    concat = tmp_path / 'colors.txt'
    concat.write_text("file 'red.png'\nduration 1\nfile 'green.png'\nduration 3\n"
                      "file 'blue.png'\nduration 2\nfile 'blue.png'\n")
    source = tmp_path / 'source.mp4'
    subprocess.run([bridge.executable('ffmpeg', {}), '-v', 'error', '-f', 'concat', '-i',
                    str(concat), '-fps_mode', 'vfr', '-pix_fmt', 'yuv420p', str(source)],
                   check=True)
    request = request_for(tmp_path)
    request['project']['clips'][0].update(sourceIn=0.5, duration=3)
    request['project']['clips'].insert(1, dict(id='middle', engine='movie', duration=1))
    request['project']['clips'][-1]['continuity']['sourceClipID'] = 'a'
    request['project']['clips'][-1]['attachments'] = [
        dict(id='saved-first', role='first', assetID='original')]
    original = copy.deepcopy(request)

    def no_model_resolution(*args, **kwargs):
        pytest.fail('Freezing a media frame must not resolve generation models.')

    monkeypatch.setattr(bridge, 'resolve_clip_generation', no_model_resolution)
    result = bridge.freeze_continuity_frame(request, tmp_path / 'frozen')
    assert result == dict(path=str((tmp_path / 'frozen/previous-frame.png').resolve()),
                          sourceClipID='a', sourceTakeID='take-a')
    assert request == original
    with Image.open(result['path']) as frame:
        red, green, blue = frame.convert('RGB').getpixel((32, 32))
    assert green > red + 80 and green > blue + 80
    with pytest.raises(FileExistsError):
        bridge.freeze_continuity_frame(request, tmp_path / 'frozen')


@pytest.mark.parametrize('mode', ['independent', 'scene', 'motion'])
def test_freeze_continuity_frame_rejects_other_modes(tmp_path, mode):
    request = request_for(tmp_path, mode=mode)
    with pytest.raises(ValueError, match='Match previous frame'):
        bridge.freeze_continuity_frame(request, tmp_path / 'frozen')
    assert not (tmp_path / 'frozen').exists()


def test_freeze_continuity_frame_rejects_source_changed_during_extraction(
    tmp_path, source_movie, monkeypatch
):
    request = request_for(tmp_path)
    run = bridge.run

    def replace_after_extraction(command, **kwargs):
        result = run(command, **kwargs)
        if 'previous-frame.png' in str(command[-1]):
            source_movie.touch()
        return result

    monkeypatch.setattr(bridge, 'run', replace_after_extraction)
    with pytest.raises(ValueError, match='source file changed'):
        bridge.freeze_continuity_frame(request, tmp_path / 'frozen')


def test_freeze_continuity_frame_cli_dispatch(tmp_path, source_movie):
    request = tmp_path / 'request.json'
    request.write_text(json.dumps(request_for(tmp_path)))
    result = subprocess.run([sys.executable, str(Path(bridge.__file__)),
                             'freeze-continuity-frame', '--request', str(request),
                             '--output', str(tmp_path / 'frozen')],
                            capture_output=True, text=True, check=True)
    response = json.loads(result.stdout.strip().splitlines()[-1])
    assert response['status'] == 'success'
    assert response['result']['sourceClipID'] == 'a'
    assert Path(response['result']['path']).is_file()
