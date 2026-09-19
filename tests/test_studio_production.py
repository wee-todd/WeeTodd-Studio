"""Persistent production behavior with the expensive renderer replaced at its boundary."""

import copy
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture
def production(tmp_path, monkeypatch):
    module = importlib.import_module("studio_production")
    calls = []

    def prepare(request, folder):
        calls.append(("prepare", copy.deepcopy(request)))
        folder.mkdir(parents=True)
        recipe = folder / "recipe.json"
        recipe.write_text("{}")
        return {"recipePath": str(recipe), "prompt": "Test", "report": {}}

    def render(recipe, folder, **kwargs):
        calls.append(("render", recipe))
        folder.mkdir(parents=True)
        target = folder / "video.mp4"
        target.write_bytes(b"rendered video")
        return {"video": str(target)}

    def export(request, target, **kwargs):
        calls.append(("export", copy.deepcopy(request)))
        target.write_bytes(b"assembled movie")
        return {"path": str(target)}

    monkeypatch.setattr(module.bridge, "prepare", prepare)
    monkeypatch.setattr(module.bridge, "render", render)
    monkeypatch.setattr(module.bridge, "export_movie", export)
    monkeypatch.setattr(
        module.bridge,
        "inspect_media",
        lambda *a, **k: {
            "kind": "video",
            "duration": 4.0 if str(a[0]).endswith(".wav") else 2.0,
            "fps": 24.0,
            "hasAudio": True,
        },
    )
    monkeypatch.setattr(module.bridge, "preflight_finishing", lambda *a: None)
    return module, calls


def request(tmp_path):
    song = tmp_path / "song.wav"
    song.write_bytes(b"original stereo")
    return dict(
        project=dict(
            id="project",
            settings=dict(format="mp4", fps=24),
            clips=[
                dict(
                    id=str(i),
                    name=f"Shot {i}",
                    engine="ltx25",
                    prompt="Test",
                    seed=42,
                    duration=2.0,
                    sourceIn=0.0,
                    sourcePath="",
                    versions=[],
                    attachments=[],
                )
                for i in range(2)
            ],
            assets=[],
            audio=[dict(path=str(song), start=0.0, sourceIn=0.0, duration=4.0, trackID="music")],
        ),
        runtime={},
        globalAssets=[],
        generateIDs=["0", "1"],
        maxRetries=1,
    )


def test_serial_continuity_uses_completed_predecessor_and_preserves_music(production, tmp_path):
    module, calls = production
    body = request(tmp_path)
    body["project"]["clips"][1]["continuity"] = {"mode": "frame", "sourceClipID": "0"}
    module.create(body, tmp_path / "job")
    result = module.run(tmp_path / "job")
    prepares = [r for kind, r in calls if kind == "prepare"]
    assert prepares[0]["project"]["clips"][0]["sourcePath"] == ""
    assert prepares[1]["project"]["clips"][0]["sourcePath"].endswith("video.mp4")
    assert prepares[1]["project"]["clips"][0]["versions"][-1]["path"].endswith("video.mp4")
    assert result["status"] == "completed"
    assert result["resolvedProject"]["audio"] == body["project"]["audio"]
    assert body["project"]["clips"][0]["sourcePath"] == ""
    assert [kind for kind, _ in calls] == ["prepare", "render", "prepare", "render", "export"]


def test_resume_reuses_verified_units_and_export(production, tmp_path):
    module, calls = production
    module.create(request(tmp_path), tmp_path / "job")
    first = module.run(tmp_path / "job")
    calls.clear()
    assert module.run(tmp_path / "job")["outputPath"] == first["outputPath"]
    assert calls == []


def test_changed_source_stops_resume_without_losing_outputs(production, tmp_path):
    module, _ = production
    body = request(tmp_path)
    module.create(body, tmp_path / "job")
    first = module.run(tmp_path / "job")
    (tmp_path / "song.wav").write_bytes(b"different song")
    with pytest.raises(ValueError, match="changed"):
        module.run(tmp_path / "job")
    assert Path(first["outputPath"]).read_bytes() == b"assembled movie"


def test_local_retry_is_bounded_and_resumes_after_failure(production, tmp_path, monkeypatch):
    module, calls = production
    module.create(request(tmp_path), tmp_path / "job")
    original = module.bridge.render
    failures = []

    def fail(*args, **kwargs):
        failures.append(1)
        raise RuntimeError("temporary renderer failure")

    monkeypatch.setattr(module.bridge, "render", fail)
    with pytest.raises(RuntimeError):
        module.run(tmp_path / "job")
    assert len(failures) == 2
    assert module.status(tmp_path / "job")["status"] == "failed"
    monkeypatch.setattr(module.bridge, "render", original)
    assert module.run(tmp_path / "job")["status"] == "completed"


def test_cancel_not_retried_and_completed_unit_retained(production, tmp_path, monkeypatch):
    module, calls = production
    module.create(request(tmp_path), tmp_path / "job")
    original = module.bridge.render
    count = []

    def cancel(*args, **kwargs):
        count.append(1)
        if len(count) == 2:
            raise KeyboardInterrupt()
        return original(*args, **kwargs)

    monkeypatch.setattr(module.bridge, "render", cancel)
    with pytest.raises(KeyboardInterrupt):
        module.run(tmp_path / "job")
    state = module.status(tmp_path / "job")
    assert state["status"] == "cancelled"
    assert state["units"][0]["status"] == "completed"
    assert len(count) == 2
    monkeypatch.setattr(module.bridge, "render", original)
    calls.clear()
    module.run(tmp_path / "job")
    assert len([1 for k, _ in calls if k == "render"]) == 1


def test_corrupt_completed_artifact_never_silently_regenerates_dependents(production, tmp_path):
    module, calls = production
    module.create(request(tmp_path), tmp_path / "job")
    result = module.run(tmp_path / "job")
    Path(result["resolvedProject"]["clips"][0]["sourcePath"]).write_bytes(b"corrupt")
    calls.clear()
    with pytest.raises(ValueError, match="changed|missing"):
        module.run(tmp_path / "job")
    assert calls == []


def test_remote_requires_explicit_authorization_before_any_work(production, tmp_path):
    module, calls = production
    body = request(tmp_path)
    body["project"]["clips"][1]["engine"] = "drawThings"
    with pytest.raises(ValueError, match="Draw Things|remote"):
        module.create(body, tmp_path / "job")
    assert calls == []


def test_scene_members_grouped_once(production, tmp_path, monkeypatch):
    module, _ = production
    body = request(tmp_path)
    monkeypatch.setattr(module, "scene_members", lambda r: r["project"]["clips"])
    summary = module.create(body, tmp_path / "job")
    assert len(summary["units"]) == 1
    assert summary["units"][0]["clipIDs"] == ["0", "1"]


def test_job_manifest_edit_rejected(production, tmp_path):
    module, _ = production
    module.create(request(tmp_path), tmp_path / "job")
    path = tmp_path / "job" / "request.json"
    doc = json.loads(path.read_text())
    doc["request"]["maxRetries"] = 3
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="manifest"):
        module.run(tmp_path / "job")


def test_interrupted_error_is_not_retried(production, tmp_path, monkeypatch):
    module, _ = production
    module.create(request(tmp_path), tmp_path / "job")
    calls = []

    def cancel(*args, **kwargs):
        calls.append(1)
        raise InterruptedError("Cancelled")

    monkeypatch.setattr(module.bridge, "render", cancel)
    with pytest.raises(InterruptedError):
        module.run(tmp_path / "job")
    assert len(calls) == 1


def test_scene_retiming_rejected_before_render(production, tmp_path, monkeypatch):
    module, calls = production
    body = request(tmp_path)
    monkeypatch.setattr(module, "scene_members", lambda r: r["project"]["clips"])
    original = module.bridge.prepare

    def prepare(*args):
        result = original(*args)
        result["report"]["scene"] = dict(
            frame_rate=24,
            members=[
                dict(clip_id="0", source_in=0.0, duration=1.8),
                dict(clip_id="1", source_in=1.8, duration=2.2),
            ],
        )
        return result

    monkeypatch.setattr(module.bridge, "prepare", prepare)
    module.create(body, tmp_path / "job")
    with pytest.raises(ValueError, match="timing"):
        module.run(tmp_path / "job")
    assert not any(kind == "render" for kind, _ in calls)


def test_previous_accepted_source_is_retained_as_version(production, tmp_path):
    module, _ = production
    body = request(tmp_path)
    old = tmp_path / "old.mp4"
    old.write_bytes(b"old accepted take")
    body["project"]["clips"][0]["sourcePath"] = str(old)
    module.create(body, tmp_path / "job")
    result = module.run(tmp_path / "job")
    versions = result["resolvedProject"]["clips"][0]["versions"]
    assert versions[0]["path"] == str(old)
    assert len(versions) == 2


def test_resume_after_final_rename_before_state_publish(production, tmp_path, monkeypatch):
    module, _ = production
    module.create(request(tmp_path), tmp_path / "job")
    original = Path.rename

    def interrupted(path, target):
        value = original(path, target)
        if Path(target) == tmp_path / "job" / "movie.mp4":
            raise KeyboardInterrupt()
        return value

    monkeypatch.setattr(Path, "rename", interrupted)
    with pytest.raises(KeyboardInterrupt):
        module.run(tmp_path / "job")
    monkeypatch.setattr(Path, "rename", original)
    assert module.run(tmp_path / "job")["status"] == "completed"


def test_actual_non_grid_scene_plan_cannot_change_reviewed_timing(production, tmp_path):
    from ltx25_mlx.chain_plan import plan_ltx25_scene

    module, _ = production
    body = request(tmp_path)
    for clip in body["project"]["clips"]:
        clip["duration"] = 3.1
    plan = plan_ltx25_scene([3.1, 3.1], frame_rate=24)
    # A native scene's delivered frame counts include special first-window accounting.
    from wee_todd_mlx.studio_scene import _plan_report

    segments = [{"clip_id": str(i)} for i in range(2)]
    scene = _plan_report(segments, plan)
    with pytest.raises(ValueError, match="timing"):
        module._validate_prepared_timing(
            {"report": {"scene": scene}}, {"clipIDs": ["0", "1"]}, body["project"]
        )


@pytest.mark.parametrize("timing_probe", [False, True])
def test_real_assembly_preserves_stereo_music_and_frame_count(tmp_path, timing_probe):
    import numpy as np
    from test_studio_bridge import fixture_project

    module = importlib.import_module("studio_production")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("ffmpeg and ffprobe required for real assembly")
    source, song = tmp_path / "source.mp4", tmp_path / "song.wav"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=blue:s=192x128:r=24:d=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "aevalsrc=0.2*sin(2*PI*440*t)|0.2*sin(2*PI*880*t):s=48000:d=2",
            "-c:a",
            "pcm_f32le",
            str(song),
        ],
        check=True,
    )
    if timing_probe:
        import wave

        # Nonperiodic audio makes a sub-frame export delay unambiguous.
        reference = np.random.default_rng(48).normal(0, 0.06, (96000, 2))
        reference = np.convolve(reference[:, 0], np.ones(5) / 5, "same")[:, None] * np.array(
            [[1, -0.7]]
        )
        reference = (reference * 32767).astype("<i2")
        with wave.open(str(song), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(48000)
            handle.writeframes(reference.tobytes())
    project = fixture_project([source, source])
    project["id"] = "project"
    project["titles"] = []
    for clip in project["clips"]:
        clip["transition"] = "cut"
        clip["volume"] = 0
    project["audioTracks"] = [{"id": "music", "muted": False, "solo": False}]
    project["audio"] = [
        dict(
            id="song",
            trackID="music",
            path=str(song),
            start=0.0,
            sourceIn=0.0,
            duration=2.0,
            volume=1.0,
            fade=0.0,
        )
    ]
    runtime = dict(ffmpegPath=ffmpeg, ffprobePath=ffprobe)
    module.create(dict(project=project, runtime=runtime, generateIDs=[]), tmp_path / "job")
    result = module.run(tmp_path / "job")
    streams = json.loads(
        subprocess.check_output(
            [ffprobe, "-v", "error", "-show_streams", "-of", "json", result["outputPath"]]
        )
    )["streams"]
    assert next(s for s in streams if s["codec_type"] == "video")["nb_frames"] == "48"
    assert next(s for s in streams if s["codec_type"] == "audio")["channels"] == 2
    pcm = np.frombuffer(
        subprocess.check_output(
            [
                ffmpeg,
                "-v",
                "error",
                "-i",
                result["outputPath"],
                "-map",
                "0:a:0",
                "-f",
                "f32le",
                "-ac",
                "2",
                "-",
            ]
        ),
        dtype=np.float32,
    ).reshape(-1, 2)
    if timing_probe:
        from scipy.signal import correlate, correlation_lags

        for channel in range(2):
            expected = reference[:, channel].astype(float)
            actual = pcm[: len(expected), channel]
            lag = correlation_lags(len(actual), len(expected))[
                np.argmax(correlate(actual, expected, method="fft"))
            ]
            assert abs(lag) <= 1, f"Final soundtrack shifted by {lag} samples"
        return
    for channel, frequency in [(0, 440), (1, 880)]:
        section = pcm[4800:43200, channel]
        peak = np.argmax(abs(np.fft.rfft(section))) * 48000 / len(section)
        assert abs(peak - frequency) < 2


def test_verify_rejects_modified_resolved_project(production, tmp_path):
    module, _ = production
    module.create(request(tmp_path), tmp_path / "job")
    module.run(tmp_path / "job")
    path = tmp_path / "job" / "resolved-project.json"
    result = json.loads(path.read_text())
    result["clips"][0]["sourcePath"] = "/changed.mp4"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="changed"):
        module.dispatch("production-verify", {"jobDirectory": str(tmp_path / "job")})


def test_cloud_rejected_during_creation_before_native_work(production, tmp_path):
    module, calls = production
    body = request(tmp_path)
    body["allowRemote"] = True
    body["project"]["clips"][1].update(engine="drawThings", drawThings={"profileID": "cloud"})
    body["drawThingsConnections"] = [{"id": "cloud", "selfHostedConfirmed": False}]
    with pytest.raises(ValueError, match="Cloud API"):
        module.create(body, tmp_path / "job")
    assert calls == []


def test_invalid_soundtrack_fails_before_generating(production, tmp_path):
    module, calls = production
    body = request(tmp_path)
    body["project"]["audio"][0]["duration"] = 8.0
    with pytest.raises(ValueError, match="audio|Audio"):
        module.create(body, tmp_path / "job")
    assert calls == []
