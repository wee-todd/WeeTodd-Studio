"""Exercise scene delivery against a tiny real audiovisual movie, without model work."""

import importlib
import json
import shutil
import subprocess

import pytest
from test_studio_scene import bridge, scene_request

renderer = importlib.import_module("render_headless")


@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize("policy", ["balanced", "strict", None])
def test_scene_calls_chain_once_and_unloads_on_all_terminal_paths(
    tmp_path, monkeypatch, failure, with_image, policy
):
    from ltx25_mlx.runtime import RUNTIME

    request = scene_request(tmp_path, 2)
    if with_image:
        from PIL import Image

        image = tmp_path / "endpoint.png"
        Image.new("RGB", (32, 32)).save(image)
        request["project"]["assets"].append(
            dict(id="end", path=str(image), kind="image", name="Endpoint")
        )
        request["project"]["clips"][1]["attachments"] = [dict(id="end", assetID="end", role="last")]
    recipe, report = bridge.compose_recipe(request)
    if policy is None:
        recipe["scene"].pop("boundary_image_policy")
    else:
        recipe["scene"]["boundary_image_policy"] = policy
    target = tmp_path / "delivered.mp4"
    native = tmp_path / "native.mp4"
    native.write_bytes(b"native fixture")
    calls, unloaded, stages = [], [], []
    monkeypatch.setattr("wee_todd_mlx.progress.render_progress",
                        lambda stage, message, **kwargs: stages.append(stage))

    def generate(spec, config, prompts, output, **kwargs):
        calls.append((config, prompts, output, kwargs))
        assert kwargs["window_frame_counts"] == [121, 145]
        assert kwargs["seeds"] == [40, 41]
        assert kwargs["boundary_image_policy"] == (policy or "strict")
        assert all("Quiet air and footsteps." in prompt for prompt in prompts)
        if with_image:
            assert [image.frame_idx for image in kwargs["images"]] == [239]
            assert kwargs["images"][0].strength == 1
        else:
            assert kwargs["images"] is None
        assert kwargs["checkpoint_dir"] == tmp_path / "checkpoints"
        if failure:
            raise failure("interrupted")
        kwargs["step_callback"](1, 22)
        kwargs["step_callback"](22, 22)
        return dict(video_path=str(native), native_marker="one decode")

    def one_shot(*a, **k):
        pytest.fail("scene must never use one-shot generation")

    def publish(source, destination, *, frames, fps, ffmpeg, source_audio=False):
        assert source == native and frames == 240 and fps == 24
        destination.write_bytes(b"delivered fixture")

    monkeypatch.setattr(RUNTIME, "generate_chain_to_file", generate)
    monkeypatch.setattr(RUNTIME, "generate_to_file", one_shot)
    monkeypatch.setattr(RUNTIME, "unload", lambda: unloaded.append(True))
    monkeypatch.setattr(renderer, "publish_scene_movie", publish, raising=False)
    if failure:
        with pytest.raises(failure, match="interrupted"):
            renderer.render_ltx(recipe, target)
        assert not target.exists()
    else:
        result = renderer.render_ltx(recipe, target)
        assert result["scene"] == report["scene"]
        assert result["video"] == str(target)
        assert result["metadata"]["native_marker"] == "one decode"
        assert result["metadata"]["delivered_frames"] == 240
        assert stages.index("sampling") < stages.index("decoding") < stages.index("publishing")
    assert len(calls) == len(unloaded) == 1


def test_scene_final_publication_has_exact_frames_and_audio(tmp_path):
    ffmpeg = bridge.executable("ffmpeg", {})
    ffprobe = bridge.executable("ffprobe", {})
    if not shutil.which(ffmpeg):
        pytest.skip("ffmpeg unavailable")
    source = tmp_path / "native.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=24:duration=10.0416666667",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=10.1",
            "-frames:v",
            "241",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    target = tmp_path / "delivered.mp4"
    renderer.publish_scene_movie(source, target, frames=240, fps=24, ffmpeg=ffmpeg)
    probe = json.loads(
        subprocess.check_output(
            [ffprobe, "-v", "error", "-show_streams", "-of", "json", str(target)]
        )
    )
    video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in probe["streams"] if stream["codec_type"] == "audio")
    assert int(video["nb_frames"]) == 240
    assert float(video["duration"]) == 10
    assert float(audio["duration"]) == pytest.approx(10, abs=1 / 48000)
    subprocess.run([ffmpeg, "-v", "error", "-i", str(target), "-f", "null", "-"], check=True)
