"""Exercise the shared headless boundary with tiny latent arrays and fake weighted stages."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_h3_continuation_artifact import component_files, context, recipe

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "save,failure", [(False, None), (True, None), (True, RuntimeError), (True, KeyboardInterrupt)]
)
@pytest.mark.parametrize("task", ["t2v", "fflf"])
def test_headless_loads_context_and_saves_only_after_publication(
    tmp_path, monkeypatch, save, failure, task
):
    from wee_todd_mlx import conditioning_media
    from wee_todd_mlx import h3_continuation_artifact as artifacts
    from wee_todd_nodes import (
        conditioning,
        decoding,
        direct_publishing,
        preflight,
        runtime,
        sampling,
    )

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import profile_fasth3_server
    import render_headless

    component_files(tmp_path)
    value = recipe(tmp_path, duration=119 / 24 if save else 124 / 24, source=True, save=save)
    value.update(prompt="A quiet landscape.", ffmpeg="ffmpeg")
    value["config"]["steps"] = 3
    if task == "fflf":
        from PIL import Image

        image = tmp_path / "last.png"
        Image.new("RGB", (32, 32)).save(image)
        value["conditioning"] = {
            "version": 1,
            "task": "fflf",
            "inputs": [
                {
                    "id": "end",
                    "kind": "image",
                    "role": "keyframe",
                    "path": str(image),
                    "frame_index": "last",
                },
            ],
        }
        value["components"]["task"] = "fl2va"
    identity = artifacts.continuation_identity(value)
    prior = artifacts.save_continuation_artifact(
        context(tmp_path),
        tmp_path / "prior",
        identity=identity,
        provenance={"published_frames": 124, "generated_frames": 124, "tail_trim_frames": 0},
    )
    value["continuation"].update(
        source_context=prior["manifest"], source_manifest_sha256=prior["manifest_sha256"]
    )
    monkeypatch.setattr(conditioning_media, "inspect_media", lambda *a: [])
    monkeypatch.setattr(preflight, "preflight_components", lambda *a: None)
    monkeypatch.setattr(profile_fasth3_server, "install_profiling", lambda *a: nullcontext())
    encoded = conditioning.H3Conditioning(
        embeddings=None,
        token_tags=None,
        token_count=0,
        prompt="test",
        load_vision=task == "fflf",
        encoder_spec=None,
        task="fl2va" if task == "fflf" else "t2va",
        cache_report={},
    )
    monkeypatch.setattr(conditioning.TEXT_ENCODER_RUNTIME, "encode", lambda *a, **k: encoded)
    monkeypatch.setattr(
        decoding.VIDEO_VAE_RUNTIME,
        "encode_keyframes",
        lambda *a, **k: np.zeros((1, 24), dtype=np.float32),
    )
    released = []
    runtimes = [
        conditioning.TEXT_ENCODER_RUNTIME,
        sampling.TRANSFORMER_RUNTIME,
        decoding.VIDEO_VAE_RUNTIME,
        decoding.AUDIO_VAE_RUNTIME,
        runtime.RUNTIME,
    ]
    for index, active in enumerate(runtimes):
        monkeypatch.setattr(active, "unload", lambda i=index: released.append(i))
    generated = 141 if save else 158

    def sample(spec, conditioning, config, **kwargs):
        assert kwargs["continuation"] is not None
        assert kwargs["continuation"].video.shape == (1, 24, 7, 2, 2)
        assert round(config.duration_seconds * 24) == generated
        if task == "fflf":
            assert conditioning.keyframe_anchors == (140 if save else 145,)
        return sampling.H3Latents(
            video=np.zeros((1, 24, (42 if save else 47), 2, 2), dtype=np.float32),
            audio=np.zeros((2, 32, (235 if save else 263)), dtype=np.float32),
            num_frames=generated,
            width=32,
            height=32,
            fps=24,
            sample_rate=32000,
            transformer_evaluations=2,
            seconds_per_evaluation=0.5,
            total_seconds=1,
            transformer_spec=spec,
            generation_config=config,
        )

    monkeypatch.setattr(sampling.TRANSFORMER_RUNTIME, "sample", sample)

    def publish(target, components, latents, **kwargs):
        assert 1 in released, "transformer must release before decode"
        assert isinstance(kwargs["video_cache"], artifacts.ContinuationVideoDecoder)
        assert isinstance(kwargs["audio_cache"], artifacts.ContinuationAudioDecoder)
        assert not (tmp_path / "render.continuation" / "manifest.json").exists()
        if failure:
            raise failure("publication interrupted")
        target.write_bytes(b"published movie fixture")
        return SimpleNamespace(video_path=target, metadata=kwargs["metadata_updates"]())

    monkeypatch.setattr(direct_publishing, "publish_latents_direct", publish)
    if failure:
        with pytest.raises(failure):
            render_headless.render_h3(value, tmp_path / "render.mp4")
        assert not (tmp_path / "render.continuation").exists()
    else:
        result = render_headless.render_h3(value, tmp_path / "render.mp4")
        report = result["metadata"]["continuation"]
        assert report["generated_frames"] == generated
        assert report["published_frames"] == (119 if save else 124)
        assert report["continuation_eligible"] is save
        if save:
            loaded, manifest = artifacts.load_continuation_artifact(
                result["continuation_artifact"]["manifest"],
                result["continuation_artifact"]["manifest_sha256"],
                identity=identity,
                context_frames=22,
            )
            assert manifest["provenance"]["source_manifest_sha256"] == prior["manifest_sha256"]
            assert loaded.audio.shape == (2, 32, 37)
        else:
            assert result["continuation_artifact"] is None
    assert set(released) == set(range(5))


def test_incompatible_context_fails_before_text_encoder_load(tmp_path, monkeypatch):
    from wee_todd_mlx import conditioning_media
    from wee_todd_mlx import h3_continuation_artifact as artifacts
    from wee_todd_nodes import conditioning, preflight

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import render_headless

    component_files(tmp_path)
    value = recipe(tmp_path, source=True)
    value.update(prompt="test", ffmpeg="ffmpeg")
    identity = artifacts.continuation_identity(value)
    prior = artifacts.save_continuation_artifact(
        context(tmp_path),
        tmp_path / "prior",
        identity=identity,
        provenance={"published_frames": 124, "generated_frames": 124, "tail_trim_frames": 0},
    )
    value["continuation"].update(
        source_context=prior["manifest"], source_manifest_sha256=prior["manifest_sha256"]
    )
    value["config"]["steps"] = 4
    monkeypatch.setattr(preflight, "preflight_components", lambda *a: None)
    monkeypatch.setattr(conditioning_media, "inspect_media", lambda *a: [])
    monkeypatch.setattr(
        conditioning.TEXT_ENCODER_RUNTIME, "encode", lambda *a, **k: pytest.fail("loaded weights")
    )
    with pytest.raises(ValueError, match="identity"):
        render_headless.render_h3(value, tmp_path / "render.mp4")


def test_preflight_only_rejects_missing_context_without_rendering(tmp_path, monkeypatch):
    import sys

    from wee_todd_mlx import asset_registry, headless_preflight

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import render_headless

    component_files(tmp_path)
    value = recipe(tmp_path, source=True)
    value.update(format="weetodd-headless-v2", candidate="continuation-test")
    recipe_file = tmp_path / "recipe.json"
    import json

    recipe_file.write_text(json.dumps(value))
    monkeypatch.setattr(asset_registry, "resolve_recipe", lambda value, registry: (value, {}))
    monkeypatch.setattr(
        headless_preflight, "preflight_recipe", lambda value: {"conditioning": {"contract": {}}}
    )
    monkeypatch.setattr(
        render_headless, "render_h3", lambda *a: pytest.fail("rendered during preflight")
    )
    # Other pytest modules exercise the maintained node catalog in this same process.
    monkeypatch.setattr(render_headless, "assert_isolated", lambda: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_headless.py",
            "--recipe",
            str(recipe_file),
            "--output-directory",
            str(tmp_path / "preflight"),
            "--preflight-only",
        ],
    )
    # main's host-isolation finder is process-global; restore it after this in-process CLI test.
    previous = list(sys.meta_path)
    try:
        with pytest.raises(FileNotFoundError):
            render_headless.main()
    finally:
        sys.meta_path[:] = previous
    result = json.loads((tmp_path / "preflight" / "result.json").read_text())
    assert result["status"] == "failed"
    assert "manifest.json" in result["error"]


def test_shared_preflight_uses_native_window_and_keeps_visible_conditioning(tmp_path, monkeypatch):
    from wee_todd_mlx import conditioning_media
    from wee_todd_mlx.headless_preflight import preflight_recipe
    from wee_todd_nodes import preflight

    value = recipe(tmp_path, duration=124 / 24, source=True)
    value["components"]["task"] = "fl2va"
    image = tmp_path / "end.png"
    image.write_bytes(b"image inspected by external media boundary")
    value["conditioning"] = {
        "version": 1,
        "task": "fflf",
        "inputs": [
            {
                "id": "end",
                "kind": "image",
                "role": "keyframe",
                "path": str(image),
                "frame_index": "last",
            },
        ],
    }
    monkeypatch.setattr(conditioning_media, "inspect_media", lambda *a: [])

    def component_report(spec, request):
        return SimpleNamespace(to_dict=lambda: preflight.estimate_h3_token_budget(request))

    monkeypatch.setattr(preflight, "preflight_components", component_report)
    report = preflight_recipe(value)
    assert report["engine_report"]["pixel_frames"] == 158
    assert report["engine_report"]["video_latent_frames"] == 47
    assert report["continuation"]["token_budget"]["condition_video_rows"] == 8
    assert report["continuation"]["token_budget"]["condition_audio_rows"] == 74
    assert report["continuation"]["token_budget"]["packed_rows"] == 1167
    assert "excludes" in report["continuation"]["memory_estimate_scope"]
    assert report["conditioning"]["contract"]["inputs"][0]["frame_index"] == 123
    assert value["config"]["duration_seconds"] == 124 / 24
