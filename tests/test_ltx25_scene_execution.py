from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from ltx25_mlx.chaining import LTX25LatentContinuation
from ltx25_mlx.pipeline import LTX25DistilledPipeline


def fake_pipeline(monkeypatch, *, fail_decode=False, cancel_after=None):
    pipeline = LTX25DistilledPipeline.__new__(LTX25DistilledPipeline)
    pipeline.low_memory = True
    pipeline.last_timings = {}
    events = []
    pipeline.video_decoder_block = SimpleNamespace(free=lambda: events.append("free_video"))
    pipeline.audio_decoder_block = SimpleNamespace(free=lambda: events.append("free_audio"))
    pipeline.prompt_encoder = SimpleNamespace(free=lambda: events.append("free_text"))
    pipeline.image_conditioner = SimpleNamespace(load=lambda: object(), free=lambda: None)
    monkeypatch.setattr(
        "ltx_pipelines_mlx.utils._orchestration.combined_image_conditionings",
        lambda images, **kw: [SimpleNamespace(clean_latent=mx.zeros((1, 4, 128)))],
    )
    pipeline._release_sampling = lambda: events.append("release_sampling")
    pipeline.encode_prompt_batch = lambda prompts, **_kw: ([(None, None, 128)] * len(prompts), {})

    def generate(prompt, **kwargs):
        events.append(("sample", prompt, kwargs))
        if (
            cancel_after is not None
            and sum(isinstance(item, tuple) for item in events) > cancel_after
        ):
            raise InterruptedError("cancelled")
        count = kwargs["num_frames"]
        video = mx.full((1, 128, (count - 1) // 8 + 1, 2, 2), float(kwargs["seed"]))
        audio = mx.full((1, 8, round(count / kwargs["frame_rate"] * 25), 16), float(kwargs["seed"]))
        if kwargs["return_continuation"]:
            frames, tokens = (
                kwargs["output_video_context_frames"],
                kwargs["output_audio_context_tokens"],
            )
            continuation = LTX25LatentContinuation(
                mx.ones((1, frames, 128)),
                mx.ones((1, frames * 4, 128)),
                mx.ones((1, tokens, 128)),
                frames,
                tokens,
            )
            return video, audio, continuation
        return video, audio

    def decode(_video_decoder, _audio_decoder, video, audio, output_path, **kwargs):
        events.append(("decode", video.shape, audio.shape))
        Path(output_path).write_bytes(b"part" if fail_decode else b"movie")
        if fail_decode:
            raise RuntimeError("failed decode")

    pipeline.generate_two_stage = generate
    monkeypatch.setattr("ltx25_mlx.chaining.decode_ltx25_chain", decode)
    return pipeline, events


@pytest.mark.parametrize("sampled_tail", [True, False])
def test_both_sampling_stages_use_interior_sampled_history_only(monkeypatch, sampled_tail):
    from ltx25_mlx import pipeline as module
    from ltx25_mlx.sampling import LTX25DenoiseOutput

    monkeypatch.setattr(module, "transformer_metadata", lambda _path: {})
    monkeypatch.setattr(module, "LTX25LatentNormalizer", lambda _path: SimpleNamespace(
        normalize_latent=lambda x: x, denormalize_latent=lambda x: x,
    ))
    pipe = LTX25DistilledPipeline(
        transformer_path="unused", text_encoder_path="unused",
        video_vae_path="unused", audio_vae_path="unused", low_memory=False,
    )
    pipe.dit = SimpleNamespace()
    pipe.upsampler = lambda x: mx.repeat(mx.repeat(x, 2, axis=3), 2, axis=4)
    pipe.load = lambda **_kwargs: None
    pipe._sampling_model = lambda: object()
    monkeypatch.setattr("ltx25_mlx.feed_forward.set_mpp_feed_forward_enabled", lambda *_a: None)
    monkeypatch.setattr("ltx25_mlx.sol_attention.ltx25_sol_attention_report", lambda *_a: {})
    states = []

    def sample(_model, video, audio, *_args, **_kwargs):
        states.append((video, audio))
        return LTX25DenoiseOutput(video.clean_latent, audio.clean_latent)

    monkeypatch.setattr(module, "euler_ancestral_denoise_loop", sample)
    low = mx.repeat(mx.array([1., 2., 3., 99.]).reshape(1, 4, 1), 128, axis=2)
    high = mx.repeat(low, 4, axis=1)
    history = LTX25LatentContinuation(
        low, high, mx.ones((1, 8, 128)), 4, 8,
        video_tail_is_terminal=sampled_tail,
    )
    pipe.generate_two_stage(
        "unchanged scene", height=64, width=64, num_frames=33, frame_rate=24,
        encoded_prompt=(mx.zeros((1, 2, 4)), mx.zeros((1, 2, 4)), 2),
        continuation=history, continuation_strength=0.5,
    )
    frames = 3 if sampled_tail else 4
    for (video, audio), area, source in zip(states, (1, 4), (low, high), strict=True):
        assert video.latent.shape[1] == (5 + frames) * area
        assert mx.array_equal(video.clean_latent[:, 5 * area:], source[:, :frames * area])
        assert mx.array_equal(audio.clean_latent[:, -8:], history.audio_tokens)
    assert pipe.last_timings["continuation_video_history"]["guided_latent_frames"] == frames


def run_scene(pipeline, tmp_path, **extra):
    kwargs = dict(
        prompts=["one", "two"],
        output_path=str(tmp_path / "scene.mp4"),
        height=64,
        width=64,
        total_frames=81,
        window_count=2,
        overlap_frames=25,
        frame_rate=24,
        seed=1,
        window_frame_counts=(33, 73),
        seeds=(12, 24),
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_identity="shared identity",
    )
    kwargs.update(extra)
    return pipeline.generate_chained_and_save(**kwargs)


def test_variable_windows_images_seeds_and_one_atomic_final_decode(tmp_path, monkeypatch):
    from ltx_pipelines_mlx.utils.args import ImageConditioningInput

    pipeline, events = fake_pipeline(monkeypatch)
    image_path = tmp_path / "anchor.png"
    image_path.write_bytes(b"anchor")
    run_scene(pipeline, tmp_path, images=[ImageConditioningInput(str(image_path), 79, 1)])
    samples = [event for event in events if isinstance(event, tuple) and event[0] == "sample"]
    assert [row[2]["num_frames"] for row in samples] == [33, 73]
    assert [row[2]["seed"] for row in samples] == [12, 24]
    assert samples[1][2]["images"][0].frame_idx == 71
    assert samples[1][2]["continuation"] is not None
    assert (
        len([event for event in events if isinstance(event, tuple) and event[0] == "decode"]) == 1
    )
    assert events.index("release_sampling") < next(
        i for i, event in enumerate(events) if isinstance(event, tuple) and event[0] == "decode"
    )
    assert (tmp_path / "scene.mp4").read_bytes() == b"movie"
    assert "free_video" in events and "free_audio" in events


def test_decode_failure_retains_checkpoints_and_does_not_publish_partial_movie(
    tmp_path, monkeypatch
):
    (tmp_path / "scene.mp4").write_bytes(b"old movie")
    pipeline, events = fake_pipeline(monkeypatch, fail_decode=True)
    with pytest.raises(RuntimeError, match="failed decode"):
        run_scene(pipeline, tmp_path)
    assert (tmp_path / "scene.mp4").read_bytes() == b"old movie"
    assert "free_video" in events and "free_audio" in events
    pipeline, events = fake_pipeline(monkeypatch)
    run_scene(pipeline, tmp_path)
    assert not [event for event in events if isinstance(event, tuple) and event[0] == "sample"]
    assert pipeline.last_timings["checkpoint_resumed_windows"] == 2


def test_cancellation_resumes_only_completed_prefix(tmp_path, monkeypatch):
    pipeline, events = fake_pipeline(monkeypatch, cancel_after=1)
    with pytest.raises(InterruptedError):
        run_scene(pipeline, tmp_path)
    assert "release_sampling" in events
    pipeline, events = fake_pipeline(monkeypatch)
    run_scene(pipeline, tmp_path)
    samples = [event for event in events if isinstance(event, tuple) and event[0] == "sample"]
    assert len(samples) == 1
    assert samples[0][1] == "two"
    assert pipeline.last_timings["checkpoint_resumed_windows"] == 1


def test_runtime_forwards_scene_inputs_and_fingerprints_model_content(tmp_path, monkeypatch):
    from dataclasses import replace

    from ltx25_mlx.runtime import LTX25ComponentSpec, LTX25GenerationConfig, LTX25RuntimeCache

    files = []
    for name in ("transformer", "text", "video", "audio", "upscaler"):
        filename = tmp_path / name
        filename.write_bytes(name.encode())
        files.append(str(filename))
    spec = LTX25ComponentSpec(*files)
    monkeypatch.setattr(
        LTX25ComponentSpec,
        "validate",
        lambda *a, **kw: {
            "video_scale_factors": [8, 32, 32],
            "model_version": "2.5",
            "video_decoder": "diffusion",
        },
    )
    captured = []
    pipeline = SimpleNamespace(
        generate_chained_and_save=lambda **kw: captured.append(kw) or kw["output_path"]
    )
    runtime = LTX25RuntimeCache()
    monkeypatch.setattr(runtime, "get", lambda *_: pipeline)
    config = replace(LTX25GenerationConfig(), duration_seconds=10)
    for _ in range(2):
        result = runtime.generate_chain_to_file(
            spec,
            config,
            ["first", "last"],
            tmp_path / "movie.mp4",
            window_count=2,
            overlap_frames=25,
            window_frame_counts=(121, 145),
            images=[],
            seeds=(1, 3),
            checkpoint_dir=tmp_path / "cache",
        )
        Path(spec.transformer_path).write_bytes(b"new model")
    assert captured[0]["checkpoint_identity"] != captured[1]["checkpoint_identity"]
    assert captured[0]["window_frame_counts"] == (121, 145)
    assert captured[0]["seeds"] == (1, 3)
    assert captured[0]["stage1_eta"] == 1.0
    assert captured[0]["ancestral_seed_offset"] == 10000
    assert result["chain_plan"]["total_frames"] == 241


def test_runtime_rejects_unsupported_scene_adapters_before_component_loading(monkeypatch, tmp_path):
    from dataclasses import replace

    from ltx25_mlx.runtime import LTX25ComponentSpec, LTX25GenerationConfig, LTX25RuntimeCache

    spec = LTX25ComponentSpec("missing", "missing", "missing", "missing", "missing")
    monkeypatch.setattr(
        LTX25ComponentSpec, "validate", lambda *a, **kw: pytest.fail("component loading attempted")
    )
    for forbidden in (
        replace(spec, ic_loras=(("control", 1.0),)),
        replace(spec, msr_lora_path="msr"),
    ):
        with pytest.raises(ValueError, match="IC-LoRA|MSR"):
            LTX25RuntimeCache().generate_chain_to_file(
                forbidden,
                LTX25GenerationConfig(),
                ["one", "two"],
                tmp_path / "result.mp4",
                window_count=2,
                overlap_frames=25,
            )


def test_resolved_scene_maximum_reports_grid_limit():
    from ltx25_mlx.chain_plan import plan_ltx25_scene

    with pytest.raises(ValueError, match="29.76 seconds at 25 fps"):
        plan_ltx25_scene([5] * 6, frame_rate=25)


def test_changing_second_prompt_preserves_first_window_but_not_descendants(tmp_path, monkeypatch):
    pipeline, _ = fake_pipeline(monkeypatch)
    run_scene(pipeline, tmp_path)
    pipeline, events = fake_pipeline(monkeypatch)
    run_scene(pipeline, tmp_path, prompts=["one", "changed two"])
    assert [event[1] for event in events if isinstance(event, tuple) and event[0] == "sample"] == [
        "changed two"
    ]
    assert pipeline.last_timings["checkpoint_resumed_windows"] == 1
    pipeline, events = fake_pipeline(monkeypatch)
    run_scene(pipeline, tmp_path, prompts=["changed one", "changed two"])
    assert [event[1] for event in events if isinstance(event, tuple) and event[0] == "sample"] == [
        "changed one",
        "changed two",
    ]


def test_corrupt_ancestor_forces_all_dependent_windows_to_resample(tmp_path, monkeypatch):
    import json

    pipeline, _ = fake_pipeline(monkeypatch)
    run_scene(pipeline, tmp_path)
    manifest = json.loads(next((tmp_path / "checkpoints").glob("window-00-*.json")).read_text())
    (tmp_path / "checkpoints" / manifest["payload"]).write_bytes(b"corrupt")
    pipeline, events = fake_pipeline(monkeypatch)
    run_scene(pipeline, tmp_path)
    assert (
        len([event for event in events if isinstance(event, tuple) and event[0] == "sample"]) == 2
    )
    assert pipeline.last_timings["checkpoint_resumed_windows"] == 0


def test_changed_image_content_invalidates_window_and_its_descendants(tmp_path, monkeypatch):
    from ltx_pipelines_mlx.utils.args import ImageConditioningInput

    filename = tmp_path / "anchor.png"
    filename.write_bytes(b"old anchor")
    images = [ImageConditioningInput(str(filename), 0, 1.0)]
    pipeline, _ = fake_pipeline(monkeypatch)
    run_scene(pipeline, tmp_path, images=images)
    filename.write_bytes(b"new anchor")
    pipeline, events = fake_pipeline(monkeypatch)
    run_scene(pipeline, tmp_path, images=images)
    assert (
        len([event for event in events if isinstance(event, tuple) and event[0] == "sample"]) == 2
    )


def test_scene_encodes_both_image_scales_before_sampling_and_releases_encoder(
    tmp_path, monkeypatch
):
    from ltx_pipelines_mlx.utils.args import ImageConditioningInput

    pipeline, events = fake_pipeline(monkeypatch)
    pipeline.image_conditioner = SimpleNamespace(
        load=lambda: events.append("load_encoder") or object(),
        free=lambda: events.append("free_encoder"),
    )

    def encode_images(images, **kwargs):
        events.append(("encode_images", kwargs["enc_h"], kwargs["enc_w"]))
        return [SimpleNamespace(clean_latent=mx.zeros((1, 4, 128)))]

    monkeypatch.setattr(
        "ltx_pipelines_mlx.utils._orchestration.combined_image_conditionings", encode_images
    )
    filename = tmp_path / "anchor.png"
    filename.write_bytes(b"image")
    run_scene(pipeline, tmp_path, images=[ImageConditioningInput(str(filename), 20, 1)])
    sample_indices = [
        i for i, event in enumerate(events) if isinstance(event, tuple) and event[0] == "sample"
    ]
    image_indices = [
        i
        for i, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == "encode_images"
    ]
    assert len(image_indices) == 4  # one overlapping anchor in each of two windows, two scales
    assert max(image_indices) < min(sample_indices)
    assert events.index("free_encoder") < min(sample_indices)
    for i in sample_indices:
        assert len(events[i][2]["_preencoded_image_conditionings"]) == 2
    assert pipeline.last_timings["scene_image_conditioning_bytes_bound"] > 0


def test_preencoded_native_image_conditions_reuse_tensors_without_reloading_encoder(monkeypatch):
    pipeline = LTX25DistilledPipeline.__new__(LTX25DistilledPipeline)
    pipeline.image_conditioner = SimpleNamespace(load=lambda: pytest.fail("encoder reloaded"))
    condition = SimpleNamespace(clean_latent=mx.zeros((1, 4, 128)))
    values = [condition]
    result = pipeline._image_conditionings(
        [], spatial_dims=(5, 2, 2), frame_rate=24, preencoded=values
    )
    assert result == values
    assert result is not values  # continuation appends must not mutate reusable anchors


def test_image_edit_during_text_encoding_cannot_publish_mislabeled_checkpoints(
    tmp_path, monkeypatch
):
    from ltx_pipelines_mlx.utils.args import ImageConditioningInput

    filename = tmp_path / "anchor.png"
    filename.write_bytes(b"old image")
    pipeline, events = fake_pipeline(monkeypatch)

    def encode_text(prompts, **kwargs):
        filename.write_bytes(b"new image")
        return [(None, None, 128)] * len(prompts), {}

    pipeline.encode_prompt_batch = encode_text
    with pytest.raises(ValueError, match="image.*changed"):
        run_scene(pipeline, tmp_path, images=[ImageConditioningInput(str(filename), 0, 1)])
    assert not [event for event in events if isinstance(event, tuple) and event[0] == "sample"]
    assert not list((tmp_path / "checkpoints").glob("window-*.json"))


def test_model_change_discards_warm_pipeline_before_new_checkpoint_identity(tmp_path, monkeypatch):
    from ltx25_mlx.runtime import LTX25ComponentSpec, LTX25GenerationConfig, LTX25RuntimeCache

    files = []
    for name in ("transformer", "text", "video", "audio", "upscaler"):
        filename = tmp_path / name
        filename.write_bytes(name.encode())
        files.append(str(filename))
    spec = LTX25ComponentSpec(*files)
    monkeypatch.setattr(
        LTX25ComponentSpec,
        "validate",
        lambda *a, **kw: {
            "video_scale_factors": [8, 32, 32],
            "model_version": "2.5",
            "video_decoder": "diffusion",
        },
    )
    instances = []

    class Pipeline:
        def __init__(self):
            instances.append(self)

        def generate_chained_and_save(self, **kwargs):
            return kwargs["output_path"]

    monkeypatch.setattr("ltx25_mlx.runtime._pipeline_class", lambda: Pipeline)
    runtime = LTX25RuntimeCache()
    config = LTX25GenerationConfig(duration_seconds=10, low_memory=False)
    for _ in range(2):
        runtime.generate_chain_to_file(
            spec,
            config,
            ["first", "last"],
            tmp_path / "movie.mp4",
            window_count=2,
            overlap_frames=25,
            window_frame_counts=(121, 145),
            checkpoint_dir=tmp_path / "cache",
            unload_after=False,
        )
        Path(spec.transformer_path).write_bytes(b"changed model")
    assert len(instances) == 2


def test_oversized_window_checkpoint_fails_before_model_work(tmp_path, monkeypatch):
    pipeline, events = fake_pipeline(monkeypatch)
    pipeline.encode_prompt_batch = lambda *a, **kw: pytest.fail("text model loaded")
    with pytest.raises(ValueError, match="byte limit"):
        run_scene(pipeline, tmp_path, height=14336, width=16384)
    assert not [event for event in events if isinstance(event, tuple) and event[0] == "sample"]


def test_runtime_rejects_ic_adapter_hidden_in_ordinary_lora_stack(tmp_path, monkeypatch):
    from ltx25_mlx.runtime import LTX25ComponentSpec, LTX25GenerationConfig, LTX25RuntimeCache

    spec = LTX25ComponentSpec(
        "missing", "missing", "missing", "missing", "missing", loras=(("control.safetensors", 1),)
    )
    monkeypatch.setattr(
        LTX25ComponentSpec,
        "validate",
        lambda *a, **kw: {
            "video_scale_factors": [8, 32, 32],
            "model_version": "2.5",
            "video_decoder": "diffusion",
            "components": [{"component": "transformer_lora_1", "adapter_role": "ic_lora"}],
        },
    )
    runtime = LTX25RuntimeCache()
    monkeypatch.setattr(runtime, "get", lambda *a: pytest.fail("model initialized"))
    with pytest.raises(ValueError, match="IC-LoRA|MSR"):
        runtime.generate_chain_to_file(
            spec,
            LTX25GenerationConfig(),
            ["one", "two"],
            tmp_path / "result.mp4",
            window_count=2,
            overlap_frames=25,
        )


def test_balanced_scene_reaches_sampler_and_cannot_resume_strict_checkpoints(tmp_path, monkeypatch):
    from ltx_pipelines_mlx.utils.args import ImageConditioningInput

    pipeline, events = fake_pipeline(monkeypatch)
    filename = tmp_path / 'anchor.png'
    filename.write_bytes(b'anchor')
    images = [ImageConditioningInput(str(filename), frame, 0.8) for frame in (31, 32, 79)]
    run_scene(pipeline, tmp_path, images=images, boundary_image_policy='strict')
    events.clear()
    run_scene(pipeline, tmp_path, images=images, boundary_image_policy='balanced')
    samples = [event for event in events if isinstance(event, tuple) and event[0] == 'sample']
    assert len(samples) == 2
    assert pipeline.last_timings['checkpoint_resumed_windows'] == 0
    assert [i.strength for i in samples[0][2]['images']] == [0.8, 0.8]
    assert [(i.frame_idx, i.strength) for i in samples[1][2]['images']] == [(71, 0.8)]
    assert all(i.strength == 0.8 for i in images)
    report = pipeline.last_timings['boundary_image_guidance']
    assert report['policy'] == 'balanced'
    assert [i['frame_index'] for i in report['inherited_anchors']] == [31, 32]
