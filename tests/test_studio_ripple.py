import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import studio_ripple as ripple


@pytest.fixture
def ripple_request(tmp_path):
    source = tmp_path / "source.mp4"
    source.touch()
    edited = tmp_path / "edited.png"
    Image.new("RGB", (64, 64), "red").save(edited)
    return {
        "runtime": {},
        "ripple": dict(
            source_path=str(source),
            source_start=0.25,
            duration=1.0,
            frame_rate=24.0,
            width=64,
            height=64,
            seed=7,
            prompt="A red subject.",
            lora_strength=1.35,
            audio_policy="preserve",
            references=[dict(frame=0, path=str(edited), strength=1.0)],
        ),
    }


def test_validates_one_and_nine_unique_exact_frames(ripple_request):
    assert ripple.validate_request(ripple_request)["frames"] == 24
    ripple_request["ripple"]["references"] = [
        dict(ripple_request["ripple"]["references"][0], frame=i * 2) for i in range(9)
    ]
    assert len(ripple.validate_request(ripple_request)["references"]) == 9


@pytest.mark.parametrize("frames", [list(range(10)), [0, 0], [24], [-1], [1.5], [True]])
def test_rejects_invalid_reference_frames(ripple_request, frames):
    ripple_request["ripple"]["references"] = [
        dict(ripple_request["ripple"]["references"][0], frame=i) for i in frames
    ]
    with pytest.raises(ValueError):
        ripple.validate_request(ripple_request)


@pytest.mark.parametrize(
    "key,value",
    [
        ("width", 65),
        ("duration", float("nan")),
        ("source_start", -1),
        ("frame_rate", 0),
        ("seed", 2.2),
        ("audio_policy", "generate"),
    ],
)
def test_rejects_invalid_settings(ripple_request, key, value):
    ripple_request["ripple"][key] = value
    with pytest.raises(ValueError):
        ripple.validate_request(ripple_request)


def test_inspection_without_weights_and_out_of_range_interval(ripple_request, monkeypatch):
    media = dict(
        kind="video",
        path=ripple_request["ripple"]["source_path"],
        duration=2,
        width=80,
        height=80,
        fps=24,
        hasAudio=False,
    )
    monkeypatch.setattr(ripple, "probe_media", lambda *_args, **_kwargs: media)
    monkeypatch.setattr(ripple, "first_source_timestamp", lambda *_: 0.25)
    report = ripple.dispatch("ripple-inspect", ripple_request)
    assert report["frames"] == 24 and report["model_frames"] == 25
    assert report["has_audio"] is False and report["source_frame_rate"] == 24
    ripple_request["ripple"]["source_start"] = 1.5
    with pytest.raises(ValueError, match="interval"):
        ripple.dispatch("ripple-inspect", ripple_request)


def test_author_baseline_prepends_edit_and_preserves_source_order(ripple_request):
    values = ripple.validate_request(ripple_request)
    source = np.zeros((24, 64, 64, 3), dtype=np.uint8)
    source[:, :, :, 1] = np.arange(24)[:, None, None]
    guide, anchors, mode = ripple.prepare_conditioning(values, source)
    assert mode == "author_first_frame"
    assert anchors == []
    assert guide.shape == (25, 64, 64, 3)
    np.testing.assert_array_equal(guide[0, 0, 0], [1, 0, 0])
    np.testing.assert_allclose(guide[1:, 0, 0, 1], np.arange(24) / 255)


def test_multiple_guides_keep_exact_independent_frame_indices(ripple_request):
    ripple_request["ripple"]["references"] = [
        dict(ripple_request["ripple"]["references"][0], frame=i) for i in (0, 7, 12)
    ]
    values = ripple.validate_request(ripple_request)
    guide, anchors, mode = ripple.prepare_conditioning(values, np.zeros((24, 64, 64, 3), np.uint8))
    assert mode == "studio_timed_anchors"
    assert [x["frame_index"] for x in anchors] == [7, 12]
    assert guide.shape[0] == 25


def test_cancelled_before_media_or_model_work(ripple_request):
    with pytest.raises(InterruptedError):
        ripple.dispatch("ripple-generate", ripple_request, cancelled=lambda: True)


def test_recipe_overrides_are_explicit_and_other_adapters_rejected(ripple_request, tmp_path):
    values = ripple.validate_request(ripple_request)
    profile = {
        "engine": "ltx25",
        "format": "weetodd-headless-v2",
        "config": {"pipeline_mode": "distilled"},
        "components": {"transformer_path": "base"},
    }
    recipe = ripple.build_recipe(profile, values, "/adapter.safetensors")
    assert recipe["config"]["stage1_sampler"] == "euler"
    assert recipe["config"]["stage1_eta"] == 0
    assert recipe["config"]["ic_lora_single_stage"] is True
    assert recipe["config"]["stage1_steps"] == 8 and recipe["config"]["stage2_steps"] == 0
    assert recipe["config"]["duration_seconds"] == 1
    assert recipe["components"]["ic_loras"] == [["/adapter.safetensors", 1.35]]
    profile["components"]["loras"] = [["other", 1]]
    with pytest.raises(ValueError, match="adapter"):
        ripple.build_recipe(profile, values, "/adapter.safetensors")


def test_ripple_identity_requires_full_pinned_digest(tmp_path, monkeypatch):
    from ltx25_mlx import ripple as native

    other = tmp_path / "LTX25_Ripple_v11.safetensors"
    other.write_bytes(b"rank64 is not identity")
    assert not native.is_verified_ripple(other)
    monkeypatch.setattr(native, "RIPPLE_BYTES", other.stat().st_size)
    assert not native.is_verified_ripple(other)


@pytest.fixture
def media_source(tmp_path):
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required for actual media contract tests")
    frames = np.zeros((60, 64, 64, 3), dtype=np.uint8)
    frames[:, :, :, 0] = np.arange(60)[:, None, None] * 3
    target = tmp_path / "source.mkv"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "64x64",
            "-r",
            "30",
            "-i",
            "pipe:0",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s16le",
            "-shortest",
            str(target),
        ],
        input=frames.tobytes(),
        check=True,
    )
    return target


def test_exact_extracted_frame_matches_native_source_with_trim_and_fps(
    ripple_request, media_source, tmp_path
):
    ripple_request["ripple"].update(
        source_path=str(media_source),
        source_start=0.3,
        duration=1.013,
        frame_rate=30,
        references=[],
    )
    values = ripple.validate_request(ripple_request, require_references=False)
    assert values["frames"] == 31 and values["model_frames"] == 33
    report = ripple.dispatch("ripple-inspect", ripple_request)
    assert report["has_audio"] is True and report["source_frame_rate"] == 30
    decoded = ripple.decode_source(values, {})
    for frame in (0, 7, 15):
        ripple_request["ripple"]["frame"] = frame
        result = ripple.dispatch("ripple-frame", ripple_request, tmp_path / "extracted")
        with Image.open(result["image_path"]) as image:
            np.testing.assert_array_equal(np.asarray(image), decoded[frame])
    # The first selected source frame is source frame 9 (0.3 s at 30 fps).
    assert decoded[0, 0, 0, 0] == 27


@pytest.mark.parametrize(
    "policy,source_audio,expected",
    [("preserve", True, True), ("silent", True, False), ("preserve", False, False)],
)
def test_publication_source_audio_or_silence_never_generated(
    ripple_request, media_source, tmp_path, policy, source_audio, expected
):
    from studio_bridge import executable

    ripple_request["ripple"].update(
        source_path=str(media_source),
        source_start=0.3,
        duration=1.013,
        frame_rate=30,
        audio_policy=policy,
    )
    values = ripple.validate_request(ripple_request)
    native = tmp_path / "native.mp4"
    # Generated audio has a different frequency; publication must replace it.
    ripple._run(
        [
            executable("ffmpeg", {}),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=blue:s=64x64:r=30:d=1.5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:sample_rate=48000:duration=1.5",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(native),
        ]
    )
    target = tmp_path / f"published-{policy}-{source_audio}.mp4"
    result = ripple.publish_video(native, values, dict(has_audio=source_audio), {}, target)
    report = ripple.probe_media(target, {})
    assert result is expected and report["hasAudio"] is expected
    assert abs(report["duration"] - 1.013) < 1 / 30 + 0.002
    if expected:
        samples = np.frombuffer(
            ripple._run(
                [
                    executable("ffmpeg", {}),
                    "-v",
                    "error",
                    "-i",
                    str(target),
                    "-map",
                    "0:a:0",
                    "-f",
                    "f32le",
                    "-ac",
                    "1",
                    "-ar",
                    "48000",
                    "pipe:1",
                ]
            ),
            dtype="<f4",
        )
        peaks = np.abs(np.fft.rfft(samples))
        frequency = np.fft.rfftfreq(len(samples), 1 / 48000)[peaks.argmax()]
        assert abs(frequency - 440) < 2


def test_reference_free_inspect_frame_allows_missing_prompt(ripple_request):
    ripple_request["ripple"].pop("prompt")
    ripple_request["ripple"]["references"] = []
    assert ripple.validate_request(ripple_request, require_references=False)["references"] == []


def test_deterministic_distilled_euler_config_is_explicit():
    from dataclasses import replace

    from ltx25_mlx.runtime import LTX25GenerationConfig

    config = LTX25GenerationConfig(
        ic_lora_single_stage=True, stage2_steps=0, stage1_sampler="euler", stage1_eta=0
    )
    config.validate()
    assert config.real_forward_passes == 8
    with pytest.raises(ValueError, match="eta"):
        replace(config, stage1_eta=1).validate()
    with pytest.raises(ValueError, match="single-stage"):
        replace(config, ic_lora_single_stage=False, stage2_steps=3).validate()


def test_ripple_identity_classification_requires_verified_content(tmp_path, monkeypatch):
    from ltx25_mlx import ripple as native
    from ltx25_mlx import transformer

    source = tmp_path / "rank64.safetensors"
    source.touch()
    # Full target inspection is independent of identity. Stub that already-tested
    # inspection to isolate the previously metadata-free classification boundary.
    pair = dict(
        target="transformer_blocks.0.attn1.to_q",
        normalized_target="transformer_blocks.0.attn1.to_q",
        logical_shape=(4096, 4096),
        alpha_tensor=None,
    )
    contract = dict(
        metadata={},
        auxiliary_tensors={},
        pairs=[pair] * 480,
        ranks=[64],
        declared_scaling=dict(rank=None, alpha=None),
        pair_schemas=["ab"],
        target_fingerprint="test",
    )
    monkeypatch.setattr(transformer, "_inspect_ltx25_adapter_contract", lambda _: contract)
    monkeypatch.setattr(native, "is_verified_ripple", lambda _: False)
    assert transformer.inspect_ltx25_lora(source)["adapter_role"] == "transformer_lora"
    monkeypatch.setattr(native, "is_verified_ripple", lambda _: True)
    report = transformer.inspect_ltx25_lora(source)
    assert report["adapter_family"] == "ripple_edit" and report["adapter_role"] == "ic_lora"
    assert report["reference_downscale_factor"] == 1
    contract["ranks"] = [32]
    assert transformer.inspect_ltx25_lora(source)["adapter_family"] != "ripple_edit"


def test_first_frame_is_required(ripple_request):
    ripple_request["ripple"]["references"][0]["frame"] = 3
    with pytest.raises(ValueError, match="frame 0"):
        ripple.validate_request(ripple_request)


def test_cadence_mismatch_rejected_for_exact_source_frames(ripple_request, media_source, tmp_path):
    ripple_request["ripple"].update(source_path=str(media_source), frame=0, frame_rate=24)
    # Inspection still succeeds so the editor can adopt the actual source rate.
    assert ripple.dispatch("ripple-inspect", ripple_request)["source_frame_rate"] == 30
    with pytest.raises(ValueError, match="frame rate must match"):
        ripple.dispatch("ripple-frame", ripple_request, tmp_path / "frames")


@pytest.mark.parametrize(
    "error", [InterruptedError("cancelled"), RuntimeError("sampler failed"), None]
)
def test_native_lifecycle_and_replay_references(ripple_request, tmp_path, monkeypatch, error):
    import json
    from types import SimpleNamespace

    from ltx25_mlx import ripple as native
    from ltx25_mlx import runtime

    monkeypatch.setattr(ripple, "first_source_timestamp", lambda *_: 0.25)

    value = ripple.validate_request(ripple_request)
    adapter = tmp_path / "adapter.safetensors"
    adapter.touch()
    settings = dict(rippleAdapterPath=str(adapter))
    components = dict(
        transformer_path="t",
        text_encoder_path="q",
        video_vae_path="v",
        audio_vae_path="a",
        spatial_upscaler_path="",
    )
    profile = dict(format="weetodd-headless-v2", engine="ltx25", config={}, components=components)
    monkeypatch.setattr(native, "is_verified_ripple", lambda _: True)
    monkeypatch.setattr(
        ripple,
        "resolve_profile",
        lambda *_: (ripple.build_recipe(profile, value, adapter), "/profile"),
    )
    monkeypatch.setattr(
        ripple,
        "probe_media",
        lambda filename, *_args, **_kwargs: dict(
            duration=1 if Path(filename).name == "ripple.mp4" else 2, fps=24, hasAudio=False
        ),
    )
    monkeypatch.setattr(
        runtime.LTX25ComponentSpec,
        "validate",
        lambda *_args, **_kwargs: dict(video_scale_factors=[8, 32, 32]),
    )
    monkeypatch.setattr(ripple, "decode_source", lambda *_: np.zeros((24, 64, 64, 3), np.uint8))
    calls = []

    def fail(*_args, **kwargs):
        calls.append(kwargs)
        if error is not None:
            raise error
        return {}

    monkeypatch.setattr(
        runtime,
        "RUNTIME",
        SimpleNamespace(generate_to_file=fail, unload=lambda: calls.append("unloaded")),
    )
    output = tmp_path / "take"
    if error is None:
        monkeypatch.setattr(ripple, "publish_video", lambda *_: False)
        original = ripple_request["ripple"]["references"][0]["path"]
        result = ripple.generate(
            value, settings, output, progress=lambda _: None, cancelled=lambda: False
        )
        frozen = result["frozen_references"][0]
        assert frozen["frame"] == 0 and frozen["strength"] == 1
        assert frozen["path"] != original
        Image.new("RGB", (64, 64), "blue").save(original)
        with Image.open(frozen["path"]) as snapshot:
            assert snapshot.getpixel((0, 0)) == (255, 0, 0)
        assert result["source_sha256"] == ripple.content_identity(value["source_path"])
    else:
        with pytest.raises(type(error)):
            ripple.generate(
                value, settings, output, progress=lambda _: None, cancelled=lambda: False
            )
    assert calls[-1] == "unloaded"
    assert calls[0]["unload_after"] is True
    assert calls[0]["image_inputs"] == []
    assert calls[0]["video_references"][0]["images"].shape[0] == 25
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == (
        "complete"
        if error is None
        else "cancelled"
        if isinstance(error, InterruptedError)
        else "failed"
    )
    if error is not None:
        assert not (output / "ripple.mp4").exists()


def test_eight_step_ripple_euler_follows_deterministic_ode_independent_of_noise_seed():
    from dataclasses import dataclass

    import mlx.core as mx

    from ltx25_mlx.runtime import LTX25_DISTILLED_SIGMAS
    from ltx25_mlx.sampling import euler_ancestral_denoise_loop

    @dataclass(frozen=True)
    class State:
        latent: object
        clean_latent: object
        denoise_mask: object

    observations = []
    for seed in (2, 999):
        observed = []
        state = State(mx.array([[[2.0]]]), mx.zeros((1, 1, 1)), mx.ones((1, 1, 1)))

        def model(observed=observed, **kwargs):
            observed.append(float(kwargs["video_latent"].item()))
            return mx.full((1, 1, 1), 0.25), mx.full((1, 1, 1), 0.25)

        result = euler_ancestral_denoise_loop(
            model,
            state,
            state,
            mx.zeros((1, 1, 1)),
            mx.zeros((1, 1, 1)),
            sigmas=LTX25_DISTILLED_SIGMAS,
            eta=0,
            noise_seed=seed,
        )
        assert float(result.video_latent.item()) == 0.25
        observations.append(observed)
    expected = [0.25 + 1.75 * sigma for sigma in LTX25_DISTILLED_SIGMAS[:-1]]
    np.testing.assert_allclose(observations[0], expected, atol=1e-6)
    np.testing.assert_array_equal(observations[0], observations[1])


def test_source_audio_leading_offset_is_preserved(ripple_request, media_source, tmp_path):
    from studio_bridge import executable

    source = tmp_path / "delayed-audio.mkv"
    binary = executable("ffmpeg", {})
    ripple._run(
        [
            binary,
            "-v",
            "error",
            "-i",
            str(media_source),
            "-itsoffset",
            "0.4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1.6",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "pcm_s16le",
            str(source),
        ]
    )
    ripple_request["ripple"].update(
        source_path=str(source), source_start=0, duration=1, frame_rate=30
    )
    values = ripple.validate_request(ripple_request)
    target = tmp_path / "preserved-offset.mp4"
    ripple.publish_video(source, values, dict(has_audio=True), {}, target)
    samples = np.frombuffer(
        ripple._run(
            [
                binary,
                "-v",
                "error",
                "-i",
                str(target),
                "-map",
                "0:a:0",
                "-f",
                "f32le",
                "-ac",
                "1",
                "-ar",
                "48000",
                "pipe:1",
            ]
        ),
        dtype="<f4",
    )
    assert np.sqrt(np.mean(samples[:14400] ** 2)) < 0.001
    assert np.sqrt(np.mean(samples[24000:36000] ** 2)) > 0.03


def test_fractional_trim_reports_actual_first_source_frame(ripple_request, media_source, tmp_path):
    ripple_request["ripple"].update(
        source_path=str(media_source),
        source_start=0.31,
        duration=1,
        frame_rate=30,
        frame=0,
        references=[],
    )
    report = ripple.dispatch("ripple-inspect", ripple_request)
    assert report["source_preview_start"] == pytest.approx(0.333, abs=0.001)
    extracted = ripple.dispatch("ripple-frame", ripple_request, tmp_path / "frame")
    assert extracted["source_time"] == report["source_preview_start"]
    with Image.open(extracted["path"]) as image:
        assert np.asarray(image)[0, 0, 0] == 30  # exact decoded source frame 10


def test_variable_frame_timestamps_rejected_before_native_models(
    ripple_request, media_source, tmp_path, monkeypatch
):
    from studio_bridge import executable

    variable = tmp_path / "variable-cadence.mkv"
    ripple._run(
        [
            executable("ffmpeg", {}),
            "-v",
            "error",
            "-i",
            str(media_source),
            "-vf",
            "setpts='if(lt(N,30),N/(30*TB),(1+(N-30)/15)/TB)'",
            "-an",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "ffv1",
            str(variable),
        ]
    )
    media = ripple.probe_media(variable, {})
    ripple_request["ripple"].update(
        source_path=str(variable),
        source_start=0.1,
        duration=2.5,
        frame_rate=media["fps"],
        frame=0,
    )
    # The declared nominal rate cannot establish the actual decoded frame cadence.
    monkeypatch.setattr(
        ripple, "resolve_profile", lambda *_: pytest.fail("model preflight reached")
    )
    for command in ("ripple-inspect", "ripple-frame", "ripple-generate"):
        with pytest.raises(ValueError, match="constant frame rate"):
            ripple.dispatch(command, ripple_request, tmp_path / command)


@pytest.mark.parametrize("rotation,sar,expected", [(90, 1, (64, 128)), (0, 2, (256, 64))])
def test_display_dimensions_match_autorotated_square_pixel_extraction(
    ripple_request, tmp_path, rotation, sar, expected
):
    from studio_bridge import executable

    binary = executable("ffmpeg", {})
    unrotated = tmp_path / "unrotated.mp4"
    source = tmp_path / "display-oriented.mp4"
    ripple._run(
        [
            binary,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=128x64:r=24:d=1",
            "-vf",
            f"setsar={sar}",
            "-c:v",
            "libx264",
            str(unrotated),
        ]
    )
    ripple._run(
        [
            binary,
            "-v",
            "error",
            "-display_rotation:v:0",
            str(rotation),
            "-i",
            str(unrotated),
            "-c",
            "copy",
            str(source),
        ]
    )
    ripple_request["ripple"].update(
        source_path=str(source),
        source_start=0,
        duration=1,
        frame_rate=24,
        frame=0,
        width=expected[0],
        height=expected[1],
        references=[],
    )
    report = ripple.dispatch("ripple-inspect", ripple_request)
    assert (report["source"]["width"], report["source"]["height"]) == expected
    extracted = ripple.dispatch("ripple-frame", ripple_request, tmp_path / "frames")
    with Image.open(extracted["path"]) as image:
        assert image.size == expected
        pixels = np.asarray(image)
    decoded = ripple.decode_source(
        ripple.validate_request(ripple_request, require_references=False), {}
    )
    np.testing.assert_array_equal(pixels, decoded[0])


def test_replay_source_hash_mismatch_rejected_before_adapter_loading(
    ripple_request, media_source, tmp_path, monkeypatch
):
    ripple_request["ripple"].update(
        source_path=str(media_source), frame_rate=30, source_sha256="0" * 64
    )
    monkeypatch.setattr(
        ripple, "resolve_profile", lambda *_: pytest.fail("model preflight reached")
    )
    with pytest.raises(ValueError, match="differs from the saved take"):
        ripple.dispatch("ripple-generate", ripple_request, tmp_path / "out")
