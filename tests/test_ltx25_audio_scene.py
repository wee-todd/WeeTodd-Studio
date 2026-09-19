"""One source timeline conditions and survives overlapping scene generation."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest


def test_global_audio_windows_reuse_exact_overlap_tokens():
    from ltx25_mlx.audio_driven import scene_audio_token_windows
    from ltx25_mlx.chain_plan import plan_ltx25_windows

    plan = plan_ltx25_windows((121, 145, 145), frame_rate=24)
    tokens = mx.broadcast_to(
        mx.arange(plan.expected_audio_tokens)[None, :, None], (1, plan.expected_audio_tokens, 128)
    )
    windows = scene_audio_token_windows(tokens, plan)
    for index, overlap in enumerate(plan.join_audio_tokens):
        assert mx.array_equal(windows[index][:, -overlap:], windows[index + 1][:, :overlap])
    rebuilt = mx.concatenate(
        [windows[0], *[w[:, n:] for w, n in zip(windows[1:], plan.join_audio_tokens, strict=True)]],
        axis=1,
    )
    assert mx.array_equal(rebuilt, tokens)


def test_scene_audio_composes_one_shared_interval_and_images(tmp_path):
    from test_studio_scene import bridge, scene_request

    request = scene_request(tmp_path, 2)
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"placeholder")
    request["project"]["assets"] = [dict(id="song", path=str(audio), kind="audio", name="Song")]
    for index, clip in enumerate(request["project"]["clips"]):
        clip["attachments"] = [
            dict(
                id=f"a{index}",
                assetID="song",
                role="audioDriver",
                strength=1,
                audioSourceStart=10 + index * 5,
                audioSourceDuration=5,
            )
        ]
    from PIL import Image

    image = tmp_path / "key.png"
    Image.new("RGB", (32, 32)).save(image)
    request["project"]["assets"].append(dict(id="key", path=str(image), kind="image", name="Key"))
    request["project"]["clips"][1]["attachments"].append(
        dict(id="key", assetID="key", role="first", strength=0.7)
    )
    recipe, _ = bridge.compose_recipe(request)
    contract = recipe["conditioning"]
    assert contract["task"] == "a2v"
    assert contract["audio_policy"] == "source"
    assert len(contract["inputs"]) == 2
    driver = next(i for i in contract["inputs"] if i["role"] == "audio_driver")
    assert driver["source_start_seconds"] == 10
    assert driver["source_duration_seconds"] == 10
    assert contract["inputs"][0]["frame_index"] == 120
    request["project"]["clips"][1]["attachments"][0]["audioSourceStart"] = 15.1
    with pytest.raises(ValueError, match="contiguous"):
        bridge.compose_recipe(request)


def test_chain_encodes_source_once_and_retains_publication(tmp_path, monkeypatch):
    from test_ltx25_scene_execution import fake_pipeline, run_scene

    pipe, events = fake_pipeline(monkeypatch)
    pipe.audio_conditioner = SimpleNamespace(free=lambda: events.append("free_audio_encoder"))
    pipe.audio_patchifier = object()
    encodes = []

    def prepare(**kwargs):
        encodes.append(kwargs)
        return (
            mx.ones((1, 84, 128)),
            mx.ones((1, 2, 162000)),
            SimpleNamespace(source_sample_rate=48000, as_dict=lambda: {}),
        )

    monkeypatch.setattr("ltx25_mlx.audio_driven.prepare_audio_driven_conditioning", prepare)
    source = {"waveform": np.zeros((1, 2, 160000), dtype=np.float32), "sample_rate": 48000}
    run_scene(pipe, tmp_path, audio_reference=source)
    assert len(encodes) == 1
    samples = [e[2] for e in events if isinstance(e, tuple) and e[0] == "sample"]
    assert samples[0]["_preencoded_audio_tokens"].shape == (1, 34, 128)
    assert samples[1]["_preencoded_audio_tokens"].shape == (1, 76, 128)
    assert pipe.last_timings["audio_join_mode"] == "single_source_frozen_global_tokens"
    first_sample = next(e for e in events if isinstance(e, tuple))
    assert events.index("free_audio_encoder") < events.index(first_sample)


def test_source_publication_never_loads_audio_decoder(tmp_path, monkeypatch):
    import ltx25_mlx.chaining as chain

    calls = []
    video = SimpleNamespace(decode_and_stream=lambda *_a, **_k: None, free=lambda: None)

    class Audio:
        def __call__(self, _):
            pytest.fail("source audio must bypass the audio decoder")

        def free(self):
            pass

    monkeypatch.setattr(
        chain.subprocess,
        "run",
        lambda command, **kw: calls.append(command) or SimpleNamespace(returncode=0, stderr=b""),
    )
    result = chain.decode_ltx25_chain(
        video,
        Audio(),
        mx.zeros((1, 128, 11, 2, 2)),
        mx.zeros((1, 8, 84, 16)),
        str(tmp_path / "scene.mp4"),
        frame_rate=24,
        source_audio=(mx.zeros((1, 2, 162000)), 48000),
    )
    assert result["audio_policy"] == "source"
    assert result["weighted_stage_order"] == "video_decode_only"
    assert calls[0][calls[0].index("-c:a") + 1] == "alac"


def test_source_tokens_stay_exact_in_both_stages_with_video_history(monkeypatch):
    from ltx25_mlx import pipeline as module
    from ltx25_mlx.chaining import LTX25LatentContinuation
    from ltx25_mlx.sampling import euler_ancestral_denoise_loop

    monkeypatch.setattr(module, "transformer_metadata", lambda _p: {})
    monkeypatch.setattr(
        module,
        "LTX25LatentNormalizer",
        lambda _p: SimpleNamespace(normalize_latent=lambda x: x, denormalize_latent=lambda x: x),
    )
    pipe = module.LTX25DistilledPipeline(
        transformer_path="unused",
        text_encoder_path="unused",
        video_vae_path="unused",
        audio_vae_path="unused",
        low_memory=False,
    )
    pipe.dit = SimpleNamespace()
    pipe.upsampler = lambda x: mx.repeat(mx.repeat(x, 2, axis=3), 2, axis=4)
    pipe.load = lambda **kw: None
    routes = []

    def model(**kwargs):
        assert bool(mx.all(kwargs["audio_timesteps"] == 0))
        return mx.zeros_like(kwargs["video_latent"]), mx.full(kwargs["audio_latent"].shape, 999.0)

    pipe._sampling_model = lambda **kw: routes.append(kw) or model
    monkeypatch.setattr("ltx25_mlx.feed_forward.set_mpp_feed_forward_enabled", lambda *_: None)
    monkeypatch.setattr("ltx25_mlx.sol_attention.ltx25_sol_attention_report", lambda *_: {})
    stages = []

    def sample(*args, **kwargs):
        assert kwargs["freeze_audio"] is True
        output = euler_ancestral_denoise_loop(*args, **kwargs)
        stages.append(output.audio_latent)
        return output

    monkeypatch.setattr(module, "euler_ancestral_denoise_loop", sample)
    tokens = mx.broadcast_to(mx.arange(34, dtype=mx.float32)[None, :, None] / 34, (1, 34, 128))
    history = LTX25LatentContinuation(
        mx.ones((1, 4, 128)), mx.ones((1, 16, 128)), mx.full((1, 8, 128), -999.0), 4, 8
    )
    pipe.generate_two_stage(
        "Audio-driven motion",
        height=64,
        width=64,
        num_frames=33,
        frame_rate=24,
        encoded_prompt=(mx.zeros((1, 2, 4)), mx.zeros((1, 2, 4)), 2),
        continuation=history,
        _preencoded_audio_tokens=tokens,
    )
    assert len(stages) == 2 and routes == [{"frozen_audio": True}]
    assert all(mx.array_equal(value, tokens) for value in stages)


def test_source_interval_decodes_exact_samples_from_long_song(tmp_path):
    import wave

    from wee_todd_mlx.conditioning_media import inspect_media, read_media

    rate = 16000
    samples = np.arange(rate * 40, dtype=np.int16)
    source = tmp_path / "long.wav"
    with wave.open(str(source), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    item = dict(
        id="song",
        kind="audio",
        role="audio_driver",
        path=str(source),
        source_start_seconds=12.5,
        source_duration_seconds=2.25,
    )
    recipe = {"engine": "ltx25"}
    report = inspect_media(recipe, {"inputs": [item]})[0]
    decoded = read_media(recipe, item, report)
    assert decoded["sample_rate"] == rate
    np.testing.assert_array_equal(
        decoded["waveform"][0, 0], samples[200000:236000].astype(np.float32) / 32768
    )
    item["source_start_seconds"] = 39
    with pytest.raises(ValueError, match="exceeds"):
        inspect_media(recipe, {"inputs": [item]})


def test_source_audio_change_invalidates_resumed_windows(tmp_path, monkeypatch):
    from test_ltx25_scene_execution import fake_pipeline, run_scene

    pipe, events = fake_pipeline(monkeypatch)
    pipe.audio_conditioner = SimpleNamespace(free=lambda: None)
    pipe.audio_patchifier = object()
    monkeypatch.setattr(
        "ltx25_mlx.audio_driven.prepare_audio_driven_conditioning",
        lambda **kw: (
            mx.ones((1, 84, 128)),
            mx.ones((1, 2, 162000)),
            SimpleNamespace(source_sample_rate=48000, as_dict=lambda: {}),
        ),
    )
    source = {"waveform": np.zeros((1, 2, 160000), dtype=np.float32), "sample_rate": 48000}
    run_scene(pipe, tmp_path, audio_reference=source)
    events.clear()
    run_scene(pipe, tmp_path, audio_reference=source)
    assert pipe.last_timings["checkpoint_resumed_windows"] == 2
    assert not any(isinstance(e, tuple) and e[0] == "sample" for e in events)
    source["waveform"][0, 0, 70000] = 0.5
    run_scene(pipe, tmp_path, audio_reference=source)
    assert pipe.last_timings["checkpoint_resumed_windows"] == 0


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda clips: clips[1]["attachments"].clear(), "Every scene member"),
        (lambda clips: clips[1]["attachments"][0].update(assetID="other"), "same source"),
        (lambda clips: clips[0]["attachments"][0].update(audioSourceDuration=4.9), "eight-frame"),
        (
            lambda clips: clips[1]["attachments"][0].update(audioSourceStart=float("nan")),
            "source_start",
        ),
        (
            lambda clips: clips[1]["attachments"][0].update(audioSourceDuration=float("inf")),
            "source_duration",
        ),
    ],
)
def test_audio_scene_rejects_incompatible_intervals_before_media_or_weights(
    tmp_path, mutation, match
):
    from test_studio_scene import bridge, scene_request

    request = scene_request(tmp_path, 2)
    for identity in ("song", "other"):
        filename = tmp_path / f"{identity}.wav"
        filename.write_bytes(b"not decodable and must never reach inference")
        request["project"]["assets"].append(
            dict(id=identity, path=str(filename), kind="audio", name=identity)
        )
    for index, clip in enumerate(request["project"]["clips"]):
        clip["attachments"] = [
            dict(
                id=f"a{index}",
                assetID="song",
                role="audioDriver",
                strength=1,
                audioSourceStart=index * 5,
                audioSourceDuration=5,
            )
        ]
    mutation(request["project"]["clips"])
    with pytest.raises(ValueError, match=match):
        bridge.compose_recipe(request)


def test_final_scene_publication_preserves_original_pcm32_interval(tmp_path):
    import subprocess
    import wave

    from ltx_core_mlx.utils.ffmpeg import find_ffmpeg
    from test_studio_scene_headless import renderer

    ffmpeg = find_ffmpeg()

    native = tmp_path / "native.mp4"
    source = tmp_path / "source.wav"
    delivered = tmp_path / "delivered.mp4"
    samples = np.arange(48000 * 3, dtype=np.int32).reshape(-1, 2) * 7919
    with wave.open(str(source), "w") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(4)
        handle.setframerate(48000)
        handle.writeframes(samples.tobytes())
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=64x64:r=24:d=1",
            "-c:v",
            "libx264",
            str(native),
        ],
        check=True,
    )
    renderer.publish_scene_movie(
        native,
        delivered,
        frames=24,
        fps=24,
        ffmpeg=ffmpeg,
        source_audio=dict(path=str(source), source_start_seconds=0.25, source_duration_seconds=1),
    )
    raw = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(delivered),
            "-map",
            "0:a:0",
            "-f",
            "s32le",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout
    published = np.frombuffer(raw, dtype="<i4").reshape(-1, 2)
    np.testing.assert_array_equal(published, samples[12000:60000])
