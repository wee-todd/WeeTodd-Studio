from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from ltx25_mlx.components import LTX25VideoDecoder


def decoder_with_native(fake):
    wrapper = LTX25VideoDecoder.__new__(LTX25VideoDecoder)
    wrapper._decoder = fake
    wrapper._conv_acceleration = None
    wrapper.last_decode_report = {}
    return wrapper


def writer_factory(events):
    class Writer:
        def __init__(self, filename, width, height, fps, ffmpeg):
            self.frames = 0
            self.filename = Path(filename)
            events.append(("writer", width, height, fps))

        def write(self, frames):
            self.frames += frames.shape[0]

        def close(self):
            events.append(("close", self.frames))
            self.filename.write_bytes(b"movie")

        def abort(self):
            events.append(("abort", self.frames))

    return Writer


def test_scene_decode_uses_explicit_temporal_tiles_and_streams_exact_timeline(
    tmp_path, monkeypatch
):
    events = []

    def tiled(latent, config):
        assert latent.shape[2] == 91
        assert config.temporal_config.tile_size_in_frames == 80
        assert config.temporal_config.tile_overlap_in_frames == 24
        events.append("tiled")
        for count in (48, *(56 for _ in range(11)), 57):
            yield mx.zeros((1, 3, count, 32, 32))

    native = SimpleNamespace(
        tiled_decode=tiled, decode_and_stream=lambda *a, **kw: pytest.fail("unbounded decoder used")
    )
    wrapper = decoder_with_native(native)
    monkeypatch.setattr("ltx25_mlx.chaining._RawVideoEncoder", writer_factory(events))
    output = tmp_path / "scene.mp4"
    wrapper.decode_scene_and_stream(mx.zeros((1, 128, 91, 1, 1)), str(output), frame_rate=24)
    assert events.count("tiled") == 1
    assert ("close", 721) in events
    assert wrapper.last_decode_report["temporal_tiling"] is True
    assert wrapper.last_decode_report["tile_frames"] == 80
    assert wrapper.last_decode_report["overlap_frames"] == 24
    assert wrapper.last_decode_report["output_frames"] == 721


@pytest.mark.parametrize("kind", ["short_convolutional", "diffusion"])
def test_short_and_diffusion_scene_decode_preserve_existing_path(kind, tmp_path):
    native = type("MLXDiffusionVideoDecoder" if kind == "diffusion" else "VideoDecoder", (), {})()
    wrapper = decoder_with_native(native)
    calls = []
    wrapper.decode_and_stream = lambda *args, **kwargs: calls.append((args, kwargs)) or args[1]
    latent = mx.zeros((1, 128, 91 if kind == "diffusion" else 16, 1, 1))
    wrapper.decode_scene_and_stream(latent, str(tmp_path / "scene.mp4"), frame_rate=24)
    assert len(calls) == 1
    expected = "existing_diffusion" if kind == "diffusion" else "existing_short_clip"
    assert wrapper.last_decode_report["scene_decode_policy"] == expected


def test_scene_decode_rejects_missing_frames_and_aborts_output(tmp_path, monkeypatch):
    events = []
    native = SimpleNamespace(tiled_decode=lambda *args: iter([mx.zeros((1, 3, 1, 32, 32))]))
    wrapper = decoder_with_native(native)
    monkeypatch.setattr("ltx25_mlx.chaining._RawVideoEncoder", writer_factory(events))
    output = tmp_path / "bad.mp4"
    with pytest.raises(RuntimeError, match="721"):
        wrapper.decode_scene_and_stream(mx.zeros((1, 128, 91, 1, 1)), str(output), frame_rate=24)
    assert any(event[0] == "abort" for event in events if isinstance(event, tuple))
    assert not output.exists()


def test_scene_decode_checks_cancellation_between_streamed_frames(tmp_path, monkeypatch):
    events = []
    native = SimpleNamespace(tiled_decode=lambda *args: iter([mx.zeros((1, 3, 721, 32, 32))]))
    wrapper = decoder_with_native(native)
    monkeypatch.setattr("ltx25_mlx.chaining._RawVideoEncoder", writer_factory(events))
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        if calls > 4:
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError):
        wrapper.decode_scene_and_stream(
            mx.zeros((1, 128, 91, 1, 1)),
            str(tmp_path / "cancelled.mp4"),
            frame_rate=24,
            check_interrupted=cancelled,
        )
    assert any(event[0] == "abort" for event in events if isinstance(event, tuple))
    assert not (tmp_path / "cancelled.mp4").exists()
