from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest


def test_full_timeline_video_then_audio_release_before_next_weighted_stage(tmp_path, monkeypatch):
    import ltx25_mlx.chaining as chain

    events = []

    def decode_video(latent, output_path, **kwargs):
        assert latent.shape[2] == 11
        events.append("video")
        Path(output_path).write_bytes(b"silent")

    def decode_audio(latent):
        assert events[-1] == "free_video"
        assert latent.shape[2] == 84
        events.append("audio")
        return mx.zeros((1, 2, 162000))

    class Audio:
        __call__ = staticmethod(decode_audio)

        def free(self):
            events.append("free_audio")

    video = SimpleNamespace(
        decode_and_stream=decode_video, free=lambda: events.append("free_video")
    )

    def mux(command, **kwargs):
        assert events[-1] == "free_audio"
        events.append("mux")
        Path(command[-1]).write_bytes(b"movie")
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(chain.subprocess, "run", mux)
    report = chain.decode_ltx25_chain(
        video,
        Audio(),
        mx.zeros((1, 128, 11, 2, 2)),
        mx.zeros((1, 8, 84, 16)),
        str(tmp_path / "out.mp4"),
        frame_rate=24,
        low_memory=True,
    )
    assert events[:5] == ["video", "free_video", "audio", "free_audio", "mux"]
    assert report["output_audio_samples"] == 162000
    assert (tmp_path / "out.mp4").read_bytes() == b"movie"
    assert list(tmp_path.iterdir()) == [tmp_path / "out.mp4"]


def test_decode_failure_unloads_video_and_never_starts_audio(tmp_path):
    import ltx25_mlx.chaining as chain

    events = []

    def fail(*args, **kwargs):
        raise RuntimeError("video failed")

    video = SimpleNamespace(decode_and_stream=fail, free=lambda: events.append("free_video"))

    class Audio:
        def __call__(self, _):
            pytest.fail("audio must not load")

        def free(self):
            events.append("free_audio")

    with pytest.raises(RuntimeError, match="video failed"):
        chain.decode_ltx25_chain(
            video,
            Audio(),
            mx.zeros((1, 128, 11, 2, 2)),
            mx.zeros((1, 8, 84, 16)),
            str(tmp_path / "out.mp4"),
            frame_rate=24,
        )
    assert "free_video" in events
    assert not list(tmp_path.iterdir())
