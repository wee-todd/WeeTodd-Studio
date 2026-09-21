import mlx.core as mx
import numpy as np

from qwen_image21_mlx.vae import ImageVAE, average_shortcut, duplicate_shortcut


def test_temporal_first_frame_shortcuts_keep_the_padded_channel_order():
    x = mx.array([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    down = average_shortcut(x, 2, temporal=2, spatial=2)
    np.testing.assert_allclose(np.array(down), [[[[0.0, 2.5]]]])
    up = duplicate_shortcut(mx.array([[[[3.0, 7.0]]]]), 1, temporal=2)
    np.testing.assert_allclose(np.array(up), np.full((1, 2, 2, 1), 7.0))


def test_small_rgba_vae_has_four_channels_and_expected_compression():
    config = dict(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=2,
        dim_mult=[1, 1],
        num_res_blocks=1,
        temperal_downsample=[False],
        latents_mean=[0.0, 0.0],
        latents_std=[1.0, 1.0],
    )
    vae = ImageVAE(config)
    pixels = mx.zeros((1, 8, 8, 4))
    latent = vae.encode(pixels)
    assert latent.shape == (1, 4, 4, 2)
    output = vae.decode(latent)
    assert output.shape == (1, 8, 8, 4)
    assert bool(mx.all(mx.isfinite(output)))


def test_decode_clamps_to_reference_pixel_range():
    config = dict(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=2,
        dim_mult=[1, 1],
        num_res_blocks=1,
        temperal_downsample=[False],
        latents_mean=[0.0, 0.0],
        latents_std=[1.0, 1.0],
    )
    model = ImageVAE(config)
    model.decoder = lambda value, cancel: mx.array([[[[-2.0, -0.5, 0.5, 2.0]]]])
    np.testing.assert_allclose(
        np.asarray(model.decode(mx.zeros((1, 1, 1, 2)))), [[[[-1.0, -0.5, 0.5, 1.0]]]]
    )


def test_vae_restores_allocator_budget_after_completion_and_cancellation(monkeypatch):
    import pytest

    config = dict(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=2,
        dim_mult=[1, 1],
        num_res_blocks=1,
        temperal_downsample=[False],
        latents_mean=[0.0, 0.0],
        latents_std=[1.0, 1.0],
    )
    model = ImageVAE(config)
    calls = []
    limit = [12345]

    def change(value):
        previous = limit[0]
        limit[0] = value
        calls.append(value)
        return previous

    monkeypatch.setattr(mx, "set_cache_limit", change)
    model.encode(mx.zeros((1, 8, 8, 4)))
    assert calls == [0, 12345]
    calls.clear()
    model.decode(mx.zeros((1, 4, 4, 2)))
    assert calls == [0, 12345]
    calls.clear()
    with pytest.raises(InterruptedError):
        model.decode(mx.zeros((1, 4, 4, 2)), cancel=lambda: True)
    assert calls == [0, 12345]
