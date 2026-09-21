import mlx.core as mx
import numpy as np

from qwen_image21_mlx.transformer import QwenImage21Transformer, block_attention, token_layout


def test_block_attention_matches_dense_mask_across_chunk_boundaries():
    rng = np.random.default_rng(4)
    q, k, v = [rng.normal(size=(1, 2, 9, 4)).astype(np.float32) for _ in range(3)]
    ids = np.array([-1, -1, 0, 0, 1, 1, -1, 2, 2])
    mask = (np.arange(9)[:, None] >= np.arange(9)[None, :]) | (
        (ids[:, None] == ids[None, :]) & (ids[:, None] >= 0)
    )
    scores = (q @ k.swapaxes(-1, -2)) / 2
    scores = np.where(mask, scores, -np.inf)
    scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    expected = (scores / scores.sum(axis=-1, keepdims=True)) @ v
    actual = block_attention(mx.array(q), mx.array(k), mx.array(v), ids, chunk_size=2)
    np.testing.assert_allclose(np.array(actual), expected, atol=1e-5)


def test_adjacent_image_blocks_remain_separate_and_positions_centered():
    image_mask = np.array([False, True, True, True, True, True, True, True, True])
    ids, positions = token_layout(image_mask, [(2, 2), (2, 2)])
    assert ids.tolist() == [-1, 0, 0, 0, 0, 1, 1, 1, 1]
    assert positions[1:5, 1:].tolist() == [[-1, -1], [-1, 0], [0, -1], [0, 0]]


def test_tiny_transformer_cached_and_uncached_agree():
    config = dict(
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        context_in_dim=8,
        mlp_ratio=2,
        axes_dims_rope=[2, 2, 4],
        eps=1e-6,
    )
    mx.random.seed(4)
    model = QwenImage21Transformer(config)
    conditioning = {
        "features": mx.random.normal((1, 3, 8)),
        "imageMask": np.array([False, True, False]),
        "shapes": [(2, 2), (2, 2)],
    }
    latents = mx.random.normal((1, 8, 4))
    cache = {}
    first = model(latents, conditioning, 0.8, cache=cache)
    uncached = model(latents, conditioning, 0.3, cache=None)
    cached = model(latents, conditioning, 0.3, cache=cache)
    assert first.shape == (1, 4, 4)
    assert len(cache) == 2
    np.testing.assert_allclose(np.array(cached), np.array(uncached), rtol=2e-5, atol=2e-5)


def test_time_embedding_matches_bias_free_checkpoint():
    from mlx.utils import tree_flatten

    from qwen_image21_mlx.transformer import TimeEmbedding

    assert not any(key.endswith("bias") for key, _ in tree_flatten(TimeEmbedding(16).parameters()))


def test_cached_target_attention_needs_no_allocated_mask(monkeypatch):
    import qwen_image21_mlx.transformer as implementation

    mx.random.seed(8)
    q = mx.random.normal((1, 2, 4, 8))
    k = mx.random.normal((1, 2, 12, 8))
    v = mx.random.normal((1, 2, 12, 8))
    expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=8**-0.5)
    original = mx.fast.scaled_dot_product_attention
    masks = []

    def observed(*args, **kwargs):
        masks.append(kwargs.get("mask"))
        return original(*args, **kwargs)

    monkeypatch.setattr(implementation.mx.fast, "scaled_dot_product_attention", observed)
    actual = block_attention(q, k, v, np.array([-1] * 4 + [0] * 4 + [1] * 4), query_offset=8)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5)
    assert all(mask is None for mask in masks)
