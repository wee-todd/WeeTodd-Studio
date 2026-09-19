import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")


def test_causal_convolution_and_transpose_sample_alignment():
    from fish_speech_mlx.codec import FishCodec

    codec = FishCodec(
        {
            "c.weight": mx.array([[[1.0, 2.0, 3.0, 4.0]]]),
            "t.weight": mx.array([[[1.0, 1.0, 1.0, 1.0]]]),
        }
    )
    x = mx.array([[[1.0], [2.0], [3.0], [4.0]]])
    y = codec.conv(x, "c", stride=2)
    np.testing.assert_allclose(np.array(y).ravel(), [11, 30])
    z = codec.conv(mx.ones((1, 2, 1)), "t", stride=2, transpose=True)
    np.testing.assert_allclose(np.array(z).ravel(), [1, 1, 2, 2])


def test_reference_prompt_preserves_codebooks_and_target_text():
    from fish_speech_mlx.model import build_prompt

    class Tokenizer:
        def encode(self, s):
            return type("Tokens", (), {"ids": list(s.encode())})()

    reference = np.arange(30).reshape(10, 3)
    prompt = np.array(build_prompt(Tokenizer(), "New words", reference, "Reference words"))
    positions = np.flatnonzero(prompt[0] >= 151678)
    np.testing.assert_array_equal(prompt[1:, positions], reference)
    decoded = bytes(prompt[0, prompt[0] < 256].tolist()).decode()
    assert "Reference words" in decoded and "New words" in decoded
    assert decoded.endswith("<|im_start|>assistant\n<|voice|>")


def test_sampling_reproducible_and_restricted():
    from wee_todd_mlx.speech_ops import sample

    mx.random.seed(71)
    a = sample(mx.array([0.0, 9.0, 2.0]), temperature=0.8, top_p=0.9, top_k=1)
    assert a == 1
