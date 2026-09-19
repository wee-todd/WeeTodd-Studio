"""Numerical and boundary contracts for native vocal isolation."""

import importlib.util
import io
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

MODULE = Path(__file__).parents[1] / "src/wee_todd_mlx/audio_analysis/separation.py"


class SeparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("separation_under_test", MODULE)
        cls.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.m)

    def test_centered_stft_matches_direct_dft_and_preserves_stereo_origin(self):
        rng = np.random.default_rng(1)
        for length in (1, 2, 511, 4096, 5517):
            audio = rng.normal(size=(length, 2)).astype(np.float32)
            spec = self.m.stft(audio)
            pad = np.pad(audio, ((2048, 2048), (0, 0)), mode="reflect" if length > 1 else "edge")
            window = np.hanning(4097)[:-1]
            expected = np.stack(
                [
                    np.fft.rfft(pad[i : i + 4096] * window[:, None], axis=0).T
                    for i in range(0, length + 1, 1024)
                ]
            )
            np.testing.assert_allclose(spec, expected, atol=2e-4, rtol=3e-5)
            np.testing.assert_allclose(self.m.istft(spec, length), audio, atol=1e-6)

    def test_overlap_add_identity_covers_all_samples_with_bounded_chunks(self):
        rng = np.random.default_rng(9)
        for length in (1, 15, 16, 17, 23, 49, 81):
            original = rng.normal(size=(length, 2)).astype(np.float32)
            seen = []

            def predict(x, seen=seen):
                seen.append(len(x))
                return x.copy()

            result = self.m.overlap_add(original, predict, chunk_size=16, overlap=4)
            np.testing.assert_allclose(result, original, atol=3e-7)
            self.assertLessEqual(max(seen), 16)
            self.assertFalse(np.shares_memory(result, original))

    def test_lstm_both_directions_match_numpy_gate_equations(self):
        try:
            import mlx.core as mx
        except ImportError:
            self.skipTest("MLX unavailable")
        rng = np.random.default_rng(7)
        x = rng.normal(size=(7, 3)).astype(np.float32)
        wi = rng.normal(0, 0.2, (8, 3)).astype(np.float32)
        wh = rng.normal(0, 0.2, (8, 2)).astype(np.float32)
        b = rng.normal(0, 0.2, 8).astype(np.float32)
        for reverse in (False, True):
            h, c, expected = np.zeros(2), np.zeros(2), []
            for row in x[::-1] if reverse else x:
                i, f, g, o = np.split(wi @ row + wh @ h + b, 4)
                c = 1 / (1 + np.exp(-f)) * c + 1 / (1 + np.exp(-i)) * np.tanh(g)
                h = 1 / (1 + np.exp(-o)) * np.tanh(c)
                expected.append(h.copy())
            expected = np.stack(expected[::-1] if reverse else expected)
            actual = self.m.lstm_direction(
                mx.array(x), mx.array(wi), mx.array(wh), mx.array(b), reverse=reverse
            )
            np.testing.assert_allclose(np.array(actual), expected, atol=2e-7)

    def test_complete_network_matches_independent_numpy_equations(self):
        try:
            import mlx.core  # noqa: F401
        except ImportError:
            self.skipTest("MLX unavailable")
        rng = np.random.default_rng(172)
        w = {}
        for key, count in [
            ("input_mean", 2),
            ("input_scale", 2),
            ("output_mean", 3),
            ("output_scale", 3),
        ]:
            w[key] = rng.normal(0.5, 0.2, count).astype(np.float32)
        for stage, shape in [(1, (4, 4)), (2, (4, 8)), (3, (6, 4))]:
            w[f"fc{stage}.weight"] = rng.normal(0, 0.2, shape).astype(np.float32)
            for suffix in ["weight", "bias", "running_mean", "running_var"]:
                w[f"bn{stage}.{suffix}"] = rng.uniform(0.2, 0.8, shape[0]).astype(np.float32)
        for layer in range(3):
            for suffix in ["", "_reverse"]:
                for key, shape in [
                    ("weight_ih", (8, 4)),
                    ("weight_hh", (8, 2)),
                    ("bias_ih", (8,)),
                    ("bias_hh", (8,)),
                ]:
                    w[f"lstm.{key}_l{layer}{suffix}"] = rng.normal(0, 0.3, shape).astype(np.float32)
        mag = rng.uniform(0, 3, (5, 2, 3)).astype(np.float32)

        def norm(x, stage):
            return (x - w[f"bn{stage}.running_mean"]) / np.sqrt(
                w[f"bn{stage}.running_var"] + 1e-5
            ) * w[f"bn{stage}.weight"] + w[f"bn{stage}.bias"]

        encoded = np.tanh(
            norm(
                ((mag[:, :, :2] + w["input_mean"]) * w["input_scale"]).reshape(5, 4)
                @ w["fc1.weight"].T,
                1,
            )
        )
        seq = encoded
        for layer in range(3):
            directions = []
            for suffix in ["", "_reverse"]:
                state, cell, rows = np.zeros(2), np.zeros(2), []
                for row in seq[::-1] if suffix else seq:
                    key = f"l{layer}{suffix}"
                    gates = (
                        w["lstm.weight_ih_" + key] @ row
                        + w["lstm.weight_hh_" + key] @ state
                        + w["lstm.bias_ih_" + key]
                        + w["lstm.bias_hh_" + key]
                    )
                    i, f, g, o = np.split(gates, 4)
                    cell = cell / (1 + np.exp(-f)) + np.tanh(g) / (1 + np.exp(-i))
                    state = np.tanh(cell) / (1 + np.exp(-o))
                    rows.append(state.copy())
                directions.append(np.stack(rows[::-1] if suffix else rows))
            seq = np.concatenate(directions, axis=-1)
        y = np.maximum(norm(np.concatenate([encoded, seq], axis=-1) @ w["fc2.weight"].T, 2), 0)
        mask = norm(y @ w["fc3.weight"].T, 3).reshape(mag.shape)
        expected = np.maximum(mask * w["output_scale"] + w["output_mean"], 0) * mag
        actual = self.m._VocalNetwork(w)(mag)
        np.testing.assert_allclose(actual, expected, atol=4e-7, rtol=3e-6)

    def test_resampling_preserves_output_length_stereo_and_time_origin(self):
        class IdentityMagnitude:
            def __init__(self, weights):
                pass

            def __call__(self, magnitude, check):
                return magnitude

        for rate in (16000, 44100, 48000):
            audio = np.zeros((rate + 13, 2), np.float32)
            audio[rate // 2, 0], audio[rate // 3, 1] = 0.5, 0.7
            with (
                patch.object(self.m, "load_checkpoint", return_value={}),
                patch.object(self.m, "_VocalNetwork", IdentityMagnitude),
            ):
                result = self.m.separate_vocals(audio, rate, "/unused")
            self.assertEqual(result["vocals"].shape, audio.shape)
            self.assertEqual(result["sample_rate"], rate)
            self.assertEqual(np.argmax(abs(result["vocals"][:, 0])), rate // 2)
            self.assertEqual(np.argmax(abs(result["vocals"][:, 1])), rate // 3)
            self.assertFalse(result["metadata"]["resident"])
            self.assertEqual(np.count_nonzero(audio), 2)

    def test_input_validation_and_cancellation_precede_weights(self):
        for audio, rate in [
            (np.zeros(20), 44100),
            (np.zeros((0, 2)), 44100),
            (np.full((3, 2), np.nan), 44100),
            (np.zeros((2, 2)), 0),
            (np.zeros((2, 2)), 44100.5),
        ]:
            with self.assertRaises(ValueError):
                self.m.separate_vocals(audio, rate, "/missing")

        def stop():
            raise InterruptedError("cancelled")

        with self.assertRaises(InterruptedError):
            self.m.separate_vocals(np.zeros((9, 2)), 44100, "/missing", check=stop)
        with self.assertRaises(FileNotFoundError):
            self.m.separate_vocals(np.zeros((9, 2)), 44100, "/missing")

    def test_checkpoint_requires_exact_size_and_hash_before_decoding(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / self.m.MODEL_ID
            target.mkdir()
            checkpoint = target / self.m.CHECKPOINT_NAME
            checkpoint.write_bytes(b"unsafe")
            with self.assertRaisesRegex(ValueError, "size"):
                self.m.load_checkpoint(folder)
            with checkpoint.open("wb") as stream:
                stream.truncate(self.m.CHECKPOINT_SIZE)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                self.m.load_checkpoint(folder)

    def test_retained_cancellation_and_failure_release_native_allocations(self):
        try:
            import mlx.core as mx
        except ImportError:
            self.skipTest("MLX unavailable")
        import gc

        gc.collect()
        mx.clear_cache()
        initial = mx.get_active_memory()
        retained = []
        for error_type in (InterruptedError, RuntimeError):

            class FailingModel:
                def __init__(self, weights):
                    self.tensor = mx.ones((256, 256))
                    mx.eval(self.tensor)

                def __call__(self, magnitude, check, failure=error_type):
                    raise failure("inference stopped")

            with (
                patch.object(self.m, "load_checkpoint", return_value={}),
                patch.object(self.m, "_VocalNetwork", FailingModel),
            ):
                try:
                    self.m.separate_vocals(np.zeros((1024, 2)), 44100, "/unused")
                except error_type as error:
                    retained.append(error)
                else:
                    self.fail("Expected inference exception")
            gc.collect()
            mx.clear_cache()
            self.assertLessEqual(mx.get_active_memory(), initial)
        self.assertEqual(len(retained), 2)
        self.assertTrue(all(error.__traceback__ is not None for error in retained))

    @unittest.skipUnless(os.environ.get("WEETODD_SEPARATION_MODELS"), "optional pinned weights")
    def test_pinned_checkpoint_real_silence_qualification(self):
        result = self.m.separate_vocals(
            np.zeros((11025, 2), np.float32), 44100, os.environ["WEETODD_SEPARATION_MODELS"]
        )
        np.testing.assert_array_equal(result["vocals"], 0)
        self.assertEqual(result["metadata"]["checkpoint_sha256"], self.m.CHECKPOINT_SHA256)

    def test_restricted_decoder_rejects_executable_globals(self):
        with self.assertRaises(pickle.UnpicklingError):
            self.m._CheckpointUnpickler(io.BytesIO(pickle.dumps(eval))).load()

    def test_cancel_between_chunks_does_not_mutate_source(self):
        original = np.ones((90, 2), np.float32)
        calls = []

        def check():
            calls.append(1)
            if len(calls) == 3:
                raise InterruptedError("cancelled")

        with self.assertRaises(InterruptedError):
            self.m.overlap_add(original, lambda x: x * 0.5, chunk_size=16, overlap=4, check=check)
        np.testing.assert_array_equal(original, 1)


if __name__ == "__main__":
    unittest.main()
