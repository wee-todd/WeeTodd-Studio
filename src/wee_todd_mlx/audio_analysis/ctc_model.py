"""Independent MLX inference for the pinned English wav2vec2-base-960h model.

Only inference-time equations are implemented here. No Transformers, Torch,
external speech runtime, training machinery or persistent model cache is used.
Frame timestamps are receptive-field centers, not calibrated word boundaries.
"""

from __future__ import annotations

import gc
import hashlib
import json
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np

MODEL_ID = "facebook/wav2vec2-base-960h"
MODEL_REVISION = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
WEIGHTS_SHA256 = "8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a"
ASSET_SHA256 = {
    "model.safetensors": WEIGHTS_SHA256,
    "config.json": "d3ec255c063d9f95057b553b19c20135b259875834a4fe9deb218a6be25b4cf3",
    "vocab.json": "19727f8944fe6459fc3f240ae2c198395b740f6a029bd23e06656266b83bcf64",
}
SAMPLE_RATE = 16000
FRAME_SAMPLES = 320
RECEPTIVE_SAMPLES = 400
CHUNK_SAMPLES = 480000
OVERLAP_SAMPLES = 64000


def verify_assets(directory, check=None) -> None:
    """Verify the exact public checkpoint, including cancellation during hashing."""
    for name, expected in ASSET_SHA256.items():
        digest = hashlib.sha256()
        with (Path(directory) / name).open("rb") as stream:
            while data := stream.read(8 * 1024 * 1024):
                if check is not None:
                    check()
                digest.update(data)
        if digest.hexdigest() != expected:
            raise ValueError(f"CTC checkpoint integrity failure: {name}")


def output_frames(sample_count: int) -> int:
    return max(0, (sample_count - RECEPTIVE_SAMPLES) // FRAME_SAMPLES + 1)


def chunk_plan(sample_count: int) -> list[tuple[int, int, int, int]]:
    """Return sample slices and retained local frame slices; no gaps/duplicates.

    Each input is at most 30 seconds, adjacent windows share four seconds, and
    their ownership boundary splits that shared context. Starts stay on the
    global convolution grid. The final partial window owns all remaining frames.
    """
    count = output_frames(sample_count)
    if count == 0:
        return []
    step = CHUNK_SAMPLES - OVERLAP_SAMPLES
    starts = [0]
    while starts[-1] + CHUNK_SAMPLES < sample_count:
        starts.append(starts[-1] + step)
    boundaries = [0]
    for start in starts[1:]:
        boundaries.append((start + OVERLAP_SAMPLES // 2) // FRAME_SAMPLES)
    boundaries.append(count)
    return [
        (
            start,
            min(start + CHUNK_SAMPLES, sample_count),
            boundaries[i] - start // FRAME_SAMPLES,
            boundaries[i + 1] - start // FRAME_SAMPLES,
        )
        for i, start in enumerate(starts)
    ]


def normalize_audio(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float32)
    return (values - values.mean()) / np.sqrt(values.var() + np.float32(1e-7))


def validate_config(config: dict) -> None:
    expected = {
        "model_type": "wav2vec2",
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "intermediate_size": 3072,
        "vocab_size": 32,
        "conv_dim": [512] * 7,
        "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
        "conv_stride": [5, 2, 2, 2, 2, 2, 2],
        "conv_bias": False,
        "feat_extract_norm": "group",
        "do_stable_layer_norm": False,
        "hidden_act": "gelu",
        "feat_extract_activation": "gelu",
        "num_conv_pos_embedding_groups": 16,
        "num_conv_pos_embeddings": 128,
        "pad_token_id": 0,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Only the supported wav2vec2-base-960h configuration can be loaded")


def layer_norm(x, weight, bias, epsilon):
    import mlx.core as mx

    return mx.fast.layer_norm(x, weight, bias, epsilon)


def attention(q, k, v):
    import mlx.core as mx

    return mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5)


class _AcousticModel:
    def __init__(self, directory: Path, config: dict):
        import mlx.core as mx

        self.weights = {}
        self.position_weight = None
        self.epsilon = float(config.get("layer_norm_eps", 1e-5))
        try:
            self.weights = mx.load(str(directory / "model.safetensors"))
            prefix = "wav2vec2.encoder.pos_conv_embed.conv"
            vector = self.weights[prefix + ".weight_v"]
            magnitude = self.weights[prefix + ".weight_g"]
            # Weight normalization retains the temporal kernel axis (PyTorch dim=2).
            normalized = (
                vector * magnitude / mx.sqrt(mx.sum(vector * vector, axis=(0, 1), keepdims=True))
            )
            self.position_weight = normalized.transpose(0, 2, 1)
            mx.eval(self.weights, self.position_weight)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.weights.clear()
        self.position_weight = None

    def linear(self, x, prefix):
        return x @ self.weights[prefix + ".weight"].T + self.weights[prefix + ".bias"]

    def norm(self, x, prefix):
        return layer_norm(
            x, self.weights[prefix + ".weight"], self.weights[prefix + ".bias"], self.epsilon
        )

    def __call__(self, samples: np.ndarray, check: Callable[[], None]):
        import mlx.core as mx
        import mlx.nn as nn

        x = mx.array(normalize_audio(samples))[None, :, None]
        for i, stride in enumerate((5, 2, 2, 2, 2, 2, 2)):
            check()
            prefix = f"wav2vec2.feature_extractor.conv_layers.{i}"
            x = mx.conv1d(
                x, self.weights[prefix + ".conv.weight"].transpose(0, 2, 1), stride=stride
            )
            if i == 0:
                # One group per channel: normalize each channel over time.
                x = (x - mx.mean(x, axis=1, keepdims=True)) * mx.rsqrt(
                    mx.var(x, axis=1, keepdims=True) + 1e-5
                )
                x = (
                    x * self.weights[prefix + ".layer_norm.weight"]
                    + self.weights[prefix + ".layer_norm.bias"]
                )
            x = nn.gelu(x)
            mx.eval(x)
        x = self.linear(
            self.norm(x, "wav2vec2.feature_projection.layer_norm"),
            "wav2vec2.feature_projection.projection",
        )
        pos = mx.conv1d(x, self.position_weight, padding=64, groups=16)[:, :-1]
        pos = pos + self.weights["wav2vec2.encoder.pos_conv_embed.conv.bias"]
        x = self.norm(x + nn.gelu(pos), "wav2vec2.encoder.layer_norm")
        for i in range(12):
            check()
            prefix = f"wav2vec2.encoder.layers.{i}"
            q, k, v = [
                self.linear(x, prefix + ".attention." + name + "_proj")
                .reshape(1, x.shape[1], 12, 64)
                .transpose(0, 2, 1, 3)
                for name in ("q", "k", "v")
            ]
            attended = attention(q, k, v).transpose(0, 2, 1, 3).reshape(x.shape)
            x = self.norm(
                x + self.linear(attended, prefix + ".attention.out_proj"), prefix + ".layer_norm"
            )
            ff = self.linear(
                nn.gelu(self.linear(x, prefix + ".feed_forward.intermediate_dense")),
                prefix + ".feed_forward.output_dense",
            )
            x = self.norm(x + ff, prefix + ".final_layer_norm")
            mx.eval(x)
        logits = self.linear(x, "lm_head")[0].astype(mx.float32)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(log_probs)
        return np.array(log_probs)


def ctc_emissions(samples16k, model_directory, check=None) -> dict:
    """Score mono 16 kHz audio and always release the process-local model.

    ``check`` raises on cancellation. Long recordings have chunk-local waveform
    and temporal group normalization; overlap suppresses edge effects but cannot
    make bounded-context scores identical to full-recording attention.
    """
    samples = np.asarray(samples16k, dtype=np.float32)
    if samples.ndim != 1 or samples.size < RECEPTIVE_SAMPLES or not np.isfinite(samples).all():
        raise ValueError("CTC requires finite mono 16 kHz samples of at least 25 ms")
    check = check or (lambda: None)
    check()
    directory = Path(model_directory)
    verify_assets(directory, check)
    config = json.loads((directory / "config.json").read_text())
    validate_config(config)
    vocabulary = json.loads((directory / "vocab.json").read_text())
    if (
        len(vocabulary) != 32
        or set(vocabulary.values()) != set(range(32))
        or vocabulary.get("<pad>") != 0
    ):
        raise ValueError("Unsupported wav2vec2 vocabulary")
    import mlx.core as mx

    model = None
    parts = []
    plan = chunk_plan(samples.size)
    try:
        model = _AcousticModel(directory, config)
        for start, end, keep_start, keep_end in plan:
            check()
            scores = model(samples[start:end], check)
            parts.append(scores[keep_start:keep_end].copy())
            del scores
            mx.clear_cache()
        check()
        return {
            "log_probs": np.concatenate(parts),
            "vocabulary": vocabulary,
            "frame_seconds": FRAME_SAMPLES / SAMPLE_RATE,
            "offset_seconds": RECEPTIVE_SAMPLES / (2 * SAMPLE_RATE),
            "metadata": {
                "model": MODEL_ID,
                "revision": MODEL_REVISION,
                "weightsSha256": WEIGHTS_SHA256,
                "backend": "native-mlx",
                "language": "en",
                "chunkCount": len(plan),
                "chunkSeconds": CHUNK_SAMPLES / SAMPLE_RATE,
                "overlapSeconds": OVERLAP_SAMPLES / SAMPLE_RATE,
                "confidenceCalibrated": False,
                "residentAfterRun": False,
            },
        }
    except BaseException as error:
        # A caller may retain its exception. Drop completed inference-frame
        # locals too, otherwise that traceback keeps Metal activations alive.
        traceback.clear_frames(error.__traceback__)
        raise
    finally:
        if model is not None:
            model.close()
        model = None
        gc.collect()
        mx.clear_cache()
