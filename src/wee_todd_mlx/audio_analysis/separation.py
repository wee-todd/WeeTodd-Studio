"""Independent native MLX UMX-HQ vocal isolation; no external inference runtime.

Single vocal magnitude estimate with mixture phase (no multi-stem Wiener EM).
Six-second crossfaded windows bound weighted execution. This is an analysis aid,
not a promise of clean stems or accurate recognition of sung words.
"""

from __future__ import annotations

import collections
import gc
import hashlib
import io
import math
import pickle
import traceback
from pathlib import Path

import numpy as np

MODEL_ID = "umxhq-vocals"
CHECKPOINT_NAME = "vocals-b62c91ce.pth"
CHECKPOINT_SHA256 = "b62c91cedbc7a066f1778ead5b5cecb377aa3a46a31af1cce7c5c8769339d083"
CHECKPOINT_SIZE = 35637796
SAMPLE_RATE = 44100
FFT_SIZE = 4096
HOP_SIZE = 1024
SETTINGS_ID = "umxhq-mlx-v1-6s-1s-mixture-phase"


def _descriptor(storage, offset, shape, strides, requires_grad=False, hooks=None):
    if not isinstance(storage, tuple) or len(storage) != 3:
        raise pickle.UnpicklingError("Invalid numerical storage")
    if len(shape) != len(strides) or len(shape) > 3:
        raise pickle.UnpicklingError("Invalid tensor rank")
    if any(type(v) is not int or v < 0 for v in (offset, *shape, *strides)):
        raise pickle.UnpicklingError("Invalid tensor extent")
    extent = offset + sum((n - 1) * s for n, s in zip(shape, strides, strict=True))
    if extent >= storage[2] or math.prod(shape) > 3_000_000:
        raise pickle.UnpicklingError("Tensor exceeds storage")
    return storage, offset, shape, strides


class _CheckpointUnpickler(pickle.Unpickler):
    """Allow only numeric descriptors; never resolve arbitrary Python globals."""

    def __init__(self, stream):
        super().__init__(stream)
        self.storages = {}

    def find_class(self, module, name):
        allowed = {
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("torch", "FloatStorage"): np.dtype("<f4"),
            ("torch", "LongStorage"): np.dtype("<i8"),
            ("torch._utils", "_rebuild_tensor_v2"): _descriptor,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError("Unsupported checkpoint descriptor")
        return allowed[module, name]

    def persistent_load(self, pid):
        if (
            not isinstance(pid, tuple)
            or len(pid) != 6
            or pid[0] != "storage"
            or not isinstance(pid[1], np.dtype)
            or not str(pid[2]).isdigit()
            or pid[3] != "cpu"
            or type(pid[4]) is not int
            or not 0 < pid[4] <= 3_000_000
            or pid[5] is not None
        ):
            raise pickle.UnpicklingError("Unsupported storage descriptor")
        value = (pid[2], pid[1], pid[4])
        if pid[2] in self.storages and self.storages[pid[2]] != value:
            raise pickle.UnpicklingError("Inconsistent storage descriptor")
        self.storages[pid[2]] = value
        return value


def load_checkpoint(model_directory, check=None):
    check = check or (lambda: None)
    check()
    root = Path(model_directory).expanduser()
    checkpoint = root / MODEL_ID / CHECKPOINT_NAME
    if not checkpoint.is_file():
        raise FileNotFoundError("Vocal isolation model missing; set up UMX-HQ vocals first")
    if checkpoint.stat().st_size != CHECKPOINT_SIZE:
        raise ValueError("Vocal checkpoint size differs from pinned UMX-HQ weights")
    digest = hashlib.sha256()
    chunks = []
    with checkpoint.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            check()
            digest.update(block)
            chunks.append(block)
    if digest.hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Vocal checkpoint SHA256 differs from pinned UMX-HQ weights")
    stream = io.BytesIO(b"".join(chunks))
    del chunks
    reader = _CheckpointUnpickler(stream)
    if reader.load() != 119547037146038801333356 or reader.load() != 1001:
        raise ValueError("Unsupported checkpoint format")
    info = reader.load()
    if not info.get("little_endian"):
        raise ValueError("Unsupported checkpoint byte order")
    descriptors = reader.load()
    keys = reader.load()
    if len(keys) != len(reader.storages) or set(keys) != set(reader.storages):
        raise ValueError("Invalid checkpoint storage index")
    arrays = {}
    for key in keys:
        check()
        _, dtype, size = reader.storages[key]
        if int.from_bytes(stream.read(8), "little") != size:
            raise ValueError("Invalid checkpoint storage size")
        raw = stream.read(size * dtype.itemsize)
        if len(raw) != size * dtype.itemsize:
            raise ValueError("Truncated checkpoint storage")
        arrays[key] = np.frombuffer(raw, dtype=dtype)
    if stream.read(1):
        raise ValueError("Unexpected checkpoint trailer")
    weights = {}
    for name, (storage, offset, shape, strides) in descriptors.items():
        weights[name] = np.ndarray(
            shape,
            dtype=storage[1],
            buffer=arrays[storage[0]],
            offset=offset * storage[1].itemsize,
            strides=tuple(s * storage[1].itemsize for s in strides),
        ).copy()
    return weights


def stft(samples):
    """Centered, reflect-padded periodic Hann, unnormalized complex FFT [T,2,F]."""
    from scipy.fft import rfft

    padded = np.pad(
        samples,
        ((FFT_SIZE // 2, FFT_SIZE // 2), (0, 0)),
        mode="reflect" if len(samples) > 1 else "edge",
    )
    frames = np.lib.stride_tricks.sliding_window_view(padded, FFT_SIZE, axis=0)[::HOP_SIZE]
    window = np.hanning(FFT_SIZE + 1)[:-1].astype(np.float32)
    return rfft(frames * window, axis=-1)


def istft(spectrum, length):
    """Weighted overlap-add with exact original sample count and time origin."""
    from scipy.fft import irfft

    window = np.hanning(FFT_SIZE + 1)[:-1].astype(np.float32)
    total = (len(spectrum) - 1) * HOP_SIZE + FFT_SIZE
    audio = np.zeros((total, 2), np.float32)
    norm = np.zeros(total, np.float32)
    frames = irfft(spectrum, n=FFT_SIZE, axis=-1) * window
    for index, frame in enumerate(frames):
        start = index * HOP_SIZE
        audio[start : start + FFT_SIZE] += frame.T
        norm[start : start + FFT_SIZE] += window * window
    start = FFT_SIZE // 2
    return audio[start : start + length] / np.maximum(norm[start : start + length, None], 1e-12)


def overlap_add(samples, predict, *, chunk_size=6 * SAMPLE_RATE, overlap=SAMPLE_RATE, check=None):
    """Crossfade bounded predictions, including short clips and final partial chunks."""
    if not 0 <= overlap < chunk_size or len(samples) < 1:
        raise ValueError("Invalid vocal chunk dimensions")
    check = check or (lambda: None)
    result = np.zeros_like(samples, dtype=np.float32)
    norm = np.zeros(len(samples), np.float32)
    stride = chunk_size - overlap
    for start in range(0, len(samples), stride):
        check()
        end = min(start + chunk_size, len(samples))
        prediction = np.asarray(predict(samples[start:end]), dtype=np.float32)
        if prediction.shape != (end - start, 2) or not np.isfinite(prediction).all():
            raise ValueError("Vocal model returned invalid samples")
        weight = np.ones(end - start, np.float32)
        fade = min(overlap, len(weight))
        if start and fade:
            weight[:fade] *= np.arange(1, fade + 1, dtype=np.float32) / (fade + 1)
        if end < len(samples) and fade:
            weight[-fade:] *= np.arange(fade, 0, -1, dtype=np.float32) / (fade + 1)
        result[start:end] += prediction * weight[:, None]
        norm[start:end] += weight
        if end == len(samples):
            break
    check()
    return result / norm[:, None]


def lstm_direction(x, input_weight, recurrent_weight, bias, *, reverse=False, check=None):
    """Evaluate i/f/g/o LSTM equations, checking cancellation every 32 frames."""
    import mlx.core as mx

    if reverse:
        x = x[::-1]
    projected = x @ input_weight.T + bias
    width = recurrent_weight.shape[1]
    hidden, cell = mx.zeros((width,)), mx.zeros((width,))
    rows = []
    for t in range(len(x)):
        gates = projected[t] + hidden @ recurrent_weight.T
        incoming, forget, candidate, outgoing = mx.split(gates, 4)
        cell = mx.sigmoid(forget) * cell + mx.sigmoid(incoming) * mx.tanh(candidate)
        hidden = mx.sigmoid(outgoing) * mx.tanh(cell)
        rows.append(hidden)
        if t % 32 == 31:
            mx.eval(rows, cell)
            if check:
                check()
    result = mx.stack(rows)
    mx.eval(result)
    return result[::-1] if reverse else result


class _VocalNetwork:
    def __init__(self, weights):
        import mlx.core as mx

        self.w = {
            k: mx.array(v)
            for k, v in weights.items()
            if k.startswith(("input_", "output_", "fc", "bn", "lstm."))
            and not k.endswith("num_batches_tracked")
        }

    def __call__(self, magnitude, check=None):
        import mlx.core as mx

        w = self.w

        def norm(x, name):
            return (x - w[name + ".running_mean"]) * mx.rsqrt(w[name + ".running_var"] + 1e-5) * w[
                name + ".weight"
            ] + w[name + ".bias"]

        mixture = mx.array(magnitude)
        encoded = (mixture[..., : len(w["input_mean"])] + w["input_mean"]) * w["input_scale"]
        encoded = mx.tanh(norm(encoded.reshape(len(mixture), -1) @ w["fc1.weight"].T, "bn1"))
        recurrent = encoded
        for layer in range(3):
            if check:
                check()
            directions = []
            for suffix in ("", "_reverse"):
                key = f"l{layer}{suffix}"
                directions.append(
                    lstm_direction(
                        recurrent,
                        w["lstm.weight_ih_" + key],
                        w["lstm.weight_hh_" + key],
                        w["lstm.bias_ih_" + key] + w["lstm.bias_hh_" + key],
                        reverse=bool(suffix),
                        check=check,
                    )
                )
            recurrent = mx.concatenate(directions, axis=-1)
        hidden = mx.maximum(
            norm(mx.concatenate((encoded, recurrent), axis=-1) @ w["fc2.weight"].T, "bn2"), 0
        )
        mask = norm(hidden @ w["fc3.weight"].T, "bn3").reshape(mixture.shape)
        mask = mx.maximum(mask * w["output_scale"] + w["output_mean"], 0)
        estimate = mask * mixture
        mx.eval(estimate)
        return np.array(estimate)


def separate_vocals(samples, sample_rate, model_directory, check=None):
    """Return vocals with the original stereo shape, sample rate, and time origin."""
    audio = np.asarray(samples, dtype=np.float32)
    if (
        audio.ndim != 2
        or audio.shape[1] != 2
        or not len(audio)
        or not np.isfinite(audio).all()
        or isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, np.integer))
        or not 8000 <= sample_rate <= 192000
    ):
        raise ValueError(
            "Vocal isolation requires finite nonempty stereo audio and an integer sample rate"
        )
    check = check or (lambda: None)
    check()
    weights = load_checkpoint(model_directory, check)
    import mlx.core as mx
    from scipy.signal import resample_poly

    model = None
    try:
        model = _VocalNetwork(weights)
        check()
        divisor = math.gcd(sample_rate, SAMPLE_RATE)
        working = (
            resample_poly(audio, SAMPLE_RATE // divisor, sample_rate // divisor, axis=0)
            if sample_rate != SAMPLE_RATE
            else audio
        )

        def predict(chunk):
            spectrum = stft(chunk)
            magnitude = np.abs(spectrum)
            estimate = model(magnitude, check)
            # With one target and no EM, estimate magnitude retains the mixture phase.
            separated = spectrum * (estimate / np.maximum(magnitude, 1e-10))
            return istft(separated, len(chunk))

        vocals = overlap_add(working, predict, check=check)
        if sample_rate != SAMPLE_RATE:
            check()
            vocals = resample_poly(vocals, sample_rate // divisor, SAMPLE_RATE // divisor, axis=0)[
                : len(audio)
            ]
        check()
        return dict(
            vocals=vocals.astype(np.float32, copy=False),
            sample_rate=sample_rate,
            metadata=dict(
                model=MODEL_ID,
                checkpoint_sha256=CHECKPOINT_SHA256,
                settings=SETTINGS_ID,
                backend="mlx",
                model_sample_rate=SAMPLE_RATE,
                chunk_seconds=6,
                overlap_seconds=1,
                phase="mixture",
                wiener_iterations=0,
                time_origin_seconds=0,
                resident=False,
            ),
        )
    except BaseException as error:
        # Retained exceptions must not keep completed inference frames and their
        # model/activation references alive after staged unloading.
        traceback.clear_frames(error.__traceback__)
        raise
    finally:
        model = None
        del weights
        gc.collect()
        mx.clear_cache()
