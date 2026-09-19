"""Independent MLX inference for the MIT-licensed Beat This! small0 weights.

Architecture and preprocessing contract: CPJKU/beat_this, revision
b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c. No external inference package is
imported. Checkpoints are immutable, hash-checked before restricted decoding.
"""

from __future__ import annotations

import collections
import gc
import hashlib
import io
import math
import pickle
import zipfile
from pathlib import Path

import numpy as np

CHECKPOINT_NAME = "small0.ckpt"
CHECKPOINT_SHA256 = "6074be2c4d490c5f6101fcc374a1ec72ae93456e23bb6019783b849f5dc7d47b"
CHECKPOINT_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/small0.ckpt"
SOURCE_REVISION = "b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c"
SAMPLE_RATE = 22050
FRAME_SECONDS = 0.02


def _tensor(storage, offset, shape, stride, requires_grad=False, hooks=None):
    """Rebuild only bounded dense numerical views, never Python/Torch objects."""
    if not isinstance(storage, np.ndarray) or len(shape) != len(stride) or len(shape) > 5:
        raise pickle.UnpicklingError("Invalid tensor descriptor")
    if any(not isinstance(v, int) or v < 0 for v in (*shape, *stride, offset)):
        raise pickle.UnpicklingError("Invalid tensor extent")
    extent = offset + sum((s - 1) * t for s, t in zip(shape, stride, strict=True) if s)
    if extent >= storage.size or math.prod(shape) > 20_000_000:
        raise pickle.UnpicklingError("Tensor exceeds bounded storage")
    return np.ndarray(
        shape,
        dtype=storage.dtype,
        buffer=storage,
        offset=offset * storage.itemsize,
        strides=tuple(v * storage.itemsize for v in stride),
    ).copy()


class _TensorUnpickler(pickle.Unpickler):
    """Allow exactly the four numerical descriptors in the pinned checkpoint."""

    def __init__(self, stream, archive, prefix):
        super().__init__(stream)
        self.archive, self.prefix = archive, prefix

    def find_class(self, module, name):
        allowed = {
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("torch", "FloatStorage"): np.dtype("<f4"),
            ("torch", "LongStorage"): np.dtype("<i8"),
            ("torch._utils", "_rebuild_tensor_v2"): _tensor,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError("Unsupported checkpoint descriptor")
        return allowed[module, name]

    def persistent_load(self, pid):
        if (
            not isinstance(pid, tuple)
            or len(pid) != 5
            or pid[0] != "storage"
            or not isinstance(pid[1], np.dtype)
            or not str(pid[2]).isdigit()
            or not isinstance(pid[4], int)
            or not 0 < pid[4] < 20_000_000
        ):
            raise pickle.UnpicklingError("Invalid storage descriptor")
        data = self.archive.read(self.prefix + "data/" + str(pid[2]))
        if len(data) != pid[4] * pid[1].itemsize:
            raise pickle.UnpicklingError("Invalid storage length")
        return np.frombuffer(data, dtype=pid[1])


def load_checkpoint(model_directory):
    """Read the verified primary checkpoint without installing PyTorch."""
    path = Path(model_directory) / CHECKPOINT_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Beat model missing: {path}; set up Beat This small0 first")
    if path.stat().st_size > 12_000_000:
        raise ValueError("Beat checkpoint exceeds expected size")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Beat checkpoint SHA256 does not match pinned small0 weights")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if len(members) > 256 or sum(m.file_size for m in members) > 12_000_000:
            raise ValueError("Beat checkpoint archive exceeds limits")
        metadata = next(n.filename for n in members if n.filename.endswith("/data.pkl"))
        prefix = metadata[: -len("data.pkl")]
        if archive.read(prefix + "byteorder") != b"little":
            raise ValueError("Unsupported checkpoint byte order")
        data = _TensorUnpickler(io.BytesIO(archive.read(metadata)), archive, prefix).load()
    expected = {
        "spect_dim": 128,
        "transformer_dim": 128,
        "n_layers": 6,
        "stem_dim": 32,
        "head_dim": 32,
        "ff_mult": 4,
        "sum_head": True,
        "partial_transformers": True,
    }
    if any(data["hyper_parameters"].get(k) != v for k, v in expected.items()):
        raise ValueError("Unsupported Beat This architecture")
    return {
        k.removeprefix("model.").replace("_orig_mod.", ""): v for k, v in data["state_dict"].items()
    }


def mel_filterbank():
    """Slaney frequency spacing, unnormalized triangles (torchaudio contract)."""

    def to_mel(freq):
        return np.where(
            freq < 1000,
            freq / (200 / 3),
            15 + np.log(np.maximum(freq, 1) / 1000) / (np.log(6.4) / 27),
        )

    points = np.linspace(to_mel(np.array(30.0)), to_mel(np.array(11000.0)), 130)
    hz = np.where(
        points < 15, points * (200 / 3), 1000 * np.exp((points - 15) * (np.log(6.4) / 27))
    )
    frequencies = np.arange(513) * SAMPLE_RATE / 1024
    left = (frequencies[:, None] - hz[:-2]) / (hz[1:-1] - hz[:-2])
    right = (hz[2:] - frequencies[:, None]) / (hz[2:] - hz[1:-1])
    return np.maximum(0, np.minimum(left, right)).astype(np.float32)


def log_mel(samples, check=None):
    """Magnitude STFT, centered reflection, periodic Hann, sqrt(N) scaling."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Beat analysis requires nonempty finite mono audio at 22050 Hz")
    from scipy.fft import rfft

    padded = np.pad(samples, (512, 512), mode="reflect" if len(samples) > 1 else "edge")
    frames = np.lib.stride_tricks.sliding_window_view(padded, 1024)[::441]
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(1024) / 1024)).astype(np.float32)
    filters = mel_filterbank()
    result = np.empty((len(frames), 128), np.float32)
    for start in range(0, len(frames), 1500):
        if check:
            check()
        magnitude = np.abs(rfft(frames[start : start + 1500] * window, axis=-1)) / 32
        result[start : start + 1500] = np.log1p(1000 * (magnitude @ filters))
    return result


def chunk_ranges(frame_count, chunk_size=1500, border=6):
    if frame_count <= 0 or chunk_size <= 2 * border:
        raise ValueError("Invalid beat chunk dimensions")
    starts = list(range(-border, frame_count - border, chunk_size - 2 * border))
    if frame_count > chunk_size - 2 * border:
        starts[-1] = frame_count - (chunk_size - border)
    return [(start, min(chunk_size, frame_count + border - start)) for start in starts]


def _rms(x, gamma):
    import mlx.core as mx

    # Match torch F.normalize's norm clamp, rather than an epsilon on mean square.
    return (
        x
        * (x.shape[-1] ** 0.5 / mx.maximum(mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True)), 1e-12))
        * gamma
    )


def _rope(x):
    import mlx.core as mx

    # Adjacent even/odd pairs, with positions reset for each attention sequence.
    size = x.shape[-1]
    angle = mx.arange(x.shape[-2], dtype=mx.float32)[:, None] * mx.exp(
        -math.log(10000) * mx.arange(0, size, 2) / size
    )
    a, b = x[..., 0::2], x[..., 1::2]
    return mx.stack(
        (a * mx.cos(angle) - b * mx.sin(angle), a * mx.sin(angle) + b * mx.cos(angle)), axis=-1
    ).reshape(x.shape)


class _BeatNetwork:
    """Functional MLX execution; no process-global weights or retained graphs."""

    def __init__(self, weights):
        import mlx.core as mx

        self.w = {
            k: mx.array(v) for k, v in weights.items() if not k.endswith("num_batches_tracked")
        }
        # PyTorch OIHW -> MLX OHWI.
        for k in self.w:
            if k.endswith("conv2d.weight"):
                self.w[k] = mx.transpose(self.w[k], (0, 2, 3, 1))

    def linear(self, x, name):
        result = x @ self.w[name + ".weight"].T
        if name + ".bias" in self.w:
            result = result + self.w[name + ".bias"]
        return result

    def bn(self, x, name):
        import mlx.core as mx

        return (x - self.w[name + ".running_mean"]) * mx.rsqrt(
            self.w[name + ".running_var"] + 1e-5
        ) * self.w[name + ".weight"] + self.w[name + ".bias"]

    def attention(self, x, name):
        import mlx.core as mx

        x = _rms(x, self.w[name + ".norm.gamma"])
        batch, length, width = x.shape
        heads = width // 32
        qkv = (
            self.linear(x, name + ".to_qkv")
            .reshape(batch, length, 3, heads, 32)
            .transpose(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = mx.fast.scaled_dot_product_attention(_rope(q), _rope(k), v, scale=32**-0.5)
        gates = mx.sigmoid(self.linear(x, name + ".to_gates")).transpose(0, 2, 1)[..., None]
        out = (out * gates).transpose(0, 2, 1, 3).reshape(batch, length, width)
        return self.linear(out, name + ".to_out.0")

    def ff(self, x, name):
        import mlx.nn as nn

        x = _rms(x, self.w[name + ".net.0.gamma"])
        return self.linear(nn.gelu(self.linear(x, name + ".net.1")), name + ".net.4")

    def __call__(self, spect, check=None):
        import mlx.core as mx
        import mlx.nn as nn

        x = self.bn(mx.array(spect)[None], "frontend.stem.bn1d")
        x = x.transpose(0, 2, 1)[..., None]
        x = nn.gelu(
            self.bn(
                mx.conv2d(x, self.w["frontend.stem.conv2d.weight"], stride=(4, 1), padding=(0, 1)),
                "frontend.stem.bn2d",
            )
        )
        for i in range(3):
            if check:
                check()
            name = f"frontend.blocks.{i}"
            b, f, t, c = x.shape
            x = x.transpose(0, 2, 1, 3).reshape(b * t, f, c)
            x = x + self.attention(x, name + ".partial.attnF")
            x = x + self.ff(x, name + ".partial.ffF")
            x = x.reshape(b, t, f, c).transpose(0, 2, 1, 3).reshape(b * f, t, c)
            x = x + self.attention(x, name + ".partial.attnT")
            x = x + self.ff(x, name + ".partial.ffT")
            x = x.reshape(b, f, t, c)
            x = nn.gelu(
                self.bn(
                    mx.conv2d(x, self.w[name + ".conv2d.weight"], stride=(2, 1), padding=(0, 1)),
                    name + ".norm",
                )
            )
            mx.eval(x)
        # Concatenation uses channel then frequency order, not NHWC flattening.
        x = x.transpose(0, 2, 3, 1).reshape(b, t, -1)
        x = self.linear(x, "frontend.linear")
        for i in range(6):
            if check:
                check()
            name = f"transformer_blocks.layers.{i}"
            x = x + self.attention(x, name + ".0")
            x = x + self.ff(x, name + ".1")
            mx.eval(x)
        x = _rms(x, self.w["transformer_blocks.norm.gamma"])
        logits = self.linear(x, "task_heads.beat_downbeat_lin")[0]
        mx.eval(logits)
        logits = np.array(logits)
        return logits[:, 0] + logits[:, 1], logits[:, 1]


def _peak_positions(logits):
    from scipy.ndimage import maximum_filter1d

    peaks = np.flatnonzero(
        (logits > 0) & (logits == maximum_filter1d(logits, size=7, mode="constant", cval=-np.inf))
    )
    # Average adjacent maxima; retain the original minimal decoder's running-mean rule.
    groups = []
    for p in peaks:
        if groups and p - groups[-1][0] <= 1:
            mean, count = groups[-1]
            groups[-1] = (mean + (p - mean) / (count + 1), count + 1)
        else:
            groups.append((float(p), 1))
    return np.array([p for p, _ in groups])


def events_from_logits(beat, downbeat):
    beat, downbeat = np.asarray(beat), np.asarray(downbeat)
    if (
        beat.ndim != 1
        or downbeat.shape != beat.shape
        or not np.isfinite(beat).all()
        or not np.isfinite(downbeat).all()
    ):
        raise ValueError("Invalid beat logits")
    b, d = _peak_positions(beat), _peak_positions(downbeat)

    def event(frame, score):
        return {
            "timeSeconds": float(frame * FRAME_SECONDS),
            "confidence": float(1 / (1 + np.exp(-np.clip(score, -80, 80)))),
        }

    beats = [event(p, np.interp(p, np.arange(len(beat)), beat)) for p in b]
    snapped = {}
    if len(b):
        for p in d:
            target = b[np.argmin(np.abs(b - p))]
            score = float(np.interp(p, np.arange(len(downbeat)), downbeat))
            snapped[target] = max(score, snapped.get(target, -np.inf))
    return beats, [event(p, score) for p, score in sorted(snapped.items())]


def beat_events(samples22050, model_directory, check=None):
    """Learn beat/downbeat events. Confidence is activation, not calibrated accuracy.

    Weighted computation is limited to 30-second windows, with original
    six-frame context trimming and keep-first overlap semantics. Cancellation
    is checked between frontend blocks, transformer layers and audio chunks.
    """
    samples = np.asarray(samples22050, dtype=np.float32)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Beat analysis requires nonempty finite mono audio at 22050 Hz")
    if check:
        check()
    weights = load_checkpoint(model_directory)
    import mlx.core as mx

    model = None
    try:
        model = _BeatNetwork(weights)
        del weights
        spect = log_mel(samples, check)
        logits = np.full((2, len(spect)), -1000, dtype=np.float32)
        covered = np.zeros(len(spect), dtype=bool)
        ranges = chunk_ranges(len(spect))
        for start, length in ranges:
            if check:
                check()
            lo, hi = max(start, 0), min(start + length, len(spect))
            chunk = np.pad(spect[lo:hi], ((lo - start, start + length - hi), (0, 0)))
            b, d = model(chunk, check)
            target = slice(start + 6, start + length - 6)
            unseen = ~covered[target]
            logits[:, target] = np.where(
                unseen[None, :], np.stack((b[6:-6], d[6:-6])), logits[:, target]
            )
            covered[target] = True
        beats, downbeats = events_from_logits(*logits)
        duration = len(samples) / SAMPLE_RATE
        return {
            "beats": [b for b in beats if b["timeSeconds"] < duration],
            "downbeats": [b for b in downbeats if b["timeSeconds"] < duration],
            "metadata": {
                "model": "beat-this-small0",
                "engine": "mlx",
                "checkpointSHA256": CHECKPOINT_SHA256,
                "sourceRevision": SOURCE_REVISION,
                "sampleRate": SAMPLE_RATE,
                "frameSeconds": FRAME_SECONDS,
                "chunkFrames": 1500,
                "chunks": len(ranges),
                "confidenceMeaning": "uncalibrated model activation",
                "postprocessing": "local peaks with downbeat snapping; no meter assumptions",
            },
        }
    finally:
        if model is not None:
            model.w.clear()
        del model
        gc.collect()
        mx.clear_cache()
