"""ECAPA-TDNN reference speaker features for Qwen3-TTS Base."""

import mlx.core as mx
import numpy as np

from wee_todd_mlx.speech_ops import Weights


def mel_spectrogram(audio):
    # Slaney's piecewise mel scale with area-normalized triangular filters.
    def mel(hz):
        hz = np.asarray(hz, dtype=np.float64)
        return np.where(
            hz < 1000, hz / (200 / 3), 15 + np.log(np.maximum(hz, 1) / 1000) / (np.log(6.4) / 27)
        )

    points = np.linspace(mel(0), mel(12000), 130)
    hz = np.where(points < 15, points * (200 / 3), 1000 * np.exp((points - 15) * np.log(6.4) / 27))
    frequency = np.linspace(0, 12000, 513)
    filters = np.maximum(
        0,
        np.minimum(
            (frequency[None] - hz[:-2, None]) / (hz[1:-1] - hz[:-2])[:, None],
            (hz[2:, None] - frequency[None]) / (hz[2:] - hz[1:-1])[:, None],
        ),
    )
    filters *= (2 / (hz[2:] - hz[:-2]))[:, None]
    padded = np.pad(np.asarray(audio, np.float32), (384, 384), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 1024)[::256]
    spectrum = np.fft.rfft(windows * np.hanning(1025)[:-1], axis=-1)
    magnitude = np.sqrt(np.abs(spectrum) ** 2 + 1e-9)
    return mx.array(np.log(np.maximum(magnitude @ filters.T, 1e-5)).astype(np.float32)[None])


class SpeakerEncoder(Weights):
    def conv(self, x, name, dilation=1, activate=False):
        w = self.w[name + ".weight"]  # This component is already in MLX O-K-I layout.
        pad = (w.shape[1] - 1) * dilation // 2
        if pad:
            x = mx.concatenate(
                [x[:, 1 : pad + 1][:, ::-1], x, x[:, -pad - 1 : -1][:, ::-1]], axis=1
            )
        y = mx.conv1d(x, w, dilation=dilation) + self.w[name + ".bias"]
        return mx.maximum(y, 0) if activate else y

    def __call__(self, audio):
        x = self.conv(mel_spectrogram(audio), "speaker_encoder.blocks.0.conv", activate=True)
        states = []
        for i, dilation in enumerate((2, 3, 4), 1):
            p = f"speaker_encoder.blocks.{i}"
            y = self.conv(x, p + ".tdnn1.conv", activate=True)
            chunks = mx.split(y, 8, axis=-1)
            parts = [chunks[0]]
            for j in range(1, 8):
                z = chunks[j] if j == 1 else chunks[j] + parts[-1]
                parts.append(self.conv(z, f"{p}.res2net_block.blocks.{j - 1}.conv", dilation, True))
            y = self.conv(mx.concatenate(parts, axis=-1), p + ".tdnn2.conv", activate=True)
            scale = self.conv(
                mx.mean(y, axis=1, keepdims=True), p + ".se_block.conv1", activate=True
            )
            scale = mx.sigmoid(self.conv(scale, p + ".se_block.conv2"))
            x = x + y * scale
            states.append(x)
        x = self.conv(mx.concatenate(states, axis=-1), "speaker_encoder.mfa.conv", activate=True)
        mean = mx.mean(x, axis=1, keepdims=True)
        std = mx.sqrt(mx.var(x, axis=1, keepdims=True) + 1e-12)
        context = mx.concatenate(
            [x, mx.broadcast_to(mean, x.shape), mx.broadcast_to(std, x.shape)], axis=-1
        )
        weights = self.conv(context, "speaker_encoder.asp.tdnn.conv", activate=True)
        weights = mx.softmax(self.conv(mx.tanh(weights), "speaker_encoder.asp.conv"), axis=1)
        mean = mx.sum(weights * x, axis=1, keepdims=True)
        std = mx.sqrt(mx.maximum(mx.sum(weights * (x - mean) ** 2, axis=1, keepdims=True), 1e-12))
        return self.conv(mx.concatenate([mean, std], axis=-1), "speaker_encoder.fc")[:, 0]
