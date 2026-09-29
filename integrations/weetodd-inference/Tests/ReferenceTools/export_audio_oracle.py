#!/usr/bin/env python3
"""Qualification only: independent MLX numerical oracle for Swift audio decoding.

No Python is called by the Swift implementation. The installed decoder supplies
weight loading/topology; the explicitly replaced equations below follow the
released checkpoint contract, independently expressed using MLX primitives:
ComfyUI b16023b004d3b1bfbbd6463414dc20da1b36cc4d,
comfy/ldm/lightricks/vocoders/vocoder.py: UpSample1d.forward,
Vocoder.forward (use_tanh_at_final=false), and _STFTFn.forward.
The BWE skip uses its internal Hann UpSample1d with replicate boundaries;
comfy.audio.resample uses zero boundaries in a separate preprocessing path.
The existing MLX source is neither edited nor treated as a byte-for-byte oracle.
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np


def alias_upsample(self, x):
    # Replicate samples BEFORE transposed convolution, then crop. MLX input BTC.
    pad = 5
    padded = mx.concatenate(
        [mx.repeat(x[:, :1], pad, axis=1), x, mx.repeat(x[:, -1:], pad, axis=1)], axis=1
    )
    b, t, c = padded.shape
    flat = padded.transpose(0, 2, 1).reshape(b * c, t, 1)
    y = mx.conv_transpose1d(flat, self.filter, stride=2) * 2
    y = y[:, 15:-15]
    return y.reshape(b, c, y.shape[1]).transpose(0, 2, 1)


def hann_bwe_resample(self, x):
    # BWE's actual primary path: 43-tap Hann filter, replicate BEFORE transpose.
    time = (np.arange(43, dtype=np.float64) / 3 - 7) * 0.99
    window = np.cos(np.clip(time, -6, 6) * np.pi / 12) ** 2
    kernel = mx.array((np.sinc(time) * window * 0.99 / 3).astype(np.float32))
    padded = mx.concatenate(
        [mx.repeat(x[:, :1], 7, axis=1), x, mx.repeat(x[:, -1:], 7, axis=1)], axis=1
    )
    result = mx.conv_transpose1d(padded[:, :, None], kernel[None, :, None], stride=3)
    return (result[:, 42:-40, 0] * 3)[:, : x.shape[1] * 3]


def base_vocoder(self, mel):
    x = self.conv_pre(mel)
    for i in range(self.num_upsamples):
        x = self.ups[i](x)
        x = (
            sum(self.resblocks[i * self.num_kernels + j](x) for j in range(self.num_kernels))
            / self.num_kernels
        )
    return mx.clip(self.conv_post(self.act_post(x)), -1, 1)


def mel_stft(self, waveform):
    x = mx.pad(waveform[:, :, None], [(0, 0), (432, 0), (0, 0)])
    spectrum = mx.conv1d(x, self.stft_fn.forward_basis, stride=80)
    magnitude = mx.sqrt(spectrum[:, :, :257] ** 2 + spectrum[:, :, 257:] ** 2)
    return mx.log(mx.maximum(magnitude @ self.mel_basis.T, 1e-5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.frames <= 26:
        parser.error("Qualification frames must be in 1...26")
    sys.path.insert(0, str(args.project / "src"))
    from ltx_core_mlx.model.audio_vae.bwe import HannSincResampler, MelSTFT, VocoderWithBWE
    from ltx_core_mlx.model.audio_vae.vocoder import UpSample1d
    from mlx.utils import tree_map

    from ltx25_mlx.components import LTX25AudioDecoder

    UpSample1d.__call__ = alias_upsample
    HannSincResampler.__call__ = hann_bwe_resample
    VocoderWithBWE._run_base_vocoder = base_vocoder
    MelSTFT.__call__ = mel_stft
    decoder = LTX25AudioDecoder(str(args.checkpoint))
    vae, vocoder = decoder.load()
    vae.update(tree_map(lambda p: p.astype(mx.float32), vae.parameters()))
    latent = (np.sin(np.arange(8 * args.frames * 16, dtype=np.float32) * 0.173) * 0.1).reshape(
        1, 8, args.frames, 16
    )
    mel = vae.decode(mx.array(latent))
    mx.eval(mel)
    wave = vocoder(mel)
    mx.eval(wave)
    payload = dict(
        frames=args.frames,
        sampleRate=48000,
        latent=latent.ravel().tolist(),
        mel=np.array(mel).ravel().tolist(),
        waveform=np.array(wave).ravel().tolist(),
        reference="MLX Float32 with released alias padding, base clamp and exact STFT magnitude",
        shape=list(wave.shape),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload))
    print(
        json.dumps(
            dict(output=str(args.output), shape=payload["shape"], peak=float(mx.max(mx.abs(wave))))
        )
    )
    decoder.free()


if __name__ == "__main__":
    main()
