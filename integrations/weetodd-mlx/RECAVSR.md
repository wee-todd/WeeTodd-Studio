# Native ReCaVSR

ReCaVSR is an experimental movie upscaler implemented in Swift/MLX and Metal.
Studio and the standalone `WeeToddReCaVSR` worker use the same engine. Setup,
checkpoint reading and inference do not require Python, ComfyUI or CUDA.

## Studio

Open **Upscale Video** in the media library and select **ReCaVSR**. Choose the
shared model folder or **Download selected model files**, then **Check setup**
and **Upscale**. Results retain the original movie and save a reproducible
receipt in Project Assets. Select a result again to restore its source/settings.

For a selected source interval, use **Clip actions → Upscale movie copy →
ReCaVSR**, adjust its upscale controls, then choose **Upscale clip**. The new
take appears in the copy's Versions. Unrelated H3/LTX setup and a text prompt
are not required.

Scale defaults to 4×; 2× and 1.5× are available. Choose BF16 reference or INT8
ConvRot, seed and optional AdaIN/wavelet correction. Default staged execution
releases model parameters between stages while retaining bounded streaming
state. **Keep both model stages in memory** is an explicit alternative subject
to memory admission. Output dimensions are rounded half-up; internal padding
never adds delivered pixels or frames.
HEVC export requires even output dimensions. Unsupported exact rasters fail
before model loading; the app does not change the selected scale silently.

## Models

Choose the shared model library during setup. ReCaVSR files live in its
`WeeTodd/ReCaVSR` family folder:

- `official/transformer.safetensors`: the released merged BF16 ReCaVSR DiT.
- `int8/transformer_int8_convrot.safetensors`: optional INT8 alternative.
- `shared/prompt.safetensors`, `shared/flashdecoder.safetensors` and the matching
  `model_config.json`, `vae_config.json`, `flashdecoder_config.json`.

Setup reuses compatible installed files and verifies pinned content hashes.
The original Wan VAE is not needed. Generic Wan weights, RelayVSR's decoder,
and arbitrary LoRA stacks are not substitutes for this released checkpoint set.

[Official code](https://github.com/kopperx/ReCaVSR) and
[official weights](https://huggingface.co/kopper/ReCaVSR) declare Apache-2.0.
The [INT8 conversion](https://huggingface.co/PedroMarinhoDev/ReCaVSR-ComfyUI)
has a separate card referring to upstream model terms. Weights are downloaded
separately from the application. The GPL ComfyUI wrapper is not imported or
used as an inference backend.

## Standalone worker

Use Swift 6.3 or later and the Xcode Metal toolchain:

```sh
swift build --build-system native --package-path integrations/weetodd-mlx \
  -c release --product WeeToddReCaVSR
bash integrations/weetodd-mlx/scripts/build_mlx_metallib.sh
integrations/weetodd-mlx/.build/release/WeeToddReCaVSR \
  input.mov output.mov --models "$RECAVSR_MODEL_DIR" --scale 4
```

Keep the matching `mlx.metallib` beside a standalone executable. Packaged
workers use the app's verified shared library. `--preflight` checks inputs,
content identities and memory admission without inference. `--help` lists
precision, color correction, seed, memory budget and residency options.
`--source-start` with `--source-duration` selects an interval; `--frames` limits
a preview. Movies retain source sample timestamps, fractional/VFR cadence,
final frame timing and source audio. There is no movie FPS override or frame
interpolation. Output is a new HEVC QuickTime movie with a JSON receipt.

The model generates a 21-frame prefix and subsequent 8-frame blocks. The
retained KV history, recycled latents, causal LQ convolutions and decoder carry
belong to one continuous shot. `--reset-frames 100,200` starts a fresh model
session at those selected-source frame indices without changing the media
clock. Automatic cut detection and original Wan VAE decoding are not included.
HDR and alpha inputs require a separately supported conversion first.

## Qualification limits

Native layout/math, stream state and delivery are tested separately. Initial
weighted checks cover BF16 and INT8 movie export, fractional source timestamps,
byte-identical AAC packet passthrough and a 101-frame rolling-cache stream.
Receipts record actual MLX peak/cleanup memory and worker/model identities.

Initial M3 Ultra measurements include asset verification, model loading and
movie export. They are small qualification runs, not resolution-wide limits:

| Profile | Source → output | Frames | Wall time | MLX peak |
| --- | --- | --- | --- | --- |
| BF16 staged | 64×48 → 128×96 | 29 | 36.00 s | Not captured |
| INT8 staged | 64×48 → 128×96 | 29 | 18.94 s | Not captured |
| INT8 resident | 64×48 → 128×96 | 101 | 17.18 s | 7.10 GB |
| INT8 staged | 256×144 → 1024×576 | 17 | 20.83 s | 8.45 GB |
| BF16 staged | 256×144 → 512×288 | 17 | 25.27 s | 11.36 GB |
| INT8 staged, wavelet, reset at 8 | 256×144 → 384×216 | 17 | 23.10 s | 6.91 GB |

Staged/resident INT8 produced identical decoded frame hashes on the same
29-frame source and seed. The 101-frame run preserved all source timestamps
and 199 AAC packet payloads; the 1.5× wavelet/reset test preserved 29 timestamps
and 58 payloads. Cancellation removed partial output and owned scratch files.
The remaining active MLX allocation after closing both model stages was about
4.2 MB for the fixed prompt. On the last two rows, macOS reported peak process
footprints of 13.44 GB and 8.93 GB respectively, separately from MLX allocation.
System memory pressure and per-stage latency have not been benchmarked.
Processing FPS is frames divided by wall time; these short jobs
are slower than their source playback rate.

GPU tile preparation now packs Q/K/V and temporal masks together from compact
cached spatial descriptors, with at most eight tiles per submission. Each tile
keeps its original SDPA shape, key order, precision and reduction. Admission
includes the added packing workspace; unsupported packed geometries retain the
original implementation. Staged execution remains the default.

On M3 Ultra with 256 GiB memory, a complete 124-frame, 24 FPS movie from
1344×768 to 2688×1536 took 480.39 seconds with this path, versus 691.21 seconds
with the previous native implementation (30.5% less wall time). Both used BF16,
staged execution, seed 42, no color correction and the same 96 GiB budget. All
decoded RGB pixels and AAC packet payloads/timing matched. A balanced 29-frame
A/B/B/A replay averaged 139.68 versus 89.00 seconds, also with exact decoded
RGB/audio equality. These are measurements on one source and machine, including
model loading and export. The complete native 4× job (5376×3072) took 1023.83
seconds versus 1988.80 seconds (48.5% less time), with all decoded RGB pixels,
AAC packets and source frame timing unchanged. The 4× replay used the packaged
release worker. Peak MLX allocations matched the old path at both scales:
19.14 GB at 2× and 43.15 GB at 4×. Individual attention operators use extra
bounded scratch, accounted for during admission. Other clips, precisions and
hardware need their own performance qualification.

Bounded asynchronous scheduling is now qualified on `g15d`: at most two
independent packed groups overlap, with every group explicitly drained before
stage cleanup. Existing two-group admission, per-tile SDPA calls, temporal masks,
precision and ordering are retained. Call-local compiled pack/SDPA wrappers were
also tested but provided no consistent extra gain and remain disabled by default.
Other GPU paths retain synchronous packed execution.

A November 14 matched complete 124-frame, 24 FPS comparison on M3 Ultra with
256 GiB measured 430.43 → 406.85 seconds at 2× (5.48% less time) and
994.46 → 922.42 seconds at native 4× (7.24% less time). Settings remained BF16,
staged residency, seed 42, no color correction and a 96 GiB budget. Exact outputs
were 2688 × 1536 and 5376 × 3072 from the same 1344 × 768 source. All selected
pairs have identical decoded RGB, AAC packets and source presentation timing.
The selected 4× control is an unchanged frozen-worker repeat: the first
1033.06-second control movie differed from frame 60, where HEVC slice QP was 26
versus 18 in the candidate. The repeat and candidate both match the archived
complete baseline. The first observation is retained/excluded; native pixels
before encoding were not captured, so its origin is not proven. No code was
changed to obtain the repeat.

Balanced 29-frame A/B/B/A means were 82.98 → 74.24 seconds (2×) and
218.71 → 199.85 (4×), with exact RGB/audio equality. Peak MLX allocations matched
at 19.14 GB and 43.15 GB; cleanup retained 4.19 MB for the existing fixed prompt.
Operator gains and preview times remain separate from complete-job throughput.
Full core/Studio validation, independent source review and isolated release
packaging/signature passed. Other source/hardware and precision/residency choices
need their own qualification.

Cross-platform image parity with the original CUDA runtime remains unverified.
MLX INT8 affine projection preserves the saved rotation and row scales but
does not reproduce CUDA's dynamic activation quantization. CUDA speed or
bit-identity claims do not apply to this implementation. Large output sizes
require explicit memory and encoder admission; installed weights alone do
not establish 4K support.

### Native RGB and HEVC qualification (November 15)

An optional `--capture-rgb /absolute/new-directory` writes the owning RGB24 bytes
at the encoder handoff, with per-frame/whole-stream SHA-256, rational source
timestamps/durations and approximate producer arrival times in `manifest.json`.
The parent directory must exist; the capture directory must be new. Capture adds
disk I/O and hashing, so use capture-off jobs for performance measurements.
`--attention-mode sync|auto` selects the diagnostic control with the same worker;
`auto` remains the normal qualified asynchronous path on `g15d`.

Complete synchronous/asynchronous native 4× runs produced identical bytes for all
124 frames (6,143,606,784 bytes at 5376 × 3072), with identical source timing. Both
movies matched the prior canonical output. This proves equality for these runs;
the historical inconsistent control has no native capture, and its exact trigger
remains unproven. A 70-second pause alone did not reproduce it.

The ReCa HEVC writer now explicitly requests VideoToolbox quality 0.9 instead of
leaving export quality to encoder defaults. Controlled replays of identical RGB
and sample timing changed with an unset expected-FPS hint versus an explicit
24 FPS hint in the old mode; the explicit quality mode produced identical decoded frames under both
hints and a 70-second pause before frame 60. Playback timing is never synthesized
from an encoder hint. No bitrate target, QP clamp or speed-priority mode is added.

Against the saved native frames, quality 0.9 measured 47.61 dB RGB PSNR versus
45.65 dB for the old canonical movie, with improved PSNR and MAE on every frame
of this clip. Video-only size increased from 34.0 MB to 84.9 MB. This is an
explicit export-quality policy and a bounded hardware qualification, not a
guarantee of encoder bit identity across machines or OS releases. Inference
arithmetic and the asynchronous queue are unchanged.

Final capture-off complete-job observations were 1118.25 seconds (sync) and
963.07 seconds (async). Concurrent GPU activity affected this timing campaign;
the values are retained as contended observations. They do not establish a new
idle speedup or a regression against the earlier campaign. Both finished movies
have identical decoded pixels, source presentation timing and AAC packets, with
43.15 GB peak MLX allocations and the same 4.19 MB staged cleanup state.
