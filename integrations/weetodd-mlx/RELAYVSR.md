# Native RelayVSR 2× and 4×

`WeeToddRelayVSR` upscales local SDR video on Apple Silicon. Swift MLX runs the
Wan keyframe transformer and Flash reconstruction network on Metal.
AVFoundation reads the source and writes an HEVC QuickTime movie.

## WeeTodd Studio

In the media library, open **Upscale Video**, select **RelayVSR**, choose **2×** or **4×**, and choose a movie.
Use **Choose RelayVSR model folder…** if Studio has not discovered the installed checkpoints.
**Upscale** adds the result to **Project Assets**.

For a timeline interval, use **Clip actions → Upscale movie copy → RelayVSR** on a
source clip, select the scale, then **Upscale clip**. This uses the reference interval and adds the take
to the copy's **Versions**.

## Build the standalone CLI

From the repository root, using Swift 6.3 or later and the Xcode Metal toolchain:

```sh
swift build --build-system native --package-path integrations/weetodd-mlx \
  -c release --product WeeToddRelayVSR
integrations/weetodd-mlx/scripts/build_mlx_metallib.sh
```

Keep `mlx.metallib` beside the executable when running it outside the build folder.

## Checkpoints

Use the matching pair from the [published checkpoint sets](https://huggingface.co/Comfy-Org/RelayVSR).
Keep weights in the shared model library, with a separate subfolder for each scale:

- **2×:** `x2/relayvsr_x2_lightx2v_int8_convrot.safetensors` and
  `x2/relayvsr_flash_vae_x2_fp32.safetensors`.
- **4×:** `x4/relayvsr_x4_int8_convrot.safetensors` and
  `x4/relayvsr_flash_vae_x4_fp32.safetensors`.

The worker reads both files directly. It uses the saved positive conditioning
and the signed INT8 Wan weights in their rotated basis. The folder picker accepts
the family folder or a scale subfolder; switching scale resolves its matching sibling
pair. Missing or mismatched pairs fail before inference. Both networks compute in BF16.
The native 4× model uses its released learned LQ projection and decoder strides.

## Run

```sh
integrations/weetodd-mlx/.build/arm64-apple-macosx/release/WeeToddRelayVSR \
  input.mp4 output.mov --models "$RELAYVSR_MODEL_DIR" --scale 4
```

Scale defaults to 2× for existing jobs. The default uses four Euler steps, keyframes 15 source frames apart, seed 42
and detail gain 1. Use `--steps 1` for one-step sampling and `--frames 16` for
a short preview. Use `--source-start` and `--source-duration` in seconds to select
an interval. `--detail` scales the learned residual over the bicubic base.
The worker writes one output frame per source frame using the source
presentation timestamps, and copies source audio over the processed timeline.
Use a new output filename.

Add `--preflight` to validate the checkpoint headers, exact output raster and estimated
Metal working set without loading weights. Output dimensions are exactly the selected
multiple of the source dimensions, even above 4K; there is no delivery-size cap.
The source must fit memory admission and the hardware encoder must support the raster.
Cross-platform pixel parity against the CUDA reference remains unqualified.

## Qualified attention work skipping

On `g15d` GPUs, native tiled attention caches compact classifications of each
32-query key block as mixed, fully visible or safely masked. The consumer
retains the pinned MLX Steel Float32 matrix arithmetic, reduction order, BF16
output and chronological keys. Fully masked blocks are skipped only after
every valid query has seen a visible key; mixed blocks keep the original mask.
A fused device safety scan disables skipping for unsafe numeric ranges.
Whole-grid, short-vector, unsupported layout and other GPU paths retain MLX.
Memory admission includes the metadata and compact safety flags.

Balanced M3 Ultra tests at native 2×/4× attention geometry retained every output
word and used approximately 3–11% less attention time. These operator figures
exclude loading, other model operations and movie export. Whole-movie timings
require their own matched measurement.

Complete 124-frame, 24 FPS qualification on M3 Ultra used fresh controls from
the frozen current app: 1344 × 768 → 2688 × 1536 measured 156.80 → 155.48 seconds;
native 5376 × 3072 measured 800.18 → 760.99 seconds. Every decoded RGB pixel,
AAC packet and source frame timestamp/duration matched. Balanced 29-frame
A/B/B/A runs averaged 40.83 → 40.71 seconds at 2× and 200.42 → 188.92 seconds
at 4×. These observations support a roughly 5% 4× gain; the small 2× difference
does not establish a repeatable gain. The earlier archived 4× movie diverged
from the fresh current control and was excluded from these comparisons.

## Bounded attention scheduling

Qualified `g15d` attention now queues at most two groups of up to eight tiles.
The CPU can build the next graph while Metal processes the previous one;
all retained groups finish before return, cancellation cleanup or unloading.
Admission explicitly reserves the gather workspace, with a synchronous fallback
when two groups exceed the attention bound. The original custom/SDPA arithmetic,
masks, head layout, dtype and inverse ordering are retained. Call-local compiled
wrappers passed word checks but showed no consistent gain over async-only and
remain disabled by default. Other GPU paths retain synchronous execution.

A November 14 matched complete 124-frame, 24 FPS comparison on M3 Ultra with
256 GiB measured 156.37 → 139.77 seconds at 2× (10.62% less time) and
773.81 → 738.18 seconds at native 4× (4.60% less time). The 1344 × 768 source
produced exact 2688 × 1536 and 5376 × 3072 movies. Every decoded RGB pixel,
AAC packet and source frame timestamp/duration matched. Balanced 29-frame
A/B/B/A means were 41.50 → 37.31 and 191.17 → 189.10 seconds respectively;
the small 4× preview difference is nearly flat. These are observed results on
one source/machine, including model loading and export; they do not establish
hardware-wide gains. Full core/Studio validation and isolated release
packaging/signature passed. The disabled compiled experiment needs shared
invalid-input validation before promotion; the selected async-only model path
retains its eager guards.

The original model architecture and streaming method are described in the
[RelayVSR project](https://kopperx.github.io/RelayVSR/).
