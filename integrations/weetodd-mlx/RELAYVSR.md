# Native RelayVSR 2×

`WeeToddRelayVSR` upscales local SDR video on Apple Silicon. Swift MLX runs the
Wan keyframe transformer and Flash reconstruction network on Metal.
AVFoundation reads the source and writes an HEVC QuickTime movie.

## WeeTodd Studio

In the media library, open **Upscale Video**, select **RelayVSR 2×**, and choose a movie.
Use **Choose RelayVSR model folder…** if Studio has not discovered the installed checkpoints.
**Upscale** adds the result to **Project Assets**.

For a timeline interval, use **Clip actions → Upscale movie copy → RelayVSR 2×** on a
source clip, then **Upscale clip**. This uses the reference interval and adds the take
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

Place the two files from the [published 2× checkpoint set](https://huggingface.co/Comfy-Org/RelayVSR)
in the same model directory:

- `relayvsr_x2_lightx2v_int8_convrot.safetensors`
- `relayvsr_flash_vae_x2_fp32.safetensors`

The worker reads both files directly. It uses the saved positive conditioning
and the signed INT8 Wan weights in their rotated basis.

## Run

```sh
integrations/weetodd-mlx/.build/arm64-apple-macosx/release/WeeToddRelayVSR \
  input.mp4 output.mov --models "$RELAYVSR_MODEL_DIR"
```

The default uses four Euler steps, keyframes 15 source frames apart, seed 42
and detail gain 1. Use `--steps 1` for one-step sampling and `--frames 16` for
a short preview. Use `--source-start` and `--source-duration` in seconds to select
an interval. `--detail` scales the learned residual over the bicubic base.
The worker writes one output frame per source frame using the source
presentation timestamps, and copies source audio over the processed timeline.
Use a new output filename.

The original model architecture and streaming method are described in the
[RelayVSR project](https://kopperx.github.io/RelayVSR/).
