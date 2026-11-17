# Native FlashVSR+ restoration

Open **Upscale Video** in the media library and choose **FlashVSR+**. The bundled Swift/MLX worker executes Turbo or Base with Tiny or Full Wan VAE. Setup, checkpoint/archive conversion and inference require no Python environment, PythonKit, PyTorch, Diffusers, Transformers or vpipe. Existing FFmpeg and FFprobe handle media transport.

## Setup

1. Choose **Model setup → Choose model library…** and select the shared AI model library.
2. Choose Turbo or Base and the decoder, then **Prepare selected FlashVSR+ models**.
3. Setup reuses checksum-verified installed files before downloading missing pinned components into the library's `FlashVSR-Plus` family folder.
4. Existing installations can use **Connect models…**. Connect FFmpeg/FFprobe if Studio has not detected them.
5. Select a movie, set its output dimensions, and use **Check setup** before rendering.

Original and converted weights stay in the model library. No weights belong in a checkout, result folder or temporary directory. Existing different files are never overwritten. The selected decoder is mandatory; a missing decoder does not select another one.

Models are pinned to [publisher revision 46b7952](https://huggingface.co/ruchangyao/FlashVSR_plus/tree/46b79520946ac67dec94d1566490690953c7debc); the fixed context comes from [official code revision b548a2cc](https://github.com/OpenImagingLab/FlashVSR_Plus/tree/b548a2cc2c929d3d4676f2973abe27876ec4dee7). Keep the original model/component licenses with acquired files. No text encoder is required.

A connected model root contains:

```text
FlashVSR_plus/
  posi_prompt.pth
  turbo/                 # or base/ for the selected model
    dit.safetensors
    lq_proj_in.safetensors
  TCDecoder.ckpt         # Tiny
  Wan2.1_VAE.pth          # Full; needed only when selected
```

Native lookup selects root-shared prompt/decoder files before variant-local copies. Hash mismatch rejects the selected file; it never retries a different candidate. Canonical paths, symlink targets and hashes are checked before execution. Historical vpipe jobs retain their original variant-first lookup.

The Swift importer reads only the three pinned Tiny, Wan and positive-context archive layouts. It parses bounded tensor metadata and raw storage without executing pickle or loading Python. `models convert` optionally creates deterministic, checksum-bound safetensors caches under the same model library.

## Controls

| Control | Default | Behavior |
| --- | --- | --- |
| Model | Turbo | Turbo and Base are separate pinned transformer/projection pairs. |
| Decoder | Tiny | Tiny is source-conditioned; Full selects Wan2.1 VAE. Both are explicit. |
| Output | 1280 × 720 | Exact even delivery dimensions, 64–8192; memory/indexing admission can reject a larger grid. |
| Aspect | Fit | Retains the picture with margins; Fill center-crops. |
| Motion compression | 4 | Values 2 and 1 use more temporal model samples and processing time. Visible frame count and source FPS remain unchanged. |
| Color | Wavelet | Wavelet, native AdaIN or Off; Off retains Tiny's required source conditioning. |
| Wavelet levels | 5 | Range 1–8. |
| Seed | 0 | Stored UInt32; Randomize resolves to a recorded value. |
| Original audio | On | Aligns the selected original interval and verifies exact decoded float PCM in native MOV delivery. |

Native execution uses stateful streaming, fixed positive context and one flow update per model chunk. It exposes no misleading CFG, prompt or editable step-count control. Overlapping Clips is a historical vpipe option; selecting it for native execution rejects before inference.

Source movies must be SDR, square-pixel, constant-rate and without alpha. Every timestamp is checked, including the final frame. Rates from 1 to 120 FPS include rational cadence such as 30000/1001. Upscaling never interpolates or silently changes FPS. HDR, VFR and unsupported pixel formats need a separately chosen preparation step.

The model canvas rounds upward to multiples of 128, with replicated right/bottom padding. A 1920 × 1080 delivery uses a 1920 × 1152 model canvas and crops only that padding. Source framing remains explicit. The standalone tool processes the complete selected movie; a typed CLI request can select an exact `firstFrame`/`frameCount` interval.

## Execution, media and resource admission

Projection, transformer and decoder execute sequentially and release weighted state between stages, on success, failure and cancellation. Transformer state is process-local. The worker reports stage times, peak MLX/process footprint and release measurements. It holds the shared inference lease, watches its owner, and releases the lease after weighted execution exits. No alternate backend or decoder starts on failure.

Native media transport retains RGB24 pixels and exact source cadence. A standalone result gets one final H.264 video encode in MOV, with original aligned `pcm_f32le` audio. Decoded PCM hashes must match the preserved source interval. Leading/trailing audio gaps are padded explicitly; video is never shortened to hide missing frames. Comparison proxies do not affect inference or delivery pixels.

Version 2 processing sequences exchange verified FFV1 RGB intermediates and encode delivery only after the final stage. H3 latents never feed FlashVSR. A Pixel stage receives a separately verified lossless MOV bridge where AVFoundation needs one. Source audio is preserved once, independently of all model stages.

Disk admission includes projection/latent spools, decoder carry state, media preparation and final delivery. Wan spatial tiles spill activation carry into job-owned scratch; weights stay in the library. Only the selected tile's carry is resident. Scratch is cleaned after completion/cancellation. The active-memory ceiling is bounded by physical memory and the worker's plan; a large Mac's successful run does not qualify a smaller machine.

Actual-weight checks cover all four model/decoder pairs at compression 4, 2 and 1, multiple groups, partial tails, 24 and 30000/1001 FPS, exact output hashes/counts and zero active MLX bytes at weighted-stage release. These are functional qualification cells, not a universal quality, 4K, OS or throughput claim. Native sparse attention uses bounded routing/score storage; CUDA numerical or speed parity is not claimed. Native AdaIN uses its recorded population-variance convention.

For the released 128-token, 128-channel attention windows, the worker groups up to eight independent windows per GPU submission. Each window retains its original rank-two matrix operations, selected-key order and eight-key-window slabs. Scratch remains below the existing 64 MiB attention reserve; other geometries and custom slab sizes keep serial scheduling. M3 Ultra synthetic operator checks retained every Float32/BF16 output bit and measured 3.32–3.51× warm operator speedups. These measurements exclude model loading, projection, other transformer operations and decoding; they do not establish whole-video throughput.

On qualified `g15d` GPUs, draft routing now computes serial Float32 dot products
and exact strict-threshold ranking on the GPU. The original host square root,
division, exponential and denominator order are retained. When the candidate
count is strictly below capacity, finite-score checks prove that every candidate
is selected. Ties retain the original first-candidate fallback. Unsupported
layouts and unsafe numerical ranges use the original CPU route. Admission adds
a bounded 512 MiB routing reserve, separate from attention scratch.

Four balanced M3 Ultra native-size routing pairs retained every selection and
measured about 5.7× local-window and 9.3× broad-window operator speedups. Routing
is only part of transformer execution; these ratios are not whole-movie gains.

Before compiled scheduling, a complete M3 Ultra 124-frame 1344 × 768 → 2688 × 1536 Turbo/Tiny
job at 24 FPS measured 2628.26 → 2505.58 seconds against the earlier complete
baseline. Every native RGB byte before encoding, delivered pixel and aligned
PCM sample matched; peak MLX allocation remained 61.24 GB, with zero active
bytes at each weighted-stage release. A newly paired 17-frame candidate/control
check, exercising both GPU routing modes, measured 478.08 → 475.56 seconds
(control → candidate). The full observation used 4.7% less time, but the paired
prefix improved only 0.5%; a repeatable 4.7% whole-job gain is unestablished.
These measurements qualify this source and 2× profile on M3 Ultra. FlashVSR+
4× was excluded from this performance task.

## Compiled attention scheduling

Qualified `g15d` GPUs compile the existing per-slab graph and queue at most
two attention slabs. The original rank-two Float32 matrix operations,
maximum/sum reductions, key order, eight-key-block slabs, eight-window groups
and final BF16 conversion are retained. Compilation state belongs to one
attention call and releases with it; queued work drains on success and
cancellation. The two-slab scratch bound remains within the existing 64 MiB
reserve. Other architectures, custom slabs/geometry and unsupported dtypes
keep the original consumer.

A fresh complete M3 Ultra comparison against the GPU-routing baseline measured
2547.05 → 1449.70 seconds (42m27s → 24m10s, 43.08% less time) for the same
124-frame 1344 × 768 → 2688 × 1536, 24 FPS Turbo/Tiny profile. Every native RGB
byte before encoding, delivered pixel and aligned PCM sample matched. Peak
MLX allocation stayed at 61.24 GB, with zero active bytes at weighted-stage
release. Full-resolution 17-frame A/B/B/A runs averaged 479.79 → 285.24 seconds
(40.55% less time). These are measurements of one source/profile on M3 Ultra;
broader sources, hardware, Base/Full timing and CUDA pixel parity remain
unqualified. FlashVSR+ 4× was excluded.

## Retry, ordered processing and saved jobs

Enable **Upscale after generation** and choose FlashVSR+, H3 Regenerate, native Pixel/CQ or an explicit ordered combination. Enabling the control starts with one FlashVSR stage. H3 → FlashVSR and Pixel/CQ → FlashVSR are selected deliberately. Existing saved recipes retain their recorded settings/backend.

All selected components pass admission before source generation. The accepted original take remains in Versions. **Upscale accepted take** retries processing without regenerating. A changed document, source, recipe or editorial revision prevents a late result from replacing the current clip.

Each completed stage has an immutable receipt and its own cache identity: decoded input RGB plus exact clock, that stage's settings, models, importer/layout and worker/media identities. Changing a downstream stage or delivery codec reuses a verified prefix. An interrupted stateful stage restarts; incomplete outputs are never completed checkpoints. Changed/corrupt artifacts are retained and retried in a new attempt. Continuous-scene members share one recipe and retain editorial ranges.

New version 2 sequences export a native v3 job envelope. Native v1/v2 readers remain supported. Generation and processing fingerprints are separate. The same executor serves Studio, Director acceptance, CLI and maintained ComfyUI handoffs. Interactive Draw Things/LTX 2.3 results can be processed; legacy exporters reject unsupported pending processing rather than dropping it.

```bash
WeeToddCLI upscale --request Upscale.json --output-directory Upscaled --preflight-only
WeeToddCLI upscale --request Upscale.json --output-directory Upscaled
WeeToddCLI process --request Processing.json --output-directory ProcessingCache --resume
WeeToddCLI models download --package flashvsr-native-turbo-tiny --model-library MODEL_LIBRARY
WeeToddCLI models convert --profile tiny --source TCDecoder.ckpt --model-library MODEL_LIBRARY
```

A command-line build outside the app needs `--catalog` for model setup. Requests reject unknown/ignored controls. The native worker and shared Metal library ship in a complete app bundle. Old explicit vpipe jobs remain under **Historical backend** and keep their external installation/OS limits; they never substitute for native execution.

The **WeeTodd FlashVSR+ Upscale** ComfyUI node is a lightweight CLI translator. The host's existing Python runs the node, while model setup/conversion/inference run in Swift. New UI prompts explicitly select `swiftMLX`; old API prompts without that selection retain their historical vpipe identity. Portable [UI](../workflows/speed/video-upscale/flashvsr_turbo_standalone.json) and [API](../examples/flashvsr_plus_standalone_api.json) templates require local paths before runtime execution.

Historical vpipe timing/quality measurements are separate from native qualification; they do not establish Swift throughput. Generative restoration can alter text and fine detail, so keep originals and compare representative footage.
