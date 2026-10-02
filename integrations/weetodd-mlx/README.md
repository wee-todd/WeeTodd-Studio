# WeeTodd Swift MLX inference

Swift owns this native MLX component; its execution does not invoke Python. Studio can select
its LTX 2.5 worker for distilled T2V, first-image I2V, first/last-frame, one-driver A2V,
after-extension, per-clip motion continuation and continuous scenes with optional
first-frame images on any shot and one continuous source-audio driver. Its H3 worker
is opt-in and experimental for T2VA, timed FL2VA and still-image Ref2VA. Other tasks retain their existing
routes while native coverage qualifies.

## Implemented

- Trained Gemma4-12B text encoding with the shared embedded tokenizer, all 48 layers,
  feature-interleaved normalized states, and the eight-layer video/audio connectors.
  Embeddings read selected rows; aggregation stages at most 64 MiB of source rows instead
  of loading multi-gigabyte matrices. Packed Q8 and dense BF16 parameters retain their storage
  format. A conservative owned-buffer plan admits states and assembly overlap before loading.
- Prompt-to-sampler orchestration validates latent/position/schedule inputs first, snapshots
  caller handles, completes and unloads text weights, then invokes transformer providers.
  Text and sampling progress remain separate; text output stays in MLX throughout handoff.

- Direct use of existing group-64 affine Q8 block pages; no duplicate checkpoint or CPU
  expansion to full Float32 matrices. Dense BF16/F16 parameters retain their stored type.
- Joint video/audio self-attention, prompt attention, cross-modal attention, split RoPE,
  adaptive normalization, learned QK normalization, attention gates and feed-forward paths.
- Ordered standard LoRA factors evaluated on the GPU as `(x Aᵀ) Bᵀ`. Shape-compatible
  LTX 2.3 factors are accepted using the shared adapter validator. Factors are read in bounded
  4 MiB windows, with at most one packed factor staging buffer plus a read window.
- One resident block with evaluated hidden states kept in MLX between blocks. Prior weights
  release before replacement; success, failure and cancellation clear owned weights and cache.
- Compiled blocks receive current weights, conditioning and ordered adapter factors as explicit
  arguments. Native safetensors handles are materialized in the reported load phase; readers
  retain only unevaluated handles and verify checkpoint identity around evaluation.
- Staged fixed input/output projections, all eight timestep heads and their complete ordered
  LoRA effects. A sampling call prepares the finite schedule once, retaining only small evaluated
  modulation/embedding tensors (bounded to 128 MiB), not timestep weights.
- Multi-step audiovisual Euler/ancestral sampling uses the shared schedule/coefficients and
  explicit noise. Latents stay in MLX across steps. Progress and latent-preview callbacks run
  with no resident transformer weights; the Studio workers emit bounded decoded-frame previews
  during video decoding, after sampling.
- Distilled two-stage execution follows the shared eight-step ancestral / spatial x2 / three-step
  Euler recipe and versioned RNG streams (legacy native or released MLX-compatible draws). Both LoRA stacks are explicit and validated before
  payload reads. Video, text and audio stay in MLX across stages. The spatial upscaler uses
  full-volume group normalization and the trained convolution/shuffle order; its independent
  installed oracle matches exactly. This does not claim whole-job same-seed bitwise equivalence.
- Experimental spatial DFR uses the same streamed denoiser with learned generated-keyframe
  markers, seam-aware canvas padding, an appended clean half-resolution reference in stage two,
  and a complete rank-32 Pixel-Spatial x2 task adapter scoped to that stage. The direct worker
  supports text, first-image and first/last-image anchors; temporal DFR rounds and Studio controls
  remain separate work.
- The developer media CLI connects trained text, both sampling stages, native video/audio VAEs,
  bounded RGB24 streaming/WAV output and an explicit external FFmpeg muxer. Components unload in order.
  Output publishes from an owned temporary directory only after all stages succeed; cooperative
  cancellation removes partial output. It preserves actual causal audio samples without stretching.
- Native reference preparation completes orientation, sRGB, optional CRF and Lanczos center crop
  before any weighted stage. Bounded RGB buffers remain on disk until needed. A single-image
  causal VAE encoder reuses the installed convolutional VAE checkpoint with one weighted layer
  resident at a time. It encodes first/last images at both sampling resolutions.
- First-frame replacement and appended last-frame tokens use per-token video timestep modulation
  and masked clean predictions. Velocity-to-clean conversion uses those per-token times
  and restores the source dtype before masking; an all-one mask retains the global timestep. Protected latents are restored after ancestral re-noising;
  deterministic Euler retains one blend for fractional strengths. Original Float32 token times stay distinct from global BF16 scalar
  heads. Appended reference tokens participate in attention and are stripped before upscaling and
  decoding; output frames are never pasted from references. Fractional strengths are explicit.
  Repeated per-token modulations retain distinct rows and token indices; blocks expand one
  parameter slice at a time instead of retaining a full thirteen-parameter video tensor.
- Extension uses a bounded, SHA-256-verified source movie tail. Video VAEs encode its low/high
  grids, the audio VAE encodes synchronized sound, and both modalities join the per-token
  denoiser as protected reference rows. Source guide tensors release before output decode.
  Publication strips the repeated video context and crops audio to the exact added-frame
  duration. Studio prepares 25 frames for after-extension and 49 for motion continuation;
  source media, timing, dimensions and combined memory are admitted before weighted work.
- Experimental continuous scenes sample two to six windows with distinct prompts and
  seeds, carrying compact interior video and audio latent guides between windows. An
  opening image on any shot is encoded at both resolutions and can coexist with the
  same source-audio driver on every shot. Scenes with a later image decode groups
  between image cuts separately and join on exact editorial frames, avoiding full-scene
  VAE bleed across that cut; other scenes retain the selected single or bounded decode
  route. Publication drops the final causal video frame and
  trims audio to the same editorial duration. Preflight admits all windows and the complete
  scene decoder before loading weights; Studio reviews and accepts one movie with per-shot ranges.
  A 384 × 256, four-second recipe passed direct Swift and saved ComfyUI recipe-node runs with
  96 published frames and four seconds of stereo audio. A six-shot, 30-second, 768 × 448
  recipe passed installed-checkpoint preflight. A separate four-second first-image
  scene passed a signed-worker render with 96 frames and four seconds of stereo audio;
  its first frame measured 33.34 dB PSNR against the supplied image. A subsequent
  two-image scene with a continuous Qwen3-TTS driver published 96 frames and four
  seconds of source audio in 57.82 seconds at a 3.52 GB Swift-process peak, excluding
  FFmpeg. Studio also rendered and accepted both ranges with Python unavailable.
  Checkpoint resume and specialized controls are not yet supported by this Swift scene route.
- Scene video decoding has an explicit headless experimental `scene.decode_mode: "windowed"`
  option. An optional `scene.decode_window_frames` cap must be `8n+1` and at least 33;
  preflight rejects a cap below 57 when the scene needs an interior decode window.
  The decoder overlaps adjacent RGB windows by 25 frames and blends them while retaining
  one assembled audio decode. Studio scenes keep the single-decode default and expose bounded
  decoding as an experimental scene-inspector choice. On a matched four-second 384 × 256
  first-image recipe, the same debug worker took 68.57/71.36 seconds
  for single/windowed total execution, with 3.78/3.46 GB peak Swift-process footprint
  and 1.66/4.15 seconds of video decode. Audio WAVs matched byte for byte; the videos
  measured 40.60 dB median frame PSNR. A signed-worker six-shot, 30-second 768 × 448
  render published 720 frames and 30 seconds of stereo audio in 552.72 seconds. Its
  Swift process peaked at 10.84 GB (external FFmpeg excluded), and video decoding took
  27.46 seconds. Preflight estimated 18.25 GB of decoder activation for one decode
  versus 10.28 GB across three windows; those estimates are not process peaks.
  Inspected contact frames showed no visible cut at either decode-window join. Broader
  long-scene quality and low-memory-hardware behavior remain unqualified.
- A 128 MiB cache target during stack execution, with explicit trimming at block boundaries
  if MLX overshoots its advisory limit. This is not a hard peak-process-memory limit.
- Header/shape/packed-storage admission and conservative activation admission, including
  quadratic memory for non-fused attention geometries. Progress reports block completion,
  load/compute time and allocator residency without returning full activations to the CPU.

Current execution uses one batch and Float32 transformer activations. Version 3 can reproduce
the released renderer’s initial noise and BF16 latent-state boundaries while retaining Float32
video refinement state. Reference denoise masks affect video
timesteps and clean predictions; arbitrary attention/inpainting masks are not supported.
`MLXDenoiserWeights` validates the complete paged manifest, all 48 page headers, fixed projection
shapes, architecture/timestep/position semantics, confined paths and selected adapter targets before
reading weight payloads. SHA strings are syntax checked; payloads are not rehashed. The existing
block/stack probes remain lower-level tools with explicitly supplied page readers.

LoRA math is not fused/requantized into the base weights. This preserves packed base weights
and avoids dense CPU delta construction, but does not claim bitwise identity to a requantized
fusion recipe. The block/stack probes apply block targets only; the denoiser and sampler apply
both fixed and block targets through `MLXDenoiserWeights`. Compatible 2.3 factors are retained.
General IC/control adapters and conditioning beyond first-frame/FFLF, one-driver A2V,
experimental after-extension/motion continuation and first-frame/audio scenes require
separate execution contracts. The experimental direct-worker MSR contract accepts
one to five ordered images through the installed rank-128 task adapter; Studio's MSR
controls and ComfyUI export have not yet switched to that Swift route.

## Build and verify

Requires Apple Silicon, a Metal device, Swift 6.3+, and Xcode's Metal Toolchain component.
If that component is missing, Xcode reports how to install it; the build script does not install
or change the developer environment. MLX Swift is pinned to immutable upstream revision
`901941965d82e4a216d4d117231d847d194c563d`, whose MLX submodule is core **0.32.2**
(`1f8e74e3f12f31365464a6867c6579f0e9b29d85`). This is an unreleased Swift revision.
Upstream's Swift package still defines the runtime version string as `0.32.0`; source identity
and the raw runtime string must be distinguished in measurements.

```bash
bash integrations/weetodd-mlx/scripts/test.sh
```

SwiftPM does not compile MLX's Metal resources. The helper builds the required non-JIT kernels
from the exact resolved dependency with fast math disabled and places `mlx.metallib` beside the
CLI and XCTest binaries. Remaining kernels use MLX's own JIT. Studio's packager bundles the
workers and this library; qualification on the minimum supported OS and a clean Mac remains work.

MLX and NNC's metal-cpp implementations conflict when linked in one test executable. They have
separate packages/executables; shared products include `TensorIO`, `LTX25Engine`, `AdapterRuntime`, `LTX25Text`,
`LTX25Video`, `LTX25Audio` and `InferenceMedia`, all independent of NNC. The model is serial and process-local; MLX cache settings are process-global, so
concurrent weighted jobs must not share this execution context.

The test suite covers packed projection math, LoRA scaling/order, an independent Python-MLX
block fixture, nonzero factorized LoRA versus independent dense fusion, buffered reads across
window boundaries, admission, release, cancellation during loading and between blocks, and retry.
It also covers full-denoiser and three-step ancestral numerical fixtures, all fixed LoRA targets
against independent CPU dense fusion, header-only manifest failures, one-time schedule preparation,
preview failures and retry. Text tests cover independent Gemma/connector fixtures, row-sliced
weights, memory admission, mid-load cancellation, prompt handoff and mutation isolation. Synthetic fixtures do not establish production media quality.
Python is used only by the optional fixture exporter, not by the Swift tests or inference.

Component developer tools support installed-checkpoint qualification:

```bash
.build/release/WeeToddMLXBlockProbe CONFIG INPUTS EXPECTED PAGE BLOCK_INDEX REPORT
.build/release/WeeToddMLXStackProbe CONFIG PAGED_ROOT BLOCKS REPORT [LORAS.json]
.build/release/WeeToddMLXDenoiserProbe denoiser|sampling FIXTURE_PREFIX PAGED_ROOT REPEATS REPORT [CPU_FIXTURE_PREFIX]
```

Paths above are relative to this package. The first compares installed block output with supplied
binary reference tensors. The second exercises actual weights with fixed synthetic activations;
it reports physical process footprint separately from MLX allocation. The third compares a complete
denoiser or trajectory against saved reference fixtures and reports process footprint and repeat
agreement. None renders media. Fixture prefixes resolve `.config.json`, `.inputs.safetensors` and
`.expected.safetensors`; sampling also uses `.schedule.json` and explicit `.noise.safetensors` for
ancestral steps. Sampling additionally requires the independently saved CPU fixture prefix and
both `.steps.safetensors` trajectories. It verifies matching inputs/noise and uses the existing
Float32 per-step gate: `max(component floor, 2 × CPU/GPU reference spread)`. Strict component
results remain reported separately; the calibration does not use the candidate output to set its limits.

Installed text qualification is opt-in through the Swift test suite. Set
`WEETODD_MLX_GEMMA_ROOT`, `WEETODD_MLX_CONNECTOR`, and `WEETODD_MLX_TEXT_REFERENCE`;
optionally provide `WEETODD_MLX_TEXT_PROMPT_FILE` (default: `A red fox.`). The saved reference
must contain matching `token_ids`, `video`, and `audio` tensors. The frozen full-text limits
are maximum absolute error 0.001 and relative L2 error 0.0001; exact token IDs are required.
The test reports encoding time, MLX allocation peak and physical process footprint separately.
Additional `WEETODD_MLX_TRANSFORMER_ROOT` and `WEETODD_MLX_LATENT_FIXTURE` opt into
prompt-to-sampler qualification using five video/three audio latent tokens and all 48 blocks;
this compares generated contexts with independently saved text contexts, not generated media.

## Developer media request

```bash
.build/release/WeeToddMLXPipelineProbe preflight REQUEST.json
.build/release/WeeToddMLXPipelineProbe render REQUEST.json /absolute/path/to/ffmpeg
```

Both commands accept trailing `--video-activation-mib N` (default 512) and
`--transformer-activation-mib N` (default 2048). Each explicit developer allowance is limited to
1–32768 MiB. Both transformer resolutions, reference modulation expansion and the decoder’s
computed workspace must fit before model payloads load. These are component admission bounds,
not peak-process-memory limits. Larger allowances do not change defaults, unload policy, geometry,
or sampling settings. The report records both allowances and the decoder’s admitted bytes.

`--video-decoder mlx` selects the BF16 MLX depth-window decoder; `mps`
remains the default. Each convolution evaluates bounded output-time windows with temporal
replication and zero spatial padding. BF16 windows remain on the GPU and use a stage-scoped
2 GiB allocator cache and 1 GiB convolution workspace limit, restored on every exit. MLX admission
includes its own activation, dtype-promotion and pinned convolution-workspace estimates,
which differ from MPS. Neither estimate is an operating-system process-memory cap.

`--save-latents` additionally writes `latents.safetensors` inside the completed job directory.
It contains Float32 decoder-layout `video` (BCFHW) and `audio` (CTF) tensors and timing metadata.
Capture remains opt-in, is published with the rest of the job, and is cleaned up on failure or
cancellation. These small frozen inputs allow decoder experiments without repeating sampling:

```bash
.build/release/WeeToddMLXDecodeProbe mlx-video /absolute/video-vae.safetensors \
  /absolute/job/latents.safetensors /absolute/new-report.json 12288
```

Use `mps-video` for the matching legacy decoder measurement. The report separates decode-only
timing and process/MLX memory; no PNG encoding, text or sampling executes. An optional `output`
tensor in a small oracle fixture enables numerical comparison. Set `WEETODD_TRACE_VIDEO_MEMORY=1`
for per-layer MLX memory records. Existing reports are never overwritten.
`mlx-video` retains the original Float32 diagnostic path; `mlx-video-bf16` selects the pipeline's
BF16 arithmetic. An optional final cache allowance (0–2048 MiB) supports decoder-only experiments;
allocator cache is reported separately from activation admission.

`--audio-decoder mlx` selects the opt-in Float32 MLX audio VAE, BigVGAN and bandwidth-extension
path; `mps` remains the default. It retains evaluated GPU activations between layers and reports
vocoder-stage progress, while preserving the trained padding, filters and 48 kHz stereo timing.
Audio has a separate conservative admission estimate and the pipeline's existing 2 GiB allowance.
The selected backend and estimate are recorded in `report.json`.

The decoder probe also accepts `mps-audio` and `mlx-audio`, using the captured `audio` tensor.
Pass 256–8192 MiB as the audio workspace allowance; inadmissible geometry fails before weights load.
It publishes a new JSON report plus sibling `.f32` channel-major samples and `.wav` audio files.
Decode timing excludes file loading and waveform publication. Existing outputs are not overwritten.
A 1344 × 768, 89-frame comparison can be explicitly admitted with 12288 MiB for each stage;
production memory optimization and longer temporal windows remain separate work.

`MLXDistilledRequest` accepts only a bounded regular JSON file. Version 1 retains exactly the T2V
fields: `version`, `engine` (`ltx25`), `task` (`t2v`, synchronized generated audio), `gemma_root`,
`transformer_root`, `connector_checkpoint`, `video_checkpoint`, `audio_checkpoint`,
`spatial_upscaler_checkpoint`, `prompt`, `width`, `height`, `frames`, `fps`, `seed`,
`output_directory`, `stage_one_loras` and `stage_two_loras`. All paths are absolute.
Each ordered adapter uses `path`, `strength` and optional `enabled`; use explicit empty arrays
for no adapters. Output must not exist. Unsupported fields/tasks fail before weighted inference.

Version 2 adds the required `reference_images` array and permits `t2v`, `i2v` and `fflf` tasks.
Use an empty array for T2V, exactly one `first` entry for I2V, and ordered `first`, `last` entries
for FFLF. Each entry has exactly `role`, absolute `path`, `strength` (0–1) and `crf` (0–51).
CRF is explicit; 0 skips the training-style codec round trip. Input files are limited to 64 MiB,
8192 pixels per side and 32 megapixels, with separate resize/encoder memory limits. Pixel preparation
is native and does not claim bitwise Pillow preprocessing equivalence. The header-only `preflight`
command checks image metadata; `render` completes all pixel preparation before loading text weights.

Version 3 additionally requires `noise_policy`: `native_box_muller_v1` preserves the previous
stream; `mlx_threefry_bf16_v1` uses explicit MLX Threefry keys, independent stage-one video/audio
seeds, ordered ancestral key splitting, the released stage-two seed/blend rules and BF16 state
boundaries. Versions 1 and 2 retain their original native stream. The policy is recorded in the
output; neither policy claims bitwise whole-pipeline equality across numerical backends.

Preflight validates geometry, decoder budgets and component headers without reading model payloads.
The current whole-context decoder budget admits the 512 × 256, 33-frame integration test but rejects
1344 × 768, 145 frames. This is an admission limit to solve with bounded decoding, not a supported
production size. JSON stdout reports stages and completion counts. The output contains `render.mp4`,
`audio.wav`, `request.json`, `report.json` and `mux.log`. MLX decoding streams GPU-converted RGB24
frames to FFmpeg; `--save-frames` retains the diagnostic PNG route and `frames/` directory.
The MPS route still uses PNGs. Reported process memory covers
Swift, excluding the external muxer. FFmpeg bundling, codecs and production distribution remain work.

Installed two-stage sampling tests use `WEETODD_MLX_TWO_STAGE_REQUEST` and
`WEETODD_MLX_TWO_STAGE_TEXT` (saved matching connector tensors). Optional
`WEETODD_MLX_TWO_STAGE_OUTPUT` writes raw video/audio latents under that prefix.
The 512 × 256, 33-frame base-only integration completed all 11 evaluations in 73.30 seconds,
with 1.48 GB peak physical process footprint. That excludes text encoding and final media decoding.
The full developer clip completed in 96.13 seconds with 1.59 GB peak Swift-process footprint,
producing 33 video frames at 24 fps and 65,760 samples of 48 kHz stereo audio. The muxer process is
excluded from that memory figure. Inspected frames and stream metadata establish a small integration
check, not production-size performance or perceptual audio qualification.

The separate 448 × 256, 33-frame Beowulf FFLF integration took 110.03 seconds and 1.78 GB peak
Swift-process footprint, excluding FFmpeg. Stream counts and frame inspection confirm the complete
reference-conditioned path. Its 1.375-second duration is too short for the unchanged cup-theft prompt;
motion assessment requires a longer test. This is not a comparison to the original 1344 × 768 generation.
The earlier small cup renders predate the post-noise anchor-restoration correction. Its regression
fixtures now cross-check against the working Python denoise loop; the full-size matched path now has explicit
workspace admission and versioned MLX RNG compatibility. A development fixture exported through
the installed production helpers checks initial/ancestral noise and both refinement blends.
Masked trajectory fixtures also execute the actual `X0Model` wrapper and production sampler,
including the BF16 clean-prediction boundary, fractional masks and post-noise anchor restoration.
Remaining Float32 encoder/transformer arithmetic and native pixel preparation can still change
output; matching a seed is not a claim of bitwise whole-pipeline parity.
The matched 1344 × 768, 89-frame Beowulf recipe (seed 43, saved prompt/references, 8+3 steps)
completed in 332.46 seconds with 10.37 GB peak Swift-process footprint and 7.49 GB peak MLX
allocation, excluding FFmpeg. Both component workspace allowances were explicitly 12288 MiB;
the decoder admitted 8,856,170,496 activation bytes. Output has 89 frames at 24 fps and 177,120
48 kHz stereo samples. Inspected frames closely follow the original cup-and-hand exit, but output
is not pixel-identical. Stage times: text 7.38 s, reference prepare/encode 4.52 s, sampling 202.49 s,
upscaling 2.90 s, video decode/publication 85.16 s, audio decode 29.52 s, mux 0.46 s.
The historical original job completed in 180.50 s (180.23 s inside the pipeline); native decoding was the largest measured time
regression. This is one matched recipe, not general production qualification or a default change.

The opt-in MLX video decoder subsequently completed the same full recipe in 293.10 seconds,
including 46.74 seconds for video decode/PNG publication, but its initial process peak was
12.06 GB. Decode-only memory tracing isolated full-activation finite-check temporaries; bounded
checks reduced standalone process peak from 12.41 to 9.84 GB with bit-identical Float32 pixels.
Final matched frozen-latent measurements are **40.69 s / 9.84 GB** for MLX versus
**81.50 s / 9.92 GB** for MPS. Both exclude PNG encoding, sampling and text; full-pipeline timing
was not repeated after the diagnostic-memory fix. The initial integrated output's audio WAV is
identical to the accepted MPS-render WAV. This does not change the developer default or qualify
all video geometries. Audio decoding took approximately 29.4 seconds in that video-only experiment.

The subsequent audio-only comparison uses the same frozen Beowulf latent: **5.67 s / 0.88 GB**
for MLX versus **28.21 s / 0.75 GB** for MPS. Peak MLX allocation is 0.415 GB. The roughly
129 MB standalone-process increase is an explicit tradeoff for the 5× speedup, below the video
stage's peak; no whole-job memory or total-speed measurement is inferred from these separate runs.
All 354,240 Float32 samples pass maximum difference 0.00001574 and RMSE 0.000001385 against MPS.
Each channel retains 177,120 samples at 48 kHz. The independent explicit-Hann oracle passes
maximum error 0.000000455 and RMSE 0.0000000854 against unchanged 0.003/0.0005 limits.
Set `WEETODD_MLX_AUDIO_VAE` and `WEETODD_MLX_AUDIO_ORACLE` to opt into this installed test;
the oracle JSON contains `frames`, `latent` and channel-major `waveform` arrays. Primitive tests
cover causal padding, dilation, transpose convolution, upsampling and replicated filter boundaries.

A historical **complete** matched run with both MLX decoders and buffered transformer reads
finished in **259.41 s**, following a **332.46 s** Swift run. Peak process footprint
was **10.31 GB** versus **10.37 GB** (FFmpeg excluded); peak MLX allocation was 8.88 GB. Sampling was
191.73 s (stage one 80.90 s, stage two 110.83 s), video decoding/PNG publication 46.71 s, and audio
decoding/publication 4.57 s. Final audiovisual latent payloads are bit-identical to the earlier
accepted MLX-video run; all 89 PNGs match and the WAV matches the standalone qualified MLX output.
The earlier run sampled in 201.50 s. The historical Python job was faster at 180.50 s (180.23 s inside its pipeline).
No swap was used; this is one matched run, not a claim for other geometries or machines.

That measurement used bounded buffered I/O. Each tensor and Q8 companion retained its stored
dtype; temporary storage is one packed tensor plus a 4 MiB read window, released after MLX copies
its bytes. Other weight callers retain mapped access by default. An eight-block matched diagnostic
reduced warm loading from 0.676–0.680 s to 0.506–0.507 s with identical output hashes, at a standalone
process-peak cost of about 230 MB. The complete result above includes that tradeoff. File identity
checks, cancellation and one-block staged release remain active.

With native text and fixed-parameter loading, three fixed-binary confirmations measured
**170.22 / 170.64 / 171.59 s internally**, median **170.64 s**. All three beat the original
**180.50 s** target; the matched-recipe performance gate passes. On the common external
stopwatch, Swift's median was **171.85 s** versus **182.15 s** for one fresh Python control.
The Python value is a single observation, not a repeated median. Maximum Swift physical
footprint was **8.854 GB** (FFmpeg excluded), maximum MLX allocation **5.903 GB**, with no swap.
Earlier candidate and repeat results remain recorded; none is substituted for the target.
The full end-to-end gain is not assigned solely to fixed loading, since the fresh Python
control also improved under the later conditions.

Native text loading completes one layer's reads together and preserves bounded embedding
and aggregation row reads. Fixed projections now reuse the validated native reader and
materialize one parameter before computation. Both retain source checks before and after
read completion. Fixed read profiling measured **0.634–0.637 s** mapped versus
**0.068–0.070 s** native over the same **855 MB**, with equal MLX peaks. There is no new
fixed-weight residency cache or raised memory limit. Rotary preparation was measured
separately at about 0.37 s and left unchanged.

All final MP4s are byte-identical to the previously qualified Swift clip; this does not claim
Swift/Python video bitwise equality. Core validation passed 2,540 Python / 176 shared Swift /
109 MLX tests (6/10/11 skips), and independent incremental review found no actionable defect.
That performance qualification preceded worker/Studio integration; no app bundle was replaced
for those three measurements. The subsequent native worker is described below.

Internal clocks differ in setup/publication scope and cannot replace the common external
stopwatch. `stage1_block_load` and `stage1_block_compute` (and stage-two equivalents) are
included in their parent stage totals; do not add them again. Raw-route `video_decode`
includes video encoding and encoder drain.

The saved independent 512 × 288, 33-frame decoder oracle passes maximum absolute error
0.00001031 and RMSE 0.000000644, within the existing 0.01/0.001 limits. Opt in with
`WEETODD_MLX_VIDEO_VAE` and `WEETODD_MLX_VIDEO_ORACLE`; the latter contains `latent` and
`output` tensors. Tests also cover window arithmetic, shuffle order, allocation lifetime,
bounded finite checks, workspace rejection, early weight rejection and cancellation/retry.

Installed single-image encoder qualification uses `WEETODD_MLX_IMAGE_ENCODER` and
`WEETODD_MLX_IMAGE_REFERENCE` (safetensors containing normalized NHWC `pixels` and packed
`latent` tensors). A 96 × 64 reference passes maximum absolute error 0.001 and relative L2 0.0001
against an independently executed full 3D encoder. Pure Swift fixture tests cover reference packing,
per-token timesteps, masked sampling, close-sigma cache isolation, callback mutation and admission.

## Remaining release work

`WeeToddLTXWorker` adapts resolved Studio recipes to the same qualified pipeline, with strict
control validation, JSON-line progress/results, signal cancellation and atomic publication.
`scripts/build_ltx_worker.py` packages the pinned executable, exact MLX Metal kernels and notices;
the Studio packager verifies their hashes before replacing a bundle. Studio can select this
worker for distilled 8 + 3 step T2V, I2V, FFLF and one-driver A2V, preserving standard LoRA
order and strength.
Video admission and execution share a configuration derived from the validated job geometry.
This fixes the hardcoded 17-latent-frame rejection for longer Studio clips. Audio decoder frame
allowances likewise follow the validated job, including the 20-second endpoint. Existing video,
transformer and audio memory limits remain enforced; duration support is not an unlimited
resolution or memory guarantee.
Actual decoded previews are capped at 640 pixels and approximately one per second; denoising
does not run an extra decoder. The worker now derives transformer and decoder allowances from
the shared conservative geometry estimates, including reference tokens and modulation. It
reserves half of physical RAM, respects Metal’s recommended working set, then leaves 4 GiB
for stage weights, bounded caches and media buffers. The existing 32 GiB per-stage engine
ceiling still applies. These are admission bounds, not eager allocations or a total-process
memory cap; live workloads can still affect memory pressure. The fixed 12 GiB worker allowance
incorrectly rejected the user’s 1344 × 768, 241-frame job on a 256 GiB Mac.
Rotary grids now use an explicit allowance inside that transformer workspace, with all four
grids checked during preflight. The former standalone 32-million-element cap also rejected
this job during refinement; standalone callers retain their conservative default.
VAE finite-value checks partition multidimensional tensors before reducing them. The former
whole-tensor flatten failed when a valid activation exceeded Int32.max elements, including
1344 × 768 at 289 frames. Rank-preserving slices retain bounded scratch and inspect every
element, including when a single spatial plane needs further subdivision.

The isolated signed app's worker passed the same 1344 × 768, 89-frame Beowulf recipe with
byte-identical MP4 output, 640 × 365 decoded previews and an 8.810 GB physical process peak
(FFmpeg excluded). Cancellation after the first transformer block exited in about 25 ms and
left no published or partial job. Installed Studio result inspection/late-edit tests passed.
The packaged run measured **186.60 s externally**, versus **180.06 s** for one fresh developer
control. That unresolved 6.54-second difference is mostly in compute subtotals; these two
observations do not establish its cause or packaged speed parity. Keep the earlier developer
repeat median separate. The worker chooses its signed Resources/LTXNative/mlx.metallib explicitly
and fails if it is absent; it cannot silently fall back to a development library in this layout.

StudioCore now performs supported native profile discovery, selection and recipe composition
without Python. It preserves editorial coverage and delegates strict component admission to this
worker. Broader installed adapter/media qualification, advanced controls and complete Python-free
distribution remain. Compatible LTX 2.3 standard
LoRA factors remain supported; specialized controls require separate qualification. The explicit
Python route remains available for those tasks. Do not extrapolate one matched recipe to every
geometry, model or machine.

One installed-checkpoint distilled 8+3 test exercised an ordinary LTX 2.3 standard LoRA at
strength 0.3 in both stages. Its five-second, 768 × 448 Swift take completed in 111.599 worker
seconds, with 3.304 GB peak MLX allocation and 5.449 GB peak worker process footprint, excluding
FFmpeg. The inspected output was distinct and coherent. This verifies that recipe and adapter,
not arbitrary LTX 2.3 LoRAs or a visual-fidelity guarantee.

The Swift A2V route accepts one local mono or stereo `audio_driver` and an explicit source interval.
It encodes the driver for both distilled stages and publishes the selected source interval as audio.
One installed-checkpoint, 512 × 320, 49-frame take at 24 fps completed in 31.82 worker seconds,
with 1.691 GB peak MLX allocation and 3.752 GB peak worker process footprint, excluding FFmpeg.
Its 98,000-sample 48 kHz stereo output WAV matched the expected trimmed/padded source by SHA-256.
Sampled frames showed one coherent speaking person; phoneme-level lip synchronization, longer
clips, multiple audio drivers and production-scale quality remain unqualified.

The experimental LTX Ripple worker also completed two real source-guided tests. The short test
requested three seconds at 768 × 448: 73 model frames yielded 72 editorial frames in 163.34
worker seconds, with 3.086 GB peak MLX allocation and 5.072 GB peak worker process footprint.
The larger test requested five seconds at 1152 × 768: 121 model frames yielded 120 editorial
frames in 865.417 worker seconds, including 722.38 seconds of sampling and 128.38 seconds of
reference encoding; peaks were 9.437 GB MLX and 11.111 GB worker process footprint. Both
movies fully decoded and sampled frames showed a coherent kitten. These two tests do not qualify
other source motions, subjects, timed-anchor fidelity, production memory across hardware or
general visual quality. Process figures exclude the external FFmpeg muxer.
The app-bundled worker reproduced the short direct take byte for byte. Studio then prepared
the same source with an aspect-preserving center crop and completed a separate three-second,
72-frame lifecycle with Python unavailable, live previews, take application and saved-project
reopening. That worker took 151.12 seconds (3.086 GB peak MLX, 5.215 GB worker footprint).
The center-cropped Studio guide differs from the manually resized direct guide, so these
two outputs are not a byte-parity comparison.

H3 FL2VA passed structural route and preflight checks plus one corrected real first/last-frame
recipe: five seconds at 768 × 448, 124 frames, four evaluations and seed 20260927. Sampled
frames showed a coherent front-to-profile turn, with first/last endpoint Pearson correlations
of 0.9903/0.9957. The worker took 511.623 seconds under a concurrent app build; peak MLX
allocation was 4,720,223,400 bytes and peak process footprint was 5,742,871,896 bytes, with
zero system swap. Stereo audio was nearly silent under a quiet-room prompt. This is one
experimental visual recipe, not audible AV quality or broad FL2VA qualification.
The Swift FL2VA recipe and Studio preparation now also admit one to eight uniquely timed
images, including interior keyframes, in ascending frame order. When full-canvas Qwen
visual patches exceed its 1,024-token window, only the Qwen copies are reduced to a
bounded 256-pixel side; the video VAE still encodes full-canvas keyframes. Three-image
and eight-position installed-checkpoint preflights and focused preparation tests pass.
A separate three-keyframe, 768 × 448, 73-frame, four-evaluation turn took 399.77 seconds
and peaked at 3.98 GB Swift process footprint, excluding FFmpeg. The first, interior
and last outputs measured 0.9902/0.9976/0.9956 correlation to their input images.
The requested 2.5 seconds aligned to 73 frames and a 3.05-second muxed movie. Studio
targets the last visible editorial frame; explicit headless `"last"` targets the raw end.
This one short turn does not qualify arbitrary multi-keyframe quality.
The signed app's bundled worker produced a byte-identical MP4 on the same recipe in 552.882
seconds. Its corrected stage report assigns 414.207 seconds to sampling and 119.716 seconds
to video decoding; peak MLX was again 4,720,223,400 bytes with zero system swap. The two
runs had different background workloads, so their elapsed times are not a speed comparison.

Earlier one-, two- and nine-still H3 Ref2VA visual observations used vertically inverted
reference pixels. They remain execution evidence, not likeness qualification. A non-symmetric
still-image fixture exposed the row-order error; after its fix, the same-seed one-image
Beowulf recipe completed again at five seconds, 768 × 448, 124 frames and four evaluations.
It fully decoded and showed coherent boxing, darker hair and beard closer to the portrait,
a clearer face during the jab and visible glove contact with the red bag; the white shirt
stripes were invented. The worker took 750.707 seconds under a concurrent app build, peaked
at 4,889,500,852 bytes MLX allocation and 5,731,812,312 bytes process footprint, and used
zero swap. Its non-silent stereo track measured -29.3 dBFS mean and -2.6 dBFS peak. This
single corrected recipe does not establish broad identity quality or speed parity.

Dependencies retain their licenses: [MLX Swift](https://github.com/ml-explore/mlx-swift/tree/901941965d82e4a216d4d117231d847d194c563d)
and its bundled MLX runtime are MIT licensed. Include their notices when packaging the worker.
Checkpoint and adapter licenses remain separate from WeeTodd source licensing.
