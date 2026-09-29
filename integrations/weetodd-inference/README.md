# WeeTodd Swift inference foundation

This package provides app-owned Swift inference components and a developer LTX 2.5 T2AV
probe with stage-one and two-stage recipes. It does **not** replace Studio's existing render routes.

The ongoing LTX migration now prioritizes [Swift + MLX](../weetodd-mlx/README.md), preserving
packed weights and the established MLX execution strategy without a Python inference dependency.
The NNC component remains a temporary comparison/fallback; its existence is not a measured
performance advantage over the MLX candidate.

- `TensorIO` validates safetensors headers without loading weights. Scoped read-only mappings
  support tensor slices in the original file. Checkpoint files must remain immutable while open.
  Bounded Float32 reconstruction supports F32/F16/BF16 and group-64 affine Q8 pages, one matrix
  or row slice at a time. Q8 reconstruction stays Float32; it is not rounded back to BF16.
- `InferenceContracts` defines versioned progress, preview, output and residency events. The client
  rejects stale jobs, invalid sequences, overlapping weighted stages and oversized preview messages.
- `InferenceHost` provides aggregate stage memory reservations and one-shot worker supervision.
  Reservations are estimates, not measured resident memory or operating-system allocation limits.
  Stage owners must synchronize GPU work and release resources before returning a reservation.
- `AdapterRuntime` validates complete dense LoRA pairs, source names, ranks, alpha and strengths.
  No adapter matrices are copied during inspection. LTX 2.3 metadata remains eligible for structurally
  compatible LTX 2.5 targets. Structural validation does not establish visual or schedule compatibility.
- `LTX25Engine` currently implements the Float32 rectified-flow Euler update on Metal, plus standard
  22B transformer LoRA target validation.
- `LTX25NNC` implements joint audiovisual transformer blocks in Float32, using the pinned
  NNC/ccv Metal backend. Explicit experimental FP16/BF16 projection policies are separate from
  this default. It includes self/text/cross-modal attention, learned QK normalization,
  attention gates, split rotary positions, adaptive normalization and both feed-forward paths.
  Its denoiser accepts one batch of packed audiovisual latents, prepared text embeddings and
  positions at a uniform timestep. It computes timestep/rotary embeddings, eight adaptive heads,
  input projections, all 48 blocks and the normalized output projections to velocity. The stack
  reuses one weighted slot and keeps hidden states on the GPU between blocks. Per-token conditioning,
  masks and guidance remain pending. A shared distilled two-stage sampling runner connects the
  spatial latent upscaler between the two transformer sessions. Text and media components
  are described below. No new Studio backend is
  advertised by this package.

Workers use bounded stdout lines and a bounded stderr tail. Cancellation escalates termination and
waits for the direct worker to exit. Workers must not spawn inference descendants. Parent-death
watchdogs and complete engine process integration remain required before production qualification.
A failed event alone never proves that a worker released its memory.

## Build and test

Run on Apple Silicon macOS with Swift 6 and a Metal device:

```bash
swift test --package-path integrations/weetodd-inference
swift build --package-path integrations/weetodd-inference -c release
```

The core project validation profile also runs these Swift tests. Metal tests report a skip when no
Metal device is available; a skipped test does not qualify GPU execution.

The developer executable performs checkpoint inspection only:

```bash
integrations/weetodd-inference/.build/release/WeeToddInference inspect-checkpoint MODEL.safetensors
integrations/weetodd-inference/.build/release/WeeToddInference inspect-ltx-standard-lora LORA.safetensors
```

The second command checks standard transformer targets. IC, MSR and other reference-conditioned
adapters require dedicated task validation and are rejected by this initial route. The current
Python reference engines and Draw Things provider remain available during migration.

## LTX block qualification

`WeeToddLTXBlockProbe` compares one selected installed block against supplied binary reference
fixtures. Header/shape and decoded-storage budgets are checked before graph compilation. Weights
are decoded and transferred one matrix at a time; no converted checkpoint or full-model copy is
created. Input/weight budgets are preflight limits, not an operating-system cap on GPU intermediates.

```bash
integrations/weetodd-inference/.build/release/WeeToddLTXBlockProbe \
  CONFIG.json INPUTS.safetensors EXPECTED.safetensors CHECKPOINT.safetensors 0 REPORT.json
```

The report includes per-stream maximum/relative error, sampled weight-decoding error, repeatability,
load/forward timing and NNC runtime storage. NNC runtime storage excludes weights and does not
represent process footprint. Measure process footprint separately; CPU RSS omits material GPU memory.

The installed distilled Q8 block 0 passed against the existing MLX block with identical Float32
dequantization and supplied inputs (five video, three audio and four text tokens). Maximum absolute
errors were 0.000094 video and 0.000035 audio; sampled weights matched exactly and repeated outputs
were identical. Peak process footprint was approximately 2.33 GB for this one-block qualification.
These small-token results establish numerical correctness for this case, not full-model visual
quality, generation speed, production memory requirements or Draw Things parity.

The synthetic test fixture contains no trained weights. Its test-only exporter lives in
`Tests/ReferenceTools/export_block_reference.py` and records the MLX reference source hash.
Reference exports may use Python/MLX; Swift execution and normal package tests do not.
NNC and ccv are pinned separately licensed dependencies. Preserve their license notices when
bundling the future generation worker; this package is not yet bundled into Studio.

## GPU-resident stack qualification

`PagedBlockWeights` checks the complete 48-page manifest, architecture and every block's headers
before graph allocation. It retains headers and file handles, not decoded pages. Manifest page
hashes are not verified by this preflight; fixed/top-level weights are outside the stack contract.
The selected page is decoded one matrix at a time into the same GPU block slot. Hidden states stay
on the GPU between blocks; only scalar health checks and progress are read during execution.

`AVStackRunner` releases its owned slot by default and on cancellation or any failure, including a
progress observer error. Explicit `retainWeights` keeps the reusable slot warm until `release()`;
it does not retain all 48 blocks. Allocator/driver residency must still be measured separately.

```bash
integrations/weetodd-inference/.build/release/WeeToddLTXStackProbe \
  CONFIG.json INPUTS.safetensors EXPECTED.safetensors PAGED_ROOT 3 REPORT.json
```

Three installed-Q8 evaluations of the small-token fixture matched the independent MLX stack and
produced identical repeated outputs. Maximum absolute error was 0.00513 video / 0.000245 audio;
relative L2 error was below 0.000002 for both. Allocation remained flat at about 1.55 GB of Metal
buffers and 20 graph variables. Each evaluation uploaded 20 input tensors once and downloaded only
the two final activation tensors. Explicit release reduced Metal allocation below 1 MB; peak
process footprint across the probe was approximately 2.34 GB. These measurements qualify bounded
stack residency, not full-generation quality, production memory or Draw Things performance parity.

Increasing Q8's bounded mapping window from 256 KiB to 4 MiB reduced the same Float32 48-block
probe from approximately 62–64 seconds to 27.5 seconds, with unchanged numerical errors and GPU
allocation. Each window uses at most two additional 256 KiB scale/bias arrays; it does not cache
decoded pages. Weight loading still dominates this tiny-token workload. These are component timings,
not complete video-generation times or a matched comparison with Draw Things.

The independent full-stack exporter is `Tests/ReferenceTools/export_stack_reference.py`. It uses
the existing MLX implementation only as a test oracle; Swift stack execution has no Python stage.

## Packed-latent denoiser qualification

`DenoiserWeights` validates the fixed checkpoint, all 48 block headers and the supported timestep,
position and latent layout before evaluation. It uses installed weights in place. Each adaptive
head is loaded, evaluated and released separately; the block slot is released before output heads
are loaded. The largest fixed matrix expands to 576 MiB in Float32 and has an explicit bounded
read allowance. The fixed file's unused text connectors are not decoded.

`DenoiserRunner` reproduces the reference renderer's BF16 latent/text/timestep input rounding,
then evaluates in Float32. Native rotary preparation uses a Float64 frequency grid and Float32
position arithmetic. Output heads use LayerNorm. CPU arrays bridge preparation, transformer and
output stages; the 48-block sequence itself retains GPU activations. This first implementation
does not cache timestep-independent preparation across denoising steps.

```bash
integrations/weetodd-inference/.build/release/WeeToddLTXDenoiserProbe \
  CONFIG.json INPUTS.safetensors EXPECTED.safetensors PAGED_ROOT 0.731 2 REPORT.json
```

The probe compares both velocity streams, repeated evaluations and stage-boundary Metal allocation.
The synthetic fixture and independent MLX exporter are in `Tests/LTX25NNCTests/Fixtures/` and
`Tests/ReferenceTools/export_denoiser_reference.py`. A velocity evaluation does not encode a prompt,
pack media, sample a complete trajectory, decode media or establish visual/audio quality.

The installed-Q8 small-token fixture initially exceeded the block-level `1e-4` relative tolerance
for video: full velocity relative L2 error was `0.001331` (0.133%), maximum absolute error `0.001079`.
All 48 blocks passed in isolation, as did the separate preparation and output-head checks.
Running the complete unchanged MLX reference on CPU versus GPU gave video relative error `0.001181`
(0.118%) and maximum absolute error `0.000959`. This comparison includes weight reconstruction and
all math stages; it is evidence of reference backend variation, not identical intermediate values.

The full-model **video fixture** limits are therefore separately calibrated and frozen at `0.002`
relative L2 and maximum absolute error. Audio retains `1e-4` for both; repeated outputs must differ
by at most `1e-6`. The report preserves the original strict-tolerance result. These limits do not
change component/isolated-block tests or establish production-precision or media-quality acceptance.
Two complete evaluations produced identical outputs. Peak process footprint was approximately
3.10 GB; transformer-stage Metal allocation stayed near 1.55 GB and returned below 1 MB after each
evaluation. These small-token measurements do not predict production-resolution memory or speed.
Use the exporter's `--device cpu`/`--device gpu` options to reproduce the reference comparison.
`--trace` also exports preparation and block-boundary tensors for the optional installed-model test:

```bash
WEETODD_DENOISER_WEIGHTS=PAGED_ROOT WEETODD_DENOISER_TRACE=FIXTURE_PREFIX \
  swift test --package-path integrations/weetodd-inference -c release --filter InstalledDenoiserTests
```

`WEETODD_DENOISER_TRACE_STACK=1` additionally checks the stack; adding
`WEETODD_DENOISER_TRACE_BLOCKS=1` compares every block in isolation and in sequence. These are
developer diagnostics requiring an explicit local checkpoint and trace, skipped in ordinary tests.


## Multi-step sampling and media components

`LTXSamplingRunner` connects the native denoiser to `EulerTrajectory`. A schedule is
validated in full before inference, including descending sigmas and their positive BF16
representation. Nonterminal ancestral steps consume explicit video noise followed by audio
noise; the terminal step consumes neither. The Float32 latent state and four Metal buffers
per stream are reused. The default job-scoped session reuses one block graph across evaluations and releases it
before the decoder stage. `reuseSession: false` retains evaluation-by-evaluation unloading. Cancellation,
invalid outputs and progress/preview observer failures unwind allocations. This baseline
still transfers packed latent arrays at the CPU/GPU stage boundary.

Progress callbacks report completed steps and weighted sub-stages. The optional preview
callback exposes a completed **latent state**. With session reuse, transformer weights remain
resident during this callback; callers requiring another weighted preview stage must use the
staged path or separately admit that overlap. The callback does not itself
decode a thumbnail or add a Studio live-preview UI; callers must schedule preview decoding.
`AVGeometry` validates dimensions/timing, builds causal token positions and unpacks channel
layouts for both VAEs. Its audio token count follows Comfy's `ceil(duration * 25)` contract.
`GaussianNoise` uses versioned SplitMix64/Box–Muller streams; the same numeric seed is not
claimed to reproduce a Draw Things or MLX random stream.

`LTX25TextEncoder` implements the installed paged-Q8 Gemma4-12B LTX pack, its embedded BPE
tokenizer, all 48 backbone layers, trained hidden-state aggregation and both eight-layer
connectors. Conditioning always has 1,024 tokens, including the trained register tail, even
for a short prompt. Dense/GQA products use Metal Performance Shaders. Attention packing,
masking and softmax execute in one GPU command with one final readback; normalization and
rotary preparation still use the CPU. Hidden states stream into their final interleaved
layout, preserving the original arithmetic. Each linear projection uploads its left operand
once and reuses it across decoded weight slabs, which are limited to 64 MiB.
`TextEncodingPlan` admits simultaneous CPU/GPU buffers for the entire stage after tokenization
and before weighted work. Its default owned-buffer budget is 3 GiB, including aggregation's
large prepared input, outputs, weight slabs, attention, the full 1,024-token connectors and
a 512 MiB reservation for tokenizer/checkpoint metadata. Tokenizer parsing temporaries have
a local autorelease scope before weighted execution.
Driver caches, allocator overhead and mapped-checkpoint residency are additional; this is
not a hard process-footprint cap. Projection scopes release prepared Metal inputs explicitly.
The vocabulary uses exact UTF-8 keys to preserve composed/decomposed forms and leading BOM
characters that Foundation string-key dictionaries can otherwise merge or alter.

`VideoDecoder` implements the released convolutional VAE, streams original checkpoint
weights one layer at a time and emits RGB frame chunks. It performs an exact decode of an
admitted window. Normalization scales are applied while packing bounded convolution windows,
avoiding a full normalized activation copy. Up to 16 windows share one command while reusing
the same workspace. Default shape caps are 17 latent frames and 64×64 spatial positions,
additionally constrained by the unchanged 512 MiB activation estimate. These shape caps are
not a promise that every combination fits: 512×288 with 33 output frames is admitted and
qualified. Over-budget windows fail
preflight; arbitrary scene tiling, halo stitching and DiffVAE are not implemented. RGB
clipping, image/video writing and presentation belong to the caller.

`AudioDecoder` implements the released stereo VAE, vocoder and 48 kHz bandwidth extension.
Its convolutions use bounded MPS matrix products. The default admission caps latent frames
and estimates resident working storage before decoding. Actual output length is
`(4 * latentFrames - 3) * 480` samples/channel. Callers can trim existing samples; the decoder
never pads or stretches audio to hide the causal offset. The internal vocoder's Hann
upsampler uses replicate boundary padding; this is distinct from the generic audio-input
resampler's zero-padding rule.

Python scripts under `Tests/ReferenceTools/` are independent developer oracles, not runtime
bridges. The decoder comparison corrects the installed MLX helper's known differences from
the released vocoder contract. Ordinary tests use small synthetic fixtures. Installed weights
and generated qualification artifacts remain external to the repository. These components
do not establish production-resolution quality, full two-stage distilled generation, complete
LoRA/conditioning coverage, or matched Draw Things speed/memory parity.


### Trajectory qualification and developer probe

The installed eight-step holdout uses an unseen synthetic input/noise seed and the released
stage-one schedule. Both native repeats pass the frozen per-step rule: each absolute and
relative error limit is the greater of its existing single-evaluation floor and twice the
independent MLX CPU/GPU trajectory spread. Floors are video `0.002`, audio `0.0001`.
The report retains strict single-evaluation failures separately; this calibration is limited
to the small-token Float32 fixture. Repeated latent arrays are identical at every step.
The runs took approximately 238 and 245 seconds, peaked at 3.10 GB process footprint and
released weighted Metal storage below 1 MB between evaluations. Weight loading still
limits this workload; these results do not establish Draw Things parity.

The same five-token full text prompt improved from 261 to 40 seconds after increasing the
bounded dense checkpoint mapping window to 4 MiB. Output conditioning remained bitwise
identical. The read-window change adds no persistent weight cache or second raw payload copy.

```bash
swift build --package-path integrations/weetodd-inference -c release
integrations/weetodd-inference/.build/release/WeeToddLTXSamplingProbe \
  GPU_FIXTURE_PREFIX CPU_FIXTURE_PREFIX PAGED_ROOT 2 REPORT.json
integrations/weetodd-inference/.build/release/WeeToddLTXPipelineProbe --preflight REQUEST.json
integrations/weetodd-inference/.build/release/WeeToddLTXPipelineProbe REQUEST.json
```

The pipeline probe requires these JSON fields: `gemma_root`, `transformer_root`,
`connector_checkpoint`, `video_checkpoint`, `audio_checkpoint`, `prompt`, `width`, `height`,
`frames`, `fps`, `seed`, and `output_directory`. All paths are absolute; the output directory
must be new with an existing parent. Headers and memory admission are checked before encoding.
The optional `spatial_upscaler_checkpoint` must name the installed released spatial x2 checkpoint;
when present it selects the two-stage recipe and requires final dimensions divisible by 64.
Explicit null, relative paths and unsupported request keys fail validation.
The probe emits PNG frames, Float32 stereo WAV, conditioning arrays for numerical comparison,
progress JSONL and a report with actual timing.
Without the optional upscaler it retains the legacy fixed eight-step stage-one recipe. With it,
`DistilledSamplingRunner` executes half-resolution stage one, neural spatial upscaling and three
deterministic refinement evaluations at final resolution. Both audio and video are re-noised at
0.909375 before refinement. Unsupported LoRAs, image conditioning and arbitrary schedules still
fail explicitly. This is a developer integration tool; Studio routing remains unfinished.


The first complete native stage-one integration smoke produced nine 64×64 frames at 24 fps
and 17,760 stereo samples/channel at 48 kHz. It took 293 seconds: text 41 seconds, eight
sampling evaluations 244 seconds, video decode 2.5 seconds, audio decode 4.5 seconds.
OS peak process footprint was 3.58 GB; sampled Metal allocation peaked at 1.68 GB. Core
validation began during the last sampling step, so this is an integration timing, not an
isolated benchmark. The separately isolated sampler repeats took 238/245 seconds.
All 112 prompt token IDs matched the independent oracle, and both conditioning streams
passed the frozen full-context gates (maximum absolute error 1e-3 and relative L2 1e-4).
The strict historical 1e-4 absolute gate remains reported separately. Actual audio duration
was 0.370 seconds versus 0.375 seconds of video; no timing compensation was applied.
This tiny smoke does not qualify image identity, perceptual quality or production performance.


### H3-inspired preparation and session improvements

Affine Q8 reconstruction now uses fused SIMD arithmetic with the same bits as the scalar
reference, including offset rows and mapping boundaries. Float32 weight validation classifies
exponent bits in vectors, preserving detection of NaNs and infinities without overflowing a
floating-point reduction. No new model format, permanent weight copy or lower precision is used.

`LTXSamplingRunner` defaults to one internal session per invocation. Each adaptive head loads
once and evaluates the admitted schedule with unchanged row-one arithmetic, then unloads before
the block graph is created. Rotary factors and rounded text are reused; timestep-dependent
conditioning is preserved. The block graph survives between evaluations and releases on success,
failure, cancellation or observer error. Every new invocation has fresh conditioning. Graph build
time is counted only on the first evaluation of that graph.

`maximumPreparationBytes` explicitly opts into one queued CPU block; zero keeps serial loading.
This parameter budgets prepared arrays, not the entire process: input mappings, scale/bias scratch,
GPU weights, activations and graph workspace still require separate admission. Providers used
with prefetch must perform CPU-only reads and remain immutable during the invocation. Queue drain
joins outstanding work before returning; only the inference thread installs GPU weights.
`sequenceAttention` is also opt-in. It orders Q/K/V producers without changing the two shared
pre-cross-modal streams. It has not demonstrated a scratch-memory reduction in the measured shape.

A controlled two-block test at 1,920 video tokens, 35 audio tokens and 1,024 text tokens
(corresponding to 768×512, 33 frames at 24 fps) produced identical output hashes across all tested
policies. Warm scalar-decoder calls took 2.18–2.30 seconds; SIMD took 1.03–1.05 seconds at roughly
the same 3.19 GB peak process footprint. Scheduled SIMD plus prefetch took 0.82–0.83 seconds but
raised peak footprint to 4.36 GB. Prefetch therefore remains optional. These are synthetic-input,
real-weight block tests, not complete video generation or production-quality qualification.

Two clean eight-step small-token session runs took 124.5 and 106.5 seconds, with 3.11 GB peak
process footprint and less than 1 MB Metal allocation after each run. Every audio/video step
matched the staged implementation bit for bit and passed the frozen CPU/GPU-reference gates.
The transformer graph compiled once per run. Earlier staged timings were affected by another
workload and are excluded from speed comparisons; the historical pre-optimization isolated
runs took 238–245 seconds. The clean measurement covers sampling only, not text or media decoding.

Developer shape/policy comparison (scalar metrics and output hashes only):

```bash
integrations/weetodd-inference/.build/release/WeeToddLTXPerformanceProbe \
  CONFIG PAGED_ROOT BLOCKS REPEATS PREPARED_BYTES SEQUENCE_0_OR_1 simd REPORT.json
```

The sampling probe additionally accepts `staged` or `session` after its report path. Reports
include per-step output hashes and separate preparation, installation, compute and health-check
costs. Background preparation durations overlap GPU execution and must not be added to wall time.


### Experimental precision and larger text/video component qualification

Float32 remains the default. `experimentalPrecision` explicitly selects `fp16-projections`
or `bf16-projections`; normalization, attention and residuals stay Float32. FP16 installation
uses checked nearest-even conversion, rejects overflow/nonfinite weights, and includes its
CPU conversion cost in installation timing. No converted checkpoint is persisted.

FP16 passed separately frozen exploratory eight-step limits (video maxabs 0.02, audio 0.005,
relative L2 0.01), with exact repeats, in 114.6/118.3 seconds at 2.38 GB peak footprint.
It did not pass the calibrated Float32 trajectory gate and has no media-quality qualification.
At the 1,920-video/35-audio/1,024-text-token two-block shape, its warm calls took 1.16–1.17 s
at 2.55 GB peak footprint, versus Float32's 1.04–1.05 s at 3.19 GB. This is a memory/speed
tradeoff, not a speed improvement. BF16 failed even the exploratory block gate and remains
an unqualified developer experiment. Reports identify both the selected precision and its
governing gate; historical strict results are retained.

The installed video decoder matched the independent Float32 MLX oracle at **512×288, 33 frames**:
maximum absolute error 0.0000227, RMSE 0.00000128 over 14,598,144 channel values. Decode took
5.83 seconds; measured process peak was 1.05 GB, including the test's reference/output arrays.
Fused normalization and batched windows also match their original separate primitive paths
exactly. This preserves full spatial/temporal context, not independent tile stitching. The
512 MiB activation admission and separate weight admission are not process-footprint caps.

Both the 112-token boxing prompt and a 1,000-token holdout match all tokenizer IDs and pass
the frozen final conditioning gates (maxabs 1e-3, relative L2 1e-4). The final 1,000-token encode
took 58.3 seconds at 3.21 GB process footprint, with bitwise-identical output before/after
the admission and release-scope fixes. The tokenizer parsing scope reduced the observed
peak from 3.27 GB modestly; it did not establish a speed improvement. Short-prompt throughput is
not materially established as faster by the 39.5-second measurement versus the earlier
41-second integration stage. Tests additionally cover nonconstant 1,024-token attention at
128/256/512 head widths, whole-stage budget boundaries, cancellation with live GPU work,
prepared-storage release and clean retry. These checks qualify components, not visual/audio
quality of a production render. The following milestone adds stage two; reference conditioning,
decoded previews, worker integration and Studio selection remain pending; active standard LoRAs are added below.


### Native spatial upscaling and two-stage sampling

`LatentUpscaler` reads the released 1,024-channel spatial x2 checkpoint in place, loading one
convolution layer at a time. It implements zero-padded 3D convolutions, full-volume 32-group
normalization, SiLU residual blocks, per-frame 2D convolution and spatial pixel shuffle.
Video latent statistics denormalize its input and normalize its output. This is separate from
the video VAE's temporal replicate padding and per-pixel RMS normalization. The 512 MiB
activation and 256 MiB single-layer budgets reject oversized full-context work before GPU
allocation. Independent tiles would change group statistics and are not silently substituted.

An installed Float32 reference comparison at packed input `[5,4,8,128]` and output `[5,8,16,128]`
passed the predeclared maxabs 0.002 / relative L2 0.0001 limits: observed maxabs 0.00010425 and
relative L2 0.00002534. It took 2.07 seconds at 0.523 GB peak process footprint. Existing VAE
primitive tests also pass after the shared convolution extension.

`TwoStageTrajectory` owns the recipe and stage ordering; `DistilledSamplingRunner` uses the same
shared sampler at both geometries. Its versioned native RNG has independent initial (`seed`),
ancestral (`seed + 10000`) and refinement (`seed + 2`) streams, with UInt64 wrapping offsets.
Each stream draws video before audio. Stage one uses eight ancestral evaluations; stage two
uses three deterministic evaluations with no additional per-step noise. This does not promise
Torch/MLX/Draw Things seed parity. Legacy one-stage probe requests retain their prior RNG path.

The installed 512×256, 33-frame sampling integration completed all 11 evaluations with one graph
build per stage: stage one 105.31 seconds, upscaling 1.98 seconds, stage two 47.94 seconds.
The process took 155.42 seconds and peaked at 3.387 GB; final Metal allocation was 17.5 MB.
This measurement reuses saved qualified text conditioning and excludes text encoding and final
media decoding. It checks finite outputs, shapes and stage execution, not full-trajectory
numerical parity or perceptual quality. Reference conditioning, decoded previews, parent-death supervision and Studio wiring remain
unimplemented in this route. Active standard LoRAs are described below. No new production backend is advertised.


The installed lifecycle check cancels after stage two has loaded more than 1 GiB of Metal
storage, verifies release below 64 MiB, and retries on the same runner. Retry outputs are
bit-for-bit identical to the independent first successful run in both modalities. Measured
stage boundaries retain 17.5 MB after either transformer and 23.8 MB after the upscaler.
Actual upscaler and sampling runners also reject synchronous reentry. The release suite
passes 164 tests with eight optional installed-model skips; those optional upscaler and
joint-sampling lifecycle checks passed separately with the local checkpoints enabled.

Core validation also passed 2,539 Python tests with six skips, and the installed 512×288×33
video decoder retained its prior numerical error after the shared convolution extension.


### Active ordered standard LoRAs

The developer pipeline accepts an optional `loras` array in selection order. Each entry needs
an absolute local `path`, finite `strength`, and optional `enabled` (default true). At most 16
selections are accepted. Disabled entries do not open files. Enabled zero-strength entries
still validate adapter structure; they do not read matrix payloads or reserve merge scratch.
Negative strengths are supported by the developer contract. Ordinary Studio controls have not
been connected to this route yet.

`LoRAWeightStack` loads one down matrix and a bounded row window of the up matrix, accumulating
`strength * alpha/rank * (B @ A)` directly into the owned, decoded Float32 base matrix through
Accelerate. The exact alpha convention comes from the validated plan; files with baked scaling
are not scaled twice. No full delta matrix, permanent merged weights or entire adapter payload
cache is created. The default 64 MiB admission includes explicit adapter arrays and the raw read
window, separately from base-model storage. BLAS scratch, metadata and OS memory are additional.
Overlapping adapter preparation is rejected. Failed/cancelled matrices are discarded.

The fixed-weight and all 48 block providers apply the same ordered stack during both sampling
stages. All active targets must match the actual model configuration before weights are loaded.
Compatible dense standard LTX 2.3 adapters do not depend on retaining the LTX 2.3 generation
engine. IC/MSR/control adapters require their own conditioning implementation and remain
unsupported here; matching a few tensors does not enable those tasks. The distilled sampler
schedule is unchanged by ordinary LoRAs. Applying distillation adapters to an already distilled
base is an explicit numerical test, not a recommended creative recipe.

The installed 2.3 rank-384 and 2.5 rank-450 adapters passed an independent Float64 reference
comparison for fixed video/audio output projections and video/audio attention projections.
Maximum absolute error was 2.98e-8 (frozen limit 2e-5), relative error at most 1.19e-8 (limit 1e-5).
The two-adapter stack admits 37.37 MB explicit scratch. This component test does not qualify a
particular adapter's style, identity or sampling quality. Default release tests also cover order,
alpha, signed/zero/disabled strengths, F64 payloads, nonfinite results, budget/shape rejection,
file mutation, cancellation/retry and fixed/last-block routing.

A resident-model profile exposed repeated small up-factor mappings as an avoidable cost.
Up factors now read in 2,048-row windows and compute in 1,024-row tiles; the explicit
window admission increases by 3.46 MB for the installed stack. Partial-window tests preserve
exact output, and the installed four-matrix reference errors remain unchanged.

LoRA payloads use bounded buffered reads to avoid repeated VM-map overhead while the model
is resident. Established base-model readers keep their mapped-access default and API. Buffered
reads enforce a 4 MiB raw window and the same file-identity and cancellation checks. Finite
validation reuses the existing SIMD checker. Installed numerical gates still pass unchanged;
the final full-block load/merge check took 1.05 seconds for the two high-rank adapters.


The installed mixed 2.3/2.5 stack completed the full 512x256, 33-frame two-stage sampling check:
stage one 449.91 s, upscaler 1.99 s, stage two 178.86 s; process 631.01 s and 3.438 GB peak footprint,
with no reported swaps. Each stage built one graph; released Metal allocation was 4.93 MB after
stage one, 11.22 MB after upscaling and 17.51 MB after stage two. Both finite latent arrays differ
from the same base recipe. Text encoding and final media decoding are excluded. Compared with
the earlier base-only 155.42 s / 3.387 GB result, two high-rank adapters substantially increase CPU
preparation time while adding about 50.7 MB to measured peak footprint. This qualifies execution
and bounded residency; active-LoRA throughput, creative quality and production-size performance
remain open. These distillation-adapter strengths are test inputs, not creative presets.

The installed cancellation test interrupts after the first adapter-modified transformer block,
then restarts the same runner and reaches weighted work again. Both attempts release Metal
allocation to 4.93 MB. The release suite executes 176 tests with ten optional installed-model
skips; the active-stack matrix, full two-stage and cancellation checks passed separately.
