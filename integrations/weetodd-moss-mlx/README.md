# Native MOSS-Image

WeeTodd Studio runs MOSS-Image Base and SFT in a dedicated Swift/MLX worker on
Apple Silicon. Model setup, tokenization, layout compilation, inference,
quantization and PNG output run natively. The UI process does not load MLX weights.
ComfyUI, MFLUX and Python are not runtime dependencies of this route.

## Studio

Open **Generate Image… → Run with → Local MOSS**, choose **Base** or **SFT**, then
**Link model library…** or **Download…** when needed. Studio automatically finds installed
components in the configured library and known `ai-models` folders on mounted volumes.
Folder selections are remembered across new drafts, projects and app restarts. Explicit
paths take priority, including a temporarily disconnected drive. SFT/Base and BF16/Q8/Q4
transformer selections are kept separate; discovery never changes the generation recipe.
It reads metadata in the background without loading, copying or downloading weights.
Select a folder in your model library to override the automatic selection.
Setup verifies pinned SHA-256 checksums and reuses matching installed files.
One shared Qwen3.5-9B/tokenizer/small-VAE package serves both transformers.
**Component folders** also accepts an existing official transformer folder and
a shared folder containing `text_encoder`, `tokenizer` and `vae`.

Write the scene prompt and use the layout canvas to draw object, character and
text regions. The inspector edits descriptions, exact words and pixel bounds.
Drag a region, resize its handles, duplicate/delete it, or use arrow keys
(Shift for ten pixels). Undo/redo groups each drag. Fit, zoom and the expanded
editor help with larger layouts. Normalized regions scale with output dimensions.
Optional guide images are editor overlays; they are not inference inputs.

The layout actions import official editable drafts, compiled prompts with the
current dimensions, and the supplied Pixaroma boxes/ComfyUI workflow format.
Import reads layout data without executing workflow nodes. Export saves an
versioned WeeTodd editable draft that preserves ownership, analysis roles and provenance;
**Copy MOSS prompt** copies the exact compiled prompt. Official model-facing drafts remain importable.
Every take retains its dimensions, authored/compiled prompt, seed and normalized
worker request. **Generation details… → Open layout as a new draft** preserves the
original take while reopening its composition.
Saved images in the Media panel expose **Open layout as a new draft** in their
context menu, so a collected project can reopen each take without a saved workspace.

Use **Check Settings**, then **Generate Image**. Enabled canvas/mood-board inputs
from another provider must be disabled explicitly. Explicit source reconstruction is
available through the layout workflow below. Control adapters and widening the
transformer input are not supported here.
Dimensions are positive multiples of 16, with sides up to 4096 and at most
4,194,304 pixels. Preflight also checks token and available-memory limits.
Overlong prompts fail instead of truncating their layout silently.

## Layout from an image

Choose **Layout from image…** in the Local MOSS workspace. Select a local image and
a complete installed **MLX Qwen3.5-4B Q4/Q8** or **Gemma4-12B unified Q8** vision model,
then **Analyze image**. A fresh selection prefers installed Gemma; your chosen analyzer
is remembered in Studio's model-folder preferences, including the previous analyzer preference.
The same lookup links MOSS's shared tokenizer before Gemma analysis. If discovery cannot
find the installation, use **Choose model folder…** and **Link model library…**.
The analyzer reuses compatible installed weights; this folder is separate from MOSS's
Qwen3.5-9B generation text encoder. Keep these components in the model library. Analysis runs in
the native worker using Swift MLX and Apple Vision OCR, with no cloud transmission.
Gemma uses its native top-first `box_2d` coordinates for visual and missed-text regions;
the worker converts them into the canvas format once. Qwen and OCR retain their own coordinate contracts.

Review the source overlay, region count and warnings, then choose **Use layout**.
This replaces the current prompt, canvas size and boxes as one undoable edit.
The regular layout inspector remains available for corrections before generation;
**Use layout** does not generate an image. Cancel or a failed analysis keeps the draft.
Changed drafts/documents reject stale proposals.

The canvas targets **2,000,000 pixels total**, with an area tolerance of 2% and
the closest available source aspect ratio on the 16-pixel grid. The whole oriented
source is preserved, including borders. A ratio that cannot reach this area under
the 4096-side limit uses a smaller canvas and reports a warning.

Swift groups detected lettering spatially into at most 32 text boxes, preserving complete
OCR lines across overlapping crops. The vision model supplies up to 40 visual regions,
owned face/hand/clothing/prop details and rich typography descriptions. Independent pose,
detected-face and prominent-heading crops refine visible evidence. Conservative heading
corrections are flagged for review; changing a text box keeps its literal wording authoritative.
Strict JSON, bounded generation, source/model identity,
the compiled prompt budget and cleanup are checked before a proposal can be applied.
Dense groups, small sources, low-confidence readings and removed duplicate lettering
descriptions carry review warnings. Recognition can still misread or miss text.
A separate heading crop helps recognize large ornamental titles. OCR rectangles that
overshoot a crop retain their visible intersection. Extra headings are deduplicated against
matching OCR geometry and split-word evidence without treating a tiny subtitle as the masthead.
The source guide and analysis provenance survive saved-take reopening. Guide pixels
alone remain absent from generation requests. New proposals enable **Enforce subject boxes**
and keep source pixels outside generation. Moving/resizing a whole subject transforms its
owned details together; deleting it also removes those details. Undo/redo preserves the group.

Detected faces become owned detail boxes. Native color measurements can correct a false
color description and identify a uniform outer background; these remain editable descriptions.
Subject-box control independently encodes background, whole subjects with body details,
face details, independent props and lettering. Native regional joint attention limits conditioning to the corresponding
areas while keeping image-to-image attention global. Partitioned queries use fused MLX
attention without a dense quadratic mask or a model pass per box. Up to eight whole subjects
and 8192 combined conditioning tokens across at most 62 streams are admitted. Shared style
uses recognized medium and lighting. Foreground regions keep their own color evidence,
so a colorful title does not colorize a monochrome face. Edit individual descriptions for precise content.
Saved takes restore this setting, and preflight rejects workers that cannot execute it.
Sampling/adapter settings remain explicit. Numeric layout prompts alone provide soft control.

The twelve-image Gemma4-12B qualification verified roughly 2 MP output and real subject-movement and
lettering edits, including generation through Studio without Python. Automatic source matching
remains uneven: faces/skin tones, poses, props, lettering and colors can change or be misread.
Review the source guide and proposed descriptions before generation; this is not qualified
as faithful reconstruction or exact identity retention.
This batch does not qualify Qwen's dense analysis or establish parity with another runtime.

### Legacy source reconstruction

Older saved layouts can retain **Source reconstruction** as an explicit canvas control.
New image analyses use a guide-only source and regional subject-box control. Legacy source
reconstruction supplies a separate, SHA-bound native input and cannot combine with regional
subject-box control. The complete oriented image is resampled
to the requested canvas without cropping. The small FLUX.2 VAE encodes its Gaussian mean,
packs and normalizes the latent, then releases before text conditioning loads.
The sampler starts at `z(s) = (1-s) * source + s * seededNoise`; the selected sigma
schedule is scaled by the actual starting sigma `s`, retaining every requested active
step. Change strength is this sigma, not an unshifted ComfyUI denoise setting.

The **Faithful reconstruction recipe** explicitly selects stock SFT/BF16, disables
adapters and sets 8 release Euler steps, CFG 1 and shift 5. The default change strength
is 0.08. Lower values preserve source likeness while restricting large prompt/box edits.
Source mode currently rejects Base, quantized transformers and active adapters with a
clear admission error. A changed or missing source fails instead of falling back to text.
Saved takes bind source identity and strength; Collect Media copies required snapshots.

This is an experimental native latent-initialization extension, not publisher-trained
image editing support. There are no source overlays, face replacements or painted text
after generation. Small originals are resampled, so missing detail cannot be restored.
Receipts record source/prepared RGB hashes, actual sigmas, latent differences, model
evaluations and stage release. The `vae-roundtrip` diagnostic produces a labeled
zero-evaluation control; it is not a generated reconstruction.

## Recipes, adapters and precision

The publisher baseline is 50 release Euler steps, shift 5, CFG 3 and negative
branch skip layer 22 throughout sampling. CFG 1 omits the negative branch.
The attachment's layout recipe is a separate selection: 25 ER-SDE steps,
Comfy `simple` sigmas, CFG 2.5 and skip layer 22 over progress 0–0.5.
ER-SDE uses the rectified-flow signal coefficient and independent native MLX
seeded noise. A seed does not reproduce PyTorch's random-number sequence.

Link ordinary SafeTensors LoRAs through **LoRAs & Groups**, then add them from
the library. Up to eight distinct adapters retain ordering, enable state and
signed strength; saved groups support Add or Replace. All projection targets,
rank/alpha and shapes must be consumed. Deltas remain separate in FP32 and
source weights stay immutable.

The four-step SFT profile recognizes the exact adapter from
[akhaliq/MOSS-Image-SFT-4-steps](https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps/tree/2ded95c37a029360245926fe72e475e30039bbfb),
`moss_image_sft_4step_v2.0.safetensors`, SHA-256
`154c7f1455c3bf9925945c2f1ea2e4f09d0d0c67256081384a094b2b585f913f`.
Enabling it applies SFT, strength 1, CFG 1, shift 5 and four release Euler steps;
4–8 steps are admitted. The shared **Preset** menu supplies Standard and pinned Fast 4/6/8 recipes; valid six/eight-step edits are preserved while Fast stays enabled. **Update…** reviews all saved/current changes before an explicit overwrite, or use **Save as new…**. See the [Studio preset guide](../../studio/README.md#models-loras-and-generation-presets). Disabling restores the saved stock recipe. Other files,
including older auxiliary data-space/time-embedding recipes, cannot impersonate
this profile by filename.

**Transformer precision** can select a separately converted native affine Q8/Q4
component. Choose **Convert official transformer…** and a new folder in the model
library. Conversion keeps FP32 scales/biases, records source and parameter hashes,
and preserves modulation, timestep, norms and input/output projections. Qwen's
decay gates stay unquantized. These are explicit quality tradeoffs; direct
ConvRot INT8 and W4A8 GPTQ layouts are not admitted as native affine weights.

Weighted stages unload in order: optional source VAE encoder, Qwen, transformer,
small VAE decoder. Streaming is the
default; explicit 2/4/8/16 GiB budgets retain a fixed transformer-block prefix.
No cross-step approximate caches or H3/LTX attention optimizations are enabled.
The VAE uses row bands with global GroupNorm statistics and full-context K/V.
Cancellation/failure releases the active stage and never publishes a partial PNG.

On the qualified M3 Ultra with 256 GB, a 16 GiB budget holds all 42 transformer
blocks. The 512-square, 50-step SFT test took 69.34 and 66.11 seconds in fresh
processes, versus 411.10 seconds with streaming. PNG bytes and conditioning,
noise, first prediction and final latent payloads matched exactly. Smaller
2/4/8/16 GiB budgets also matched exactly in four-step dense cache probes.
These are filesystem-warm/uncontrolled-cache measurements, not disk-cold or
persistent-engine timings. Use the larger budget when memory admission allows it;
the component still unloads before VAE decoding.

## Build and CLI

The standalone worker build/distribution entry point is Swift:

```sh
swift scripts/build_moss_worker.swift --root "$PWD" --configuration release
```

Run model-free native tests with
`bash integrations/weetodd-moss-mlx/scripts/test.sh`; it builds and installs the
matching pinned Metal resources beside the test executable.

The existing app packager includes the worker, matching pinned Metal kernels,
dependency notices and a verified manifest. Its Python build orchestration does
not become a MOSS runtime dependency. The macOS 14 product floor is retained.

The worker accepts `preflight`, `inspect-model` and `render --request request.json`.
`analyze-layout --request analysis-request.json --output proposal-directory` uses the
versioned `MOSSLayoutAnalysisRequest` contract and writes a source snapshot, OCR
evidence, editable draft, compiled prompt and bound analysis receipt. Studio creates
these requests and reviews the result without requiring JSON editing.
For Gemma analysis, the request also supplies `mossTokenizerDirectory` from the MOSS
shared components. Final layout budgeting always uses MOSS's compatible Qwen vocabulary;
the vision analyzer's tokenizer handles its own image/chat input only. Both tokenizer
identity and analyzer family are bound to the reviewed proposal.
`--output take-directory` produces `image.png` and its receipt; `--capture directory`
saves conditioning, initial noise, first prediction and final latents for diagnosis.
Optional `--profile` records synchronized operator groups for tensor IO/upload,
Qwen attention/GDN, transformer projections/refinement/norms/SiTU, adapter deltas
and VAE convolution/norm/attention. These nested groups overlap and introduce
synchronization overhead; use ordinary runs for latency comparisons.
The versioned request is defined in `MOSSImageRequest.swift`; Studio generates it
without requiring users to edit JSON. `SIGINT` cancels a job.

Native conversion also supports a shared text encoder:

```sh
WeeToddMOSSImageWorker convert-model --input "$MOSS_LIBRARY/Shared/text_encoder" \
  --output "$MOSS_LIBRARY/Native-Q8/Shared/text_encoder" --bits 8 --component text_encoder
```

Link that shared folder after placing links to the original `tokenizer` and `vae`
beside it. Keep all original/derived model files and adapters in the model library.

## Qualification

Model-free tests cover schedules, injected-noise ER-SDE, CFG/skip intervals,
tensor admission/identity, Qwen math, a tiny DiT, FP32 adapters, affine conversion,
VAE global statistics/halos and PNG/lifecycle behavior. Separate layout and Studio
tests cover interchange, normalized editing, persistence and direct native routing.

Initial complete native SFT runs produced a coherent 512-square cup with both
the 50-step baseline and the pinned four-step adapter. The supplied 896×1152 serum
layout produced the requested bottle placement and both text lines with ER-SDE.
These functional checks do not establish universal typography, model quality or
speed, nor numerical parity with the publisher's PyTorch pipeline. Additional
qualification results belong in the dated status record.

The controlled 896×1152 serum comparison used the same seed, layout and
25-step ER-SDE recipe (empty negative prompt):

| Transformer / Qwen | Worker seconds | Peak MLX GB | Visual observation |
|---|---:|---:|---|
| BF16 / BF16 | 137.00 | 15.96 | Both text lines and central bottle correct |
| Native Q8 / Q8 | 182.99 | 9.93 | Text correct; small detail/composition changes |
| Native Q4 / Q8 | 180.61 | 9.93 | Text correct; larger bottle/texture changes |

GB is decimal. Qwen dominates the two quantized peak measurements, so the
smaller Q4 transformer did not lower the whole-job peak here. These single
fresh-process M3 Ultra trials establish a memory option; BF16 was faster.

## Sources and licenses

Independent implementation researched from
[OpenMOSS](https://github.com/OpenMOSS/MOSS-Image),
[MFLUX](https://github.com/mflux-community/mflux),
[the Swift MOSS port](https://github.com/xocialize/moss-image-swift),
[FLUX.2 VAE Swift](https://github.com/xocialize/flux2-vae-mlx-swift),
and the [ER-SDE paper implementation](https://github.com/QinpengCui/ER-SDE-Solver).
Draw Things and ComfyUI informed research into contracts and memory/sampling;
their GPL implementation code is not incorporated into this worker.
Official MOSS model weights use Apache-2.0 and install separately. Bundled
dependency notices retain their respective licenses.
