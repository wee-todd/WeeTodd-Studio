# WeeTodd Studio headless jobs

The standalone app exports movie and clip jobs for the same shared renderer used by Studio and the
maintained ComfyUI nodes. Choose Draw Things for its supported inference tasks or native H3/LTX for
additional models and conditioning; an exported job retains that explicit backend choice. Supported
model files can be shared in place, but every executing machine must have access to the referenced
components and media. See the [project overview](../../README.md#choose-how-to-generate).

## Draw Things portable jobs

The `drawthings_image.json` and `drawthings_video.json` files show the secret-free portion of a
Studio job v3 export. Select a connection and a model discovered from that exact endpoint, then
export from Studio to create an executable job with its project, runtime, and manifest hash. Model
IDs are endpoint identifiers and paths are local files; relocating either does not rewrite the
other. Credentials are read only at execution through `credentialRef` (the examples use an
environment variable). Never put a token value in a job.

Self-hosted gRPC jobs require their server to remain running with cloud offload disabled and
explicitly confirmed in the exported connection. Direct Cloud jobs do not require the Draw Things
app. DT+ App Bridge generation is blocked because its free-only billing route is unverified. Every
new request refreshes discovery, capability, CU, and billing eligibility before submission. Completed
artifacts are reused by hash with `--resume`. A submission interrupted after it may have reached
the endpoint is recorded as ambiguous and is not automatically retried.

These JSON files are documentation fragments, not directly executable jobs. Direct Cloud requires
verified remaining free requests, explicit PAYG-disabled status, and an estimate strictly below the
current per-job CU limit. Unknown allowance blocks submission. Fixture tests cover the shared job
runner; real Draw Things model and authenticated Cloud generation remain unqualified. See the
[Studio qualification table](../../studio/README.md#headless-jobs-and-qualification).

`drawthings_image_then_video.json` documents the dependency ordering contract. Its
`inputBindings.first` value names the prior image job. After that image completes, the runner
verifies its hash and resolves an absolute `{role, path, sha256, frameIndex, strength}` first-frame
input before video preparation and submission. Other generated conditioning roles are rejected
until their endpoint and helper contracts are implemented.

Studio’s **Model setup** creates validated recipes from existing components without JSON editing.
The [LTX 2.5 distilled Q8 example](ltx25_distilled_q8_t2v.json) also documents the direct format.
Replace every `/REPLACE/WITH/YOUR/MODELS/` value and supply your prompt before use. These placeholders
are portable documentation, not a runtime-ready checkpoint layout.

## Download and create an LTX 2.5 recipe

Run these commands from the WeeTodd Studio checkout with its compatible Python environment. Accept access to
the [prepared model repository](https://huggingface.co/Vayden/LTX-2.5-MLX-Q8-Paged) and sign in with
`hf auth login` before downloading. Studio can instead store a read token in macOS Keychain for
its own setup downloads; that saved token is not a general shell login.

```bash
python scripts/setup_models.py download ltx25-distilled-q8-preconverted \
  --destination /path/to/shared-models
python scripts/setup_models.py scan ltx25-text \
  /path/to/shared-models/ltx25-distilled-q8-preconverted
```

The destination is a parent library folder. Setup creates the named package directory within it.
Optionally add `--existing-root /path/to/ComfyUI/models` to the download command to reuse files with
matching checksums. Keep the whole package, including all pages, manifests and notices.

Create a recipe using its five component paths:

```bash
python scripts/setup_models.py create ltx25-text \
  --component transformer_path=/path/to/shared-models/ltx25-distilled-q8-preconverted/transformer \
  --component text_encoder_path=/path/to/shared-models/ltx25-distilled-q8-preconverted/gemma \
  --component video_vae_path=/path/to/shared-models/ltx25-distilled-q8-preconverted/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
  --component audio_vae_path=/path/to/shared-models/ltx25-distilled-q8-preconverted/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --component spatial_upscaler_path=/path/to/shared-models/ltx25-distilled-q8-preconverted/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --profiles-directory /path/to/model-recipes --memory-mode lower_memory
```

Replace the example paths with yours; quote each entire `KEY=PATH` argument if it contains spaces.
Creation runs component/configuration preflight and returns `recipePath` for a new, uniquely named
file. Import that file into Studio, select it for an LTX 2.5 text clip, and enter the clip prompt.
For direct headless use, edit the generated recipe's `prompt` and desired `config` values, then run
the preflight/render commands below with that file. The lower-memory policy enables supported
savings; it is not a measured 36 GB fit guarantee. Reference/control tasks need their additional
media and compatible adapters; the distilled base package does not include those adapters.

## LTX 2.5 images and ordinary LoRAs

For a first-frame clip, use the same compatible recipe with a timed keyframe:

```json
"conditioning": {
  "version": 1,
  "task": "fflf",
  "inputs": [
    {"id": "first", "kind": "image", "role": "keyframe", "path": "/local/media/first.png", "frame_index": 0, "strength": 1.0}
  ]
}
```

The task name also covers first-only, last-only and first/last clips. For ordinary LTX 2.5
LoRAs, add ordered path/strength pairs inside `components`:

```json
"loras": [["/local/models/style.safetensors", 0.75]]
```

Use compatible transformer LoRAs; MSR and IC controls require their dedicated task fields and
conditioning. Studio's LoRA library writes this same native recipe format. Run preflight again
after changing adapters or media. If an older runner reports `PosixPath is not JSON serializable`
while writing the LoRA result, update the renderer; no model conversion is needed.

## Download H3 components

Use the same compatible Python environment and Hugging Face access described above. For text,
first-frame or first/last-frame clips, download the FL2VA transformer and its support package:

```bash
python scripts/setup_models.py download h3-fl2va-q8-preconverted --destination /path/to/shared-models
python scripts/setup_models.py download h3-fl2va-support --destination /path/to/shared-models
```

For reference clips, choose the genuine Ref2VA pair instead:

```bash
python scripts/setup_models.py download h3-ref2va-q8-preconverted --destination /path/to/shared-models
python scripts/setup_models.py download h3-ref2va-support --destination /path/to/shared-models
```

Both use the shared encoder and video VAE:

```bash
python scripts/setup_models.py download h3-qwen-q8-vision-preconverted --destination /path/to/shared-models
python scripts/setup_models.py download h3-video-vae-q8-preconverted --destination /path/to/shared-models
python scripts/setup_models.py scan h3-reference /path/to/shared-models
```

Use `h3-text` or `h3-image` instead of `h3-reference` when scanning for those tasks. In Studio,
select the corresponding preset and scan the same library. Set **H3 task manifest** to the matching
support package root, **Audio VAE** to its `audio_vae/` folder, and tokenizer/processor to its
`tokenizer/` and `processor/` folders. Select the transformer and Qwen package roots and the
`video_vae_affine_q8.safetensors` file inside the video VAE package. **Create Recipe** validates the
component set; attach required media before full clip preflight.

Keep manifests and support files with their weights. Downloads preserve original model terms and
hash-verify reused files. The FL2VA transformer cannot replace genuine Ref2VA weights. Optional
Turbo LoRAs and control adapters are not part of these base component downloads.

Singularity V2 INT8 can be linked as the transformer in the existing Ref2VA setup preset, using
its matching support components. The base downloads above do not include this separate finetune.
See [H3 INT8 finetunes](../../studio/README.md#h3-int8-finetunes) for the exact checkpoint and
[V2 qualification](../../STATUS.md#october-25-singularity-v2-two-stage-qualification) for the tested
Turbo4 + learned upscale + standard3 recipe. Run preflight with your resolved component paths
before rendering an exported job.

## Native H3 two-stage exports

Enable **Two-stage upscale** in Studio before exporting a native clip or movie job. New selections
use four strength-1 Turbo draft evaluations, the learned H3 latent upscaler, and three standard
refinement evaluations. Link compatible task-specific Turbo and learned weights in Studio; base
component downloads do not include them. Set the exact final width/height and adjust **Second-pass
strength** in **Settings → Basic** as needed; its default is 0.35. The exported job freezes these
choices, resolves the smaller draft, and preserves its audio for the single final decode.

Run the exported native job with the bundled CLI:

```bash
/path/to/WeeTodd.app/Contents/MacOS/WeeToddCLI \
  --job /path/to/H3.weetodd-job.json \
  --output-directory /path/to/new-preflight --preflight-only
```

Omit `--preflight-only` and select a new output directory to render. This is the native Studio/job
contract; the Python `--recipe` commands below retain their existing recipe support. A Turbo draft
uses five schedule points for four evaluations; the independently recorded refinement budget is
three actual evaluations. Custom final dimensions remain subject to H3's grid, edge and complete-row
limits. See the [Studio controls and qualification](../../studio/README.md#current-swift-controls-and-qualification).

## Shared recipe format

- `format`: `weetodd-headless-v2`.
- `engine`: `h3`, `ltx23`, or `ltx25`; selects the shared renderer adapter.
- `candidate`: descriptive identifier used in the run record.
- `components`: engine-specific paths or registered asset references. LTX 2.5 requires transformer,
  Gemma text encoder, video VAE, audio VAE and (for two-stage generation) spatial upscaler paths.
  Q8-paged transformer/text paths identify directories; the other example paths identify files.
- `config`: the selected engine’s generation configuration. The example uses distilled 8+3 sampling,
  staged unloading and Q8 block streaming, 768×512, five seconds, 24 fps. These are a starting point,
  not measured 36 GB qualification. Optional configuration uses the engine’s defaults.
- `prompt`: the generation prompt; Studio replaces it with the clip prompt.
- `conditioning`: `{ "version": 1, "task": "t2v", "inputs": [], "audio_policy": "generated" }`
  for this example. Image/reference/control tasks require their corresponding media inputs.
- H3-only `block_residency`: `checkpoint_default` (default) or `resident`. Resident sampling
  requires `config.memory_mode="normal"` and `config.paging_cache_gb=0`; it retains all transformer
  blocks during sampling and unloads them before decoding. Use only with ample unified memory.
  For paged weights with larger working buffers, keep `block_residency="checkpoint_default"`
  and select `config.memory_mode="normal"`; `"low_memory_bf16"` uses the lower-memory policy.
  `config.projection_backend="auto"` enables the existing hardware-qualified projection selector;
  `"mlx"` selects standard MLX. Results report requested/resolved backend and residency evidence.

Studio resolves app acceleration preferences and clip overrides before export. Its exported native
recipes contain concrete engine configuration, so the command-line runner does not need Studio's
settings or recipe picker. Native H3 ordinary Euler `config.steps` is a schedule-point count: 20
points execute 19 evaluations. Do not apply that conversion to fixed or specialized schedules.

Run from the repository with its Python environment:

```bash
python scripts/render_headless.py --recipe /path/to/edited-recipe.json \
  --output-directory /path/to/new-preflight --preflight-only
python scripts/render_headless.py --recipe /path/to/edited-recipe.json \
  --output-directory /path/to/new-render
```

Output directories must be new. FFmpeg must be available to the renderer (or supplied with the
recipe’s `ffmpeg` field). In Studio, **Import model recipes…** accepts the edited file. Clip
geometry, duration, seed and prompt override the corresponding recipe values. Final media preflight
is authoritative; importing a recipe does not qualify every future clip’s settings.

## LTX 2.3 single-pass distilled 1.1

Use the [single-pass example](ltx23_single_pass_distilled_t2v.json), replacing both component
paths and the prompt. It uses the existing MLX distilled 1.1 bundle directly. Keep its config,
split/quantization metadata, connector, video VAEs, audio VAE and vocoder together. Gemma is a
separate local encoder directory. This route requires no spatial upscaler or Dev-named copy.

Guided setup creates the same contract without editing JSON:

```bash
python scripts/setup_models.py create ltx23-text-single-pass \
  --component model_dir=/path/to/ltx-2.3-mlx-q8 \
  --component gemma_model=/path/to/gemma-3-12b-it-qat-8bit \
  --profiles-directory /path/to/recipes --memory-mode lower_memory
python scripts/render_headless.py --recipe /path/to/recipes/created-recipe.json \
  --output-directory /path/to/new-preflight-directory --preflight-only
python scripts/render_headless.py --recipe /path/to/recipes/created-recipe.json \
  --output-directory /path/to/new-render-directory
```

Each invocation requires a new output directory, including preflight. No model download is
implicit. Setup starts at eight evaluations, Shift 5, CFG 1/STG 0 and zero refinement steps.
`config.stage1_steps` and `config.shift` can be changed; the eight-step/Shift-5 combination is the
matched tested configuration. The recipe is T2V with generated audio only. Studio clip exports
retain these fields and use this same renderer. Physical 36 GB qualification remains open.

## Experimental FastH3 Omni · eight evaluations

[The portable Omni recipe](h3_fasth3_omni_ref2va.json) uses the shared Swift H3 worker's
`fasth3_omni` wrapper and native Ref2VA route. [The ComfyUI API example](../h3_fasth3_omni_swift_api.json)
passes that saved recipe unchanged through the existing `WeeToddSwiftVideoGenerate` node. Native
headless execution uses the same recipe with `scripts/render_headless.py --recipe ...
--swift-worker ... --output-directory ...`; the worker owns preflight and inference. No Python model is loaded
by the recipe-backed host. These files are portable templates, **not runtime-ready recipes**.
Replace every `/REPLACE/WITH/YOUR/` path, compute the actual media SHA256 (the zero digest is a
placeholder), and choose an executable FFmpeg before preflight or rendering. The model-free
`preflight_h3_workflow.py --all-api` sweep validates explicit portable placeholders and reports
`runtime_ready: null`; a saved actual ComfyUI prompt still requires `--workflow` and `--comfy-root`
for installed worker/media admission.

Studio Model setup offers **MiniMax H3 · FastH3 Omni · 8 evaluations · Experimental Swift**.
Link the existing affine Q8 transformer from
`ddalcu/FastH3-Omni-MLX-Serve-8bit@3277ecd1f31f46037f8d51e4882d24139be8d112` and the separate
`fastvideo_inference.json` from
`FastVideo/FastVideo-FastH3-Omni-8-Step-V1@53e0d26a1691aa473b6262d8196a8feb2c297df0`.
The official FastH3-Omni-MLX kit has no weights. Source/checksum verification runs in the background;
compatible installed Qwen/vision, tokenizer and original-architecture VAE components are reused in place.
Use a supported native-format video VAE; qualification uses the existing affine-Q8 conversion
of the original quality VAE with default FP32 decode, not a fast or distilled decoder. Raw
upstream SafeTensors are not automatically converted or silently substituted; unsupported
format/normalization metadata fails preflight.
The optional pinned download creates `FastH3-Omni/h3-fasth3-omni-q8` under the chosen destination
library (or `h3-fasth3-omni-q8` when the chosen folder is already named `FastH3-Omni`). Partial
files and staging stay in that family folder. Keep model LICENSE, NOTICE and MODIFICATIONS metadata;
these weights retain the MiniMax H3 Community License Agreement, separately from product code.

Omni is independent Ref2VA with at least one image or movie appearance reference, in attachment order.
It admits at most **12 total** references: **9 images, 3 videos and 3 audio sources**, where included
movie soundtracks also consume audio slots. The output canvas stays user-selected on a 32-pixel grid,
with each axis up to 2048, a requested duration of 5–15 seconds, and a complete **40,000-row** packed
budget for text, target and media checked by the worker. Media preparation also checks the existing
Qwen visual-pad and RGB budgets; passing canvas checks alone does not establish media admission.
`config.steps=9` records grid nodes and means **8 actual PDD evaluations**, with coarse Euler,
CFG 1 and fixed trained VSA. The profile owns PDD 32, boundaries `[0,4,8,12,16,20,24,28,32]`, grid
maximum 0.999, video/audio shifts 12/3, tile 128, sparsity 0.9, `p2_multi_region` and keep rate 0.1.
Omit `attention_policy`, including explicit `dense`. Still images, anchors/timing, continuation,
refinement, model/sampler/motion controls, extra LoRAs, Preview/VDN/Sol and approximation caches are unsupported.
CPU reference preparation remains explicit: images accept `image_pixel_budget_percent` 50–400;
movies accept `size_policy=match_output|native_h3` and `temporal_density=full` only. These policies
resize down without enlarging smaller originals and do not change the sampling contract. The
example uses an explicit 400% image budget to retain more source detail, subject to Qwen and packed-row limits.
Half, quarter and automatic temporal thinning reject. Omitted preparation fields retain existing defaults.

Selecting the profile keeps authored work. Retained incompatible settings block preparation/export;
**Apply required Omni sampling settings** explicitly clears incompatible sampling, Turbo/two-stage,
refinement, reference-noise and cache overrides. Disable extra LoRAs or remove incompatible input
purposes in their visible controls, and clear timed placement or non-full density explicitly.
Clearing unsupported reference settings preserves valid sizing and soundtrack selections. Prompt, canvas,
duration and ordered attachments survive. This experimental profile remains unqualified for
production; portable schema tests do not establish installed-model generation or creative quality. See the
[measured qualification scope](../../STATUS.md#october-26-experimental-native-fasth3-omni) for actual AV and app-lifecycle checks.
