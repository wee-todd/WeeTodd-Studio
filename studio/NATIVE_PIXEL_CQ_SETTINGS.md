# Native Pixel Spatial + CQ recipe pass

Enable **Upscale after generation** in an H3 or LTX clip inspector. New finishing starts with one native FlashVSR stage. Select **Native Pixel + CQ → FlashVSR+ Turbo** explicitly for the native 2× pass followed by Turbo/Tiny at 1920 × 1080.
Select **Native Pixel Spatial + CQ · 2×** to use only the native pass. Existing saved FlashVSR recipes retain their selected engine and settings.
The native pass runs in the bundled Swift/MLX LTX worker after the original generation has been accepted and its worker has exited.

## Settings

| Control | Default | Range and behavior |
| --- | --- | --- |
| Pixel Spatial strength | 1.0 | 0.05–2.0; scales the official LTX 2.5 Pixel Spatial x2 IC LoRA |
| CQ strength | 0.5 | 0–2.0; scales CQ Enhancer V2; zero disables CQ |
| Refinement strength | 0.75 | Advanced; controls source revision/noise separately from either LoRA strength |
| Native LTX components | Automatic | Reuses installed distilled model components from native Model Setup profiles |
| Pixel Spatial adapter | Native model profile | Override with an installed compatible adapter using the file picker |
| CQ adapter | Installed LoRA library | Uses one matching CQ V2 file; choose the file explicitly when missing or ambiguous |

Pixel and CQ apply in the same three-evaluation distilled refinement pass. CQ does not add a second render.
The advanced refinement slider covers 0.05–0.9. Higher refinement can increase detail and can change texture, expression or identity.
The worker contract accepts at most 0.909375; saved out-of-range values reject before inference.
First-frame source anchoring uses strength 0.7. The pass uses a fixed preservation prompt and seed 20261028 for reproducible retries.
Generation prompts, generation LoRA stacks and generation seeds remain independent.

The tested native combination is Pixel 1.0, CQ 0.5 and refinement 0.75.
A five-second 672 × 384 source completed at 1344 × 768 in 4m20s on one M3 Ultra, excluding build and delivery assembly.
That measurement comes from the earlier native worker qualification, not an app latency guarantee.
This three-evaluation result has a milder enhancement than the earlier eight-evaluation Dev experiment. These observations do not establish general throughput or quality parity.

## Source timing, dimensions and audio

The pass preserves the complete original movie's frame count and constant frame rate. It never inherits the LTX profile's FPS or timeline FPS.
The native movie contract accepts 1–60 FPS. It requires source dimensions divisible by 32 and produces exactly twice each dimension.
For example, 672 × 384 becomes 1344 × 768. It does not fit or crop automatically to another target.
Unsupported cadence or dimensions stop before weighted execution; choose an explicit source conversion or a different upscaler before retrying.

The delivered version copies the original audio stream without re-encoding and verifies its audio packet hash.
A silent original stays silent. The result is a MOV containing the native video and original audio.
If the source audio codec cannot be copied into that container, the pass fails with the original intact; it does not silently replace or transcode audio.
The worker's internal audio conditioning does not replace the delivered source soundtrack.

The accepted version retains the clip's editorial source-in, duration, volume and scene membership.
Normal final timeline export still uses the movie's explicit finishing settings; the upscaled source version retains its own timing.

## Models and admission

Use native LTX 2.5 distilled transformer pages, Gemma 4 pages, the convolution video VAE, audio VAE and learned spatial x2 model already linked in Model Setup.
The Pixel Spatial adapter can come from a compatible DFR spatial profile without adopting that profile's generation duration or sampler settings.
Select CQ Enhancer V2 from the LoRA library or use **Choose CQ V2…**. Component files are reused in place.
This feature does not download weights or require an optional ComfyUI, Python inference or vpipe runtime.
The FlashVSR stage uses its separate bundled native Swift/MLX worker. Historical explicit vpipe recipes retain that adapter. Both stages must pass setup admission before source generation starts; selecting native-only does not require vpipe.

Before generation, a one-frame synthetic movie request checks model headers, CQ target admission and the current native worker.
It runs preflight only. After generation, the actual source geometry receives its own preflight before any upscale weights execute.
If native model setup is incomplete, generation does not start.

## Retry and saved takes

Use **Upscale accepted take** to run or retry processing on an existing accepted source.
This action does not call H3 or LTX generation. It always uses the original take recorded in Versions, including after a processed version is selected.
Changing either LoRA strength or refinement strength creates a new upscale attempt. Previous versions remain available.
An identical completed request reuses verified output. A damaged or incomplete output gets a new attempt while the prior files remain intact.
Failure and cancellation retain the original generation and do not publish a completed processing checkpoint.

Cache identity binds the original source, both strengths, refinement settings, model identity and native worker/media tools.
Changing processing settings or processing runtime does not invalidate the generation identity.
A late result cannot replace a clip after its project, source, settings or document session changes.
Continuous scenes use the same processing recipe across their members and preserve member ranges.

## Native job export

Native H3 and LTX 2.5 jobs export a v3 envelope for version 2 processing sequences; existing v1/v2 job readers remain supported.
The frozen job includes native upscaler component paths separately from the generation recipes.
Run it with `WeeToddCLI --job JOB --output-directory OUTPUT`; use `--resume` after an interrupted upscale.
Processing settings can change while the original generation checkpoint remains reusable.
Interactive LTX 2.3 and Draw Things takes can use the pass; their legacy headless exporters continue to reject pending processing explicitly.
A version 2 recipe supports ordered H3 Regenerate, FlashVSR and native Pixel/CQ stages. The ordinary controls offer native-only, FlashVSR-only, H3-only, H3→Flash and Pixel/CQ→Flash. The two-stage chain has independent caches: changing Tiny/Full or the Flash target reuses a verified native intermediate. Changing native strengths reruns native processing and its downstream Flash stage while keeping the original generation. Legacy v1 chains copy the original encoded audio packets. Version 2 sequences retain aligned original float PCM and deliver pcm_f32le, verifying decoded sample identity without an additional AAC encode. New sequences exchange verified lossless FFV1 RGB and use a verified lossless MOV bridge when Pixel source inspection requires AVFoundation. Intermediate-only model stages produce no H.264 delivery encode or audio replacement; the sequence preserves original PCM and encodes delivery once after the final stage. Earlier v1 receipts retain their historical behavior.

The [portable chain](../examples/native_pixel_cq_flashvsr_recipe.json) and [native-only recipe fragment](../examples/native_pixel_cq_recipe.json) contains no model paths, FPS override or audio override.
Model selection belongs to local setup and frozen exported jobs.

## Same-size CQ-only enhancement

The native movie worker also accepts `mode: "enhance"` for a separate same-size source-video pass.
It skips both the learned spatial upscaler and Pixel Spatial adapter. The full source video supplies a clean guide at 1× spatial positions and guide strength 1. The requested CQ-only run uses no extra endpoint anchor, so denoise 0.5 applies uniformly to generated video tokens.
Source dimensions must already be on the 32-pixel grid; resizing and cropping reject. Source FPS and frame count remain unchanged.

Denoising/refinement strength and CQ LoRA amount are different controls. The requested test uses `refinement_strength: 0.5` and a CQ V2 refinement adapter with `strength: 1.0` (100%).
This does not change the first native upscale’s defaults of Pixel 1.0, CQ 0.5 and refinement 0.75.
The same-size mode retains three distilled transformer evaluations; it is an optimized native adaptation, not the publisher’s eight-evaluation Dev recipe.
A 1088p same-size guide has more reference tokens than the earlier 768p upscale, so elapsed time must be measured separately.

The current movie engine caps per-stage activation admission at 32 GiB. A full five-second 1920 × 1088 clean-guide pass can exceed that limit. Explicit `chunking: true` with `chunk_frame_megapixel_budget: 150` admits two 60-frame windows for the 120-frame qualification source. Every visible frame is retained; no FPS conversion occurs. The two windows require six total transformer evaluations and a separate join-quality check. Chunking is explicit and does not change either adapter amount or denoising strength.

The measured five-second CQ-only qualification completed in 21m52s at 1920 × 1088/24 FPS. Source VAE encoding took 5m36s and six refinement evaluations took 15m38s. Original audio packets and decoded PCM match, and all 120 frames decode. Sampled join frames are coherent; human quality approval remains separate. This optional extra pass is substantially heavier than the earlier 768p native upscale and is separate from the explicitly selected native-to-Flash sequence.
