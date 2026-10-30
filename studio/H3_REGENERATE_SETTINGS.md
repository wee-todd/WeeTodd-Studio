# H3 Regenerate in Swift/MLX

Choose **Upscale Video → H3 Regenerate**, or select **H3 Regenerate** in a clip's **Upscale after generation** controls. The dedicated native H3 worker performs source encoding, learned latent upscaling, the publisher's fixed three-step refinement and the selected video decoder. It uses no Python runtime or conversion environment.

## Native model setup

Choose the shared AI model library, then **Prepare H3 restoration models**. Setup reuses checksum-verified installed components before acquiring missing pinned originals. The Swift worker prepares the original Ref2VA base and publisher F32 encoder; it never invokes a Python converter. Setup creates an ordinary native Ref2VA component profile in Studio's profile directory. Select an existing compatible profile with **Connect existing Ref2VA profile…** when appropriate.

Required components are the original Ref2VA base, paged native Qwen/tokenizer, Q8 video VAE, audio VAE, learned H3 latent upscaler, released Refiner/Turbo/Sharp stack and its separate Sharp timestep coordinates. The optional Fast H3 decoder downloads only when selected. Original and prepared weights stay in the selected library, organized by H3 family; profiles/results contain paths and identities, not copied weights.

The published Qwen package is publicly listed with automatic account gating. Its download requires source access and a Hugging Face read token. **Hugging Face access** in Model setup supports Keychain storage; an existing `HF_TOKEN` or standard Hugging Face login is reused. Credentials stay out of projects, recipes and logs. Installed compatible weights and licensed metadata remain reusable without another download.

The Swift base preparation namespaces the pinned original INT8/ConvRot tensors and casts six fixed F32 projections to BF16. All 1,035 resulting tensor names, dtypes, shapes and payload checksums were compared with the existing compatible native base. Both verified representations are admitted explicitly. A similar-shaped fused Turbo model does not qualify as this base.

The F32 encoder converter reads the two pinned original video-VAE shards and writes the 118 encoder tensors in the native layout without reducing their precision. Conversion uses a private staging output, validates its identity, then publishes exclusively. Cancellation or a changed/partial output never creates a usable model cache. Existing valid outputs can be reused; existing different files remain intact.

```bash
WeeToddCLI models prepare-h3 --model-library MODEL_LIBRARY --profile-directory PROFILE_DIRECTORY --worker H3_WORKER
WeeToddCLI regenerate --request H3.json --output-directory Restored --preflight-only
WeeToddCLI regenerate --request H3.json --output-directory Restored
```

For a command-line build outside the app, pass the model catalog explicitly. See `WeeToddCLI --help` for the exact converter subcommands and arguments. The full packaged app supplies its native helper and Metal library.

## Source and target contract

H3 preserves a **24 FPS** SDR source's selected frame interval and original audio. Other source rates reject before weighted inference. No reference workflow FPS overrides the source. Choose FlashVSR+ for other source rates, or explicitly prepare a separate 24 FPS copy before selecting H3.

The standalone tool processes the complete selected movie. Typed requests can bind `firstFrame` and `frameCount`. Temporal padding repeats the last source frame to the native `5 + 17n` length, and only visible frames are delivered. The limit is 362 padded frames, subject to token, decoder and memory qualification.

Publisher source preparation pads dimensions to the encoder's 32-pixel grid, or explicitly center-crops into that source grid. The learned upscaler then resizes each latent axis to the selected target grid. **These H3 publisher settings can change picture aspect when the grid proportions differ.** They do not have FlashVSR's Fit/Fill delivery semantics. The app shows source/model grids after admission. Target dimensions round upward to the model grid; final output center-crops only the declared target padding.

Direct publisher latent upscaling supports at most two times each axis and does not shrink it. Explicit native multihop is a separate preprocessing profile for larger scale changes; it is recorded and does not claim publisher parity. The source encoder canvas is bounded to one megapixel. Unsupported targets stop rather than selecting another method or decoder.

An explicitly supplied native latent artifact must bind the original source hash, exact interval, padded length, grid, F32 row layout and sampler normalization. A malformed or stale artifact rejects; it never silently falls back to movie encoding. Different model families exchange verified RGB video. H3 latents are never supplied to FlashVSR.

## Sampling and controls

| Control | Behavior |
| --- | --- |
| Preservation prompt | Describe the source subjects, appearance, scene, camera and action that should remain. |
| Seed | Reproducible MLX noise seed; stored with the take. |
| Source encoder | Publisher sampled F32 posterior, with a separate posterior seed; or an explicitly selected native F16 posterior-mean profile. |
| Decoder | Released H3 VAE or Fast H3, selected explicitly. |
| Schedule | Publisher three-step refinement; fixed flow convention, CFG 1 and required adapter stack. |
| Latent upscale | Direct publisher or explicit native multihop. |

The three-step noise schedule is `[0.909375, 6/7, 2/3, 0]`; model time is `1 − sigma`. Encoded silence stays at audio sigma zero while refining video. The delivered soundtrack always comes from the original source. H3 generates no replacement soundtrack for this tool.

The released adapter combines Refiner, folded Turbo and Sharp. Native merging applies each required tensor permutation/scale exactly once, with bounded FP32 projection merges before BF16 execution. Sharp's original 1,025 × 8 timestep coordinates are separate from the native base's different 1,001 × 64 table. LoRA rank or matching tensor shapes alone do not establish compatibility. Extracted rank-384/rank-16 adapters are separate import profiles and do not silently replace the released stack.

No free CFG, denoising-step or generic LoRA-strength sliders imply support that this fixed restoration profile lacks. The ordinary generation engine's settings remain independent.

## Memory and qualification

Weighted stages load and unload sequentially: text, source encoder, learned upscaler, silence encoder, transformer and selected decoder. Failure and cancellation release the active component before another weighted stage starts. Per-stage times, MLX peaks, process footprint and released residency are recorded. Large checkpoint hashing and conversion use bounded host storage.

Publisher F32 full-spatial encoding is substantially heavier than native F16 preprocessing. A measured 1,280 × 736 source reached about 94.2 GB decimal process footprint on the qualification machine; the conservative admission estimate is approximately 100.5 GB. A 32/64 GiB stage budget cannot admit that profile. A native F16 choice is explicit and is not substituted automatically. Publisher tiled-encoder numerical parity remains unqualified.

The transformer uses full-key attention with bounded query windows and FFN chunks. Query splitting does not turn the model into a local-attention approximation. Large-row execution and decoder grids have explicit qualification/indexing gates. A real block or a synthetic decoder run is not proof of complete restoration quality or throughput.

On the qualified M3 Ultra Metal route, native dense execution batches all 56 heads with up to 8,192 queries per fused attention call. It retains the original 256-row output projections. Other devices and unsupported geometries retain conservative execution windows. The worker result records the resolved attention plan, actual hardware route and window sizes.

A matched installed-weight test of one full-size transformer block, with 133,718 packed rows and seeded synthetic activations, improved from **49.75 to 31.11 seconds**. Its complete BF16 output was byte-identical. Peak MLX allocation increased from **11.87 to 14.66 GB**. This is a block measurement, not an end-to-end video speedup. Larger output-projection batches were rejected from this exact path because they changed floating-point rounding.

**Complete five-second 2K restoration remains unqualified.** The ordinary renderer's gate above 64,000 packed rows remains closed. The interrupted long trial and the successful single-block test do not establish a completed full-clip result. Unsupported long/high-row requests reject before weighted inference.

Fast H3 uses whole-clip noncausal context, with required halos for partial output. Its mixed precision, ImageNet normalization and ties-to-even RGB conversion are recorded. Full-context 2K decoder execution was tested independently; that does not establish that every 4K/long-video shape fits or that it is faster than the released decoder on every canvas. Unsupported indexing geometry rejects explicitly.

An October 29 trial on Apple M3 Ultra with 256 GiB unified memory restored 22 frames at 24 FPS (0.917 seconds) to 2,560 × 1,440. The learned latent upscale took **3.97 seconds**, three-step refinement **501.55 seconds**, and released-VAE decoding **85.77 seconds**. The complete released-decoder job took **647.95 seconds**, including preparation and other stages. These are distinct operations: the learned upscale timer alone does not represent a finished video.

Fast H3 decoded those exact saved refined latents in **34.21 seconds**, an observed 2.51× decoder speedup. That Fast trial overlapped another GPU job, so it is not an isolated benchmark or a measured end-to-end speedup. The two decoders showed visible pose, framing and detail differences. Fast H3 remains an explicit decoder choice; this measurement does not establish equal visual quality.

Generative refinement can change identity, action, text or texture. Keep the original and compare the result. The qualification ledger distinguishes source roundtrip, model execution, full-context decode and visual fidelity. Production availability follows the worker's recorded gates; a process success alone does not qualify a new grid.

## Ordered finishing and other frontends

Choose **H3 Regenerate → FlashVSR+** deliberately to add both stages. Enabling finishing alone starts with one FlashVSR stage. Native Pixel/CQ is another independent choice. All selected stages are admitted before generation; the original accepted take remains in Versions.

Version 2 sequences use verified lossless RGB handoff, independent completed-stage caches, original source PCM and one sequence delivery encode. Changing a downstream target/decoder or delivery codec reuses a matching completed prefix. Interrupted stateful stages restart. A changed source/model/worker/importer invalidates affected stages. A late result cannot overwrite a changed document or clip revision.

Studio, Director acceptance, exported native v3 jobs and `WeeToddCLI process` share the same typed sequence executor. Older native v1/v2 jobs and earlier single/two-stage recipes remain readable. The maintained **WeeTodd H3 Regenerate** ComfyUI node translates its controls to the same `regenerate` request; model inference remains in Swift. Portable [UI](../workflows/speed/video-upscale/h3_regenerate_standalone.json) and [API](../examples/h3_regenerate_standalone_api.json) examples require local paths before execution.

The source profile is based on the pinned [Open-H3 Regenerate release](https://nvlabs.github.io/Sana/Open-H3) and [official model repository](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K). Component licenses and distribution limits remain attached to their acquired files. Native integration does not imply CUDA speed/numerical parity or compatibility with every community workflow/checkpoint.
