#!/usr/bin/env python3
"""Host-independent H3/LTX recipe runner; no ComfyUI graph or node imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from render_h3_headless import NoComfyImports, assert_isolated


def assemble_external_extension(
    source,
    generated,
    target,
    *,
    source_frames,
    context_frames,
    fps,
    sample_rate,
    ffmpeg,
):
    """Append a generated window after removing its repeated continuation context."""

    from wee_todd_mlx.conditioning_media import assemble_external_extension as assemble

    return assemble(
        source,
        generated,
        target,
        source_frames=source_frames,
        context_frames=context_frames,
        fps=fps,
        sample_rate=sample_rate,
        ffmpeg=ffmpeg,
    )


def assemble_h3_extension(
    source, generated, target, *, source_frames, context_frames, fps, ffmpeg
):
    """Backward-compatible H3 assembly entry point."""

    return assemble_external_extension(
        source,
        generated,
        target,
        source_frames=source_frames,
        context_frames=context_frames,
        fps=fps,
        sample_rate=32000,
        ffmpeg=ffmpeg,
    )


def validate_h3_adapter_keys(names):
    """Do not partially apply a foreign adapter while silently discarding extra tensors."""
    endings = (
        ".lora_A.turbo.weight",
        ".lora_B.turbo.weight",
        ".lora_A.default.weight",
        ".lora_B.default.weight",
        ".lora_A.weight",
        ".lora_B.weight",
        ".lora_down.weight",
        ".lora_up.weight",
        ".alpha",
        ".diff",
        ".diff_b",
    )
    unknown = [name for name in names if not name.endswith(endings)]
    if unknown:
        raise ValueError(
            "This H3 adapter contains unsupported tensor fields; refusing partial application. "
            "A format-specific converter or runtime implementation is required: "
            + ", ".join(unknown[:5])
        )


def render_h3(recipe, target):
    from wee_todd_mlx.conditioning_media import (
        inspect_media,
        read_embedded_video_audio,
        read_media,
    )
    from wee_todd_mlx.progress import render_progress
    from wee_todd_mlx.task_conditioning import validate_conditioning

    task_report = validate_conditioning(recipe)
    contract = task_report["contract"]
    media_report = inspect_media(recipe, contract)
    from profile_fasth3_server import install_profiling

    from wee_todd_nodes.conditioning import TEXT_ENCODER_RUNTIME, H3TextEncoderSpec
    from wee_todd_nodes.decoding import (
        AUDIO_VAE_RUNTIME,
        VIDEO_VAE_RUNTIME,
        H3AudioVAESpec,
        H3VideoVAESpec,
    )
    from wee_todd_nodes.direct_publishing import publish_latents_direct
    from wee_todd_nodes.preflight import (
        H3ComponentSetSpec,
        H3PreflightRequest,
        preflight_components,
    )
    from wee_todd_nodes.preview import H3PreviewConfig
    from wee_todd_nodes.residency import prepare_low_memory_stage
    from wee_todd_nodes.runtime import RUNTIME, H3GenerationConfig
    from wee_todd_nodes.sampling import TRANSFORMER_RUNTIME, H3TransformerSpec

    fields = dict(recipe["components"])
    preview = fields.pop("preview_override", None)
    fun_controlnet_path = fields.pop("fun_controlnet", None)
    components = H3ComponentSetSpec(
        **fields, preview_override=H3PreviewConfig(**preview) if preview else None
    )
    config = H3GenerationConfig(**recipe["config"])
    config.validate()
    continuation_plan = None
    continuation_report = None
    if "continuation" in recipe:
        from wee_todd_mlx.h3_continuation_artifact import prepare_continuation
        from wee_todd_nodes.continuation import validate_continuation_for_sample

        continuation_plan = prepare_continuation(recipe, task=contract["task"])
        continuation_report = continuation_plan["request"]
        identity = continuation_plan["identity"]
        components = replace(
            components,
            checkpoint=identity["checkpoint"],
            **{name: value["path"] for name, value in identity["components"].items()},
        )
        if continuation_plan["context"] is not None:
            config = replace(
                config, duration_seconds=continuation_report["sample_duration_seconds"]
            )
            config.validate()
            validate_continuation_for_sample(
                continuation_plan["context"], H3TransformerSpec.from_components(components), config
            )
        if continuation_report["save_context"] and target.with_suffix(".continuation").exists():
            raise FileExistsError("Continuation output already exists; choose a new output take")
    block_residency = recipe.get("block_residency", "checkpoint_default")
    config.validate_paging(components.resolved_paths()["transformer"], block_residency)
    preflight_components(
        components,
        H3PreflightRequest(
            duration_seconds=config.duration_seconds,
            steps=config.steps,
            width=config.width,
            height=config.height,
        ),
    )
    runtimes = (
        TEXT_ENCODER_RUNTIME,
        TRANSFORMER_RUNTIME,
        VIDEO_VAE_RUNTIME,
        AUDIO_VAE_RUNTIME,
        RUNTIME,
    )

    sampling_finished = False

    def prepare(stage):
        labels = {
            "text_encoder": "Encoding prompt and references",
            "transformer": "Loading transformer",
            "video_vae": "Decoding video" if sampling_finished else "Encoding images",
            "audio_vae": "Decoding audio" if sampling_finished else "Encoding audio",
        }
        render_progress(stage, labels[stage])
        return prepare_low_memory_stage(stage, config.memory_mode)

    attention = None
    if recipe.get("attention"):
        from minimax_h3_mlx.vsa_h3 import FastH3VSAConfig

        fields = dict(recipe["attention"])
        for name in ("prefix_segments", "video_grid"):
            fields[name] = tuple(fields.get(name, ()))
        attention = FastH3VSAConfig(**fields)
    fastvideo = None
    if recipe.get("fastvideo"):
        from minimax_h3_mlx.fasth3_approx import FastH3ApproximationConfig

        fastvideo = FastH3ApproximationConfig(**recipe["fastvideo"])
    vdn = loras = None
    if recipe.get("loras"):
        from wee_todd_nodes.lora import H3LoRASpec, H3LoRAStack
        from wee_todd_nodes.preflight import read_safetensors_header

        loras = H3LoRAStack()
        for value in recipe["loras"]["adapters"]:
            loras = loras.append(H3LoRASpec(**value))
        loras.validate_for_steps(config.steps)
        for adapter in loras.adapters:
            validate_h3_adapter_keys(read_safetensors_header(Path(adapter.path)).tensor_names)
    if recipe.get("vdn"):
        from wee_todd_nodes.vdn import H3VDNSpec

        vdn = H3VDNSpec(**recipe["vdn"])
        vdn.validate_sampling(config, loras)
    try:
        prepared = None
        images = None
        anchors = None
        continuation = continuation_plan["context"] if continuation_plan else None
        fun_control_spec = None
        fun_control_latent = None
        extension_source = None
        extension_report = None
        if contract["task"] == "extension":
            from minimax_h3_mlx.packing import align_num_frames
            from wee_todd_nodes.conditioning_inputs import h3_external_extension_references

            item = contract["inputs"][0]
            extension_report = media_report[0]
            source_video = read_media(recipe, item, extension_report)
            if len(source_video) != extension_report["num_frames"]:
                raise ValueError("H3 extension decoded frame count differs from ffprobe")
            source_audio = read_embedded_video_audio(recipe, item, extension_report)
            stack = h3_external_extension_references(
                source_video, source_audio, extension_report["fps"]
            )
            prepared = stack.prepare(
                target_width=config.width,
                target_height=config.height,
                target_num_frames=align_num_frames(round(config.duration_seconds * 24)),
            )
            extension_source = item["path"]
        if contract["task"] == "control":
            from minimax_h3_mlx.controlnet import H3FunControlSpec
            from minimax_h3_mlx.packing import align_num_frames
            from wee_todd_nodes.h3_controlnet import prepare_h3_control_frames

            item = next(value for value in contract["inputs"] if value["role"] == "control")
            report = media_report[contract["inputs"].index(item)]
            source = read_media(recipe, item, report)
            prepared_control = prepare_h3_control_frames(
                source,
                target_frames=align_num_frames(round(config.duration_seconds * 24)),
                height=config.height,
                width=config.width,
            )
            fun_control_spec = H3FunControlSpec(
                str(fun_controlnet_path), float(item["strength"])
            )
            fun_control_latent = VIDEO_VAE_RUNTIME.encode_continuation(
                H3VideoVAESpec.from_components(components),
                prepared_control,
                unload_after=True,
                prepare_stage=lambda: prepare("video_vae"),
            )
        if components.task == "ref2va":
            if prepared is None:
                from minimax_h3_mlx.packing import align_num_frames
                from wee_todd_nodes.conditioning_inputs import H3ReferenceInput, H3ReferenceStack

                stack = H3ReferenceStack()
                for item, report in zip(contract["inputs"], media_report, strict=True):
                    soundtrack = None
                    if item.get("soundtrack_path"):
                        soundtrack = read_media(
                            recipe,
                            {
                                "id": item["id"] + "-soundtrack",
                                "kind": "audio",
                                "path": item["soundtrack_path"],
                            },
                            report["soundtrack"],
                        )
                    stack = stack.append(
                        H3ReferenceInput(
                            kind=item["kind"],
                            media=read_media(recipe, item, report),
                            fps=report.get("fps"),
                            target_frame=(
                                0
                                if contract["task"] == "a2v"
                                and item["role"] == "audio_driver"
                                else item.get("frame_index")
                            ),
                            soundtrack=soundtrack,
                        )
                    )
                prepared = stack.prepare(
                    target_width=config.width,
                    target_height=config.height,
                    target_num_frames=align_num_frames(round(config.duration_seconds * 24)),
                )
        elif components.task == "fl2va":
            from PIL import Image

            from minimax_h3_mlx.packing import prepare_keyframe_image

            ordered = sorted(contract["inputs"], key=lambda item: item["frame_index"])
            overlap = continuation_report["overlap_frames"] if continuation_report else 0
            anchors = tuple(item["frame_index"] + overlap for item in ordered)
            images = []
            for item in ordered:
                with Image.open(item["path"]) as source:
                    images.append(
                        prepare_keyframe_image(
                            source.convert("RGB"),
                            config.height,
                            config.width,
                            stretch=item["frame_index"] == 0,
                        )
                    )
        needs_vision = images is not None or (
            prepared is not None and any(reference.kind != "audio" for reference in prepared)
        )
        conditioning = TEXT_ENCODER_RUNTIME.encode(
            H3TextEncoderSpec.from_components(
                components, load_vision=needs_vision
            ),
            recipe["prompt"],
            images=images,
            references=prepared,
            task=components.task,
            unload_after=True,
            prepare_stage=lambda: prepare("text_encoder"),
            cache_directory=recipe.get("cache_directory")
            if prepared is None and images is None
            else None,
        )
        if prepared is not None:
            rows = VIDEO_VAE_RUNTIME.encode_references(
                H3VideoVAESpec.from_components(components),
                prepared,
                unload_after=True,
                prepare_stage=lambda: prepare("video_vae"),
            )
            audio_rows = None
            if any(reference.has_audio for reference in prepared):
                audio_rows = AUDIO_VAE_RUNTIME.encode_references(
                    H3AudioVAESpec.from_components(components),
                    prepared,
                    unload_after=True,
                    prepare_stage=lambda: prepare("audio_vae"),
                )
            conditioning = replace(
                conditioning,
                condition_video_rows=rows,
                condition_audio_rows=audio_rows,
                references=tuple(prepared),
            )
        elif images is not None:
            rows = VIDEO_VAE_RUNTIME.encode_keyframes(
                H3VideoVAESpec.from_components(components),
                images,
                height=config.height,
                width=config.width,
                unload_after=True,
                prepare_stage=lambda: prepare("video_vae"),
            )
            conditioning = replace(
                conditioning, condition_video_rows=rows, keyframe_anchors=anchors
            )
        with install_profiling(target.parent / "raw", "off"):
            latents = TRANSFORMER_RUNTIME.sample(
                H3TransformerSpec.from_components(components),
                conditioning,
                config,
                unload_after=True,
                block_residency=block_residency,
                sol_attention=attention,
                fastvideo=fastvideo,
                vdn=vdn,
                loras=loras,
                continuation=continuation,
                preview_config=components.preview_override,
                prepare_stage=lambda: prepare("transformer"),
                step_callback=lambda done, total: render_progress(
                    "sampling", "Sampling", completed=done, total=total
                ),
                fun_control_spec=fun_control_spec,
                fun_control_latent=fun_control_latent,
            )
        if latents.transformer_evaluations != config.steps - 1:
            raise RuntimeError("Unexpected transformer evaluation count")
        if attention and (
            latents.sol_attention_report.get("executed_calls")
            != (config.steps - 1)
            * (fastvideo.active_layers if fastvideo and fastvideo.active_layers else 50)
            or latents.sol_attention_report.get("fallback_calls") != 0
        ):
            raise RuntimeError("Native VSA execution proof failed")
        # Enforce the stage boundary even for a resident sampling policy.
        TRANSFORMER_RUNTIME.unload()
        sampling_finished = True
        publication = dict(recipe.get("publication", {}))
        if continuation_report:
            if latents.num_frames != continuation_report["generated_frames"]:
                raise ValueError("H3 sampler returned a different continuation frame window")
            publication["metadata_updates"] = lambda: {"continuation": continuation_report}
            if continuation_report["overlap_frames"]:
                from wee_todd_mlx.h3_continuation_artifact import (
                    ContinuationAudioDecoder,
                    ContinuationVideoDecoder,
                )

                publication["video_cache"] = ContinuationVideoDecoder(
                    VIDEO_VAE_RUNTIME, continuation_report
                )
                publication["audio_cache"] = ContinuationAudioDecoder(
                    AUDIO_VAE_RUNTIME, continuation_report
                )
        publish_target = (
            target if extension_source is None else target.with_name("generated-window.mp4")
        )
        result = publish_latents_direct(
            publish_target,
            components,
            latents,
            **publication,
            ffmpeg_path=recipe["ffmpeg"],
            prepare_video_stage=lambda: prepare("video_vae"),
            prepare_audio_stage=lambda: prepare("audio_vae"),
        )
        extension_metadata = None
        video_path = result.video_path
        continuation_artifact = None
        if continuation_report and continuation_report["save_context"]:
            from wee_todd_mlx.h3_continuation_artifact import save_continuation_artifact
            from wee_todd_nodes.continuation import continuation_context_from_latents

            context = continuation_context_from_latents(
                latents, continuation_report["context_frames"]
            )
            with Path(result.video_path).open("rb") as stream:
                movie_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            continuation_artifact = save_continuation_artifact(
                context,
                Path(result.video_path).with_suffix(".continuation"),
                identity=continuation_plan["identity"],
                provenance={
                    "output_take_id": continuation_report.get("output_take_id"),
                    "video": str(result.video_path),
                    "video_sha256": movie_sha256,
                    "source_manifest_sha256": continuation_report.get("source_manifest_sha256"),
                    "generated_frames": continuation_report["generated_frames"],
                    "published_frames": continuation_report["published_frames"],
                    "overlap_frames": continuation_report["overlap_frames"],
                    "tail_trim_frames": continuation_report["tail_trim_frames"],
                },
            )
        if extension_source is not None:
            assert extension_source is not None and extension_report is not None
            assemble_h3_extension(
                extension_source,
                result.video_path,
                target,
                source_frames=extension_report["num_frames"],
                context_frames=0,
                fps=24,
                ffmpeg=recipe["ffmpeg"],
            )
            video_path = target
            extension_metadata = {
                "direction": "after",
                "source_frames": extension_report["num_frames"],
                "context_frames": 0,
                "conditioning": "released_ref2va_source_plus_first_frame_anchor",
                "additional_frames": contract["extension"]["additional_frames"],
                "output_frames": extension_report["num_frames"]
                + contract["extension"]["additional_frames"],
                "audio_policy": contract["audio_policy"],
                "generated_window": str(result.video_path),
            }
        return {
            "video": str(video_path),
            "metadata": {
                **result.metadata,
                "conditioning_task": contract["task"],
                "extension": extension_metadata,
            },
            "evaluations": latents.transformer_evaluations,
            "sampling_seconds": latents.total_seconds,
            "paging": latents.paging_report,
            "block_residency": latents.block_residency_report,
            "projection_backend": latents.projection_backend_report,
            "projection_backend_runtime": latents.projection_backend_runtime,
            "attention": latents.sol_attention_report,
            "vdn": latents.vdn_report,
            "preview": latents.preview_report,
            "conditioning_cache": conditioning.cache_report,
            "conditioning_rows": {
                "video": 0
                if conditioning.condition_video_rows is None
                else int(conditioning.condition_video_rows.shape[0]),
                "audio": 0
                if conditioning.condition_audio_rows is None
                else int(conditioning.condition_audio_rows.shape[0]),
                "keyframe_anchors": list(anchors or ()),
            },
            "conditioning": task_report,
            "consumed_input_ids": task_report["input_ids"],
            **({"continuation_artifact": continuation_artifact} if continuation_report else {}),
            "runtime_loaded": [runtime.loaded for runtime in runtimes],
        }
    finally:
        for runtime in runtimes:
            runtime.unload()



def publish_scene_movie(source, target, *, frames, fps, ffmpeg, source_audio=None):
    """Trim the native extra frame/audio and atomically publish the delivered scene."""
    import os
    import subprocess

    target = Path(target)
    if target.exists():
        raise ValueError("Scene output already exists; choose a new output location.")
    duration = frames / fps
    with tempfile.TemporaryDirectory(prefix=".scene-publish-", dir=target.parent) as work:
        temporary = Path(work) / "scene.mp4"
        inputs = [ffmpeg, "-v", "error", "-i", str(source)]
        audio_stream = "0:a:0"
        if source_audio is not None:
            inputs += ["-ss", str(source_audio["source_start_seconds"]),
                       "-t", str(source_audio["source_duration_seconds"]),
                       "-i", source_audio["path"]]
            audio_stream = "1:a:0"
        subprocess.run(inputs + [
            "-filter_complex",
            f"[0:v:0]trim=end_frame={frames},setpts=PTS-STARTPTS[v];"
            f"[{audio_stream}]atrim=end={duration:.12f},asetpts=PTS-STARTPTS[a]",
            "-map", "[v]", "-map", "[a]", "-r", str(fps), "-fps_mode", "cfr",
            "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p",
            "-c:a", "pcm_s32le" if source_audio is not None else "aac", "-t", f"{duration:.12f}",
            "-movflags", "+faststart",
            "-n", str(temporary),
        ], check=True)
        os.replace(temporary, target)

def render_ltx(recipe, target, *, checkpoint_directory=None):
    from wee_todd_mlx.progress import render_progress
    if recipe.get("loras") and (
        recipe["engine"] == "ltx25" or not isinstance(recipe["loras"], dict)
    ):
        raise ValueError(
            "Invalid adapter declaration; refusing to render without requested adapters"
        )
    from wee_todd_mlx.conditioning_media import inspect_media, ltx_conditioning_kwargs
    from wee_todd_mlx.studio_scene import validate_scene_recipe
    from wee_todd_mlx.task_conditioning import (
        apply_ltx25_msr_prompt_guide,
        validate_conditioning,
        validate_ltx25_control_families,
    )

    scene_report = validate_scene_recipe(recipe)
    task_report = validate_conditioning(recipe)
    contract = task_report["contract"]
    media_report = inspect_media(recipe, contract)
    source_audio_item = next((item for item in contract["inputs"]
                              if scene_report and item["role"] == "audio_driver"), None)
    source_audio_sha256 = None
    if source_audio_item is not None:
        from ltx25_mlx.chain_checkpoints import content_identity

        source_audio_sha256 = content_identity(source_audio_item["path"])
    if recipe["engine"] == "ltx23":
        from ltx23_mlx.ic_lora import recipe_ic_specs
        from ltx23_mlx.lora import recipe_specs
        from ltx23_mlx.runtime import RUNTIME, LTX23GenerationConfig, LTX23ModelSpec

        loras = recipe_specs(recipe)
        fields = dict(recipe["components"])
        fields.pop("loras", None)
        fields.pop("ic_loras", None)
        spec = LTX23ModelSpec(**fields, loras=loras, ic_loras=recipe_ic_specs(recipe))
        config = LTX23GenerationConfig(**recipe["config"])
    else:
        if recipe.get("loras"):
            raise ValueError(
                "LTX 2.5 adapters must be declared under components.loras; "
                "refusing to render without the requested adapters."
            )
        from ltx25_mlx.runtime import RUNTIME, LTX25ComponentSpec, LTX25GenerationConfig

        fields = dict(recipe["components"])
        for name in ("loras", "ic_loras"):
            fields[name] = tuple(tuple(item) for item in fields.get(name, ()))
        spec = LTX25ComponentSpec(**fields)
        if spec.loras:
            from ltx25_mlx.transformer import inspect_ltx25_lora

            for filename, _strength in spec.loras:
                inspect_ltx25_lora(filename)
        fields = dict(recipe["config"])
        fields["stg_blocks"] = tuple(fields.get("stg_blocks", ()))
        config = LTX25GenerationConfig(**fields)
        if any(item["role"] == "control" for item in contract["inputs"]):
            report = spec.validate(
                config.pipeline_mode, require_spatial_upscaler=not config.ic_lora_single_stage
            )
            validate_ltx25_control_families(contract, report)
    conditioning_kwargs = ltx_conditioning_kwargs(recipe, contract, media_report)
    effective_prompt = apply_ltx25_msr_prompt_guide(
        recipe["prompt"], task_report.get("prompt_guide", "")
    )
    ingredients_dir = None
    if recipe["engine"] == "ltx23" and "reference_sheet_input" in conditioning_kwargs:
        from wee_todd_mlx.conditioning_media import materialize_ingredients_video

        ingredients_dir = tempfile.TemporaryDirectory(prefix="weetodd-ltx23-ingredients-")
        item = conditioning_kwargs.pop("reference_sheet_input")
        conditioning_kwargs["control_inputs"] = [
            materialize_ingredients_video(
                item, config, ingredients_dir.name, ffmpeg=recipe.get("ffmpeg")
            )
        ]
    audio_interval_dir = None
    try:
        if "audio_interval_input" in conditioning_kwargs:
            from wee_todd_mlx.conditioning_media import materialize_audio_interval

            interval = conditioning_kwargs.pop("audio_interval_input")
            audio_interval_dir = tempfile.TemporaryDirectory(prefix="weetodd-audio-interval-")
            conditioning_kwargs["audio_path"] = materialize_audio_interval(
                recipe, interval["item"], interval["report"], audio_interval_dir.name)
        extension = recipe["engine"] in {"ltx23", "ltx25"} and contract[
            "task"
        ] == "extension"
        generation_target = target.with_name("generated-window.mp4") if extension else target
        render_progress("encoding", "Loading models and encoding conditioning")
        if scene_report:
            from wee_todd_mlx.conditioning_media import media_binary
            from wee_todd_mlx.studio_scene import scene_window_prompts

            images = None
            if conditioning_kwargs.get("image_inputs"):
                from ltx_pipelines_mlx.utils.args import ImageConditioningInput

                images = [ImageConditioningInput(path=item["path"],
                                                 frame_idx=item["frame_index"],
                                                 strength=item["strength"])
                          for item in conditioning_kwargs["image_inputs"]]
            native_target = target.with_name("scene-native.mp4")

            def scene_progress(done, total):
                if done == total:
                    render_progress("decoding", "Decoding the complete scene and shared audio")
                else:
                    render_progress("sampling", "Sampling continuous scene",
                                    completed=done, total=total)

            result = RUNTIME.generate_chain_to_file(
                spec, config, scene_window_prompts(recipe), native_target,
                window_count=len(recipe["scene"]["segments"]),
                overlap_frames=recipe["scene"]["overlap_frames"],
                boundary_image_policy=recipe["scene"].get("boundary_image_policy", "strict"),
                window_frame_counts=scene_report["plan"]["window_frame_counts"],
                audio_reference=conditioning_kwargs.get("audio_reference"),
                images=images, seeds=[item["seed"] for item in recipe["scene"]["segments"]],
                checkpoint_dir=checkpoint_directory or target.parent / "checkpoints",
                unload_after=True,
                step_callback=scene_progress,
            )
            frames = scene_report["plan"]["total_frames"] - 1
            render_progress("publishing", "Trimming scene to its delivered timeline")
            if source_audio_item is not None:
                if content_identity(source_audio_item["path"]) != source_audio_sha256:
                    raise ValueError("Scene source audio changed during generation.")
                result.update(source_audio_sha256=source_audio_sha256,
                              source_audio_publication="original_interval_pcm_s32le")
            publish_scene_movie(Path(result["video_path"]), target, frames=frames,
                                fps=config.frame_rate, ffmpeg=media_binary(recipe, "ffmpeg"),
                                source_audio=source_audio_item)
            result.update(native_video_path=result["video_path"], video_path=str(target),
                          delivered_frames=frames,
                          delivered_duration_seconds=frames / config.frame_rate,
                          publication_mode=scene_report["scene"]["publication_mode"])
        else:
            result = RUNTIME.generate_to_file(
                spec,
                config,
                effective_prompt,
                generation_target,
                unload_after=True,
                step_callback=lambda done, total: render_progress(
                    "sampling", "Sampling", completed=done, total=total
                ),
                **conditioning_kwargs,
            )
        if extension:
            source_report = media_report[0]
            context_frames = (
                contract["extension"]["context_frames"]
                if recipe["engine"] == "ltx25"
                else source_report["num_frames"]
            )
            from wee_todd_mlx.conditioning_media import media_binary

            assemble_external_extension(
                contract["inputs"][0]["path"],
                generation_target,
                target,
                source_frames=source_report["num_frames"],
                context_frames=context_frames,
                fps=source_report["fps"],
                sample_rate=48000,
                ffmpeg=media_binary(recipe, "ffmpeg"),
            )
            result["video_path"] = str(target)
            result["extension"] = {
                "direction": "after",
                "source_frames": source_report["num_frames"],
                "context_frames": context_frames,
                "additional_frames": contract["extension"]["additional_frames"],
                "output_frames": source_report["num_frames"]
                + contract["extension"]["additional_frames"],
                "audio_policy": contract["audio_policy"],
                "generated_window": str(generation_target),
            }
            result["audio_policy"] = contract["audio_policy"]
        return {
            "video": result["video_path"],
            **({"scene": scene_report["scene"]} if scene_report else {}),
            "metadata": result,
            "conditioning": task_report,
            "consumed_input_ids": task_report["input_ids"],
            "runtime_loaded": [RUNTIME.loaded],
        }
    finally:
        RUNTIME.unload()
        if ingredients_dir is not None:
            ingredients_dir.cleanup()
        if audio_interval_dir is not None:
            audio_interval_dir.cleanup()



def execute_native_render(recipe, output, *, checkpoint_directory=None):
    from wee_todd_mlx.inference_lease import InferenceLease
    from wee_todd_mlx.progress import render_progress
    with InferenceLease(progress=lambda event: render_progress("waiting", event["message"])):
        if recipe["engine"] == "h3":
            return render_h3(recipe, output)
        return render_ltx(recipe, output, checkpoint_directory=checkpoint_directory)

def main():
    if "--job" in sys.argv[1:]:
        from studio_job import cli as run_studio_job

        return run_studio_job()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model-library", type=Path, help="Registry for local asset references")
    parser.add_argument("--preflight-only", action="store_true", help="Validate without rendering")
    parser.add_argument("--checkpoint-directory", type=Path,
                        help="Persistent native scene windows for an exported job resume")
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    if recipe.get("format") != "weetodd-headless-v2" or recipe.get("engine") not in {
        "h3",
        "ltx23",
        "ltx25",
    }:
        parser.error("Unsupported recipe format/engine")
    sys.meta_path.insert(0, NoComfyImports())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from wee_todd_mlx.progress import render_progress
    assert_isolated()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = {
        "candidate": recipe["candidate"],
        "recipe": str(args.recipe.resolve()),
        "recipe_sha256": hashlib.sha256(args.recipe.read_bytes()).hexdigest(),
        "status": "failed",
    }
    try:
        render_progress("preflight", "Validating models and conditioning")
        from wee_todd_mlx.asset_registry import AssetRegistry, resolve_recipe

        recipe, resolution = resolve_recipe(
            recipe, AssetRegistry(args.model_library) if args.model_library else None
        )
        record["asset_resolution"] = resolution
        (output / "resolved-recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
        from wee_todd_mlx.headless_preflight import preflight_recipe

        record["preflight"] = preflight_recipe(recipe)
        (output / "effective-conditioning.json").write_text(
            json.dumps(record["preflight"]["conditioning"]["contract"], indent=2) + "\n"
        )
        if args.preflight_only:
            if "continuation" in recipe:
                from wee_todd_mlx.h3_continuation_artifact import prepare_continuation

                prepared = prepare_continuation(recipe, load_arrays=False)
                record["preflight"]["continuation"] = {
                    **record["preflight"].get("continuation", {}),
                    "request": prepared["request"],
                    "identity": prepared["identity"],
                    "source": prepared["source"],
                }
            record.update(
                status="preflight_passed",
                seconds=time.perf_counter() - started,
                isolation=assert_isolated(),
            )
            print(json.dumps({"status": record["status"], "output": str(output)}), flush=True)
            return
        record.update(execute_native_render(recipe, output / "render.mp4",
                                            checkpoint_directory=args.checkpoint_directory))
        record["seconds"] = time.perf_counter() - started
        if any(record["runtime_loaded"]):
            raise RuntimeError("A weighted runtime was not released")
        record["isolation"] = assert_isolated()
        render_progress("publishing", "Verifying finished movie")
        record["mp4_sha256"] = hashlib.sha256(Path(record["video"]).read_bytes()).hexdigest()
        record["status"] = "success"
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        import resource

        # macOS reports bytes; Linux reports KiB. Each native recipe has its own process.
        record["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )
        (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {k: record[k] for k in ("candidate", "status", "seconds", "video", "mp4_sha256")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
