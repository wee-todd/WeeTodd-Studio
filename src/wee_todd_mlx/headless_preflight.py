"""Run existing engine contracts without constructing a weighted runtime."""

from __future__ import annotations


def preflight_recipe(recipe):
    from .conditioning_media import inspect_media
    from .studio_scene import validate_scene_recipe
    from .task_conditioning import validate_conditioning, validate_ltx25_control_families

    scene_report = validate_scene_recipe(recipe)
    task_report = validate_conditioning(recipe)
    media_report = inspect_media(recipe, task_report["contract"])
    engine = recipe["engine"]
    adapters = []
    continuation_report = None
    if engine == "h3":
        from dataclasses import replace

        from wee_todd_nodes.lora import H3LoRAStack
        from wee_todd_nodes.preflight import (
            H3ComponentSetSpec,
            H3PreflightRequest,
            estimate_h3_token_budget,
            preflight_components,
        )
        from wee_todd_nodes.preview import H3PreviewConfig
        from wee_todd_nodes.runtime import H3GenerationConfig

        from .h3_continuation_artifact import continuation_request

        fields = dict(recipe["components"])
        preview = fields.pop("preview_override", None)
        spec = H3ComponentSetSpec(
            **fields, preview_override=H3PreviewConfig(**preview) if preview else None
        )
        config = H3GenerationConfig(**recipe["config"])
        config.validate()
        continuation = continuation_request(recipe)
        if continuation and continuation.get("source_context"):
            config = replace(config, duration_seconds=continuation["sample_duration_seconds"])
            config.validate()
        config.validate_paging(
            spec.resolved_paths()["transformer"],
            recipe.get("block_residency", "checkpoint_default"),
        )
        request = H3PreflightRequest(
            duration_seconds=config.duration_seconds,
            steps=config.steps,
            width=config.width,
            height=config.height,
        )
        report = preflight_components(spec, request).to_dict()
        if continuation:
            overlap = continuation["overlap_frames"]
            spatial_rows = (config.width // 32) * (config.height // 32)
            context_video_frames = (overlap - 5) // 17 * 5 + 2 if overlap else 0
            keyframes = sum(
                item["role"] == "keyframe" for item in task_report["contract"]["inputs"]
            )
            continuation_report = {
                "request": continuation,
                "token_budget": estimate_h3_token_budget(
                    request,
                    condition_video_rows=(context_video_frames + keyframes) * spatial_rows,
                    condition_audio_rows=round(overlap / 24 * 40) * 2,
                ),
                "memory_estimate_scope": (
                    "Engine memory uses the complete native generated window, but excludes "
                    "additional conditioning-prefix and vision workspaces. The separate token "
                    "budget includes continuation and keyframe rows with an estimated text length."
                ),
            }
        if task_report["contract"]["task"] == "extension":
            from wee_todd_nodes.conditioning_inputs import H3ReferenceInput, H3ReferenceStack

            if not hasattr(H3ReferenceStack, "prepare") or not hasattr(
                H3ReferenceInput, "validate"
            ):
                raise ValueError("Installed H3 runtime cannot consume Ref2VA continuation media")
        stack = H3LoRAStack.from_recipe(recipe)
        stack.validate_for_steps(config.steps)
        adapters = stack.metadata()
        if recipe.get("vdn"):
            from wee_todd_nodes.vdn import H3VDNSpec

            vdn = H3VDNSpec(**recipe["vdn"])
            vdn.validate()
            vdn.validate_sampling(config, stack if stack.adapters else None)
    elif engine == "ltx23":
        from ltx23_mlx.ic_lora import recipe_ic_specs, validate_control_inputs, validate_ic_stack
        from ltx23_mlx.lora import recipe_specs, validate_stack
        from ltx23_mlx.runtime import LTX23GenerationConfig, LTX23ModelSpec

        fields = dict(recipe["components"])
        fields.pop("loras", None)
        fields.pop("ic_loras", None)
        spec = LTX23ModelSpec(
            **fields, loras=recipe_specs(recipe), ic_loras=recipe_ic_specs(recipe)
        )
        config = LTX23GenerationConfig(**recipe["config"])
        config.validate()
        task = task_report["contract"]["task"]
        if config.pipeline_mode == "distilled_single_stage" and task != "t2v":
            raise ValueError("Single-pass distilled currently supports text-to-video only")
        if task in {"fflf", "a2v", "extension"}:
            import inspect

            from ltx23_mlx.runtime import _pipeline_class

            pipeline_class = _pipeline_class(
                "keyframe"
                if task == "fflf"
                else (
                    "extension_distilled"
                    if task == "extension" and config.pipeline_mode == "distilled"
                    else "extension"
                )
                if task == "extension"
                else "a2v"
            )
            if task == "extension":
                parameters = inspect.signature(pipeline_class.extend_from_video).parameters
                required = {"video_path", "extend_frames", "direction"}
            else:
                parameters = inspect.signature(pipeline_class.generate_and_save).parameters
                required = (
                    {"audio_path"}
                    if task == "a2v"
                    else {
                        "keyframe_images",
                        "keyframe_indices",
                        "keyframe_strengths",
                        "video_guider_params",
                    }
                )
            if required - parameters.keys():
                raise ValueError("Installed LTX 2.3 pipeline cannot consume requested conditioning")
            constructor = inspect.signature(pipeline_class.__init__).parameters
            constructor_required = (
                set()
                if task == "extension" and config.pipeline_mode == "distilled"
                else {"dev_transformer"}
                if task == "extension"
                else {"dev_transformer", "distilled_lora"}
            )
            if constructor_required - constructor.keys():
                raise ValueError(
                    "Installed LTX 2.3 conditioning pipeline cannot bind Dev/refinement weights"
                )
        spec.validate(config.pipeline_mode)
        adapters = validate_stack(
            spec.loras,
            spec.root(),
            config.pipeline_mode,
            low_ram_streaming=config.low_ram_streaming,
        )
        report = spec.inventory(config.pipeline_mode)
        report["ic_loras"] = validate_ic_stack(spec, config)
        if task in {"control", "ref2va"}:
            import inspect

            from ltx23_mlx.runtime import _pipeline_class

            cls = _pipeline_class("control")
            if "video_conditioning" not in inspect.signature(cls.generate_and_save).parameters:
                raise ValueError("Installed IC pipeline cannot consume control video")
            validate_control_inputs(
                spec,
                config,
                [
                    {k: i[k] for k in ("path", "kind", "strength", "control_type")}
                    for i in task_report["contract"]["inputs"]
                ],
            )
        report["limitation"] = (
            "Bundle presence check; not full tensor/task compatibility qualification"
        )
    elif engine == "ltx25":
        import inspect

        from ltx25_mlx.runtime import LTX25ComponentSpec, LTX25GenerationConfig

        if recipe.get("loras"):
            raise ValueError("LTX 2.5 adapters must be declared under components.loras")
        fields = dict(recipe["components"])
        for key in ("loras", "ic_loras"):
            fields[key] = tuple(tuple(item) for item in fields.get(key, ()))
        spec = LTX25ComponentSpec(**fields)
        fields = dict(recipe["config"])
        fields["stg_blocks"] = tuple(fields.get("stg_blocks", ()))
        config = LTX25GenerationConfig(**fields)
        report = spec.validate(
            config.pipeline_mode, require_spatial_upscaler=not config.ic_lora_single_stage
        )
        if scene_report and any(item.get("adapter_role") == "ic_lora"
                                for item in report.get("transformer_baked_loras", ())):
            raise ValueError("Continuous scenes do not support baked IC-LoRA adapters.")
        validate_ltx25_control_families(task_report["contract"], report)
        config.validate(
            scale_factors=tuple(report["video_scale_factors"]),
            reference_downscale_factor=int(report.get("ic_lora_reference_downscale_factor") or 1),
        )
        if task_report.get("prompt_guide"):
            from ltx25_mlx.runtime import LTX25RuntimeCache

            if "msr_references" not in inspect.signature(
                LTX25RuntimeCache.generate_to_file
            ).parameters:
                raise ValueError("Installed LTX 2.5 runtime cannot consume MSR references")
        if task_report["contract"]["task"] == "extension":
            from ltx25_mlx.pipeline import LTX25DistilledPipeline

            runtime_parameters = inspect.signature(
                __import__("ltx25_mlx.runtime", fromlist=["LTX25RuntimeCache"])
                .LTX25RuntimeCache.generate_to_file
            ).parameters
            if "extension_input" not in runtime_parameters or not hasattr(
                LTX25DistilledPipeline, "encode_external_continuation"
            ):
                raise ValueError("Installed LTX 2.5 runtime cannot encode external continuation")
        if spec.loras:
            from ltx25_mlx.transformer import inspect_ltx25_lora

            adapters = []
            for path, _strength in spec.loras:
                adapter = inspect_ltx25_lora(path)
                if scene_report and adapter["adapter_role"] != "transformer_lora":
                    raise ValueError(
                        "Continuous scenes do not support IC-LoRA or MSR adapters, "
                        "including entries in components.loras."
                    )
                adapters.append({**adapter, "path": str(adapter["path"])})
    else:
        raise ValueError(f"Unsupported engine: {engine}")
    return {
        "format": "weetodd-headless-preflight-v1",
        "engine": engine,
        "status": "preflight_passed",
        "engine_report": report,
        "adapters": adapters,
        "conditioning": task_report,
        "media": media_report,
        "render_qualification": "not_evaluated",
        "weights_loaded": False,
        **({"continuation": continuation_report} if continuation_report else {}),
        **({"scene": scene_report["scene"], "scenePlan": scene_report["plan"]}
           if scene_report else {}),
    }
