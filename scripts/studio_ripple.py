"""Local Ripple editing bridge, using the shared LTX 2.5 runtime.

Inspection and source-frame extraction are weight-free. The only recognized Ripple
checkpoint is the pinned author v11 file; additional timed anchors are a Studio
extension of the author's first-frame workflow, not a separate sampler.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

DEFAULT_RIPPLE_PROMPT = (
    "Use the reference video for motion, timing, camera movement, composition, and unchanged "
    "scene content, while consistently propagating the visual edit established in the first "
    "frame throughout the video."
)


def _check(cancelled):
    if cancelled():
        raise InterruptedError("Ripple editing cancelled.")


def content_identity(filename, cancelled=lambda: False):
    digest = hashlib.sha256()
    with Path(filename).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            _check(cancelled)
            digest.update(block)
    return digest.hexdigest()


def _run(command, cancelled=lambda: False):
    _check(cancelled)
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            _check(cancelled)
            try:
                output, error = child.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                continue
        if child.returncode:
            raise ValueError(
                error.decode("utf-8", errors="replace")[-2000:]
                or f"{Path(command[0]).name} failed."
            )
        return output
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate()


def _number(value, name, minimum=None, maximum=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Ripple {name} must be a number.")
    if not math.isfinite(value) or integer and type(value) is not int:
        raise ValueError(f"Ripple {name} must be a finite {'integer' if integer else 'number'}.")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"Ripple {name} must be between {minimum} and {maximum}.")
    return value


def validate_request(request, *, require_references=True):
    raw = request.get("ripple")
    if not isinstance(raw, dict):
        raise ValueError("A Ripple request is required.")
    if raw.get("version", 1) != 1:
        raise ValueError("Unsupported Ripple request version.")
    value = copy.deepcopy(raw)
    source = Path(str(value.get("source_path", ""))).expanduser().resolve()
    if not source.is_file():
        raise ValueError("Select an existing source movie for Ripple.")
    value["source_path"] = str(source)
    value["source_start"] = _number(value.get("source_start", 0), "source start", 0)
    value["duration"] = _number(value.get("duration"), "duration", 0.001, 30)
    value["frame_rate"] = _number(value.get("frame_rate"), "frame rate", 1, 60)
    for field in ("width", "height"):
        value[field] = _number(value.get(field), field, 32, 1920, integer=True)
        if value[field] % 32:
            raise ValueError("Ripple width and height must be multiples of 32.")
    value["seed"] = _number(value.get("seed", 0), "seed", 0, 2**32 - 1, integer=True)
    value["lora_strength"] = _number(value.get("lora_strength", 1.35), "adapter strength", 0.001, 3)
    value["audio_policy"] = value.get("audio_policy", "preserve")
    if value["audio_policy"] not in {"preserve", "silent"}:
        raise ValueError("Ripple audio policy must be preserve or silent.")
    value["frames"] = max(1, math.ceil(value["duration"] * value["frame_rate"] - 1e-7))
    intervals = max(
        1, math.ceil((value["frames"] - 1) / 8), math.ceil(0.25 * value["frame_rate"] / 8)
    )
    value["model_frames"] = 1 + 8 * intervals
    if (value["model_frames"] - 1) / value["frame_rate"] > 30:
        raise ValueError("The Ripple interval plus model padding exceeds 30 seconds.")
    references = value.get("references", [])
    if (
        not isinstance(references, list)
        or len(references) > 9
        or require_references
        and not references
    ):
        raise ValueError("Ripple requires one to nine edited reference images.")
    seen = set()
    for ref in references:
        if not isinstance(ref, dict):
            raise ValueError("Each Ripple reference needs a frame and edited image.")
        frame = _number(ref.get("frame"), "reference frame", 0, value["frames"] - 1, integer=True)
        if frame in seen:
            raise ValueError("Each Ripple reference must target a different source frame.")
        seen.add(frame)
        ref["strength"] = _number(ref.get("strength", 1), "reference strength", 0, 1)
        image_path = Path(str(ref.get("path", ""))).expanduser().resolve()
        if not image_path.is_file():
            raise ValueError("A Ripple edited reference image is missing.")
        from PIL import Image

        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            raise ValueError("A Ripple reference is not a readable image.") from exc
        ref["path"] = str(image_path)
    value["references"] = sorted(references, key=lambda item: item["frame"])
    if require_references and 0 not in seen:
        raise ValueError("Ripple requires an edited reference for source frame 0.")
    return value


def probe_media(source, settings, *, cancelled=lambda: False):
    from studio_bridge import executable

    data = json.loads(
        _run(
            [
                executable("ffprobe", settings),
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(source),
            ],
            cancelled,
        )
    )
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("Ripple requires a source movie with a video stream.")

    def rate(text):
        numerator, denominator = (text or "0/1").split("/")
        return float(numerator) / float(denominator) if float(denominator) else 0

    fps = rate(video.get("avg_frame_rate")) or rate(video.get("r_frame_rate"))
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0)
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("The Ripple source has no usable duration or frame rate.")
    rotation = next(
        (float(item["rotation"]) for item in video.get("side_data_list", []) if "rotation" in item),
        float(video.get("tags", {}).get("rotate", 0)),
    )
    width, height = int(video["width"]), int(video["height"])
    sar_text = video.get("sample_aspect_ratio", "1:1")
    try:
        numerator, denominator = (float(item) for item in sar_text.split(":"))
        sar = numerator / denominator if denominator else 1
    except (ValueError, AttributeError):
        sar = 1
    if not math.isfinite(sar) or sar <= 0:
        sar = 1
    if round(rotation) % 180 == 90:
        width, height, sar = height, width, 1 / sar
    width = math.floor(width * sar + 0.5)
    return dict(
        kind="video",
        path=str(source),
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        hasAudio=any(s.get("codec_type") == "audio" for s in data.get("streams", [])),
        start_time=float(data.get("format", {}).get("start_time") or 0),
        timestamp_tick=rate(video.get("time_base")),
        rotation=rotation,
    )


def first_source_timestamp(values, source, settings, cancelled=lambda: False):
    """Validate interval cadence and locate the first actual decoded frame."""
    from studio_bridge import executable

    origin = source.get("start_time", 0)
    threshold = values["source_start"] + origin
    data = json.loads(
        _run(
            [
                executable("ffprobe", settings),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-read_intervals",
                f"{threshold:.12f}%{threshold + values['duration']:.12f}",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "json",
                values["source_path"],
            ],
            cancelled,
        )
    )
    timestamps = []
    for frame in data.get("frames", []):
        timestamp = float(frame.get("best_effort_timestamp_time", "nan"))
        if not math.isfinite(timestamp):
            raise ValueError("Ripple requires a constant frame rate source with valid timestamps.")
        if threshold - 1e-7 <= timestamp < threshold + values["duration"] - 1e-7:
            timestamps.append(timestamp)
    if not timestamps:
        raise ValueError("No decoded source frame was found at the Ripple interval start.")
    # Container time bases quantize valid CFR timestamps (e.g. Matroska milliseconds).
    # Compare all decoded timestamps with the cadence, allowing one container tick.
    tolerance = max(2e-6, source.get("timestamp_tick", 0) * 1.05)
    for index, timestamp in enumerate(timestamps):
        expected = timestamps[0] + index / source["fps"]
        if abs(timestamp - expected) > tolerance:
            raise ValueError(
                "Ripple requires a constant frame rate source for exact frame and audio alignment. "
                "Convert this variable frame rate movie to constant frame rate before editing."
            )
    return timestamps[0] - origin


def inspect(values, settings, cancelled=lambda: False):
    source = probe_media(values["source_path"], settings, cancelled=cancelled)
    if values["source_start"] + values["duration"] > source["duration"] + 0.001:
        raise ValueError("The selected Ripple interval extends beyond the source movie.")
    preview_start = first_source_timestamp(values, source, settings, cancelled)
    return dict(
        source=source,
        frames=values["frames"],
        frame_count=values["frames"],
        model_frames=values["model_frames"],
        frame_rate=values["frame_rate"],
        source_frame_rate=source["fps"],
        duration=values["duration"],
        source_start=values["source_start"],
        source_preview_start=preview_start,
        width=values["width"],
        height=values["height"],
        has_audio=source["hasAudio"],
    )


def source_filter(values):
    # Both still extraction and the native guide decode execute this same timeline.
    return (
        f"trim=start={values['source_start']:.12f}:duration={values['duration']:.12f},"
        "setpts=PTS-STARTPTS"
    )


def require_source_cadence(values, report):
    if not math.isclose(values["frame_rate"], report["source_frame_rate"], rel_tol=1e-5):
        raise ValueError(
            "Ripple frame rate must match the source movie for exact frame alignment. "
            f"Use {report['source_frame_rate']:.8g} fps."
        )


def decode_source(values, settings, cancelled=lambda: False):
    import numpy as np
    from studio_bridge import executable

    width, height = values["width"], values["height"]
    fit = (
        "format=rgb24,scale=trunc(iw*sar+0.5):ih,setsar=1,"
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1"
    )
    filters = source_filter(values) + f",{fit},tpad=stop_mode=clone:stop_duration=1"
    raw = _run(
        [
            executable("ffmpeg", settings),
            "-v",
            "error",
            "-nostdin",
            "-i",
            values["source_path"],
            "-vf",
            filters,
            "-an",
            "-frames:v",
            str(values["frames"]),
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        cancelled,
    )
    expected = values["frames"] * width * height * 3
    if len(raw) != expected:
        raise ValueError("Ripple could not decode every frame in the selected source interval.")
    return np.frombuffer(raw, dtype=np.uint8).reshape(values["frames"], height, width, 3)


def extract_frame(values, settings, output, cancelled=lambda: False):
    from studio_bridge import executable

    frame = _number(values.get("frame"), "source frame", 0, values["frames"] - 1, integer=True)
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"source-frame-{frame:06d}.png"
    # Keep original source resolution for editing; native image and video paths both
    # center crop to generation geometry later.
    filters = (
        source_filter(values) + f",tpad=stop_mode=clone:stop_duration=1,select=eq(n\\,{frame}),"
        "format=rgb24,scale=trunc(iw*sar+0.5):ih,setsar=1"
    )
    _run(
        [
            executable("ffmpeg", settings),
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            values["source_path"],
            "-vf",
            filters,
            "-frames:v",
            "1",
            "-an",
            str(target),
        ],
        cancelled,
    )
    if not target.is_file():
        raise ValueError("Ripple source frame extraction returned no image.")
    return dict(
        image_path=str(target),
        path=str(target),
        frame=frame,
        source_time=values.get("source_preview_start", values["source_start"])
        + frame / values["frame_rate"],
    )


def prepare_conditioning(values, source):
    import numpy as np
    from PIL import Image, ImageOps

    first = values["references"][0]
    with Image.open(first["path"]) as image:
        edited = np.asarray(
            ImageOps.fit(
                image.convert("RGB"),
                (values["width"], values["height"]),
                method=Image.Resampling.LANCZOS,
            )
        )
    # Independently reproduced from the author's FFAF workflow: prepend the edit,
    # then trim to the causal VAE canvas. Pad only outside the editorial interval.
    guide = np.concatenate([edited[None], source], axis=0)
    if len(guide) < values["model_frames"]:
        guide = np.concatenate(
            [guide, np.repeat(guide[-1:], values["model_frames"] - len(guide), axis=0)]
        )
    guide = np.ascontiguousarray(guide[: values["model_frames"]], dtype=np.float32) / 255
    author_mode = len(values["references"]) == 1 and first["frame"] == 0 and first["strength"] == 1
    anchors = [
        dict(path=ref["path"], frame_index=ref["frame"], strength=ref["strength"])
        for ref in values["references"][1:]
    ]
    return guide, anchors, "author_first_frame" if author_mode else "studio_timed_anchors"


def build_recipe(profile, values, adapter):
    if (
        profile.get("format") != "weetodd-headless-v2"
        or profile.get("engine") != "ltx25"
        or profile.get("config", {}).get("pipeline_mode", "distilled") != "distilled"
    ):
        raise ValueError("Ripple requires a native LTX 2.5 distilled profile.")
    components = copy.deepcopy(profile["components"])
    if any(
        components.get(key) for key in ("loras", "ic_loras", "msr_lora_path", "distilled_lora_path")
    ):
        raise ValueError("Choose a plain distilled profile without other adapters for Ripple.")
    components["ic_loras"] = [[str(adapter), values["lora_strength"]]]
    # Carry only execution/memory settings. Hidden generation settings from another
    # task must never silently change this dedicated author recipe.
    carry = {
        key: item
        for key, item in profile.get("config", {}).items()
        if key
        in {
            "low_ram_streaming",
            "prompt_context",
            "diffvae_optimization",
            "diffvae_query_chunk_size",
            "diffvae_context_width_chunks",
            "diffvae_stage4_tile_width",
        }
    }
    config = dict(
        carry,
        pipeline_mode="distilled",
        width=values["width"],
        height=values["height"],
        duration_seconds=(values["model_frames"] - 1) / values["frame_rate"],
        duration_mode="manual",
        frame_rate=values["frame_rate"],
        seed=values["seed"],
        stage1_steps=8,
        stage2_steps=0,
        ic_lora_single_stage=True,
        stage1_sampler="euler",
        stage1_eta=0.0,
        stage1_s_noise=1.0,
        stage2_sampler="euler",
        video_cfg_scale=1.0,
        audio_cfg_scale=1.0,
        low_memory=True,
        feed_forward_backend="reference_fp32",
        sol_attention_profile="disabled",
    )
    return dict(
        format="weetodd-headless-v2",
        engine="ltx25",
        components=components,
        config=config,
        prompt=values.get("prompt", ""),
    )


def resolve_profile(settings, values, adapter):
    selected = settings.get("rippleProfileID", "").strip()
    explicit = selected not in {"", "auto"}
    candidates = (
        [Path(selected).expanduser()]
        if explicit
        else sorted(Path(settings.get("profilesDirectory", "")).expanduser().glob("*.json"))
    )
    errors = []
    for candidate in candidates:
        try:
            recipe = build_recipe(json.loads(candidate.read_text()), values, adapter)
            # Component presence is established before importing an MLX module.
            for key in (
                "transformer_path",
                "text_encoder_path",
                "video_vae_path",
                "audio_vae_path",
            ):
                source = Path(recipe["components"].get(key, "")).expanduser()
                if not source.exists() or not recipe["components"].get(key):
                    raise ValueError(f"Ripple component {key} is missing.")
                recipe["components"][key] = str(source.resolve())
            return recipe, str(candidate.resolve())
        except (OSError, ValueError, KeyError) as exc:
            errors.append(str(exc))
    detail = (
        errors[0] if explicit and errors else "No compatible installed distilled profile was found."
    )
    raise ValueError(f"{detail} Select a native LTX 2.5 profile in Runtime Settings.")


def publish_video(native, values, source_report, settings, target, cancelled=lambda: False):
    from studio_bridge import executable

    command = [executable("ffmpeg", settings), "-v", "error", "-nostdin", "-y", "-i", str(native)]
    has_audio = values["audio_policy"] == "preserve" and source_report["has_audio"]
    if has_audio:
        command += [
            "-i",
            values["source_path"],
            "-filter_complex",
            f"[1:a:0]atrim=start={values['source_start']:.12f}:duration={values['duration']:.12f},"
            f"asetpts=PTS-{values['source_start']:.12f}/TB,aresample=async=1:first_pts=0,"
            f"apad=whole_dur={values['duration']:.12f},atrim=duration={values['duration']:.12f}"
            "[source_audio]",
            "-map",
            "0:v:0",
            "-map",
            "[source_audio]",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
        ]
    else:
        command += ["-map", "0:v:0", "-an"]
    command += [
        "-vf",
        f"trim=end_frame={values['frames']},setpts=PTS-STARTPTS",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "fast",
        "-bf",
        "0",
        "-t",
        f"{values['duration']:.12f}",
        "-movflags",
        "+faststart",
        str(target),
    ]
    _run(command, cancelled)
    return has_audio


def generate(values, settings, output, *, progress, cancelled):
    from ltx25_mlx.ripple import (
        RIPPLE_REPOSITORY,
        RIPPLE_REVISION,
        RIPPLE_SHA256,
        is_verified_ripple,
    )

    source_report = inspect(values, settings, cancelled)
    require_source_cadence(values, source_report)
    source_identity = content_identity(values["source_path"], cancelled)
    expected_source_identity = values.get("source_sha256")
    if expected_source_identity is not None and expected_source_identity != source_identity:
        raise ValueError(
            "The Ripple source movie differs from the saved take. Relink its original source."
        )
    prompt = values.get("prompt", "")
    if not isinstance(prompt, str):
        raise ValueError("The Ripple prompt must be text.")
    prompt = prompt.strip() or DEFAULT_RIPPLE_PROMPT
    values["prompt"] = prompt
    adapter = Path(settings.get("rippleAdapterPath", "")).expanduser().resolve()
    if not adapter.is_file() or not is_verified_ripple(adapter):
        raise ValueError(
            "Select the verified author LTX25_Ripple_v11.safetensors in Runtime Settings."
        )
    _check(cancelled)
    recipe, profile = resolve_profile(settings, values, adapter)
    from ltx25_mlx.runtime import RUNTIME, LTX25ComponentSpec, LTX25GenerationConfig

    fields = dict(recipe["components"])
    for name in ("loras", "ic_loras"):
        fields[name] = tuple(tuple(item) for item in fields.get(name, ()))
    spec, config = LTX25ComponentSpec(**fields), LTX25GenerationConfig(**recipe["config"])
    report = spec.validate("distilled", require_spatial_upscaler=False)
    if report.get("transformer_baked_loras"):
        raise ValueError("Ripple requires a plain distilled transformer without baked adapters.")
    config.validate(
        scale_factors=tuple(report["video_scale_factors"]), reference_downscale_factor=1
    )
    _check(cancelled)
    output.mkdir(parents=True, exist_ok=True)
    # Freeze images used by both priming and exact timed anchors. Reproducible takes
    # survive subsequent edits to the user's reference files.
    from PIL import Image

    for index, ref in enumerate(values["references"]):
        frozen = output / f"edited-{index + 1:02d}-frame-{ref['frame']:06d}.png"
        with Image.open(ref["path"]) as image:
            image.convert("RGB").save(frozen)
        ref["original_path"], ref["path"] = ref["path"], str(frozen)
        ref["sha256"] = content_identity(frozen, cancelled)
    progress(dict(event="progress", message="Preparing source frames and Ripple references"))
    source = decode_source(values, settings, cancelled)
    guide, anchors, mode = prepare_conditioning(values, source)
    del source
    video_reference = dict(
        images=guide,
        start_frame=0,
        end_frame=config.num_frames - 1,
        strength=values["references"][0]["strength"],
        attention_strength=1.0,
        mask=None,
        control_type="custom_preprocessed",
        reference_size_policy="quality",
        reference_role="",
    )
    receipt = dict(
        format="weetodd-ripple-take-v1",
        request=values,
        recipe=recipe,
        profile_path=profile,
        adapter=dict(
            repository=RIPPLE_REPOSITORY,
            revision=RIPPLE_REVISION,
            sha256=RIPPLE_SHA256,
            path=str(adapter),
        ),
        conditioning_mode=mode,
        source_guide="earliest edited reference prepended to selected source frames",
        source_frame_mapping="trim at source time; preserve source cadence and decoded frame order",
        publication_audio="Source interval re-encoded as AAC; no generated audio is published.",
        image_inputs=anchors,
        source=source_report,
        source_sha256=source_identity,
        status="prepared",
        qualification="Author first-frame layout; native MLX quality requires visual qualification."
        if mode == "author_first_frame"
        else (
            "Studio extension: first edit primes IC guide; independently timed image anchors "
            "are not an author-proven multi-guide workflow."
        ),
    )
    receipt_path = output / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    native = output / "native-padded.mp4"
    try:
        result = RUNTIME.generate_to_file(
            spec,
            config,
            prompt,
            native,
            image_inputs=anchors,
            video_references=[video_reference],
            unload_after=True,
            check_interrupted=lambda: _check(cancelled),
            step_callback=lambda done, total: progress(
                dict(event="progress", message="Sampling Ripple edit", fraction=done / total)
            ),
        )
        _check(cancelled)
        if content_identity(values["source_path"], cancelled) != source_identity:
            raise ValueError("The source movie changed during Ripple generation. Retry the edit.")
        progress(dict(event="progress", message="Preserving source audio and editorial duration"))
        target = output / "ripple.mp4"
        has_audio = publish_video(native, values, source_report, settings, target, cancelled)
        actual = probe_media(target, settings, cancelled=cancelled)
        if actual["hasAudio"] != has_audio:
            raise ValueError("Ripple publication audio does not match the selected policy.")
        if abs(actual["duration"] - values["duration"]) > 1 / values["frame_rate"] + 0.002:
            raise ValueError("Ripple publication does not match the selected interval.")
        receipt.update(native_result=result, publication=actual, status="complete")
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        return dict(
            video_path=str(target),
            path=str(target),
            duration=values["duration"],
            frames=values["frames"],
            frame_rate=values["frame_rate"],
            width=values["width"],
            height=values["height"],
            has_audio=has_audio,
            artifacts_directory=str(output),
            receipt_path=str(receipt_path),
            conditioning_mode=mode,
            frozen_references=[
                {key: ref[key] for key in ("frame", "path", "strength")}
                for ref in values["references"]
            ],
            source_sha256=source_identity,
        )
    except BaseException as exc:
        receipt.update(
            status="cancelled"
            if isinstance(exc, (InterruptedError, KeyboardInterrupt))
            else "failed",
            error=str(exc),
        )
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        # A failed take never leaves a file that can be mistaken for an accepted output.
        (output / "ripple.mp4").unlink(missing_ok=True)
        raise
    finally:
        RUNTIME.unload()


def dispatch(
    command, request, output=None, *, progress=lambda event: None, cancelled=lambda: False
):
    _check(cancelled)
    if command not in {"ripple-inspect", "ripple-frame", "ripple-generate"}:
        raise ValueError(f"Unknown Ripple command: {command}")
    values = validate_request(request, require_references=command == "ripple-generate")
    settings = request.get("runtime", {})
    if command == "ripple-inspect":
        return inspect(values, settings, cancelled)
    if output is None:
        raise ValueError("Ripple frame extraction and generation need an output directory.")
    destination = Path(output).expanduser().resolve()
    if command == "ripple-frame":
        report = inspect(values, settings, cancelled)
        require_source_cadence(values, report)
        values["source_preview_start"] = report["source_preview_start"]
        return extract_frame(values, settings, destination, cancelled)
    # A fresh take directory prevents accidental overwrites of earlier results.
    destination.mkdir(parents=True, exist_ok=True)
    take = Path(tempfile.mkdtemp(prefix="ripple-", dir=destination))
    return generate(values, settings, take, progress=progress, cancelled=cancelled)
