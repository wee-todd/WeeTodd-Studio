"""Model-free Studio membership and native LTX 2.5 scene recipe contracts."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

PUBLICATION_MODE = "single_decode_native_latent_chain"


def _mode(clip):
    settings = clip.get("continuity") or {}
    if (
        not isinstance(settings, dict)
        or set(settings) - {"mode", "sourceClipID", "saveContext", "boundaryImagePolicy"}
        or type(settings.get("saveContext", False)) is not bool
    ):
        raise ValueError("Unsupported continuity settings.")
    from ltx25_mlx.chain_plan import validate_boundary_image_policy

    validate_boundary_image_policy(settings.get("boundaryImagePolicy", "balanced"))
    return settings.get("mode", "independent")


def _duration(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("Scene durations must be finite positive numbers.")
    return value


def scene_members(request):
    """Return the maximal contiguous group, or [] for an ordinary clip."""
    clips = request["project"]["clips"]
    ids = [clip["id"] for clip in clips]
    if len(ids) != len(set(ids)):
        raise ValueError("Continuous scenes require unique clip identities.")
    if request["clipID"] not in ids:
        raise ValueError("The selected scene member no longer exists.")
    index = ids.index(request["clipID"])
    begin = index
    while begin > 0 and _mode(clips[begin]) == "scene":
        begin -= 1
    end = index + 1
    while end < len(clips) and _mode(clips[end]) == "scene":
        end += 1
    if end - begin == 1 and _mode(clips[index]) != "scene":
        return []
    members = clips[begin:end]
    if _mode(members[0]) != "independent":
        raise ValueError("The first continuous scene member must be independent.")
    if not 2 <= len(members) <= 6:
        raise ValueError("Continuous scenes require two to six contiguous shots.")
    if sum(_duration(clip["duration"]) for clip in members) > 30 + 1e-9:
        raise ValueError("Continuous scenes support at most 30 seconds.")
    for offset, clip in enumerate(members):
        if clip["engine"] != "ltx25":
            raise ValueError("Continuous scenes require local native LTX 2.5 for every shot.")
        if offset and (clip.get("continuity") or {}).get("sourceClipID") not in {
            None,
            members[offset - 1]["id"],
        }:
            raise ValueError("A scene shot must join its immediately preceding shot.")
        if clip.get("extensionDirection") or clip.get("extensionSource"):
            raise ValueError("Clear video extension before preparing a continuous scene.")
        for attachment in clip.get("attachments", []):
            if attachment.get("role") not in {"first", "last", "keyframe", "lora"}:
                raise ValueError(
                    "Continuous scenes support endpoint/keyframe images and ordinary "
                    "LoRAs only; MSR, audio drivers and controls are not qualified."
                )
    return members


def _effective_settings(recipe):
    from ltx25_mlx.runtime import LTX25ComponentSpec, LTX25GenerationConfig

    config = asdict(LTX25GenerationConfig(**recipe["config"]))
    for key in ("duration_seconds", "seed"):
        config.pop(key)
    components = asdict(LTX25ComponentSpec(**recipe["components"]))
    # Canonical JSON normalizes tuples/lists without changing adapter order.
    from .generation_selection import fingerprint

    return fingerprint(
        {
            "config": config,
            "components": components,
            "other": {
                k: v
                for k, v in recipe.items()
                if k
                not in {
                    "config",
                    "components",
                    "prompt",
                    "candidate",
                    "conditioning",
                    "scene",
                    "ffmpeg",
                    "ffprobe",
                }
            },
        }
    )


def _plan_report(segments, plan):
    return {
        "version": 1,
        "members": [
            {
                "clip_id": segment["clip_id"],
                "source_in": start / plan.frame_rate,
                "duration": count / plan.frame_rate,
            }
            for segment, start, count in zip(
                segments, plan.segment_start_frames, plan.segment_frame_counts, strict=True
            )
        ],
        "frame_rate": plan.frame_rate,
        "publication_mode": PUBLICATION_MODE,
    }


def validate_scene_recipe(recipe):
    """Validate the complete scene before file inspection or weighted work."""
    if "scene" not in recipe:
        return None
    from ltx25_mlx.chain_plan import (
        boundary_image_guidance_report,
        plan_ltx25_scene,
        scene_image_conditioning_bytes,
        validate_boundary_image_policy,
    )
    from ltx25_mlx.runtime import LTX25GenerationConfig

    scene = recipe["scene"]
    if recipe.get("engine") != "ltx25":
        raise ValueError("Continuous scenes require native LTX 2.5.")
    if (
        not isinstance(scene, dict)
        or type(scene.get("version")) is not int
        or scene["version"] != 1
        or set(scene) - {"boundary_image_policy"}
        != {"version", "segments", "overlap_frames", "soundscape", "music"}
    ):
        raise ValueError("Unsupported scene schema.")
    policy = scene.get("boundary_image_policy", "strict")
    validate_boundary_image_policy(policy)
    segments = scene["segments"]
    if not isinstance(segments, list) or not 2 <= len(segments) <= 6:
        raise ValueError("Continuous scenes require two to six segments.")
    ids = set()
    for segment in segments:
        if not isinstance(segment, dict) or set(segment) != {
            "clip_id",
            "prompt",
            "duration_seconds",
            "seed",
        }:
            raise ValueError("Unsupported scene segment schema.")
        if (
            not isinstance(segment["clip_id"], str)
            or not segment["clip_id"]
            or segment["clip_id"] in ids
        ):
            raise ValueError("Scene segment identities must be unique nonempty strings.")
        ids.add(segment["clip_id"])
        if not isinstance(segment["prompt"], str) or not segment["prompt"].strip():
            raise ValueError("Every scene member requires a prompt.")
        if type(segment["seed"]) is not int or not 0 <= segment["seed"] < 2**32:
            raise ValueError("Scene member seeds must be unsigned 32-bit integers.")
        _duration(segment["duration_seconds"])
    if not all(isinstance(scene[key], str) for key in ("soundscape", "music")):
        raise ValueError("Scene soundscape and music must be strings.")
    if recipe.get("continuation") or recipe.get("reference_images") or recipe.get("loras"):
        raise ValueError("Unsupported scene continuation or adapter declaration.")
    if recipe.get("components", {}).get("ic_loras") or recipe.get("components", {}).get(
        "msr_lora_path"
    ):
        raise ValueError("Continuous scenes do not support MSR or IC-LoRA controls.")
    config = LTX25GenerationConfig(**recipe["config"])
    config.validate_chain_support()
    contract = recipe.get("conditioning", {})
    if (
        contract.get("task") not in {"t2v", "fflf"}
        or contract.get("extension")
        or contract.get("audio_policy", "generated") != "generated"
    ):
        raise ValueError("Continuous scenes support generated audio and image keyframes only.")
    plan = plan_ltx25_scene(
        [s["duration_seconds"] for s in segments],
        overlap_frames=scene["overlap_frames"],
        frame_rate=config.frame_rate,
    )
    if abs(config.duration_seconds - (plan.total_frames - 1) / plan.frame_rate) > 1e-8:
        raise ValueError("Scene config duration must match the resolved scene timeline.")
    anchors = {}
    for item in contract.get("inputs", []):
        if item.get("kind") != "image" or item.get("role") != "keyframe":
            raise ValueError("Continuous scenes support endpoint/keyframe images only.")
        frame = item.get("frame_index")
        if type(frame) is not int or not 0 <= frame < plan.total_frames - 1:
            raise ValueError("Scene keyframes must target the visible delivered timeline.")
        strength = item.get("strength", 1.0)
        if (
            isinstance(strength, bool) or not isinstance(strength, (int, float))
            or not math.isfinite(strength) or not 0 <= strength <= 1
        ):
            raise ValueError("Scene image strength must be a finite number from zero to one.")
        if frame in anchors:
            raise ValueError("Conflicting scene anchors target the same global frame.")
        anchors[frame] = item
    if len(anchors) > 32:
        raise ValueError("LTX 2.5 scenes support at most 32 image anchors.")
    window_images = [
        [item for frame, item in anchors.items() if start <= frame < start + count]
        for start, count in zip(plan.window_start_frames, plan.window_frame_counts, strict=True)
    ]
    scene_image_conditioning_bytes(
        window_images, height=config.height, width=config.width,
        single_stage=config.ic_lora_single_stage,
    )
    guidance = boundary_image_guidance_report(
        [SimpleNamespace(frame_idx=frame, strength=item.get("strength", 1.0))
         for frame, item in anchors.items()], plan, policy,
    )
    return {"scene": _plan_report(segments, plan), "plan": plan.as_dict(),
            "imageGuidance": guidance}


def compose_scene_recipe(request, compose_clip):
    """Resolve every member through the existing composer, without recursive grouping."""
    from ltx25_mlx.chain_plan import plan_ltx25_scene

    from .generation_selection import fingerprint
    from .task_conditioning import validate_conditioning

    members = scene_members(request)
    if not members:
        raise ValueError("Select a complete continuous scene.")
    composed = []
    for member in members:
        current = copy.deepcopy(request)
        current["clipID"] = member["id"]
        current["_sceneResolved"] = True
        current.pop("_continuityResolved", None)
        clip = next(c for c in current["project"]["clips"] if c["id"] == member["id"])
        clip["continuity"] = {"mode": "independent"}
        composed.append(compose_clip(current))
    recipes = [value[0] for value in composed]
    expected = _effective_settings(recipes[0])
    for member, recipe in zip(members, recipes, strict=True):
        if _effective_settings(recipe) != expected:
            raise ValueError(
                f"{member.get('name', member['id'])}: all scene members need "
                "compatible effective component, adapter and sampling settings."
            )
    recipe, report = copy.deepcopy(composed[0])
    fps = recipe["config"].get("frame_rate", 24)
    plan = plan_ltx25_scene([c["duration"] for c in members], frame_rate=fps)
    scene = {
        "version": 1,
        "segments": [
            {
                "clip_id": c["id"],
                "prompt": r["prompt"],
                "duration_seconds": c["duration"],
                "seed": r["config"]["seed"],
            }
            for c, r in zip(members, recipes, strict=True)
        ],
        "overlap_frames": 25,
        "soundscape": members[0].get("soundscape", ""),
        "music": members[0].get("music", ""),
        "boundary_image_policy": (members[0].get("continuity") or {}).get(
            "boundaryImagePolicy", "balanced"
        ),
    }
    inputs = {}
    for member, per_clip, start, length in zip(
        members, recipes, plan.segment_start_frames, plan.segment_frame_counts, strict=True
    ):
        for original in per_clip["conditioning"]["inputs"]:
            item = copy.deepcopy(original)
            local = item["frame_index"]
            # A last endpoint is the last visible delivered frame, never the extra VAE frame.
            if local != "last" and (type(local) is not int or not 0 <= local < length):
                raise ValueError(
                    "Timed scene images must lie within their resolved member range; "
                    "use Last frame for a terminal endpoint."
                )
            global_frame = start + (length - 1 if local == "last" else local)
            item.update(id=f"{member['id']}:{item['id']}", frame_index=global_frame)
            if global_frame in inputs:
                prior = inputs[global_frame]
                if (Path(prior["path"]).resolve(), prior.get("strength", 1)) != (
                    Path(item["path"]).resolve(),
                    item.get("strength", 1),
                ):
                    raise ValueError("Conflicting scene anchors target the same global frame.")
                continue
            inputs[global_frame] = item
    recipe["scene"] = scene
    recipe["config"]["duration_seconds"] = (plan.total_frames - 1) / fps
    recipe["conditioning"] = {
        "version": 1,
        "task": "fflf" if inputs else "t2v",
        "inputs": [inputs[key] for key in sorted(inputs)],
    }
    recipe["prompt"] = "\n\n".join(
        [f"Shot {i}: {s['prompt']}" for i, s in enumerate(scene["segments"], 1)]
        + [f"Shared soundscape: {scene['soundscape']}", f"Music: {scene['music']}"]
    )
    scene_report = validate_scene_recipe(recipe)
    anchor_warnings = []
    for index, boundary in enumerate(plan.segment_start_frames[1:], 1):
        before, after = inputs.get(boundary - 1), inputs.get(boundary)
        if before and after and Path(before["path"]).resolve() != Path(after["path"]).resolve():
            anchor_warnings.append(
                f"{members[index - 1].get('name', 'Previous shot')} → "
                f"{members[index].get('name', 'Next shot')}: different image files guide "
                "consecutive frames at this join. Compare their poses and lighting; "
                "Automatic avoids repeated overlap guidance but cannot reconcile "
                "contradictory images."
            )
    report.update(
        scene=scene_report["scene"],
        scenePlan=scene_report["plan"],
        conditioning=validate_conditioning(recipe),
        task=recipe["conditioning"]["task"],
        resolvedFingerprint=fingerprint({"recipe": recipe, "runtime": request["runtime"]}),
        warnings=[w for _, r in composed for w in r.get("warnings", [])],
        sceneAnchorWarnings=anchor_warnings,
        sceneImageGuidance=scene_report["imageGuidance"],
    )
    return recipe, report


def scene_window_prompts(recipe):
    scene = recipe["scene"]
    return [
        f"{s['prompt']}\n\nShared soundscape: {scene['soundscape']}\nMusic: {scene['music']}"
        for s in scene["segments"]
    ]
