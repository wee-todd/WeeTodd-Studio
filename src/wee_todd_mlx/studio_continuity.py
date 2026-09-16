"""Editorial continuity, resolved before shared native generation. No model imports.

Prepared inputs are immutable job-local snapshots. Source takes remain untouched.
H3 reuses its saved native state; per-clip LTX uses the media extension adapter.
Continuous LTX 2.5 scenes are composed separately in studio_scene.
"""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path


def _stat(filename):
    source = Path(filename).expanduser().resolve(strict=True)
    stat = source.stat()
    if not source.is_file():
        raise ValueError("Continuity source must be a local movie file.")
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino]


def _number(value, label, *, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"Continuity {label} must be a finite {qualifier} number.")
    return value


def continuity_state(request):
    """Freeze the selected earlier clip, active take and visible interval."""
    clips = request["project"]["clips"]
    ids = [clip["id"] for clip in clips]
    if len(ids) != len(set(ids)):
        raise ValueError("Continuity requires unique clip identities.")
    index = ids.index(request["clipID"])
    clip = clips[index]
    intent = clip.get("continuity") or {}
    if not isinstance(intent, dict) or set(intent) - {
        "mode", "sourceClipID", "saveContext", "boundaryImagePolicy"
    }:
        raise ValueError("Unsupported continuity settings.")
    from ltx25_mlx.chain_plan import validate_boundary_image_policy

    validate_boundary_image_policy(intent.get("boundaryImagePolicy", "balanced"))
    mode = intent.get("mode", "independent")
    if mode not in {"independent", "frame", "motion", "scene"}:
        raise ValueError("Choose a supported connection in Clip Continuity.")
    save = intent.get("saveContext", False)
    if type(save) is not bool:
        raise ValueError("Save motion context must be a boolean.")
    # Capturing the predecessor is opt-in, either directly or by a dependent H3 clip.
    if clip["engine"] == "h3":
        for child_index in range(index + 1, len(clips)):
            child = clips[child_index]
            settings = child.get("continuity") or {}
            source_id = settings.get("sourceClipID") or clips[child_index - 1]["id"]
            if (
                child["engine"] == "h3"
                and settings.get("mode") == "motion"
                and source_id == clip["id"]
            ):
                save = True
    # Preserve the H3 preference across engine switches; it is inactive for LTX.
    save = save and clip["engine"] == "h3"
    state = {"version": 1, "mode": mode, "engine": clip["engine"], "saveContext": save}
    if mode == "scene":
        from .studio_scene import scene_members

        state["memberIDs"] = [member["id"] for member in scene_members(request)]
        return state
    if mode == "independent":
        return state
    if clip["engine"] not in {"h3", "ltx23", "ltx25"}:
        raise ValueError("These continuity controls require a native H3 or LTX clip.")
    source_id = intent.get("sourceClipID") or (clips[index - 1]["id"] if index else None)
    if source_id not in ids[:index]:
        raise ValueError("Choose an earlier clip as the continuity source.")
    source = clips[ids.index(source_id)]
    filename = source.get("sourcePath")
    if not filename:
        raise ValueError("Render and accept the source clip before preparing continuity.")
    filename = str(Path(filename).expanduser().resolve())
    take = next(
        (
            item
            for item in reversed(source.get("versions", []))
            if str(Path(item["path"]).expanduser().resolve()) == filename
        ),
        {},
    )
    start = _number(source.get("sourceIn", 0), "source trim")
    duration = _number(source["duration"], "source duration", positive=True)
    state.update(
        sourceClipID=source_id,
        sourceTakeID=take.get("id"),
        sourcePath=filename,
        sourceIn=start,
        duration=duration,
        sourceStat=_stat(filename),
    )
    if clip.get("extensionDirection") or clip.get("extensionSource"):
        raise ValueError("Clear the separate video-extension setting before selecting continuity.")
    if mode == "frame":
        if any(
            item["role"] not in {"first", "last", "keyframe", "lora"}
            for item in clip.get("attachments", [])
        ):
            raise ValueError("Match previous frame supports endpoint images and LoRAs only.")
        return state
    if clip["engine"] == "h3":
        artifact = take.get("continuationArtifact")
        if source["engine"] != "h3" or not artifact:
            raise ValueError(
                "Enable Save motion context on an H3 source clip, "
                "render it, and accept that take first."
            )
        usable_start = _number(take.get("usableSourceIn", 0), "take start")
        usable_duration = _number(take.get("usableDuration", 0), "take duration", positive=True)
        if abs(start + duration - usable_start - usable_duration) > 0.5 / 24:
            raise ValueError(
                "The source take's terminal frames were trimmed. Restore its endpoint "
                "or render and accept a new context; Match previous frame supports trims."
            )
        state["artifact"] = copy.deepcopy(artifact)
    elif any(item["role"] != "lora" for item in clip.get("attachments", [])):
        raise ValueError(
            "LTX motion/audio continuation cannot combine endpoint, reference "
            "or audio attachments. "
            "Use Match previous frame for endpoint-guided clips, "
            "or remove these attachments explicitly."
        )
    return state


def verify_source(state):
    if state.get("sourcePath") and _stat(state["sourcePath"]) != state["sourceStat"]:
        raise ValueError("The continuity source file changed. Prepare the clip again.")


def effective_clip(clip, state):
    """Task intent used by the selector, including when no inputs are materialized yet."""
    result = copy.deepcopy(clip)
    mode = state["mode"]
    if mode == "frame" or (mode == "motion" and clip["engine"] != "h3"):
        selection = result.setdefault("generationSelection", {})
        if selection is None:
            selection = result["generationSelection"] = {}
        selection["task"] = (
            (
                "fflf"
                if any(item["role"] in {"last", "keyframe"} for item in clip.get("attachments", []))
                else "i2v"
            )
            if mode == "frame"
            else "extension"
        )
    if mode == "motion" and clip["engine"] != "h3":
        result.update(extensionDirection="after", extensionSource=state["sourcePath"])
    return result


def prepare_request(request, directory, bridge, *, fps=24):
    state = continuity_state(request)
    updated = copy.deepcopy(request)
    clips = updated["project"]["clips"]
    index = next(i for i, clip in enumerate(clips) if clip["id"] == request["clipID"])
    clip = clips[index] = effective_clip(clips[index], state)
    if state["mode"] == "independent" or (state["mode"] == "motion" and clip["engine"] == "h3"):
        return updated, state
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    media = bridge.inspect_media(state["sourcePath"], request["runtime"])
    source_fps = _number(media["fps"], "source frame rate", positive=True)
    if state["sourceIn"] + state["duration"] > media["duration"] + 0.5 / source_fps:
        raise ValueError("The visible continuity interval extends beyond the source movie.")
    end = state["sourceIn"] + state["duration"]
    if state["mode"] == "frame":
        destination = directory / "previous-frame.png"
        command = [
            bridge.executable("ffmpeg", request["runtime"]),
            "-v",
            "error",
            "-i",
            state["sourcePath"],
            "-vf",
            f"trim=start={state['sourceIn']:.12f}:end={end:.12f},setpts=PTS-STARTPTS",
            "-fps_mode",
            "vfr",
            "-update",
            "1",
            "-atomic_writing",
            "1",
            "-n",
            str(destination),
        ]
        bridge.run(command)
        if not destination.is_file():
            raise ValueError("The source movie did not contain the selected visible frame.")
        asset_id = f"continuity-{clip['id']}"
        updated["project"]["assets"].append(
            {
                "id": asset_id,
                "kind": "image",
                "name": "Previous visible frame",
                "path": str(destination.resolve()),
            }
        )
        clip["attachments"] = [
            item for item in clip.get("attachments", []) if item["role"] != "first"
        ]
        clip["attachments"].append(
            {"id": asset_id, "assetID": asset_id, "role": "first", "strength": 1}
        )
        # Update one image while streaming. Timestamps handle VFR movie endpoints.
        state["sourceTimeEnd"] = end
    else:
        if not media["hasAudio"]:
            raise ValueError(
                "Motion/audio continuation needs a source movie with audio. "
                "Use Match previous frame for silent sources."
            )
        # 49 frames is a bounded 8n+1 context at both 24 and 25fps, longer than
        # the LTX2.5 25-frame native overlap. Only the visible tail is re-encoded.
        count = 49
        length = count / fps
        if state["duration"] + 1e-7 < length:
            raise ValueError(
                f"Motion/audio continuation needs at least {length:.3f} seconds "
                "of visible source footage."
            )
        begin = end - length
        destination = directory / "previous-tail.mp4"
        graph = (
            f"[0:v]trim=start={begin:.12f}:end={end:.12f},setpts=PTS-STARTPTS,"
            f"fps={fps},tpad=stop_mode=clone:stop_duration=1,trim=end_frame={count}[v];"
            f"[0:a]atrim=start={begin:.12f}:end={end:.12f},asetpts=PTS-STARTPTS,"
            f"aresample=48000,apad,atrim=end={length:.12f}[a]"
        )
        bridge.run(
            [
                bridge.executable("ffmpeg", request["runtime"]),
                "-v",
                "error",
                "-i",
                state["sourcePath"],
                "-filter_complex",
                graph,
                "-map",
                "[v]",
                "-map",
                "[a]",
                "-c:v",
                "libx264",
                "-crf",
                "15",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-t",
                f"{length:.12f}",
                "-n",
                str(destination),
            ]
        )
        clip["extensionSource"] = str(destination.resolve())
        state.update(
            sourceFrames=count,
            nativeFPS=fps,
            usableSourceIn=length,
            usableDuration=clip["duration"],
        )
    verify_source(state)
    return updated, state


def configure_recipe(recipe, state):
    """Preserve sampler and adapter choices; apply only explicit continuity intent."""
    if recipe["engine"] == "h3" and (state["saveContext"] or state["mode"] == "motion"):
        context = {"version": 1, "context_frames": 22, "save_context": state["saveContext"]}
        if state["mode"] == "motion":
            context.update(
                source_context=state["artifact"]["manifest"],
                source_manifest_sha256=state["artifact"]["manifest_sha256"],
            )
        recipe["continuation"] = context
        from .h3_continuation_artifact import continuation_request

        geometry = continuation_request(recipe)
        if state["mode"] == "motion" and recipe.get("conditioning", {}).get("task") == "fflf":
            # The renderer shifts image anchors into the sampled overlap+new window.
            # Show those same times in Studio's resolved prompt, preserving shot direction.
            prompt = recipe["prompt"]
            alignment = r"\s*Picture \d+ is fully referenced at \d+(?:\.\d+)? seconds\."
            prompt = re.sub(r"^(?:" + alignment + r")+\s*", "", prompt)
            frames = [
                geometry["published_frames"] - 1
                if item["frame_index"] == "last" else item["frame_index"]
                for item in recipe["conditioning"]["inputs"]
            ]
            if any(type(frame) is not int or not 0 <= frame < geometry["published_frames"]
                   for frame in frames):
                raise ValueError("H3 picture alignment requires valid visible-frame anchors.")
            prefix = " ".join(
                f"Picture {index} is fully referenced at "
                f"{(frame + geometry['overlap_frames']) / 24:.6f} seconds."
                for index, frame in enumerate(sorted(frames), 1)
            )
            recipe["prompt"] = prefix + "\n" + prompt
        if (
            state["saveContext"]
            and state["mode"] != "motion"
            and abs(recipe["config"]["duration_seconds"] * 24 - geometry["published_frames"]) > 0.5
        ):
            raise ValueError(
                "Saving H3 motion context requires the clip to reach its generated endpoint. "
                f"Set duration to {geometry['published_frames'] / 24:.6f} seconds, "
                "or turn off Save motion context."
            )
        state.update(usableSourceIn=0, usableDuration=geometry["published_duration_seconds"])
    elif state["mode"] == "motion" and recipe["engine"] in {"ltx23", "ltx25"}:
        extension = recipe["conditioning"]["extension"]
        fps = recipe["config"].get("frame_rate", 24)
        requested = state["usableDuration"] * fps
        if abs(requested - extension["additional_frames"]) > 0.5:
            raise ValueError(
                "LTX motion continuation adds multiples of 8 frames. "
                f"Set duration to {extension['additional_frames'] / fps:.6f} seconds "
                "or use Match previous frame."
            )
        if recipe["engine"] == "ltx25":
            recipe["config"]["duration_seconds"] = (
                extension["context_frames"] + extension["additional_frames"] - 1
            ) / fps
        state["usableDuration"] = extension["additional_frames"] / fps
