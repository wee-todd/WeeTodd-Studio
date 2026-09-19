"""Guided music-video planning through existing human-owned brief/object/shot reviews."""

from __future__ import annotations

import copy
import json
import math
import re

from ..music_video import analyze_audio, clip_bounds, optimize_timing
from .creative import prepare_brief, require_answers
from .operations import parse_value


def prepare(inputs, ctx):
    source = inputs["brief"]
    prepared = prepare_brief(
        {**inputs, "brief": source or "Create a music video for the supplied track."}, ctx
    )
    prepared["sourceText"] = source
    if not source.strip():
        prepared["facts"] = []
        prepared["questions"].insert(
            0,
            {
                "id": "music-concept",
                "prompt": "What should the music video show?",
                "options": ["A performance", "A visual story", "Abstract imagery"],
                "requiresExplicitChoice": False,
                "answer": "",
            },
        )
    status = inputs["lyrics_status"]
    if status not in {"unknown", "supplied", "instrumental"}:
        raise ValueError("Choose supplied, unknown, or instrumental lyrics status")
    add_music_lyrics_question(prepared, inputs["lyrics"], status)
    # Keep schema's maximum six questions; deterministic essential questions take precedence.
    prepared["questions"] = sorted(
        prepared["questions"], key=lambda q: not q["id"].startswith("music-")
    )[:6]
    return {"creative_brief": prepared}


_NO_LIP_SYNC = re.compile(
    r"\b(?:(?:no|without)\s+(?:lip[\s-]*sync(?:ing)?|singing|vocals?)|visual[\s-]+only)\b", re.I
)
_LIP_SYNC = re.compile(r"\blip[\s-]*sync(?:ing|hronization)?\b", re.I)
_PERFORMANCE = re.compile(
    r"\b(?:sing|sings|singing|singer|singers|sang|sung|vocal|vocals|vocalist|vocalists|performance)\b",
    re.I,
)


def needs_music_lyrics(brief, lyrics, lyrics_status="unknown"):
    """Recheck current user choices; lexical boundaries must not mistake 'single' for singing."""
    if lyrics_status == "supplied" and lyrics.strip():
        return False
    answers = [q.get("answer", "") for q in brief["questions"] if q["id"] != "music-lyrics"]
    preferences = [v for v in brief.get("preferences", {}).values() if isinstance(v, str)]
    source = brief.get("sourceText", "")
    lyric_answers = [q.get("answer", "") for q in brief["questions"] if q["id"] == "music-lyrics"]
    waiver = any(_NO_LIP_SYNC.search(text) for text in lyric_answers)
    # A prior visual-only answer resolves the original request, but cannot authorize a new
    # conflicting lip-sync request in a later answer or editable creative preference.
    explicit_new = any(
        _LIP_SYNC.search(_NO_LIP_SYNC.sub("", text)) for text in answers + preferences
    )
    explicit_source = _LIP_SYNC.search(_NO_LIP_SYNC.sub("", source)) is not None
    if explicit_new or explicit_source and not waiver:
        return True
    if waiver or any(_NO_LIP_SYNC.search(text) for text in [source, *answers, *preferences]):
        return False
    if lyric_answers or lyrics_status == "supplied":
        return True
    return lyrics_status != "instrumental" and any(
        _PERFORMANCE.search(text) for text in [source, *answers, *preferences]
    )


def add_music_lyrics_question(brief, lyrics, lyrics_status="unknown"):
    if not needs_music_lyrics(brief, lyrics, lyrics_status):
        return
    if any(q["id"] == "music-lyrics" for q in brief["questions"]):
        return
    brief["questions"].append(
        {
            "id": "music-lyrics",
            "prompt": (
                "Supply the lyrics in the music-video intake, or choose "
                "visual-only performance without lip sync."
            ),
            "options": ["I will supply lyrics", "Visual-only performance without lip sync"],
            "requiresExplicitChoice": True,
            "answer": "",
        }
    )


def require_music_answers(brief, lyrics, lyrics_status="unknown"):
    require_answers(brief)
    if needs_music_lyrics(brief, lyrics, lyrics_status):
        raise ValueError(
            "Supply lyrics in the music-video intake, or edit the Brief answers/preferences "
            "to explicitly choose visual-only performance without lip sync"
        )


def timing(inputs, ctx):
    analysis = inputs["analysis"]
    brief = inputs["creative_brief"]
    require_music_answers(brief, inputs["lyrics"], inputs["lyrics_status"])
    fps = brief["preferences"]["frameRate"]
    duration = brief["preferences"]["durationSeconds"]
    source_start = inputs["source_start_seconds"]
    if not math.isfinite(source_start) or source_start < 0 or duration <= 0:
        raise ValueError("Choose a finite source start and positive duration")
    if source_start + duration > analysis["durationSeconds"] + 1 / analysis["sampleRate"]:
        raise ValueError("Selected song interval extends past the source audio")
    bounds = clip_bounds(
        inputs["generation_engine"],
        inputs["generation_task"],
        fps,
        minimum_seconds=inputs["min_clip_seconds"],
        maximum_seconds=inputs["max_clip_seconds"],
        backend_minimum_seconds=inputs["backend_min_seconds"],
        backend_maximum_seconds=inputs["backend_max_seconds"],
    )
    total = math.ceil(duration * fps - 1e-9)
    # Creative pacing is a soft preference. It cannot move locked markers or change coverage.
    preferred = parse_value(
        ctx.ask(
            "Choose a preferred shot duration in seconds for this music-video concept. "
            "Return ONLY a JSON number within bounds. This is a soft creative pacing preference; "
            "The host allocates frames using audio evidence and reviewed markers. "
            "Do not invent lyrics, "
            "choruses, word times, or confident beats from these cues.",
            json.dumps(
                {
                    "brief": brief["sourceText"],
                    "answers": brief["questions"],
                    "pacing": inputs["pacing"],
                    "bounds": bounds,
                    "beatSuggestion": analysis["beatSuggestion"],
                }
            ),
        ),
        "number",
    )
    if (
        not math.isfinite(preferred)
        or not bounds["minimumSeconds"] <= preferred <= bounds["maximumSeconds"]
    ):
        raise ValueError("Pacing suggestion must fit the selected duration bounds")
    from ..audio_analysis.structure import editing_cues

    learned_cues = editing_cues(analysis, fps=fps, source_start=source_start, duration=duration)
    cues = [
        {"frame": round((c["timeSeconds"] - source_start) * fps), "strength": c["strength"]}
        for c in analysis["cues"]
        if source_start <= c["timeSeconds"] <= source_start + duration
    ]
    cues = [c for c in cues if 0 <= c["frame"] <= total]
    cues.extend({"frame": c["frame"], "strength": c["confidence"]} for c in learned_cues)
    from .io import _constant, _pairs

    markers = json.loads(
        inputs["timing_markers"], object_pairs_hook=_pairs, parse_constant=_constant
    )
    if not isinstance(markers, list) or len(markers) > 1000:
        raise ValueError("Timing markers must be a JSON list of at most 1000 source timestamps")
    locks = []
    for marker in markers:
        if not isinstance(marker, dict) or set(marker) - {"timeSeconds", "locked", "label"}:
            raise ValueError(
                "Markers accept only timeSeconds, locked, and an optional supplied label"
            )
        second = marker.get("timeSeconds")
        if (
            isinstance(second, bool)
            or not isinstance(second, (int, float))
            or not math.isfinite(second)
        ):
            raise ValueError("Marker timeSeconds must be finite")
        if not source_start < second < source_start + duration:
            raise ValueError("Markers must lie inside the selected source interval")
        if type(marker.get("locked", False)) is not bool:
            raise ValueError("Marker locked must be boolean")
        frame = math.floor((second - source_start) * fps + 0.5)
        cues.append({"frame": frame, "strength": 1.0})
        if marker.get("locked"):
            locks.append(frame)
    quantum = 8 if inputs["continuity_enabled"] and inputs["generation_engine"] == "ltx25" else 1
    plan = optimize_timing(
        total_frames=total,
        fps=fps,
        minimum_frames=bounds["minimumFrames"],
        maximum_frames=bounds["maximumFrames"],
        preferred_frames=round(preferred * fps),
        cues=cues,
        locked_frames=locks,
        protected_ranges=[
            (
                max(0, round((w["startSeconds"] - source_start) * fps)),
                min(total, round((w["endSeconds"] - source_start) * fps)),
            )
            for w in (analysis.get("alignment") or {}).get("words", [])
            if w.get("startSeconds") is not None
            and w.get("confidence", 0) >= 0.5
            and w["endSeconds"] > source_start
            and w["startSeconds"] < source_start + duration
        ],
        quantum=quantum,
        check=getattr(ctx, "check", None),
    )
    clips = []
    for clip in plan["clips"]:
        frames = clip["frameCount"]
        render_frames = (
            math.ceil(frames / 8) * 8 + 1
            if inputs["generation_engine"] in {"ltx25", "ltx23"}
            else frames
        )
        if inputs["generation_engine"] == "h3":
            render_frames = frames + (5 - frames) % 17
        source_duration = min(frames / fps, duration - clip["startFrame"] / fps)
        fractional_tail = source_duration < frames / fps - 1e-9
        clips.append(
            {
                "clipID": clip["id"],
                "startFrame": clip["startFrame"],
                "frameCount": frames,
                "sourceStartSeconds": source_start + clip["startFrame"] / fps,
                "sourceDurationSeconds": source_duration,
                "renderFrames": render_frames,
                "extraTailFrames": render_frames - frames,
                "sceneEligible": quantum == 8
                and frames % 8 == 0
                and frames >= 8
                and not fractional_tail,
            }
        )
    lyrics = inputs["lyrics"] if inputs["lyrics_status"] == "supplied" else ""
    alignment = analysis.get("alignment") or {}
    return {
        "music_timing": {
            "sourceAudio": {
                "path": inputs["audio_path"],
                "sha256": inputs["audio_sha256"],
                "sourceStartSeconds": source_start,
                "sourceEndSeconds": source_start + duration,
            },
            "fps": fps,
            "totalFrames": total,
            "clips": clips,
            "allocation": plan,
            "engine": inputs["generation_engine"],
            "task": inputs["generation_task"],
            "suppliedLyrics": lyrics,
            "lyricStatus": alignment.get("status")
            or (
                "supplied_unaligned"
                if lyrics.strip()
                else ("instrumental" if inputs["lyrics_status"] == "instrumental" else "unknown")
            ),
            "analysis": analysis,
            "bounds": bounds,
            "preferredSeconds": preferred,
            "timingQuantum": quantum,
            "markers": markers,
            "warnings": [
                "Audio word and section suggestions require review; scores are uncalibrated."
                if alignment
                else "Supplied lyrics are unaligned; word times are not inferred.",
                "Render handles must be trimmed to each exact source interval; "
                "retain the original song bed.",
            ],
        }
    }


def execute_music(operation, inputs, parameters, ctx):
    if operation in {"music.analyze@2", "music.analyze@3"}:
        from ..audio_analysis.service import analyze

        return {
            "analysis": analyze(
                inputs["audio_path"],
                expected_sha256=inputs["audio_sha256"],
                model_directory=inputs["analysis_model_directory"],
                lyrics=inputs["lyrics"],
                lyrics_status=inputs["lyrics_status"],
                mode=inputs["analysis_mode"],
                vocal_mode=inputs.get("analysis_vocal_mode", "mixed"),
                cache_directory=ctx.runner.directory.parent / "music-analysis-cache",
                check=ctx.check,
            )
        }
    if operation == "music.analyze@1":
        ctx.check()
        result = analyze_audio(
            inputs["audio_path"],
            expected_sha256=inputs["audio_sha256"],
            cache_directory=ctx.runner.directory.parent / "music-analysis-cache",
            check=ctx.check,
        )
        ctx.check()
        return {"analysis": result}
    if operation == "music.prepare_brief@1":
        return prepare(inputs, ctx)
    if operation == "music.plan_timing@1":
        return timing(inputs, ctx)
    if operation in {"music.plan_treatment@1", "music.plan_beats@1"}:
        from .creative import execute_creative
        from .structured import plan_beats

        bounded = {key: value for key, value in inputs.items() if key != "music_timing"}
        bounded["_allocation"] = copy.deepcopy(inputs["music_timing"]["allocation"])
        bounded["_music_timing"] = inputs["music_timing"]
        if operation == "music.plan_beats@1":
            require_answers(inputs["creative_brief"])
            return plan_beats(bounded, parameters, ctx)
        from .music_context import treatment_brief

        bounded["brief"] = treatment_brief(bounded["brief"], inputs["music_timing"])
        return execute_creative("movie.plan_treatment@1", bounded, parameters, ctx)
    if operation == "music.check_plan@1":
        from .structured import execute_structured

        bounded = {key: value for key, value in inputs.items() if key != "music_timing"}
        bounded["duration_seconds"] = (
            inputs["music_timing"]["totalFrames"] / inputs["music_timing"]["fps"]
        )
        return execute_structured("movie.check_plan@2", bounded, parameters, ctx)
    if operation == "music.compile_prompts@1":
        from .creative import compile_prompts

        review = inputs["planning_review"]
        if review.get("structure") == "invalid" or any(
            i["severity"] == "error" for i in review["items"]
        ):
            raise ValueError("Resolve structural errors before previewing prompts")
        preview = compile_prompts(
            inputs["creative_brief"], inputs["clips"], inputs["subjects"], model_neutral=True
        )
        preview["warnings"] += ["Plan review: " + item["message"] for item in review["items"]]
        preview["warnings"] = list(dict.fromkeys(preview["warnings"]))[:32]
        return {"prompt_plan": preview}
    raise ValueError("Unknown music planning operation")
