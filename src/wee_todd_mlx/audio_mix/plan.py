"""Pure project-to-audio compiler. All times resolve before file I/O."""

from __future__ import annotations

import math

from .reverb import validate_reverb


def number(value, label, low=0, high=3600):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"Invalid audio {label}")
    return float(value)


def compile_mix(project, *, start=0, duration=None, purpose="preview", selection=None):
    if purpose not in {"preview", "export", "driver"}:
        raise ValueError("Unknown audio mix purpose")
    policy = project.get("audioMixPolicy") or "legacy-v1"
    if policy not in {"legacy-v1", "studio-v1"}:
        raise ValueError("Unsupported audio mix policy")
    tracks = {t["id"]: t for t in project.get("audioTracks", [])}
    if len(tracks) != len(project.get("audioTracks", [])):
        raise ValueError("Duplicate audio track IDs")
    reverbs = {}
    for track in tracks.values():
        number(track.get("gainDb", 0), "track gain", -60, 12)
        number(track.get("pan", 0), "pan", -1, 1)
        reverbs[track["id"]] = validate_reverb(track.get("reverb"))
    clips, positions, overlaps = project.get("clips", []), {}, []
    cursor = 0
    for i, clip in enumerate(clips):
        length = number(clip["duration"], "clip duration", 0.00001)
        overlap = (
            0
            if i == 0 or clip.get("transition", "cut") == "cut"
            else min(
                number(clip.get("transitionDuration", 0.5), "transition"),
                length / 2,
                clips[i - 1]["duration"] / 2,
            )
        )
        cursor -= overlap
        positions[clip["id"]] = cursor
        overlaps.append(overlap)
        cursor += length
    start = number(start, "mix start")
    duration = number(cursor - start if duration is None else duration, "mix duration", 1 / 48000)
    if start + duration > 3600:
        raise ValueError("Audio preparation supports a maximum one-hour window")
    selection = selection or {"mode": "voiceAndMusic"}
    mode = selection.get("mode", "voiceAndMusic")
    if mode not in {"voice", "music", "voiceAndMusic"}:
        raise ValueError("Choose Voice, Music or Voice + Music as driver")
    selected = set()
    if purpose == "driver":
        for role in ("voice", "music"):
            if mode not in {role, "voiceAndMusic"}:
                continue
            explicit = selection.get(role + "TrackIDs", [])
            if any(
                i not in tracks
                or tracks[i].get("role", "music" if tracks[i].get("name") == "Music" else "other")
                != role
                for i in explicit
            ):
                raise ValueError("A selected audio driver track is missing or has changed role")
            selected.update(
                explicit
                or [
                    i
                    for i, t in tracks.items()
                    if t.get("role", "music" if t.get("name") == "Music" else "other") == role
                ]
            )
    solo = (purpose == "preview" or (purpose == "export" and policy == "legacy-v1")) and any(
        t.get("solo", False) for t in tracks.values()
    )
    inputs, replacements = [], []
    for region in project.get("audio", []):
        track = tracks.get(region.get("trackID"), {})
        if (
            track.get("muted")
            or (solo and not track.get("solo"))
            or (purpose == "driver" and region.get("trackID") not in selected)
        ):
            continue
        anchor = region.get("anchor")
        if anchor and anchor.get("clipID") not in positions:
            raise ValueError("An audio region references a missing clip")
        at = (
            positions[anchor["clipID"]] + number(anchor["offsetSeconds"], "clip offset")
            if anchor
            else number(region.get("start", 0), "region start")
        )
        length = number(region["duration"], "region duration", 1 / 48000)
        if at >= start + duration + 0.05 or at + length <= 0:
            continue
        fade_in = number(region.get("fadeIn", region.get("fade", 0.2)), "fade in")
        fade_out = number(region.get("fadeOut", region.get("fade", 0.2)), "fade out")
        if policy == "legacy-v1":
            if region.get("fadeIn") is None:
                fade_in = min(fade_in, length / 2)
            if region.get("fadeOut") is None:
                fade_out = min(fade_out, length / 2)
        curve = region.get("fadeCurve") or "linear"
        if curve not in {"linear", "equalPower"}:
            raise ValueError("Unsupported audio fade curve")
        window = region.get("envelope")
        envelope_offset, envelope_duration = 0, length
        if window:
            envelope_offset = number(window["offset"], "envelope offset")
            envelope_duration = number(window["duration"], "envelope duration", 1 / 48000)
            fade_in = number(window["fadeIn"], "envelope fade in")
            fade_out = number(window["fadeOut"], "envelope fade out")
            curve = window["curve"]
            if curve not in {"linear", "equalPower"}:
                raise ValueError("Unsupported envelope curve")
        role = track.get("role", "music" if track.get("name") == "Music" else "other")
        inputs.append(
            dict(
                id=region["id"],
                path=region["path"],
                start=at,
                duration=length,
                source_in=number(region.get("sourceIn", 0), "source trim"),
                gain=number(region.get("volume", 0.8), "region volume", 0, 4)
                * 10 ** (track.get("gainDb", 0) / 20),
                pan=track.get("pan", 0),
                fade_in=min(fade_in, envelope_duration),
                fade_out=min(fade_out, envelope_duration),
                envelope_offset=envelope_offset,
                envelope_duration=envelope_duration,
                curve=curve,
                role=role,
                ducking=track.get("ducking") if role == "music" else None,
                replacements=[],
            )
        )
        if effect := reverbs.get(region.get("trackID")):
            inputs[-1].update(reverb=effect, bus=region["trackID"])
        if track.get("replacesSource"):
            replacements.append([at, at + length])
    if purpose != "driver" and not solo:
        for i, clip in enumerate(clips):
            path = clip.get("sourcePath", "")
            if not path or clip.get("volume", 1) == 0:
                continue
            inputs.append(
                dict(
                    id=clip["id"],
                    path=path,
                    start=positions[clip["id"]],
                    duration=clip["duration"],
                    source_in=number(clip.get("sourceIn", 0), "source trim"),
                    gain=number(clip.get("volume", 1), "source volume", 0, 2),
                    pan=number(clip.get("sourcePan", 0) or 0, "source pan", -1, 1),
                    fade_in=overlaps[i],
                    fade_out=overlaps[i + 1] if i + 1 < len(clips) else 0,
                    curve="linear",
                    role="source",
                    ducking=None,
                    replacements=replacements,
                )
            )
    for item in inputs:
        d = item["ducking"]
        if d:
            number(d.get("amountDb", 12), "duck amount", 0, 36)
            number(d.get("thresholdDb", -36), "duck threshold", -80, 0)
            number(d.get("attack", 0.02), "duck attack", 0.001, 5)
            number(d.get("release", 0.25), "duck release", 0.001, 10)
    return dict(
        version=1,
        policy=policy,
        start=start,
        duration=duration,
        frames=round(duration * 48000),
        sample_rate=48000,
        channels=2,
        purpose=purpose,
        inputs=inputs,
    )
