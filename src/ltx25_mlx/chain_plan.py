"""Model-free LTX 2.5 native chain and delivered scene timing contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import accumulate


@dataclass(frozen=True)
class LTX25ChainPlan:
    """An exact, temporally aligned LTX 2.5 chained timeline."""

    total_frames: int
    window_count: int
    window_frames: int
    overlap_frames: int
    video_overlap_latent_frames: int
    window_audio_tokens: int
    join_audio_tokens: tuple[int, ...]
    frame_rate: float

    @property
    def window_frame_counts(self) -> tuple[int, ...]:
        return (self.window_frames,) * self.window_count

    @property
    def window_audio_token_counts(self) -> tuple[int, ...]:
        return (self.window_audio_tokens,) * self.window_count

    @property
    def expected_audio_tokens(self) -> int:
        return round(self.total_frames / self.frame_rate * 25.0)

    @property
    def window_start_frames(self) -> tuple[int, ...]:
        stride = self.window_frames - self.overlap_frames
        return tuple(index * stride for index in range(self.window_count))

    @property
    def assembled_video_seam_frames(self) -> tuple[int, ...]:
        """Frame indices where the assembled output first switches windows."""
        stride = self.window_frames - self.overlap_frames
        return tuple(self.window_frames + index * stride for index in range(self.window_count - 1))

    @property
    def assembled_audio_seam_tokens(self) -> tuple[int, ...]:
        """Audio-token indices where the assembled output first switches windows."""
        seams = []
        total = self.window_audio_tokens
        for trim in self.join_audio_tokens:
            seams.append(total)
            total += self.window_audio_tokens - trim
        return tuple(seams)

    def as_dict(self) -> dict[str, object]:
        return {
            "total_frames": self.total_frames,
            "window_count": self.window_count,
            "window_frames": self.window_frames,
            "overlap_frames": self.overlap_frames,
            "video_overlap_latent_frames": self.video_overlap_latent_frames,
            "window_audio_tokens": self.window_audio_tokens,
            "join_audio_tokens": list(self.join_audio_tokens),
            "window_start_frames": list(self.window_start_frames),
            "assembled_video_seam_frames": list(self.assembled_video_seam_frames),
            "assembled_audio_seam_tokens": list(self.assembled_audio_seam_tokens),
            "frame_rate": self.frame_rate,
            "delivered_duration_seconds": (self.total_frames - 1) / self.frame_rate,
        }


def plan_ltx25_chain(
    *,
    total_frames: int,
    window_count: int,
    overlap_frames: int,
    frame_rate: float,
) -> LTX25ChainPlan:
    """Resolve an exact equal-window chain on the LTX temporal grid."""
    if window_count < 2 or window_count > 4:
        raise ValueError("LTX 2.5 chained timelines support two to four windows.")
    if frame_rate <= 0:
        raise ValueError("LTX 2.5 chained timeline frame rate must be positive.")
    if total_frames < 1 or (total_frames - 1) % 8:
        raise ValueError("LTX 2.5 total frames must equal 8n+1.")
    if overlap_frames < 1 or (overlap_frames - 1) % 8:
        raise ValueError("LTX 2.5 overlap frames must equal 8n+1.")
    numerator = total_frames + (window_count - 1) * overlap_frames
    if numerator % window_count:
        raise ValueError(
            "The selected total, window count, and overlap do not produce equal integer windows."
        )
    window_frames = numerator // window_count
    if window_frames <= overlap_frames:
        raise ValueError("LTX 2.5 chained windows must be longer than their overlap.")
    if (window_frames - 1) % 8:
        raise ValueError("Resolved LTX 2.5 window frames must equal 8n+1.")

    window_audio_tokens = round(window_frames / frame_rate * 25.0)
    stride = window_frames - overlap_frames
    previous_total_audio = window_audio_tokens
    join_audio_tokens: list[int] = []
    for index in range(1, window_count):
        cumulative_frames = window_frames + index * stride
        cumulative_audio = round(cumulative_frames / frame_rate * 25.0)
        new_audio = cumulative_audio - previous_total_audio
        trim = window_audio_tokens - new_audio
        if trim <= 0 or trim >= window_audio_tokens:
            raise ValueError("Resolved LTX 2.5 audio overlap is invalid.")
        join_audio_tokens.append(trim)
        previous_total_audio = cumulative_audio

    return LTX25ChainPlan(
        total_frames=total_frames,
        window_count=window_count,
        window_frames=window_frames,
        overlap_frames=overlap_frames,
        video_overlap_latent_frames=(overlap_frames - 1) // 8 + 1,
        window_audio_tokens=window_audio_tokens,
        join_audio_tokens=tuple(join_audio_tokens),
        frame_rate=float(frame_rate),
    )


@dataclass(frozen=True)
class LTX25ScenePlan:
    window_frame_counts: tuple[int, ...]
    window_start_frames: tuple[int, ...]
    segment_frame_counts: tuple[int, ...]
    segment_start_frames: tuple[int, ...]
    total_frames: int
    overlap_frames: int
    frame_rate: float
    window_audio_token_counts: tuple[int, ...]
    join_audio_tokens: tuple[int, ...]
    requested_durations: tuple[float, ...] = ()

    @property
    def window_count(self) -> int:
        return len(self.window_frame_counts)

    @property
    def video_overlap_latent_frames(self) -> int:
        return (self.overlap_frames - 1) // 8 + 1

    @property
    def expected_audio_tokens(self) -> int:
        return round(self.total_frames / self.frame_rate * 25)

    def as_dict(self) -> dict[str, object]:
        return {
            "total_frames": self.total_frames,
            "window_count": self.window_count,
            "window_frame_counts": list(self.window_frame_counts),
            "window_start_frames": list(self.window_start_frames),
            "segment_frame_counts": list(self.segment_frame_counts),
            "segment_start_frames": list(self.segment_start_frames),
            "overlap_frames": self.overlap_frames,
            "video_overlap_latent_frames": self.video_overlap_latent_frames,
            "window_audio_token_counts": list(self.window_audio_token_counts),
            "join_audio_tokens": list(self.join_audio_tokens),
            "expected_audio_tokens": self.expected_audio_tokens,
            "frame_rate": self.frame_rate,
            "requested_durations": list(self.requested_durations),
            "resolved_durations": [
                frames / self.frame_rate for frames in self.segment_frame_counts
            ],
            "delivered_duration_seconds": (self.total_frames - 1) / self.frame_rate,
        }


def plan_ltx25_windows(window_frame_counts, *, overlap_frames=25, frame_rate=24) -> LTX25ScenePlan:
    """Validate explicit scene windows before any inference or checkpoint reads."""
    counts = tuple(window_frame_counts)
    if not 2 <= len(counts) <= 6:
        raise ValueError("LTX 2.5 scenes support two to six windows.")
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError("LTX 2.5 scene frame rate must be finite and positive.")
    if type(overlap_frames) is not int or overlap_frames < 9 or (overlap_frames - 1) % 8:
        raise ValueError("LTX 2.5 scene overlap must be 8n+1, at least 9 frames.")
    if any(
        type(count) is not int or count <= overlap_frames or (count - 1) % 8 for count in counts
    ):
        raise ValueError("LTX 2.5 scene windows must be 8n+1 and longer than the overlap.")
    segments = (counts[0] - 1,) + tuple(count - overlap_frames for count in counts[1:])
    boundaries = (0, *accumulate(segments))
    total = boundaries[-1] + 1
    if (total - 1) / frame_rate > 30:
        maximum = math.floor(30 * frame_rate / 8) * 8 / frame_rate
        raise ValueError(
            "Resolved LTX 2.5 scene exceeds 30 seconds; "
            f"use at most {maximum:g} seconds at {frame_rate:g} fps."
        )
    starts = (0,) + tuple(boundary - overlap_frames + 1 for boundary in boundaries[1:-1])
    audio_counts = tuple(round(count / frame_rate * 25) for count in counts)
    cumulative_audio = tuple(round((boundary + 1) / frame_rate * 25) for boundary in boundaries[1:])
    trims = tuple(
        audio_counts[i] - (cumulative_audio[i] - cumulative_audio[i - 1])
        for i in range(1, len(counts))
    )
    if any(
        trim <= 0 or trim >= min(audio_counts[i], audio_counts[i + 1])
        for i, trim in enumerate(trims)
    ):
        raise ValueError("LTX 2.5 scene audio overlap is invalid.")
    return LTX25ScenePlan(
        counts,
        starts,
        segments,
        boundaries[:-1],
        total,
        overlap_frames,
        float(frame_rate),
        audio_counts,
        trims,
    )


def plan_ltx25_scene(durations, *, overlap_frames=25, frame_rate=24) -> LTX25ScenePlan:
    """Quantize cumulative delivered boundaries, keeping rounding out of each shot."""
    from dataclasses import replace

    requested = tuple(float(duration) for duration in durations)
    if not 2 <= len(requested) <= 6:
        raise ValueError("LTX 2.5 scenes support two to six segments.")
    if any(not math.isfinite(duration) or duration <= 0 for duration in requested):
        raise ValueError("LTX 2.5 scene durations must be finite and positive.")
    if math.fsum(requested) > 30:
        raise ValueError("LTX 2.5 scenes support at most 30 seconds.")
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError("LTX 2.5 scene frame rate must be finite and positive.")
    boundaries = (
        0,
        *(
            round(math.fsum(requested[:i]) * frame_rate / 8) * 8
            for i in range(1, len(requested) + 1)
        ),
    )
    segments = tuple(b - a for a, b in zip(boundaries[:-1], boundaries[1:], strict=True))
    windows = (segments[0] + 1,) + tuple(frames + overlap_frames for frames in segments[1:])
    return replace(
        plan_ltx25_windows(windows, overlap_frames=overlap_frames, frame_rate=frame_rate),
        requested_durations=requested,
    )


def validate_boundary_image_policy(policy):
    if not isinstance(policy, str) or policy not in {"balanced", "strict"}:
        raise ValueError("LTX 2.5 boundary image policy must be balanced or strict.")


def _image_window_owner(plan, frame):
    """First covering window owns the anchor; later overlaps inherit its history."""
    return next(
        index for index, (start, count) in enumerate(
            zip(plan.window_start_frames, plan.window_frame_counts, strict=True)
        ) if start <= frame < start + count
    )


def boundary_image_guidance_report(images, plan, policy):
    """Expose anchor ownership without changing images, timestamps or strengths."""
    validate_boundary_image_policy(policy)
    by_frame = _validated_scene_images(images, plan)
    inherited = []
    if policy == "balanced":
        for index, (start, count) in enumerate(
            zip(plan.window_start_frames, plan.window_frame_counts, strict=True)
        ):
            for frame, image in sorted(by_frame.items()):
                owner = _image_window_owner(plan, frame)
                if owner < index and start <= frame < start + count:
                    inherited.append({
                        "window": index + 1, "source_window": owner + 1,
                        "frame_index": frame, "strength": image.strength,
                    })
    return {"policy": policy, "inherited_anchors": inherited}


def _validated_scene_images(images, plan):
    by_frame = {}
    for image in images or ():
        frame = image.frame_idx
        if type(frame) is not int or not 0 <= frame < plan.total_frames - 1:
            raise ValueError("LTX 2.5 image anchor is outside the delivered scene timeline.")
        if not math.isfinite(image.strength) or not 0 <= image.strength <= 1:
            raise ValueError("LTX 2.5 image strength must be between zero and one.")
        if frame in by_frame and image != by_frame[frame]:
            raise ValueError(f"Conflicting LTX 2.5 image anchors at global frame {frame}.")
        by_frame[frame] = image
    if len(by_frame) > 32:
        raise ValueError("LTX 2.5 scenes support at most 32 image anchors.")
    return by_frame


def route_scene_images(images, plan, *, boundary_image_policy="strict") -> tuple[tuple, ...]:
    """Route global anchors once automatically, or to every overlap in strict mode.

    Reapplying an image in a following overlap competes with the motion history
    already conditioned by it and can create an exposure pulse. Automatic mode
    retains every image at its requested strength in its first covering window;
    subsequent windows inherit that conditioning through native video history.
    Appended keyframes keep exact pixel timestamps, including the last delivered
    frame (native total minus two), without 8n snapping.
    """
    validate_boundary_image_policy(boundary_image_policy)
    by_frame = _validated_scene_images(images, plan)
    return tuple(
        tuple(
            image._replace(frame_idx=frame - start)
            for frame, image in sorted(by_frame.items())
            if start <= frame < start + count
            and (boundary_image_policy == "strict" or _image_window_owner(plan, frame) == index)
        )
        for index, (start, count) in enumerate(
            zip(plan.window_start_frames, plan.window_frame_counts, strict=True)
        )
    )


def scene_image_conditioning_bytes(window_images, *, height, width, single_stage):
    """Bound retained scene anchor tensors before any weighted component load."""
    full_area = (height // 32) * (width // 32)
    low_area = 0 if single_stage else (height // 64) * (width // 64)
    # 128 latent channels plus three temporal/spatial position coordinates.
    # Use float32 as an upper bound even when the encoder returns bfloat16.
    count = sum(len(images) for images in window_images)
    size = count * (full_area + low_area) * 131 * 4
    if size > 256 * 1024 * 1024:
        raise ValueError("LTX 2.5 scene image conditioning exceeds its 256 MiB limit.")
    return size
