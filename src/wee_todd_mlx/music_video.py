"""Weight-free music cues and exact editorial timing; no transcription or semantic claims."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

ANALYSIS_VERSION = "native-dsp-v1"


def analyze_samples(samples, sample_rate, *, check=None):
    """Independent short-time energy/spectral-flux implementation using existing NumPy."""
    check = check or (lambda: None)
    check()
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
        raise ValueError("Audio must contain finite mono samples")
    if not 1000 <= sample_rate <= 192000:
        raise ValueError("Unsupported analysis sample rate")
    hop = max(1, round(sample_rate * 0.02))
    size = 1 << max(8, round(sample_rate * 0.04).bit_length())
    window = np.hanning(size)
    previous = np.zeros(size // 2 + 1)
    energies, fluxes = [], []
    for index, offset in enumerate(range(0, samples.size, hop)):
        if index % 128 == 0:
            check()
        chunk = samples[offset : offset + size]
        padded = np.pad(chunk, (0, size - len(chunk)))
        spectrum = np.abs(np.fft.rfft(padded * window))
        energies.append(float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2))))
        fluxes.append(float(np.maximum(spectrum - previous, 0).sum()))
        previous = spectrum
    energy, flux = np.asarray(energies), np.asarray(fluxes)
    positive = flux[flux > 1e-9]
    cues, last_time = [], -1.0
    if positive.size:
        threshold = max(float(np.percentile(positive, 60)), float(flux.max()) * 0.08)
        for i in range(1, len(flux) - 1):
            if i % 1024 == 0:
                check()
            time = i * hop / sample_rate
            if flux[i] >= threshold and flux[i] > flux[i - 1] and flux[i] >= flux[i + 1]:
                if time - last_time >= 0.12:
                    cues.append(
                        {
                            "timeSeconds": round(time, 6),
                            "kind": "onset",
                            "strength": round(float(flux[i] / flux.max()), 5),
                            "provisional": True,
                        }
                    )
                    last_time = time
    # Coarse changes are dynamics hints, never inferred verse/chorus labels.
    stride = max(1, round(sample_rate / hop))
    envelope = [float(np.mean(energy[i : i + stride])) for i in range(0, len(energy), stride)]
    peak = max(envelope, default=0)
    if peak:
        for i in range(1, len(envelope)):
            difference = abs(envelope[i] - envelope[i - 1]) / peak
            if difference >= 0.25:
                cues.append(
                    {
                        "timeSeconds": float(i),
                        "kind": "energy_change",
                        "strength": round(difference, 5),
                        "provisional": True,
                    }
                )
    onset_times = [c["timeSeconds"] for c in cues if c["kind"] == "onset"]
    intervals = np.diff(onset_times)
    intervals = intervals[(intervals >= 0.25) & (intervals <= 1.5)]
    beat = None
    if len(intervals) >= 3:
        median = float(np.median(intervals))
        if float(np.median(np.abs(intervals - median))) < 0.12 * median:
            beat = {
                "bpm": round(60 / median, 2),
                "provisional": True,
                "basis": "median stable onset spacing; metrical level unverified",
            }
    check()
    return {
        "method": ANALYSIS_VERSION,
        "sampleRate": sample_rate,
        "durationSeconds": len(samples) / sample_rate,
        "cues": sorted(cues, key=lambda c: c["timeSeconds"]),
        "energyEnvelope": envelope,
        "energyHopSeconds": 1,
        "beatSuggestion": beat,
        "limitations": [
            "DSP cues are provisional; no learned beat/downbeat analysis.",
            "No transcription, word alignment, or semantic song-section detection.",
        ],
    }


def verify_source_audio(source, expected_sha256, *, check=None):
    check = check or (lambda: None)
    check()
    source = Path(source).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("Select a source audio file")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            check()
            digest.update(chunk)
    content_hash = digest.hexdigest()
    if content_hash != expected_sha256:
        raise ValueError("Source audio changed; refresh its content hash before planning")
    return source, content_hash


def _decode_audio(command, *, check, timeout=180):
    """Drain ffmpeg pipes while giving the workflow a chance to pause every 100ms."""
    deadline = time.monotonic() + timeout
    check()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            check()
            if time.monotonic() >= deadline:
                raise TimeoutError("Source audio decoding exceeded its time limit")
            try:
                output, error = process.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.communicate()
        raise
    check()
    if process.returncode:
        raise ValueError("Cannot decode source audio: " + error.decode(errors="replace")[-600:])
    return output


def analyze_audio(source, *, expected_sha256, cache_directory, sample_rate=8000, check=None):
    from .conditioning_media import media_binary

    check = check or (lambda: None)
    source_reference = source
    source, content_hash = verify_source_audio(source, expected_sha256, check=check)
    settings = {"sha256": content_hash, "sampleRate": sample_rate, "method": ANALYSIS_VERSION}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    directory = Path(cache_directory)
    destination = directory / (key + ".json")
    if destination.is_file():
        try:
            saved = json.loads(destination.read_text())
            if saved["cacheKey"] == key and saved["sourceSHA256"] == content_hash:
                return saved
        except (ValueError, KeyError):
            pass
    # Bound decoding to one hour plus a sentinel sample. Never rewrite the original song.
    command = [
        media_binary({}, "ffmpeg"),
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-t",
        "3600.001",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    samples = np.frombuffer(_decode_audio(command, check=check), dtype="<f4")
    if len(samples) > sample_rate * 3600:
        raise ValueError("Music analysis supports source audio up to one hour")
    analysis = analyze_samples(samples, sample_rate, check=check)
    verify_source_audio(source_reference, expected_sha256, check=check)
    analysis.update(
        sourceSHA256=content_hash,
        cacheKey=key,
        decode={"channels": 1, "sampleRate": sample_rate, "format": "float32"},
    )
    directory.mkdir(parents=True, exist_ok=True)
    # Atomic publication permits concurrent runs to share the immutable cache.
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as stream:
        json.dump(analysis, stream, allow_nan=False)
        temporary = Path(stream.name)
    temporary.replace(destination)
    return analysis


def clip_bounds(
    engine,
    task,
    fps,
    *,
    minimum_seconds=0,
    maximum_seconds=15,
    backend_minimum_seconds=0,
    backend_maximum_seconds=0,
):
    numeric = [
        fps,
        minimum_seconds,
        maximum_seconds,
        backend_minimum_seconds,
        backend_maximum_seconds,
    ]
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in numeric
    ):
        raise ValueError("Timing bounds must be finite numbers")
    if task not in {"a2v", "t2v"}:
        raise ValueError("Music planning supports a2v or t2v soundtrack mode")
    if engine in {"ltx25", "ltx23"}:
        low, high, basis = 0.25, 30.0, engine + ".runtime.GenerationConfig.validate"
        if not 1 <= fps <= 60:
            raise ValueError("LTX requires FPS between 1 and 60")
    elif engine == "h3":
        low, high, basis = 2.5, 15.0, "minimax_h3_mlx.pipeline duration contract"
        if fps != 24:
            raise ValueError("H3 requires 24 FPS")
    elif engine == "drawThings":
        if not 0 < backend_minimum_seconds <= backend_maximum_seconds:
            raise ValueError(
                "Select a Draw Things model with explicit task capability duration bounds"
            )
        low, high, basis = (
            backend_minimum_seconds,
            backend_maximum_seconds,
            "selected model capability",
        )
    else:
        raise ValueError("Select a supported generation engine")
    if minimum_seconds and minimum_seconds < low:
        raise ValueError("Minimum clip duration is below the supported model minimum")
    minimum, maximum = max(low, minimum_seconds), min(high, maximum_seconds)
    if not all(math.isfinite(v) for v in [minimum, maximum, fps]) or not 1 <= fps <= 120:
        raise ValueError("Timing bounds must be finite and FPS must be 1–120")
    if not 0 < minimum <= maximum:
        raise ValueError("Minimum clip duration must fit the selected maximum")
    return {
        "minimumFrames": math.ceil(minimum * fps),
        "maximumFrames": math.floor(maximum * fps),
        "minimumSeconds": minimum,
        "maximumSeconds": maximum,
        "basis": basis,
        "modelMinimumSeconds": low,
        "modelMaximumSeconds": high,
    }


def optimize_timing(
    *,
    total_frames,
    fps,
    minimum_frames,
    maximum_frames,
    preferred_frames,
    cues=(),
    locked_frames=(),
    protected_ranges=(),
    quantum=1,
    max_clips=200,
    check=None,
):
    """DP with an adaptive cut penalty; locks and exact coverage are always hard constraints.

    A single cheapest state cannot safely enforce a hard count by discarding later edges.
    Instead, solve unconstrained count for each cut penalty, then raise the penalty until
    its optimal path fits the budget. A dominating final penalty minimizes count and proves
    feasibility; bounded bisection preserves useful creative pacing. This is not a claim of
    globally optimal original pacing cost under the hard count constraint.
    """
    check = check or (lambda: None)
    check()
    if not 1 <= minimum_frames <= maximum_frames or not 1 <= total_frames <= 432000:
        raise ValueError("Invalid timing coverage bounds")
    if quantum not in {1, 8}:
        raise ValueError("Unsupported timing quantum")
    if any(type(frame) is not int or not 0 < frame < total_frames for frame in locked_frames):
        raise ValueError("Locked markers must be interior integer frames")
    protected = np.zeros(total_frames + 1, dtype=bool)
    for first, last in protected_ranges:
        if (
            type(first) is not int
            or type(last) is not int
            or not 0 <= first <= last <= total_frames
        ):
            raise ValueError("Protected word ranges must be bounded integer frames")
        protected[first + 1 : last] = True
    rewards = np.zeros(total_frames + 1)
    for cue in cues:
        frame, strength = cue["frame"], cue["strength"]
        if type(frame) is not int or not 0 <= frame <= total_frames or not 0 <= strength <= 1:
            raise ValueError("Cue frames and strengths must be in range")
        # A cue can attract a nearby legal frame-grid boundary without claiming exact beat timing.
        for target in range(max(0, frame - 2), min(total_frames, frame + 2) + 1):
            rewards[target] = max(rewards[target], strength * (1 - abs(target - frame) / 3))
    preferred_frames = min(maximum_frames, max(minimum_frames, preferred_frames))
    if type(max_clips) is not int or max_clips < 1:
        raise ValueError("Shot budget must be a positive integer")
    stops = sorted(set(locked_frames)) + [total_frames]
    if total_frames > max_clips * maximum_frames or len(stops) > max_clips:
        raise ValueError("Cannot cover source interval within the shot budget")
    duration_costs = (
        0.45 + 2 * ((np.arange(maximum_frames + 1) - preferred_frames) / preferred_frames) ** 2
    )

    def solve(cut_penalty):
        check()
        boundaries = [0]
        for stop in stops:
            start = boundaries[-1]
            length = stop - start
            score = np.full(length + 1, np.inf)
            parent = np.full(length + 1, -1, dtype=np.int64)
            score[0] = 0
            for end in range(minimum_frames, length + 1):
                if end % 256 == 0:
                    check()
                if quantum > 1 and (start + end) % quantum and end != length:
                    continue
                low, high = max(0, end - maximum_frames), end - minimum_frames
                costs = score[low : high + 1] + duration_costs[end - high : end - low + 1][::-1]
                index = int(np.argmin(costs))
                if np.isfinite(costs[index]):
                    score[end] = costs[index] + cut_penalty
                    if end != length:
                        score[end] -= 0.35 * rewards[start + end]
                        score[end] += 2.0 * protected[start + end]
                    parent[end] = low + index
            if parent[length] < 0:
                raise ValueError(
                    "Cannot cover source interval with these clip bounds and locked markers"
                )
            segment, cursor = [], length
            while cursor:
                segment.append(start + cursor)
                cursor = int(parent[cursor])
            boundaries.extend(reversed(segment))
        return boundaries

    boundaries = solve(0)
    if len(boundaries) - 1 > max_clips:
        lower = 0.0
        upper = max(1.0, 2 * ((total_frames / (max_clips * preferred_frames)) ** 2 - 1))
        # Every path's nonnegative base cost is at most this value. One additional
        # cut then costs more than any possible improvement in all other edge costs.
        dominating = (total_frames // minimum_frames) * float(
            duration_costs[minimum_frames:].max() + 2.0
        ) + 1
        for _ in range(8):
            candidate = solve(min(upper, dominating))
            if len(candidate) - 1 <= max_clips:
                break
            lower, upper = upper, upper * 2
        else:
            upper = dominating
            candidate = solve(upper)
            if len(candidate) - 1 > max_clips:
                raise ValueError("Cannot cover source interval within the shot budget")
        boundaries = candidate
        for _ in range(6):
            if len(boundaries) - 1 == max_clips:
                break
            middle = (lower + upper) / 2
            candidate = solve(middle)
            if len(candidate) - 1 <= max_clips:
                boundaries, upper = candidate, middle
            else:
                lower = middle
    clips = [
        {
            "id": f"clip-{i + 1}",
            "startFrame": a,
            "frameCount": b - a,
            "action": "",
            "continuity": "cut" if i == 0 else "continue",
        }
        for i, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True))
    ]
    return {"fps": fps, "totalFrames": total_frames, "clips": clips}
