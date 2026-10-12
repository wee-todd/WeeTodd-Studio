"""Bounded audio visualization and caption geometry; no model imports at module load."""

import math

import numpy as np


def presentation_duration(project):
    from .audio_mix.plan import number

    cursor, positions = 0.0, {}
    clips = project.get("clips", [])
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
        cursor += length
    for region in project.get("audio", []):
        anchor = region.get("anchor")
        if anchor and anchor["clipID"] not in positions:
            raise ValueError("An audio region references a missing clip")
        start = (
            positions[anchor["clipID"]] + number(anchor["offsetSeconds"], "offset")
            if anchor
            else number(region.get("start", 0), "start")
        )
        cursor = max(cursor, start + number(region["duration"], "duration", 1 / 48000))
    for title in project.get("titles", []):
        cursor = max(
            cursor,
            number(title["start"], "title start")
            + number(title["duration"], "title duration", 0.00001),
        )
    return number(cursor, "presentation duration", 1 / 48000)


def audio_features(samples, rate):
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        # Avoid cancelling opposite-polarity stereo channels.
        samples = samples[:, np.argmax(np.mean(samples**2, axis=0))]
    if (
        samples.ndim != 1
        or not np.isfinite(samples).all()
        or not 8000 <= rate <= 192000
        or len(samples) > rate * 3600
    ):
        raise ValueError("Choose finite audio of at most one hour")
    n, hop = 2048, 0.1
    padded = np.pad(samples, (n // 2, n))
    edges = np.geomspace(40, min(rate / 2, 16000), 33)
    frequencies = np.fft.rfftfreq(n, 1 / rate)
    bins = [
        np.flatnonzero((frequencies >= a) & (frequencies < b))
        for a, b in zip(edges, edges[1:], strict=False)
    ]
    window = np.hanning(n)
    waves, spectra = [], []
    for frame in range(math.ceil(len(samples) / rate / hop) + 1):
        start = round(frame * rate * hop)
        block = padded[start : start + n]
        if len(block) < n:
            block = np.pad(block, (0, n - len(block)))
        spectrum = np.abs(np.fft.rfft(block * window)) * 2 / window.sum()
        spectra.append(
            [round(float(min(1, np.sqrt(spectrum[b].max()) if len(b) else 0)), 4) for b in bins]
        )
        waves.append(np.round(np.clip(block.reshape(64, 32).mean(axis=1) * 3, -1, 1), 4).tolist())
    return dict(hop=hop, waveform=waves, spectrum=spectra)


def caption_cues(words, duration):
    cues, group = [], []

    def flush():
        if group:
            cues.append(
                dict(
                    start=group[0]["startSeconds"],
                    end=min(duration, group[-1]["endSeconds"]),
                    text=" ".join(w["text"] for w in group),
                )
            )
            group.clear()

    for word in words:
        start, end = word.get("startSeconds"), word.get("endSeconds")
        if (
            not isinstance(start, (float, int))
            or not isinstance(end, (float, int))
            or not math.isfinite(start + end)
            or not 0 <= start < end <= duration + 0.05
        ):
            continue
        if not str(word.get("text", "")).strip():
            continue
        if group and (
            start - group[-1]["endSeconds"] > 0.6
            or end - group[0]["startSeconds"] > 4
            or len(" ".join(w["text"] for w in group)) > 48
        ):
            flush()
        group.append(word)
    flush()
    return cues
