"""Bounded musical structure suggestions and editable frame-snapped cut candidates."""

from __future__ import annotations

import math

import numpy as np


def structure_analysis(samples22050, *, beats, downbeats, words=None, lines=None, check=None):
    check = check or (lambda: None)
    check()
    audio = np.asarray(samples22050, dtype=np.float32)
    duration = len(audio) / 22050
    bars = []
    for left, right in zip(downbeats, downbeats[1:], strict=False):
        start, end = left["timeSeconds"], right["timeSeconds"]
        if not 0 <= start < end <= duration:
            continue
        count = sum(start - 0.04 <= b["timeSeconds"] < end - 0.04 for b in beats)
        bars.append(
            dict(
                startSeconds=start,
                endSeconds=end,
                beatCount=count,
                confidence=min(left["confidence"], right["confidence"]),
                inferred=True,
            )
        )
    # One feature per second bounds analysis at 3600 feature rows, without an N² song matrix.
    features = []
    size = 8192
    frequencies = np.fft.rfftfreq(size, 1 / 22050)
    mask = (frequencies >= 65) & (frequencies <= 4000)
    pitches = np.rint(69 + 12 * np.log2(frequencies[mask] / 440)).astype(int) % 12
    for second in range(math.ceil(duration)):
        check()
        chunk = audio[second * 22050 : second * 22050 + size]
        spectrum = np.abs(np.fft.rfft(np.pad(chunk, (0, size - len(chunk))) * np.hanning(size)))
        chroma = np.bincount(pitches, weights=spectrum[mask], minlength=12)
        chroma = np.sqrt(chroma)
        features.append(chroma / max(float(np.linalg.norm(chroma)), 1e-8))
    features = np.asarray(features)
    sections = []
    if len(features) >= 8 and np.max(np.abs(audio), initial=0) > 1e-5:
        novelty = np.zeros(len(features))
        for i in range(3, len(features) - 3):
            a, b = features[i - 3 : i].mean(axis=0), features[i : i + 3].mean(axis=0)
            novelty[i] = (
                0
                if max(np.linalg.norm(a), np.linalg.norm(b)) < 1e-6
                else max(0, 1 - np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8))
            )
        candidates = [
            i
            for i in range(4, len(features) - 4)
            if novelty[i] > max(0.12, np.percentile(novelty, 80))
            and novelty[i] >= max(novelty[i - 2 : i + 3])
        ]
        boundaries = [0.0]
        for i in candidates:
            if i - boundaries[-1] >= 4:
                nearby = min(
                    (b["timeSeconds"] for b in downbeats),
                    key=lambda x: abs(x - i),
                    default=float(i),
                )
                boundary = nearby if abs(nearby - i) <= 1 else float(i)
                if boundary - boundaries[-1] >= 4 and duration - boundary >= 4:
                    boundaries.append(boundary)
        boundaries.append(duration)
        signatures = []
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:], strict=False)):
            vector = features[int(start) : max(int(start) + 1, int(end))].mean(axis=0)
            vector /= max(np.linalg.norm(vector), 1e-8)
            group = next(
                (g for g, prior in enumerate(signatures) if np.dot(prior, vector) > 0.985),
                len(signatures),
            )
            if group == len(signatures):
                signatures.append(vector)
            sections.append(
                dict(
                    startSeconds=start,
                    endSeconds=end,
                    label=f"Section {index + 1}",
                    repetitionGroup=group,
                    confidence=float(novelty[min(int(start), len(novelty) - 1)]),
                    labelSource="spectral_novelty_candidate",
                    provisional=True,
                )
            )
    # Supplied section headings are explicitly text-derived, not an audio classifier verdict.
    for line in lines or []:
        if line.get("startSeconds") is not None and line.get("sectionLabel"):
            matching = next(
                (
                    s
                    for s in sections
                    if s["startSeconds"] <= line["startSeconds"] < s["endSeconds"]
                ),
                None,
            )
            if matching and matching["labelSource"] != "supplied_lyrics":
                matching["label"] = line["sectionLabel"]
                matching["labelSource"] = "supplied_lyrics"
    timed = sorted(
        (w for w in words or [] if w.get("startSeconds") is not None),
        key=lambda w: w["startSeconds"],
    )
    gaps = []
    for left, right in zip(timed, timed[1:], strict=False):
        if right["startSeconds"] - left["endSeconds"] >= 2:
            gaps.append(
                dict(
                    startSeconds=left["endSeconds"],
                    endSeconds=right["startSeconds"],
                    label="Possible vocal break",
                    provisional=True,
                )
            )
    return dict(
        bars=bars,
        sections=sections,
        vocalGaps=gaps,
        structureMethod="bounded-chroma-novelty-v1",
        structureLimitations=[
            "Sections and repetition groups need review; similar harmony is not proof of chorus.",
            "Vocal gaps may be recognition failures, not instrumental passages.",
        ],
    )


def editing_cues(analysis, *, fps, source_start=0, duration=None):
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("Use a positive finite editing frame rate")
    duration = (
        analysis.get("durationSeconds", 3600) - source_start if duration is None else duration
    )
    if not math.isfinite(source_start + duration) or source_start < 0 or duration <= 0:
        raise ValueError("Use a finite positive audio interval")
    candidates = [
        (c["timeSeconds"], c["kind"], c["strength"], c["kind"].replace("_", " ").title())
        for c in analysis.get("cues", [])
        if c["kind"] in {"onset", "energy_change"}
    ]
    for kind in ("beat", "downbeat"):
        candidates += [
            (b["timeSeconds"], kind, b["confidence"], kind.title())
            for b in analysis.get(kind + "s", [])
        ]
    candidates += [
        (s["startSeconds"], "section", s["confidence"], s["label"])
        for s in analysis.get("sections", [])
    ]
    alignment = analysis.get("alignment", analysis) or {}
    for name, kind in (("words", "word_end"), ("lines", "line_end")):
        candidates += [
            (w["endSeconds"], kind, w["confidence"], w["text"])
            for w in alignment.get(name, [])
            if w.get("endSeconds") is not None
        ]
    candidates += [(p["startSeconds"], "pause", 0.5, "Pause") for p in alignment.get("pauses", [])]
    priority = dict(
        onset=-2, energy_change=-1, beat=0, word_end=1, pause=2, line_end=3, downbeat=4, section=5
    )
    chosen = {}
    for second, kind, confidence, label in candidates:
        if not source_start < second < source_start + duration:
            continue
        frame = math.floor((second - source_start) * fps + 0.5)
        if not 0 < frame < math.ceil(duration * fps):
            continue
        cue = dict(
            id=f"{kind}-{frame}",
            kind=kind,
            timeSeconds=float(second),
            frame=frame,
            snapErrorSeconds=source_start + frame / fps - second,
            confidence=max(0, min(1, float(confidence))),
            label=label,
            locked=False,
        )
        if frame not in chosen or priority[kind] > priority[chosen[frame]["kind"]]:
            chosen[frame] = cue
    return [chosen[f] for f in sorted(chosen)]
