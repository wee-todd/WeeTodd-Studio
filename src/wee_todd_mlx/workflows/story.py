"""Deterministic story scoping and conservative planning warnings; no model imports."""

from __future__ import annotations

import re
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal

TIME = r"\d+(?::\d+)?(?:\.\d+)?"
SECTION = re.compile(
    rf"^\s*(?:[#*\[]+\s*)?({TIME})\s*[–—-]\s*({TIME})\s*(?:s|seconds)?[^\n]*$",
    re.MULTILINE,
)
LABEL = re.compile(r"^\s*[*\-\s]*(Visual|Action|Audio|Sound|Camera)\s*:[*\s]*(.*)$", re.I)


def _seconds(value):
    parts = value.split(":")
    return Decimal(parts[0]) * 60 + Decimal(parts[1]) if len(parts) == 2 else Decimal(value)


def _actions(body):
    groups = {"visual": [], "action": [], "other": []}
    active = "other"
    for line in body.splitlines():
        match = LABEL.match(line)
        if match:
            active = match[1].lower()
            line = match[2]
        elif re.match(r"^\s*[#*\s]*(?:title|outline)\s*:", line, re.I):
            continue
        if active in groups:
            clean = re.sub(r"^\s*(?:[-*#]+|\d+[.)])\s*", "", line.strip()).replace("**", "")
            if clean:
                groups[active].append(clean)
    # A labeled visual description supplies staging and action; don't duplicate its audio/summary.
    selected = groups["visual"] or groups["action"] or groups["other"]
    text = " ".join(selected)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    actions, framing = [], ""
    for sentence in sentences:
        if re.fullmatch(
            r"(?:extreme |medium |wide |long |close[- ]up |establishing )?shot[.!]?", sentence, re.I
        ):
            framing += sentence + " "
        else:
            actions.append(framing + sentence)
            framing = ""
    if framing:
        if actions:
            actions[-1] += " " + framing.strip()
        else:
            actions.append(framing.strip())
    return actions


def scoped_actions(story, clips, total, fps):
    matches = list(SECTION.finditer(story))
    sections = []
    if matches:
        cursor = 0
        for index, match in enumerate(matches):
            start, end = [
                int((_seconds(value) * fps).to_integral_value(rounding=ROUND_HALF_UP))
                for value in (match[1], match[2])
            ]
            if start < cursor or end <= start:
                raise ValueError("Story time sections overlap or have invalid ranges")
            if start != cursor or end > total:
                raise ValueError("Story time sections must cover the movie with no gaps")
            body_end = matches[index + 1].start() if index + 1 < len(matches) else len(story)
            sections.append((start, end, _actions(story[match.end() : body_end])))
            cursor = end
        if cursor != total:
            raise ValueError("Story time sections must cover the entire requested movie")
    else:
        sections = [(0, total, _actions(story))]
    result = [[] for _ in clips]
    for start, end, actions in sections:
        if not actions:
            raise ValueError("Every story section needs a visible action before clip allocation")
        slots = [
            i
            for i, clip in enumerate(clips)
            if clip["startFrame"] < end and clip["startFrame"] + clip["frameCount"] > start
        ]
        # Partition sentences, not characters: every action stays intact and in sequence.
        # Sparse outlines can repeat a beat; review flags that rather than inventing new events.
        for position, index in enumerate(slots):
            first = position * len(actions) // len(slots)
            last = max(first + 1, (position + 1) * len(actions) // len(slots))
            result[index].extend(actions[first:last])
    return [" ".join(actions) for actions in result]


def _normalized(text):
    text = re.sub(r"^Story segment \d+ of \d+\.\s*", "", text)
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


STOP = set(
    """this that with from into through while where their there these those after before
only then when still also against around above below camera shot scene frame first last story
segment description visible another character moment image foreground background showing shows
looking looks follows keeps remains continues slightly slowly softly quietly very some more
than each same other have has been being are were will would could should its his her the and
""".split()
)


def _phrases(text):
    words = _normalized(text).split()
    return {
        a + " " + b
        for a, b in zip(words, words[1:], strict=False)
        if len(a) >= 4 and len(b) >= 4 and a not in STOP and b not in STOP
    }


def pacing_warnings(clips, endpoints, *, known_context=""):
    """Heuristics are warnings for human review, never semantic proof or automatic rewrites."""
    items = []

    def warn(cid, message):
        items.append({"severity": "warning", "clipID": cid, "message": message})

    counts = Counter(_normalized(c["action"]) for c in clips)
    for action, count in counts.items():
        if count >= 3:
            cid = next(c["id"] for c in clips if _normalized(c["action"]) == action)
            warn(
                cid,
                f"The same action is repeated across {count} clips. Add distinct story beats "
                "or confirm that this is an intentional hold.",
            )
    actions = [_phrases(c["action"]) for c in clips]
    by_id = {e["clipID"]: e for e in endpoints}
    previous, introduced, seen = [], _phrases(known_context), {}
    for index, clip in enumerate(clips):
        introduced |= actions[index]
        pair = by_id.get(clip["id"])
        if pair is None:
            continue
        end = pair["last"]["description"]
        normalized = _normalized(end)
        pairs = set(zip(normalized.split(), normalized.split()[1:], strict=False))
        repeated = seen.get(normalized) or next(
            (
                cid
                for cid, before in previous[-8:]
                if pairs and before and 2 * len(pairs & before) / (len(pairs) + len(before)) >= 0.90
            ),
            None,
        )
        if repeated:
            warn(
                clip["id"],
                f"Last-frame description repeats {repeated} with little change. "
                "Check that the clip advances its assigned action.",
            )
        previous.append((clip["id"], pairs))
        seen.setdefault(normalized, clip["id"])
        future = (
            set().union(*actions[index + 1 :]) - introduced if index + 1 < len(clips) else set()
        )
        matches = sorted(_phrases(end) & future)
        if len(matches) >= 2:
            warn(
                clip["id"],
                "Last frame may introduce future story details too early: "
                + ", ".join(matches[:8])
                + ". Compare it with this clip's assigned action.",
            )
        if index < len(clips) - 1 and re.search(r"fad\w* to black|story ends|the end", end, re.I):
            warn(clip["id"], "An ending/fade appears before the final clip. Check story pacing.")
    return items
