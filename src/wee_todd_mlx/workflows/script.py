"""Read explicit source shots without asking a model to invent their grouping or timing."""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal

SHOT = re.compile(r"(?m)^\s*\[Shot\s+(\d+)\]\s*", re.I)
GLOBAL = re.compile(r"(?mi)^\s*(?:overall_soundscape|non_diegetic_music|negative_prompt)\s*:")
TIME = re.compile(r"^At\s+(\d{1,2}):(\d{2}(?:\.\d+)?)", re.I)
QUOTED = (
    r'''"(?:\\.|[^"\\])*"|“[^”]*”|'''
    r"(?<!\w)'(?:\\.|[^'\\]|(?<=\w)'(?=\w))*'(?!\w)|‘(?:[^’]|(?<=\w)’(?=\w))*’"
)
SPEECH_VERB = r"\b(?:says?|said|whispers?|shouts?|asks?|replies|speaks?|yells?)\b"
SPEECH = re.compile(
    rf"(?:{SPEECH_VERB}"
    r"[^.!?\n]{0,80}?|(?:^|\n)[ \t]*[\w .'-]{1,60}:[ \t]*)"
    rf"(?P<quote>{QUOTED})|(?P<tag><d>.*?</d>)",
    re.IGNORECASE | re.DOTALL,
)


def shot_action_text(source):
    """Remove timeline metadata after authored_shots has validated its start time."""
    return re.sub(r"^At\s+\d{1,2}:\d{2}(?:\.\d+)?\s*[,.:]?\s*", "", source, flags=re.I)


def _dialogue(text):
    """Copy explicit speech with its source context; never infer missing spoken words."""
    # Authored shot headings and start times are metadata, never a speaker prefix.
    text = "\n".join(
        shot_action_text(re.sub(r"^\s*\[Shot\s+\d+\]\s*", "", line, flags=re.I))
        for line in text.splitlines()
    )
    fragments = []
    previous_end = 0
    for match in SPEECH.finditer(text):
        speech_start = match.start()
        while text[speech_start].isspace():
            speech_start += 1
        boundaries = list(re.finditer(r'''[.!?]["”’']?\s+|\n+''', text[:speech_start]))
        start = max(previous_end, boundaries[-1].end() if boundaries else 0)
        fragment = text[start:match.end()].strip()
        token = match["quote"] or match["tag"]
        # A single explicit name/pronoun is safe to compare across an action edit.
        # More complex attribution must retain its exact source prefix; never guess.
        attribution_end = match.start()
        if match["quote"] and not re.match(SPEECH_VERB, match.group(), re.IGNORECASE):
            # In speaker-label syntax the label belongs to the match itself.
            attribution_end = match.start("quote")
        prefix = text[start:attribution_end].strip().rstrip(":").strip()
        prefix = re.sub(
            r"^(?:inside|outside|at|in|on|near|beneath|above|beside)\b[^,\n]+,\s*", "", prefix,
            flags=re.IGNORECASE,
        )
        names = re.findall(r"\b[A-Z][\w’'-]*(?:\s+[A-Z][\w’'-]*)*", prefix)
        speaker = (
            names[0] if len(names) == 1 and prefix.startswith(names[0])
            and names[0] not in {"The", "A", "An"} else prefix
        )
        fragments.append((fragment, token, speaker))
        previous_end = match.end()
    return fragments


def validate_dialogue(text, source):
    """Validate spoken words and explicit attribution without assigning missing lines."""
    supplied = Counter((token, speaker) for _, token, speaker in _dialogue(source))
    generated = Counter((token, speaker) for _, token, speaker in _dialogue(text))
    # Every literal quote is checked, even when its surrounding verb is unfamiliar.
    source_quotes = Counter(re.findall(QUOTED, source))
    generated_quotes = Counter(re.findall(QUOTED, text))
    if generated - supplied or generated_quotes - source_quotes:
        raise ValueError(
            "Preserve source dialogue exactly, including its speaker. Return visible "
            "action only; the host retains the supplied spoken lines."
        )


def require_dialogue_assignment(actions, source):
    """Unlabeled prose may be reviewed only when its explicit speech has a shot assignment."""
    joined = "\n".join(actions)
    validate_dialogue(joined, source)
    supplied = Counter((token, speaker) for _, token, speaker in _dialogue(source))
    assigned = Counter((token, speaker) for _, token, speaker in _dialogue(joined))
    if supplied - assigned:
        raise ValueError(
            "Source dialogue needs an exact shot assignment. Keep every supplied spoken "
            "line in its intended action, or label the source [Shot 1], [Shot 2], etc. "
            "No spoken lines were discarded."
        )


def preserve_dialogue(action, source, maximum):
    """The model writes visible action; exact source speech remains host-owned."""
    validate_dialogue(action, source)
    existing = Counter((token, speaker) for _, token, speaker in _dialogue(action))
    missing = []
    for line, token, speaker in _dialogue(source):
        if existing[token, speaker]:
            existing[token, speaker] -= 1
        else:
            missing.append(line)
    # Newlines preserve speaker-label syntax and keep action prose outside attribution.
    action = "\n".join([action.strip(), *missing]).strip()
    if len(action) > maximum:
        raise ValueError(
            f"Action and exact source dialogue exceed {maximum} characters; shorten the "
            "visible action or split this source shot. No dialogue was discarded."
        )
    require_dialogue_assignment([action], source)
    return action


def authored_shots(brief, plan):
    matches = list(SHOT.finditer(brief))
    if not matches:
        return []
    count = len(plan["clips"])
    if len(matches) != count:
        raise ValueError(
            f"Your script has {len(matches)} labeled shots, but movie settings allocate "
            f"{count} clips. Adjust movie length/clip length or the shot list to match."
        )
    if [int(m[1]) for m in matches] != list(range(1, count + 1)):
        raise ValueError("Number source shots consecutively, starting with [Shot 1].")
    shots = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < count else len(brief)
        source = brief[match.end() : end].strip()
        boundary = GLOBAL.search(source)
        if boundary:
            source = source[: boundary.start()].strip()
        if not source:
            raise ValueError(f"Shot {index + 1} has no description")
        stamp = TIME.match(source)
        if stamp:
            seconds = Decimal(stamp[1]) * 60 + Decimal(stamp[2])
            expected = Decimal(plan["clips"][index]["startFrame"]) / plan["fps"]
            if abs(seconds - expected) > Decimal("0.5") / plan["fps"]:
                raise ValueError(
                    f"Shot {index + 1} timing is {seconds}s; movie settings place it "
                    f"at {expected}s. Match the script timing and movie settings."
                )
        shots.append(source)
    return shots
