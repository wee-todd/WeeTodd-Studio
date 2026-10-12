"""Symbolic-v1 interchange vocabulary and independent event/ABC serialization.

Token IDs and label ordering are checkpoint data contracts. No external tokenizer
or notation implementation is imported or executed.
"""

from fractions import Fraction

import numpy as np

STRUCTURES = (
    "silence",
    "intro",
    "outro",
    "verse",
    "chorus",
    "bridge",
    "pre-chorus",
    "post-chorus",
    "interlude",
    "fade-out",
    "loop",
    "rap",
    "preshot",
    "irregular",
    "instrumental",
    "intro and verse",
    "pre-chorus and chorus",
    "verse and pre-chorus",
    "solo",
    "theme",
    "development",
    "variation",
    "pre-outro",
)
PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
DURATIONS = (
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
)
PROMPTS = ("timestamp", "downbeat_meter", "structure", "key", "chord_full", "melody_full")


class Vocabulary:
    size = 31678
    prefix = [1, 4, 5, 6, 7, 9, 11, 3]
    ranges = {
        "shift": (260, 517),
        "time": (517, 30517),
        "meter": (30517, 30709),
        "eighth": (30709, 30965),
        "structure": (30965, 30988),
        "key": (30988, 31012),
        "major_minor": (31012, 31037),
        "chord": (31037, 31398),
        "pitch": (31398, 31654),
        "duration": (31654, 31678),
    }

    def __init__(self):
        self.chords = ["N"]
        for quality in (
            "maj",
            "min",
            "dim",
            "aug",
            "maj7",
            "min7",
            "7",
            "hdim7",
            "dim7",
            "minmaj7",
            "sus2",
            "sus4",
            "sus4(b7)",
            "maj6",
            "min6",
        ):
            inversions = {
                "maj": ("2", "3", "5"),
                "min": ("2", "b3", "5"),
                "maj7": ("3", "5", "7"),
                "min7": ("b3", "5", "b7"),
                "7": ("3", "5", "b7"),
            }.get(quality, ())
            for pitch in PITCH_CLASSES:
                self.chords.extend(f"{pitch}:{quality}/{bass}" for bass in inversions)
                self.chords.append(f"{pitch}:{quality}")

    def kind(self, token):
        for name, (start, end) in self.ranges.items():
            if start <= token < end:
                return name
        return {0: "pad", 1: "bos", 2: "eos", 3: "out"}.get(token, "unknown")


class Grammar:
    """Ordered event fields, optional note lengths, and bounded shift runs."""

    ranks = {
        "time": 0,
        "meter": 1,
        "eighth": 1,
        "structure": 2,
        "key": 3,
        "chord": 4,
        "pitch": 5,
        "duration": 5,
    }

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.rank = -1
        self.payload = 0
        self.shifts = 0
        self.last = "shift"

    def allowed(self):
        mask = np.zeros(self.vocabulary.size, dtype=bool)
        if self.last == "meter":
            start, end = self.vocabulary.ranges["eighth"]
            mask[start:end] = True
            return mask
        if self.payload == 0 and self.shifts == 0:
            mask[260:517] = True
            return mask
        mask[2] = self.payload > 0
        if (self.payload or self.last == "shift") and self.shifts < 4:
            mask[260:517] = True
        if self.last == "pitch":
            names = ["pitch", "duration"]
        else:
            names = [
                name
                for name, rank in self.ranks.items()
                if name != "duration" and (rank > self.rank or name == "pitch")
            ]
        for name in names:
            start, end = self.vocabulary.ranges[name]
            mask[start:end] = True
        return mask

    def accept(self, token):
        if not 0 <= token < self.vocabulary.size or not self.allowed()[token]:
            raise ValueError(f"Invalid symbolic event transition at token {token}")
        kind = self.vocabulary.kind(token)
        if kind == "eos":
            return True
        if kind == "shift":
            if self.last != "shift":
                self.payload = 0
                self.rank = -1
            self.shifts += 1
        else:
            self.payload += 1
            self.shifts = 0
            self.rank = self.ranks[kind]
        self.last = kind
        return False


def decode(tokens, duration):
    """Retain original token events and derive note times only from predicted anchors."""
    v = Vocabulary()
    if list(tokens[:8]) != v.prefix or not tokens or tokens[-1] != 2:
        raise ValueError("SheetSage2 sequence is missing its task prefix or EOS")
    events = []
    current = None
    position = 0
    shifted = False
    for token in tokens[8:-1]:
        kind = v.kind(token)
        if kind == "shift":
            if current is not None:
                events.append(current)
                current = None
            position += token - 260
            shifted = True
            continue
        if not shifted:
            raise ValueError("A SheetSage2 event has no subbeat shift")
        if current is None:
            current = {"subbeat": position, "values": {}, "tokens": []}
        values = current["values"]
        current["tokens"].append(token)
        index = token - v.ranges.get(kind, (0, 0))[0]
        if kind == "time":
            values["timestamp"] = index / 100
        elif kind == "meter":
            values.setdefault("rhythm", {})["meter"] = [
                index // 6 + 1,
                (1, 2, 4, 8, 16, 32)[index % 6],
            ]
        elif kind == "eighth":
            values.setdefault("rhythm", {})["eighth_position"] = index
        elif kind == "structure":
            values["structure"] = STRUCTURES[index]
        elif kind == "key":
            values["key"] = PITCH_CLASSES[index % 12] + (":minor" if index >= 12 else ":major")
        elif kind == "chord":
            values["chord"] = v.chords[index]
        elif kind == "pitch":
            values.setdefault("melody", []).append(
                {
                    "pitch": index % 128,
                    "track": index // 128,
                    "duration_bin": 0,
                    "duration_steps": 1,
                }
            )
        elif kind == "duration" and values.get("melody"):
            values["melody"][-1].update(duration_bin=index, duration_steps=DURATIONS[index])
        else:
            raise ValueError(f"Unexpected SheetSage2 event token {token}")
    if current is not None:
        events.append(current)
    anchors = {}
    warnings = []
    for event in events:
        if "timestamp" in event["values"]:
            anchors[event["subbeat"]] = event["values"]["timestamp"]
    if len(anchors) < 2:
        raise ValueError(
            "SheetSage2 did not infer enough timestamp anchors to recover musical timing; "
            "choose a longer musical excerpt"
        )
    valid = []
    for step, stamp in sorted(anchors.items()):
        if valid and stamp <= valid[-1][1]:
            warnings.append(
                f"Non-increasing predicted timestamp at subbeat {step}; timing needs review"
            )
        else:
            valid.append((step, stamp))
    if len(valid) < 2:
        raise ValueError(
            "SheetSage2 predicted inconsistent timestamp timing; choose another excerpt"
        )
    steps, stamps = np.array(valid, dtype=np.float64).T
    seconds_per_step = float(np.median(np.diff(stamps) / np.diff(steps)))

    def time_at(step):
        if step < steps[0]:
            value = stamps[0] + (step - steps[0]) * seconds_per_step
        elif step > steps[-1]:
            value = stamps[-1] + (step - steps[-1]) * seconds_per_step
        else:
            value = np.interp(step, steps, stamps)
        return float(np.clip(value, 0, duration))

    retained = []
    for event in events:
        event["time"] = time_at(event["subbeat"])
        if event["time"] >= duration:
            continue
        for note in event["values"].get("melody", []):
            note["startSeconds"] = event["time"]
            note["endSeconds"] = time_at(event["subbeat"] + note["duration_steps"])
        retained.append(event)
    return {
        "events": retained,
        "warnings": list(dict.fromkeys(warnings)),
        "seconds_per_subbeat": seconds_per_step,
        "schema_version": "v1",
    }


def _pitch(midi):
    # Explicit accidentals avoid leaking accidentals across tied polyphonic segments.
    letter = ("=C", "^C", "=D", "^D", "=E", "=F", "^F", "=G", "^G", "=A", "^A", "=B")[midi % 12]
    octave = midi // 12 - 5
    if octave >= 1:
        return letter.lower() + "'" * (octave - 1)
    return letter + "," * (-octave)


def _length(value):
    if value == 1:
        return ""
    return (
        str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    )


def to_abc(events, melody_only=True):
    """Render both predicted melody tracks with exact symbolic lengths and rests.

    This serializer preserves overlapping notes as polyphony instead of silently
    deleting predicted pitches. All source events remain available for editing.
    """
    keys = [e["values"]["key"] for e in events if "key" in e["values"]]
    meters = [
        e["values"]["rhythm"]["meter"] for e in events if "meter" in e["values"].get("rhythm", {})
    ]
    notes = []
    for event in events:
        for note in event["values"].get("melody", []):
            if note["endSeconds"] > note["startSeconds"]:
                notes.append(
                    (
                        event["subbeat"],
                        event["subbeat"] + note["duration_steps"],
                        note["pitch"],
                        note["track"],
                    )
                )
    if not notes:
        raise ValueError("SheetSage2 did not detect melody notes; no score was fabricated")
    if not keys or not meters:
        raise ValueError(
            "SheetSage2 did not detect a key and meter; review timed events before creating a score"
        )
    first, final = min(e["subbeat"] for e in events), max(note[1] for note in notes)
    numerator, denominator = meters[0]
    key = keys[0].replace(":major", "").replace(":minor", "m")
    anchors = [(e["subbeat"], e["time"]) for e in events if "timestamp" in e["values"]]
    periods = [
        (b[1] - a[1]) / (b[0] - a[0])
        for a, b in zip(anchors, anchors[1:], strict=False)
        if b[0] > a[0] and b[1] > a[1]
    ]
    if not periods:
        raise ValueError("SheetSage2 timing cannot determine a score tempo")
    # One model subbeat is a quarter of a denominator beat.
    tempo = 60 / (float(np.median(periods)) * denominator)
    lines = [
        "X:1",
        "T:Reference transcription",
        f"M:{numerator}/{denominator}",
        "L:1/16",
        f"Q:1/4={max(1, round(tempo))}",
        f"K:{key}",
        'V:Vocal name="Vocal melody"',
        'V:Ins name="Instrumental melody"',
    ]
    changes = {event["subbeat"]: event["values"] for event in events}
    # Merge same-position metadata: the decoder can emit separate events at zero shift.
    for event in events:
        changes[event["subbeat"]] = dict(changes.get(event["subbeat"], {}), **event["values"])
    boundaries = {first, final}
    boundaries.update(point for note in notes for point in note[:2])
    boundaries.update(step for step in changes if first <= step <= final)
    barlines = set()
    # Explicit decoded downbeats define bars. Extrapolate only within observed meter sections.
    meter_changes = [(first, meters[0])]
    for event in events:
        rhythm = event["values"].get("rhythm", {})
        if "meter" in rhythm:
            meter_changes.append((event["subbeat"], rhythm["meter"]))
        if rhythm.get("eighth_position") == 0:
            barlines.add(event["subbeat"])
    for index, (start, meter) in enumerate(meter_changes):
        end = meter_changes[index + 1][0] if index + 1 < len(meter_changes) else final
        anchors_here = [step for step in barlines if start <= step < end]
        cursor = min(anchors_here) if anchors_here else start
        while cursor < end:
            barlines.add(cursor)
            cursor += meter[0] * 4
    boundaries.update(step for step in barlines if first <= step <= final)
    ordered = sorted(boundaries)
    for track, name in enumerate(("Vocal", "Ins")):
        pieces = [f"[V:{name}]"]
        current_denominator = denominator
        for start, end in zip(ordered, ordered[1:], strict=False):
            metadata = changes.get(start, {})
            if start in barlines and start != first:
                pieces.append("|")
            if "structure" in metadata:
                pieces.append(f"[P:{metadata['structure']}]")
            if "key" in metadata:
                pieces.append(
                    "[K:" + metadata["key"].replace(":major", "").replace(":minor", "m") + "]"
                )
            if "meter" in metadata.get("rhythm", {}):
                num, current_denominator = metadata["rhythm"]["meter"]
                pieces.append(f"[M:{num}/{current_denominator}]")
            if not melody_only and metadata.get("chord", "N") != "N":
                # Chord labels are quoted annotations; preserve the exact prediction.
                pieces.append('"' + metadata["chord"] + '"')
            active = [note for note in notes if note[3] == track and note[0] <= start < note[1]]
            units = Fraction((end - start) * 4, current_denominator)
            if not active:
                pieces.append("z" + _length(units))
            elif len(active) == 1:
                note = active[0]
                pieces.append(_pitch(note[2]) + _length(units) + ("-" if note[1] > end else ""))
            else:
                pitches = "".join(
                    _pitch(note[2]) + ("-" if note[1] > end else "") for note in active
                )
                pieces.append("[" + pitches + "]" + _length(units))
        pieces.append("|]")
        lines.append(" ".join(pieces))
    return "\n".join(lines) + "\n"
