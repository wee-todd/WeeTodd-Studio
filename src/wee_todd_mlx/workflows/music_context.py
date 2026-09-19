"""Planning views of saved music evidence; no inferred timestamps or transport records."""

import copy
import json
import math
import re

MUSIC_PLANNING_VERSION = 2
MUSIC_EVIDENCE_RULES = (
    "Original song audio is preserved. All acoustic times are source seconds, not verified "
    "lyric synchronization. Scores are uncalibrated support, never probabilities. Preserve "
    "flags, uncertainty and provisional labels; do not infer timed syllables, speech or "
    "verified choruses. Untimed lyrics remain global source evidence without invented times. "
)
MUSIC_WINDOW_RULES = (
    "Use the assigned phase as coarse narrative guidance, not a verified song section. "
    "Relevant source-window lyrics may override the coarse outline's event placement. "
    "Respect the source chronology and uncertainty; repeated supplied lyrics may use "
    "recurring visual motifs or explicit holds rather than replaying completed story events. "
    + MUSIC_EVIDENCE_RULES
)
SUBJECT_SELECTION_RULES = (
    "subjectIndex lists every legal identity; omittedAppearanceIDs have unknown appearance "
    "in this request, not permission to redesign them. Prefer the exact supplied definitions. "
    "If selecting an omitted identity or location, the host must supply its full definition "
    "and replan before accepting the shot. Relationship placements are conditional context, "
    "not instructions to display every linked object. "
)
VISIBLE_SUBJECT_RULES = (
    "Also return visibleSubjectIDs: a distinct array of known subjectIndex IDs for every "
    "visible character, location, prop and garment, including objects described by shorthand. "
    "This metadata is required for full shots and scoped repairs and must cover the entire "
    "resulting shot. Name each declared noncharacter by its exact approved name or alias in "
    "action, startState, endState or location; shorthand or an ID alone is insufficient. "
    "Declare every location visibly shown in action or states, even when location names "
    "only the primary setting. Do not declare absent or merely contextual objects. "
    "The host derives characters from the character-kind IDs in visibleSubjectIDs; "
    "omit the redundant characters field. During repairs outside characters, preserve "
    "the existing cast in visibleSubjectIDs. "
)


def action_array_format(count, *, retry=False):
    """Teach the wire format explicitly without treating rejected prose as story evidence."""
    return (
        "\nOutput format: one JSON array with double-quoted strings separated by commas. "
        "Use one outer pair of square brackets, never [unquoted prose] per line. "
        "No Markdown, headings or text outside the array. Lyric labels such as [Verse] and "
        "[Chorus] are source data, not the output format. Replace every placeholder with "
        "one chronological action from the source; do not copy the placeholder words. "
        + (
            "The rejected reply is not source evidence: reread the original source and produce "
            "fresh actions in the required JSON format. "
            if retry
            else ""
        )
        + "\nJSON array skeleton (replace every placeholder):\n"
        + json.dumps([f"Source action {i + 1}." for i in range(count)])
    )


def shot_object_format(fields, *, constraints=None):
    """Show only the requested decision fields, never example production identities."""
    placeholders = {
        "action": "VISIBLE_ACTION",
        "startState": "VISIBLE_START_STATE",
        "endState": "VISIBLE_END_STATE",
        "location": "EXACT_ALLOWED_LOCATION_NAME",
        "continuity": "cut",
    }
    constraints = constraints or {}
    for field, key in (("location", "requiredLocation"), ("continuity", "requiredContinuity")):
        if key in constraints:
            placeholders[field] = constraints[key]
    skeleton = {key: placeholders[key] for key in fields if key != "characters"}
    skeleton["visibleSubjectIDs"] = ["KNOWN_VISIBLE_SUBJECT_ID"]
    return (
        "\nReturn one flat JSON object: all fields at the root, no shot/result wrapper. "
        "visibleSubjectIDs is an array of ID strings, never objects with id/name. "
        "The host derives characters from those explicit IDs; omit characters. "
        "Use only known IDs from the assignment; use [] when none apply. "
        "visibleSubjectIDs covers the entire resulting shot, including unchanged repair fields. "
        "For cut, startState must show the new shot at its chosen location; do not copy "
        "the previous scene's endState. Only continue carries the previous endState forward. "
        "Return only the fields shown below. Do not copy placeholder values: replace them "
        "with the assigned action, exact approved names and IDs. "
        + (
            "Use requiredContinuity exactly. "
            if "requiredContinuity" in constraints
            else "Choose cut or continue from the continuity rules; the example cut is not "
            "a story decision. "
        )
        + "No Markdown or prose "
        "outside JSON. "
        + (
            "requiredLocation and requiredContinuity are exact host constraints. Copy their "
            "values when those fields are editable; never change immutable repair fields. "
            if constraints
            else ""
        )
        + "\nFlat shot JSON skeleton (replace placeholders; no wrapper):\n"
        + json.dumps(skeleton)
    )


def _positive_match(text, phrase, *, background=False):
    for sentence in re.split(r"[.!?;]|\bbut\b", text.casefold()):
        normalized = " " + " ".join(re.findall(r"\w+", sentence)) + " "
        for match in re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", normalized):
            prefix = normalized[: match.start()]
            # Qualify this noun only; a distant mountain cannot hide a foreground sword.
            if background and (
                re.search(r"\b(?:background|distant)\s+$", prefix)
                or re.match(
                    r"\s+(?:appears\s+)?in\s+(?:the\s+)?(?:distant\s+)?background\b",
                    normalized[match.end() :],
                )
            ):
                continue
            if not re.search(r"\b(?:no|without|not|neither|absent)\b(?:\s+\w+){0,5}\s*$", prefix):
                return True
    return False


def named_subject_ids(
    subjects, text, *, include_ids=True, positive_only=False, reject_ambiguous=False
):
    # Prefer a complete known name/alias at each occurrence. A namesake nested
    # inside another identity (Grendel in Grendel's mother) is not a second actor.
    phrases = {
        (row["id"], phrase)
        for row in subjects
        for term in ([row["id"]] if include_ids else [])
        + [row["name"], *row.get("aliases", [])]
        for phrase in [" ".join(re.findall(r"\w+", term.casefold()))]
        if phrase
    }
    selected = set()
    for sentence in re.split(r"[.!?;]|\bbut\b", text.casefold()):
        normalized = " " + " ".join(re.findall(r"\w+", sentence)) + " "
        matches = [
            (sid, match.start(), match.end())
            for sid, phrase in phrases
            for match in re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", normalized)
        ]
        for sid, start, end in matches:
            if any(
                other_start <= start and end <= other_end
                and other_end - other_start > end - start
                for _, other_start, other_end in matches
            ):
                continue
            # Apply negation to this occurrence, not another same-name mention.
            if positive_only and re.search(
                r"\b(?:no|without|not|neither|absent)\b(?:\s+\w+){0,5}\s*$",
                normalized[:start],
            ):
                continue
            if reject_ambiguous and len({
                other for other, left, right in matches if left == start and right == end
            }) > 1:
                raise ValueError("A shot names multiple possible subjects; clarify its identity")
            selected.add(sid)
    return selected


def candidate_subject_ids(subjects, candidate):
    text = " ".join(
        candidate.get(key, "") for key in ("action", "startState", "endState", "location")
    )
    # A generic noun does not identify a qualified production object. In
    # particular, a unique catalog sword is not automatically every story sword.
    return named_subject_ids(subjects, text, positive_only=True) | set(
        candidate.get("characters", [])
    )


def _ambiguous_object_heads(subjects, text, exact):
    """Ask about unresolved variants without assigning their identities or appearances."""
    heads = {}
    for row in subjects:
        if row["kind"] == "character":
            continue
        words = re.findall(r"\w+", row["name"].casefold())
        if words and len(words[-1]) >= 3:
            heads.setdefault(words[-1], set()).add(row["id"])
    return {
        head: ids
        for head, ids in heads.items()
        if len(ids) > 1 and not ids & exact and _positive_match(text, head, background=True)
    }


def translate_subject_ids(text, mapping):
    """Translate ID references in repair diagnostics, never source prose."""
    if not mapping:
        return text
    pattern = r"(?<![\w-])(?:" + "|".join(re.escape(sid) for sid in mapping) + r")(?![\w-])"
    return re.sub(pattern, lambda match: mapping[match[0]], text)


def compact_subject_ids(payload):
    """Alias long opaque IDs on the wire; saved assignments retain canonical IDs."""
    subjects = payload["assignment"].get("subjectIndex", [])
    known = {row["id"] for row in subjects}
    mapping = {}
    for row in subjects:
        sid = row["id"]
        if len(sid) < 16:
            continue
        alias = f"ref{len(mapping) + 1}"
        while alias in known or alias in mapping.values():
            alias = "_" + alias
        mapping[sid] = alias

    def packed(value, key=None):
        if isinstance(value, dict):
            return {k: packed(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [packed(v, key) for v in value]
        if isinstance(value, str):
            if key in {"id", "targetID", "characters", "visibleSubjectIDs",
                       "selectedAppearanceIDs", "omittedAppearanceIDs"}:
                return mapping.get(value, value)
            if key == "repairInstruction":
                return translate_subject_ids(value, mapping)
        return value

    return packed(payload), {alias: sid for sid, alias in mapping.items()}


def parse_visible_subjects(text, subjects, *, immutable_characters=None, aliases=None):
    """Project explicit music identities into the unchanged generic shot/repair schema."""
    from .io import _constant, _pairs, check_json
    from .operations import _root_closer

    raw = text.strip()
    if raw.startswith("```json\n") and raw.endswith("```"):
        raw = raw[8:-3].strip()
    value = json.loads(_root_closer(raw), object_pairs_hook=_pairs, parse_constant=_constant)
    check_json(value)
    if isinstance(value, dict):
        if "visibleSubjectIDs" not in value and any(
            isinstance(nested, dict) and "visibleSubjectIDs" in nested for nested in value.values()
        ):
            raise ValueError(
                "Return a flat shot object: visibleSubjectIDs and shot fields belong at the "
                "root, never inside a shot/result wrapper"
            )
        if "characters" in value and (
            not isinstance(value["characters"], list)
            or any(not isinstance(sid, str) for sid in value["characters"])
        ):
            raise ValueError("characters must be an array of ID strings, never id/name objects")
    declared = value.pop("visibleSubjectIDs", None) if isinstance(value, dict) else None
    if aliases:
        if isinstance(declared, list):
            declared = [aliases.get(sid, sid) if isinstance(sid, str) else sid for sid in declared]
        if isinstance(value, dict) and isinstance(value.get("characters"), list):
            value["characters"] = [aliases.get(sid, sid) for sid in value["characters"]]
    known = {row["id"] for row in subjects}
    if (
        not isinstance(declared, list)
        or any(not isinstance(sid, str) for sid in declared)
        or len(declared) != len(set(declared))
        or set(declared) - known
    ):
        raise ValueError("visibleSubjectIDs must list distinct known subjectIndex IDs")
    character_ids = {row["id"] for row in subjects if row["kind"] == "character"}
    projected = [sid for sid in declared if sid in character_ids]
    if immutable_characters is not None and set(projected) != set(immutable_characters):
        raise ValueError("visibleSubjectIDs must preserve immutable characters for this repair")
    if "characters" in value:
        if len(value["characters"]) != len(set(value["characters"])) or set(
            value["characters"]
        ) != set(projected):
            raise ValueError(
                "characters must match the character IDs declared in visibleSubjectIDs"
            )
    elif immutable_characters is None:
        value["characters"] = projected
    return json.dumps(value, ensure_ascii=False), set(declared)


def validate_visible_subjects(subjects, value, declared):
    names = {row["id"]: row["name"] for row in subjects}

    def labels(ids):
        return ", ".join(f"{names[sid]} ({sid})" for sid in sorted(ids))

    required = candidate_subject_ids(subjects, value)
    if required - declared:
        raise ValueError(
            "Declare visibleSubjectIDs and use exact approved names for: "
            + labels(required - declared)
        )
    text = " ".join(value[key] for key in ("action", "startState", "endState", "location"))
    named = named_subject_ids(subjects, text, include_ids=False, positive_only=True)
    ambiguous = _ambiguous_object_heads(subjects, text, named)
    if ambiguous:
        choices = "; ".join(f"{head}: {labels(ids)}" for head, ids in sorted(ambiguous.items()))
        raise ValueError(
            "Ambiguous object reference — " + choices + ". Use the exact approved names "
            "for the visible objects, or resolve the missing object in the catalog. Do not "
            "assume every listed variant is visible. A generic noun alone does not select "
            "a catalog identity."
        )
    missing = {
        row["id"]
        for row in subjects
        if row["kind"] != "character" and row["id"] in declared and row["id"] not in named
    }
    if missing:
        raise ValueError(
            "Name each visible object by its exact approved name or alias: "
            + labels(missing)
        )


def scoped_subjects(subjects, assigned, previous, window, required=()):
    """Select full definitions semantically, never trim them to a runtime token limit."""
    previous = previous or {}
    text = "\n".join(
        [
            assigned,
            previous.get("endState", ""),
            previous.get("location", ""),
            *[line["text"] for line in window["lines"]],
        ]
    )
    roots = named_subject_ids(subjects, text) | set(previous.get("characters", [])) | set(required)
    inventory = {row["id"]: row for row in subjects}
    selected = set(roots)
    # One hop only. Contained scenery/props and other locations require an explicit match;
    # following the whole production graph would put the entire inventory in every shot.
    for sid in roots:
        row = inventory.get(sid, {})
        for link in row.get("relationships", []):
            target = inventory.get(link["targetID"], {})
            if (
                link["role"] in {"wears", "holds", "uses", "part_of"}
                and target.get("kind") not in {"set", "environment", "location"}
            ) or (row.get("kind") == "set" and link["role"] == "located_in"):
                selected.add(link["targetID"])
    rows = [
        {
            key: copy.deepcopy(row[key])
            for key in ("id", "name", "kind", "description", "relationships")
            if key in row
        }
        for row in subjects
        if row["id"] in selected
    ]
    return {
        "characters": [row for row in rows if row["kind"] == "character"],
        "approvedSubjects": [row for row in rows if row["kind"] != "character"],
        "subjectIndex": [
            {
                key: copy.deepcopy(row[key])
                for key in ("id", "name", "kind", "aliases")
                if key in row
            }
            for row in subjects
        ],
        "selectedAppearanceIDs": [row["id"] for row in subjects if row["id"] in selected],
        "omittedAppearanceIDs": [row["id"] for row in subjects if row["id"] not in selected],
        "appearanceSelection": SUBJECT_SELECTION_RULES + "Full originals remain saved unchanged.",
    }


def _timed(row):
    return (
        all(
            isinstance(row.get(key), (int, float)) and math.isfinite(row[key])
            for key in ("startSeconds", "endSeconds")
        )
        and row["endSeconds"] > row["startSeconds"]
    )


def _alignment(timing):
    return (timing.get("analysis") or {}).get("alignment") or {}


def _uncertainty(timing):
    alignment = _alignment(timing)
    return {
        "contextVersion": MUSIC_PLANNING_VERSION,
        "lyricStatus": timing.get("lyricStatus", "unknown"),
        "confidenceMeaning": alignment.get("confidenceMeaning", "Uncalibrated acoustic support"),
        "untimedLineCount": sum(not _timed(row) for row in alignment.get("lines", [])),
        "direction": MUSIC_EVIDENCE_RULES,
    }


def source_window(timing, clips):
    """Expose exact overlapping line support, never all word/transport metadata."""
    allocation = timing["allocation"]
    fps = allocation["fps"]
    audio = timing.get("sourceAudio", {})
    origin = audio.get("sourceStartSeconds", 0)
    end = audio.get("sourceEndSeconds", origin + allocation["totalFrames"] / fps)
    windows = [
        {
            "clipID": clip["id"],
            "sourceStartSeconds": origin + clip["startFrame"] / fps,
            "sourceEndSeconds": min(end, origin + (clip["startFrame"] + clip["frameCount"]) / fps),
        }
        for clip in clips
    ]
    start, stop = windows[0]["sourceStartSeconds"], windows[-1]["sourceEndSeconds"]

    def overlaps(row):
        return _timed(row) and row["startSeconds"] < stop and row["endSeconds"] > start

    return {
        **_uncertainty(timing),
        "sourceStartSeconds": start,
        "sourceEndSeconds": stop,
        "clips": windows,
        "lines": [
            {
                key: copy.deepcopy(row[key])
                for key in (
                    "text",
                    "startSeconds",
                    "endSeconds",
                    "confidence",
                    "flags",
                    "sectionLabel",
                )
                if key in row
            }
            for row in _alignment(timing).get("lines", [])
            if overlaps(row)
        ],
        "sections": [
            copy.deepcopy(row)
            for row in timing.get("analysis", {}).get("sections", [])
            if overlaps(row)
        ],
        "lineSectionLabelMeaning": "Supplied lyric labels; not verified acoustic sections.",
        "markers": [
            copy.deepcopy(row)
            for row in timing.get("markers", [])
            if start <= row["timeSeconds"] < stop
        ],
        "selection": "Only overlapping timed lines/sections and in-window user markers are shown. "
        "Missing local evidence does not imply silence or absence. Untimed lines and full "
        "supplied lyrics remain in the global treatment source; full analysis remains saved.",
    }


def treatment_brief(brief, timing):
    """Keep complete authored text; global outline needs a summary, not a transport dump."""
    alignment = _alignment(timing)
    allocation = timing["allocation"]
    audio = timing.get("sourceAudio", {})
    origin = audio.get("sourceStartSeconds", 0)
    summary = {
        **_uncertainty(timing),
        "sourceStartSeconds": origin,
        "sourceEndSeconds": audio.get(
            "sourceEndSeconds", origin + allocation["totalFrames"] / allocation["fps"]
        ),
        "clipCount": len(allocation["clips"]),
        "timedLineCount": sum(_timed(row) for row in alignment.get("lines", [])),
        "timingFlags": sorted(
            {flag for row in alignment.get("lines", []) for flag in row.get("flags", [])}
        ),
        "warnings": timing.get("warnings", []),
        "selection": "Full supplied lyrics follow as source text. Detailed acoustic line evidence "
        "is selected separately for each expansion/shot window; no word timing is asserted here.",
    }
    lyrics = timing.get("suppliedLyrics", "")
    # Exact duplicates need not consume context twice; never trim or paraphrase source text.
    if lyrics and lyrics not in brief:
        brief += (
            "\nSupplied lyrics (unaltered source text; labels are not verified sections):\n"
            + lyrics
        )
    return brief + "\nMusic evidence summary (data): " + json.dumps(summary, ensure_ascii=False)
