"""CTC evidence alignment. Scores measure acoustic support, not calibrated correctness."""

from __future__ import annotations

import difflib
import re
import unicodedata

import numpy as np


def forced_token_spans(log_probs, tokens, *, blank=0, check=None):
    """Viterbi CTC alignment with bounded traceback; repeated letters need a blank."""
    check = check or (lambda: None)
    check()
    scores = np.asarray(log_probs, dtype=np.float32)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("CTC scores must be a finite frame/token matrix")
    if not tokens:
        return []
    target = np.full(2 * len(tokens) + 1, blank, dtype=np.int32)
    target[1::2] = tokens
    if min(tokens) < 0 or max(tokens) >= scores.shape[1] or blank in tokens:
        raise ValueError("Invalid CTC target tokens")
    if len(scores) < len(tokens) or len(scores) * len(target) > 40_000_000:
        raise ValueError("Cannot align this text in the bounded audio window")
    previous = np.full(len(target), -np.inf, dtype=np.float32)
    previous[0] = 0
    trace = np.empty((len(scores), len(target)), dtype=np.uint8)
    skip = np.zeros(len(target), dtype=bool)
    skip[2:] = (target[2:] != blank) & (target[2:] != target[:-2])
    for t, frame in enumerate(scores):
        if t % 128 == 0:
            check()
        choices = np.stack(
            (previous, np.r_[-np.inf, previous[:-1]], np.r_[-np.inf, -np.inf, previous[:-2]])
        )
        choices[2, ~skip] = -np.inf
        trace[t] = choices.argmax(axis=0)
        previous = choices.max(axis=0) + frame[target]
    state = len(target) - (1 if previous[-1] >= previous[-2] else 2)
    if not np.isfinite(previous[state]):
        raise ValueError("Cannot align target tokens to audio")
    frames = [[] for _ in tokens]
    for t in range(len(scores) - 1, -1, -1):
        if state % 2:
            frames[state // 2].append(t)
        state -= int(trace[t, state])
    if any(not values for values in frames):
        raise ValueError("Cannot align every target token")
    return [(values[-1], values[0] + 1) for values in frames]


def _normal(text):
    return "".join(
        c for c in unicodedata.normalize("NFKC", text).upper() if c.isalnum() or c == "'"
    )


def _recognized(scores, vocabulary, frame_seconds, offset):
    inverse = {v: k for k, v in vocabulary.items()}
    blank = vocabulary.get("<pad>", 0)
    best = scores.argmax(axis=1)
    words, chars = [], []

    def flush():
        if chars:
            text = "".join(c[0] for c in chars)
            start, end = chars[0][1], chars[-1][2]
            words.append(
                dict(
                    text=text,
                    startSeconds=max(0, offset + (start - 0.5) * frame_seconds),
                    endSeconds=offset + (end - 0.5) * frame_seconds,
                    confidence=float(np.exp(np.mean([c[3] for c in chars]))),
                    flags=[],
                    _first=start,
                    _last=end,
                )
            )
            chars.clear()

    boundaries = np.r_[0, np.flatnonzero(np.diff(best)) + 1, len(best)]
    for first, last in zip(boundaries[:-1], boundaries[1:], strict=True):
        token = int(best[first])
        char = inverse.get(token, "")
        if token == blank:
            continue
        if char == "|":
            flush()
        elif len(char) == 1 and (char.isalnum() or char == "'"):
            chars.append((char, int(first), int(last), float(scores[first:last, token].mean())))
        else:
            flush()
    flush()
    return words


def _match(expected, observed, check, line_indices=None, phrase_support=None):
    """Word edit alignment with bounded phrase corrections and acoustic rechecking.

    Trace entries describe expected/observed consumption; rolling rows keep the
    numeric workspace linear while one byte per cell permits global traceback.
    Fuzzy substitutions are cheaper than deletion+insertion only at >=.65
    spelling similarity; acoustic support is still required after matching.
    """
    n, m = len(expected), len(observed)
    if n > 6000 or m > 12000 or n * m > 40_000_000:
        raise ValueError("Too many words for one alignment; analyze a shorter audio interval")
    trace = np.zeros((n + 1, m + 1), np.uint8)
    trace[0, 1:] = 2
    history = [np.arange(m + 1, dtype=np.float32)]
    indices = np.arange(1, m + 1)
    observed_array = np.array(observed)
    # Codes: diagonal, delete, insert, merge2, merge3, split2, split3.
    phrase_pairs = ((2, 2), (2, 3), (3, 2), (3, 3))
    consumed = ((1, 1), (1, 0), (0, 1), (2, 1), (3, 1), (1, 2), (1, 3)) + phrase_pairs
    observed_phrases = {
        count: np.array(["".join(observed[j - count + 1 : j + 1]) for j in range(count - 1, m)])
        for count in (2, 3)
    }
    for i, word in enumerate(expected, 1):
        check()
        previous = history[-1]
        costs = np.full(m, 2.2, dtype=np.float32)
        ratios = {}
        for j, other in enumerate(observed):
            if j % 256 == 0:
                check()
            if other not in ratios:
                upper = 2 * min(len(word), len(other)) / max(1, len(word) + len(other))
                ratio = difflib.SequenceMatcher(None, word, other).ratio() if upper >= 0.65 else 0
                ratios[other] = ratio
            ratio = ratios[other]
            if ratio == 1:
                costs[j] = 0
            elif ratio >= 0.65 and min(len(word), len(other)) >= 3:
                costs[j] = 1.5 - ratio
        diagonal = previous[:-1] + costs
        delete = previous[1:] + 1
        candidates = [diagonal, delete, np.full(m, np.inf)]
        for count in (2, 3):
            joined = "".join(expected[i - count : i]) if i >= count else None
            merged = np.full(m, np.inf)
            if joined is not None:
                merged = history[-count][:-1] + np.where(observed_array == joined, 0.15, np.inf)
            candidates.append(merged)
        for count in (2, 3):
            split = np.full(m, np.inf)
            for j in range(count - 1, m):
                if "".join(observed[j - count + 1 : j + 1]) == word:
                    split[j] = previous[j - count + 1] + 0.15
            candidates.append(split)
        for ne, no in phrase_pairs:
            phrase = np.full(m, np.inf)
            same_line = line_indices is None or (
                i >= ne and line_indices[i - ne] == line_indices[i - 1]
            )
            if i >= ne and m >= no and same_line:
                target = "".join(expected[i - ne : i])
                costs = np.where(observed_phrases[no] == target, 0.15, np.inf)
                # Two exact outer words constrain a misheard middle word. The
                # later acoustic pass still has to support every proposed letter.
                if ne == no == 3 and len(target) >= 8:
                    anchors = (observed_array[:-2] == expected[i - 3]) & (
                        observed_array[2:] == expected[i - 1]
                    )
                    for j in np.flatnonzero(anchors):
                        check()
                        ratio = difflib.SequenceMatcher(
                            None, target, observed_phrases[3][j]
                        ).ratio()
                        if ratio >= 0.8:
                            costs[j] = min(costs[j], 0.15 + 2 * (1 - ratio))
                if ne == no:
                    identical = np.ones(m - no + 1, dtype=bool)
                    for k in range(ne):
                        identical &= observed_array[k : k + m - no + 1] == expected[i - ne + k]
                    # Identical word sequences already have a cheaper diagonal
                    # path; do not repeat their acoustic work as phrase candidates.
                    costs[identical] = np.inf
                phrase[no - 1 :] = history[-ne][: m - no + 1] + costs
                if phrase_support is not None:
                    for j in np.flatnonzero(np.isfinite(phrase)):
                        if not phrase_support((int(j - no + 1), int(j + 1), i - ne, i)):
                            phrase[j] = np.inf
            candidates.append(phrase)
        base = np.min(candidates, axis=0)
        row = np.r_[float(i), np.minimum.accumulate(np.r_[float(i), base - indices])[1:] + indices]
        candidates[2] = row[:-1] + 1
        trace[i, 1:] = np.argmin(candidates, axis=0)
        trace[i, 0] = 1
        history.append(row)
        history = history[-3:]
    mapping, extra = {}, []
    i, j = n, m
    while i or j:
        ne, no = consumed[trace[i, j]]
        if ne and no:
            for index in range(i - ne, i):
                mapping[index] = (j - no, j, i - ne, i)
        elif no:
            extra.append(j - 1)
        i -= ne
        j -= no
    return mapping, sorted(extra)


def _matched_word_evidence(scores, vocabulary, target_words, observed_words, first, last, check):
    target = "".join(target_words)
    observed_tokens = [_normal(word["text"]) for word in observed_words]
    observed = "".join(observed_tokens)
    boundary_mismatch = len(target_words) != len(observed_words) or (
        target == observed and target_words != observed_tokens
    )
    similarity = difflib.SequenceMatcher(None, target, observed).ratio()
    if similarity < 0.65 or min(word["confidence"] for word in observed_words) < 0.15:
        return None
    spans = forced_token_spans(
        scores[first:last],
        [vocabulary[c] for c in target],
        blank=vocabulary.get("<pad>", 0),
        check=check,
    )
    result, position = [], 0
    for index, word in enumerate(target_words):
        word_spans = spans[position : position + len(word)]
        support = [
            float(scores[first + a : first + b, vocabulary[c]].mean())
            for c, (a, b) in zip(word, word_spans, strict=True)
        ]
        confidence = float(np.exp(np.mean(support)))
        # A long word must not average away a completely unsupported letter.
        if confidence < 0.15 or min(support) < np.log(0.05):
            result.append(None)
        else:
            flags = ["word_boundary_mismatch"] if boundary_mismatch else []
            if similarity < 1 and (
                len(target_words) != len(observed_tokens) or word != observed_tokens[index]
            ):
                flags.append("text_mismatch")
            result.append((first + word_spans[0][0], first + word_spans[-1][1], confidence, flags))
        position += len(word)
    return result


def align_lyrics(
    log_probs, vocabulary, lyrics, *, frame_seconds=0.02, offset_seconds=0.0125, check=None
):
    check = check or (lambda: None)
    check()
    scores = np.asarray(log_probs, dtype=np.float32)
    if scores.ndim != 2 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Expected finite CTC acoustic evidence")
    if frame_seconds <= 0 or not np.isfinite(frame_seconds + offset_seconds):
        raise ValueError("Invalid acoustic frame timing")
    recognized = _recognized(scores, vocabulary, frame_seconds, offset_seconds)
    lines, expected, section = [], [], ""
    for raw in lyrics.splitlines():
        text = raw.strip()
        if not text:
            continue
        if re.fullmatch(r"\[[^\]]+\]", text):
            section = text[1:-1]
            continue
        index = len(lines)
        words = [word for word in text.split() if _normal(word)]
        if not words:
            continue
        lines.append(
            dict(
                text=text,
                sectionLabel=section,
                wordIndices=list(range(len(expected), len(expected) + len(words))),
            )
        )
        expected.extend(dict(text=word, lineIndex=index) for word in words)
    extras = []
    corrections = {}
    if expected:
        normalized = [_normal(w["text"]) for w in expected]
        observed = [_normal(w["text"]) for w in recognized]
        grouped_evidence = {}

        def evidence_for(group_key):
            obs_first, obs_last, target_first, target_last = group_key
            target_words = normalized[target_first:target_last]
            # Repeated lyric occurrences share the same solve for each observed
            # window; target indices do not change its acoustic evidence.
            key = (obs_first, obs_last, tuple(target_words))
            if key not in grouped_evidence:
                if len(grouped_evidence) >= 4096:
                    grouped_evidence.pop(next(iter(grouped_evidence)))
                evidence = recognized[obs_first:obs_last]
                try:
                    grouped_evidence[key] = _matched_word_evidence(
                        scores,
                        vocabulary,
                        target_words,
                        evidence,
                        evidence[0]["_first"],
                        evidence[-1]["_last"],
                        check,
                    )
                except (ValueError, KeyError):
                    grouped_evidence[key] = None
            return grouped_evidence[key]

        def phrase_support(group_key):
            check()
            grouped = evidence_for(group_key)
            return grouped is not None and all(item is not None for item in grouped)

        mapping, insertions = _match(
            normalized, observed, check, [w["lineIndex"] for w in expected], phrase_support
        )
        words = []
        for i, supplied in enumerate(expected):
            check()
            word = dict(
                supplied,
                startSeconds=None,
                endSeconds=None,
                confidence=0.0,
                flags=[],
                verification="unresolved",
            )
            target = normalized[i]
            if any(char not in vocabulary for char in target):
                word["flags"].append("unsupported_characters")
            elif i not in mapping:
                word["flags"].append("possible_omission")
            else:
                obs_first, obs_last, target_first, target_last = mapping[i]
                evidence = recognized[obs_first:obs_last]
                word["observedText"] = (
                    evidence[i - target_first]["text"]
                    if obs_last - obs_first == target_last - target_first
                    else " ".join(item["text"] for item in evidence)
                )
                try:
                    grouped = evidence_for(mapping[i])
                    if (
                        grouped
                        and all(item is not None for item in grouped)
                        and (normalized[target_first:target_last] != observed[obs_first:obs_last])
                    ):
                        corrections[obs_first] = (
                            obs_last,
                            " ".join(w["text"] for w in expected[target_first:target_last]),
                        )
                    item = grouped[i - target_first] if grouped is not None else None
                    if item is None:
                        word["flags"].append("low_acoustic_support")
                    else:
                        begin, end, confidence, flags = item
                        word.update(
                            startSeconds=max(0, offset_seconds + (begin - 0.5) * frame_seconds),
                            endSeconds=offset_seconds + (end - 0.5) * frame_seconds,
                            confidence=confidence,
                        )
                        word["flags"].extend(flags)
                        word["verification"] = "lyric_assisted" if flags else "matched"
                except ValueError:
                    word["flags"].append("unrecognized")
            words.append(word)
        for j in insertions:
            word = {k: v for k, v in recognized[j].items() if not k.startswith("_")}
            word["flags"] = [
                "possible_repeat" if observed[j] in normalized else "extra_recognized_word"
            ]
            extras.append(word)
        status = "aligned_needs_review"
    else:
        words = [{k: v for k, v in w.items() if not k.startswith("_")} for w in recognized]
        status = "transcribed_unreviewed" if words else "no_words_detected"
    assisted, index = [], 0
    while index < len(recognized):
        if index in corrections:
            index, text = corrections[index]
            assisted.append(text)
        else:
            assisted.append(recognized[index]["text"])
            index += 1
    for line in lines:
        members = [words[i] for i in line["wordIndices"]]
        timed = [w for w in members if w["startSeconds"] is not None]
        line.update(
            startSeconds=min((w["startSeconds"] for w in timed), default=None),
            endSeconds=max((w["endSeconds"] for w in timed), default=None),
            confidence=sum(w["confidence"] for w in members) / len(members),
            flags=sorted({f for w in members for f in w["flags"]}),
        )
    timed = sorted(
        (w for w in words + extras if w["startSeconds"] is not None),
        key=lambda w: w["startSeconds"],
    )
    pauses, regions = [], []
    for word in timed:
        if regions and word["startSeconds"] - regions[-1]["endSeconds"] <= 0.35:
            regions[-1]["endSeconds"] = max(regions[-1]["endSeconds"], word["endSeconds"])
        else:
            if regions:
                start, end = regions[-1]["endSeconds"], word["startSeconds"]
                pauses.append(dict(startSeconds=start, endSeconds=end, durationSeconds=end - start))
            regions.append(dict(startSeconds=word["startSeconds"], endSeconds=word["endSeconds"]))
    return dict(
        method="ctc-evidence-alignment-v2",
        status=status,
        language="en",
        words=words,
        lines=lines,
        extraWords=extras,
        pauses=pauses,
        vocalRegions=regions,
        recognizedText=" ".join(w["text"] for w in recognized),
        lyricAssistedText=" ".join(assisted),
        confidenceMeaning="Uncalibrated acoustic support; not probability of correct lyrics",
        limitations=[
            "English acoustic model; singing and mixed music need human review.",
            "Missing recognition is not proof a lyric was omitted; "
            "vocal regions are acoustic candidates.",
        ],
    )
