"""Sequential learned audio analysis with content provenance and reusable acoustic evidence."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

from ..music_video import _decode_audio, analyze_audio, verify_source_audio
from .alignment import align_lyrics
from .setup import model_identity
from .structure import structure_analysis

VERSION = "native-audio-analysis-v1"
CACHE_REVISION = 3
VOCAL_CACHE_REVISION = 1


def _number(value, low=0, high=float("inf")):
    return type(value) in (int, float) and np.isfinite(value) and low <= value <= high


def _read_json(path, check):
    check()
    try:
        if path.stat().st_size > 32 * 1024 * 1024:
            return None
        value = json.loads(path.read_text())
    except (OSError, ValueError, RecursionError):
        return None
    check()
    # Reject NaN/Infinity, including overflowing ordinary JSON numeric literals.
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, float) and not np.isfinite(item):
            return None
    return value


def _events_valid(value, duration):
    if not isinstance(value, dict):
        return False
    for kind in ("beats", "downbeats"):
        events = value.get(kind)
        if not isinstance(events, list):
            return False
        previous = -1.0
        for event in events:
            if (
                not isinstance(event, dict)
                or not _number(event.get("timeSeconds"), 0, duration)
                or not _number(event.get("confidence"), 0, 1)
                or event["timeSeconds"] <= previous
            ):
                return False
            previous = event["timeSeconds"]
    return True


def _interval_valid(item, duration, *, optional=False):
    if not isinstance(item, dict) or "startSeconds" not in item or "endSeconds" not in item:
        return False
    start, end = item["startSeconds"], item["endSeconds"]
    if optional and start is None and end is None:
        return True
    return _number(start, 0, duration) and _number(end, start, duration + 0.02)


def _result_valid(value, key, content_hash, models, lyrics_status, vocal_provenance):
    if not isinstance(value, dict):
        return False
    if (
        value.get("cacheKey") != key
        or value.get("sourceSHA256") != content_hash
        or value.get("method") != VERSION
        or value.get("modelProvenance") != models
        or value.get("vocalAnalysis") != vocal_provenance
        or not _number(value.get("durationSeconds"), 0.00001, 3600)
        or not _number(value.get("sampleRate"), 1, 192000)
    ):
        return False
    from ..workflows.validation import validate_value

    try:
        if validate_value("music_analysis", value):
            return False
    except (ValueError, RecursionError):
        return False
    duration = value["durationSeconds"]
    if not _events_valid(value, duration):
        return False
    required_lists = ("cues", "energyEnvelope", "bars", "sections", "vocalGaps", "limitations")
    if any(not isinstance(value.get(name), list) for name in required_lists):
        return False
    if (
        "beatSuggestion" not in value
        or not _number(value.get("energyHopSeconds"), 0.00001)
        or any(not _number(e) for e in value["energyEnvelope"])
    ):
        return False
    for cue in value["cues"]:
        if (
            not isinstance(cue, dict)
            or not isinstance(cue.get("kind"), str)
            or not _number(cue.get("timeSeconds"), 0, duration)
            or not _number(cue.get("strength"))
        ):
            return False
    for kind in ("bars", "sections", "vocalGaps"):
        for item in value[kind]:
            if not _interval_valid(item, duration):
                return False
            if kind != "vocalGaps" and not _number(item.get("confidence"), 0, 1):
                return False
            if kind == "bars" and (type(item.get("beatCount")) is not int or item["beatCount"] < 0):
                return False
            if kind == "sections" and not isinstance(item.get("label"), str):
                return False
    if "alignment" not in value:
        return False
    alignment = value["alignment"]
    if lyrics_status == "instrumental":
        return alignment is None
    if not isinstance(alignment, dict) or not isinstance(alignment.get("status"), str):
        return False
    if not all(
        isinstance(alignment.get(name), str) for name in ("recognizedText", "lyricAssistedText")
    ):
        return False
    for name in ("words", "lines", "extraWords", "pauses", "vocalRegions"):
        if not isinstance(alignment.get(name), list):
            return False
        for item in alignment[name]:
            if (
                name == "words"
                and alignment["status"] == "aligned_needs_review"
                and (
                    not isinstance(item, dict)
                    or item.get("verification") not in ("matched", "lyric_assisted", "unresolved")
                )
            ):
                return False
            if not _interval_valid(item, duration, optional=name in ("words", "lines")):
                return False
            if "observedText" in item and not isinstance(item["observedText"], str):
                return False
            if name in ("words", "lines", "extraWords") and (
                not isinstance(item.get("text"), str) or not _number(item.get("confidence"), 0, 1)
            ):
                return False
    return True


def _read_ctc(path, root, sample_count, check):
    from .ctc_model import output_frames

    check()
    vocabulary = json.loads((root / "wav2vec2-base-960h" / "vocab.json").read_text())
    frames = output_frames((sample_count * 320 + 440) // 441)
    # Bound archive expansion before NumPy allocation; only scalar timing, vocab and scores.
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if (
                len(members) != 4
                or sum(item.file_size for item in members) > frames * len(vocabulary) * 4 + 32768
            ):
                return None
        with np.load(path, allow_pickle=False) as arrays:
            evidence = dict(
                log_probs=arrays["scores"],
                vocabulary=json.loads(str(arrays["vocabulary"])),
                frame_seconds=float(arrays["frame_seconds"]),
                offset_seconds=float(arrays["offset_seconds"]),
            )
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
        return None
    check()
    scores = evidence["log_probs"]
    if (
        scores.dtype != np.float32
        or scores.shape != (frames, len(vocabulary))
        or evidence["vocabulary"] != vocabulary
        or evidence["frame_seconds"] != 0.02
        or evidence["offset_seconds"] != 0.0125
    ):
        return None
    for start in range(0, len(scores), 8192):
        check()
        block = scores[start : start + 8192]
        if (
            not np.isfinite(block).all()
            or np.max(block, initial=-np.inf) > 1e-4
            or not np.allclose(np.exp(block).sum(axis=1), 1, atol=0.002, rtol=0)
        ):
            return None
    return evidence


def _publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, allow_nan=False)
        temporary = Path(stream.name)
    temporary.replace(path)


def vocal_settings(mode):
    if mode == "isolated":
        from .separation import SETTINGS_ID

        return SETTINGS_ID
    return None


def _cache_key(**identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _model_files(models, model_id):
    return {name: digest for name, digest in models.items() if name.startswith(model_id + "/")}


def _file_digest(path, check):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            check()
            digest.update(block)
    return digest.hexdigest()


def _vocals(source, content_hash, root, directory, key, sample_count, check):
    """Cache only a verified time-preserving stereo waveform; never replace source media."""
    target = directory / (key + "-vocals.npy")
    receipt_path = directory / (key + "-vocals.json")
    receipt = _read_json(receipt_path, check)
    try:
        if (
            isinstance(receipt, dict)
            and receipt.get("cacheKey") == key
            and receipt.get("sampleRate") == 44100
            and target.stat().st_size <= (sample_count * 2 + 1) * 8 + 1024
            and receipt.get("sha256") == _file_digest(target, check)
        ):
            cached = np.load(target, allow_pickle=False, mmap_mode="r")
            if not isinstance(cached, np.ndarray):
                cached.close()
                raise ValueError("Expected a cached vocal waveform")
            if cached.dtype == np.float32 and cached.ndim == 2 and cached.shape[1] == 2:
                if abs(len(cached) - sample_count * 2) <= 1:
                    for start in range(0, len(cached), 44100 * 30):
                        check()
                        if not np.isfinite(cached[start : start + 44100 * 30]).all():
                            break
                    else:
                        return cached
    except InterruptedError:
        raise
    except (OSError, ValueError, TypeError, EOFError):
        pass
    from ..conditioning_media import media_binary
    from .separation import separate_vocals

    check()
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
        "2",
        "-ar",
        "44100",
        "-f",
        "f32le",
        "pipe:1",
    ]
    samples = np.frombuffer(_decode_audio(command, check=check), dtype="<f4").reshape(-1, 2)
    if len(samples) > 3600 * 44100 or abs(len(samples) - sample_count * 2) > 1:
        raise ValueError("Vocal decode must retain the complete song timeline")
    result = separate_vocals(samples, 44100, root, check=check)
    check()
    vocals = np.asarray(result["vocals"])
    if (
        result["sample_rate"] != 44100
        or vocals.shape != samples.shape
        or vocals.dtype != np.float32
    ):
        raise ValueError("Vocal isolation changed the source timeline or stereo format")
    for start in range(0, len(vocals), 44100 * 30):
        check()
        if not np.isfinite(vocals[start : start + 44100 * 30]).all():
            raise ValueError("Vocal isolation returned nonfinite audio")
    del samples, result
    verify_source_audio(source, content_hash, check=check)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".npy", delete=False) as stream:
            temporary = Path(stream.name)
            np.save(stream, vocals, allow_pickle=False)
        receipt = dict(cacheKey=key, sampleRate=44100, sha256=_file_digest(temporary, check))
        check()
        temporary.replace(target)
        _publish(receipt_path, receipt)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return vocals


def analyze(
    source,
    *,
    expected_sha256,
    model_directory="",
    lyrics="",
    lyrics_status="unknown",
    mode="neural",
    vocal_mode="mixed",
    cache_directory,
    check=None,
):
    check = check or (lambda: None)
    check()
    if mode not in {"fast", "neural"} or lyrics_status not in {
        "unknown",
        "supplied",
        "instrumental",
    }:
        raise ValueError("Choose a supported audio analysis mode and lyrics status")
    if vocal_mode not in {"mixed", "isolated"}:
        raise ValueError("Choose mixed or isolated vocal analysis")
    if mode == "fast" and vocal_mode == "isolated":
        raise ValueError("Vocal isolation requires learned (neural) audio analysis")
    if lyrics_status == "instrumental":
        vocal_mode = "mixed"
    lyrics = lyrics if lyrics_status == "supplied" else ""
    source, content_hash = verify_source_audio(source, expected_sha256, check=check)
    directory = Path(cache_directory)
    if mode == "fast":
        return analyze_audio(
            source, expected_sha256=content_hash, cache_directory=directory, check=check
        )
    root = Path(model_directory).expanduser()
    models = model_identity(root, check=check, include_vocals=vocal_mode == "isolated")
    settings = vocal_settings(vocal_mode)
    vocal_info = dict(
        mode=vocal_mode,
        **(
            {
                "modelID": "umxhq-vocals",
                "settings": settings,
                "method": "Native stereo UMX magnitude mask; mixture phase",
            }
            if vocal_mode == "isolated"
            else {}
        ),
    )
    identity = dict(
        version=VERSION,
        implementation=CACHE_REVISION,
        source=content_hash,
        models=models,
        lyrics=lyrics,
        lyricsStatus=lyrics_status,
        vocalMode=vocal_mode,
        vocalSettings=settings,
    )
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination = directory / (key + ".json")
    saved = _read_json(destination, check)
    if _result_valid(saved, key, content_hash, models, lyrics_status, vocal_info):
        check()
        return saved
    started = time.perf_counter()
    basic = analyze_audio(
        source, expected_sha256=content_hash, cache_directory=directory, check=check
    )
    from ..conditioning_media import media_binary

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
        "22050",
        "-f",
        "f32le",
        "pipe:1",
    ]
    audio = np.frombuffer(_decode_audio(command, check=check), dtype="<f4")
    if len(audio) > 3600 * 22050:
        raise ValueError("Analyze audio up to one hour")
    beat_key = _cache_key(
        stage="beat-this-v1", source=content_hash, models=_model_files(models, "beat-this-small0")
    )
    beat_path = directory / (beat_key + "-beats.json")
    from .beat_model import beat_events

    beats = _read_json(beat_path, check)
    if not _events_valid(beats, len(audio) / 22050):
        check()
        beats = beat_events(audio, root / "beat-this-small0", check=check)
        verify_source_audio(source, content_hash, check=check)
        _publish(beat_path, beats)
    alignment = None
    if lyrics_status != "instrumental":
        from scipy.signal import resample_poly

        from .ctc_model import ctc_emissions

        vocal_key = _cache_key(
            stage="umxhq-vocals",
            implementation=VOCAL_CACHE_REVISION,
            source=content_hash,
            settings=settings,
            models=_model_files(models, "umxhq-vocals"),
        )
        acoustic_key = _cache_key(
            stage="wav2vec2-v1",
            source=content_hash,
            models=_model_files(models, "wav2vec2-base-960h"),
            vocals=vocal_key if vocal_mode == "isolated" else "mixed",
        )
        acoustic_path = directory / (acoustic_key + "-ctc.npz")
        evidence = _read_ctc(acoustic_path, root, len(audio), check)
        if evidence is None:
            check()
            if vocal_mode == "isolated":
                vocals = _vocals(
                    source, content_hash, root, directory, vocal_key, len(audio), check
                )
                verify_source_audio(source, content_hash, check=check)
                # Preserve the exact mixed-waveform frame grid, including odd resample lengths.
                acoustic_audio = resample_poly(vocals.mean(axis=1), 160, 441).astype(np.float32)
                acoustic_audio = acoustic_audio[: (len(audio) * 320 + 440) // 441]
                del vocals
            else:
                acoustic_audio = resample_poly(audio, 320, 441).astype(np.float32)
            check()
            evidence = ctc_emissions(
                acoustic_audio,
                root / "wav2vec2-base-960h",
                check=check,
            )
            del acoustic_audio
            verify_source_audio(source, content_hash, check=check)
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".npz", delete=False) as stream:
                np.savez_compressed(
                    stream,
                    scores=evidence["log_probs"],
                    vocabulary=json.dumps(evidence["vocabulary"]),
                    frame_seconds=evidence["frame_seconds"],
                    offset_seconds=evidence["offset_seconds"],
                )
                temporary = Path(stream.name)
            temporary.replace(acoustic_path)
        alignment = align_lyrics(
            evidence["log_probs"],
            evidence["vocabulary"],
            lyrics,
            frame_seconds=evidence["frame_seconds"],
            offset_seconds=evidence["offset_seconds"],
            check=check,
        )
        del evidence
    structure = structure_analysis(
        audio,
        beats=beats["beats"],
        downbeats=beats["downbeats"],
        words=(alignment or {}).get("words", []) + (alignment or {}).get("extraWords", []),
        lines=(alignment or {}).get("lines"),
        check=check,
    )
    for kind in ("beat", "downbeat"):
        basic["cues"].extend(
            dict(
                timeSeconds=b["timeSeconds"], kind=kind, strength=b["confidence"], provisional=True
            )
            for b in beats[kind + "s"]
        )
    basic["cues"].extend(
        dict(
            timeSeconds=s["startSeconds"],
            kind="section",
            strength=max(0.1, s["confidence"]),
            provisional=True,
        )
        for s in structure["sections"]
    )
    basic["cues"].sort(key=lambda c: c["timeSeconds"])
    if len(beats["beats"]) >= 3:
        intervals = np.diff([b["timeSeconds"] for b in beats["beats"]])
        basic["beatSuggestion"] = {
            "bpm": round(float(60 / np.median(intervals)), 2),
            "provisional": True,
            "basis": "Median learned beat spacing; tempo may vary",
        }
    basic.update(
        method=VERSION,
        cacheKey=key,
        alignment=alignment,
        beats=beats["beats"],
        downbeats=beats["downbeats"],
        **structure,
        modelProvenance=models,
        vocalAnalysis=vocal_info,
        elapsedSeconds=time.perf_counter() - started,
        limitations=[
            "Model activation and acoustic-support scores are not "
            "calibrated correctness probabilities.",
            "English acoustic model; singing, lyric omissions/repeats "
            "and structure require review.",
            "No speaker diarization or guaranteed lip synchronization.",
            *(
                [
                    "Vocal isolation may retain instruments or remove sung sounds; "
                    "it does not guarantee usable word timing."
                ]
                if vocal_mode == "isolated"
                else []
            ),
        ],
    )
    verify_source_audio(source, content_hash, check=check)
    check()
    _publish(destination, basic)
    return basic
