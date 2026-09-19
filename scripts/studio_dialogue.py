"""Sequential native dialogue renders with prepared references and immutable turn takes."""

from __future__ import annotations

import copy
import json
import os
import struct
import time
from pathlib import Path

from wee_todd_mlx.audio_mix.plan import number
from wee_todd_mlx.speech_reference import digest_file, prepare_reference


def validate_request(value):
    from studio_voice import validate_request as validate_voice

    if not isinstance(value, dict):
        raise ValueError("Choose a dialogue request")
    turns = value.get("turns")
    if not isinstance(turns, list) or not 1 <= len(turns) <= 64:
        raise ValueError("Dialogue needs between 1 and 64 turns")
    shared = {
        k: copy.deepcopy(v)
        for k, v in value.items()
        if k in {"engine", "model_path", "precision", "language", "seed", "sampling"}
    }
    seed = shared.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be a 32-bit nonnegative integer")
    if not isinstance(shared.get("sampling", {}), dict):
        raise ValueError("Choose speech sampling settings")
    language = shared.get("language", "auto")
    if not isinstance(language, str) or not language.strip():
        raise ValueError("Choose a speech language")
    normalized, requests, ids = [], [], set()
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError("Choose a dialogue turn")
        for field in ("id", "speaker_id", "speaker_name", "text"):
            if not isinstance(turn.get(field), str) or not turn[field].strip():
                raise ValueError(f"Each dialogue turn needs {field}")
        if turn["id"] in ids:
            raise ValueError("Dialogue turn IDs must be unique")
        ids.add(turn["id"])
        gap = number(turn.get("gap_after", 0), "dialogue pause", 0, 10)
        request = validate_voice(
            dict(
                shared,
                text=turn["text"],
                reference_mode=turn.get("reference_mode", "audioAndTranscript"),
                reference=copy.deepcopy(turn.get("reference")),
                seed=(seed + index) % 2**32,
                **{key: turn[key] for key in ("speaker", "instruct") if key in turn},
            )
        )
        normalized.append(
            dict(
                id=turn["id"],
                speaker_id=turn["speaker_id"],
                speaker_name=turn["speaker_name"],
                text=request["text"],
                reference_mode=request["reference_mode"],
                reference=request.get("reference"),
                gap_after=gap,
                **{key: request[key] for key in ("speaker", "instruct") if key in request},
            )
        )
        requests.append(request)
    shared.update(seed=seed, sampling=copy.deepcopy(requests[0]["sampling"]), turns=normalized)
    return shared, requests


def _assemble(target, turns, frames, rate, check):
    """Stream IEEE float WAV to keep long dialogues from duplicating audio in memory."""
    import numpy as np
    from scipy.io import wavfile

    size = frames * 4
    if size + 48 >= 2**32:
        raise ValueError("Dialogue exceeds the WAV file size limit")
    with target.open("wb") as stream:
        stream.write(
            struct.pack(
                "<4sI4s4sIHHIIHH4sII4sI",
                b"RIFF",
                size + 48,
                b"WAVE",
                b"fmt ",
                16,
                3,
                1,
                rate,
                rate * 4,
                4,
                32,
                b"fact",
                4,
                frames,
                b"data",
                size,
            )
        )
        for turn in turns:
            check()
            _, audio = wavfile.read(turn["audio"], mmap=True)
            for offset in range(0, len(audio), 262144):
                check()
                stream.write(np.asarray(audio[offset : offset + 262144], dtype="<f4").tobytes())
            remaining = turn["gap_frames"]
            while remaining:
                check()
                count = min(remaining, 262144)
                stream.write(bytes(count * 4))
                remaining -= count


def dispatch(request, output=None, progress=None, cancelled=None):
    cancelled = cancelled or (lambda: False)

    def check():
        if cancelled():
            raise InterruptedError("Dialogue generation cancelled")

    def report(message):
        if progress:
            progress(dict(event="progress", message=message))

    check()
    value, requests = validate_request(request.get("voice", {}))
    if output is None:
        raise ValueError("Choose a new folder for the dialogue take")
    from wee_todd_mlx.speech_checkpoint import inspect

    report("Checking local speech checkpoint")
    check()
    spec = inspect(
        value["model_path"], engine=value["engine"], precision=value.get("precision", "auto")
    )
    from studio_voice import validate_checkpoint_request

    for turn in requests:
        validate_checkpoint_request(turn, spec)
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        check()
        (root / "request.json").write_text(json.dumps(value, indent=2, allow_nan=False))
        references = []
        for index, turn in enumerate(requests):
            check()
            report(f"Preparing dialogue reference {index + 1} of {len(requests)}")
            reference = None
            if turn["reference_mode"] not in {"synthetic", "customVoice"}:
                reference = prepare_reference(
                    turn["reference"],
                    engine=value["engine"],
                    mode=turn["reference_mode"],
                    model_identity=spec["identity"],
                    output=root.parent / "reference-cache",
                    runtime=request.get("runtime", {}),
                    cancelled=cancelled,
                )
            references.append(reference)
        check()
        if value["engine"] == "fishS2Pro":
            from fish_speech_mlx.pipeline import generate
        else:
            from qwen3_tts_mlx.pipeline import generate
        import numpy as np
        from scipy.io import wavfile

        rate = 44100 if value["engine"] == "fishS2Pro" else 24000
        turns, cursor = [], 0
        for index, (turn, reference) in enumerate(zip(requests, references, strict=True)):
            check()
            report(f"Rendering dialogue turn {index + 1} of {len(requests)}")
            folder = root / "turns" / f"{index + 1:03d}"
            folder.mkdir(parents=True)
            (folder / "request.json").write_text(json.dumps(turn, indent=2, allow_nan=False))
            result = generate(
                turn,
                reference_path=reference["path"] if reference else None,
                inspection=spec,
                cancelled=cancelled,
                progress=report,
            )
            check()
            audio = np.asarray(result["audio"], dtype=np.float32)
            if (
                audio.ndim != 1
                or not len(audio)
                or not np.isfinite(audio).all()
                or np.max(abs(audio)) < 1e-7
            ):
                raise ValueError("Speech did not produce a finite audible take")
            if result["sample_rate"] != rate:
                raise ValueError("Dialogue turn has an unexpected sample rate")
            metadata = value["turns"][index]
            gap_frames = round(metadata["gap_after"] * rate)
            response = dict(
                audio=str(folder / "take.wav"),
                duration=len(audio) / rate,
                frames=len(audio),
                sample_rate=rate,
                channels=1,
                artifacts=str(folder),
                truncated=bool(result["truncated"]),
                request=turn,
                id=metadata["id"],
                speaker_id=metadata["speaker_id"],
                speaker_name=metadata["speaker_name"],
                start_frame=cursor,
                end_frame=cursor + len(audio),
                gap_frames=gap_frames,
            )
            check()
            pending = folder / "take.pending.wav"
            wavfile.write(pending, rate, audio)
            np.save(folder / "codes.npy", result["codes"], allow_pickle=False)
            receipt = dict(
                version=1,
                state="complete",
                result=response,
                model_identity=spec["identity"],
                reference=reference,
                audio_sha256=digest_file(pending),
                codes_sha256=digest_file(folder / "codes.npy"),
            )
            (folder / "receipt.pending.json").write_text(
                json.dumps(receipt, indent=2, allow_nan=False)
            )
            check()
            os.replace(pending, folder / "take.wav")
            check()
            os.replace(folder / "receipt.pending.json", folder / "receipt.json")
            turns.append(response)
            cursor += len(audio) + gap_frames
            del audio, result
        check()
        report("Assembling dialogue take")
        pending = root / "take.pending.wav"
        _assemble(pending, turns, cursor, rate, check)
        response = dict(
            audio=str(root / "take.wav"),
            duration=cursor / rate,
            frames=cursor,
            sample_rate=rate,
            channels=1,
            artifacts=str(root),
            truncated=any(t["truncated"] for t in turns),
            request=value,
            turns=turns,
        )
        check()
        receipt = dict(
            version=1,
            state="complete",
            result=response,
            model_identity=spec["identity"],
            audio_sha256=digest_file(pending),
            elapsed=time.monotonic() - started,
        )
        (root / "receipt.pending.json").write_text(json.dumps(receipt, indent=2, allow_nan=False))
        check()
        os.replace(pending, root / "take.wav")
        check()
        os.replace(root / "receipt.pending.json", root / "receipt.json")
        return response
    except BaseException as error:
        (root / "failure.json").write_text(
            json.dumps(
                dict(
                    state="cancelled" if isinstance(error, InterruptedError) else "failed",
                    error=str(error),
                )
            )
        )
        raise
