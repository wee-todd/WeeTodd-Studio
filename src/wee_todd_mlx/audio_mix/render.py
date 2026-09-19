"""Sample-accurate bounded-memory mixing with FFmpeg decode and output limiting.

Mono uses equal-power panning; stereo uses balance (centre leaves both channels intact).
Ducking uses fixed 10-ms RMS windows before pan, attack/release interpolation and a
bounded attenuation. All windows start at movie sample zero for repeatable subranges.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path

from wee_todd_mlx.speech_reference import digest_file, run_media, tool

from .cache import decoded as cached_decode
from .cache import locked


def render_mix(plan, output, runtime, *, cancelled=None):
    import numpy as np
    from scipy.io import wavfile

    cancelled = cancelled or (lambda: False)
    ffmpeg, ffprobe = tool("ffmpeg", runtime), tool("ffprobe", runtime)
    identity = json.loads(json.dumps(plan))
    # Purpose doesn't change PCM; mute/solo and driver selection already compiled into inputs.
    identity.pop("purpose", None)
    source_hashes = {}
    probes = {}
    for item in plan["inputs"]:
        source = Path(item["path"])
        if not source.is_file():
            raise ValueError("An audible audio file is missing. Relink it before mixing.")
        if str(source) not in source_hashes:
            source_hashes[str(source)] = digest_file(source)
            probe = json.loads(
                run_media(
                    [
                        ffprobe,
                        "-v",
                        "error",
                        "-select_streams",
                        "a:0",
                        "-show_entries",
                        "stream=channels",
                        "-of",
                        "json",
                        str(source),
                    ],
                    cancelled=cancelled,
                )
            )
            streams = probe.get("streams", [])
            probes[str(source)] = int(streams[0]["channels"]) if streams else 0
        if probes[str(source)] not in {0, 1, 2}:
            raise ValueError("Mixing supports mono and stereo audio")
        if not probes[str(source)] and item["role"] != "source":
            raise ValueError("An audio region has no audio stream")
    version = run_media([ffmpeg, "-version"], cancelled=cancelled).decode().splitlines()[0]
    identity.update(sources=source_hashes, ffmpeg=version, renderer="studio-pcm-v3")
    key = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    folder = Path(output).expanduser().resolve() / key
    folder.mkdir(parents=True, exist_ok=True)
    target, receipt = folder / "mix.wav", folder / "mix.json"
    with locked(folder / "build.lock", cancelled):
        if target.is_file() and receipt.is_file():
            try:
                saved = json.loads(receipt.read_text())
                if saved["sha256"] == digest_file(target) and saved["identity"] == identity:
                    result = saved["result"]
                    if plan["purpose"] == "driver" and result["peak"] < 1e-7:
                        raise ValueError("The selected audio driver is silent")
                    os.utime(target, None)
                    return result
            except (OSError, KeyError, json.JSONDecodeError):
                pass
        if cancelled():
            raise InterruptedError("Audio mix cancelled")
        rate = 48000
        count = round((plan["start"] + plan["duration"]) * rate)
        first = round(plan["start"] * rate)
        count = max(count, first + plan["frames"]) + 2400  # Limiter lookahead/right context.
        with (
            ExitStack() as leases,
            tempfile.TemporaryDirectory(prefix="prepare-", dir=folder) as temp,
        ):
            temp = Path(temp)
            voice = np.memmap(temp / "voice.pcm", dtype="float32", mode="w+", shape=(count, 2))
            mixed = np.memmap(temp / "sum.pcm", dtype="float32", mode="w+", shape=(count, 2))
            mixed[:] = 0
            voice[:] = 0
            decoded = []
            for item in plan["inputs"]:
                if cancelled():
                    raise InterruptedError("Audio mix cancelled")
                channels = probes[item["path"]]
                if not channels:
                    continue
                frames = min(round(item["duration"] * rate), count - round(item["start"] * rate))
                if frames <= 0:
                    continue
                source_start = round(item["source_in"] * rate)

                def build(
                    pcm, item=item, source_start=source_start, frames=frames, channels=channels
                ):
                    run_media(
                        [
                            ffmpeg,
                            "-v",
                            "error",
                            "-nostdin",
                            "-i",
                            item["path"],
                            "-map",
                            "0:a:0",
                            "-af",
                            (
                                f"aresample={rate},atrim=start_sample={source_start}:"
                                f"end_sample={source_start + frames},asetpts=PTS-STARTPTS"
                            ),
                            "-ac",
                            str(channels),
                            "-f",
                            "f32le",
                            str(pcm),
                        ],
                        cancelled=cancelled,
                    )

                decode_identity = dict(
                    source=source_hashes[item["path"]],
                    ffmpeg=version,
                    rate=rate,
                    channels=channels,
                    start=source_start,
                    frames=frames,
                )
                pcm = leases.enter_context(
                    cached_decode(Path(output) / "decoded", decode_identity, build, cancelled)
                )
                available = pcm.stat().st_size // (4 * channels)
                if not available:
                    continue
                data = np.memmap(pcm, dtype="float32", mode="r", shape=(available, channels))
                decoded.append((item, data, frames))
                if item["role"] == "voice":
                    at = round(item["start"] * rate)
                    for j in range(0, min(frames, available), 48000):
                        k = min(j + 48000, frames, available)
                        voice[at + j : at + k] += (
                            data[j:k] * envelope(item, j, k, rate)[:, None] * item["gain"]
                        )
            # 10 ms RMS buckets, aligned globally. Voice is measured before pan.
            rms = np.zeros(math.ceil(count / 480), dtype=np.float32)
            for j in range(len(rms)):
                v = voice[j * 480 : min(count, (j + 1) * 480)]
                rms[j] = np.sqrt(np.mean(v * v))
            duck_cache = {}

            def duck_values(d):
                duck = None
                if d:
                    dk = json.dumps(d, sort_keys=True)
                    if dk not in duck_cache:
                        threshold = 10 ** (d.get("thresholdDb", -36) / 20)
                        target_db = np.clip(
                            20 * np.log10(np.maximum(rms, 1e-12) / threshold),
                            0,
                            d.get("amountDb", 12),
                        )
                        values = np.zeros(len(rms) + 1, dtype=np.float32)
                        state = 0.0
                        for j, value in enumerate(target_db):
                            tau = d.get("attack", 0.02) if value > state else d.get("release", 0.25)
                            state = value + (state - value) * math.exp(-0.01 / tau)
                            values[j + 1] = state
                        duck_cache[dk] = values
                    duck = duck_cache[dk]
                return duck

            def mix_item(item, data, frames, destination, *, apply_duck=True):
                duck = duck_values(item["ducking"]) if apply_duck else None
                at = round(item["start"] * rate)
                for j in range(0, min(frames, len(data)), 48000):
                    if cancelled():
                        raise InterruptedError("Audio mix cancelled")
                    k = min(j + 48000, frames, len(data))
                    samples = np.array(data[j:k])
                    if not np.isfinite(samples).all():
                        raise ValueError("An audio source contains non-finite samples")
                    pan = item["pan"]
                    if samples.shape[1] == 1:
                        angle = (pan + 1) * math.pi / 4
                        samples = samples * np.array([[math.cos(angle), math.sin(angle)]])
                    else:
                        samples *= [min(1, 1 - pan), min(1, 1 + pan)]
                    gain = envelope(item, j, k, rate) * item["gain"]
                    if duck is not None:
                        db = np.interp(
                            (np.arange(at + j, at + k)) / 480, np.arange(len(duck)), duck
                        )
                        gain *= 10 ** (-db / 20)
                    for left, right in item["replacements"]:
                        indices = np.arange(at + j, at + k)
                        gain[(indices >= round(left * rate)) & (indices < round(right * rate))] = 0
                    destination[at + j : at + k] += samples * gain[:, None]

            buses = {}
            for item, data, frames in decoded:
                if item.get("reverb"):
                    buses.setdefault(item["bus"], []).append((item, data, frames))
                else:
                    mix_item(item, data, frames, mixed)
            if buses:
                from .reverb import add_reverb

                # Process one track at a time. Disk-backed buses keep memory bounded even
                # for long movies; all regions share a single tail, including split regions.
                for index, items in enumerate(buses.values()):
                    bus_file = temp / f"bus-{index}.pcm"
                    bus = np.memmap(bus_file, dtype="float32", mode="w+", shape=(count, 2))
                    bus[:] = 0
                    for item, data, frames in items:
                        mix_item(item, data, frames, bus, apply_duck=False)
                    settings = items[0][0]["reverb"]
                    duck = duck_values(items[0][0]["ducking"])
                    if duck is None:
                        add_reverb(bus, mixed, settings, sample_rate=rate, cancelled=cancelled)
                    else:
                        # Duck the complete music bus, including previously excited tails.
                        wet_file = temp / f"wet-{index}.pcm"
                        wet = np.memmap(wet_file, dtype="float32", mode="w+", shape=(count, 2))
                        wet[:] = 0
                        add_reverb(bus, wet, settings, sample_rate=rate, cancelled=cancelled)
                        for j in range(0, count, 48000):
                            if cancelled():
                                raise InterruptedError("Audio mix cancelled")
                            k = min(j + 48000, count)
                            db = np.interp(np.arange(j, k) / 480, np.arange(len(duck)), duck)
                            mixed[j:k] += wet[j:k] * (10 ** (-db / 20))[:, None]
                        del wet
                        wet_file.unlink()
                    del bus
                    bus_file.unlink()
            mixed.flush()
            pending = temp / "mix.wav"
            # Disable auto makeup gain and compensate lookahead; emit the exact requested frames.
            limiter = "alimiter=limit=0.95:level=0:latency=1,"
            if plan["policy"] == "legacy-v1":
                limiter = (
                    "alimiter=limit=0.95:level=1:latency=1,"
                    if any(i["role"] != "source" for i in plan["inputs"])
                    else ""
                )
            run_media(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-nostdin",
                    "-f",
                    "f32le",
                    "-ar",
                    str(rate),
                    "-ac",
                    "2",
                    "-i",
                    str(temp / "sum.pcm"),
                    "-af",
                    (
                        f"{limiter}atrim=start_sample={first}:"
                        f"end_sample={first + plan['frames']},asetpts=PTS-STARTPTS"
                    ),
                    "-c:a",
                    "pcm_f32le",
                    str(pending),
                ],
                cancelled=cancelled,
            )
            actual_rate, audio = wavfile.read(pending, mmap=True)
            if actual_rate != rate or audio.shape != (plan["frames"], 2):
                raise ValueError("Audio mix failed its sample-count verification")
            peak = 0.0
            for j in range(0, len(audio), rate):
                if cancelled():
                    raise InterruptedError("Audio mix cancelled")
                block = audio[j : j + rate]
                if not np.isfinite(block).all():
                    raise ValueError("Audio mix contains non-finite samples")
                peak = max(peak, float(np.max(np.abs(block))))
            if plan["purpose"] == "driver" and peak < 1e-7:
                raise ValueError("The selected audio driver is silent")
            for source, sha in source_hashes.items():
                if digest_file(source) != sha:
                    raise ValueError("An audio source changed while preparing the mix; retry")
            result = dict(
                path=str(target),
                frames=plan["frames"],
                sample_rate=rate,
                channels=2,
                mix_key=key,
                peak=peak,
                duration=plan["frames"] / rate,
                input_receipt=identity,
                applied_policy=plan["policy"],
            )
            pending_receipt = temp / "mix.json"
            pending_receipt.write_text(
                json.dumps(
                    dict(identity=identity, sha256=digest_file(pending), result=result), indent=2
                )
            )
            del audio, mixed, voice
            os.replace(pending, target)
            os.replace(pending_receipt, receipt)
            return result


def envelope(item, begin, end, rate):
    import numpy as np

    t = np.arange(begin, end, dtype=np.float64) / rate + item.get("envelope_offset", 0)
    a = np.ones(end - begin)
    if item["fade_in"] > 0:
        a *= (
            np.clip(t / item["fade_in"], 0, 1)
            if item["curve"] == "linear"
            else np.sin(np.clip(t / item["fade_in"], 0, 1) * np.pi / 2)
        )
    if item["fade_out"] > 0:
        a *= (
            np.clip((item.get("envelope_duration", item["duration"]) - t) / item["fade_out"], 0, 1)
            if item["curve"] == "linear"
            else np.sin(
                np.clip(
                    (item.get("envelope_duration", item["duration"]) - t) / item["fade_out"], 0, 1
                )
                * np.pi
                / 2
            )
        )
    return a
