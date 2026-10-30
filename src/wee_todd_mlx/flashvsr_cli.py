"""Model-free handoff to Studio's backend-neutral standalone upscaler."""
from __future__ import annotations

import hashlib
import json
import os
import select
import subprocess
import tempfile
from pathlib import Path


def run_flashvsr_cli(*, cli, source, output, model_root, vpipe, ffmpeg, ffprobe,
                     width=1280, height=720, decoder="tiny", seed=0, keep_audio=True,
                     temporal_compression=4, color_fix="wavelet", backend="vpipe", variant="turbo",
                     swift_worker="",
                     on_progress=lambda event: None, check_interrupted=lambda: None):
    if (type(seed) is not int or not 0 <= seed <= 4294967295
            or any(type(n) is not int or not 64 <= n <= 8192 or n % 2 for n in (width, height))
            or decoder not in {"tiny", "vae"} or type(keep_audio) is not bool
            or type(temporal_compression) is not int or temporal_compression not in {1, 2, 4}
            or color_fix not in {"wavelet", "adain", "off"}
            or backend not in {"swiftMLX", "vpipe"} or variant not in {"turbo", "base"}
            or (backend == "vpipe" and variant != "turbo")):
        raise ValueError("Invalid FlashVSR+ seed, dimensions, decoder or processing settings")
    cli, source, output, model_root, ffmpeg, ffprobe = (
        Path(p).expanduser().resolve()
        for p in (cli, source, output, model_root, ffmpeg, ffprobe)
    )
    tools = [cli, ffmpeg, ffprobe]
    vpipe_path = ""
    if backend == "vpipe":
        if not str(vpipe).strip():
            raise FileNotFoundError("Connect the vpipe executable recorded by this legacy job")
        vpipe_path = str(Path(vpipe).expanduser().resolve())
        tools.append(Path(vpipe_path))
    if swift_worker:
        tools.append(Path(swift_worker).expanduser().resolve())
    for tool in tools:
        if not tool.is_file() or not os.access(tool, os.X_OK):
            raise FileNotFoundError(f"Connect the selected local executable: {tool.name}")
    if not source.is_file() or not model_root.is_dir():
        raise FileNotFoundError("Select an existing source movie and shared model folder")
    if output.exists():
        raise FileExistsError("Choose a new upscale output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    request = {
        "format": "weetodd-video-upscale-v1", "sourcePath": str(source),
        "sourceSHA256": digest, "firstFrame": 0,
        "settings": {"version": 1, "backend": backend, "variant": variant, "decoder": decoder,
                     "width": width, "height": height, "fit": "fit", "seed": seed,
                     "temporalCompression": temporal_compression, "temporalSigma": 1,
                     "colorFix": color_fix, "waveletLevels": 5, "continuity": "stream",
                     "preserveAudio": keep_audio},
        "runtime": {"vpipePath": vpipe_path, "modelRoot": str(model_root),
                    "supervisorPath": str(cli), "ffmpegPath": str(ffmpeg),
                    "ffprobePath": str(ffprobe), "leasePath": "",
                    **({"swiftWorkerPath": str(Path(swift_worker).expanduser().resolve())}
                       if swift_worker else {})},
    }
    return _run_cli_request(cli=cli, output=output, request=request, command="upscale",
        on_progress=on_progress, check_interrupted=check_interrupted)


def _run_cli_request(*, cli, output, request, command, on_progress, check_interrupted,
                     preflight_only=False):
    with tempfile.TemporaryDirectory(prefix="weetodd-upscale-", dir=output.parent) as scratch:
        filename = Path(scratch) / "request.json"
        filename.write_text(json.dumps(request))
        process = subprocess.Popen(
            [str(cli), command, "--request", str(filename), "--output-directory", str(output)]
            + (["--preflight-only"] if preflight_only else []),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        terminal, pending = None, b""
        try:
            while True:
                check_interrupted()
                ready, _, _ = select.select([process.stdout], [], [], 0.1)
                if not ready:
                    if process.poll() is not None:
                        break
                    continue
                chunk = os.read(process.stdout.fileno(), 16384)
                if not chunk:
                    break
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if len(line) > 8 * 1024 * 1024:
                        raise ValueError("Upscaler output exceeds its 8 MiB receipt bound")
                    try:
                        event = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError) as error:
                        raise RuntimeError(line[-1000:].decode(errors="replace")) from error
                    if not isinstance(event, dict):
                        raise RuntimeError("Invalid upscaler event")
                    if terminal is not None:
                        raise RuntimeError("Upscaler output continued after completion")
                    if "status" in event:
                        if terminal is not None:
                            raise RuntimeError("Duplicate upscaler completion")
                        terminal = event
                    else:
                        if len(line) > 65536:
                            raise RuntimeError("Oversized upscaler progress event")
                        on_progress(event)
                if len(pending) > 8 * 1024 * 1024:
                    raise RuntimeError("Oversized upscaler event")
            process.wait(timeout=5)
            if (pending.strip() or process.returncode or not terminal
                    or terminal.get("status") != "success"):
                raise RuntimeError("The upscaler failed or returned no verified completion")
            if preflight_only:
                if (terminal.get("preflight") is not True
                        or not isinstance(terminal.get("plan"), dict)):
                    raise RuntimeError("The upscaler returned no admitted preflight plan")
                if output.exists():
                    raise RuntimeError(
                        "The upscaler preflight unexpectedly published an output directory"
                    )
                return terminal["plan"]
            result = terminal["result"]
            video = result["video"]
            artifact = Path(video["path"]).resolve()
            if output not in artifact.parents or not artifact.is_file():
                raise RuntimeError("The upscaler result escaped its output directory")
            with artifact.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != video["sha256"]:
                    raise RuntimeError("The upscaler artifact changed after verification")
            return result
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()


def validate_h3_regenerate_settings(*, width, height, decoder, seed, prompt):
    if (type(seed) is not int or not 0 <= seed <= 4294967295
            or any(type(n) is not int or not 32 <= n <= 4096 or n % 2 for n in (width, height))
            or decoder not in {"releasedH3", "fastH3"}
            or not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode()) > 100000):
        raise ValueError("Invalid H3 source, target, fixed decoder profile or preservation prompt")


def run_h3_regenerate_cli(*, cli, source, output, profile, model_library, ffmpeg, ffprobe,
                        width=2560, height=1440, decoder="releasedH3", seed=0,
                        prompt="Preserve the source subjects, composition and motion.",
                        swift_worker="", on_progress=lambda event: None,
                        check_interrupted=lambda: None, preflight_only=False):
    validate_h3_regenerate_settings(
        width=width, height=height, decoder=decoder, seed=seed, prompt=prompt
    )
    if type(preflight_only) is not bool:
        raise ValueError("H3 preflight_only must be a boolean")
    cli, source, output, profile, library, ffmpeg, ffprobe = (
        Path(p).expanduser().resolve()
        for p in (cli, source, output, profile, model_library, ffmpeg, ffprobe)
    )
    tools = [cli, ffmpeg, ffprobe]
    if swift_worker:
        tools.append(Path(swift_worker).expanduser().resolve())
    for tool in tools:
        if not tool.is_file() or not os.access(tool, os.X_OK):
            raise FileNotFoundError(f"Connect the selected native executable: {tool.name}")
    if not source.is_file() or not profile.is_file() or not library.is_dir():
        raise FileNotFoundError(
            "Connect the source movie, ordinary Ref2VA profile and shared model library"
        )
    if output.exists():
        raise FileExistsError("Choose a new restoration output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    request = {
        "format": "weetodd-h3-regenerate-v1", "sourcePath": str(source),
        "sourceSHA256": digest, "firstFrame": 0,
        "settings": {
            "version": 1, "targetWidth": width, "targetHeight": height, "decoder": decoder,
            "encoderProfile": "publisherPosteriorFP32", "schedule": "publisherThreeStep",
            "sourceMode": "movie", "fit": "fit", "seed": seed, "posteriorSeed": 0,
            "latentUpscaleMode": "directPublisher", "baseProfilePath": str(profile),
            "modelLibraryRoot": str(library), "publisherEncoderPath": "", "fastDecoderPath": "",
            "prompt": prompt,
        },
        "runtime": {"workerPath": str(Path(swift_worker).expanduser().resolve())
                    if swift_worker else "",
                    "ffmpegPath": str(ffmpeg), "ffprobePath": str(ffprobe)},
    }
    return _run_cli_request(cli=cli, output=output, request=request, command="regenerate",
        on_progress=on_progress, check_interrupted=check_interrupted, preflight_only=preflight_only)
