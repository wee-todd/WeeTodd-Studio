"""Local audio visualization, caption inference and presentation export adapters."""

import copy
import json
import os
import signal
import subprocess
import tempfile
import uuid
from pathlib import Path


def features(request, output):
    import numpy as np
    from scipy.io import wavfile

    from wee_todd_mlx.audio_mix import compile_mix, render_mix
    from wee_todd_mlx.audio_presentation import audio_features, presentation_duration

    project = copy.deepcopy(request["project"])
    duration = presentation_duration(project)
    source = (project.get("audioPresentation") or {}).get("source", "mix")
    if source not in {"mix", "voice", "music", "voiceAndMusic"}:
        raise ValueError("Choose a supported visualizer audio source")
    if source != "mix":
        roles = {"voice", "music"} if source == "voiceAndMusic" else {source}
        for track in project.get("audioTracks", []):
            if track.get("role", "music" if track.get("name") == "Music" else "other") not in roles:
                track["muted"] = True
        for clip in project.get("clips", []):
            clip["volume"] = 0
    plan = compile_mix(project, duration=duration, purpose=request.get("purpose", "preview"))
    mixed = render_mix(plan, output / "Mix", request.get("runtime", {}))
    target = output / (mixed["mix_key"] + "-visual-v1.json")
    if not target.is_file():
        rate, audio = wavfile.read(mixed["path"])
        if audio.dtype.kind in "iu":
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        result = audio_features(audio, rate)
        temporary = target.with_suffix("." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False))
        temporary.replace(target)
    return dict(path=str(target), duration=duration)


def captions(request, output):
    import numpy as np
    from studio_bridge import executable, inspect_media, run

    from wee_todd_mlx.audio_analysis.alignment import align_lyrics
    from wee_todd_mlx.audio_analysis.ctc_model import ctc_emissions
    from wee_todd_mlx.audio_presentation import caption_cues
    from wee_todd_mlx.speech_reference import digest_file

    source = Path(request["path"])
    info = inspect_media(source, request.get("runtime", {}))
    duration = info["duration"]
    if not 0 < duration <= 3600:
        raise ValueError("Caption audio must be within one hour")
    root = Path(
        request.get("modelDirectory")
        or request.get("runtime", {}).get("captionModelDirectory")
        or ""
    )
    if (root / "wav2vec2-base-960h").is_dir():
        root = root / "wav2vec2-base-960h"
    if not (root / "model.safetensors").is_file():
        raise ValueError("Set up the local English caption model in Audio visuals & captions.")
    stamp = digest_file(source)
    output.mkdir(parents=True, exist_ok=True)
    pcm = output / "caption-input.f32"
    run(
        [
            executable("ffmpeg", request.get("runtime", {})),
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            str(pcm),
        ]
    )
    try:
        samples = np.fromfile(pcm, dtype=np.float32)
        evidence = ctc_emissions(samples, root)
        aligned = align_lyrics(
            evidence["log_probs"],
            evidence["vocabulary"],
            request.get("text", ""),
            frame_seconds=evidence["frame_seconds"],
            offset_seconds=evidence["offset_seconds"],
        )
        if digest_file(source) != stamp:
            raise ValueError("Audio changed during caption generation. Generate again.")
        cues = caption_cues(aligned["words"], duration)
        return dict(
            cues=cues,
            sourceSHA256=stamp,
            warnings=aligned["limitations"],
            recognizedText=aligned["recognizedText"],
        )
    finally:
        pcm.unlink(missing_ok=True)


def export_audio(request, destination):
    from studio_bridge import executable, inspect_media, run

    from wee_todd_mlx.audio_mix import compile_mix, render_mix
    from wee_todd_mlx.audio_presentation import presentation_duration

    if destination.exists():
        raise ValueError("Choose a new export filename")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".audio-export-", dir=destination.parent) as folder:
        work = Path(folder)
        mixed = render_mix(
            compile_mix(
                request["project"],
                duration=presentation_duration(request["project"]),
                purpose="export",
            ),
            work / "mix",
            request["runtime"],
        )
        target = work / "audio.wav"
        run(
            [
                executable("ffmpeg", request["runtime"]),
                "-v",
                "error",
                "-nostdin",
                "-i",
                mixed["path"],
                "-c:a",
                "pcm_s24le",
                str(target),
            ]
        )
        info = inspect_media(target, request["runtime"])
        if abs(info["duration"] - mixed["duration"]) > 0.02:
            raise RuntimeError("Audio duration verification failed")
        os.replace(target, destination)
    return dict(audio=str(destination), duration=mixed["duration"])


def render_frames(project, feature, base, target, runtime, geometry, work):
    from studio_bridge import executable, write_json

    from wee_todd_mlx.audio_presentation import presentation_duration

    helper = runtime.get("presentationHelperPath") or str(
        Path(runtime.get("metalPath", "")).with_name("WeeToddCLI")
    )
    width, height, fps = geometry["outputWidth"], geometry["outputHeight"], geometry["outputFPS"]
    project_file = work / (uuid.uuid4().hex + ".json")
    write_json(project_file, project)
    command = [
        executable("ffmpeg", runtime),
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(base),
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgra",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-filter_complex",
        "[0:v][1:v]overlay=shortest=1:format=auto:alpha=premultiplied[v]",
        "-map",
        "[v]",
        "-map",
        "0:a",
        "-t",
        str(presentation_duration(project)),
    ]
    if geometry.get("format") == "proRes":
        command += [
            "-c:v",
            "prores_ks",
            "-profile:v",
            "3",
            "-pix_fmt",
            "yuv422p10le",
            "-c:a",
            "pcm_s24le",
        ]
    else:
        command += [
            "-c:v",
            "libx264",
            "-crf",
            str(geometry.get("quality", 18)),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
        ]
    encoder = subprocess.Popen(command + [str(target)], stdin=subprocess.PIPE)
    renderer = None
    try:
        renderer = subprocess.Popen(
            [
                helper,
                "--audio-presentation",
                str(project_file),
                feature["path"] if feature else "-",
                str(width),
                str(height),
                str(fps),
            ],
            stdout=encoder.stdin,
        )
        encoder.stdin.close()
        if renderer.wait() != 0:
            raise RuntimeError("Presentation frame rendering failed")
        if encoder.wait() != 0:
            raise RuntimeError("Presentation encoding failed")
    finally:
        for child in [renderer, encoder]:
            if child and child.poll() is None:
                child.send_signal(signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def export_presentation(request, destination, cache_directory=None):
    from studio_bridge import (
        executable,
        export_movie_base,
        inspect_media,
        resolved_settings,
        run,
        write_json,
    )

    from wee_todd_mlx.audio_presentation import presentation_duration

    project = request["project"]
    settings = project.get("audioPresentation")
    if not settings:
        return export_movie_base(request, destination, cache_directory=cache_directory)
    if destination.exists():
        raise ValueError("Choose a new export filename")
    duration = presentation_duration(project)
    runtime = request["runtime"]
    helper = runtime.get("presentationHelperPath") or str(
        Path(runtime.get("metalPath", "")).with_name("WeeToddCLI")
    )
    if not Path(helper).is_file():
        raise ValueError("Update the Studio app to enable presentation export")
    geometry = resolved_settings(project, {})
    width, height, fps = geometry["outputWidth"], geometry["outputHeight"], geometry["outputFPS"]
    if not 1 <= fps <= 240:
        raise ValueError("Presentation frame rate must be 1–240 fps")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".presentation-", dir=destination.parent) as folder:
        work = Path(folder)
        feature = (
            features(dict(request, purpose="export"), work / "Features")
            if settings.get("enabled")
            else None
        )
        p = copy.deepcopy(project)
        p.pop("audioPresentation", None)
        # Titles/captions are drawn after assembly by the same renderer as the live preview.
        p["titles"] = []
        cursor = 0
        clips = p.setdefault("clips", [])
        for i, c in enumerate(clips):
            overlap = (
                0
                if i == 0 or c.get("transition", "cut") == "cut"
                else min(
                    c.get("transitionDuration", 0.5),
                    c["duration"] / 2,
                    clips[i - 1]["duration"] / 2,
                )
            )
            cursor += c["duration"] - overlap
        if duration > cursor + 0.00001:
            clips.append(
                dict(
                    id=str(uuid.uuid4()),
                    name="Audio canvas",
                    engine="movie",
                    duration=duration - cursor,
                    sourceIn=0,
                    sourcePath="",
                    transition="cut",
                    volume=0,
                )
            )
        blank = work / "canvas.mp4"
        if any(not c.get("sourcePath") for c in clips):
            run(
                [
                    executable("ffmpeg", runtime),
                    "-v",
                    "error",
                    "-nostdin",
                    "-f",
                    "lavfi",
                    "-i",
                    f"color=c=black:s={width}x{height}:r={fps}",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=48000:cl=stereo",
                    "-t",
                    str(duration),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    str(blank),
                ]
            )
            if settings.get("enabled"):
                background = copy.deepcopy(project)
                background["titles"] = [dict(title, text="") for title in project.get("titles", [])]
                background["audioPresentation"].update(voiceCaptions=False, lyrics=False)
                for c in background.get("clips", []):
                    c["sourcePath"] = ""
                    c["versions"] = []
                visual = work / "visual.mp4"
                render_frames(
                    background, feature, blank, visual, runtime, dict(geometry, format="mp4"), work
                )
                blank = visual
            cursor = 0
            for i, c in enumerate(clips):
                overlap = (
                    0
                    if i == 0 or c.get("transition", "cut") == "cut"
                    else min(
                        c.get("transitionDuration", 0.5),
                        c["duration"] / 2,
                        clips[i - 1]["duration"] / 2,
                    )
                )
                cursor -= overlap
                if not c.get("sourcePath"):
                    c.update(sourcePath=str(blank), sourceIn=cursor, engine="movie", volume=0)
                cursor += c["duration"]
        # Placing the visualization inside empty clips preserves their actual video transitions.
        fmt = geometry.get("format", "mp4")
        p["settings"] = dict(p["settings"], format="proRes" if fmt == "proRes" else "mp4")
        base = work / ("base.mov" if fmt == "proRes" else "base.mp4")
        export_movie_base(dict(request, project=p), base, cache_directory=cache_directory)
        overlay = copy.deepcopy(project)
        overlay["audioPresentation"]["enabled"] = False
        final = work / ("final.mov" if fmt in {"mov", "proRes"} else "final.mp4")
        render_frames(overlay, None, base, final, runtime, geometry, work)
        info = inspect_media(final, runtime)
        if (
            abs(info["duration"] - duration) > max(0.1, 2 / fps)
            or (info["width"], info["height"]) != (width, height)
            or not info["hasAudio"]
        ):
            raise RuntimeError("Presentation media verification failed")
        if fmt == "pngSequence":
            sequence = work / "sequence"
            sequence.mkdir()
            run(
                [
                    executable("ffmpeg", runtime),
                    "-v",
                    "error",
                    "-i",
                    str(final),
                    str(sequence / "frame-%06d.png"),
                ]
            )
            run(
                [
                    executable("ffmpeg", runtime),
                    "-v",
                    "error",
                    "-i",
                    str(final),
                    "-vn",
                    "-c:a",
                    "pcm_s24le",
                    str(sequence / "audio.wav"),
                ]
            )
            write_json(
                sequence / "manifest.json",
                dict(fps=fps, duration=duration, width=width, height=height),
            )
            os.replace(sequence, destination)
        else:
            os.replace(final, destination)
        write_json(
            destination.with_name(destination.name + ".json"),
            dict(format="weetodd-export-v1", project=project, media=info),
        )
    return dict(video=str(destination), duration=duration, width=width, height=height, fps=fps)
