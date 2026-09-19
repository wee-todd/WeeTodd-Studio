"""Media-only Studio mix worker. Never imports an inference engine."""

import json
from pathlib import Path

from wee_todd_mlx.audio_mix import compile_mix, render_mix
from wee_todd_mlx.speech_reference import digest_file


def driver_plan(request):
    from studio_music import driver_excerpt_request

    project = request["project"]
    clips = project["clips"]
    clip = next(c for c in clips if c["id"] == request["clipID"])
    if clip.get("engine") not in {"h3", "ltx23", "ltx25"}:
        raise ValueError("Timeline audio driving requires a supported native H3 or LTX model")
    if clip.get("musicSource") or (clip.get("continuity") or {}).get("mode") == "scene":
        raise ValueError(
            "Keep the planned continuous scene audio source, or separate this clip first"
        )
    selection = request.get("selection") or clip.get("audioDriverSelection")
    if not selection:
        raise ValueError("Choose Voice, Music or Voice + Music")
    geometry = driver_excerpt_request(
        dict(video_request=request, runtime=request.get("runtime", {}))
    )
    cursor = 0
    for i, c in enumerate(clips):
        overlap = (
            0
            if i == 0 or c.get("transition", "cut") == "cut"
            else min(
                max(0, c.get("transitionDuration", 0.5)),
                c["duration"] / 2,
                clips[i - 1]["duration"] / 2,
            )
        )
        cursor -= overlap
        if c["id"] == clip["id"]:
            break
        cursor += c["duration"]
    return compile_mix(
        project, start=cursor, duration=geometry["duration"], purpose="driver", selection=selection
    )


def verify_driver(request, clip):
    if not clip.get("audioDriverSelection"):
        return
    key = clip.get("audioDriverMixKey")
    if not key:
        raise ValueError(
            "The timeline audio driver changed. Prepare its current mix before generation."
        )
    attachments = [a for a in clip.get("attachments", []) if a["role"] == "audioDriver"]
    if len(attachments) != 1:
        raise ValueError("Prepare exactly one timeline audio driver")
    assets = {a["id"]: a for a in request["project"].get("assets", [])}
    source = Path(assets.get(attachments[0]["assetID"], {}).get("path", ""))
    receipt = source.parent / "mix.json"
    if not receipt.is_file():
        raise ValueError("The audio driver mix receipt is missing; prepare it again")
    saved = json.loads(receipt.read_text())
    plan = driver_plan(request)
    plan.pop("purpose")
    identity = saved["identity"]
    if key != saved["result"]["mix_key"] or any(identity.get(k) != v for k, v in plan.items()):
        raise ValueError("The audio driver mix is stale; prepare it again")
    if digest_file(source) != saved["sha256"]:
        raise ValueError("The audio driver file changed; prepare it again")
    for name, sha in identity["sources"].items():
        if not Path(name).is_file() or digest_file(name) != sha:
            raise ValueError("A source in the audio driver changed; prepare it again")


def dispatch(command, request, output=None, progress=None, cancelled=None):
    if command not in {"audio-mix", "audio-driver"} or output is None:
        raise ValueError("Choose an audio mix cache folder")
    if progress:
        progress(dict(event="progress", message="Preparing timeline audio"))
    plan = (
        driver_plan(request)
        if command == "audio-driver"
        else compile_mix(
            request["project"],
            start=request.get("start", 0),
            duration=request.get("duration"),
            purpose=request.get("purpose", "preview"),
            selection=request.get("selection"),
        )
    )
    result = render_mix(plan, output, request.get("runtime", {}), cancelled=cancelled)
    if command == "audio-mix" and request.get("disposablePreview") is True:
        from wee_todd_mlx.audio_mix.cache import prune_previews

        prune_previews(output)
    return result
