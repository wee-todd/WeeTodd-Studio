"""Resumable movie production using Studio's existing render and assembly adapters."""

from __future__ import annotations

import copy
import fcntl
import json
import math
import time
import uuid
from pathlib import Path

import studio_bridge as bridge
from studio_job import artifact_hash, atomic_json, digest, inputs_fingerprint, renderer_fingerprint

from wee_todd_mlx.studio_scene import scene_members

FORMAT = "weetodd-production-v1"


def _read(path):
    if path.stat().st_size > 64_000_000:
        raise ValueError("Production record exceeds its size limit.")
    return json.loads(path.read_text())


def _inputs(request):
    # Profiles are a runtime input, including their referenced weight paths.
    profiles = {}
    directory = request.get("runtime", {}).get("profilesDirectory")
    if directory:
        for filename in sorted(Path(directory).glob("*.json")):
            profiles[str(filename)] = _read(filename)
    return inputs_fingerprint(dict(request, recipes=profiles))


def _units(request):
    clips = request["project"]["clips"]
    ids = [c["id"] for c in clips]
    if not clips or len(set(ids)) != len(ids) or len(ids) > 1000:
        raise ValueError("Production needs 1–1000 uniquely identified clips.")
    generating = set(request.get("generateIDs", []))
    if not generating.issubset(ids):
        raise ValueError("A queued shot no longer exists.")
    result, seen = [], set()
    for clip in clips:
        if clip["id"] in seen:
            continue
        members = scene_members(dict(request, clipID=clip["id"])) or [clip]
        member_ids = [c["id"] for c in members]
        generate = bool(generating.intersection(member_ids))
        if generate and clip["engine"] == "movie":
            raise ValueError("Imported movies cannot be generated.")
        if generate and clip["engine"] == "drawThings" and request.get("allowRemote") is not True:
            raise ValueError(
                "Enable Draw Things execution explicitly, or generate its shots manually first."
            )
        if generate and clip["engine"] == "drawThings":
            profile_id = (clip.get("drawThings") or {}).get("profileID")
            connection = next(
                (c for c in request.get("drawThingsConnections", []) if c.get("id") == profile_id),
                None,
            )
            if not connection or connection.get("selfHostedConfirmed") is not True:
                raise ValueError(
                    "Generate Cloud API shots with their individual cost confirmation "
                    "before production, or select a confirmed self-hosted connection."
                )
        seen.update(member_ids)
        result.append(
            dict(
                id=member_ids[0],
                name=clip["name"],
                clipIDs=member_ids,
                generate=generate,
                engine=clip["engine"],
                status="pending",
                attempts=0,
                priorTakeIDs={c["id"]: str(uuid.uuid4()) for c in members},
            )
        )
    return result


def create(request, directory):
    directory = Path(directory).expanduser().resolve()
    request = copy.deepcopy(request)
    maximum = request.get("maxRetries", 1)
    if type(maximum) is not int or not 0 <= maximum <= 3:
        raise ValueError("Choose 0–3 automatic local retries.")
    request["maxRetries"] = maximum
    units = _units(request)
    reused_ids = {clip_id for unit in units if not unit["generate"] for clip_id in unit["clipIDs"]}
    for clip in request["project"]["clips"]:
        duration = clip.get("duration")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError("All shots need a positive finite duration.")
        bridge.preflight_finishing(request["project"], clip, request["runtime"])
        if clip["id"] in reused_ids:
            media = bridge.inspect_media(clip["sourcePath"], request["runtime"])
            if (
                media["kind"] != "image"
                and clip.get("sourceIn", 0) + duration > media["duration"] + 0.04
            ):
                raise ValueError("An existing take is shorter than its planned shot.")
    for audio in request["project"].get("audio", []):
        media = bridge.inspect_media(audio["path"], request["runtime"])
        values = [audio.get("sourceIn", 0), audio.get("duration", 0), audio.get("start", 0)]
        if (
            not all(
                isinstance(v, (float, int))
                and not isinstance(v, bool)
                and math.isfinite(v)
                and v >= 0
                for v in values
            )
            or values[1] <= 0
            or not media.get("hasAudio")
            or values[0] + values[1] > media["duration"] + 0.04
        ):
            raise ValueError("An audio region has no audio or extends beyond its source.")
    record = dict(
        format=FORMAT, request=request, inputs=_inputs(request), renderer=renderer_fingerprint()
    )
    record["sha256"] = digest(record)
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / "request.json", record)
    atomic_json(
        directory / "state.json",
        dict(format=FORMAT, manifestSHA256=record["sha256"], status="ready", units=units),
    )
    return status(directory)


def _load(directory):
    record = _read(directory / "request.json")
    expected = record.pop("sha256", None)
    if record.get("format") != FORMAT or digest(record) != expected:
        raise ValueError("The production manifest changed; create a new production.")
    state = _read(directory / "state.json")
    if state.get("manifestSHA256") != expected or state.get("format") != FORMAT:
        raise ValueError("The production state does not match its manifest.")
    return record, state


def status(directory):
    directory = Path(directory).expanduser().resolve()
    record, state = _load(directory)
    result = dict(state, jobDirectory=str(directory), projectID=record["request"]["project"]["id"])
    if state.get("status") == "completed":
        result["resolvedProject"] = _read(directory / "resolved-project.json")
    return result


def _verify_result(unit):
    result = unit.get("result", {})
    path = result.get("video")
    if not path or not Path(path).is_file() or artifact_hash(path) != result.get("sha256"):
        raise ValueError(
            "A completed take is missing or changed. "
            "Start a new production; old outputs are preserved."
        )
    artifact = result.get("continuation_artifact")
    if artifact and unit.get("continuationHash") != digest(artifact):
        raise ValueError("Saved motion continuity changed. Start a new production.")


def _apply(project, unit):
    result, prepared = unit["result"], unit.get("prepared", {})
    scene = result.get("scene")
    ranges = (
        {m["clip_id"]: (m["source_in"], m["duration"]) for m in scene["members"]} if scene else {}
    )
    for clip in project["clips"]:
        if clip["id"] not in unit["clipIDs"]:
            continue
        old_source = clip.get("sourcePath")
        versions = clip.setdefault("versions", [])
        if old_source and not any(v["path"] == old_source for v in versions):
            versions.append(
                dict(
                    id=unit["priorTakeIDs"][clip["id"]],
                    created=unit["created"],
                    path=old_source,
                    seed=clip.get("seed", 42),
                    prompt=clip.get("prompt", ""),
                    recipePath="",
                    usableSourceIn=clip.get("sourceIn", 0),
                    usableDuration=clip["duration"],
                )
            )
        start, duration = ranges.get(clip["id"], (result["usable_source_in"], clip["duration"]))
        clip.update(
            sourcePath=result["video"],
            sourceIn=start,
            duration=duration,
            renderedSignature="",
            validatedSignature="",
        )
        version = dict(
            id=unit["takeIDs"][clip["id"]],
            created=unit["created"],
            path=result["video"],
            seed=clip.get("seed", 42),
            prompt=clip.get("prompt", ""),
            recipePath=prepared.get("recipePath", ""),
            usableSourceIn=start,
            usableDuration=duration,
        )
        report = prepared.get("report", {})
        if report.get("generation"):
            version["generationSettings"] = report["generation"]
        if report.get("resolvedFingerprint"):
            version["resolvedFingerprint"] = report["resolvedFingerprint"]
        if result.get("continuation_artifact"):
            version["continuationArtifact"] = result["continuation_artifact"]
        if scene:
            version.update(
                sceneMembers=copy.deepcopy(scene["members"]),
                sceneTakeID=unit["sceneTakeID"],
                sceneFrameRate=scene["frame_rate"],
                sceneInputFingerprint="",
            )
        clip.setdefault("versions", []).append(version)


def _validated_result(result, prepared, unit, project, runtime):
    path = result.get("video")
    info = bridge.inspect_media(path, runtime)
    if info.get("kind") != "video" or not math.isfinite(info.get("duration", 0)):
        raise ValueError("Renderer returned no valid movie.")
    first = next(c for c in project["clips"] if c["id"] == unit["id"])
    expected_scene = prepared.get("report", {}).get("scene")
    if len(unit["clipIDs"]) > 1:
        if not expected_scene or result.get("scene") != expected_scene:
            raise ValueError("Scene result does not match the complete prepared member ranges.")
        if [m["clip_id"] for m in expected_scene["members"]] != unit["clipIDs"]:
            raise ValueError("Prepared scene changed member identities.")
        duration = sum(m["duration"] for m in expected_scene["members"])
        if abs(info["duration"] - duration) > 1 / expected_scene["frame_rate"]:
            raise ValueError("Scene movie duration does not cover the planned timing.")
        start = 0.0
    else:
        if result.get("scene"):
            raise ValueError("A single shot returned an unexpected scene.")
        start = result.get("usable_source_in", 0.0)
        available = result.get("usable_duration", info["duration"] - start)
        if first.get("extensionSource") and "usable_source_in" not in result:
            context = bridge.inspect_media(first["extensionSource"], runtime)["duration"]
            available -= context
            if first.get("extensionDirection") == "after":
                start = context
        duration = first["duration"]
        tolerance = 1 / project["settings"].get("fps", 24)
        if (
            not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in (start, available)
            )
            or start < 0
            or available + tolerance < duration
            or start + duration > info["duration"] + tolerance
        ):
            raise ValueError(
                "Generated take is shorter than its planned shot; timing was preserved."
            )
    return dict(
        result, sha256=artifact_hash(path), usable_source_in=start, usable_duration=duration
    )


def _validate_prepared_timing(prepared, unit, project):
    scene = prepared.get("report", {}).get("scene")
    if len(unit["clipIDs"]) <= 1:
        return
    if not scene or [m["clip_id"] for m in scene["members"]] != unit["clipIDs"]:
        raise ValueError("Scene preparation did not preserve the planned members.")
    clips = {c["id"]: c for c in project["clips"]}
    for member in scene["members"]:
        if abs(member["duration"] - clips[member["clip_id"]]["duration"]) > 0.000001:
            raise ValueError(
                "Continuous-scene frame constraints would change shot timing. "
                "Adjust the shots to supported scene lengths before production."
            )


def run(directory):
    directory = Path(directory).expanduser().resolve()
    with (directory / ".production.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("This production is already running.") from error
        return _run_locked(directory)


def _run_locked(directory):
    record, state = _load(directory)
    request = record["request"]
    if record["inputs"] != _inputs(request) or record["renderer"] != renderer_fingerprint():
        raise ValueError(
            "Source inputs or renderer changed. "
            "Start a new production; existing outputs are preserved."
        )
    expected = _units(request)
    if [(u["id"], u["clipIDs"], u["generate"]) for u in state["units"]] != [
        (u["id"], u["clipIDs"], u["generate"]) for u in expected
    ]:
        raise ValueError("Production unit manifest changed.")
    for unit in state["units"]:
        if unit["status"] == "completed" and unit["generate"]:
            _verify_result(unit)
    project = copy.deepcopy(request["project"])
    state["status"] = "running"
    state.pop("error", None)

    def save():
        atomic_json(directory / "state.json", state)

    save()
    try:
        for index, unit in enumerate(state["units"]):
            if not unit["generate"]:
                for clip in project["clips"]:
                    if clip["id"] in unit["clipIDs"]:
                        info = bridge.inspect_media(clip["sourcePath"], request["runtime"])
                        if (
                            info["kind"] != "image"
                            and clip.get("sourceIn", 0) + clip["duration"] > info["duration"] + 0.04
                        ):
                            raise ValueError("An existing take is shorter than its planned shot.")
                unit["status"] = "completed"
                save()
                continue
            if unit["status"] == "completed":
                _apply(project, unit)
                continue
            if unit.get("remoteSubmitted"):
                raise ValueError(
                    "A prior Draw Things submission has no verified take. "
                    "Review it manually; no automatic resubmission."
                )
            remote = unit["engine"] == "drawThings"
            for retry in range(1 + (0 if remote else request["maxRetries"])):
                unit["attempts"] += 1
                unit["status"] = "running"
                unit.pop("error", None)
                save()
                bridge.emit(
                    event="progress",
                    message=f"Rendering {index + 1}/{len(state['units'])}: {unit['name']}",
                )
                folder = directory / "takes" / uuid.uuid4().hex
                folder.mkdir(parents=True)
                body = dict(request, project=project, clipID=unit["id"])
                try:
                    if remote:
                        from studio_drawthings import dispatch

                        selection = next(c for c in project["clips"] if c["id"] == unit["id"])[
                            "drawThings"
                        ]
                        connection = next(
                            (
                                c
                                for c in request.get("drawThingsConnections", [])
                                if c["id"] == selection["profileID"]
                            ),
                            None,
                        )
                        if connection is None:
                            raise ValueError("The selected Draw Things connection is unavailable.")
                        body["connection"] = connection
                        estimate = dispatch("dt-prepare-clip", body, None)
                        if estimate.get("eligibility") != "allowed":
                            raise ValueError(
                                "Draw Things preflight did not authorize this request."
                            )
                        # Cloud execution stays in the existing individual-shot confirmation flow.
                        if connection.get("selfHostedConfirmed") is not True:
                            raise ValueError(
                                "Generate Cloud API shots manually with their cost confirmation, "
                                "then start production."
                            )
                        prepared = {}
                        unit["remoteSubmitted"] = True
                        save()
                        result = dispatch(
                            "dt-generate-clip",
                            body,
                            folder / "render",
                            progress=lambda e: bridge.emit(event="progress", message=str(e)),
                        )
                    else:
                        prepared = bridge.prepare(body, folder / "prepared")
                        _validate_prepared_timing(prepared, unit, project)
                        unit["prepared"] = prepared
                        save()
                        options = (
                            dict(
                                checkpoint_directory=directory / "checkpoints" / digest(unit["id"])
                            )
                            if len(unit["clipIDs"]) > 1
                            else {}
                        )
                        result = bridge.render(prepared["recipePath"], folder / "render", **options)
                    result = _validated_result(result, prepared, unit, project, request["runtime"])
                    unit.update(
                        result=result,
                        prepared=prepared,
                        status="completed",
                        created=time.time() - 978307200.0,
                        sceneTakeID=str(uuid.uuid4()),
                        takeIDs={clip_id: str(uuid.uuid4()) for clip_id in unit["clipIDs"]},
                    )
                    if result.get("continuation_artifact"):
                        unit["continuationHash"] = digest(result["continuation_artifact"])
                    save()
                    _apply(project, unit)
                    break
                except InterruptedError:
                    raise
                except Exception as error:
                    unit["error"] = str(error)
                    unit["status"] = "failed"
                    save()
                    if isinstance(error, (ValueError, TypeError, FileNotFoundError)) or retry == (
                        0 if remote else request["maxRetries"]
                    ):
                        raise
        if record["inputs"] != _inputs(request):
            raise ValueError(
                "Source inputs changed during production; generated takes were retained."
            )
        format_name = project["settings"].get("format", "mp4")
        filename = (
            "movie-frames"
            if format_name == "pngSequence"
            else "movie.mov"
            if format_name in {"mov", "proRes"}
            else "movie.mp4"
        )
        final = directory / filename
        previous = state.get("outputSHA256")
        pending = state.get("pendingExport")
        if not previous and pending:
            pending_path = Path(pending["path"])
            candidate = final if final.exists() else pending_path
            if artifact_hash(candidate) != pending["sha256"]:
                raise ValueError("The pending assembly changed; start a new production.")
            if candidate != final:
                candidate.rename(final)
            previous = state["outputSHA256"] = pending["sha256"]
        if previous:
            if artifact_hash(final) != previous:
                raise ValueError(
                    "The assembled movie is missing or changed; start a new production."
                )
        else:
            # Each attempt has its own output so interrupted exports cannot block resume.
            attempt = directory / "assembly" / uuid.uuid4().hex
            attempt.mkdir(parents=True)
            bridge.export_movie(
                dict(project=project, runtime=request["runtime"]),
                attempt / filename,
                cache_directory=directory / "finished-clips",
            )
            if record["inputs"] != _inputs(request):
                raise ValueError(
                    "Source inputs changed during assembly; "
                    "output remains in the production folder."
                )
            if final.exists():
                raise ValueError("An unverified movie already exists; preserve it before resuming.")
            output_hash = artifact_hash(attempt / filename)
            if not output_hash:
                raise ValueError("Assembly returned no output.")
            state["pendingExport"] = dict(path=str(attempt / filename), sha256=output_hash)
            save()
            (attempt / filename).rename(final)
            state["outputSHA256"] = output_hash
        atomic_json(directory / "resolved-project.json", project)
        state["resolvedProjectSHA256"] = artifact_hash(directory / "resolved-project.json")
        state.update(status="completed", outputPath=str(final))
        save()
        return status(directory)
    except BaseException as error:
        state["status"] = (
            "cancelled" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed"
        )
        state["error"] = str(error) or "Production paused. Completed takes are retained."
        save()
        raise


def dispatch(command, request, output=None):
    if command == "production-create":
        if output is None:
            raise ValueError("Choose a production folder.")
        result = create(request, output)
        return _summary(result)
    directory = request.get("jobDirectory")
    if not isinstance(directory, str) or not directory:
        raise ValueError("Choose an existing production.")
    if command == "production-run":
        return _summary(run(directory))
    if command == "production-status":
        return _summary(status(directory))
    if command == "production-verify":
        return _summary(verify(Path(directory).expanduser().resolve()))
    raise ValueError("Unknown production action.")


def verify(directory):
    record, state = _load(directory)
    if record["inputs"] != _inputs(record["request"]):
        raise ValueError("Original source inputs changed after production.")
    if state.get("status") != "completed":
        raise ValueError("Production has not completed.")
    for unit in state["units"]:
        if unit["generate"]:
            _verify_result(unit)
    if artifact_hash(directory / "resolved-project.json") != state.get("resolvedProjectSHA256"):
        raise ValueError("The resolved production project changed or is missing.")
    if not state.get("outputPath") or artifact_hash(state["outputPath"]) != state.get(
        "outputSHA256"
    ):
        raise ValueError("The assembled movie changed or is missing.")
    return status(directory)


def _summary(result):
    summary = {
        k: result[k]
        for k in ("status", "jobDirectory", "projectID", "error", "outputPath")
        if k in result
    }
    summary["units"] = [
        {k: unit[k] for k in ("id", "name", "clipIDs", "status", "attempts", "error") if k in unit}
        for unit in result["units"]
    ]
    if result["status"] == "completed":
        summary["resolvedProjectPath"] = str(Path(result["jobDirectory"]) / "resolved-project.json")
    return summary
