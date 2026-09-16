"""Opt-in, local H3 synchronized latent tails with verified provenance.

This module performs no model loading. Manifests are committed only after their bounded
SafeTensors payload is durable. Content identity is conservative: all component files and
adapter bytes are hashed, with only a per-call cache for shared files.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import struct
from dataclasses import asdict, replace
from pathlib import Path

FORMAT = "weetodd-h3-continuation-v1"
CONTEXT_FRAMES = (5, 22, 39, 56)
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _hash_file(filename):
    digest = hashlib.sha256()
    with Path(filename).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_hash(value, label):
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def continuation_request(recipe, *, task=None):
    """Validate optional intent and derive native versus published frame geometry."""
    if "continuation" not in recipe:
        return None
    value = recipe["continuation"]
    allowed = {
        "version",
        "context_frames",
        "source_context",
        "source_manifest_sha256",
        "save_context",
        "output_take_id",
    }
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("Unsupported continuation fields; expected a continuation object")
    if recipe.get("engine") != "h3":
        raise ValueError("Native continuation is supported for H3 only")
    if type(value.get("version")) is not int or value["version"] != 1:
        raise ValueError("continuation.version must be 1")
    context = value.get("context_frames")
    if type(context) is not int or context not in CONTEXT_FRAMES:
        raise ValueError(f"continuation.context_frames must be one of {CONTEXT_FRAMES}")
    save = value.get("save_context", False)
    if type(save) is not bool:
        raise ValueError("continuation.save_context must be a boolean")
    source = value.get("source_context")
    digest = value.get("source_manifest_sha256")
    if ("source_context" in value) != ("source_manifest_sha256" in value):
        raise ValueError("source_context and source_manifest_sha256 must be paired")
    if "source_context" in value:
        if (
            not isinstance(source, str)
            or not source
            or "://" in source
            or not Path(source).is_absolute()
        ):
            raise ValueError("continuation.source_context must be an absolute local manifest path")
        _checked_hash(digest, "source_manifest_sha256")
    if not source and not save:
        raise ValueError("continuation must request load or save of a context")
    if "output_take_id" in value and (
        not isinstance(value["output_take_id"], str)
        or not value["output_take_id"].strip()
        or len(value["output_take_id"]) > 256
    ):
        raise ValueError(
            "continuation.output_take_id must be a nonempty string up to 256 characters"
        )
    task = (
        task
        or recipe.get("conditioning", {}).get("task")
        or {
            "t2va": "t2v",
            "fl2va": "fflf",
        }.get(recipe.get("components", {}).get("task", "t2va"))
    )
    if task not in {"t2v", "fflf"}:
        raise ValueError("Experimental native continuation supports t2v and fflf only")
    if any(recipe.get(name) for name in ("attention", "fastvideo", "vdn")):
        raise ValueError("Native continuation with attention approximations or VDN is unqualified")
    if recipe.get("components", {}).get("fun_controlnet") or any(
        "refin" in key.lower() for key in recipe.get("config", {})
    ):
        raise ValueError("Native continuation cannot combine control or refinement")
    duration = recipe.get("config", {}).get("duration_seconds", 5.0)
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or not 2.5 <= duration <= 15
    ):
        raise ValueError("Continuation duration must be a finite value from 2.5 to 15 seconds")
    requested = round(duration * 24)
    overlap = context if source else 0
    generated = requested + overlap
    generated += (5 - generated) % 17
    if generated > 362:
        raise ValueError("Continuation plus new frames exceeds the native H3 15-second window")
    # A save-only request keeps the ordinary H3 output grid. A loaded context describes
    # new visible frames, so only that route needs an editorial tail trim.
    published = requested if source else generated
    tail = generated - overlap - published
    if save and tail:
        nearest = [count for count in range(60, 363 - context) if count % 17 == 0]
        below = max((count for count in nearest if count < requested), default=None)
        above = min((count for count in nearest if count > requested), default=None)
        choices = ", ".join(
            f"{count} frames ({count / 24:.6f} seconds)"
            for count in (below, above)
            if count is not None
        )
        raise ValueError(
            "Cannot save continuation after trimming the generated tail: its state would "
            f"extend beyond the visible movie. Nearest eligible new durations: {choices}"
        )
    return {
        **value,
        "save_context": save,
        "requested_frames": requested,
        "generated_frames": generated,
        "published_frames": published,
        "overlap_frames": overlap,
        "tail_trim_frames": tail,
        "continuation_eligible": tail == 0,
        "sample_duration_seconds": min(generated / 24, 15.0),
        "published_duration_seconds": published / 24,
    }


def continuation_identity(recipe):
    """Hash actual local components and effective adapter/sampling settings, without weights."""
    from wee_todd_nodes.conditioning import H3TextEncoderSpec
    from wee_todd_nodes.lora import H3LoRASpec
    from wee_todd_nodes.preflight import H3ComponentSetSpec
    from wee_todd_nodes.runtime import H3GenerationConfig

    fields = dict(recipe["components"])
    fields.pop("preview_override", None)
    components = H3ComponentSetSpec(**fields)
    config = H3GenerationConfig(**recipe["config"])
    config.validate()
    cache = {}

    def file_record(filename):
        from .model_hash_cache import cached_model_hash

        resolved = filename.resolve(strict=True)
        stat = resolved.stat()
        if not resolved.is_file():
            raise ValueError(f"Continuation identity requires regular files: {filename}")
        key = (str(resolved), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key not in cache:
            digest = cached_model_hash(resolved, _hash_file)
            after = resolved.stat()
            if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != key[1:]:
                raise ValueError(
                    f"Component changed while computing continuation identity: {filename}"
                )
            cache[key] = digest
        return {"bytes": stat.st_size, "sha256": cache[key]}

    def component_record(location):
        location = Path(location).expanduser().resolve(strict=True)
        if location.is_file():
            files = {location.name: file_record(location)}
            with location.open("rb") as stream:
                is_dt = stream.read(16) == b"SQLite format 3\0"
            if is_dt:
                for suffix in ("-tensordata", "-wal"):
                    sidecar = Path(str(location) + suffix)
                    if sidecar.exists():
                        files[sidecar.name] = file_record(sidecar)
            return {"path": str(location), "files": files}
        files = {}
        visited = 0
        for folder, directories, names in os.walk(location, followlinks=False):
            visited += 1
            if visited > 10000 or len(files) + len(names) > 10000:
                raise ValueError("Continuation component inventory exceeds 10000 entries")
            for directory in directories:
                if (Path(folder) / directory).is_symlink():
                    raise ValueError(
                        "Continuation component identity cannot traverse nested directory symlinks"
                    )
            for name in sorted(names):
                filename = Path(folder) / name
                files[str(filename.relative_to(location))] = file_record(filename)
        if not files:
            raise ValueError(f"Continuation component has no files: {location}")
        from minimax_h3_mlx.dt_source import dt_source_files

        references = {
            str(filename): file_record(filename) for filename in dt_source_files(location)
        }
        return {"path": str(location), "files": files, "referenced_files": references}

    index = Path(components.checkpoint).expanduser() / "model_index.json"
    settings = asdict(config)
    for name in ("duration_seconds", "seed", "resolution_mode", "resolution_tier", "aspect_ratio"):
        settings.pop(name)
    adapters = []
    for fields in recipe.get("loras", {}).get("adapters", []):
        adapter = asdict(H3LoRASpec(**fields))
        adapter["path"] = str(Path(adapter["path"]).expanduser().resolve(strict=True))
        grid = None
        if adapter["adaln_input_grid"] is not None:
            adapter["adaln_input_grid"] = str(
                Path(adapter["adaln_input_grid"]).expanduser().resolve(strict=True)
            )
            grid = file_record(Path(adapter["adaln_input_grid"]))
        adapters.append(
            {
                "settings": adapter,
                "content": file_record(Path(adapter["path"])),
                "adaln_input_grid": grid,
            }
        )
    encoder_config = H3TextEncoderSpec.from_components(components).config_path
    identity = {
        "version": 1,
        "engine": "h3",
        "fps": 24,
        "sample_rate": 32000,
        "audio_channels": 2,
        "width": config.width,
        "height": config.height,
        "task": components.task,
        "checkpoint": str(Path(components.checkpoint).expanduser().resolve(strict=True)),
        "model_index": file_record(index),
        "text_architecture_config": (
            {
                "path": str(Path(encoder_config).resolve()),
                "content": file_record(Path(encoder_config)),
            }
            if encoder_config
            else None
        ),
        "components": {
            name: component_record(location)
            for name, location in components.resolved_paths().items()
        },
        "sampling": settings,
        "schedule": "native_h3_shifted_flow_v1",
        "block_residency": recipe.get("block_residency", "checkpoint_default"),
        "loras": adapters,
    }
    if len(_json_bytes(identity)) > MAX_MANIFEST_BYTES // 2:
        raise ValueError("Continuation identity inventory is too large")
    return identity


def _shapes(context_frames, width, height):
    if type(context_frames) is not int or context_frames not in CONTEXT_FRAMES:
        raise ValueError("Invalid continuation context_frames")
    if any(
        type(size) is not int or not 32 <= size <= 1920 or size % 32 for size in (width, height)
    ):
        raise ValueError("Invalid continuation canvas")
    return {
        "video": [1, 24, (context_frames - 5) // 17 * 5 + 2, height // 16, width // 16],
        "audio": [2, 32, round(context_frames / 24 * 40)],
    }


def _validate_provenance(provenance, context_frames):
    if not isinstance(provenance, dict) or provenance.get("tail_trim_frames") != 0:
        raise ValueError("Cannot persist continuation from a trimmed or unknown tail")
    published = provenance.get("published_frames")
    generated = provenance.get("generated_frames")
    overlap = provenance.get("overlap_frames", 0)
    if (
        type(published) is not int
        or published <= context_frames
        or type(generated) is not int
        or not 5 < generated <= 362
        or (generated - 5) % 17
        or type(overlap) is not int
        or overlap not in (0, *CONTEXT_FRAMES)
        or generated - overlap != published
    ):
        raise ValueError("Continuation provenance does not identify an untrimmed visible tail")


def _payload_header(filename, shapes):
    size = filename.stat().st_size
    if not 8 < size <= MAX_PAYLOAD_BYTES:
        raise ValueError("Continuation payload exceeds its bounded size")
    with filename.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if not 1 <= header_size <= MAX_HEADER_BYTES or header_size + 8 > size:
            raise ValueError("Invalid continuation SafeTensors header size")
        header = json.loads(stream.read(header_size))
    if not isinstance(header, dict) or set(header) - {"__metadata__"} != {"video", "audio"}:
        raise ValueError("Continuation payload must contain only video and audio tensors")
    offsets = []
    for name, shape in shapes.items():
        info = header[name]
        if (
            not isinstance(info, dict)
            or info.get("shape") != shape
            or info.get("dtype") not in {"F16", "BF16", "F32"}
        ):
            raise ValueError(f"Continuation {name} shape or dtype is incompatible")
        offset = info.get("data_offsets")
        if (
            not isinstance(offset, list)
            or len(offset) != 2
            or any(type(n) is not int for n in offset)
        ):
            raise ValueError("Invalid continuation tensor offsets")
        nbytes = math.prod(shape) * (4 if info["dtype"] == "F32" else 2)
        if offset[0] < 0 or offset[1] - offset[0] != nbytes:
            raise ValueError("Invalid continuation tensor byte count")
        offsets.append(offset)
    offsets.sort()
    if (
        offsets[0][0] != 0
        or offsets[0][1] != offsets[1][0]
        or offsets[1][1] != size - header_size - 8
    ):
        raise ValueError("Continuation tensor ranges do not cover the payload")
    return {name: header[name] for name in shapes}


def save_continuation_artifact(context, directory, *, identity, provenance):
    """Commit a new artifact; never replace an existing take's context."""
    shapes = _shapes(context.context_frames, context.width, context.height)
    _validate_provenance(provenance, context.context_frames)
    if (
        str(Path(context.transformer_checkpoint).expanduser().resolve()) != identity["checkpoint"]
        or str(Path(context.transformer_path).expanduser().resolve())
        != identity["components"]["transformer"]["path"]
    ):
        raise ValueError("Continuation transformer provenance differs from identity")
    if (context.width, context.height, context.fps, context.sample_rate) != (
        identity["width"],
        identity["height"],
        identity["fps"],
        identity["sample_rate"],
    ) or (context.fps, context.sample_rate) != (24, 32000):
        raise ValueError("Continuation context timing/canvas differs from identity")
    for name, shape in shapes.items():
        if list(getattr(context, name).shape) != shape:
            raise ValueError(f"Continuation {name} shape differs from the native canvas")
    import mlx.core as mx

    arrays = {name: mx.array(getattr(context, name)) for name in shapes}
    if any(
        array.dtype not in (mx.float16, mx.bfloat16, mx.float32)
        or not mx.all(mx.isfinite(array)).item()
        for array in arrays.values()
    ):
        raise ValueError(
            "Continuation tensors must contain finite float16, bfloat16, or float32 values"
        )
    directory = Path(directory).absolute()
    directory.mkdir(parents=True, exist_ok=False)
    try:
        payload = directory / "latents.safetensors"
        mx.save_safetensors(str(payload), arrays)
        tensors = _payload_header(payload, shapes)
        with payload.open("rb") as stream:
            os.fsync(stream.fileno())
        manifest = {
            "format": FORMAT,
            "context_frames": context.context_frames,
            "identity": identity,
            "provenance": provenance,
            "payload": {
                "file": payload.name,
                "bytes": payload.stat().st_size,
                "sha256": _hash_file(payload),
                "tensors": tensors,
            },
        }
        encoded = _json_bytes(manifest)
        if len(encoded) > MAX_MANIFEST_BYTES:
            raise ValueError("Continuation manifest exceeds its bounded size")
        pending = directory / ".manifest.partial.json"
        with pending.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, directory / "manifest.json")
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        shutil.rmtree(directory)
        raise
    return {
        "manifest": str(directory / "manifest.json"),
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "payload_sha256": manifest["payload"]["sha256"],
    }


def load_continuation_artifact(
    filename, expected_sha256, *, identity, context_frames, load_arrays=True
):
    """Verify the manifest, complete payload, geometry and identity before array allocation."""
    _checked_hash(expected_sha256, "source manifest hash")
    filename = Path(filename)
    with filename.open("rb") as stream:
        encoded = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise ValueError("Continuation manifest exceeds its bounded size")
    if hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise ValueError("Continuation manifest hash mismatch")
    manifest = json.loads(encoded)
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ValueError("Unsupported continuation artifact format")
    if manifest.get("identity") != identity:
        raise ValueError(
            "Continuation identity differs: components, canvas, timing, adapters "
            "or schedule changed"
        )
    if manifest.get("context_frames") != context_frames:
        raise ValueError("Continuation context_frames differs from the saved artifact")
    shapes = _shapes(context_frames, identity["width"], identity["height"])
    _validate_provenance(manifest.get("provenance"), context_frames)
    payload = manifest.get("payload")
    if not isinstance(payload, dict) or payload.get("file") != "latents.safetensors":
        raise ValueError("Continuation payload must use the local latents.safetensors file")
    filename = filename.parent / "latents.safetensors"
    if filename.is_symlink() or not filename.is_file():
        raise ValueError("Continuation payload must be a regular local artifact file")
    tensors = _payload_header(filename, shapes)
    if filename.stat().st_size != payload.get("bytes") or tensors != payload.get("tensors"):
        raise ValueError("Continuation payload geometry or size differs from manifest")
    # Keep the verified bytes immutable through deserialization, even if another process
    # replaces the source file between verification and mx.load. Both copies are bounded.
    with filename.open("rb") as stream:
        payload_bytes = stream.read(MAX_PAYLOAD_BYTES + 1)
    if len(payload_bytes) > MAX_PAYLOAD_BYTES or hashlib.sha256(
        payload_bytes
    ).hexdigest() != _checked_hash(payload.get("sha256"), "payload hash"):
        raise ValueError("Continuation payload hash mismatch")
    if not load_arrays:
        return None, manifest
    import mlx.core as mx

    from wee_todd_nodes.continuation import H3ContinuationContext

    arrays = mx.load(io.BytesIO(payload_bytes), format="safetensors")
    if any(not mx.all(mx.isfinite(array)).item() for array in arrays.values()):
        raise ValueError("Continuation payload contains nonfinite latents")
    context = H3ContinuationContext(
        video=arrays["video"],
        audio=arrays["audio"],
        context_frames=context_frames,
        width=identity["width"],
        height=identity["height"],
        fps=identity["fps"],
        sample_rate=identity["sample_rate"],
        transformer_checkpoint=identity["checkpoint"],
        transformer_path=identity["components"]["transformer"]["path"],
    )
    return context, manifest


def prepare_continuation(recipe, *, task=None, load_arrays=True):
    request = continuation_request(recipe, task=task)
    if request is None:
        return None
    identity = continuation_identity(recipe)
    context = source = None
    if request.get("source_context"):
        context, source = load_continuation_artifact(
            request["source_context"],
            request["source_manifest_sha256"],
            identity=identity,
            context_frames=request["context_frames"],
            load_arrays=load_arrays,
        )
    return {"request": request, "identity": identity, "context": context, "source": source}


class ContinuationVideoDecoder:
    """Filter bounded decoded chunks before the ordinary direct publisher sees them."""

    def __init__(self, delegate, request):
        self.delegate = delegate
        self.request = request

    def decode_stream(self, spec, latents, emit, **kwargs):
        seen = written = 0
        start = self.request["overlap_frames"]
        stop = start + self.request["published_frames"]

        def publish_chunk(chunk):
            nonlocal seen, written
            count = int(chunk.shape[0])
            first = max(0, start - seen)
            last = min(count, stop - seen)
            if last > first:
                emit(chunk[first:last])
                written += last - first
            seen += count

        result = self.delegate.decode_stream(spec, latents, publish_chunk, **kwargs)
        if (
            seen != self.request["generated_frames"]
            or result.num_frames != seen
            or written != self.request["published_frames"]
        ):
            raise ValueError("Continuation video decode differs from the planned frame window")
        return replace(result, num_frames=written)

    def unload(self):
        self.delegate.unload()


class ContinuationAudioDecoder:
    """Use the shared synchronized overlap trim, then the same editorial tail as video."""

    def __init__(self, delegate, request):
        self.delegate = delegate
        self.request = request

    def decode(self, spec, latents, **kwargs):
        import numpy as np

        from wee_todd_nodes.continuation import trim_continuation_overlap

        result = self.delegate.decode(spec, latents, **kwargs)
        if result.video_frames != self.request["generated_frames"] or result.fps != 24:
            raise ValueError("Continuation audio decode differs from the planned frame window")
        # The existing trim helper only uses IMAGE's leading dimension for timing. Zero
        # spatial dimensions retain that contract with zero bytes of RGB allocation.
        geometry = np.empty((result.video_frames, 0, 0, 3), dtype=np.uint8)
        _, audio, _ = trim_continuation_overlap(
            geometry,
            {"waveform": result.waveform, "sample_rate": result.sample_rate},
            self.request["overlap_frames"],
        )
        samples = round(self.request["published_frames"] / 24 * result.sample_rate)
        waveform = audio["waveform"][..., :samples]
        if waveform.shape[-1] != samples:
            raise ValueError("Continuation audio is shorter than the published frame window")
        return replace(
            result,
            waveform=waveform,
            num_samples=samples,
            duration_seconds=samples / result.sample_rate,
            video_frames=self.request["published_frames"],
        )

    def unload(self):
        self.delegate.unload()
