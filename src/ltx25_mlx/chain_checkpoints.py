"""Bounded, content-identified SafeTensors artifacts for completed native windows.

A manifest is published last. Missing, partial or corrupt artifacts are cache
misses, never inputs to an unchecked tensor loader. No pickle is used.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import uuid
from pathlib import Path

MAX_HEADER_BYTES = 64 * 1024
MAX_MANIFEST_BYTES = 64 * 1024
MAX_PAYLOAD_BYTES = 1024 * 1024 * 1024
FORMAT_VERSION = 1


def prefix_identity(previous: str, window_input: dict) -> str:
    encoded = json.dumps(
        {"version": FORMAT_VERSION, "previous": previous, "input": window_input},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(filename: Path) -> str:
    digest = hashlib.sha256()
    with filename.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_state(filename: str | Path) -> tuple[int, ...]:
    """Modification identity; ordinary reads changing access time are irrelevant."""
    stat = Path(filename).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def content_identity(filename: str | Path, *, model_file: bool = False) -> str:
    """Hash content; only model files use the shared durable identity cache."""
    source = Path(filename).expanduser().resolve(strict=True)
    if source.is_dir():
        files = sorted(item for item in source.rglob("*") if item.is_file())
        if not files or len(files) > 10000:
            raise ValueError("LTX 2.5 model directory has an invalid file count.")
        return prefix_identity(
            "directory",
            {
                str(item.relative_to(source)): content_identity(item, model_file=model_file)
                for item in files
            },
        )
    before = file_state(source)
    if model_file:
        from wee_todd_mlx.model_hash_cache import cached_model_hash

        digest = cached_model_hash(source, _sha256)
    else:
        digest = _sha256(source)
    if file_state(source) != before:
        raise ValueError("LTX 2.5 input changed while computing checkpoint identity.")
    return digest


def validate_checkpoint_shapes(shapes):
    if not shapes or len(shapes) > 5:
        raise ValueError("Invalid LTX 2.5 checkpoint tensor count.")
    size = 0
    for shape in shapes.values():
        if not 1 <= len(shape) <= 5 or any(
            type(n) is not int or not 0 < n <= 1000000 for n in shape
        ):
            raise ValueError("Invalid LTX 2.5 checkpoint tensor shape.")
        size += math.prod(shape) * 4
    if size > MAX_PAYLOAD_BYTES - MAX_HEADER_BYTES:
        raise ValueError("LTX 2.5 window checkpoint exceeds its byte limit.")


def _validate_header(filename: Path | bytes, shapes):
    validate_checkpoint_shapes(shapes)
    size = len(filename) if isinstance(filename, bytes) else filename.stat().st_size
    if not 8 < size <= MAX_PAYLOAD_BYTES:
        raise ValueError("Invalid LTX 2.5 checkpoint payload size.")
    with io.BytesIO(filename) if isinstance(filename, bytes) else filename.open("rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
        if not 2 <= header_size <= min(MAX_HEADER_BYTES, size - 8):
            raise ValueError("Invalid LTX 2.5 checkpoint header size.")
        header = json.loads(stream.read(header_size))
    if isinstance(header, dict):
        header.pop("__metadata__", None)
    if not isinstance(header, dict) or set(header) != set(shapes):
        raise ValueError("LTX 2.5 checkpoint tensor names do not match.")
    spans = []
    for name, expected in shapes.items():
        tensor = header[name]
        if not isinstance(tensor, dict) or tensor.get("shape") != list(expected):
            raise ValueError("LTX 2.5 checkpoint shape does not match its window.")
        item_bytes = {"F32": 4, "BF16": 2, "F16": 2}.get(tensor.get("dtype"))
        offsets = tensor.get("data_offsets")
        if (
            not item_bytes
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(n) is not int for n in offsets)
        ):
            raise ValueError("Invalid LTX 2.5 checkpoint tensor encoding.")
        start, end = offsets
        if start < 0 or end - start != math.prod(expected) * item_bytes:
            raise ValueError("Invalid LTX 2.5 checkpoint tensor byte range.")
        spans.append((start, end))
    cursor = 0
    for start, end in sorted(spans):
        if start != cursor:
            raise ValueError("LTX 2.5 checkpoint contains overlapping or missing data.")
        cursor = end
    if cursor != size - header_size - 8:
        raise ValueError("LTX 2.5 checkpoint data size is inconsistent.")


class ChainCheckpointStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)

    def _manifest(self, index, prefix):
        if (
            type(index) is not int
            or not 0 <= index < 6
            or len(prefix) != 64
            or any(c not in "0123456789abcdef" for c in prefix)
        ):
            raise ValueError("Invalid LTX 2.5 checkpoint window identity.")
        return self.directory / f"window-{index:02d}-{prefix}.json"

    def load(self, index: int, prefix: str, shapes: dict):
        import mlx.core as mx

        manifest_path = self._manifest(index, prefix)
        try:
            validate_checkpoint_shapes(shapes)
            if manifest_path.is_symlink():
                return None
            with manifest_path.open("rb") as stream:
                encoded = stream.read(MAX_MANIFEST_BYTES + 1)
            if len(encoded) > MAX_MANIFEST_BYTES:
                return None
            manifest = json.loads(encoded)
            if (
                manifest.get("version") != FORMAT_VERSION
                or manifest.get("prefix") != prefix
                or manifest.get("index") != index
            ):
                return None
            digest = manifest["sha256"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                return None
            if manifest["payload"] != f"{digest}.safetensors":
                return None
            payload = self.directory / manifest["payload"]
            if payload.is_symlink():
                return None
            _validate_header(payload, shapes)
            limit = min(
                MAX_PAYLOAD_BYTES,
                8 + MAX_HEADER_BYTES + sum(math.prod(shape) * 4 for shape in shapes.values()),
            )
            with payload.open("rb") as stream:
                data = stream.read(limit + 1)
            if len(data) > limit or hashlib.sha256(data).hexdigest() != digest:
                return None
            # Validate and load the same bounded immutable bytes. Reopening a
            # path after header inspection would allow a concurrent replacement
            # to bypass the allocation limits.
            _validate_header(data, shapes)
            arrays = mx.load(io.BytesIO(data), format="safetensors")
            if set(arrays) != set(shapes) or any(
                tuple(arrays[key].shape) != tuple(shapes[key]) for key in shapes
            ):
                return None
            if any(not bool(mx.all(mx.isfinite(array)).item()) for array in arrays.values()):
                return None
            return arrays
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            OverflowError,
            RuntimeError,
            AttributeError,
        ):
            return None

    def save(self, index: int, prefix: str, arrays: dict, shapes: dict) -> None:
        import mlx.core as mx

        manifest_path = self._manifest(index, prefix)
        validate_checkpoint_shapes(shapes)
        if set(arrays) != set(shapes) or any(
            tuple(arrays[key].shape) != tuple(shapes[key]) for key in shapes
        ):
            raise ValueError("LTX 2.5 checkpoint output shape does not match its plan.")
        if any(
            array.dtype not in (mx.float16, mx.bfloat16, mx.float32) for array in arrays.values()
        ):
            raise ValueError("LTX 2.5 checkpoint supports only floating point latents.")
        if any(not bool(mx.all(mx.isfinite(array)).item()) for array in arrays.values()):
            raise ValueError("LTX 2.5 checkpoint cannot contain nonfinite latents.")
        token = uuid.uuid4().hex
        temporary = self.directory / f".{token}.safetensors"
        pending = self.directory / f".{token}.json"
        try:
            mx.save_safetensors(
                str(temporary), {key: mx.contiguous(value) for key, value in arrays.items()}
            )
            _validate_header(temporary, shapes)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            digest = _sha256(temporary)
            payload = self.directory / f"{digest}.safetensors"
            os.replace(temporary, payload)
            manifest = {
                "version": FORMAT_VERSION,
                "index": index,
                "prefix": prefix,
                "payload": payload.name,
                "sha256": digest,
            }
            with pending.open("w") as stream:
                json.dump(manifest, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(pending, manifest_path)
            descriptor = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)
            pending.unlink(missing_ok=True)
