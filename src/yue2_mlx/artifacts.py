"""Content-addressed local stage bundles; no model or numerical imports at import time."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import time
from pathlib import Path

from .checkpoint import inspect_checkpoint
from .config import validate_request
from .protocol import CODEC_OFFSET, CODEC_SIZE, EOD, negative_tokens, prefix_tokens

FILES = frozenset(
    {
        "request.json",
        "effective.json",
        "score.abc",
        "tokens.json",
        "noise.npy",
        "latents.npy",
        "master.wav",
        "result.json",
    }
)
BASE = frozenset({"request.json", "effective.json", "score.abc", "tokens.json", "result.json"})
STAGES = frozenset({"plan", "generate", "resynthesize", "decode_latents", "legacy_import"})


def digest(filename, cancelled=None):
    filename = Path(filename)
    before = filename.stat()
    value = hashlib.sha256()
    with filename.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            if cancelled and cancelled():
                raise InterruptedError("YuE2 artifact hashing cancelled")
            value.update(chunk)
    after = filename.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(f"File changed during integrity check: {filename}")
    return value.hexdigest()


def checkpoint_fingerprints(inspection, cancelled=None):
    result = {}
    for role, identity in inspection["identities"].items():
        result[role] = dict(identity, sha256=digest(inspection["files"][role], cancelled))
    return result


def engine_identity():
    code = hashlib.sha256()
    for filename in sorted(Path(__file__).parent.glob("*.py")):
        code.update(filename.name.encode())
        code.update(filename.read_bytes())
    versions = {"python": platform.python_version()}
    for package in ("mlx", "numpy", "tiktoken"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return dict(source_sha256=code.hexdigest(), runtime=versions)


def seal(directory, stage, cancelled=None):
    directory = Path(directory)
    if stage not in STAGES:
        raise ValueError("Unknown YuE2 artifact stage")
    present = {name for name in FILES if (directory / name).is_file()}
    required = BASE if stage == "plan" else BASE | {"noise.npy", "latents.npy"}
    if not required <= present:
        raise ValueError(f"Incomplete stage bundle: {sorted(required - present)}")
    manifest = dict(
        format="weetodd-yue2-artifacts-v1",
        stage=stage,
        files={
            name: dict(
                sha256=digest(directory / name, cancelled), bytes=(directory / name).stat().st_size
            )
            for name in sorted(present)
        },
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _read(directory, name):
    value = json.loads((directory / name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def validate_contents(directory, request, inspection, stage):
    """Validate independently meaningful relationships, beyond byte integrity."""
    from .tokenizer import Tokenizer

    tokenizer = Tokenizer(inspection["files"]["tokenizer"])
    tokens = _read(directory, "tokens.json")
    abc = tokens.get("abc")
    if not isinstance(abc, list) or any(type(i) is not int or not 0 <= i < EOD for i in abc):
        raise ValueError("Invalid saved score IDs")
    if request["cot"] == "off" and abc:
        raise ValueError("Score IDs conflict with cot=off")
    if request["abc"] is not None and tokenizer.encode(request["abc"]) != abc:
        raise ValueError("Saved score differs from the supplied request score")
    if (directory / "score.abc").read_text(encoding="utf-8") != tokenizer.decode(abc):
        raise ValueError("Score text does not match saved score IDs")
    if tokens.get("prefix") != prefix_tokens(request, tokenizer, abc):
        raise ValueError("Saved prefix does not match the request and score")
    negative = negative_tokens(request, tokenizer, abc) if request["cfg_scale"] != 1 else None
    if tokens.get("negative_prefix") != negative:
        raise ValueError("Saved guided prefix does not match the request")
    truncated = tokens.get("truncated")
    if (
        not isinstance(truncated, dict)
        or set(truncated) != {"abc", "semantic"}
        or any(type(x) is not bool for x in truncated.values())
    ):
        raise ValueError("Invalid token truncation metadata")
    codec = tokens.get("codec")
    if not isinstance(codec, list) or any(
        type(i) is not int or not 0 <= i < CODEC_SIZE for i in codec
    ):
        raise ValueError("Invalid saved codec IDs")
    if tokens.get("semantic") != [i + CODEC_OFFSET for i in codec]:
        raise ValueError("Semantic IDs do not match codec IDs")
    if stage == "plan":
        if codec:
            raise ValueError("Plan artifact contains unexpected semantic tokens")
    else:
        if not codec:
            raise ValueError("Saved artifact contains no semantic tokens")
        import numpy as np

        for name in ("noise.npy", "latents.npy"):
            array = np.load(directory / name, allow_pickle=False, mmap_mode="r")
            if (
                array.shape != (len(codec), 64)
                or array.dtype != np.float32
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"Invalid saved {name}: require finite float32 [frames,64]")
    return tokens


def verify(source, cancelled=None):
    started = time.perf_counter()
    directory = Path(source).expanduser().resolve()
    manifest = _read(directory, "manifest.json")
    if manifest.get("format") != "weetodd-yue2-artifacts-v1" or manifest.get("stage") not in STAGES:
        raise ValueError("Unsupported YuE2 artifact manifest")
    inventory = manifest.get("files")
    required = BASE if manifest["stage"] == "plan" else BASE | {"noise.npy", "latents.npy"}
    if (
        not isinstance(inventory, dict)
        or not required <= set(inventory)
        or not set(inventory) <= FILES
    ):
        raise ValueError("Incomplete or invalid artifact integrity inventory")
    for name, entry in inventory.items():
        filename = directory / name
        if not isinstance(entry, dict) or not re.fullmatch(
            "[0-9a-f]{64}", str(entry.get("sha256", ""))
        ):
            raise ValueError("Invalid artifact integrity digest")
        if (
            not filename.is_file()
            or filename.stat().st_size != entry.get("bytes")
            or digest(filename, cancelled) != entry["sha256"]
        ):
            raise ValueError(f"Artifact integrity check failed: {name}")
    request = _read(directory, "request.json")
    if validate_request(request) != request:
        raise ValueError("Saved request is not normalized")
    effective = _read(directory, "effective.json")
    if effective.get("protocol") != "yue2-native-v1":
        raise ValueError("Unsupported saved generation protocol")
    for key, setting in (
        ("steps", "steps"),
        ("guidance", "cfg_scale"),
        ("memory_mode", "memory_mode"),
    ):
        if effective.get(key) != request[setting]:
            raise ValueError(f"Effective {key} conflicts with the saved request")
    if effective.get("sampling") != {
        "abc": request["abc_sampling"],
        "semantic": request["semantic_sampling"],
    }:
        raise ValueError("Effective sampling metadata conflicts with the saved request")
    result = _read(directory, "result.json")
    if result.get("stage") != manifest["stage"]:
        raise ValueError("Result stage metadata conflicts with the manifest")
    if result.get("request") != request:
        raise ValueError("Result request differs from the saved request")
    inspection = inspect_checkpoint(
        request["model_path"], request["vae_path"], request["precision"]
    )
    if (
        effective.get("precision") != inspection["precision"]
        or effective.get("layout") != inspection["layout"]
    ):
        raise ValueError("Effective precision/layout metadata conflicts with the checkpoint")
    current = checkpoint_fingerprints(inspection, cancelled)
    expected = effective.get("models")
    if not isinstance(expected, dict) or set(expected) != set(current):
        raise ValueError("Missing checkpoint content provenance")
    for role, identity in current.items():
        if (
            expected[role].get("sha256") != identity["sha256"]
            or expected[role].get("bytes") != identity["bytes"]
        ):
            raise ValueError(f"Checkpoint content mismatch: {role}")
    tokens = validate_contents(directory, request, inspection, manifest["stage"])
    if result.get("truncated") != tokens["truncated"]:
        raise ValueError("Result truncation metadata conflicts with the saved tokens")
    return dict(
        directory=directory,
        request=request,
        effective=effective,
        tokens=tokens,
        inspection=inspection,
        manifest=manifest,
        manifest_sha256=digest(directory / "manifest.json"),
        verification_seconds=time.perf_counter() - started,
    )


def copy_verified(checked, name, destination, cancelled=None):
    """Copy only a manifest member and recheck its bytes before consuming it."""
    import shutil

    if name not in checked["manifest"]["files"] or name not in FILES:
        raise ValueError("Unknown source artifact")
    destination = Path(destination)
    shutil.copyfile(checked["directory"] / name, destination)
    expected = checked["manifest"]["files"][name]
    if (
        destination.stat().st_size != expected["bytes"]
        or digest(destination, cancelled) != expected["sha256"]
    ):
        raise ValueError(f"Copied artifact integrity check failed: {name}")
