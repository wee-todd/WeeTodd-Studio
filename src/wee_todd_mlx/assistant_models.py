"""Explicit assistant setup, separate from native video encoders and clip recipes.

Imports are model/network free. The source manifest identifies the separately
linked helper's exact checkpoint format; it is not a universal Qwen loader.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import zlib
from pathlib import Path
from uuid import uuid4

from .model_downloads import (
    RESERVE_BYTES,
    DownloadFile,
    _check_space,
    download_file,
    partial_path,
)

CHECKPOINT = DownloadFile(
    "drawthingsai/draw-things-community",
    "08e798b5ad59c3db78b2be53f0ed60b071653302",
    "qwen_3.5_4b_i8x.ckpt",
    4888076288,
    "9e07b480a4e093d1d304ee59a625fc3169a53dc3a597d17f69a2dc167958ee9c",
    "qwen_3.5_4b_i8x.ckpt",
    provider="drawthings-static",
)
RECEIPT_NAME = "qwen_3.5_4b_i8x.install.json"
FILENAMES = ("qwen_3.5_4b_i8x.ckpt", "qwen_3.5_9b_i5x.ckpt")


def catalog():
    return {
        "id": "qwen35-4b-dt-i8x",
        "name": "Qwen3.5 4B · text and vision",
        "filename": CHECKPOINT.filename,
        "runtime": "drawthings-qwen-local",
        "downloadBytes": CHECKPOINT.size,
        "requiredDiskBytes": CHECKPOINT.size + RESERVE_BYTES,
        "sha256": CHECKPOINT.sha256,
        "sourceURL": CHECKPOINT.url,
        "sourceRevision": CHECKPOINT.revision,
        "licenseURL": "https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE",
        "notice": "Uses the separately linked local Qwen helper. Draw Things app is not required. "
        "Other Qwen formats and H3 encoders are not interchangeable.",
    }


def _cancel(cancelled):
    if cancelled():
        raise InterruptedError("Assistant model setup cancelled; retry to resume")


def _checksum(path, cancelled):
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            _cancel(cancelled)
            digest.update(block)
    after = path.stat()

    def identity(s):
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    if identity(before) != identity(after):
        raise ValueError("Checkpoint changed during verification; retry")
    return digest.hexdigest()


def inspect_model(source, *, cancelled=lambda: False):
    """Check an installed file in place; never write receipts beside user files."""
    path = Path(source).expanduser().absolute()
    _cancel(cancelled)
    if path.name not in FILENAMES or not path.is_file():
        raise ValueError("Locate the supported Qwen3.5 4B or 9B checkpoint")
    with path.open("rb") as stream:
        if stream.read(16) != b"SQLite format 3\x00":
            raise ValueError("Checkpoint is not the supported SQLite format")
    verified = False
    if path.name == CHECKPOINT.filename:
        if path.stat().st_size == CHECKPOINT.size:
            verified = _checksum(path, cancelled) == CHECKPOINT.sha256
            if not verified:
                raise ValueError(
                    "Checkpoint checksum differs; preserve it and select a verified copy"
                )
        elif not Path(str(path) + "-tensordata").is_file():
            raise ValueError("Checkpoint is incomplete or uses an unsupported layout")
    return {
        "path": str(path),
        "status": "checksum_verified" if verified else "format_checked",
        "inferenceChecked": False,
        "requiresHealthCheck": not verified,
        "vision": path.name == FILENAMES[0],
        "message": "Checkpoint checksum verified. Inference has not been checked."
        if verified
        else "Existing checkpoint format found. Run the health check before using it.",
    }


def install(destination, *, opener=None, progress=lambda *_: None, cancelled=lambda: False):
    """Install at the selected location, preserving unrelated files and resumable partials."""
    root = Path(destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / CHECKPOINT.filename
    part = partial_path(CHECKPOINT, target)
    receipt = root / RECEIPT_NAME
    lock = root / ".qwen35-4b-setup.lock"
    for value in (target, part, receipt, lock):
        if value.is_symlink():
            raise ValueError("Refusing a symlink in the assistant install destination")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError(
            "Assistant setup is already active here. If interrupted by a crash, "
            "confirm no setup is running before removing .qwen35-4b-setup.lock"
        ) from None
    try:
        os.close(fd)
        _cancel(cancelled)
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if (
                not isinstance(previous, dict)
                or previous.get("format") != "weetodd-assistant-install-v1"
                or previous.get("sha256") != CHECKPOINT.sha256
            ):
                raise ValueError("Existing install receipt differs; move it aside before retrying")
        if part.exists() and part.stat().st_size > CHECKPOINT.size:
            part.unlink()  # This checksum-named partial is owned by this installer.
        remaining = (
            0
            if target.exists()
            else CHECKPOINT.size - (part.stat().st_size if part.exists() else 0)
        )
        _check_space(root, remaining + RESERVE_BYTES)
        download_file(CHECKPOINT, target, opener=opener, progress=progress, cancelled=cancelled)
        _cancel(cancelled)
        with target.open("rb") as stream:
            if stream.read(16) != b"SQLite format 3\x00":
                raise ValueError("Verified download is not the supported SQLite format")
        result = {
            "path": str(target),
            "status": "checksum_verified",
            "vision": True,
            "inferenceChecked": False,
            "requiresHealthCheck": False,
            "message": "Checkpoint checksum verified. Inference has not been checked.",
        }
        record = {
            "format": "weetodd-assistant-install-v1",
            **catalog(),
            "path": str(target),
            "inferenceChecked": False,
        }
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if (
                previous.get("format") != record["format"]
                or previous.get("sha256") != CHECKPOINT.sha256
            ):
                raise ValueError("Existing install receipt differs; move it aside before retrying")
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=root, prefix=".assistant-receipt-", delete=False
            ) as stream:
                temporary = Path(stream.name)
                try:
                    json.dump(record, stream, indent=2)
                    stream.flush()
                    os.fsync(stream.fileno())
                    os.link(temporary, receipt)
                finally:
                    temporary.unlink(missing_ok=True)
        return result
    finally:
        lock.unlink(missing_ok=True)


def _test_image(path):
    """Small deterministic visual input; does not use image generation or dependencies."""

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    pixels = b"".join(b"\x00" + b"\xff\x00\x00" * 64 + b"\x00\x00\xff" * 64 for _ in range(128))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 128, 128, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )


def health_check(source, helper, *, progress=lambda *_: None, cancelled=lambda: False, invoke=None):
    """Explicit local inference smoke test, not quality or adapter qualification."""
    from wee_todd_remote.client import invoke_helper

    progress("Verifying checkpoint before local inference…", 0.0)
    inspected = inspect_model(source, cancelled=cancelled)
    helper = Path(helper).expanduser().absolute()
    if not helper.is_file() or not os.access(helper, os.X_OK):
        raise ValueError("Set up the local Qwen helper before running its health check")
    results = []
    with tempfile.TemporaryDirectory(prefix="weetodd-assistant-health-") as folder:
        image = Path(folder) / "two-colors.png"
        _test_image(image)
        cases = [("text", "Reply with the single word READY.", [])]
        if inspected["vision"]:
            cases.append(
                (
                    "vision",
                    "Name the solid color on the left and the solid color on the right.",
                    [{"path": str(image), "label": "Two solid colors"}],
                )
            )
        for name, prompt, images in cases:
            _cancel(cancelled)
            progress("Checking local " + name + " inference…", 0.0)
            payload = {
                "requestID": str(uuid4()),
                "modelPath": inspected["path"],
                "systemPrompt": "Follow the instruction briefly.",
                "prompt": prompt,
                "maxTokens": 64,
                "images": images,
            }
            value = None
            for event in (invoke or invoke_helper)(
                "text", payload, helper=helper, cancelled=cancelled, timeout=600
            ):
                if event.get("type") == "result":
                    value = event.get("value")
            if (
                not isinstance(value, dict)
                or not str(value.get("text", "")).strip()
                or value.get("truncated")
            ):
                raise ValueError("Local " + name + " inference did not return a complete response")
            if images and value.get("imagesUsed") != 1:
                raise ValueError("Vision health check did not confirm reading the test image")
            results.append({"kind": name, "result": value})
    return {
        **inspected,
        "status": "inference_checked",
        "inferenceChecked": True,
        "requiresHealthCheck": False,
        "checks": results,
        "message": "Local inference returned text"
        + (" and read a test image" if inspected["vision"] else "")
        + ". This is a smoke test, not a quality certification.",
    }
