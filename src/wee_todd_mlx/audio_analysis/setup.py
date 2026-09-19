"""Pinned small analysis models through Studio's existing verified download transport."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..model_downloads import DownloadFile, download_file


def catalog():
    return json.loads(Path(__file__).with_name("catalog.json").read_text())


def model_identity(directory, *, check=None, include_vocals=False):
    check = check or (lambda: None)
    root = Path(directory).expanduser()
    hashes = {}
    for model in catalog()["models"]:
        if model.get("optional") and not include_vocals:
            continue
        for record in model["files"]:
            check()
            path = root / record["target"]
            if not path.is_file() or path.stat().st_size != record["size"]:
                raise ValueError(
                    "Set up the audio analysis models, or choose their complete model folder"
                )
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    check()
                    digest.update(chunk)
            if digest.hexdigest() != record["sha256"]:
                raise ValueError(
                    "Analysis checkpoint verification failed; restore the pinned model files"
                )
            hashes[record["target"]] = record["sha256"]
    return hashes


def download(directory, *, progress=None, cancelled=None, include_vocals=False):
    cancelled = cancelled or (lambda: False)

    def check():
        if cancelled():
            raise InterruptedError("Analysis model setup cancelled; retry to resume")

    if type(include_vocals) is not bool:
        raise ValueError("Choose whether to include the optional vocal model")
    check()
    root = Path(directory).expanduser().resolve() / "WeeTodd-Analysis"
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".setup-lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError("Another analysis model setup is using this folder") from None
    try:
        os.close(fd)
        selected = [
            model for model in catalog()["models"] if include_vocals or not model.get("optional")
        ]
        records = [item for model in selected for item in model["files"]]
        for i, record in enumerate(records):
            item = DownloadFile(**record)

            def report(message, fraction, index=i):
                if progress:
                    progress(
                        dict(
                            event="progress",
                            message=message,
                            fraction=(index + fraction) / len(records),
                        )
                    )

            download_file(item, root / item.target, progress=report, cancelled=cancelled)
        model_identity(root, check=check, include_vocals=include_vocals)
        (root / "weetodd-model-source.json").write_text(
            json.dumps({**catalog(), "models": selected}, indent=2)
        )
        return dict(model_directory=str(root), bytes=sum(item["size"] for item in records))
    finally:
        lock.unlink(missing_ok=True)
