"""Pinned music checkpoint setup, sharing Studio's verified download transport."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .model_downloads import DownloadFile, download_file


def catalog():
    return json.loads(Path(__file__).with_name("music_model_catalog.json").read_text())


def download(destination, *, progress=None):
    record = catalog()
    root = Path(destination).expanduser().resolve() / record["id"]
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".setup-lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError("Another music setup owns this folder. Wait for it to finish.") from None
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()))
        files = [DownloadFile(**item) for item in record["files"]]
        remaining = sum(item.size for item in files if not (root / item.target).exists())
        if shutil.disk_usage(root).free < remaining + 128 * 1024 * 1024:
            raise ValueError("The selected library does not have enough free space for YuE2.")
        for index, item in enumerate(files):

            def report(message, fraction, index=index):
                if progress:
                    progress(
                        dict(
                            event="progress",
                            message=message,
                            fraction=(index + fraction) / len(files),
                        )
                    )

            download_file(item, root / item.target, progress=report)
        (root / "weetodd-model-source.json").write_text(json.dumps(record, indent=2))
        from yue2_mlx.checkpoint import inspect_checkpoint

        inspect_checkpoint(root, precision="8bit")
        return dict(model_path=str(root), license=record["license"], source=record["source"])
    finally:
        lock.unlink(missing_ok=True)
