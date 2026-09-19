"""Explicit speech model setup using pinned, integrity-checked downloads."""

from __future__ import annotations

import fcntl
import json
import shutil
from pathlib import Path

from .model_downloads import DownloadFile, _hash, download_file


def catalog():
    return json.loads(Path(__file__).with_name("speech_model_catalog.json").read_text())


def download(model_id, destination, *, progress=None, cancelled=None):
    record = next((r for r in catalog() if r["id"] == model_id), None)
    if record is None:
        raise ValueError("Choose a speech model from the verified catalog")
    root = Path(destination).expanduser().resolve() / model_id
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".setup-lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError("A speech setup is already using this folder") from None
    try:
        files = [DownloadFile(**f) for f in record["files"]]
        remaining = 0
        for item in files:
            target = root / item.target
            verified = (
                target.is_file()
                and target.stat().st_size == item.size
                and _hash(target, cancelled or (lambda: False)) == item.sha256
            )
            if not verified:
                remaining += item.size
        if shutil.disk_usage(root).free < remaining + 128 * 1024**2:
            raise ValueError("Not enough space for the verified speech model")
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

            download_file(
                item, root / item.target, progress=report, cancelled=cancelled or (lambda: False)
            )
        pending = root / "weetodd-model-source.pending.json"
        pending.write_text(json.dumps(record, indent=2))
        pending.replace(root / "weetodd-model-source.json")
        return dict(model_path=str(root), engine=record["engine"], license=record["license"])
    finally:
        lock.close()
