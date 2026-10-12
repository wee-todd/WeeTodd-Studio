"""Pinned optional music weights; no model imports or downloads at module import."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .model_downloads import DownloadFile, download_file


def catalog():
    return json.loads(Path(__file__).with_name("music_extension_catalog.json").read_text())


def inspect(path, kind, decoder_path=None):
    root = Path(path).expanduser().resolve()
    if kind == "instrumental":
        if root.is_dir():
            candidates = [
                root / name
                for name in (
                    "ar_lora_inst_v3abc.bf16.safetensors",
                    "ar_lora_inst_v3abc.safetensors",
                )
            ]
            root = next((x for x in candidates if x.is_file()), root)
        from yue2_mlx.lora import inspect_lora

        info = inspect_lora(root)
        name = "YuE2 Instrumental · " + root.stem
    elif kind == "sheetsage":
        from sheetsage2_mlx import inspect_checkpoint

        info = inspect_checkpoint(root)
        name = "SheetSage2 · " + root.stem
    elif kind == "generator":
        from yue2_mlx.checkpoint import inspect_checkpoint

        info = inspect_checkpoint(root, vae_path=decoder_path)
        name = "YuE2 · " + root.name
    else:
        raise ValueError("Choose a supported music model kind.")
    return dict(
        model=dict(path=str(root), kind=kind, name=name, decoder_path=decoder_path), inspection=info
    )


def download(destination, kind, *, progress=None, cancelled=None):
    record = next((x for x in catalog() if x["kind"] == kind), None)
    if record is None:
        raise ValueError("Choose Instrumental LoRA or SheetSage2 to download.")
    root = Path(destination).expanduser().resolve() / record["id"]
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".setup-lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError("Another setup owns this folder. Wait for it to finish.") from None
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(str(os.getpid()))
        files = [DownloadFile(**spec) for spec in record["files"]]
        if (
            shutil.disk_usage(root).free
            < sum(x.size for x in files if not (root / x.target).exists()) + 128 * 1024**2
        ):
            raise ValueError("Not enough free space for the selected music model.")
        for index, spec in enumerate(files):

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
                spec, root / spec.target, progress=report, cancelled=cancelled or (lambda: False)
            )
        (root / "weetodd-model-source.json").write_text(json.dumps(record, indent=2))
        return inspect(root / files[0].target, kind)
    finally:
        lock.unlink(missing_ok=True)
