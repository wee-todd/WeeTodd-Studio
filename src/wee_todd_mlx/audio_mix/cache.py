"""Bounded disposable PCM cache. Shared leases protect readers from LRU eviction."""

import fcntl
import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from wee_todd_mlx.speech_reference import digest_file


@contextmanager
def locked(filename, cancelled, *, shared=False):
    with Path(filename).open("a") as stream:
        while True:
            if cancelled():
                raise InterruptedError("Audio preparation cancelled")
            try:
                fcntl.flock(stream, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.025)
        try:
            yield stream
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def decoded(root, identity, build, cancelled):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    pcm, receipt = root / (key + ".pcm"), root / (key + ".json")
    with locked(root / (key + ".lock"), cancelled, shared=True):
        with locked(root / (key + ".build"), cancelled):
            valid = False
            if pcm.is_file() and receipt.is_file():
                try:
                    saved = json.loads(receipt.read_text())
                    valid = saved["identity"] == identity and saved["sha256"] == digest_file(pcm)
                except (OSError, ValueError, KeyError):
                    pass
            if not valid:
                with tempfile.TemporaryDirectory(dir=root, prefix="decode-") as temp:
                    pending = Path(temp) / "source.pcm"
                    build(pending)
                    meta = Path(temp) / "source.json"
                    meta.write_text(
                        json.dumps(dict(identity=identity, sha256=digest_file(pending)))
                    )
                    os.replace(pending, pcm)
                    os.replace(meta, receipt)
            os.utime(pcm, None)
        yield pcm
    prune(root)


def prune(root, budget=1024**3):
    entries = []
    for p in root.glob("*.pcm"):
        try:
            stat = p.stat()
            entries.append((stat.st_mtime, stat.st_size, p))
        except FileNotFoundError:
            continue
    files = [p for _, _, p in sorted(entries)]
    total = sum(size for _, size, _ in entries)
    for pcm in files:
        if total <= budget:
            break
        with pcm.with_suffix(".lock").open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if pcm.exists():
                total -= pcm.stat().st_size
                pcm.unlink()
                pcm.with_suffix(".json").unlink(missing_ok=True)


def prune_previews(root, *, budget=2 * 1024**3, grace=120):
    """Only disposable preview roots call this; driver/export artifacts are durable."""
    root = Path(root)
    with locked(root / "eviction.lock", lambda: False):
        candidates = []
        for wav in root.glob("*/mix.wav"):
            try:
                candidates.append((wav.stat().st_mtime, wav.stat().st_size, wav))
            except FileNotFoundError:
                continue
        total = sum(size for _, size, _ in candidates)
        for accessed, size, wav in sorted(candidates):
            if total <= budget:
                break
            if time.time() - accessed < grace:
                continue
            with (
                (wav.parent / "lease.lock").open("a") as lease,
                (wav.parent / "build.lock").open("a") as build,
            ):
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(build, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                wav.unlink(missing_ok=True)
                (wav.parent / "mix.json").unlink(missing_ok=True)
                total -= size
