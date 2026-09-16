"""Bounded local model checksum cache keyed by full filesystem identity.

Only immutable model-file checks use this cache. Generated continuation manifests and
payloads still receive a complete hash check on every load.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path


def _identity(source):
    stat = source.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _database():
    if sys.platform == "darwin":
        root = Path.home() / "Library/Caches/WeeTodd Studio"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "weetodd"
    return root / "model-sha256.sqlite3"


def _connect(database):
    database.parent.mkdir(parents=True, exist_ok=True)
    if database.exists() and database.stat().st_size > 16 * 1024**2:
        raise OSError("Model hash cache exceeds its 16 MiB limit")
    db = sqlite3.connect(database, timeout=0.1)
    try:
        db.execute("CREATE TABLE IF NOT EXISTS model_hashes "
                   "(path TEXT PRIMARY KEY, identity TEXT, sha256 TEXT, used INTEGER)")
    except sqlite3.Error:
        db.close()
        raise
    return db


def cached_model_hash(source, compute, *, database=None, max_entries=10000):
    """Reuse a checksum only while device/inode/size/mtime/ctime all match.

    Read-only, unavailable or corrupt caches fall back to complete hashing. Disable
    reuse with WEETODD_DISABLE_MODEL_HASH_CACHE=1 for a full re-verification run.
    """
    source = Path(source).expanduser().resolve(strict=True)
    expected = _identity(source)
    signature = json.dumps(expected)
    database = Path(database) if database is not None else _database()
    enabled = os.environ.get("WEETODD_DISABLE_MODEL_HASH_CACHE") != "1"
    record = None
    if enabled:
        try:
            db = _connect(database)
            try:
                record = db.execute("SELECT identity, sha256 FROM model_hashes WHERE path=?",
                                    (str(source),)).fetchone()
            finally:
                db.close()
        except (sqlite3.Error, OSError):
            pass
    if (record and record[0] == signature and isinstance(record[1], str)
            and re.fullmatch(r"[0-9a-f]{64}", record[1])):
        result = record[1]
    else:
        result = compute(source)
    if _identity(source) != expected:
        raise ValueError(f"Model file changed while verifying its checksum: {source}")
    if not isinstance(result, str) or not re.fullmatch(r"[0-9a-f]{64}", result):
        raise ValueError("Model checksum must be a complete SHA-256 digest")
    if enabled:
        try:
            db = _connect(database)
            try:
                with db:
                    db.execute("INSERT OR REPLACE INTO model_hashes VALUES (?, ?, ?, ?)",
                               (str(source), signature, result, time.time_ns()))
                    db.execute("DELETE FROM model_hashes WHERE path IN "
                               "(SELECT path FROM model_hashes ORDER BY used DESC "
                               "LIMIT -1 OFFSET ?)",
                               (max_entries,))
            finally:
                db.close()
        except (sqlite3.Error, OSError):
            pass
    return result
