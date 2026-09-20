"""Pinned identity for the metadata-free author Ripple v11 IC adapter.

No model or MLX imports: a compatible rank or a filename cannot establish its task.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

RIPPLE_REPOSITORY = "WepeNerd/LTX-Ripple"
RIPPLE_REVISION = "8658110161061ab50c931ff633c9c5ead8dcebfa"
RIPPLE_SHA256 = "bde543610144e5bba78782fb0ac18daa663fc087c2212244b8116deb1925ee2a"
RIPPLE_BYTES = 654443392


@lru_cache(maxsize=16)
def _digest(filename, device, inode, size, mtime_ns, ctime_ns):
    del device, inode, size, mtime_ns, ctime_ns
    digest = hashlib.sha256()
    with Path(filename).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_verified_ripple(path: str | Path) -> bool:
    source = Path(path).expanduser().resolve()
    stat = source.stat()
    if stat.st_size != RIPPLE_BYTES:
        return False
    signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    digest = _digest(str(source), *signature)
    after = source.stat()
    if signature != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("Ripple adapter changed while its identity was being verified.")
    return digest == RIPPLE_SHA256
