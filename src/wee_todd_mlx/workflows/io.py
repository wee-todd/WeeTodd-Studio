"""Bounded, strict JSON import. Never fetch schemas, files or code from a definition."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024
MAX_TURN_LOG_BYTES = 16 * 1024 * 1024
TURN_LOG_OVERHEAD_BYTES = 64 * 1024


def storage_budgets(limits: dict) -> tuple[int, int]:
    """Reserve old/new checkpoints and SQLite/rollback storage without overlap.

    Small jobs retain their original 2 MiB checkpoint allowance when affordable;
    larger jobs split the declared storage evenly, up to each file's hard cap.
    The history allocation includes conservative transaction/header overhead.
    """
    working = limits.get("maxWorkingBytes") if isinstance(limits, dict) else None
    if type(working) is not int or working < 1:
        raise ValueError("Workflow storage budget must be a positive integer")
    checkpoint = min(MAX_CHECKPOINT_BYTES, working // 2,
                     max(MAX_DOCUMENT_BYTES, working // 4))
    history = min(MAX_TURN_LOG_BYTES, (working - 2 * checkpoint) // 2)
    return checkpoint, history


def check_json(value: Any, depth: int = 0) -> None:
    if depth > 64:
        raise ValueError("Workflow JSON nesting exceeds 64 levels")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            check_json(item, depth + 1)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            check_json(item, depth + 1)
        return
    raise ValueError("Workflow definitions must contain finite JSON values")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(_value):
    raise ValueError("JSON numbers must be finite")


def _decode(data: bytes) -> dict:
    try:
        value = json.loads(data, object_pairs_hook=_pairs, parse_constant=_constant)
        check_json(value)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Invalid or excessively nested workflow JSON") from error
    if not isinstance(value, dict):
        raise ValueError("A workflow or adapter definition must be a JSON object")
    return value


def load_document(path: str | Path) -> dict:
    with Path(path).open("rb") as source:
        data = source.read(MAX_DOCUMENT_BYTES + 1)
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ValueError("Workflow definitions must be at most 2 MiB")
    return _decode(data)


def load_checkpoint(path: str | Path, *, limits: dict | None = None) -> dict:
    """Read only resume state with the larger cap, preserving strict JSON checks."""
    source = Path(path)
    if source.is_symlink():
        raise ValueError("Workflow checkpoint cannot be a symlink")
    cap = storage_budgets(limits)[0] if limits is not None else MAX_CHECKPOINT_BYTES
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        data = stream.read(cap + 1)
    if len(data) > cap:
        raise ValueError("Workflow checkpoint exceeds its JSON/disk budget (at most 16 MiB)")
    value = _decode(data)
    definition = value.get("definition", {})
    if not isinstance(definition, dict):
        raise ValueError("Workflow checkpoint definition must be a JSON object")
    saved_limits = definition.get("limits")
    if saved_limits is not None and len(data) > storage_budgets(saved_limits)[0]:
        raise ValueError("Workflow checkpoint exceeds its saved disk budget")
    return value
