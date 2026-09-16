"""Bounded, strict JSON import. Never fetch schemas, files or code from a definition."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

MAX_DOCUMENT_BYTES = 2 * 1024 * 1024


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


def load_document(path: str | Path) -> dict:
    with Path(path).open("rb") as source:
        data = source.read(MAX_DOCUMENT_BYTES + 1)
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ValueError("Workflow definitions must be at most 2 MiB")
    try:
        value = json.loads(data, object_pairs_hook=_pairs, parse_constant=_constant)
        check_json(value)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Invalid or excessively nested workflow JSON") from error
    if not isinstance(value, dict):
        raise ValueError("A workflow or adapter definition must be a JSON object")
    return value
