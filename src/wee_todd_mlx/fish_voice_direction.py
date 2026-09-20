"""Lightweight Fish delivery tags; no inference imports or spoken-script mutation."""

from __future__ import annotations

import unicodedata


def normalize_voice_direction(value) -> list[str]:
    """Validate the optional wire field and return an independent, trimmed tag list."""
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 8:
        raise ValueError("Fish voice direction must be a list of at most 8 tags")
    normalized = []
    for tag in value:
        if not isinstance(tag, str) or any(
            char in "[]<>" or unicodedata.category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
            for char in tag
        ):
            raise ValueError("Voice direction tags cannot contain brackets or control characters")
        tag = tag.strip()
        if not 1 <= len(tag) <= 120:
            raise ValueError("Each voice direction tag must contain 1 to 120 Unicode characters")
        normalized.append(tag)
    return normalized


def conditioned_text(text: str, direction=None) -> str:
    """Prefix Fish's target text only, preserving existing inline tags and script spacing."""
    tags = normalize_voice_direction(direction)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a speech script of at most 32,000 UTF-8 bytes")
    result = " ".join(f"[{tag}]" for tag in tags) + " " + text if tags else text
    try:
        size = len(result.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ValueError("Enter a speech script with valid Unicode characters") from error
    if size > 32000:
        raise ValueError(
            "Speech script including voice direction must be at most 32,000 UTF-8 bytes"
        )
    return result
