"""Strict validation for review-only character extraction proposals."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

ROLES = frozenset({"character", "style", "text"})
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RUNTIME_TOKEN = re.compile(r"<\|(?:im_start|im_end|endoftext)\|>", re.IGNORECASE)
_MEASUREMENT = re.compile(
    r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s+(?:mm|cm|m|in|ft)$",
    re.IGNORECASE,
)


def field_id(metadata: Mapping[str, Any]) -> str:
    value = metadata.get("id", metadata.get("key"))
    if not isinstance(value, str) or not value:
        raise ValueError("Each catalog field needs a nonempty id or key")
    return value


def metadata_for(path: str, catalog: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Resolve an exact path or a repeatable template such as garments[].type."""
    for metadata in catalog:
        template = field_id(metadata)
        if template == path:
            return metadata
        pattern = re.escape(template).replace(r"\[\]", r"\[[^\[\]]+\]")
        if "[]" in template and re.fullmatch(pattern, path):
            return metadata
    return None


def _bounded_text(value: Any, *, name: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Proposal {name} must be text")
    if len(value) > limit:
        raise ValueError(f"Proposal {name} exceeds {limit} characters")
    if _CONTROL.search(value) or _RUNTIME_TOKEN.search(value):
        raise ValueError(f"Proposal {name} contains control tokens")
    return value


def _validate_value(value: Any, metadata: Mapping[str, Any]) -> Any:
    limit = metadata.get("maxLength", 240)
    if not isinstance(limit, int) or limit < 1 or limit > 4000:
        limit = 240
    result = _bounded_text(value, name="value", limit=limit)
    if not result.strip():
        raise ValueError("Proposal value must not be blank")
    kind = metadata.get("valueKind", metadata.get("kind", "text"))
    if kind == "measurement" and not _MEASUREMENT.fullmatch(result.strip()):
        raise ValueError("Proposal measurement value must include a number and unit")
    if kind == "color" and (
        result.lstrip().startswith("#")
        or re.match(r"^(?:rgb|hsl)a?\s*\(", result.strip(), re.IGNORECASE)
    ):
        raise ValueError("Proposal color value must be a bounded color name")
    return result


def _validate_evidence(value: Any) -> Any:
    if isinstance(value, str):
        return _bounded_text(value, name="evidence", limit=400)
    if not isinstance(value, dict) or len(value) > 8:
        raise ValueError("Proposal evidence must be text or a bounded record")
    allowed = {"summary", "visibility", "confidence"}
    if not set(value) <= allowed:
        raise ValueError("Proposal evidence contains unknown fields")
    result = {}
    for key, item in value.items():
        if key == "confidence":
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not 0 <= item <= 1:
                raise ValueError("Proposal evidence confidence must be between 0 and 1")
            result[key] = item
        else:
            result[key] = _bounded_text(item, name=f"evidence {key}", limit=400)
    return result


def validate_proposals(
    value: dict,
    *,
    role: str,
    allowed_fields: set[str],
    catalog: Iterable[Mapping[str, Any]] | None = None,
) -> dict:
    """Validate one model response atomically and return normalized proposals."""
    if role not in ROLES:
        raise ValueError("Choose a valid extraction role")
    if any("[]" in path for path in allowed_fields):
        raise ValueError("Proposal fields require an exact host-created record path")
    if not isinstance(value, dict) or set(value) != {"proposals"}:
        raise ValueError("Extraction JSON must contain only proposals")
    proposals = value["proposals"]
    if not isinstance(proposals, list) or len(proposals) > 64:
        raise ValueError("Extraction proposals must be a list of at most 64 items")
    catalog = tuple(catalog or ())
    normalized = []
    seen = set()
    for index, proposal in enumerate(proposals):
        if not isinstance(proposal, dict):
            raise ValueError(f"Proposal {index} must be an object")
        permitted = {"id", "field", "value", "state", "evidence", "uncertainty"}
        required = {"field", "value", "evidence", "uncertainty"}
        if not set(proposal) <= permitted or not required <= set(proposal):
            raise ValueError(f"Proposal {index} has an invalid schema")
        path = proposal["field"]
        if isinstance(path, str) and "[]" in path:
            raise ValueError("Proposal fields require an exact host-created record path")
        if not isinstance(path, str) or path not in allowed_fields:
            raise ValueError(f"Proposal field is not allowed for {role}: {path!r}")
        if path in seen:
            raise ValueError(f"Proposal field appears more than once: {path}")
        seen.add(path)
        field_metadata = metadata_for(path, catalog) or {
            "id": path,
            "kind": "text",
            "maxLength": 240,
        }
        if role != "text" and (
            field_metadata.get("authoredOnly") is True
            or path
            in {
                "identity.authoredAge",
                "identity.authoredSexGender",
                "identity.authoredAncestry",
            }
        ):
            raise ValueError(f"Authored-only field cannot come from image analysis: {path}")
        state = proposal.get("state", "value")
        if state != "value":
            raise ValueError("Populated proposal state must be value")
        evidence = _validate_evidence(proposal["evidence"])
        uncertainty = _bounded_text(proposal["uncertainty"], name="uncertainty", limit=400)
        item = {
            "field": path,
            "value": _validate_value(proposal["value"], field_metadata),
            "state": state,
            "evidence": evidence,
            "uncertainty": uncertainty,
        }
        identity = json.dumps(item, sort_keys=True, ensure_ascii=False, allow_nan=False)
        item["id"] = proposal.get("id") or hashlib.sha256(identity.encode()).hexdigest()[:20]
        if not isinstance(item["id"], str) or not item["id"] or len(item["id"]) > 128:
            raise ValueError("Proposal id must be bounded text")
        normalized.append(item)
    return {"proposals": normalized}
