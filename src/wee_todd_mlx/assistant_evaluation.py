"""Small, explicit task probes; these scores do not certify complete movie quality."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from jsonschema import Draft202012Validator


def _pointer(value, pointer):
    if pointer == "":
        return value
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("Assertions require a JSON pointer")
    for key in pointer[1:].split("/"):
        key = key.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def load_suite(source: Path) -> list[dict]:
    if source.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Evaluation suite exceeds 2 MiB")
    document = json.loads(source.read_text())
    if document.get("format") != "weetodd-assistant-evaluation-v1":
        raise ValueError("Unsupported evaluation format")
    cases = document.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 200:
        raise ValueError("Evaluation needs 1–200 cases")
    seen, groups = set(), {}
    for item in cases:
        for field in ("id", "category", "group", "split", "system", "prompt"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"Case needs {field}")
        if item["id"] in seen:
            raise ValueError("Duplicate evaluation ID")
        seen.add(item["id"])
        if item["split"] not in {"development", "holdout"}:
            raise ValueError("Unsupported evaluation split")
        if groups.setdefault(item["group"], item["split"]) != item["split"]:
            raise ValueError("A source group cannot appear in multiple splits")
        schema = item.get("schema", document.get("schemas", {}).get(item.get("schemaID")))
        if not isinstance(schema, dict) or '"$ref"' in json.dumps(schema):
            raise ValueError("Use a self-contained evaluation schema")
        Draft202012Validator.check_schema(schema)
        item["schema"] = schema
        assertions = item.get("assertions")
        if not isinstance(assertions, list) or not assertions:
            raise ValueError("Each case needs independent expected assertions")
        for check in assertions:
            if set(check) != {"path", "equals"}:
                raise ValueError("Use path/equals assertions")
            if not isinstance(check["path"], str) or (
                check["path"] and not check["path"].startswith("/")
            ):
                raise ValueError("Assertions require a JSON pointer")
    return cases


def evaluate_response(case: dict, result: dict) -> dict:
    verdict = {"passed": False, "structuralValid": False, "failedAssertions": []}
    if result.get("truncated"):
        return {**verdict, "error": "truncated"}
    try:
        value = json.loads(result["text"])
    except (KeyError, TypeError, ValueError):
        return {**verdict, "error": "invalid_json"}
    if list(Draft202012Validator(case["schema"]).iter_errors(value)):
        return {**verdict, "error": "invalid_schema"}
    verdict["structuralValid"] = True
    for check in case["assertions"]:
        try:
            actual = _pointer(value, check["path"])
            # JSON booleans are not interchangeable with numeric choices.
            matches = actual == check["equals"] and type(actual) is type(check["equals"])
        except (KeyError, IndexError, TypeError, ValueError):
            matches = False
        if not matches:
            verdict["failedAssertions"].append(check["path"])
    verdict["passed"] = not verdict["failedAssertions"]
    return verdict


def summarize(rows: list[dict]) -> dict:
    result = {
        "attempted": len(rows), "passed": sum(bool(r.get("passed")) for r in rows),
        "structuralValid": sum(bool(r.get("structuralValid")) for r in rows), "categories": {},
    }
    result["passRate"] = result["passed"] / len(rows) if rows else None
    times = sorted(r["wallSeconds"] for r in rows if isinstance(r.get("wallSeconds"),
                                                               (int, float)))
    result["wallSeconds"] = {
        "total": sum(times), "p50": statistics.median(times) if times else None,
        "p95": times[max(0, math.ceil(len(times) * 0.95) - 1)] if times else None,
    }
    for row in rows:
        category = result["categories"].setdefault(row["category"], {"attempted": 0, "passed": 0})
        category["attempted"] += 1
        category["passed"] += bool(row.get("passed"))
    return result
