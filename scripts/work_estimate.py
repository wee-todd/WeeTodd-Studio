#!/usr/bin/env python3
"""Report explicit, evidence-backed estimates and measured intervals.

Input JSON has original_scope (immutable task IDs), tasks, optional intervals and
original_forecast. Keep successive snapshots outside the repository. This is
arithmetic/accounting, not a learned predictor or a probability interval.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def bounds(value, label):
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not 0 <= x < float("inf")
            for x in value
        )
        or value[0] > value[1]
    ):
        raise ValueError(f"{label} requires finite nonnegative lower/upper seconds")
    return list(value)


def compare_baseline(document, baseline):
    """Reject changed original scope/forecast against a separately retained baseline."""
    for key in ("original_scope", "original_forecast"):
        if document.get(key) != baseline.get(key):
            raise ValueError(f"Original {key} differs from baseline; report new scope separately")


def summarize(document):
    original = document["original_scope"]
    tasks = document["tasks"]
    ids = [t["id"] for t in tasks]
    if not original or any(not isinstance(x, str) or not x for x in original + ids):
        raise ValueError("Scope requires nonempty task IDs")
    if (
        len(set(ids)) != len(ids)
        or len(set(original)) != len(original)
        or not set(original) <= set(ids)
    ):
        raise ValueError("Duplicate tasks or original scope removed; preserve original IDs")
    work, wait, unknown = [0, 0], [0, 0], []
    done = set()
    for t in tasks:
        if t["status"] not in {"pending", "active", "done", "blocked"}:
            raise ValueError("Unsupported task status")
        if not isinstance(t.get("basis"), str) or not t["basis"].strip():
            raise ValueError("Every estimate requires its basis, including unknowns")
        w, q = bounds(t.get("work_seconds"), t["id"]), bounds(t.get("wait_seconds"), t["id"])
        if t["status"] == "done":
            if not isinstance(t.get("evidence"), str) or not t["evidence"].strip():
                raise ValueError("Completion requires a verification evidence reference")
            done.add(t["id"])
            continue
        if w is None or q is None:
            unknown.append(t["id"])
        else:
            work = [a + b for a, b in zip(work, w, strict=True)]
            wait = [a + b for a, b in zip(wait, q, strict=True)]
    return dict(
        original_checks_done=len(done & set(original)),
        original_checks_total=len(original),
        added_scope=[x for x in ids if x not in original],
        completed_added_checks=len(done - set(original)),
        unknown_tasks=unknown,
        remaining_work_seconds=None if unknown else work,
        remaining_wait_seconds=None if unknown else wait,
        serial_remaining_seconds=None
        if unknown
        else [a + b for a, b in zip(work, wait, strict=True)],
        time_confidence="not statistically calibrated",
        elapsed_assumption=("Serial work plus external waits; overlap can reduce elapsed time."),
        progress_scope="Verified checklist counts, not effort or full-product completion.",
        original_forecast=document.get("original_forecast"),
    )


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Intervals require timezone-qualified timestamps")
    return parsed.timestamp()


def union_seconds(intervals):
    end = None
    total = 0
    for a, b in sorted(intervals):
        total += max(0, b - max(a, end if end is not None else a))
        end = max(b, end if end is not None else b)
    return total


def actuals(records):
    groups = {kind: [] for kind in ["work", "wait", "rework"]}
    all_intervals = []
    for r in records:
        if r["kind"] not in groups or not r.get("evidence"):
            raise ValueError("Measured intervals require work/wait/rework and evidence")
        a, b = timestamp(r["start"]), timestamp(r["end"])
        if b < a:
            raise ValueError("Interval ends before it starts")
        groups[r["kind"]].append((a, b))
        all_intervals.append((a, b))
    wall = max((b for _, b in all_intervals), default=0) - min(
        (a for a, _ in all_intervals), default=0
    )
    result = {kind + "_union_seconds": union_seconds(rows) for kind, rows in groups.items()}
    result.update(
        wall_seconds=wall,
        uncovered_seconds=wall - union_seconds(all_intervals),
        measurement_scope=(
            "Recorded wall intervals, not CPU time or billable effort. "
            "Categories may overlap; do not add them."
        ),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--baseline", type=Path, help="Retained original JSON snapshot")
    args = parser.parse_args()
    try:
        document = json.loads(args.manifest.read_text())
        if args.baseline:
            compare_baseline(document, json.loads(args.baseline.read_text()))
        report = summarize(document)
        report["actuals"] = actuals(document.get("intervals", []))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
