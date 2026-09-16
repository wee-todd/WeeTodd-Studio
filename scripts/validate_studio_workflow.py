#!/usr/bin/env python3
"""Validate a workflow or adapter definition without running it or loading a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wee_todd_mlx.workflows import load_document, validate_file  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("definition", type=Path)
    parser.add_argument("--adapter", type=Path, action="append", default=[])
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--require-runnable", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = validate_file(args.definition, adapters=[load_document(p) for p in args.adapter])
    except (OSError, ValueError, RecursionError) as error:
        report = {
            "valid": False,
            "executionStatus": "invalid",
            "topologicalOrder": [],
            "issues": [{"code": "import", "path": "$", "message": str(error)}],
            "warnings": [],
        }
    if args.as_json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print(f"Definition: {'valid' if report['valid'] else 'invalid'}")
        print(f"Execution: {report['executionStatus']}")
        if report["topologicalOrder"]:
            print("Step order: " + " → ".join(report["topologicalOrder"]))
        for level in ("issues", "warnings"):
            for item in report[level]:
                print(f"{level}: {item['path']}: {item['message']} ({item['code']})")
    if not report["valid"]:
        return 1
    return 2 if args.require_runnable and report["executionStatus"] != "ready" else 0


if __name__ == "__main__":
    raise SystemExit(main())
