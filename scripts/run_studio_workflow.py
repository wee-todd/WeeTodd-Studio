#!/usr/bin/env python3
"""Run/resume a local Studio workflow job. Uses referenced models/media in place."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wee_todd_mlx.workflows.io import load_document  # noqa: E402
from wee_todd_mlx.workflows.service import dispatch  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "request",
        type=Path,
        help="JSON job with definition/builtin, inputs, bindings and runDirectory",
    )
    parser.add_argument(
        "--next", action="store_true", help="Run the next incomplete step and checkpoint"
    )
    parser.add_argument("--regenerate", help="Invalidate this step and its descendants")
    parser.add_argument(
        "--review",
        type=Path,
        help="JSON review action with stepID, expectedRevision, action and optional itemID/outputs",
    )
    args = parser.parse_args(argv)
    stopped = False

    def stop(_number, _frame):
        nonlocal stopped
        stopped = True

    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, stop)
    try:
        request = load_document(args.request)
        if args.next:
            request["maxSteps"] = 1
        if args.regenerate:
            request["regenerate"] = args.regenerate
        result = dispatch(
            "workflow-review" if args.review else "workflow-run",
            {**request, "review": load_document(args.review)} if args.review else request,
            cancelled=lambda: stopped,
            progress=lambda event: print(json.dumps({"event": "progress", **event}), flush=True),
        )
        print(json.dumps({"event": "result", "result": result}, allow_nan=False), flush=True)
        return 1 if result["status"] == "failed" else 130 if result["status"] == "cancelled" else 0
    except (ValueError, OSError) as error:
        print(json.dumps({"event": "error", "message": str(error)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
