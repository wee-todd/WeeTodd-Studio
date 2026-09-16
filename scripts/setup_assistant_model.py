#!/usr/bin/env python3
"""Install or explicitly check local Qwen assistance without the Draw Things app."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    from studio_assistant_models import dispatch

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("catalog")
    commands.add_parser("inspect").add_argument("path")
    commands.add_parser("download").add_argument("--destination", required=True)
    health = commands.add_parser("health")
    health.add_argument("path")
    health.add_argument("--helper", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    args["runtime"] = {"drawThingsHelperPath": args.pop("helper", "")}
    result = dispatch(
        "assistant-model-" + command,
        args,
        progress=lambda message, fraction: print(message, file=sys.stderr, flush=True),
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Cancelled; retry the same destination to resume.", file=sys.stderr)
        raise SystemExit(130) from None
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None
