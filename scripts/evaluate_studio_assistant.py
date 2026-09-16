#!/usr/bin/env python3
"""Validate curated assistant probes, or explicitly run them against a local Qwen helper.

Default invocation is model-free. --run requires existing model/helper paths and a new report
directory. Tests probe small tasks, not subjective story quality or the whole movie workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wee_todd_mlx.assistant_evaluation import evaluate_response, load_suite, summarize  # noqa: E402
from wee_todd_remote.client import invoke_helper  # noqa: E402


def file_identity(source):
    info = source.stat()
    return {"path": str(source.resolve()), "bytes": info.st_size, "modifiedNS": info.st_mtime_ns}


def file_digest(source):
    with source.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def output_shape(schema):
    """Only schema types, never expected answers or source facts."""
    kind = schema.get("type")
    if kind == "object":
        return {key: output_shape(value) for key, value in schema.get("properties", {}).items()}
    if kind == "array":
        return [output_shape(schema["items"])]
    return {"string": "<string>", "boolean": "<boolean>", "integer": "<integer>",
            "number": "<number>", "null": None}.get(kind, "<value>")


def system_instruction(case, prompt_format):
    if prompt_format == "schema":
        return case["system"] + " Return only JSON matching this schema: " + json.dumps(
            case["schema"], separators=(",", ":"))
    return case["system"] + (
        " Return only the populated JSON answer. Output shape (replace type placeholders "
        "with your answer; arrays may have zero or more entries): "
    ) + json.dumps(output_shape(case["schema"]), separators=(",", ":"))


def create_image(spec, destination):
    from PIL import Image, ImageDraw

    colors = {"red": "#ed2727", "blue": "#234fef", "green": "#00b248", "yellow": "#f3dc17"}
    if spec["color"] not in colors or spec["shape"] not in {"circle", "square", "triangle"}:
        raise ValueError("Unsupported synthetic image fixture")
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    fill = colors[spec["color"]]
    if spec["shape"] == "circle":
        draw.ellipse((48, 48, 208, 208), fill=fill)
    elif spec["shape"] == "square":
        draw.rectangle((48, 48, 208, 208), fill=fill)
    else:
        draw.polygon(((128, 35), (222, 216), (34, 216)), fill=fill)
    image.save(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path,
                        default=ROOT / "examples/assistant-evaluation/cases.json")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--helper", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("all", "development", "holdout"), default="all")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--prompt-format", choices=("schema", "shape"), default="schema")
    args = parser.parse_args(argv)
    cases = load_suite(args.suite)
    corpus_revision = json.loads(args.suite.read_text()).get("corpusRevision", 1)
    if (not 1 <= args.limit <= 200 or not 1 <= args.max_tokens <= 1024
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("Use limit1–200, max-tokens1–1024 and a positive timeout")
    cases = [c for c in cases if args.split == "all" or c["split"] == args.split][:args.limit]
    if not args.run:
        print(json.dumps({"status": "validated", "cases": len(cases),
                          "categories": sorted({c["category"] for c in cases}),
                          "corpusRevision": corpus_revision, "modelCalls": 0}, indent=2))
        return 0
    if not args.model or not args.helper or not args.output:
        parser.error("--run requires --model, --helper and --output")
    model, helper = args.model.resolve(strict=True), args.helper.resolve(strict=True)
    metadata = {
        "format": "weetodd-assistant-evaluation-run-v1",
        "suiteSHA256": file_digest(args.suite),
        "model": file_identity(model), "helper": file_identity(helper),
        "helperSHA256": file_digest(helper), "harnessSHA256": file_digest(Path(__file__)),
        "machine": platform.machine(), "platform": platform.platform(),
        "maxTokens": args.max_tokens, "split": args.split,
        "promptFormat": args.prompt_format,
        "corpusRevision": corpus_revision,
        "lifecycle": "request-owned-process; OS file cache may warm between cases",
        "qualification": "bounded task probes; not full movie or hardware qualification",
    }
    sidecar = Path(str(model) + "-tensordata")
    if sidecar.exists():
        metadata["modelSidecar"] = file_identity(sidecar)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "suite.json").write_bytes(args.suite.read_bytes())
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows, failures = [], 0
    with (args.output / "responses.jsonl").open("x") as log:
        for case in cases:
            request = {
                "requestID": str(uuid4()), "modelPath": str(model),
                "systemPrompt": system_instruction(case, args.prompt_format),
                "prompt": case["prompt"], "maxTokens": args.max_tokens, "images": [],
            }
            if spec := case.get("image"):
                image_path = args.output / f"image-{len(rows):03d}.png"
                create_image(spec, image_path)
                request["images"] = [{"path": str(image_path.resolve()), "label": "Test image"}]
            row = {k: case[k] for k in ("id", "category", "group", "split")}
            start = time.monotonic()
            try:
                response = None
                for event in invoke_helper("text", request, helper=helper,
                                           cancelled=lambda: False, timeout=args.timeout):
                    if event["type"] == "result":
                        response = event["value"]
                if response is None:
                    raise RuntimeError("Helper exited without a result")
                row.update(evaluate_response(case, response), response=response)
                failures = 0
            except (RuntimeError, ValueError, OSError, TimeoutError) as error:
                row.update(passed=False, structuralValid=False, error=str(error))
                failures += 1
            row["wallSeconds"] = time.monotonic() - start
            rows.append(row)
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            print(json.dumps({"case": row["id"], "passed": row["passed"],
                              "seconds": round(row["wallSeconds"], 3)}), flush=True)
            if failures >= 3:
                break
    summary = {**summarize(rows), "planned": len(cases), "completed": len(rows) == len(cases)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
