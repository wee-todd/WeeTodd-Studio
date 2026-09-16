#!/usr/bin/env python3
"""Frozen, bounded evaluation of the real guided Director runner. Default: model-free.

Weighted runs use --run, preserve all transcripts externally, and simulate explicit review
choices. A recorded production 'human' event is an evaluation driver action, never evidence
of human acceptance. --source-root imports before/after engines in separate processes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "examples/director-production-evaluation/sessions.json"


def digest_file(source):
    with Path(source).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(destination, value):
    Path(destination).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_suite(source, expected_digest=None):
    source = Path(source)
    if source.stat().st_size > 1024 * 1024:
        raise ValueError("Suite exceeds 1 MiB")
    if expected_digest is not None and digest_file(source) != expected_digest:
        raise ValueError("Frozen suite digest mismatch")
    suite = json.loads(source.read_text())
    if suite.get("format") != "weetodd-director-production-evaluation-v1":
        raise ValueError("Unknown production evaluation format")
    sessions = suite.get("sessions", [])
    if not 1 <= len(sessions) <= 3:
        raise ValueError("Use 1–3 bounded production sessions")
    seen = set()
    for session in sessions:
        identifier = session["id"]
        if not identifier or set(identifier) - set("abcdefghijklmnopqrstuvwxyz0123456789-"):
            raise ValueError("Use simple unique session IDs")
        if identifier in seen:
            raise ValueError("Duplicate session ID")
        seen.add(identifier)
        inputs, expected = session["inputs"], session["expected"]
        shots = expected["shots"]
        if len(shots) not in (2, 3):
            raise ValueError("Use two or three authored shots")
        if inputs["duration_seconds"] != len(shots) * inputs["target_clip_seconds"]:
            raise ValueError("Fixture timing must match authored shots")
        for index, shot in enumerate(shots, 1):
            if f"[Shot {index}]" not in inputs["brief"]:
                raise ValueError("Fixture lacks an authored shot marker")
            if any(quote not in inputs["brief"] for quote in shot["dialogue"]):
                raise ValueError("Expected dialogue must be literal source text")
        if session["correction"]["fieldScope"] != "action":
            raise ValueError("This evaluation measures one action-field correction")
    return suite


def freeze_suite(source, destination):
    load_suite(source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "suite.json").write_bytes(Path(source).read_bytes())
    digest = digest_file(destination / "suite.json")
    write_json(
        destination / "fixture-freeze.json",
        {
            "sha256": digest,
            "frozenAt": time.time(),
            "policy": "Expected facts and text are frozen before any model call.",
        },
    )
    return digest


def step_output(state, step, port, default=None):
    return state.get("steps", {}).get(step, {}).get("outputs", {}).get(port, default)


def dialogue_violations(text, expected):
    """These fixtures reserve quoted spans for speech; contractions are not delimiters."""
    endings = {"“": "”", '"': '"', "‘": "’", "'": "'"}
    quoted, index = [], 0
    while index < len(text):
        char = text[index]
        if char not in endings or (char == "'" and index > 0 and text[index - 1].isalnum()):
            index += 1
            continue
        cursor = index + 1
        while cursor < len(text):
            if text[cursor] == endings[char]:
                contraction = (
                    text[cursor] in {"'", "’"}
                    and text[cursor - 1].isalnum()
                    and cursor + 1 < len(text)
                    and text[cursor + 1].isalnum()
                )
                if not contraction:
                    quoted.append(text[index + 1 : cursor])
                    break
            cursor += 1
        index = cursor + 1
    result = ["unexpected:" + words for words in dict.fromkeys(quoted) if words not in expected]
    result.extend("duplicate:" + words for words in expected if quoted.count(words) > 1)
    return result


def score_session(case, state):
    violations, measured = [], []
    brief = step_output(state, "creative_brief", "creative_brief")
    if brief is not None:
        measured.append("source")
        if brief.get("sourceText") != case["inputs"]["brief"]:
            violations.append("source.text")
        evidence = "\n".join(row["evidence"] for row in brief["facts"])
        if evidence != case["inputs"]["brief"]:
            violations.append("source.facts")
    subjects = step_output(state, "subjects_coverage", "subjects")
    if subjects is not None:
        measured.append("subjects")
        for expected in case["expected"]["subjects"]:
            matching = [
                row for row in subjects if row["name"].casefold() == expected["name"].casefold()
            ]
            if len(matching) != 1:
                violations.append("subjects.identity:" + expected["name"])
                continue
            row = matching[0]
            if row["kind"] != expected["kind"]:
                violations.append("subjects.kind:" + expected["name"])
            for term in expected["terms"]:
                if term.casefold() not in row["description"].casefold():
                    violations.append("subjects.fact:" + expected["name"] + ":" + term)
    story = step_output(state, "story", "story")
    plan = step_output(state, "clips", "clips")
    for stage, values in (
        ("story", story.get("beats") if story else None),
        ("clips", [row["action"] for row in plan["clips"]] if plan else None),
    ):
        if values is None:
            continue
        measured.append(stage)
        if len(values) != len(case["expected"]["shots"]):
            violations.append(stage + ".shot_count")
        for index, (expected, text) in enumerate(
            zip(case["expected"]["shots"], values, strict=False), 1
        ):
            for quote in expected["dialogue"]:
                if quote not in text:
                    violations.append(f"{stage}.shot-{index}.dialogue:{quote}")
            violations.extend(
                f"{stage}.shot-{index}.dialogue_{error}"
                for error in dialogue_violations(text, expected["dialogue"])
            )
            for term in expected["actionTerms"]:
                if term.casefold() not in text.casefold():
                    violations.append(f"{stage}.shot-{index}.action_term:{term}")
    if subjects is not None:
        cast = [
            {"id": row["id"], "description": row["description"]}
            for row in subjects
            if row["kind"] == "character"
        ]
        for stage, value in (("story", story), ("clips", plan)):
            if value is not None and value["characters"] != cast:
                violations.append(stage + ".approved_characters")
    if plan is not None:
        fps = case["inputs"]["frame_rate"]
        count = case["inputs"]["target_clip_seconds"] * fps
        expected_layout = [
            (f"clip-{i + 1}", i * count, count) for i in range(len(case["expected"]["shots"]))
        ]
        actual_layout = [(row["id"], row["startFrame"], row["frameCount"]) for row in plan["clips"]]
        if (
            actual_layout != expected_layout
            or plan["fps"] != fps
            or plan["totalFrames"] != case["inputs"]["duration_seconds"] * fps
        ):
            violations.append("clips.timing")
        known = {row["id"] for row in plan["characters"]}
        if any(set(row["characters"]) - known for row in plan["clips"]):
            violations.append("clips.character_ids")
    prompts = step_output(state, "prompt_preview", "h3_prompts")
    if prompts is not None:
        measured.append("compiled_prompts")
        for index, (expected, row) in enumerate(
            zip(case["expected"]["shots"], prompts["prompts"], strict=False), 1
        ):
            for quote in expected["dialogue"]:
                if quote not in row["prompt"]:
                    violations.append(f"compiled.shot-{index}.dialogue:{quote}")
            violations.extend(
                f"compiled.shot-{index}.dialogue_{error}"
                for error in dialogue_violations(row["prompt"], expected["dialogue"])
            )
    return {
        "measured": measured,
        "violations": violations,
        "complete": state.get("status") == "completed",
        "humanAcceptance": "not_measured",
    }


def score_repair(before, after, clip_id, expected_action):
    violations = []
    if {k: v for k, v in before.items() if k != "clips"} != {
        k: v for k, v in after.items() if k != "clips"
    }:
        violations.append("repair.plan_metadata")
    if [row["id"] for row in before["clips"]] != [row["id"] for row in after["clips"]]:
        violations.append("repair.membership")
    old_rows = {row["id"]: row for row in before["clips"]}
    new_rows = {row["id"]: row for row in after["clips"]}
    if new_rows.get(clip_id, {}).get("action") != expected_action:
        violations.append("repair.requested_action")
    for identifier, old in old_rows.items():
        new = new_rows.get(identifier, {})
        if identifier != clip_id:
            if old != new:
                violations.append("repair.unrelated:" + identifier)
        else:
            for field in old.keys() | new.keys():
                if field != "action" and old.get(field) != new.get(field):
                    violations.append("repair.unselected:" + field)
    return {"violations": violations, "fieldScope": "action"}


def summarize_turns(directory):
    source = Path(directory) / "model-turns.sqlite"
    if not source.exists():
        return {
            "modelTurns": 0,
            "reusedTurns": 0,
            "validationFailures": 0,
            "retryPromptCalls": 0,
            "callSeconds": 0,
            "perStep": {},
        }
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as db:
        rows = [
            json.loads(row[0]) for row in db.execute("SELECT detail FROM turns ORDER BY started")
        ]
    write_json(Path(directory) / "model-turns-export.json", rows)
    model = [row for row in rows if row["kind"] == "model"]
    by_step = {}
    for row in model:
        by_step[row["stepID"]] = by_step.get(row["stepID"], 0) + 1
    retry_markers = (
        "Previous attempt failed validation:",
        "Previous response error:",
        "Previous problem:",
        "Previous response problem:",
    )
    return {
        "modelTurns": len(model),
        "reusedTurns": len(rows) - len(model),
        "validationFailures": sum(row["validation"]["status"] == "failed" for row in model),
        "failedCalls": sum(
            row["status"] in {"failed", "cancelled", "interrupted"} for row in model
        ),
        "retryPromptCalls": sum(
            any(mark in row["prompt"] for mark in retry_markers) for row in model
        ),
        "callSeconds": sum(row.get("seconds") or 0 for row in model),
        "perStep": by_step,
    }


def run_session(case, args, dispatch, builtin, backend_class):
    directory = args.output / case["id"]
    directory.mkdir()
    trace = (directory / "helper-calls.jsonl").open("x")
    started = time.monotonic()

    class RecordedBackend(backend_class):
        calls = 0

        def generate(self, *positional, **keywords):
            from wee_todd_mlx.workflows.context_budget import NonRetryableAssistantError

            if self.calls >= args.max_calls or time.monotonic() - started >= args.session_seconds:
                raise NonRetryableAssistantError(
                    "Evaluation model-call or wall-time budget exhausted"
                )
            self.calls += 1
            call_start = time.monotonic()
            row = {"call": self.calls, "system": positional[1], "prompt": positional[2]}
            keywords["timeout"] = min(
                keywords["timeout"],
                args.call_seconds,
                max(0.01, args.session_seconds - (call_start - started)),
            )
            try:
                result = super().generate(*positional, **keywords)
                row["result"] = result
                return result
            except BaseException as error:
                row["error"] = str(error)
                raise
            finally:
                row["wallSeconds"] = time.monotonic() - call_start
                trace.write(json.dumps(row, ensure_ascii=False) + "\n")
                trace.flush()
                print(
                    json.dumps(
                        {
                            "session": case["id"],
                            "call": self.calls,
                            "seconds": round(row["wallSeconds"], 2),
                        }
                    ),
                    flush=True,
                )

    backend = RecordedBackend({"assistant": str(args.model)}, {}, args.helper)
    definition = builtin("guided-movie-planning")
    request = {
        "definition": definition,
        "inputs": case["inputs"],
        "runDirectory": str(directory / "workflow"),
        "maxTokens": args.max_tokens,
    }
    write_json(directory / "request.json", request)
    decisions, state, error, repair = [], {}, None, None
    first_score = None

    def review(action, step, **extra):
        mutation = {
            "stepID": step,
            "expectedRevision": state["revision"],
            "action": action,
            **extra,
        }
        updated = dispatch("workflow-review", {**request, "review": mutation}, backend=backend)
        decisions.append(
            {
                "actor": "evaluation_driver",
                "creativeAcceptance": False,
                "mutation": mutation,
                "resultRevision": updated["revision"],
            }
        )
        write_json(directory / "evaluation-decisions.json", decisions)
        return updated

    try:
        for _ in range(30):
            state = dispatch("workflow-run", request, backend=backend)
            if state["status"] != "awaiting_approval":
                break
            sid = state["awaitingStep"]
            write_json(directory / f"before-review-{sid}.json", state)
            if sid == "creative_brief":
                outputs = copy.deepcopy(state["steps"][sid]["outputs"])
                questions = outputs["creative_brief"]["questions"]
                if questions:
                    for question in questions:
                        question["answer"] = case["clarificationAnswer"]
                    state = review("edit", sid, outputs=outputs)
            state = review("approve", sid)
        write_json(directory / "before-correction.json", state)
        first_score = score_session(case, state)
        before = step_output(state, "clips", "clips")
        if state.get("status") == "completed" and before is not None:
            correction = case["correction"]
            cid = correction["clipID"]
            state = review("unapprove", "clips", itemID=cid)
            scope = {"fieldScope": "action"} if args.repair_scope == "action" else {}
            state = review(
                "repair",
                "clips",
                itemID=cid,
                instruction="Change only the action field to exactly: "
                + correction["action"]
                + " Retain every other field, timing, identity, and unrelated shot unchanged.",
                **scope,
            )
            state = dispatch("workflow-run", request, backend=backend)
            after = step_output(state, "clips", "clips")
            clip_step = state.get("steps", {}).get("clips", {})
            completed_repair = (
                state.get("status") in {"awaiting_approval", "completed"}
                and clip_step.get("status") == "completed"
                and clip_step.get("items", {}).get(cid, {}).get("status") == "completed"
            )
            if after is not None and completed_repair:
                repair = score_repair(before, after, cid, correction["action"])
                repair["runStatus"] = state.get("status")
            else:
                repair = {
                    "violations": ["repair.no_completed_result"],
                    "runStatus": state.get("status"),
                    "error": state.get("error"),
                }
            write_json(directory / "after-correction.json", state)
    except (ValueError, RuntimeError, OSError, TimeoutError) as caught:
        error = str(caught)
        checkpoint = directory / "workflow/run.json"
        if checkpoint.exists():
            state = json.loads(checkpoint.read_text())
        if first_score is None:
            first_score = score_session(case, state)
        elif repair is None:
            repair = {
                "violations": ["repair.execution_error"],
                "runStatus": state.get("status"),
                "error": error,
            }
    finally:
        trace.close()
    summary = {
        "session": case["id"],
        "status": state.get("status"),
        "error": error or state.get("error"),
        "objective": first_score,
        "repair": repair,
        "weightedCalls": backend.calls,
        "wallSeconds": time.monotonic() - started,
        "scriptedApprovals": sum(row["mutation"]["action"] == "approve" for row in decisions),
        "scriptedClarificationEdits": sum(row["mutation"]["action"] == "edit" for row in decisions),
        "humanCreativeAcceptance": "not_measured",
        **summarize_turns(directory / "workflow"),
    }
    write_json(directory / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--expected-digest")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--helper", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repair-scope", choices=("legacy", "action"), default="action")
    parser.add_argument("--max-calls", type=int, default=64)
    parser.add_argument("--session-seconds", type=float, default=1500)
    parser.add_argument("--call-seconds", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args(argv)
    suite = load_suite(args.suite, args.expected_digest)
    if not args.run:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "sessions": len(suite["sessions"]),
                    "suiteSHA256": digest_file(args.suite),
                    "modelCalls": 0,
                }
            )
        )
        return 0
    if not args.model or not args.helper or not args.output:
        parser.error("--run requires --model, --helper and --output")
    if (
        not 1 <= args.max_calls <= 100
        or not 1 <= args.max_tokens <= 1024
        or not all(
            math.isfinite(value) and value > 0
            for value in (args.call_seconds, args.session_seconds)
        )
    ):
        parser.error("Use bounded positive budgets (at most 100 calls, 1024 tokens)")
    args.source_root = args.source_root.resolve(strict=True)
    args.model, args.helper = args.model.resolve(strict=True), args.helper.resolve(strict=True)
    args.output = args.output.resolve()
    if args.output.is_relative_to(ROOT) or args.output.is_relative_to(args.source_root):
        parser.error("Keep evaluation output outside all source checkouts")
    if not os.access(args.helper, os.X_OK):
        parser.error("Helper must be executable")
    digest = freeze_suite(args.suite, args.output)
    suite = load_suite(args.output / "suite.json", digest)

    def identity(source):
        stat = source.stat()
        return {"path": str(source), "bytes": stat.st_size, "modifiedNS": stat.st_mtime_ns}

    manifest = {
        "suiteSHA256": digest,
        "harnessSHA256": digest_file(Path(__file__)),
        "sourceRoot": str(args.source_root),
        "model": identity(args.model),
        "helper": {**identity(args.helper), "sha256": digest_file(args.helper)},
        "repairScope": args.repair_scope,
        "maxCallsPerSession": args.max_calls,
        "maxTokens": args.max_tokens,
        "callSeconds": args.call_seconds,
        "sessionSeconds": args.session_seconds,
        "approvalPolicy": "Evaluation driver structural decisions, not human creative acceptance",
        "sourceSHA256": {
            str(p.relative_to(args.source_root)): digest_file(p)
            for p in sorted((args.source_root / "src/wee_todd_mlx/workflows").rglob("*"))
            if p.suffix in {".py", ".json"}
        },
    }
    sidecar = Path(str(args.model) + "-tensordata")
    if sidecar.exists():
        manifest["modelSidecar"] = identity(sidecar)
    write_json(args.output / "manifest.json", manifest)
    sys.path.insert(0, str(args.source_root / "src"))
    from wee_todd_mlx.workflows.backend import LocalQwenBackend
    from wee_todd_mlx.workflows.service import builtin, dispatch

    summaries = [
        run_session(case, args, dispatch, builtin, LocalQwenBackend) for case in suite["sessions"]
    ]
    summary = {
        "sessions": summaries,
        "suiteSHA256": digest,
        "completedBeforeCorrection": sum(row["objective"]["complete"] for row in summaries),
        "weightedCalls": sum(row["weightedCalls"] for row in summaries),
        "objectiveViolations": sum(len(row["objective"]["violations"]) for row in summaries),
        "repairViolations": sum(
            len(row["repair"]["violations"]) for row in summaries if row["repair"]
        ),
        "repairsMeasured": sum(row["repair"] is not None for row in summaries),
        "humanCreativeAcceptance": "not_measured",
    }
    write_json(args.output / "summary.json", summary)
    return 0 if summary["completedBeforeCorrection"] == len(summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
