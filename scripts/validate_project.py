#!/usr/bin/env python3
"""Run composable, model-free project validation profiles from a checkout or source snapshot.

Examples:
  python scripts/validate_project.py --profile core
  python scripts/validate_project.py --profile studio --profile workflows
  python scripts/validate_project.py --profile remote --list

Profiles never install dependencies, generate media, record manual review or replace an app bundle.
Studio and remote profiles include native Swift tests; run release packaging separately.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

PROFILES = ("core", "studio", "workflows", "remote")
# This file imports a separately maintained Diffusers reference implementation.
# Algorithm-search experiments are outside the top-level test glob as well.
OPTIONAL_TESTS = {"test_packing_parity.py"}
WORKFLOW_TESTS = {
    "test_prompt_assist.py", "test_subject_description_review.py", "test_movie_story_allocation.py",
    "test_director_engine_foundation.py",
    "test_director_production_evaluation.py",
    "test_director_source_preservation.py",
}


def test_profile(name: str) -> str:
    if (name.startswith(("test_workflow_", "test_studio_workflow_"))
            or name in WORKFLOW_TESTS):
        return "workflows"
    if name.startswith("test_studio_") or name == "test_assistant_models.py":
        return "studio"
    if name.startswith("test_drawthings_"):
        return "remote"
    return "core"


def build_checks(project: Path, profiles: list[str], python: str) -> list[list[str]]:
    selected = set(profiles)
    if not selected or not selected.issubset(PROFILES):
        raise ValueError("Select at least one known validation profile")
    checks = [[python, "scripts/preflight_python_environment.py", "--project", ".",
               "--python", python, "--require-architecture", "arm64"]]
    lint_paths = {"tests", "scripts/validate_project.py"}
    if "core" in selected:
        checks.extend([
            [python, "-m", "compileall", "-q", "src", "__init__.py"],
            [python, "scripts/lint_docs.py"],
            [python, "scripts/audit_workflow_catalog.py", "--project", "."],
            [python, "scripts/update_readme_node_catalog.py", "--project", ".", "--check"],
            [python, "scripts/preflight_h3_workflow.py", "--project", ".", "--all-api"],
        ])
        lint_paths.add("src")
    if "workflows" in selected:
        definitions = sorted((project / "examples/studio-workflows").rglob("*.json"))
        if not definitions:
            raise ValueError("No shipped Studio workflow definitions found")
        checks.extend([python, "scripts/validate_studio_workflow.py", str(p.relative_to(project))]
                      for p in definitions)
        checks.append([python, "scripts/evaluate_studio_assistant.py"])
        checks.append([python, "scripts/evaluate_director_production.py"])
        lint_paths.update({
            "src/wee_todd_mlx/workflows", "scripts/run_studio_workflow.py",
            "scripts/validate_studio_workflow.py", "scripts/studio_prompt_assist.py",
            "src/wee_todd_mlx/assistant_evaluation.py", "scripts/evaluate_studio_assistant.py",
            "scripts/evaluate_director_production.py",
        })
    if "studio" in selected:
        lint_paths.update({"scripts/studio_bridge.py", "scripts/studio_references.py",
                           "scripts/studio_lora.py", "scripts/studio_music.py",
                           "scripts/studio_production.py",
                           "scripts/build_studio_app.py",
                           "scripts/install_studio_runtime.py", "scripts/setup_assistant_model.py",
                           "scripts/studio_assistant_models.py",
                           "src/wee_todd_mlx/assistant_models.py",
                           "src/wee_todd_mlx/model_downloads.py"})
    if "remote" in selected:
        lint_paths.update({"src/wee_todd_remote", "scripts/studio_drawthings.py",
                           "scripts/build_drawthings_client.py"})
    tests = [str(p.relative_to(project)) for p in sorted((project / "tests").glob("test_*.py"))
             if p.name not in OPTIONAL_TESTS and test_profile(p.name) in selected]
    if not tests:
        raise ValueError("No tests found for the selected validation profiles")
    checks.append([python, "-m", "pytest", "-q", *tests])
    # A combined core run already covers every engine/adapter below src/.
    if "src" in lint_paths:
        lint_paths = {p for p in lint_paths if not p.startswith("src/")}
    checks.append([python, "-m", "ruff", "check", *sorted(lint_paths)])
    if "studio" in selected:
        checks.append(["swift", "test", "--package-path", "studio", "--jobs", "2"])
    if "remote" in selected:
        checks.append(["swift", "test", "--package-path", "integrations/drawthings-client",
                       "--jobs", "2"])
    return checks


def run_checks(project: Path, checks: list[list[str]]) -> int:
    for command in checks:
        print(f"Running: {shlex.join(command)}", flush=True)
        result = subprocess.run(command, cwd=project, check=False)
        if result.returncode:
            print(f"Validation stopped: check exited {result.returncode}", file=sys.stderr)
            return result.returncode
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--profile", choices=PROFILES, action="append")
    parser.add_argument("--list", action="store_true", help="Print command plan as JSON; no runs")
    args = parser.parse_args(argv)
    project = args.project.resolve()
    try:
        checks = build_checks(project, args.profile or ["core"], sys.executable)
    except ValueError as error:
        parser.error(str(error))
    if args.list:
        print(json.dumps(checks, indent=2))
        return 0
    return run_checks(project, checks)


if __name__ == "__main__":
    raise SystemExit(main())
