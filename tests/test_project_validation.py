"""Validation selection must cover shipped interfaces without repeating checks."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_project.py"


def plan(*profiles):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--project", str(ROOT), "--list", *profiles],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_combined_validation_covers_every_fast_test_once():
    checks = plan("--profile", "core", "--profile", "studio", "--profile", "workflows",
                  "--profile", "remote", "--profile", "core")
    selected = [arg for command in checks for arg in command if arg.startswith("tests/test_")]
    assert len(selected) == len(set(selected))
    expected = {str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("test_*.py")}
    expected.remove("tests/test_packing_parity.py")  # Optional external reference runtime.
    assert set(selected) == expected
    assert len([c for c in checks if "scripts/preflight_python_environment.py" in c]) == 1


def test_workflow_profile_validates_all_shipped_definitions():
    checks = plan("--profile", "workflows")
    selected = {c[-1] for c in checks if c[1] == "scripts/validate_studio_workflow.py"}
    assert selected == {str(p.relative_to(ROOT)) for p in
                        (ROOT / "examples/studio-workflows").rglob("*.json")}
    assert "tests/test_workflow_turns.py" in next(c for c in checks if "pytest" in c)
    workflow_tests = next(c for c in checks if "pytest" in c)
    assert "tests/test_director_production_evaluation.py" in workflow_tests
    assert any(c[1:] == ["scripts/evaluate_director_production.py"] for c in checks)
    assert not any(c[0] == "swift" for c in checks)


def test_native_profiles_include_both_swift_packages():
    checks = plan("--profile", "studio", "--profile", "remote")
    swift = [c for c in checks if c[0] == "swift"]
    assert ["swift", "test", "--package-path", "studio", "--jobs", "2"] in swift
    assert ["swift", "test", "--package-path", "integrations/drawthings-client",
            "--jobs", "2"] in swift


def test_failed_check_stops_before_later_commands(tmp_path):
    assert SCRIPT.is_file()
    spec = importlib.util.spec_from_file_location("project_validation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    marker = tmp_path / "unexpected"
    code = "from pathlib import Path; Path('unexpected').write_text('ran')"
    checks = [[sys.executable, "-c", "raise SystemExit(7)"], [sys.executable, "-c", code]]
    assert module.run_checks(tmp_path, checks) == 7
    assert not marker.exists()
