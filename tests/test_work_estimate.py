"""Estimates preserve scope and expose unknowns; no inferred effort percentages."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def tool():
    spec = importlib.util.spec_from_file_location(
        "work_estimate", ROOT / "scripts/work_estimate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task(name, status="pending", work=(10, 20), wait=(0, 0), basis="measured", evidence="test.log"):
    return dict(
        id=name, status=status, work_seconds=work, wait_seconds=wait, basis=basis, evidence=evidence
    )


def test_unknown_work_prevents_false_total_and_completion_is_checklist_only():
    result = tool().summarize(
        {
            "original_scope": ["a", "b"],
            "tasks": [task("a", "done"), task("b", work=None), task("c")],
        }
    )
    assert result["remaining_work_seconds"] is None
    assert result["unknown_tasks"] == ["b"]
    assert result["original_checks_done"] == 1
    assert result["original_checks_total"] == 2
    assert result["added_scope"] == ["c"]
    assert "effort_percent" not in result


def test_completed_work_disappears_from_remaining_and_work_wait_stay_separate():
    result = tool().summarize(
        {
            "original_scope": ["a", "b"],
            "tasks": [
                task("a", "done", (100, 200), (500, 600)),
                task("b", work=(15, 25), wait=(30, 40)),
            ],
        }
    )
    assert result["remaining_work_seconds"] == [15, 25]
    assert result["remaining_wait_seconds"] == [30, 40]
    assert result["serial_remaining_seconds"] == [45, 65]
    assert result["time_confidence"] == "not statistically calibrated"


def test_estimates_require_basis_and_completed_checks_require_evidence():
    for item in [
        task("a", work=(-1, 1)),
        task("a", work=(True, 20)),
        task("a", work=(20, 10)),
        task("a", basis=""),
        task("a", "done", evidence=""),
    ]:
        with pytest.raises(ValueError):
            tool().summarize({"original_scope": ["a"], "tasks": [item]})
    with pytest.raises(ValueError):
        tool().summarize({"original_scope": ["a"], "tasks": [task("a"), task("a")]})
    with pytest.raises(ValueError):
        tool().summarize({"original_scope": ["missing"], "tasks": [task("a")]})


def test_overlapping_wait_intervals_are_not_double_counted_and_rework_is_visible():
    result = tool().actuals(
        [
            dict(
                kind="wait",
                start="2026-10-05T00:00:00Z",
                end="2026-10-05T00:01:00Z",
                evidence="build.log",
            ),
            dict(
                kind="wait",
                start="2026-10-05T00:00:30Z",
                end="2026-10-05T00:02:00Z",
                evidence="render.log",
            ),
            dict(
                kind="rework",
                start="2026-10-05T00:00:40Z",
                end="2026-10-05T00:00:50Z",
                evidence="error.log",
            ),
        ]
    )
    assert result["wait_union_seconds"] == 120
    assert result["rework_union_seconds"] == 10
    assert result["wall_seconds"] == 120
    assert result["uncovered_seconds"] == 0
    with pytest.raises(ValueError):
        tool().actuals(
            [
                dict(
                    kind="wait",
                    start="2026-10-05T00:00:01Z",
                    end="2026-10-05T00:00:00Z",
                    evidence="x",
                )
            ]
        )


def test_baseline_check_exposes_original_forecast_or_scope_rewriting():
    original = {"original_scope": ["a"], "original_forecast": {"hours": [4, 7]}}
    tool().compare_baseline(dict(original), original)
    for key, changed in [("original_scope", ["b"]), ("original_forecast", {"hours": [1, 2]})]:
        update = dict(original)
        update[key] = changed
        with pytest.raises(ValueError):
            tool().compare_baseline(update, original)
