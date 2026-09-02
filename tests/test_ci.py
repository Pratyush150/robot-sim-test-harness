"""Baselines, tolerances and the CI verdict."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import minimal_scenario_dict
from simharness.ci import (
    DEFAULT_TOLERANCES,
    MetricTolerance,
    build_baseline,
    compare_to_baseline,
    load_baseline,
    load_tolerances,
    save_baseline,
)
from simharness.runner import RunResult, SuiteResult, run_scenario
from simharness.scenario import Scenario

STATUS_PASS = "pass"
STATUS_FAIL = "fail"


def _suite(*results: RunResult) -> SuiteResult:
    return SuiteResult(results=list(results), wall_time_s=0.5)


def _result(name: str, status: str = STATUS_PASS, **metrics: float) -> RunResult:
    return RunResult(scenario=name, status=status, simulator="mock", seed=1, metrics=metrics)


# -- tolerances ------------------------------------------------------------


def test_absolute_tolerance_admits_small_moves():
    tol = MetricTolerance(abs_tol=0.05)
    assert tol.within(1.0, 1.04) is True
    assert tol.within(1.0, 1.06) is False


def test_relative_tolerance_admits_proportional_moves():
    tol = MetricTolerance(rel_tol=0.10)
    assert tol.within(100.0, 109.0) is True
    assert tol.within(100.0, 111.0) is False


def test_either_band_is_enough():
    tol = MetricTolerance(abs_tol=0.01, rel_tol=0.5)
    assert tol.within(0.001, 0.009) is True   # inside the absolute band
    assert tol.within(100.0, 140.0) is True   # inside the relative band
    assert tol.within(100.0, 200.0) is False


def test_relative_tolerance_is_ignored_against_a_zero_baseline():
    assert MetricTolerance(rel_tol=1.0).within(0.0, 5.0) is False


def test_default_tolerances_cover_the_headline_metrics():
    for metric in ("path_length_m", "min_clearance_m", "time_to_goal_s", "energy_wh"):
        assert metric in DEFAULT_TOLERANCES


def test_load_tolerances_from_yaml(tmp_path: Path):
    path = tmp_path / "tol.yaml"
    path.write_text("path_length_m: {abs: 0.5, rel: 0.2}\ncustom_metric: {abs: 1.0}\n", encoding="utf-8")
    tolerances = load_tolerances(path)
    assert tolerances["path_length_m"] == MetricTolerance(abs_tol=0.5, rel_tol=0.2)
    assert tolerances["custom_metric"].abs_tol == 1.0
    assert "min_clearance_m" in tolerances, "defaults must survive a partial override"


def test_load_tolerances_from_json(tmp_path: Path):
    path = tmp_path / "tol.json"
    path.write_text(json.dumps({"energy_wh": {"rel": 0.25}}), encoding="utf-8")
    assert load_tolerances(path)["energy_wh"].rel_tol == 0.25


def test_load_tolerances_rejects_a_bad_document(tmp_path: Path):
    path = tmp_path / "tol.json"
    path.write_text(json.dumps({"energy_wh": 0.25}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_tolerances(path)


# -- baselines -------------------------------------------------------------


def test_baseline_round_trips_through_disk(tmp_path: Path):
    suite = _suite(_result("a", path_length_m=10.0), _result("b", STATUS_FAIL, path_length_m=3.0))
    path = save_baseline(suite, tmp_path / "ci" / "baseline.json", note="first run")
    loaded = load_baseline(path)
    assert set(loaded.entries) == {"a", "b"}
    assert loaded.entries["a"]["metrics"]["path_length_m"] == 10.0
    assert loaded.note == "first run"
    assert loaded.recorded_at > 0.0


def test_baseline_rejects_an_unknown_format(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"format": "something-else", "scenarios": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_baseline(path)


def test_baseline_records_a_real_run():
    scenario = Scenario.from_dict(
        minimal_scenario_dict(
            sim={"dt": 0.05, "time_limit": 30.0, "seed": 1},
            assertions=[{"type": "reached_goal", "name": "arrives"}],
        )
    )
    baseline = build_baseline(_suite(run_scenario(scenario, keep_trace=False)))
    entry = baseline.entries["unit"]
    assert entry["status"] == STATUS_PASS
    assert entry["assertions"] == {"arrives": True}
    assert entry["metrics"]["path_length_m"] > 0


# -- comparisons -----------------------------------------------------------


def test_no_change_is_a_clean_pass():
    suite = _suite(_result("a", path_length_m=10.0))
    comparison = compare_to_baseline(suite, build_baseline(suite))
    assert comparison.ok is True
    assert comparison.regressions == []
    assert comparison.drift == []
    assert "No status changes" in comparison.report()


def test_a_green_to_red_transition_is_a_regression():
    baseline = build_baseline(_suite(_result("a", path_length_m=10.0)))
    now = _suite(_result("a", STATUS_FAIL, path_length_m=10.0))
    comparison = compare_to_baseline(now, baseline)
    assert comparison.regressions == [("a", STATUS_PASS, STATUS_FAIL)]
    assert comparison.ok is False
    assert "Regressions" in comparison.report()


def test_a_red_to_green_transition_is_an_improvement_not_a_failure():
    baseline = build_baseline(_suite(_result("a", STATUS_FAIL, path_length_m=10.0)))
    now = _suite(_result("a", STATUS_PASS, path_length_m=10.0))
    comparison = compare_to_baseline(now, baseline)
    assert comparison.improvements == [("a", STATUS_FAIL, STATUS_PASS)]
    assert comparison.regressions == []
    assert comparison.ok is True


def test_expected_failure_stays_green_against_a_pass_baseline():
    baseline = build_baseline(_suite(_result("a", STATUS_PASS)))
    now = _suite(_result("a", "expected_failure"))
    comparison = compare_to_baseline(now, baseline)
    assert comparison.regressions == []
    assert comparison.status_changes == [("a", "pass", "expected_failure")]
    assert comparison.ok is True


def test_unexpected_pass_is_a_regression():
    baseline = build_baseline(_suite(_result("a", "expected_failure")))
    now = _suite(_result("a", "unexpected_pass"))
    comparison = compare_to_baseline(now, baseline)
    assert comparison.regressions == [("a", "expected_failure", "unexpected_pass")]
    assert comparison.ok is False


def test_metric_drift_is_advisory_by_default_and_fatal_under_strict():
    baseline = build_baseline(_suite(_result("a", path_length_m=10.0)))
    now = _suite(_result("a", path_length_m=13.0))
    lenient = compare_to_baseline(now, baseline)
    strict = compare_to_baseline(now, baseline, strict_metrics=True)
    assert len(lenient.drift) == 1
    assert lenient.ok is True
    assert strict.ok is False
    assert "advisory" in lenient.report()
    assert "failing" in strict.report()


def test_metric_drift_reports_direction_and_percentage():
    baseline = build_baseline(_suite(_result("a", path_length_m=10.0)))
    comparison = compare_to_baseline(_suite(_result("a", path_length_m=12.0)), baseline)
    drift = comparison.drift[0]
    assert drift.metric == "path_length_m"
    assert drift.delta == pytest.approx(2.0)
    assert drift.percent == pytest.approx(20.0)
    assert "+20.0%" in drift.describe()


def test_drift_within_tolerance_is_not_reported():
    baseline = build_baseline(_suite(_result("a", path_length_m=10.0)))
    comparison = compare_to_baseline(_suite(_result("a", path_length_m=10.3)), baseline)
    assert comparison.drift == []


def test_custom_tolerance_overrides_the_default():
    baseline = build_baseline(_suite(_result("a", path_length_m=10.0)))
    now = _suite(_result("a", path_length_m=13.0))
    loose = compare_to_baseline(now, baseline, tolerances={"path_length_m": MetricTolerance(rel_tol=0.5)})
    assert loose.drift == []


def test_a_new_scenario_is_reported_but_does_not_fail_the_build():
    baseline = build_baseline(_suite(_result("a")))
    comparison = compare_to_baseline(_suite(_result("a"), _result("b")), baseline)
    assert comparison.new_scenarios == ["b"]
    assert comparison.ok is True


def test_a_missing_scenario_fails_the_build():
    baseline = build_baseline(_suite(_result("a"), _result("b")))
    comparison = compare_to_baseline(_suite(_result("a")), baseline)
    assert comparison.missing_scenarios == ["b"]
    assert comparison.ok is False
    assert "Missing" in comparison.report()


def test_missing_scenarios_can_be_ignored():
    baseline = build_baseline(_suite(_result("a"), _result("b")))
    comparison = compare_to_baseline(_suite(_result("a")), baseline, ignore_missing=True)
    assert comparison.missing_scenarios == []
    assert comparison.ok is True


def test_a_red_suite_fails_even_with_a_matching_baseline():
    suite = _suite(_result("a", STATUS_FAIL))
    comparison = compare_to_baseline(suite, build_baseline(suite))
    assert comparison.regressions == []
    assert comparison.suite_ok is False
    assert comparison.ok is False


def test_comparison_serialises():
    baseline = build_baseline(_suite(_result("a", path_length_m=10.0)))
    comparison = compare_to_baseline(_suite(_result("a", STATUS_FAIL, path_length_m=15.0)), baseline)
    payload = json.loads(json.dumps(comparison.to_dict()))
    assert payload["ok"] is False
    assert payload["regressions"][0]["scenario"] == "a"
    assert payload["drift"][0]["metric"] == "path_length_m"
