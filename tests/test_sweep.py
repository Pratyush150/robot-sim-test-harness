"""Sweeps, Monte Carlo and failure-boundary bisection."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import minimal_scenario_dict
from simharness.scenario import Scenario, load_scenario
from simharness.sweep import find_failure_boundary, find_scenario_boundary, grid_sweep, monte_carlo


def _scenario() -> Scenario:
    return Scenario.from_dict(
        minimal_scenario_dict(
            sim={"dt": 0.05, "time_limit": 30.0, "seed": 5},
            assertions=[{"type": "reached_goal", "name": "arrives", "within": 20.0}],
            goal={"x": 10.0, "y": 0.0, "tolerance": 0.25, "within": 20.0},
        )
    )


# -- bisection on a known analytic boundary --------------------------------


def test_bisection_finds_a_known_boundary_within_tolerance():
    result = find_failure_boundary(lambda v: v < 3.14159, low=0.0, high=10.0, tolerance=0.01)
    assert result.found is True
    assert result.status == "bracketed"
    assert result.boundary == pytest.approx(3.14159, abs=0.01)
    assert result.first_fail - result.last_pass <= 0.01
    assert result.last_pass < 3.14159 <= result.first_fail


def test_bisection_tightens_with_a_smaller_tolerance():
    coarse = find_failure_boundary(lambda v: v < 2.5, low=0.0, high=10.0, tolerance=0.5)
    fine = find_failure_boundary(lambda v: v < 2.5, low=0.0, high=10.0, tolerance=0.001)
    assert coarse.first_fail - coarse.last_pass <= 0.5
    assert fine.first_fail - fine.last_pass <= 0.001
    assert fine.evaluations > coarse.evaluations
    assert fine.boundary == pytest.approx(2.5, abs=0.001)


def test_bisection_needs_only_logarithmically_many_evaluations():
    result = find_failure_boundary(lambda v: v < 512.0, low=0.0, high=1024.0, tolerance=1.0)
    # 1024 -> 1 is 10 halvings, plus the two endpoint probes.
    assert result.evaluations <= 13
    assert result.boundary == pytest.approx(512.0, abs=1.0)


def test_bisection_reports_when_everything_passes():
    result = find_failure_boundary(lambda v: True, low=0.0, high=5.0, tolerance=0.01)
    assert result.status == "passes_everywhere"
    assert result.found is False
    assert result.boundary is None
    assert "still passing" in result.summary()


def test_bisection_reports_when_everything_fails():
    result = find_failure_boundary(lambda v: False, low=0.0, high=5.0, tolerance=0.01)
    assert result.status == "fails_everywhere"
    assert result.found is False
    assert "already failing" in result.summary()


def test_bisection_rejects_an_inverted_range():
    with pytest.raises(ValueError):
        find_failure_boundary(lambda v: True, low=5.0, high=1.0)


def test_bisection_rejects_a_non_positive_tolerance():
    with pytest.raises(ValueError):
        find_failure_boundary(lambda v: True, low=0.0, high=1.0, tolerance=0.0)


def test_bisection_history_is_recorded_in_probe_order():
    result = find_failure_boundary(lambda v: v < 4.0, low=0.0, high=8.0, tolerance=0.1)
    assert result.history[0] == (0.0, True)
    assert result.history[1] == (8.0, False)
    assert all(isinstance(value, float) and isinstance(ok, bool) for value, ok in result.history)


def test_bisection_stops_at_max_iterations():
    result = find_failure_boundary(lambda v: v < 0.5, low=0.0, high=1.0, tolerance=1e-12, max_iterations=3)
    assert result.iterations == 3
    assert "max_iterations" in result.note


# -- boundary on a real scenario -------------------------------------------


def test_scenario_boundary_finds_where_the_deadline_starts_being_missed():
    """Lowering max_speed eventually makes a 20 s deadline unreachable.

    The vehicle has 10 m to cover in 20 s, so the boundary sits just under
    0.5 m/s plus a small allowance for the acceleration ramp. Bisection is
    searched downward by flipping the sign, since the harness bisects an
    increasing parameter.
    """
    scenario = _scenario()
    boundary = find_failure_boundary(
        lambda inverse_speed: _reaches_goal(scenario, 1.0 / inverse_speed),
        low=1.0,       # 1.0 m/s: easy
        high=4.0,      # 0.25 m/s: 40 s of travel for a 20 s deadline
        tolerance=0.02,
        parameter="1/max_speed",
    )
    assert boundary.found is True
    assert 1.0 / boundary.boundary == pytest.approx(0.53, abs=0.06)


def _reaches_goal(scenario: Scenario, max_speed: float) -> bool:
    from simharness.runner import GREEN_STATUSES, run_scenario

    variant = scenario.with_overrides({"robot.max_speed": max_speed})
    return run_scenario(variant, keep_trace=False).status in GREEN_STATUSES


def test_wind_boundary_on_the_shipped_scenario(scenario_dir: Path):
    scenario = load_scenario(scenario_dir / "wind_disturbance.yaml")
    boundary = find_scenario_boundary(scenario, "disturbance.wind[1]", low=0.0, high=14.0, tolerance=0.25)
    assert boundary.found is True
    assert 6.0 < boundary.boundary < 10.0
    assert boundary.last_pass < boundary.first_fail


# -- grids -----------------------------------------------------------------


def test_grid_sweep_covers_the_cross_product():
    result = grid_sweep(_scenario(), {"sim.seed": [1, 2], "robot.max_speed": [0.8, 1.0, 1.2]})
    assert result.total == 6
    assert result.mode == "grid"
    assert set(result.parameters) == {"sim.seed", "robot.max_speed"}
    assert all(len(p.overrides) == 2 for p in result.points)


def test_grid_sweep_pass_rate_and_table():
    result = grid_sweep(_scenario(), {"robot.max_speed": [0.2, 0.3, 1.0, 1.2]})
    assert 0.0 < result.pass_rate < 1.0, "this grid should straddle the boundary"
    table = result.format_table("robot.max_speed")
    assert "passed" in table and "total" in table
    assert table.count("\n") >= 5
    rows = result.pass_rate_by("robot.max_speed")
    assert [value for value, _, _ in rows] == [0.2, 0.3, 1.0, 1.2]
    assert sum(total for _, _, total in rows) == result.total


def test_grid_sweep_first_failing_value():
    result = grid_sweep(_scenario(), {"robot.max_speed": [0.2, 0.3, 1.0]})
    assert result.first_failing_value("robot.max_speed") == 0.2


def test_grid_sweep_needs_parameters():
    with pytest.raises(ValueError):
        grid_sweep(_scenario(), {})
    with pytest.raises(ValueError):
        grid_sweep(_scenario(), {"sim.seed": []})


def test_grid_sweep_records_a_bad_override_as_an_error_point():
    result = grid_sweep(_scenario(), {"robot.max_sped": [1.0]})
    assert result.total == 1
    assert result.points[0].status == "error"
    assert "ScenarioError" in result.points[0].error
    assert result.pass_rate == 0.0


def test_grid_sweep_runs_in_parallel_with_the_same_answer():
    serial = grid_sweep(_scenario(), {"sim.seed": [1, 2, 3, 4]}, workers=1)
    parallel = grid_sweep(_scenario(), {"sim.seed": [1, 2, 3, 4]}, workers=3)
    assert [p.status for p in serial.points] == [p.status for p in parallel.points]
    assert [p.overrides for p in serial.points] == [p.overrides for p in parallel.points]


# -- Monte Carlo -----------------------------------------------------------


def test_monte_carlo_is_repeatable_for_a_given_seed():
    a = monte_carlo(_scenario(), {"robot.max_speed": (0.4, 1.2)}, samples=8, seed=99)
    b = monte_carlo(_scenario(), {"robot.max_speed": (0.4, 1.2)}, samples=8, seed=99)
    assert [p.overrides for p in a.points] == [p.overrides for p in b.points]
    assert [p.status for p in a.points] == [p.status for p in b.points]


def test_monte_carlo_varies_the_sim_seed_by_default():
    result = monte_carlo(_scenario(), {"robot.max_speed": (0.6, 1.2)}, samples=6, seed=1)
    seeds = {p.overrides["sim.seed"] for p in result.points}
    assert len(seeds) == 6
    assert "sim.seed" in result.parameters


def test_monte_carlo_can_hold_the_sim_seed_fixed():
    result = monte_carlo(_scenario(), {"robot.max_speed": (0.6, 1.2)}, samples=4, seed=1, vary_sim_seed=False)
    assert all("sim.seed" not in p.overrides for p in result.points)


def test_monte_carlo_samples_stay_inside_the_range():
    result = monte_carlo(_scenario(), {"robot.max_speed": (0.6, 1.1)}, samples=12, seed=3)
    values = [p.overrides["robot.max_speed"] for p in result.points]
    assert all(0.6 <= v <= 1.1 for v in values)
    assert len(set(values)) == 12


def test_monte_carlo_rejects_bad_inputs():
    with pytest.raises(ValueError):
        monte_carlo(_scenario(), {"robot.max_speed": (1.2, 0.6)}, samples=2)
    with pytest.raises(ValueError):
        monte_carlo(_scenario(), {"robot.max_speed": (0.6, 1.2)}, samples=0)


def test_sweep_result_serialises():
    import json

    result = grid_sweep(_scenario(), {"sim.seed": [1, 2]})
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["total"] == 2
    assert payload["pass_rate"] == pytest.approx(result.pass_rate)
