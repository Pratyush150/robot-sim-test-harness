"""Scalar behavioural assertions, checked against hand-built traces.

Every expectation here is a number a human can verify by reading the column
that was fed in, which is the point: if these tests only compared against a
simulation they would be testing nothing.
"""

from __future__ import annotations

import math

import pytest

from conftest import minimal_scenario_dict
from simharness.assertions import build_signals, evaluate_all, evaluate_assertion, summarise
from simharness.scenario import AssertionSpec, Scenario, ScenarioError


def _evaluate(make_trace, spec_dict, scenario_overrides=None, dt=0.1, **columns):
    data = minimal_scenario_dict(**(scenario_overrides or {}))
    scenario = Scenario.from_dict(data)
    trace = make_trace(columns, dt=dt, meta={"goal": list(scenario.goal_xyz())})
    ctx = build_signals(trace, scenario)
    spec = AssertionSpec(
        type=spec_dict["type"],
        params={k: v for k, v in spec_dict.items() if k not in ("type", "name")},
        name=spec_dict.get("name"),
        path="assertions[0]",
    )
    return evaluate_assertion(spec, ctx)


# -- clearance -------------------------------------------------------------


def test_min_clearance_reports_the_exact_worst_value_and_timestamp(make_trace):
    clearance = [1.0, 0.8, 0.42, 0.17, 0.31, 0.9]
    result = _evaluate(make_trace, {"type": "min_clearance", "threshold": 0.2}, clearance=clearance)
    assert result.passed is False
    assert result.worst_value == pytest.approx(0.17)
    assert result.worst_index == 3
    assert result.worst_time == pytest.approx(0.3)
    assert result.threshold == pytest.approx(0.2)
    assert result.units == "m"
    assert "0.170" in result.message and "0.30" in result.message


def test_min_clearance_passes_at_exactly_the_threshold(make_trace):
    result = _evaluate(make_trace, {"type": "min_clearance", "threshold": 0.2}, clearance=[0.5, 0.2, 0.4])
    assert result.passed is True
    assert result.worst_value == pytest.approx(0.2)


def test_min_clearance_after_window_skips_early_samples(make_trace):
    clearance = [0.01, 0.02, 0.5, 0.6]
    result = _evaluate(make_trace, {"type": "min_clearance", "threshold": 0.2, "after": 0.2}, clearance=clearance)
    assert result.passed is True
    assert result.worst_value == pytest.approx(0.5)


def test_min_clearance_requires_a_threshold(make_trace):
    with pytest.raises(ScenarioError) as exc:
        _evaluate(make_trace, {"type": "min_clearance"}, clearance=[1.0])
    assert exc.value.key == "assertions[0].threshold"


def test_min_clearance_with_no_obstacles_is_unbounded(make_trace):
    result = _evaluate(make_trace, {"type": "min_clearance", "threshold": 0.2}, clearance=[math.inf, math.inf])
    assert result.passed is True
    assert result.worst_value is None


# -- collision -------------------------------------------------------------


def test_no_collision_reports_the_first_flagged_step(make_trace):
    result = _evaluate(
        make_trace,
        {"type": "no_collision"},
        collided=[0.0, 0.0, 1.0, 1.0],
        clearance=[0.5, 0.1, -0.05, -0.2],
    )
    assert result.passed is False
    assert result.worst_index == 2
    assert result.worst_time == pytest.approx(0.2)
    assert result.worst_value == pytest.approx(-0.05)


def test_no_collision_passes_and_reports_closest_approach(make_trace):
    result = _evaluate(make_trace, {"type": "no_collision"}, collided=[0.0] * 4, clearance=[0.9, 0.3, 0.11, 0.7])
    assert result.passed is True
    assert result.worst_value == pytest.approx(0.11)
    assert result.worst_index == 2


def test_no_collision_names_the_obstacle(make_trace):
    scenario_overrides = {"obstacles": [{"id": "pillar_a", "x": 5.0, "y": 0.0, "radius": 0.5}]}
    result = _evaluate(
        make_trace,
        {"type": "no_collision"},
        scenario_overrides=scenario_overrides,
        x=[0.0, 4.9],
        collided=[0.0, 1.0],
        clearance=[5.0, -0.1],
    )
    assert result.details["obstacle"] == "pillar_a"


# -- geofence --------------------------------------------------------------


def test_geofence_uses_world_bounds_by_default(make_trace):
    result = _evaluate(make_trace, {"type": "geofence"}, x=[0.0, 4.0], y=[0.0, 0.0], z=[0.0, 0.0])
    assert result.passed is True


def test_geofence_reports_the_excursion(make_trace):
    result = _evaluate(make_trace, {"type": "geofence"}, x=[0.0, 14.0, 16.5], y=[0.0, 0.0, 0.0])
    assert result.passed is False
    assert result.worst_index == 2
    assert result.worst_value == pytest.approx(-1.5)


def test_geofence_accepts_an_explicit_box(make_trace):
    result = _evaluate(
        make_trace,
        {"type": "geofence", "min_xy": [-1.0, -1.0], "max_xy": [2.0, 1.0], "min_z": -5.0, "max_z": 5.0},
        x=[0.0, 3.0],
        y=[0.0, 0.0],
    )
    assert result.passed is False
    assert result.worst_value == pytest.approx(-1.0)


def test_geofence_accepts_a_polygon(make_trace):
    polygon = [[-1.0, -1.0], [5.0, -1.0], [5.0, 1.0], [-1.0, 1.0]]
    inside = _evaluate(make_trace, {"type": "geofence", "polygon": polygon, "min_z": -5.0, "max_z": 5.0}, x=[0.0, 4.0], y=[0.0, 0.0])
    outside = _evaluate(make_trace, {"type": "geofence", "polygon": polygon, "min_z": -5.0, "max_z": 5.0}, x=[0.0, 7.0], y=[0.0, 0.0])
    assert inside.passed is True
    assert outside.passed is False


def test_geofence_polygon_needs_three_vertices(make_trace):
    with pytest.raises(ScenarioError):
        _evaluate(make_trace, {"type": "geofence", "polygon": [[0, 0], [1, 1]]}, x=[0.0])


# -- reached goal ----------------------------------------------------------


def test_reached_goal_reports_the_arrival_time(make_trace):
    result = _evaluate(make_trace, {"type": "reached_goal"}, x=[0.0, 5.0, 9.9, 10.0])
    assert result.passed is True
    assert result.worst_index == 2
    assert result.worst_time == pytest.approx(0.2)


def test_reached_goal_fails_and_reports_closest_approach(make_trace):
    result = _evaluate(make_trace, {"type": "reached_goal"}, x=[0.0, 3.0, 6.0])
    assert result.passed is False
    assert result.worst_value == pytest.approx(4.0)
    assert result.details["closest_approach_m"] == pytest.approx(4.0)


def test_reached_goal_deadline_is_enforced(make_trace):
    result = _evaluate(make_trace, {"type": "reached_goal", "within": 0.1}, x=[0.0, 5.0, 10.0])
    assert result.passed is False
    assert "deadline" in result.message
    assert result.worst_time == pytest.approx(0.2)


def test_reached_goal_after_window_ignores_the_start(make_trace):
    # Goal is 10 m away; a trace that starts at the goal, leaves and returns.
    result = _evaluate(make_trace, {"type": "reached_goal", "after": 0.2}, x=[10.0, 4.0, 2.0, 10.0])
    assert result.passed is True
    assert result.worst_index == 3


# -- limits ----------------------------------------------------------------


def test_max_velocity_reports_the_peak(make_trace):
    result = _evaluate(make_trace, {"type": "max_velocity", "limit": 1.0}, vx=[0.0, 0.5, 1.4, 0.2])
    assert result.passed is False
    assert result.worst_value == pytest.approx(1.4)
    assert result.worst_index == 2


def test_max_acceleration_uses_backward_differences(make_trace):
    # vx steps 0 -> 0 -> 1 over dt = 0.1, so accel = 10 m/s^2 at index 2.
    result = _evaluate(make_trace, {"type": "max_acceleration", "limit": 5.0}, vx=[0.0, 0.0, 1.0, 1.0])
    assert result.passed is False
    assert result.worst_value == pytest.approx(10.0)
    assert result.worst_index == 2


def test_first_sample_derivative_is_zero_not_extrapolated(make_trace):
    result = _evaluate(make_trace, {"type": "max_acceleration", "limit": 1e9}, vx=[5.0, 5.0])
    assert result.worst_value == pytest.approx(0.0)


def test_max_jerk_is_the_derivative_of_acceleration(make_trace):
    result = _evaluate(make_trace, {"type": "max_jerk", "limit": 1e9}, vx=[0.0, 0.0, 1.0, 1.0])
    assert result.worst_value == pytest.approx(100.0)  # accel 0 -> 10 over dt = 0.1


def test_heading_error_bounds(make_trace):
    # Goal is at +x, so a yaw of pi is a 180 degree heading error.
    result = _evaluate(make_trace, {"type": "heading_error", "limit": 0.5}, yaw=[0.0, math.pi])
    assert result.passed is False
    assert result.worst_value == pytest.approx(math.pi, abs=1e-9)


# -- path length -----------------------------------------------------------


def test_path_length_ratio_against_the_straight_line(make_trace):
    result = _evaluate(make_trace, {"type": "path_length", "max_ratio": 1.2}, x=[0.0, 5.0, 10.0])
    assert result.passed is True
    assert result.worst_value == pytest.approx(1.0)
    assert result.details["optimal_m"] == pytest.approx(10.0)


def test_path_length_detects_a_detour(make_trace):
    result = _evaluate(
        make_trace, {"type": "path_length", "max_ratio": 1.2}, x=[0.0, 5.0, 5.0, 10.0], y=[0.0, 0.0, 5.0, 0.0]
    )
    assert result.passed is False
    assert result.details["path_length_m"] == pytest.approx(5.0 + 5.0 + math.hypot(5.0, 5.0))


def test_path_length_accepts_an_explicit_optimum(make_trace):
    result = _evaluate(make_trace, {"type": "path_length", "max_ratio": 1.1, "optimal": 20.0}, x=[0.0, 10.0])
    assert result.passed is True
    assert result.worst_value == pytest.approx(0.5)


# -- settling and overshoot -----------------------------------------------


def test_settling_time_uses_the_last_entry_into_the_band(make_trace):
    result = _evaluate(
        make_trace,
        {"type": "settling_time", "signal": "x", "target": 10.0, "band": 0.5, "limit": 0.5},
        x=[0.0, 9.8, 5.0, 9.9, 10.0, 10.1],
    )
    assert result.passed is True
    assert result.worst_time == pytest.approx(0.3)


def test_settling_time_fails_when_the_signal_leaves_the_band_at_the_end(make_trace):
    result = _evaluate(
        make_trace,
        {"type": "settling_time", "signal": "x", "target": 10.0, "band": 0.5, "limit": 5.0},
        x=[10.0, 10.0, 10.0, 3.0],
    )
    assert result.passed is False
    assert "never settled" in result.message


def test_settling_time_fails_past_its_deadline(make_trace):
    result = _evaluate(
        make_trace,
        {"type": "settling_time", "signal": "x", "target": 10.0, "band": 0.5, "limit": 0.1},
        x=[0.0, 0.0, 0.0, 10.0, 10.0],
    )
    assert result.passed is False
    assert result.worst_value == pytest.approx(0.3)


def test_overshoot_percentage(make_trace):
    # distance_to_goal starts at 10 and dips 1 m past the goal: 10% overshoot.
    result = _evaluate(make_trace, {"type": "overshoot", "signal": "x", "target": 10.0, "max_percent": 5.0}, x=[0.0, 8.0, 11.0, 10.0])
    assert result.passed is False
    assert result.worst_value == pytest.approx(10.0)
    assert result.worst_index == 2


def test_no_overshoot_reports_zero(make_trace):
    result = _evaluate(make_trace, {"type": "overshoot", "signal": "x", "target": 10.0, "max_percent": 5.0}, x=[0.0, 8.0, 10.0])
    assert result.passed is True
    assert result.worst_value == pytest.approx(0.0)


# -- oscillation -----------------------------------------------------------


def test_no_oscillation_detects_a_known_frequency(make_trace):
    # 2 Hz sine, sampled at 100 Hz for 2 s: 8 zero crossings over 2 s -> 2 Hz.
    dt = 0.01
    values = [math.sin(2.0 * math.pi * 2.0 * i * dt) for i in range(201)]
    result = _evaluate(
        make_trace,
        {"type": "no_oscillation", "signal": "x", "max_frequency": 1.0, "min_amplitude": 0.1},
        dt=dt,
        x=values,
    )
    assert result.passed is False
    assert result.worst_value == pytest.approx(2.0, abs=0.3)


def test_no_oscillation_passes_below_the_frequency_limit(make_trace):
    dt = 0.01
    values = [math.sin(2.0 * math.pi * 0.25 * i * dt) for i in range(201)]
    result = _evaluate(
        make_trace,
        {"type": "no_oscillation", "signal": "x", "max_frequency": 1.0, "min_amplitude": 0.1},
        dt=dt,
        x=values,
    )
    assert result.passed is True


def test_no_oscillation_ignores_small_amplitude_noise(make_trace):
    values = [0.001 * (1 if i % 2 else -1) for i in range(50)]
    result = _evaluate(
        make_trace, {"type": "no_oscillation", "signal": "x", "max_frequency": 0.5, "min_amplitude": 0.05}, x=values
    )
    assert result.passed is True
    assert "below" in result.message


def test_no_oscillation_detrends_a_ramp(make_trace):
    values = [0.05 * i for i in range(60)]
    result = _evaluate(
        make_trace, {"type": "no_oscillation", "signal": "x", "max_frequency": 0.5, "min_amplitude": 0.01}, x=values
    )
    assert result.passed is True, "a monotone ramp is not an oscillation"


# -- energy ----------------------------------------------------------------


def test_energy_budget_in_watt_hours(make_trace):
    joules = [0.0, 1800.0, 3600.0, 7200.0]
    result = _evaluate(make_trace, {"type": "energy_budget", "max_wh": 1.5}, energy_j=joules)
    assert result.passed is False
    assert result.worst_value == pytest.approx(2.0)
    assert result.units == "Wh"


def test_energy_budget_in_joules_passes(make_trace):
    result = _evaluate(make_trace, {"type": "energy_budget", "max_j": 100.0}, energy_j=[0.0, 40.0, 80.0])
    assert result.passed is True
    assert result.details["fraction_of_budget"] == pytest.approx(0.8)


def test_energy_budget_requires_a_limit(make_trace):
    with pytest.raises(ScenarioError):
        _evaluate(make_trace, {"type": "energy_budget"}, energy_j=[0.0])


# -- empty traces and summarising -----------------------------------------


def test_assertions_on_an_empty_trace_do_not_crash():
    from simharness.trace import Trace

    scenario = Scenario.from_dict(minimal_scenario_dict())
    results = evaluate_all(Trace(), scenario)
    assert len(results) == 1
    assert results[0].passed is False
    assert "empty" in results[0].message


def test_summarise_counts_and_names_failures(make_trace):
    good = _evaluate(make_trace, {"type": "min_clearance", "name": "ok", "threshold": 0.1}, clearance=[1.0])
    bad = _evaluate(make_trace, {"type": "min_clearance", "name": "bad", "threshold": 5.0}, clearance=[1.0])
    summary = summarise([good, bad])
    assert summary == {"total": 2, "passed": 1, "failed": 1, "all_passed": False, "failures": ["bad"]}
