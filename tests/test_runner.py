"""Runner behaviour: statuses, expected failures, crash isolation, suites."""

from __future__ import annotations

from pathlib import Path

from conftest import minimal_scenario_dict
from simharness.runner import (
    GREEN_STATUSES,
    STATUS_ERROR,
    STATUS_EXPECTED_FAILURE,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_TIMEOUT,
    STATUS_UNEXPECTED_PASS,
    RunResult,
    SuiteResult,
    run_scenario,
    run_scenario_file,
    run_suite,
)
from simharness.scenario import Scenario, load_scenario, load_scenario_dir
from simharness.simulators.base import Simulator, SimulatorError


def _passing() -> Scenario:
    return Scenario.from_dict(
        minimal_scenario_dict(
            sim={"dt": 0.05, "time_limit": 30.0, "seed": 1},
            assertions=[{"type": "reached_goal", "name": "arrives"}, {"type": "no_collision"}],
        )
    )


def _failing() -> Scenario:
    data = minimal_scenario_dict(
        sim={"dt": 0.05, "time_limit": 30.0, "seed": 1},
        assertions=[{"type": "min_clearance", "name": "impossible", "threshold": 99.0}],
        obstacles=[{"id": "blob", "x": 6.0, "y": 3.0, "radius": 0.5}],
    )
    return Scenario.from_dict(data)


def test_a_passing_scenario_is_reported_as_pass():
    result = run_scenario(_passing())
    assert result.status == STATUS_PASS
    assert result.ok is True
    assert result.error is None
    assert result.simulator == "mock"
    assert len(result.assertions) == 2
    assert result.steps > 1


def test_a_failing_scenario_is_reported_as_fail():
    result = run_scenario(_failing())
    assert result.status == STATUS_FAIL
    assert result.ok is False
    assert [a.name for a in result.failed_assertions] == ["impossible"]


def test_expect_failure_turns_a_failure_into_a_green_xfail():
    data = _failing().to_dict()
    data["expect_failure"] = True
    result = run_scenario(Scenario.from_dict(data))
    assert result.status == STATUS_EXPECTED_FAILURE
    assert result.ok is True
    assert STATUS_EXPECTED_FAILURE in GREEN_STATUSES


def test_expect_failure_that_passes_is_an_unexpected_pass_and_fails_the_suite():
    data = _passing().to_dict()
    data["expect_failure"] = True
    result = run_scenario(Scenario.from_dict(data))
    assert result.status == STATUS_UNEXPECTED_PASS
    assert result.ok is False


def test_metrics_are_measured_from_the_trace():
    result = run_scenario(_passing())
    assert result.metrics["path_length_m"] > 9.0
    assert result.metrics["final_distance_to_goal_m"] <= 0.25
    assert result.metrics["time_to_goal_s"] > 0.0
    assert result.metrics["steps"] == float(result.steps)


def test_trace_metadata_describes_the_run():
    result = run_scenario(_passing())
    meta = result.trace.meta
    assert meta["robot"] == "diff_drive"
    assert meta["goal"] == [10.0, 0.0, 0.0]
    assert meta["connection"]
    assert result.trace.scenario == "unit"


def test_keep_trace_false_drops_the_trace_but_keeps_the_metrics():
    result = run_scenario(_passing(), keep_trace=False)
    assert result.trace is None
    assert result.metrics


def test_trace_dir_writes_a_trace_file(tmp_path: Path):
    result = run_scenario(_passing(), trace_dir=str(tmp_path))
    assert result.trace_path is not None
    assert Path(result.trace_path).exists()
    assert Path(result.trace_path).name == "unit.trace.json"


# -- crash isolation -------------------------------------------------------


class _ExplodingSimulator(Simulator):
    """A backend that fails mid-run, the way a dropped RPC connection would."""

    name = "exploding"
    connection = "test double"

    def __init__(self, fail_after: int = 3) -> None:
        self.fail_after = fail_after
        self.steps = 0
        self.closed = False
        self._inner = None

    @classmethod
    def availability(cls):
        return (True, "test double")

    def reset(self, scenario):
        from simharness.simulators.mock import MockSimulator

        self._inner = MockSimulator()
        return self._inner.reset(scenario)

    def step(self, dt):
        self.steps += 1
        if self.steps > self.fail_after:
            raise SimulatorError("backend connection dropped")
        return self._inner.step(dt)

    def get_state(self):
        return self._inner.get_state()

    def send_command(self, command):
        self._inner.send_command(command)

    def spawn(self, model_id, pose, **kwargs):
        self._inner.spawn(model_id, pose, **kwargs)

    def close(self):
        self.closed = True


def test_a_backend_that_dies_becomes_an_error_result_not_an_exception():
    result = run_scenario(_passing(), simulator_instance=_ExplodingSimulator())
    assert result.status == STATUS_ERROR
    assert result.ok is False
    assert "connection dropped" in result.error
    assert "SimulatorError" in result.traceback


def test_one_bad_scenario_does_not_stop_the_suite():
    class _AlwaysExplodes(_ExplodingSimulator):
        def __init__(self):
            super().__init__(fail_after=0)

    good = _passing()
    bad = Scenario.from_dict(dict(_passing().to_dict(), name="broken"))
    results = [run_scenario(good), run_scenario(bad, simulator_instance=_AlwaysExplodes()), run_scenario(good)]
    suite = SuiteResult(results=results)
    assert [r.status for r in results] == [STATUS_PASS, STATUS_ERROR, STATUS_PASS]
    assert suite.ok is False
    assert suite.by_status() == {STATUS_PASS: 2, STATUS_ERROR: 1}


def test_run_scenario_file_turns_a_bad_yaml_into_an_error_result(tmp_path: Path):
    path = tmp_path / "broken.yaml"
    path.write_text("name: broken\nrobot: {type: submarine}\n", encoding="utf-8")
    result = run_scenario_file(str(path))
    assert result.status == STATUS_ERROR
    assert "robot.type" in result.error


def test_wall_timeout_produces_a_timeout_status():
    slow = Scenario.from_dict(
        minimal_scenario_dict(
            sim={"dt": 0.001, "time_limit": 60.0, "seed": 1},
            goal={"x": 10.0, "y": 0.0, "tolerance": 0.01},
            assertions=[{"type": "no_collision"}],
        )
    )
    result = run_scenario(slow, wall_timeout_s=0.0)
    assert result.status == STATUS_TIMEOUT
    assert result.ok is False
    assert result.trace.events_of("wall_timeout")


# -- suites ----------------------------------------------------------------


def test_run_suite_preserves_input_order(scenario_dir: Path):
    scenarios = load_scenario_dir(scenario_dir)
    suite = run_suite(scenarios, keep_traces=False)
    assert [r.scenario for r in suite] == [s.name for s in scenarios]
    assert len(suite) == len(scenarios)


def test_run_suite_accepts_a_directory_path(scenario_dir: Path):
    suite = run_suite(str(scenario_dir), keep_traces=False)
    assert len(suite) >= 8
    assert suite.get("straight_line") is not None
    assert suite.get("not_a_scenario") is None


def test_shipped_suite_is_green_and_contains_one_expected_failure(scenario_dir: Path):
    suite = run_suite(str(scenario_dir), keep_traces=False, workers=2)
    counts = suite.by_status()
    assert suite.ok is True, [r.summary_line() for r in suite if not r.ok]
    assert counts.get(STATUS_EXPECTED_FAILURE) == 1
    assert counts.get(STATUS_PASS, 0) >= 7
    assert STATUS_ERROR not in counts


def test_expected_failure_scenario_is_not_an_error(scenario_dir: Path):
    result = run_scenario(load_scenario(scenario_dir / "expected_failure_no_avoidance.yaml"))
    assert result.status == STATUS_EXPECTED_FAILURE
    assert result.error is None
    assert result.traceback is None
    assert result.ok is True
    assert any(a.type == "no_collision" and not a.passed for a in result.assertions)


def test_suite_to_dict_is_serialisable(scenario_dir: Path):
    import json

    suite = run_suite(load_scenario_dir(scenario_dir)[:2], keep_traces=False)
    payload = json.loads(json.dumps(suite.to_dict()))
    assert payload["counts"]
    assert len(payload["results"]) == 2


def test_run_result_round_trips_through_dict():
    result = run_scenario(_passing(), keep_trace=False)
    again = RunResult.from_dict(result.to_dict())
    assert again.scenario == result.scenario
    assert again.status == result.status
    assert [a.name for a in again.assertions] == [a.name for a in result.assertions]
    assert again.metrics == result.metrics


def test_on_result_callback_fires_per_scenario(scenario_dir: Path):
    seen = []
    run_suite(load_scenario_dir(scenario_dir)[:3], keep_traces=False, on_result=seen.append)
    assert len(seen) == 3
