"""Scenario execution: fixed-step, seeded, isolated, parallel.

Four properties this module is responsible for.

**Determinism.** Every stochastic source in a run is derived from
``scenario.sim.seed``. There is no ``random`` module global use, no wall-clock
seeding, no dict-ordering dependence. Two runs of the same scenario with the
same seed produce byte-identical trace JSON, and ``tests/test_determinism.py``
asserts exactly that.

**Fixed stepping.** The loop advances by ``scenario.sim.dt`` and never by a
wall-clock delta. Step count is computed up front from ``time_limit / dt``, so
a slow machine produces the same trace as a fast one.

**Crash isolation.** A scenario that raises -- bad YAML, a backend that drops
its connection, a controller that divides by zero -- becomes a ``RunResult``
with ``status="error"`` and the traceback attached. It does not take the suite
down with it. In parallel mode, a worker process that dies outright is caught
and the remaining scenarios are re-run serially.

**Honest expected failures.** A scenario marked ``expect_failure: true`` that
fails is reported as ``expected_failure`` and does *not* fail the suite. The
same scenario passing is reported as ``unexpected_pass`` and *does* fail the
suite, because a regression test that has silently started passing is a
regression test you no longer have.
"""

from __future__ import annotations

import logging
import time
import traceback as tb_module
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .assertions import AssertionResult, evaluate_all, summarise
from .controllers import make_controller
from .scenario import Scenario, ScenarioError, load_scenario, load_scenario_dir
from .simulators import Simulator, create_simulator
from .simulators.base import Command
from .trace import Sample, Trace

__all__ = [
    "RunResult",
    "SuiteResult",
    "run_scenario",
    "run_scenario_file",
    "run_suite",
    "STATUS_PASS",
    "STATUS_FAIL",
    "STATUS_EXPECTED_FAILURE",
    "STATUS_UNEXPECTED_PASS",
    "STATUS_ERROR",
    "STATUS_TIMEOUT",
]

LOG = logging.getLogger("simharness.runner")

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_EXPECTED_FAILURE = "expected_failure"
STATUS_UNEXPECTED_PASS = "unexpected_pass"
STATUS_ERROR = "error"
STATUS_TIMEOUT = "timeout"

#: Statuses that do not fail the suite.
GREEN_STATUSES = (STATUS_PASS, STATUS_EXPECTED_FAILURE)


@dataclass
class RunResult:
    """The outcome of one scenario."""

    scenario: str
    status: str
    seed: int = 0
    simulator: str = ""
    expect_failure: bool = False
    sim_time_s: float = 0.0
    wall_time_s: float = 0.0
    steps: int = 0
    assertions: List[AssertionResult] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    trace: Optional[Trace] = None
    trace_path: Optional[str] = None
    source: Optional[str] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    tags: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Does this result leave the suite green?"""
        return self.status in GREEN_STATUSES

    @property
    def failed_assertions(self) -> List[AssertionResult]:
        return [a for a in self.assertions if not a.passed]

    def to_dict(self, *, include_trace: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "scenario": self.scenario,
            "status": self.status,
            "ok": self.ok,
            "seed": self.seed,
            "simulator": self.simulator,
            "expect_failure": self.expect_failure,
            "sim_time_s": self.sim_time_s,
            "wall_time_s": self.wall_time_s,
            "steps": self.steps,
            "metrics": self.metrics,
            "assertions": [a.to_dict() for a in self.assertions],
            "source": self.source,
            "tags": list(self.tags),
        }
        if self.error:
            out["error"] = self.error
            out["traceback"] = self.traceback
        if self.trace_path:
            out["trace_path"] = self.trace_path
        if include_trace and self.trace is not None:
            out["trace"] = self.trace.to_dict()
        return out

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "RunResult":
        """Rebuild a result from :meth:`to_dict` output (used by ``report``)."""
        result = RunResult(
            scenario=str(data.get("scenario", "")),
            status=str(data.get("status", STATUS_ERROR)),
            seed=int(data.get("seed", 0)),
            simulator=str(data.get("simulator", "")),
            expect_failure=bool(data.get("expect_failure", False)),
            sim_time_s=float(data.get("sim_time_s", 0.0)),
            wall_time_s=float(data.get("wall_time_s", 0.0)),
            steps=int(data.get("steps", 0)),
            metrics={k: float(v) for k, v in (data.get("metrics") or {}).items()},
            source=data.get("source"),
            error=data.get("error"),
            traceback=data.get("traceback"),
            trace_path=data.get("trace_path"),
            tags=tuple(data.get("tags", ())),
        )
        result.assertions = [
            AssertionResult(
                name=a.get("name", ""),
                type=a.get("type", ""),
                passed=bool(a.get("passed")),
                message=a.get("message", ""),
                worst_value=a.get("worst_value"),
                worst_time=a.get("worst_time"),
                worst_index=a.get("worst_index"),
                threshold=a.get("threshold"),
                units=a.get("units", ""),
                details=dict(a.get("details", {})),
            )
            for a in data.get("assertions", [])
        ]
        if data.get("trace"):
            result.trace = Trace.from_dict(data["trace"])
        return result

    def summary_line(self) -> str:
        counts = summarise(self.assertions)
        detail = f"{counts['passed']}/{counts['total']} assertions"
        if self.error:
            detail = self.error.splitlines()[0] if self.error else "error"
        return f"{self.status.upper():17s} {self.scenario:28s} {detail}  ({self.sim_time_s:.2f}s sim)"


@dataclass
class SuiteResult:
    """Every :class:`RunResult` from one suite invocation."""

    results: List[RunResult] = field(default_factory=list)
    wall_time_s: float = 0.0
    workers: int = 1

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)

    def by_status(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1
        return counts

    def get(self, name: str) -> Optional[RunResult]:
        for result in self.results:
            if result.scenario == name:
                return result
        return None

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "SuiteResult":
        """Rebuild a suite result from :meth:`to_dict` output."""
        return SuiteResult(
            results=[RunResult.from_dict(r) for r in data.get("results", [])],
            wall_time_s=float(data.get("wall_time_s", 0.0)),
            workers=int(data.get("workers", 1)),
        )

    def to_dict(self, *, include_traces: bool = False) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "wall_time_s": self.wall_time_s,
            "workers": self.workers,
            "counts": self.by_status(),
            "results": [r.to_dict(include_trace=include_traces) for r in self.results],
        }


# --------------------------------------------------------------------------
# single scenario
# --------------------------------------------------------------------------


def run_scenario(
    scenario: Scenario,
    *,
    simulator: Optional[str] = None,
    simulator_instance: Optional[Simulator] = None,
    keep_trace: bool = True,
    trace_dir: Optional[str] = None,
    wall_timeout_s: Optional[float] = None,
) -> RunResult:
    """Execute one scenario and evaluate its assertions.

    Never raises for a scenario-level problem: a failure inside the run is
    turned into a ``RunResult`` with ``status="error"``.
    """
    started = time.perf_counter()
    result = RunResult(
        scenario=scenario.name,
        status=STATUS_ERROR,
        seed=scenario.sim.seed,
        expect_failure=scenario.expect_failure,
        source=scenario.source,
        tags=scenario.tags,
    )
    sim: Optional[Simulator] = None
    try:
        sim = simulator_instance if simulator_instance is not None else create_simulator(simulator)
        result.simulator = sim.name
        trace, timed_out = _simulate(sim, scenario, wall_timeout_s)
        result.trace = trace
        result.steps = len(trace)
        result.sim_time_s = trace.duration
        result.assertions = evaluate_all(trace, scenario)
        result.metrics = _metrics(trace, scenario)
        result.status = _classify(result, timed_out)
        if trace_dir is not None:
            path = Path(trace_dir) / f"{scenario.name}.trace.json"
            trace.save_json(path)
            result.trace_path = str(path)
        if not keep_trace:
            result.trace = None
    except Exception as exc:  # crash isolation: one bad scenario must not kill the suite
        result.status = STATUS_ERROR
        result.error = f"{type(exc).__name__}: {exc}"
        result.traceback = tb_module.format_exc()
        LOG.warning("scenario '%s' errored: %s", scenario.name, result.error)
    finally:
        if sim is not None and simulator_instance is None:
            try:
                sim.close()
            except Exception:  # a failing close() must not mask the real result
                LOG.debug("simulator close() failed for '%s'", scenario.name, exc_info=True)
        result.wall_time_s = time.perf_counter() - started
    return result


def _simulate(sim: Simulator, scenario: Scenario, wall_timeout_s: Optional[float]) -> Tuple[Trace, bool]:
    """Run the fixed-step loop and return ``(trace, timed_out)``."""
    dt = scenario.sim.dt
    max_steps = int(round(scenario.sim.time_limit / dt))
    controller = make_controller(scenario)
    controller.reset()

    state = sim.reset(scenario)
    trace = Trace(
        scenario=scenario.name,
        simulator=sim.name,
        seed=scenario.sim.seed,
        dt=dt,
        meta={
            "robot": scenario.robot.type,
            "controller": scenario.controller.type,
            "time_limit_s": scenario.sim.time_limit,
            "goal": list(scenario.goal_xyz()),
            "goal_tolerance": scenario.goal.tolerance,
            "spawn": list(scenario.spawn_xyz()),
            "obstacles": [o.to_dict() for o in scenario.obstacles],
            "world": scenario.world.to_dict(),
            "connection": sim.connection,
        },
    )
    trace.record(_sample(state, Command()))

    goal_time: Optional[float] = None
    timed_out = False
    deadline = None if wall_timeout_s is None else time.perf_counter() + wall_timeout_s

    for step in range(max_steps):
        if deadline is not None and time.perf_counter() > deadline:
            timed_out = True
            trace.add_event(state.t, "wall_timeout", f"wall-clock budget of {wall_timeout_s}s exhausted at step {step}")
            break
        command = controller.compute(state, dt)
        sim.send_command(command)
        state = sim.step(dt)
        trace.record(_sample(state, command))
        for t, kind, detail in getattr(sim, "drain_events", list)():
            trace.add_event(t, kind, detail)

        distance = _distance_to_goal(state, scenario)
        if distance <= scenario.goal.tolerance:
            if goal_time is None:
                goal_time = state.t
                trace.add_event(state.t, "goal_reached", f"within {distance:.3f} m of the goal")
            if scenario.sim.stop_on_goal and state.t - goal_time >= scenario.sim.settle_time:
                break
        elif goal_time is not None and distance > scenario.goal.tolerance * 1.5:
            goal_time = None
            trace.add_event(state.t, "goal_lost", f"drifted back out to {distance:.3f} m")

    if not timed_out and len(trace) >= max_steps + 1:
        trace.add_event(state.t, "time_limit", f"hit the {scenario.sim.time_limit}s scenario time limit")
    trace.meta["goal_reached_at"] = goal_time
    trace.meta["controller_status"] = controller.status()
    return trace, timed_out


def _sample(state: Any, command: Command) -> Sample:
    return Sample(
        t=state.t,
        x=state.position[0],
        y=state.position[1],
        z=state.position[2],
        yaw=state.yaw,
        vx=state.velocity[0],
        vy=state.velocity[1],
        vz=state.velocity[2],
        yaw_rate=state.yaw_rate,
        cmd_x=command.linear[0],
        cmd_y=command.linear[1],
        cmd_z=command.linear[2],
        cmd_yaw_rate=command.yaw_rate,
        est_x=state.est_position[0],
        est_y=state.est_position[1],
        est_z=state.est_position[2],
        est_yaw=state.est_yaw,
        clearance=state.clearance,
        energy_j=state.energy_j,
        sensor_valid=state.sensor_valid,
        collided=state.collided,
    )


def _distance_to_goal(state: Any, scenario: Scenario) -> float:
    gx, gy, gz = scenario.goal_xyz()
    px, py, pz = state.position
    return ((px - gx) ** 2 + (py - gy) ** 2 + (pz - gz) ** 2) ** 0.5


def _metrics(trace: Trace, scenario: Scenario) -> Dict[str, float]:
    """Headline numbers, all measured from the trace, none invented."""
    if not len(trace):
        return {}
    last = trace.samples[-1]
    gx, gy, gz = scenario.goal_xyz()
    speeds = [s.speed() for s in trace.samples]
    clearances = [s.clearance for s in trace.samples if s.clearance != float("inf")]
    metrics = {
        "final_distance_to_goal_m": ((last.x - gx) ** 2 + (last.y - gy) ** 2 + (last.z - gz) ** 2) ** 0.5,
        "path_length_m": trace.path_length(),
        "max_speed_mps": max(speeds) if speeds else 0.0,
        "energy_wh": last.energy_j / 3600.0,
        "sim_time_s": trace.duration,
        "steps": float(len(trace)),
    }
    if clearances:
        metrics["min_clearance_m"] = min(clearances)
    goal_time = trace.meta.get("goal_reached_at")
    if goal_time is not None:
        metrics["time_to_goal_s"] = float(goal_time)
    return {k: round(v, 6) for k, v in metrics.items()}


def _classify(result: RunResult, timed_out: bool) -> str:
    if timed_out:
        return STATUS_TIMEOUT
    all_passed = all(a.passed for a in result.assertions)
    if result.expect_failure:
        return STATUS_UNEXPECTED_PASS if all_passed else STATUS_EXPECTED_FAILURE
    return STATUS_PASS if all_passed else STATUS_FAIL


def run_scenario_file(path: str, **kwargs: Any) -> RunResult:
    """Load a scenario YAML and run it. Load errors become error results."""
    try:
        scenario = load_scenario(path)
    except ScenarioError as exc:
        return RunResult(
            scenario=Path(path).stem,
            status=STATUS_ERROR,
            source=str(path),
            error=f"ScenarioError: {exc}",
            traceback=tb_module.format_exc(),
        )
    return run_scenario(scenario, **kwargs)


# --------------------------------------------------------------------------
# suites
# --------------------------------------------------------------------------


def _worker(payload: Tuple[Scenario, Dict[str, Any]]) -> RunResult:
    """Process-pool entry point. Must be module level to be picklable."""
    scenario, options = payload
    return run_scenario(scenario, **options)


def run_suite(
    scenarios: Iterable[Scenario] | str,
    *,
    simulator: Optional[str] = None,
    workers: int = 1,
    keep_traces: bool = True,
    trace_dir: Optional[str] = None,
    wall_timeout_s: Optional[float] = None,
    on_result: Optional[Any] = None,
) -> SuiteResult:
    """Run a set of scenarios, optionally across a worker pool.

    ``scenarios`` may be a directory path or an iterable of :class:`Scenario`.
    Results come back in the input order regardless of completion order, so a
    parallel run and a serial run produce the same report.
    """
    if isinstance(scenarios, (str, Path)):
        items = load_scenario_dir(scenarios)
    else:
        items = list(scenarios)
    options: Dict[str, Any] = {
        "simulator": simulator,
        "keep_trace": keep_traces,
        "trace_dir": trace_dir,
        "wall_timeout_s": wall_timeout_s,
    }
    started = time.perf_counter()
    workers = max(1, int(workers))

    if workers == 1 or len(items) <= 1:
        results = []
        for scenario in items:
            result = run_scenario(scenario, **options)
            if on_result is not None:
                on_result(result)
            results.append(result)
        return SuiteResult(results=results, wall_time_s=time.perf_counter() - started, workers=1)

    ordered: Dict[int, RunResult] = {}
    pending: Dict[int, Scenario] = dict(enumerate(items))
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, (scenario, options)): index for index, scenario in enumerate(items)}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = RunResult(
                        scenario=items[index].name,
                        status=STATUS_ERROR,
                        source=items[index].source,
                        error=f"worker process failed: {type(exc).__name__}: {exc}",
                        traceback=tb_module.format_exc(),
                    )
                ordered[index] = result
                pending.pop(index, None)
                if on_result is not None:
                    on_result(result)
    except Exception as exc:  # BrokenProcessPool and friends: finish the job serially
        LOG.warning("worker pool failed (%s); finishing %d scenario(s) serially", exc, len(pending))
        for index, scenario in sorted(pending.items()):
            result = run_scenario(scenario, **options)
            if result.error:
                result.error += " [re-run serially after worker pool failure]"
            ordered[index] = result
            if on_result is not None:
                on_result(result)

    results = [ordered[i] for i in sorted(ordered)]
    return SuiteResult(results=results, wall_time_s=time.perf_counter() - started, workers=workers)
