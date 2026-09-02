"""Parameter sweeps, Monte Carlo, and failure-boundary bisection.

One passing run tells you the controller works at *that* wind speed, with
*that* noise seed, at *those* gains. It tells you nothing about the envelope.
This module answers the question a reviewer actually asks: **at what value does
it start failing?**

Three tools:

``grid_sweep``
    Cross-product of explicit parameter values. Reports a pass rate per cell.

``monte_carlo``
    Random samples from ranges, with the noise seed varied per sample so you
    are measuring robustness rather than re-running one lucky seed.

``find_failure_boundary``
    Bisection on a monotone pass/fail predicate. Give it a value where the
    scenario passes and one where it fails, and it returns the crossover to a
    tolerance you choose, plus the number of runs it took. This is the number
    that goes in a spec: "holds up to 4.6 m/s of crosswind, fails by 4.8".

Monotonicity is an assumption, and the code says so out loud: if the endpoints
do not bracket a transition, you get a result with ``status`` explaining why
rather than a confident wrong answer.
"""

from __future__ import annotations

import itertools
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .runner import GREEN_STATUSES, RunResult, run_scenario
from .scenario import Scenario, ScenarioError

__all__ = [
    "SweepPoint",
    "SweepResult",
    "BoundaryResult",
    "grid_sweep",
    "monte_carlo",
    "find_failure_boundary",
    "find_scenario_boundary",
]

LOG = logging.getLogger("simharness.sweep")


@dataclass
class SweepPoint:
    """One scenario variant and how it went."""

    index: int
    overrides: Dict[str, Any]
    status: str
    passed: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    failed_assertions: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def label(self) -> str:
        return ", ".join(f"{k}={_fmt(v)}" for k, v in self.overrides.items())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "overrides": self.overrides,
            "status": self.status,
            "passed": self.passed,
            "metrics": self.metrics,
            "failed_assertions": self.failed_assertions,
            "error": self.error,
        }


@dataclass
class SweepResult:
    """Every point in a sweep plus the aggregate pass rate."""

    scenario: str
    mode: str
    parameters: List[str]
    points: List[SweepPoint] = field(default_factory=list)
    wall_time_s: float = 0.0

    @property
    def total(self) -> int:
        return len(self.points)

    @property
    def passed(self) -> int:
        return sum(1 for p in self.points if p.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def failures(self) -> List[SweepPoint]:
        return [p for p in self.points if not p.passed]

    def pass_rate_by(self, parameter: str) -> List[Tuple[Any, int, int]]:
        """``(value, passed, total)`` for each distinct value of ``parameter``."""
        buckets: Dict[Any, List[int]] = {}
        for point in self.points:
            if parameter not in point.overrides:
                continue
            key = point.overrides[parameter]
            bucket = buckets.setdefault(key, [0, 0])
            bucket[1] += 1
            if point.passed:
                bucket[0] += 1
        return [(value, counts[0], counts[1]) for value, counts in sorted(buckets.items(), key=_sort_key)]

    def format_table(self, parameter: Optional[str] = None) -> str:
        """A plain-text pass-rate table, suitable for pasting into a report."""
        parameter = parameter or (self.parameters[0] if self.parameters else "")
        rows = self.pass_rate_by(parameter) if parameter else []
        if not rows:
            return f"{self.scenario}: {self.passed}/{self.total} passed ({self.pass_rate * 100:.1f}%)"
        width = max(len(parameter), max(len(_fmt(v)) for v, _, _ in rows))
        lines = [f"{parameter:>{width}}  passed  total  rate", f"{'-' * width}  ------  -----  ----"]
        for value, passed, total in rows:
            lines.append(f"{_fmt(value):>{width}}  {passed:>6}  {total:>5}  {passed / total * 100:5.1f}%")
        lines.append(f"{'total':>{width}}  {self.passed:>6}  {self.total:>5}  {self.pass_rate * 100:5.1f}%")
        return "\n".join(lines)

    def first_failing_value(self, parameter: str) -> Optional[Any]:
        """Lowest value of ``parameter`` at which any point failed."""
        failing = [p.overrides[parameter] for p in self.failures() if parameter in p.overrides]
        if not failing:
            return None
        try:
            return min(failing)
        except TypeError:  # mixed types: fall back to first seen
            return failing[0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "mode": self.mode,
            "parameters": self.parameters,
            "total": self.total,
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "wall_time_s": self.wall_time_s,
            "points": [p.to_dict() for p in self.points],
        }


def _sort_key(item: Tuple[Any, Any]) -> Tuple[int, Any]:
    value = item[0]
    if isinstance(value, (int, float)):
        return (0, value)
    return (1, str(value))


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


# --------------------------------------------------------------------------
# running variants
# --------------------------------------------------------------------------


def _evaluate_point(
    scenario: Scenario,
    index: int,
    overrides: Mapping[str, Any],
    run_options: Mapping[str, Any],
) -> SweepPoint:
    try:
        variant = scenario.with_overrides(overrides)
    except ScenarioError as exc:
        return SweepPoint(
            index=index,
            overrides=dict(overrides),
            status="error",
            passed=False,
            error=f"ScenarioError: {exc}",
        )
    result = run_scenario(variant, keep_trace=False, **run_options)
    return _point_from_result(index, overrides, result)


def _point_from_result(index: int, overrides: Mapping[str, Any], result: RunResult) -> SweepPoint:
    return SweepPoint(
        index=index,
        overrides=dict(overrides),
        status=result.status,
        passed=result.status in GREEN_STATUSES,
        metrics=dict(result.metrics),
        failed_assertions=[a.name for a in result.failed_assertions],
        error=result.error,
    )


def _run_points(
    scenario: Scenario,
    combos: Sequence[Mapping[str, Any]],
    run_options: Mapping[str, Any],
    workers: int,
) -> List[SweepPoint]:
    if workers <= 1 or len(combos) <= 1:
        return [_evaluate_point(scenario, i, combo, run_options) for i, combo in enumerate(combos)]

    from concurrent.futures import ProcessPoolExecutor, as_completed

    ordered: Dict[int, SweepPoint] = {}
    pending = dict(enumerate(combos))
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_evaluate_point, scenario, i, dict(combo), dict(run_options)): i
                for i, combo in enumerate(combos)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    ordered[i] = future.result()
                except Exception as exc:
                    ordered[i] = SweepPoint(
                        index=i,
                        overrides=dict(combos[i]),
                        status="error",
                        passed=False,
                        error=f"worker failed: {type(exc).__name__}: {exc}",
                    )
                pending.pop(i, None)
    except Exception as exc:  # pool died: finish serially rather than lose the sweep
        LOG.warning("sweep worker pool failed (%s); finishing %d point(s) serially", exc, len(pending))
        for i, combo in sorted(pending.items()):
            ordered[i] = _evaluate_point(scenario, i, combo, run_options)
    return [ordered[i] for i in sorted(ordered)]


def grid_sweep(
    scenario: Scenario,
    parameters: Mapping[str, Sequence[Any]],
    *,
    workers: int = 1,
    simulator: Optional[str] = None,
    wall_timeout_s: Optional[float] = None,
) -> SweepResult:
    """Run the full cross-product of ``parameters``.

    ``parameters`` maps dotted scenario paths to the values to try::

        grid_sweep(scenario, {"disturbance.wind[0]": [0, 1, 2, 3],
                              "controller.kp_linear": [0.6, 1.0]})
    """
    if not parameters:
        raise ValueError("grid_sweep() needs at least one parameter")
    names = list(parameters)
    for name in names:
        if not parameters[name]:
            raise ValueError(f"parameter '{name}' has no values to sweep")
    combos = [dict(zip(names, values)) for values in itertools.product(*(parameters[n] for n in names))]
    started = time.perf_counter()
    options = {"simulator": simulator, "wall_timeout_s": wall_timeout_s}
    points = _run_points(scenario, combos, options, workers)
    return SweepResult(
        scenario=scenario.name,
        mode="grid",
        parameters=names,
        points=points,
        wall_time_s=time.perf_counter() - started,
    )


def monte_carlo(
    scenario: Scenario,
    parameters: Mapping[str, Tuple[float, float]],
    *,
    samples: int = 50,
    seed: int = 0,
    vary_sim_seed: bool = True,
    workers: int = 1,
    simulator: Optional[str] = None,
    wall_timeout_s: Optional[float] = None,
) -> SweepResult:
    """Sample ``parameters`` uniformly from ``(low, high)`` ranges.

    ``vary_sim_seed`` also redraws ``sim.seed`` for each sample, which is
    usually what you want: holding the noise seed fixed while varying wind
    measures the controller against one particular realisation of the noise,
    not against the noise.

    The sampling RNG is seeded from ``seed``, so the whole sweep is repeatable.
    """
    if samples < 1:
        raise ValueError("monte_carlo() needs samples >= 1")
    rng = random.Random(f"simharness:sweep:{seed}")
    names = list(parameters)
    for name in names:
        low, high = parameters[name]
        if high < low:
            raise ValueError(f"parameter '{name}' has an inverted range ({low} > {high})")
    combos: List[Dict[str, Any]] = []
    for _ in range(samples):
        combo: Dict[str, Any] = {}
        for name in names:
            low, high = parameters[name]
            combo[name] = rng.uniform(low, high)
        if vary_sim_seed:
            combo["sim.seed"] = rng.randrange(1, 2**31 - 1)
        combos.append(combo)
    started = time.perf_counter()
    options = {"simulator": simulator, "wall_timeout_s": wall_timeout_s}
    points = _run_points(scenario, combos, options, workers)
    return SweepResult(
        scenario=scenario.name,
        mode="monte_carlo",
        parameters=names + (["sim.seed"] if vary_sim_seed else []),
        points=points,
        wall_time_s=time.perf_counter() - started,
    )


# --------------------------------------------------------------------------
# failure boundary
# --------------------------------------------------------------------------


@dataclass
class BoundaryResult:
    """Where a scenario stops passing.

    ``status`` is one of:

    ``bracketed``
        Normal result. ``last_pass`` and ``first_fail`` bracket the boundary to
        within ``tolerance``.
    ``passes_everywhere``
        The scenario also passed at ``high``: the envelope is wider than the
        search range, so widen it.
    ``fails_everywhere``
        The scenario already failed at ``low``: it is broken below the search
        range, so lower it or fix the scenario.
    """

    parameter: str
    status: str
    low: float
    high: float
    tolerance: float
    last_pass: Optional[float] = None
    first_fail: Optional[float] = None
    iterations: int = 0
    evaluations: int = 0
    history: List[Tuple[float, bool]] = field(default_factory=list)
    note: str = ""

    @property
    def boundary(self) -> Optional[float]:
        """Midpoint of the bracket: the best single-number answer."""
        if self.last_pass is None or self.first_fail is None:
            return None
        return 0.5 * (self.last_pass + self.first_fail)

    @property
    def found(self) -> bool:
        return self.status == "bracketed"

    def summary(self) -> str:
        if self.status == "bracketed":
            return (
                f"{self.parameter}: passes up to {self.last_pass:.6g}, fails from {self.first_fail:.6g} "
                f"-> boundary {self.boundary:.6g} +/-{(self.first_fail - self.last_pass) / 2:.3g} "
                f"after {self.evaluations} runs"
            )
        if self.status == "passes_everywhere":
            return f"{self.parameter}: still passing at {self.high:.6g}; no boundary inside the search range"
        if self.status == "fails_everywhere":
            return f"{self.parameter}: already failing at {self.low:.6g}; the boundary is below the search range"
        return f"{self.parameter}: {self.note}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter": self.parameter,
            "status": self.status,
            "low": self.low,
            "high": self.high,
            "tolerance": self.tolerance,
            "last_pass": self.last_pass,
            "first_fail": self.first_fail,
            "boundary": self.boundary,
            "iterations": self.iterations,
            "evaluations": self.evaluations,
            "history": [[v, p] for v, p in self.history],
            "note": self.note,
            "summary": self.summary(),
        }


def find_failure_boundary(
    predicate: Callable[[float], bool],
    *,
    low: float,
    high: float,
    tolerance: float = 0.01,
    max_iterations: int = 40,
    parameter: str = "value",
) -> BoundaryResult:
    """Bisect a monotone pass/fail ``predicate`` between ``low`` and ``high``.

    ``predicate(value)`` returns ``True`` for pass. The search assumes passing
    at ``low`` and failing at ``high``; both endpoints are evaluated first and
    a missing bracket is reported rather than papered over.

    The returned bracket satisfies ``first_fail - last_pass <= tolerance``.
    """
    if high <= low:
        raise ValueError(f"find_failure_boundary() needs high > low, got low={low}, high={high}")
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be > 0, got {tolerance}")

    history: List[Tuple[float, bool]] = []

    def check(value: float) -> bool:
        outcome = bool(predicate(value))
        history.append((value, outcome))
        return outcome

    result = BoundaryResult(parameter=parameter, status="bracketed", low=low, high=high, tolerance=tolerance)

    if not check(low):
        result.status = "fails_everywhere"
        result.first_fail = low
        result.evaluations = len(history)
        result.history = history
        result.note = f"predicate already false at low={low:g}"
        return result
    if check(high):
        result.status = "passes_everywhere"
        result.last_pass = high
        result.evaluations = len(history)
        result.history = history
        result.note = f"predicate still true at high={high:g}"
        return result

    last_pass, first_fail = low, high
    iterations = 0
    while first_fail - last_pass > tolerance and iterations < max_iterations:
        mid = 0.5 * (last_pass + first_fail)
        if check(mid):
            last_pass = mid
        else:
            first_fail = mid
        iterations += 1

    result.last_pass = last_pass
    result.first_fail = first_fail
    result.iterations = iterations
    result.evaluations = len(history)
    result.history = history
    if first_fail - last_pass > tolerance:
        result.note = f"hit max_iterations={max_iterations} before reaching the tolerance"
    return result


def find_scenario_boundary(
    scenario: Scenario,
    parameter: str,
    *,
    low: float,
    high: float,
    tolerance: float = 0.05,
    max_iterations: int = 40,
    simulator: Optional[str] = None,
    wall_timeout_s: Optional[float] = None,
) -> BoundaryResult:
    """Bisect a scenario parameter to find where the scenario starts failing.

    Example: ``find_scenario_boundary(sc, "disturbance.wind[0]", low=0, high=6)``
    answers "how much headwind before the vehicle misses its deadline".
    """

    def predicate(value: float) -> bool:
        variant = scenario.with_overrides({parameter: value})
        outcome = run_scenario(
            variant, keep_trace=False, simulator=simulator, wall_timeout_s=wall_timeout_s
        )
        LOG.debug("boundary probe %s=%g -> %s", parameter, value, outcome.status)
        return outcome.status in GREEN_STATUSES

    return find_failure_boundary(
        predicate,
        low=low,
        high=high,
        tolerance=tolerance,
        max_iterations=max_iterations,
        parameter=parameter,
    )
