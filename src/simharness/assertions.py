"""Behavioural assertions over a recorded run.

This is the part that turns "I launched the simulator and it looked fine" into
a regression test. An assertion reads a :class:`~simharness.trace.Trace` and
returns a :class:`AssertionResult` carrying not just pass/fail but **the
worst-case value, the timestamp it happened at, and a sentence a human can
act on**. A failure that says ``min clearance 0.083 m at t=6.34 s (limit
0.200 m), obstacle 'pillar'`` is worth ten that say ``assert False``.

Two families live here.

**Scalar assertions** reduce the run to one number and compare it to a limit:
reached-goal, collision, clearance, geofence, velocity/acceleration/jerk,
path-length ratio, settling time, overshoot, heading error, oscillation
frequency, energy budget.

**Temporal assertions** quantify a predicate over time, in the style of linear
temporal logic:

============================ ===============================================
``always P``                 P holds at every step
``eventually P``             P holds at some step (optionally by a deadline)
``eventually_always P``      P holds from some step onward and never lapses
``P until Q``                Q eventually holds, and P holds at every step before
============================ ===============================================

Finite-trace semantics, stated plainly
--------------------------------------
A recorded run is finite, so these operators need a convention:

* ``eventually P`` with no deadline is satisfied by any step in the trace.
* ``eventually_always P`` is satisfied when there is a step after which P never
  becomes false again. On a finite trace that reduces to "P holds at the last
  step", which is why the assertion additionally reports the *stabilisation
  time* (the first step of that final run of trues) and accepts ``settle_by``
  to bound it. Without ``settle_by`` you are only testing the final step.
* ``until`` is **strong** until: Q must actually occur. ``A until B`` where B
  never happens is a failure, not a vacuous pass.
* Vacuous truth is reported, not hidden. ``always P`` on an empty trace passes
  and says so in the message.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .geometry import box_signed_distance, polygon_signed_distance, wrap_angle
from .scenario import AssertionSpec, Scenario, ScenarioError
from .trace import Trace

__all__ = [
    "AssertionResult",
    "EvalContext",
    "SIGNALS",
    "ASSERTIONS",
    "register_assertion",
    "build_signals",
    "compile_predicate",
    "evaluate_assertion",
    "evaluate_all",
    "summarise",
]

Predicate = Callable[["EvalContext"], List[bool]]

_OPS: Dict[str, Callable[[float, float], bool]] = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


# --------------------------------------------------------------------------
# result
# --------------------------------------------------------------------------


@dataclass
class AssertionResult:
    """Everything you need to debug one assertion without opening the trace."""

    name: str
    type: str
    passed: bool
    message: str
    worst_value: Optional[float] = None
    worst_time: Optional[float] = None
    worst_index: Optional[int] = None
    threshold: Optional[float] = None
    units: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "passed": self.passed,
            "message": self.message,
            "worst_value": self.worst_value,
            "worst_time": self.worst_time,
            "worst_index": self.worst_index,
            "threshold": self.threshold,
            "units": self.units,
            "details": self.details,
        }

    def __str__(self) -> str:
        return f"[{self.status}] {self.name}: {self.message}"


# --------------------------------------------------------------------------
# signals
# --------------------------------------------------------------------------


@dataclass
class EvalContext:
    """A trace plus every derived signal, computed once and shared."""

    trace: Trace
    scenario: Scenario
    signals: Dict[str, List[float]]

    @property
    def n(self) -> int:
        return len(self.trace.samples)

    @property
    def times(self) -> List[float]:
        return self.signals["t"]

    def signal(self, name: str, key: str = "signal") -> List[float]:
        if name not in self.signals:
            raise ScenarioError(key, f"unknown signal '{name}'; available: {', '.join(sorted(self.signals))}")
        return self.signals[name]

    def time_at(self, index: int) -> float:
        if not self.times:
            return 0.0
        return self.times[max(0, min(index, len(self.times) - 1))]


SIGNALS: Tuple[str, ...] = (
    "t",
    "x",
    "y",
    "z",
    "yaw",
    "vx",
    "vy",
    "vz",
    "speed",
    "accel",
    "jerk",
    "yaw_rate",
    "cmd_speed",
    "distance_to_goal",
    "clearance",
    "heading_error",
    "geofence_margin",
    "energy_j",
    "energy_wh",
    "estimate_error",
    "sensor_valid",
    "collided",
)


def build_signals(trace: Trace, scenario: Scenario) -> EvalContext:
    """Derive every signal the assertions can reference.

    Derivatives are computed by backward difference on the recorded (already
    quantised) trace, so two identical traces always give identical
    derivatives. Index 0 of every derivative is 0.0 rather than an
    extrapolation, because inventing a value at t=0 is how a jerk assertion
    ends up failing on the spawn step.
    """
    n = len(trace.samples)
    goal = scenario.goal_xyz()
    world = scenario.world

    sig: Dict[str, List[float]] = {name: [0.0] * n for name in SIGNALS}
    if n == 0:
        return EvalContext(trace=trace, scenario=scenario, signals=sig)

    for i, s in enumerate(trace.samples):
        sig["t"][i] = s.t
        sig["x"][i] = s.x
        sig["y"][i] = s.y
        sig["z"][i] = s.z
        sig["yaw"][i] = s.yaw
        sig["vx"][i] = s.vx
        sig["vy"][i] = s.vy
        sig["vz"][i] = s.vz
        sig["yaw_rate"][i] = s.yaw_rate
        sig["speed"][i] = s.speed()
        sig["cmd_speed"][i] = math.sqrt(s.cmd_x**2 + s.cmd_y**2 + s.cmd_z**2)
        sig["clearance"][i] = s.clearance
        sig["energy_j"][i] = s.energy_j
        sig["energy_wh"][i] = s.energy_j / 3600.0
        sig["sensor_valid"][i] = 1.0 if s.sensor_valid else 0.0
        sig["collided"][i] = 1.0 if s.collided else 0.0
        sig["distance_to_goal"][i] = math.dist((s.x, s.y, s.z), goal)
        sig["estimate_error"][i] = math.dist((s.x, s.y, s.z), (s.est_x, s.est_y, s.est_z))
        bearing = math.atan2(goal[1] - s.y, goal[0] - s.x)
        sig["heading_error"][i] = abs(wrap_angle(bearing - s.yaw))
        margin = box_signed_distance(s.x, s.y, world.min_xy, world.max_xy)
        margin = min(margin, s.z - world.min_z, world.max_z - s.z)
        sig["geofence_margin"][i] = margin

    for i in range(1, n):
        dt = trace.samples[i].t - trace.samples[i - 1].t
        if dt <= 0.0:
            continue
        dvx = trace.samples[i].vx - trace.samples[i - 1].vx
        dvy = trace.samples[i].vy - trace.samples[i - 1].vy
        dvz = trace.samples[i].vz - trace.samples[i - 1].vz
        sig["accel"][i] = math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz) / dt
    for i in range(1, n):
        dt = trace.samples[i].t - trace.samples[i - 1].t
        if dt <= 0.0:
            continue
        sig["jerk"][i] = abs(sig["accel"][i] - sig["accel"][i - 1]) / dt

    return EvalContext(trace=trace, scenario=scenario, signals=sig)


# --------------------------------------------------------------------------
# predicates
# --------------------------------------------------------------------------


def compile_predicate(spec: Any, key: str) -> Predicate:
    """Compile a predicate mapping into a callable returning a boolean series.

    Accepted forms::

        {signal: clearance, op: ">=", value: 0.2}
        {all: [<predicate>, ...]}
        {any: [<predicate>, ...]}
        {not: <predicate>}
    """
    if not isinstance(spec, Mapping):
        raise ScenarioError(key, f"predicate must be a mapping, got {type(spec).__name__}")
    if "all" in spec or "any" in spec:
        combiner = "all" if "all" in spec else "any"
        items = spec[combiner]
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)) or not items:
            raise ScenarioError(f"{key}.{combiner}", "expected a non-empty list of predicates")
        subs = [compile_predicate(item, f"{key}.{combiner}[{i}]") for i, item in enumerate(items)]
        reducer = all if combiner == "all" else any

        def _combined(ctx: EvalContext) -> List[bool]:
            series = [sub(ctx) for sub in subs]
            return [reducer(s[i] for s in series) for i in range(ctx.n)]

        return _combined
    if "not" in spec:
        inner = compile_predicate(spec["not"], f"{key}.not")
        return lambda ctx: [not v for v in inner(ctx)]

    unknown = set(spec) - {"signal", "op", "value"}
    if unknown:
        raise ScenarioError(key, f"unknown predicate key(s): {', '.join(sorted(unknown))}")
    if "signal" not in spec:
        raise ScenarioError(f"{key}.signal", "predicate needs a 'signal'")
    name = spec["signal"]
    if not isinstance(name, str):
        raise ScenarioError(f"{key}.signal", f"expected a signal name, got {type(name).__name__}")
    op = spec.get("op", "<=")
    if op not in _OPS:
        raise ScenarioError(f"{key}.op", f"unknown operator '{op}'; use one of {', '.join(_OPS)}")
    if "value" not in spec:
        raise ScenarioError(f"{key}.value", "predicate needs a 'value'")
    value = spec["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScenarioError(f"{key}.value", f"expected a number, got {type(value).__name__}")
    fn = _OPS[op]
    threshold = float(value)

    def _atom(ctx: EvalContext) -> List[bool]:
        return [fn(v, threshold) for v in ctx.signal(name, f"{key}.signal")]

    _atom.description = f"{name} {op} {threshold:g}"  # type: ignore[attr-defined]
    return _atom


def describe_predicate(spec: Any) -> str:
    """Render a predicate mapping as a readable expression."""
    if not isinstance(spec, Mapping):
        return str(spec)
    if "all" in spec:
        return "(" + " and ".join(describe_predicate(s) for s in spec["all"]) + ")"
    if "any" in spec:
        return "(" + " or ".join(describe_predicate(s) for s in spec["any"]) + ")"
    if "not" in spec:
        return f"not {describe_predicate(spec['not'])}"
    return f"{spec.get('signal')} {spec.get('op', '<=')} {spec.get('value')}"


# --------------------------------------------------------------------------
# temporal operators (pure functions over boolean series, so they are testable)
# --------------------------------------------------------------------------


def op_always(series: Sequence[bool]) -> Tuple[bool, Optional[int]]:
    """``always P``. Returns ``(holds, first_violation_index)``."""
    for i, value in enumerate(series):
        if not value:
            return (False, i)
    return (True, None)


def op_eventually(series: Sequence[bool]) -> Tuple[bool, Optional[int]]:
    """``eventually P``. Returns ``(holds, first_satisfying_index)``."""
    for i, value in enumerate(series):
        if value:
            return (True, i)
    return (False, None)


def op_eventually_always(series: Sequence[bool]) -> Tuple[bool, Optional[int]]:
    """``eventually always P``. Returns ``(holds, stabilisation_index)``.

    The stabilisation index is the start of the final unbroken run of trues.
    On a finite trace the operator holds exactly when the last element is true;
    the index is what makes the result useful (it is a settling time).
    """
    if not series:
        return (True, 0)
    if not series[-1]:
        return (False, None)
    index = len(series) - 1
    while index > 0 and series[index - 1]:
        index -= 1
    return (True, index)


def op_until(left: Sequence[bool], right: Sequence[bool]) -> Tuple[bool, Optional[int], Optional[int]]:
    """Strong ``A until B``.

    Returns ``(holds, release_index, violation_index)``. ``release_index`` is
    where B first held; ``violation_index`` is where A first failed before B
    held. B never holding is a failure with ``release_index`` of ``None``.
    """
    n = min(len(left), len(right))
    for i in range(n):
        if right[i]:
            return (True, i, None)
        if not left[i]:
            return (False, None, i)
    return (False, None, None)


# --------------------------------------------------------------------------
# assertion implementations
# --------------------------------------------------------------------------

ASSERTIONS: Dict[str, Callable[[AssertionSpec, EvalContext], AssertionResult]] = {}


def register_assertion(
    name: str,
) -> Callable[[Callable[[AssertionSpec, "EvalContext"], AssertionResult]], Callable[..., AssertionResult]]:
    """Register a new assertion type, usable from YAML as ``type: <name>``.

    The function receives the validated :class:`~simharness.scenario.AssertionSpec`
    and an :class:`EvalContext`, and must return an :class:`AssertionResult`.
    Anything you can compute from the trace is fair game::

        @register_assertion("max_lateral_error")
        def _lateral(spec, ctx):
            worst = max(abs(v) for v in ctx.signal("y"))
            ...
    """

    def deco(fn: Callable[..., AssertionResult]) -> Callable[..., AssertionResult]:
        if name in ASSERTIONS:
            raise ValueError(f"assertion type '{name}' is already registered")
        ASSERTIONS[name] = fn
        return fn

    return deco


def _register(name: str) -> Callable[[Callable[..., AssertionResult]], Callable[..., AssertionResult]]:
    def deco(fn: Callable[..., AssertionResult]) -> Callable[..., AssertionResult]:
        ASSERTIONS[name] = fn
        return fn

    return deco


def _param(spec: AssertionSpec, key: str, default: Any = None, *, required: bool = False) -> Any:
    if key in spec.params and spec.params[key] is not None:
        return spec.params[key]
    if required:
        raise ScenarioError(f"{spec.path or spec.type}.{key}", f"assertion '{spec.type}' requires '{key}'")
    return default


def _number(spec: AssertionSpec, key: str, default: Optional[float] = None, *, required: bool = False) -> float:
    value = _param(spec, key, default, required=required)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScenarioError(f"{spec.path or spec.type}.{key}", f"expected a number, got {type(value).__name__}")
    return float(value)


def _argmax(values: Sequence[float]) -> Tuple[int, float]:
    best_i, best = 0, -math.inf
    for i, v in enumerate(values):
        if v > best:
            best_i, best = i, v
    return best_i, best


def _argmin(values: Sequence[float]) -> Tuple[int, float]:
    best_i, best = 0, math.inf
    for i, v in enumerate(values):
        if v < best:
            best_i, best = i, v
    return best_i, best


def _empty(spec: AssertionSpec, note: str) -> AssertionResult:
    return AssertionResult(
        name=spec.label,
        type=spec.type,
        passed=False,
        message=f"trace is empty, so {note} cannot be evaluated",
    )


@_register("reached_goal")
def _reached_goal(spec: AssertionSpec, ctx: EvalContext) -> AssertionResult:
    """Robot came within ``tolerance`` of the goal, no later than ``within``.

    ``after`` ignores the first N seconds, which matters for a return-to-home
    mission whose goal is also (near) its spawn: without it the assertion is
    satisfied at t=0 before the vehicle has flown anything.
    """
    if ctx.n == 0:
        return _empty(spec, "goal arrival")
    tol = _number(spec, "tolerance", ctx.scenario.goal.tolerance)
    after = _number(spec, "after", 0.0)
    limit = _param(spec, "within", ctx.scenario.goal.within)
    deadline = float(limit) if limit is not None else ctx.times[-1]
    dist = ctx.signal("distance_to_goal")
    arrive_index: Optional[int] = None
    for i, d in enumerate(dist):
        if d <= tol and ctx.times[i] >= after:
            arrive_index = i
            break
    windowed = [i for i in range(ctx.n) if ctx.times[i] >= after] or list(range(ctx.n))
    closest_i = min(windowed, key=lambda i: dist[i])
    closest = dist[closest_i]
    if arrive_index is None:
        return AssertionResult(
            name=spec.label,
            type=spec.type,
            passed=False,
            message=(
                f"never reached the goal: closest approach {closest:.3f} m at t={ctx.time_at(closest_i):.2f}s "
                f"(tolerance {tol:.3f} m)"
            ),
            worst_value=closest,
            worst_time=ctx.time_at(closest_i),
            worst_index=closest_i,
            threshold=tol,
            units="m",
            details={"deadline_s": deadline, "closest_approach_m": closest},
        )
    arrive_t = ctx.time_at(arrive_index)
    if arrive_t > deadline + 1e-9:
        return AssertionResult(
            name=spec.label,
            type=spec.type,
            passed=False,
            message=f"reached the goal at t={arrive_t:.2f}s, after the {deadline:.2f}s deadline",
            worst_value=arrive_t,
            worst_time=arrive_t,
            worst_index=arrive_index,
            threshold=deadline,
            units="s",
            details={"tolerance_m": tol, "closest_approach_m": closest},
        )
    return AssertionResult(
        name=spec.label,
        type=spec.type,
        passed=True,
        message=f"reached the goal at t={arrive_t:.2f}s within {tol:.3f} m (deadline {deadline:.2f}s)",
        worst_value=arrive_t,
        worst_time=arrive_t,
        worst_index=arrive_index,
        threshold=deadline,
        units="s",
        details={"tolerance_m": tol, "closest_approach_m": closest},
    )


@_register("no_collision")
def _no_collision(spec: AssertionSpec, ctx: EvalContext) -> AssertionResult:
    """No obstacle contact at any point in the run."""
    if ctx.n == 0:
        return _empty(spec, "collision")
    collided = ctx.signal("collided")
    for i, value in enumerate(collided):
        if value >= 1.0:
            clearance = ctx.signal("clearance")[i]
            obstacle = _nearest_obstacle(ctx, i)
            return AssertionResult(
                name=spec.label,
                type=spec.type,
                passed=False,
                message=(
                    f"collision at t={ctx.time_at(i):.2f}s (step {i}), penetration {abs(min(clearance, 0.0)):.3f} m"
                    + (f", nearest obstacle '{obstacle}'" if obstacle else "")
                ),
                worst_value=clearance,
                worst_time=ctx.time_at(i),
                worst_index=i,
                threshold=0.0,
                units="m",
                details={"obstacle": obstacle},
            )
    min_i, min_clearance = _argmin(ctx.signal("clearance"))
    if math.isinf(min_clearance):
        return AssertionResult(
            name=spec.label,
            type=spec.type,
            passed=True,
            message="no collision; the scenario has no obstacles in the vehicle's path",
            threshold=0.0,
            units="m",
        )
    return AssertionResult(
        name=spec.label,
        type=spec.type,
        passed=True,
        message=f"no collision; closest approach was {min_clearance:.3f} m at t={ctx.time_at(min_i):.2f}s",
        worst_value=min_clearance,
        worst_time=ctx.time_at(min_i),
        worst_index=min_i,
        threshold=0.0,
        units="m",
    )


@_register("min_clearance")
def _min_clearance(spec: AssertionSpec, ctx: EvalContext) -> AssertionResult:
    """Hull-to-obstacle gap never dropped below ``threshold`` metres."""
    if ctx.n == 0:
        return _empty(spec, "clearance")
    threshold = _number(spec, "threshold", required=True)
    after = _number(spec, "after", 0.0)
    clearance = ctx.signal("clearance")
    indices = [i for i in range(ctx.n) if ctx.times[i] >= after]
    if not indices:
        return _empty(spec, "clearance")
    worst_i = min(indices, key=lambda i: clearance[i])
    worst = clearance[worst_i]
    passed = worst >= threshold
    if math.isinf(worst):
        return AssertionResult(
            name=spec.label,
            type=spec.type,
            passed=True,
            message="no obstacles in the scenario, so clearance is unbounded",
            worst_value=None,
            threshold=threshold,
            units="m",
        )
    obstacle = _nearest_obstacle(ctx, worst_i)
    verb = "held" if passed else "dropped to"
    return AssertionResult(
        name=spec.label,
        type=spec.type,
        passed=passed,
        message=(
            f"minimum clearance {verb} {worst:.3f} m at t={ctx.time_at(worst_i):.2f}s (step {worst_i}), "
            f"limit {threshold:.3f} m" + (f", obstacle '{obstacle}'" if obstacle else "")
        ),
        worst_value=worst,
        worst_time=ctx.time_at(worst_i),
        worst_index=worst_i,
        threshold=threshold,
        units="m",
        details={"obstacle": obstacle},
    )


def _nearest_obstacle(ctx: EvalContext, index: int) -> Optional[str]:
    """Which obstacle was closest at ``index`` -- used to name the culprit."""
    if not ctx.scenario.obstacles or index >= ctx.n:
        return None
    sample = ctx.trace.samples[index]
    best_id: Optional[str] = None
    best = math.inf
    for obstacle in ctx.scenario.obstacles:
        gap = obstacle.clearance_from(sample.x, sample.y, sample.t)
        if gap < best:
            best, best_id = gap, obstacle.id
    return best_id


@_register("geofence")
def _geofence(spec: AssertionSpec, ctx: EvalContext) -> AssertionResult:
    """Robot stayed inside a box or polygon fence for the whole run."""
    if ctx.n == 0:
        return _empty(spec, "geofence containment")
    polygon = _param(spec, "polygon")
    min_xy = _param(spec, "min_xy")
    max_xy = _param(spec, "max_xy")
    min_z = _number(spec, "min_z", ctx.scenario.world.min_z)
    max_z = _number(spec, "max_z", ctx.scenario.world.max_z)

    margins: List[float] = []
    if polygon is not None:
        if not isinstance(polygon, Sequence) or len(polygon) < 3:
            raise ScenarioError(f"{spec.path or spec.type}.polygon", "needs at least 3 vertices")
        for s in ctx.trace.samples:
            margins.append(min(polygon_signed_distance(s.x, s.y, polygon), s.z - min_z, max_z - s.z))
    elif min_xy is not None or max_xy is not None:
        lo = tuple(min_xy) if min_xy is not None else ctx.scenario.world.min_xy
        hi = tuple(max_xy) if max_xy is not None else ctx.scenario.world.max_xy
        for s in ctx.trace.samples:
            margins.append(min(box_signed_distance(s.x, s.y, lo, hi), s.z - min_z, max_z - s.z))
    else:
        margins = list(ctx.signal("geofence_margin"))

    worst_i, worst = _argmin(margins)
    passed = worst >= 0.0
    return AssertionResult(
        name=spec.label,
        type=spec.type,
        passed=passed,
        message=(
            f"stayed inside the fence, closest approach to the boundary {worst:.3f} m at t={ctx.time_at(worst_i):.2f}s"
            if passed
            else f"left the fence by {abs(worst):.3f} m at t={ctx.time_at(worst_i):.2f}s (step {worst_i})"
        ),
        worst_value=worst,
        worst_time=ctx.time_at(worst_i),
        worst_index=worst_i,
        threshold=0.0,
        units="m",
    )


def _limit_assertion(
    spec: AssertionSpec, ctx: EvalContext, signal_name: str, units: str, noun: str
) -> AssertionResult:
    if ctx.n == 0:
        return _empty(spec, noun)
    limit = _number(spec, "limit", required=True)
    after = _number(spec, "after", 0.0)
    values = ctx.signal(signal_name)
    indices = [i for i in range(ctx.n) if ctx.times[i] >= after]
    worst_i = max(indices, key=lambda i: values[i])
    worst = values[worst_i]
    passed = worst <= limit
    return AssertionResult(
        name=spec.label,
        type=spec.type,
        passed=passed,
        message=(
            f"peak {noun} {worst:.4f} {units} at t={ctx.time_at(worst_i):.2f}s (step {worst_i}), "
            f"limit {limit:.4f} {units}"
        ),
        worst_value=worst,
        worst_time=ctx.time_at(worst_i),
        worst_index=worst_i,
        threshold=limit,
        units=units,
    )


@_register("max_velocity")
def _max_velocity(spec: AssertionSpec, ctx: EvalContext) -> AssertionResult:
    """Ground speed never exceeded ``limit`` m/s."""
    return _limit_assertion(spec, ctx, "speed", "m/s", "speed")


@_register("max_acceleration")
def _max_acceleration(spec: AssertionSpec, ctx: EvalContext) -> AssertionResult:
    """Acceleration magnitude never exceeded ``limit`` m/s^2."""
    return _limit_assertion(spec, ctx, "accel", "m/s^2", "acceleration")


@_register("max_jerk")
def _max_jerk(spec: AssertionSpec, ctx: EvalContext) -> AssertionResult:
    """Jerk magnitude never exceeded ``limit`` m/s^3."""
    return _limit_assertion(spec, ctx, "jerk", "m/s^3", "jerk")


@_register("heading_error")
def _heading_error(spec: AssertionSpec, ctx: EvalContext) -> AssertionResult:
    """Heading error relative to the bearing-to-goal stayed under ``limit`` rad.

    ``after`` skips the initial turn-to-face, which is otherwise the only thing
    this assertion ever reports.
    """
    return _limit_assertion(spec, ctx, "heading_error", "rad", "heading error")


@_register("path_length")
def _path_length(spec: AssertionSpec, ctx: EvalContext) -> AssertionResult:
    """Travelled distance within ``max_ratio`` times the straight-line optimum."""
    if ctx.n < 2:
        return _empty(spec, "path length")
    max_ratio = _number(spec, "max_ratio", 1.25)
    optimal = _param(spec, "optimal")
    baseline = float(optimal) if optimal is not None else ctx.scenario.straight_line_distance()
    if baseline <= 0.0:
        return AssertionResult(
            name=spec.label,
            type=spec.type,
            passed=True,
            message="spawn and goal coincide, so there is no path-length baseline",
            threshold=max_ratio,
        )
    actual = ctx.trace.path_length()
    ratio = actual / baseline
    passed = ratio <= max_ratio
    return AssertionResult(
        name=spec.label,
        type=spec.type,
        passed=passed,
        message=(
            f"path length {actual:.2f} m vs optimal {baseline:.2f} m = {ratio:.3f}x "
            f"(limit {max_ratio:.3f}x)"
        ),
        worst_value=ratio,
        worst_time=ctx.times[-1],
        worst_index=ctx.n - 1,
        threshold=max_ratio,
        units="x",
        details={"path_length_m": actual, "optimal_m": baseline},
    )


@_register("settling_time")
def _settling_time(spec: AssertionSpec, ctx: EvalContext) -> AssertionResult:
    """Signal entered a band around ``target`` and stayed there, by ``limit`` seconds.

    This is ``eventually always |signal - target| <= band`` with the
    stabilisation time compared to a deadline. It fails loudly if the signal
    enters the band and then leaves it, which a plain "first time within
    tolerance" check would miss.
    """
    if ctx.n == 0:
        return _empty(spec, "settling time")
    name = str(_param(spec, "signal", "distance_to_goal"))
    target = _number(spec, "target", 0.0)
    band = _number(spec, "band", ctx.scenario.goal.tolerance)
    limit = _number(spec, "limit", required=True)
    values = ctx.signal(name, f"{spec.path or spec.type}.signal")
    inside = [abs(v - target) <= band for v in values]
    holds, index = op_eventually_always(inside)
    if not holds:
        final = abs(values[-1] - target)
        return AssertionResult(
            name=spec.label,
            type=spec.type,
            passed=False,
            message=(
                f"'{name}' never settled: it is {final:.4f} from target {target:g} at the end of the run "
                f"(band {band:g})"
            ),
            worst_value=final,
            worst_time=ctx.times[-1],
            worst_index=ctx.n - 1,
            threshold=limit,
            units="s",
        )
    settle_t = ctx.time_at(index or 0)
    passed = settle_t <= limit + 1e-9
    return AssertionResult(
        name=spec.label,
        type=spec.type,
        passed=passed,
        message=(
            f"'{name}' settled into +/-{band:g} of {target:g} at t={settle_t:.2f}s "
            f"(limit {limit:.2f}s)"
        ),
        worst_value=settle_t,
        worst_time=settle_t,
        worst_index=index,
        threshold=limit,
        units="s",
    )


@_register("overshoot")
def _overshoot(spec: AssertionSpec, ctx: EvalContext) -> AssertionResult:
    """Peak excursion past ``target``, as a percentage of the initial step size."""
    if ctx.n < 2:
        return _empty(spec, "overshoot")
    name = str(_param(spec, "signal", "distance_to_goal"))
    target = _number(spec, "target", 0.0)
    max_percent = _number(spec, "max_percent", required=True)
    values = ctx.signal(name, f"{spec.path or spec.type}.signal")
    start = values[0]
    step = start - target
    if abs(step) < 1e-9:
        return AssertionResult(
            name=spec.label,
            type=spec.type,
            passed=True,
            message=f"'{name}' starts at the target, so overshoot is undefined and treated as 0%",
            worst_value=0.0,
            threshold=max_percent,
            units="%",
        )
    excursions = [(v - target) / step for v in values]
    worst_i, worst_ratio = _argmin(excursions)
    overshoot_pct = max(0.0, -worst_ratio) * 100.0
    passed = overshoot_pct <= max_percent
    return AssertionResult(
        name=spec.label,
        type=spec.type,
        passed=passed,
        message=(
            f"'{name}' overshot the target by {overshoot_pct:.2f}% at t={ctx.time_at(worst_i):.2f}s "
            f"(limit {max_percent:.2f}%)"
        ),
        worst_value=overshoot_pct,
        worst_time=ctx.time_at(worst_i),
        worst_index=worst_i,
        threshold=max_percent,
        units="%",
    )


@_register("no_oscillation")
def _no_oscillation(spec: AssertionSpec, ctx: EvalContext) -> AssertionResult:
    """No sustained oscillation above ``max_frequency`` Hz.

    The signal is linearly detrended, then zero crossings of the residual give
    the dominant frequency (``crossings / (2 * duration)``). Oscillation whose
    peak-to-peak residual is under ``min_amplitude`` is ignored: sensor noise
    crosses zero constantly and is not what this assertion is about.
    """
    if ctx.n < 4:
        return _empty(spec, "oscillation")
    name = str(_param(spec, "signal", "yaw_rate"))
    max_frequency = _number(spec, "max_frequency", required=True)
    min_amplitude = _number(spec, "min_amplitude", 0.05)
    after = _number(spec, "after", 0.0)

    idx = [i for i in range(ctx.n) if ctx.times[i] >= after]
    if len(idx) < 4:
        return _empty(spec, "oscillation")
    times = [ctx.times[i] for i in idx]
    values = [ctx.signal(name, f"{spec.path or spec.type}.signal")[i] for i in idx]
    residual = _detrend(times, values)
    amplitude = 0.5 * (max(residual) - min(residual))
    duration = times[-1] - times[0]
    if duration <= 0.0:
        return _empty(spec, "oscillation")
    if amplitude < min_amplitude:
        return AssertionResult(
            name=spec.label,
            type=spec.type,
            passed=True,
            message=(
                f"'{name}' amplitude {amplitude:.4f} is below the {min_amplitude:g} threshold; "
                "no oscillation to speak of"
            ),
            worst_value=0.0,
            threshold=max_frequency,
            units="Hz",
            details={"amplitude": amplitude},
        )
    crossings = 0
    worst_i = idx[0]
    for a, b, i in zip(residual, residual[1:], idx[1:]):
        if (a > 0.0 and b <= 0.0) or (a < 0.0 and b >= 0.0):
            crossings += 1
            worst_i = i
    frequency = crossings / (2.0 * duration)
    passed = frequency <= max_frequency
    return AssertionResult(
        name=spec.label,
        type=spec.type,
        passed=passed,
        message=(
            f"'{name}' oscillates at ~{frequency:.3f} Hz (amplitude {amplitude:.4f}, "
            f"{crossings} zero crossings over {duration:.2f}s), limit {max_frequency:.3f} Hz"
        ),
        worst_value=frequency,
        worst_time=ctx.time_at(worst_i),
        worst_index=worst_i,
        threshold=max_frequency,
        units="Hz",
        details={"amplitude": amplitude, "crossings": crossings},
    )


def _detrend(times: Sequence[float], values: Sequence[float]) -> List[float]:
    """Subtract a least-squares straight line so a ramp is not read as offset."""
    n = len(values)
    mean_t = math.fsum(times) / n
    mean_v = math.fsum(values) / n
    sxx = math.fsum((t - mean_t) ** 2 for t in times)
    if sxx <= 0.0:
        return [v - mean_v for v in values]
    sxy = math.fsum((t - mean_t) * (v - mean_v) for t, v in zip(times, values))
    slope = sxy / sxx
    return [v - (mean_v + slope * (t - mean_t)) for t, v in zip(times, values)]


@_register("energy_budget")
def _energy_budget(spec: AssertionSpec, ctx: EvalContext) -> AssertionResult:
    """Total energy stayed within a joule or watt-hour budget."""
    if ctx.n == 0:
        return _empty(spec, "energy use")
    max_wh = _param(spec, "max_wh")
    max_j = _param(spec, "max_j")
    if max_wh is None and max_j is None:
        raise ScenarioError(f"{spec.path or spec.type}", "energy_budget needs 'max_wh' or 'max_j'")
    if max_wh is not None:
        limit, units, series = float(max_wh), "Wh", ctx.signal("energy_wh")
    else:
        limit, units, series = float(max_j), "J", ctx.signal("energy_j")
    used = series[-1]
    worst_i = ctx.n - 1
    for i, v in enumerate(series):
        if v > limit:
            worst_i = i
            break
    passed = used <= limit
    fraction = used / limit if limit > 0 else math.inf
    return AssertionResult(
        name=spec.label,
        type=spec.type,
        passed=passed,
        message=(
            f"used {used:.3f} {units} of a {limit:.3f} {units} budget ({fraction * 100:.1f}%)"
            + ("" if passed else f"; budget exceeded at t={ctx.time_at(worst_i):.2f}s")
        ),
        worst_value=used,
        worst_time=ctx.time_at(worst_i),
        worst_index=worst_i,
        threshold=limit,
        units=units,
        details={"fraction_of_budget": fraction},
    )


# -- temporal ---------------------------------------------------------------


@_register("always")
def _always(spec: AssertionSpec, ctx: EvalContext) -> AssertionResult:
    """``always P``: the predicate holds at every recorded step."""
    raw = _param(spec, "predicate", required=True)
    series = compile_predicate(raw, f"{spec.path or spec.type}.predicate")(ctx)
    holds, index = op_always(series)
    text = describe_predicate(raw)
    if ctx.n == 0:
        return AssertionResult(
            name=spec.label,
            type=spec.type,
            passed=True,
            message=f"always({text}) holds vacuously: the trace is empty",
            details={"vacuous": True, "predicate": text},
        )
    if holds:
        return AssertionResult(
            name=spec.label,
            type=spec.type,
            passed=True,
            message=f"always({text}) held for all {ctx.n} steps",
            details={"predicate": text, "steps": ctx.n},
        )
    return AssertionResult(
        name=spec.label,
        type=spec.type,
        passed=False,
        message=f"always({text}) first violated at t={ctx.time_at(index or 0):.2f}s (step {index})",
        worst_time=ctx.time_at(index or 0),
        worst_index=index,
        details={"predicate": text},
    )


@_register("never")
def _never(spec: AssertionSpec, ctx: EvalContext) -> AssertionResult:
    """``never P``, i.e. ``always not P``."""
    raw = _param(spec, "predicate", required=True)
    series = compile_predicate(raw, f"{spec.path or spec.type}.predicate")(ctx)
    holds, index = op_eventually(series)
    text = describe_predicate(raw)
    if not holds:
        return AssertionResult(
            name=spec.label,
            type=spec.type,
            passed=True,
            message=f"never({text}) held: the predicate was false at all {ctx.n} steps",
            details={"predicate": text},
        )
    return AssertionResult(
        name=spec.label,
        type=spec.type,
        passed=False,
        message=f"never({text}) violated at t={ctx.time_at(index or 0):.2f}s (step {index})",
        worst_time=ctx.time_at(index or 0),
        worst_index=index,
        details={"predicate": text},
    )


@_register("eventually")
def _eventually(spec: AssertionSpec, ctx: EvalContext) -> AssertionResult:
    """``eventually P``, optionally by a ``within`` deadline in seconds."""
    raw = _param(spec, "predicate", required=True)
    series = compile_predicate(raw, f"{spec.path or spec.type}.predicate")(ctx)
    within = _param(spec, "within")
    deadline = float(within) if within is not None else None
    if deadline is not None:
        series = [v and ctx.times[i] <= deadline + 1e-9 for i, v in enumerate(series)]
    holds, index = op_eventually(series)
    text = describe_predicate(raw)
    suffix = f" within {deadline:.2f}s" if deadline is not None else ""
    if holds:
        return AssertionResult(
            name=spec.label,
            type=spec.type,
            passed=True,
            message=f"eventually({text}) satisfied at t={ctx.time_at(index or 0):.2f}s (step {index}){suffix}",
            worst_value=ctx.time_at(index or 0),
            worst_time=ctx.time_at(index or 0),
            worst_index=index,
            threshold=deadline,
            units="s",
            details={"predicate": text},
        )
    return AssertionResult(
        name=spec.label,
        type=spec.type,
        passed=False,
        message=f"eventually({text}) never became true{suffix} over {ctx.n} steps",
        threshold=deadline,
        units="s",
        details={"predicate": text},
    )


@_register("eventually_always")
def _eventually_always(spec: AssertionSpec, ctx: EvalContext) -> AssertionResult:
    """``eventually always P``: P becomes true and never lapses again.

    ``settle_by`` bounds the stabilisation time. Without it, this only tests
    that P holds at the final step -- see the finite-trace note at the top of
    this module.
    """
    raw = _param(spec, "predicate", required=True)
    series = compile_predicate(raw, f"{spec.path or spec.type}.predicate")(ctx)
    settle_by = _param(spec, "settle_by")
    deadline = float(settle_by) if settle_by is not None else None
    holds, index = op_eventually_always(series)
    text = describe_predicate(raw)
    if not holds:
        last_false = max(i for i, v in enumerate(series) if not v)
        return AssertionResult(
            name=spec.label,
            type=spec.type,
            passed=False,
            message=(
                f"eventually_always({text}) failed: the predicate is false at the end of the run "
                f"(last false at t={ctx.time_at(last_false):.2f}s)"
            ),
            worst_time=ctx.time_at(last_false),
            worst_index=last_false,
            threshold=deadline,
            units="s",
            details={"predicate": text},
        )
    settle_t = ctx.time_at(index or 0)
    passed = deadline is None or settle_t <= deadline + 1e-9
    return AssertionResult(
        name=spec.label,
        type=spec.type,
        passed=passed,
        message=(
            f"eventually_always({text}) stabilised at t={settle_t:.2f}s (step {index})"
            + (f", deadline {deadline:.2f}s" if deadline is not None else "")
        ),
        worst_value=settle_t,
        worst_time=settle_t,
        worst_index=index,
        threshold=deadline,
        units="s",
        details={"predicate": text},
    )


@_register("until")
def _until(spec: AssertionSpec, ctx: EvalContext) -> AssertionResult:
    """Strong ``A until B``: B must happen, and A must hold up to that point."""
    raw_a = _param(spec, "condition", required=True)
    raw_b = _param(spec, "release", required=True)
    key = spec.path or spec.type
    left = compile_predicate(raw_a, f"{key}.condition")(ctx)
    right = compile_predicate(raw_b, f"{key}.release")(ctx)
    within = _param(spec, "within")
    deadline = float(within) if within is not None else None
    if deadline is not None:
        right = [v and ctx.times[i] <= deadline + 1e-9 for i, v in enumerate(right)]
    holds, release, violation = op_until(left, right)
    text_a, text_b = describe_predicate(raw_a), describe_predicate(raw_b)
    if holds:
        return AssertionResult(
            name=spec.label,
            type=spec.type,
            passed=True,
            message=(
                f"({text_a}) until ({text_b}): release condition met at t={ctx.time_at(release or 0):.2f}s "
                f"(step {release}) with the condition holding throughout"
            ),
            worst_value=ctx.time_at(release or 0),
            worst_time=ctx.time_at(release or 0),
            worst_index=release,
            threshold=deadline,
            units="s",
            details={"condition": text_a, "release": text_b},
        )
    if violation is not None:
        return AssertionResult(
            name=spec.label,
            type=spec.type,
            passed=False,
            message=(
                f"({text_a}) until ({text_b}): the condition broke at t={ctx.time_at(violation):.2f}s "
                f"(step {violation}) before the release condition was ever met"
            ),
            worst_time=ctx.time_at(violation),
            worst_index=violation,
            threshold=deadline,
            units="s",
            details={"condition": text_a, "release": text_b},
        )
    return AssertionResult(
        name=spec.label,
        type=spec.type,
        passed=False,
        message=(
            f"({text_a}) until ({text_b}): the release condition never occurred"
            + (f" within {deadline:.2f}s" if deadline is not None else "")
            + " (strong until, so this is a failure)"
        ),
        threshold=deadline,
        units="s",
        details={"condition": text_a, "release": text_b},
    )


# --------------------------------------------------------------------------
# driving the whole set
# --------------------------------------------------------------------------


def evaluate_assertion(spec: AssertionSpec, ctx: EvalContext) -> AssertionResult:
    """Evaluate one assertion. Unknown types raise :class:`ScenarioError`."""
    fn = ASSERTIONS.get(spec.type)
    if fn is None:
        raise ScenarioError(
            f"{spec.path or 'assertions'}.type",
            f"unknown assertion type '{spec.type}'; known types: {', '.join(sorted(ASSERTIONS))}",
        )
    return fn(spec, ctx)


def evaluate_all(trace: Trace, scenario: Scenario) -> List[AssertionResult]:
    """Evaluate every assertion in the scenario against the trace."""
    ctx = build_signals(trace, scenario)
    return [evaluate_assertion(spec, ctx) for spec in scenario.assertions]


def summarise(results: Sequence[AssertionResult]) -> Dict[str, Any]:
    """Counts plus the failures, for report headers."""
    failures = [r for r in results if not r.passed]
    return {
        "total": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "all_passed": not failures,
        "failures": [r.name for r in failures],
    }
