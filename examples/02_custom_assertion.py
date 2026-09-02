#!/usr/bin/env python3
"""Add your own assertion type and use it from a scenario.

    python3 examples/02_custom_assertion.py

The assertion registry is open. A new ``type:`` in YAML is one decorated
function away, and it gets the same rich result shape as the built-ins:
pass/fail, worst value, timestamp and a human-readable message.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simharness.assertions import AssertionResult, EvalContext, register_assertion
from simharness.runner import run_scenario
from simharness.scenario import AssertionSpec, Scenario


@register_assertion("max_cross_track_error")
def max_cross_track_error(spec: AssertionSpec, ctx: EvalContext) -> AssertionResult:
    """Perpendicular distance from the straight spawn-to-goal line stays under ``limit``.

    Nothing special about this one: it reads a signal off the context, finds
    the worst sample, and reports where it happened.
    """
    limit = float(spec.params["limit"])
    sx, sy, _ = ctx.scenario.spawn_xyz()
    gx, gy, _ = ctx.scenario.goal_xyz()
    dx, dy = gx - sx, gy - sy
    length = (dx * dx + dy * dy) ** 0.5 or 1.0

    worst_index, worst = 0, 0.0
    for i, (x, y) in enumerate(zip(ctx.signal("x"), ctx.signal("y"))):
        error = abs((x - sx) * dy - (y - sy) * dx) / length
        if error > worst:
            worst_index, worst = i, error

    return AssertionResult(
        name=spec.label,
        type=spec.type,
        passed=worst <= limit,
        message=(
            f"peak cross-track error {worst:.3f} m at t={ctx.time_at(worst_index):.2f}s "
            f"(step {worst_index}), limit {limit:.3f} m"
        ),
        worst_value=worst,
        worst_time=ctx.time_at(worst_index),
        worst_index=worst_index,
        threshold=limit,
        units="m",
    )


SCENARIO = {
    "name": "cross_track_demo",
    "world": {"min_xy": [-2.0, -6.0], "max_xy": [14.0, 6.0]},
    "robot": {"type": "diff_drive", "max_speed": 1.2, "spawn": {"x": 0.0, "y": 0.0}},
    "goal": {"x": 12.0, "y": 0.0, "tolerance": 0.3, "within": 30.0},
    "obstacles": [{"id": "pillar", "shape": "circle", "x": 6.0, "y": 0.3, "radius": 0.6}],
    "controller": {"type": "goto_goal", "kp_linear": 0.9, "avoid_gain": 1.2, "avoid_range": 1.4},
    "sim": {"dt": 0.02, "time_limit": 40.0, "seed": 11},
    "assertions": [
        {"type": "reached_goal"},
        {"type": "no_collision"},
        # The custom type, used exactly like a built-in.
        {"type": "max_cross_track_error", "name": "stays_near_the_line", "limit": 2.0},
        {"type": "max_cross_track_error", "name": "hugs_the_line", "limit": 0.2},
    ],
}


def main() -> int:
    result = run_scenario(Scenario.from_dict(SCENARIO))
    print(f"status: {result.status}")
    for assertion in result.assertions:
        print(f"  {'PASS' if assertion.passed else 'FAIL'}  {assertion.name}: {assertion.message}")
    print()
    print("The second custom assertion is expected to fail: the vehicle has to")
    print("leave the straight line to get around the pillar. That is the point --")
    print("an assertion that can never fail is not testing anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
