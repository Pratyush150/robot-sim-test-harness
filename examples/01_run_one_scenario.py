#!/usr/bin/env python3
"""Run one scenario and read the result programmatically.

    python3 examples/01_run_one_scenario.py

Prints every assertion with its worst-case value and the timestamp it
happened at, then the metrics measured from the trace.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simharness import load_scenario
from simharness.runner import run_scenario

SCENARIO = Path(__file__).resolve().parents[1] / "scenarios" / "obstacle_slalom.yaml"


def main() -> int:
    scenario = load_scenario(SCENARIO)
    print(f"scenario : {scenario.name}")
    print(f"robot    : {scenario.robot.type}, spawn {scenario.spawn_xyz()}, goal {scenario.goal_xyz()}")
    print(f"obstacles: {', '.join(o.id for o in scenario.obstacles) or 'none'}")
    print(f"seed     : {scenario.sim.seed}, dt {scenario.sim.dt}s, limit {scenario.sim.time_limit}s")
    print()

    result = run_scenario(scenario)
    print(f"status   : {result.status}  ({result.steps} steps, {result.sim_time_s:.2f}s simulated)")
    print()
    for assertion in result.assertions:
        mark = "PASS" if assertion.passed else "FAIL"
        worst = "" if assertion.worst_value is None else f"  [worst {assertion.worst_value:.4g} {assertion.units}]"
        print(f"  {mark}  {assertion.name}")
        print(f"        {assertion.message}{worst}")
    print()
    print("metrics measured from the trace:")
    for key, value in sorted(result.metrics.items()):
        print(f"  {key:28s} {value:g}")

    # The trace is a plain object: 21 fields per timestep, plus events.
    trace = result.trace
    print()
    print(f"trace: {len(trace)} samples, {len(trace.events)} events")
    for event in trace.events:
        print(f"  t={event.t:6.2f}s  {event.kind:16s} {event.detail}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
