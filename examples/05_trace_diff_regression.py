#!/usr/bin/env python3
"""Localise a regression: which timestep did two versions stop agreeing?

    python3 examples/05_trace_diff_regression.py

"The obstacle test broke" is not a debugging lead. "The two runs are identical
for 3.42 s and then the yaw rate diverges at step 171, half a second before
the collision" is.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simharness import load_scenario
from simharness.runner import run_scenario
from simharness.trace import diff_traces

SCENARIO = Path(__file__).resolve().parents[1] / "scenarios" / "obstacle_slalom.yaml"


def main() -> int:
    scenario = load_scenario(SCENARIO)

    baseline = run_scenario(scenario)
    # Stand-in for "someone changed the controller": nudge one gain by 2%.
    candidate = run_scenario(scenario.with_overrides({"controller.kp_angular": scenario.controller.kp_angular * 1.02}))

    print(f"baseline : {baseline.status}, {baseline.steps} steps, path {baseline.metrics['path_length_m']:.3f} m")
    print(f"candidate: {candidate.status}, {candidate.steps} steps, path {candidate.metrics['path_length_m']:.3f} m")
    print()

    exact = diff_traces(baseline.trace, candidate.trace)
    print("exact comparison:")
    print(f"  {exact.summary()}")
    print(f"  largest deviation over the run: '{exact.worst_field}' "
          f"up to {exact.max_abs_diff[exact.worst_field]:.6g}")
    print()

    # A tolerance answers a different question: did the robot actually go
    # somewhere else, or did a float wobble in the last few bits?
    loose = diff_traces(baseline.trace, candidate.trace, tolerance=0.05, fields=["x", "y", "yaw"])
    print("comparison at 0.05 tolerance on position and heading:")
    print(f"  {loose.summary()}")
    print()

    # Re-running the same scenario must produce a byte-identical trace.
    repeat = run_scenario(scenario)
    identical = diff_traces(baseline.trace, repeat.trace)
    print(f"determinism check (same seed, same scenario): {identical.summary()}")
    assert identical.identical, "a repeat run must be byte-identical"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
