#!/usr/bin/env python3
"""Sweep the envelope and find where it breaks.

    python3 examples/03_sweep_and_boundary.py

A grid sweep tells you the pass rate. Bisection tells you the number you can
put in a spec sheet: the crosswind at which this controller stops meeting its
deadline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simharness import load_scenario
from simharness.sweep import find_scenario_boundary, grid_sweep, monte_carlo

SCENARIO = Path(__file__).resolve().parents[1] / "scenarios" / "wind_disturbance.yaml"
PARAMETER = "disturbance.wind[1]"


def main() -> int:
    scenario = load_scenario(SCENARIO)

    print("=== grid sweep over crosswind ===")
    grid = grid_sweep(scenario, {PARAMETER: [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]}, workers=4)
    print(grid.format_table(PARAMETER))
    print(f"\nran {grid.total} variants in {grid.wall_time_s:.2f}s")
    for point in grid.failures():
        print(f"  fail at {point.label()}: {', '.join(point.failed_assertions)}")

    print("\n=== Monte Carlo over crosswind and noise seed ===")
    random_sweep = monte_carlo(scenario, {PARAMETER: (0.0, 12.0)}, samples=40, seed=7, workers=4)
    print(
        f"{random_sweep.passed}/{random_sweep.total} passed "
        f"({random_sweep.pass_rate * 100:.1f}%) in {random_sweep.wall_time_s:.2f}s"
    )
    lowest_failure = random_sweep.first_failing_value(PARAMETER)
    if lowest_failure is not None:
        print(f"lowest failing sample: {PARAMETER} = {lowest_failure:.3f}")

    print("\n=== failure boundary by bisection ===")
    boundary = find_scenario_boundary(scenario, PARAMETER, low=0.0, high=14.0, tolerance=0.05)
    print(boundary.summary())
    print("probe order:")
    for value, passed in boundary.history:
        print(f"  {PARAMETER} = {value:8.4f}  ->  {'pass' if passed else 'fail'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
