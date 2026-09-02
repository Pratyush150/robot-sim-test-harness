#!/usr/bin/env python3
"""Plug your own simulator into the harness.

    python3 examples/04_custom_simulator.py

Six methods is the whole contract. Implement them against your own backend --
a ROS 2 bridge, an in-house physics engine, a hardware-in-the-loop rig -- and
every scenario, assertion, sweep and report in this repo works unchanged.

The backend here is deliberately trivial (a perfect integrator with no
dynamics) so you can see the shape of the interface without physics in the
way. It also demonstrates why the harness is worth having: this "vehicle" has
no actuator lag at all, so it passes a tighter path-length assertion than the
real model does.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simharness.runner import run_scenario
from simharness.scenario import Pose, Scenario
from simharness.simulators import register
from simharness.simulators.base import Command, SimState, Simulator, SimulatorError


class PerfectIntegrator(Simulator):
    """A kinematic ideal: the vehicle does exactly what it is told, instantly."""

    name = "perfect"
    connection = "in-process; no backend at all, for demonstration"

    def __init__(self) -> None:
        self._scenario: Scenario | None = None
        self._t = 0.0
        self._pose = (0.0, 0.0, 0.0)
        self._yaw = 0.0
        self._velocity = (0.0, 0.0, 0.0)
        self._command = Command()
        self._closed = False

    @classmethod
    def availability(cls) -> Tuple[bool, str]:
        return (True, "pure Python demonstration backend")

    def reset(self, scenario: Scenario) -> SimState:
        self._scenario = scenario
        self._t = 0.0
        self._pose = scenario.spawn_xyz()
        self._yaw = scenario.spawn.yaw
        self._velocity = (0.0, 0.0, 0.0)
        return self.get_state()

    def send_command(self, command: Command) -> None:
        if self._scenario is None:
            raise SimulatorError("send_command() before reset()")
        self._command = command.clipped(self._scenario.robot.max_speed, self._scenario.robot.max_yaw_rate)

    def step(self, dt: float) -> SimState:
        if self._scenario is None:
            raise SimulatorError("step() before reset()")
        self._yaw += self._command.yaw_rate * dt
        if self._scenario.robot.type == "diff_drive":
            speed = self._command.forward
            self._velocity = (speed * math.cos(self._yaw), speed * math.sin(self._yaw), 0.0)
        else:
            self._velocity = self._command.linear
        self._pose = tuple(p + v * dt for p, v in zip(self._pose, self._velocity))  # type: ignore[assignment]
        self._t += dt
        return self.get_state()

    def get_state(self) -> SimState:
        clearance = float("inf")
        if self._scenario is not None:
            for obstacle in self._scenario.obstacles:
                gap = obstacle.clearance_from(self._pose[0], self._pose[1], self._t) - self._scenario.robot.radius
                clearance = min(clearance, gap)
        return SimState(
            t=self._t,
            position=self._pose,
            velocity=self._velocity,
            yaw=self._yaw,
            est_position=self._pose,
            est_yaw=self._yaw,
            est_velocity=self._velocity,
            clearance=clearance,
            collided=clearance < 0.0,
        )

    def spawn(self, model_id: str, pose: Pose, **kwargs: Any) -> None:
        raise SimulatorError("this demonstration backend does not support spawning")

    def close(self) -> None:
        self._closed = True


register("perfect", lambda: PerfectIntegrator)

SCENARIO = {
    "name": "backend_comparison",
    "world": {"min_xy": [-2.0, -6.0], "max_xy": [14.0, 6.0]},
    "robot": {"type": "diff_drive", "max_speed": 1.2, "max_accel": 1.5, "actuator_tau": 0.2,
              "spawn": {"x": 0.0, "y": 0.0}},
    "goal": {"x": 12.0, "y": 0.0, "tolerance": 0.3, "within": 30.0},
    "controller": {"type": "goto_goal", "kp_linear": 0.9},
    "sim": {"dt": 0.02, "time_limit": 40.0, "seed": 3},
    "assertions": [
        {"type": "reached_goal"},
        {"type": "no_collision"},
        {"type": "max_acceleration", "name": "accel_limit", "limit": 2.0},
    ],
}


def main() -> int:
    scenario = Scenario.from_dict(SCENARIO)
    for backend in ("mock", "perfect"):
        result = run_scenario(scenario, simulator=backend)
        arrival = result.metrics.get("time_to_goal_s")
        print(f"{backend:8s} {result.status:6s} "
              f"time_to_goal={arrival if arrival is None else f'{arrival:.2f}s':>7} "
              f"path={result.metrics['path_length_m']:.3f} m")
        for assertion in result.assertions:
            if not assertion.passed:
                print(f"         FAIL {assertion.name}: {assertion.message}")
    print()
    print("Same scenario, same controller, same assertions -- two backends.")
    print("The 'perfect' backend has no actuator lag, so it accelerates")
    print("instantaneously and blows the acceleration limit the mock respects.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
