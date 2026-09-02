"""The controllers under test.

The harness has to drive *something*, and that something has to be realistic
enough for the assertions to mean anything. These are deliberately simple,
deliberately imperfect controllers: a proportional go-to-goal law with a
potential-field avoidance term, and a waypoint sequencer built on top of it.

Two design choices worth stating outright:

* **The controller only ever sees the estimate**, never ground truth. During a
  sensor dropout it dead-reckons from the last valid fix using its own
  commanded velocity, and drifts. That is what makes the dropout scenario a
  real test rather than a formality.
* **The controller has a known obstacle map.** This harness tests *trajectory
  behaviour*, not SLAM. If you want to test perception, swap in your own
  controller; the interface is three methods.

Gains come from the scenario's ``controller:`` block, which is what lets
:mod:`simharness.sweep` sweep them and find the gain at which the vehicle
starts clipping obstacles.
"""

from __future__ import annotations

import abc
import math
from typing import Any, Dict, List, Tuple

from .geometry import clamp, wrap_angle
from .scenario import Scenario
from .simulators.base import Command, SimState

__all__ = ["Controller", "GotoGoalController", "WaypointMissionController", "make_controller"]

#: How hard the avoidance term steers *around* an obstacle relative to *away*
#: from it. Above ~2 the vehicle orbits; below ~1 it stalls on head-on
#: approaches. 1.6 is what the slalom and corridor scenarios are tuned against.
TANGENTIAL_WEIGHT = 1.6

Vec3 = Tuple[float, float, float]


class Controller(abc.ABC):
    """Something that turns a state estimate into a setpoint."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.spec = scenario.controller
        self.robot = scenario.robot
        self._estimate: Vec3 = scenario.spawn_xyz()
        self._est_yaw: float = scenario.spawn.yaw
        self._last_command = Command()
        self._dead_reckoning_steps = 0

    def reset(self) -> None:
        """Return the controller to its initial state."""
        self._estimate = self.scenario.spawn_xyz()
        self._est_yaw = self.scenario.spawn.yaw
        self._last_command = Command()
        self._dead_reckoning_steps = 0

    @abc.abstractmethod
    def target(self, state: SimState) -> Vec3:
        """The point the controller is currently steering at."""

    def compute(self, state: SimState, dt: float) -> Command:
        """Produce the next setpoint from ``state``."""
        self._update_estimate(state, dt)
        target = self.target(state)
        if self.robot.type == "diff_drive":
            command = self._diff_drive_command(target)
        else:
            command = self._quadrotor_command(target)
        self._last_command = command
        return command

    def status(self) -> Dict[str, Any]:
        """Controller-internal state worth putting in the trace metadata."""
        return {"dead_reckoning_steps": self._dead_reckoning_steps}

    # -- estimation --------------------------------------------------------

    def _update_estimate(self, state: SimState, dt: float) -> None:
        """Track the position estimate, dead-reckoning through dropouts."""
        if state.sensor_valid:
            self._estimate = state.est_position
            self._est_yaw = state.est_yaw
            return
        self._dead_reckoning_steps += 1
        if self.robot.type == "diff_drive":
            speed = self._last_command.forward
            self._est_yaw = wrap_angle(self._est_yaw + self._last_command.yaw_rate * dt)
            self._estimate = (
                self._estimate[0] + speed * math.cos(self._est_yaw) * dt,
                self._estimate[1] + speed * math.sin(self._est_yaw) * dt,
                self._estimate[2],
            )
        else:
            lin = self._last_command.linear
            self._estimate = (
                self._estimate[0] + lin[0] * dt,
                self._estimate[1] + lin[1] * dt,
                self._estimate[2] + lin[2] * dt,
            )
            self._est_yaw = wrap_angle(self._est_yaw + self._last_command.yaw_rate * dt)

    # -- steering ----------------------------------------------------------

    def _repulsion(self, position: Vec3, t: float, target: Vec3) -> Tuple[float, float]:
        """Potential-field push away from any obstacle inside ``avoid_range``.

        A purely radial push is not enough. Approach a round pillar head-on and
        the radial term points straight back down your own path, so the vehicle
        decelerates into the obstacle instead of going around it. The fix is a
        tangential term: pick the side of the obstacle the goal is already on
        and push along it. That is the difference between an avoidance law that
        works in a slalom and one that only works when the obstacle is
        conveniently off to one side.
        """
        spec = self.spec
        if spec.avoid_gain <= 0.0:
            return (0.0, 0.0)
        px, py = position[0], position[1]
        gx, gy = target[0] - px, target[1] - py
        gnorm = math.hypot(gx, gy)
        if gnorm > 1e-9:
            gx, gy = gx / gnorm, gy / gnorm
        else:
            gx, gy = math.cos(self._est_yaw), math.sin(self._est_yaw)

        rx = ry = 0.0
        for obstacle in self.scenario.obstacles:
            if position[2] > obstacle.z + obstacle.height:
                continue
            gap = obstacle.clearance_from(px, py, t) - self.robot.radius
            if gap >= spec.avoid_range:
                continue
            cx, cy, _ = obstacle.position_at(t)
            ox, oy = cx - px, cy - py
            dist = math.hypot(ox, oy)
            if dist < 1e-9:
                ox, oy, dist = -gx, -gy, 1.0
            ox, oy = ox / dist, oy / dist

            strength = spec.avoid_gain * (spec.avoid_range - max(gap, 0.0)) / spec.avoid_range
            if gap <= 0.0:
                strength *= 4.0
            # Radial: straight away from the obstacle centre.
            rx -= strength * ox
            ry -= strength * oy
            # Tangential: around the side the goal is already on.
            cross = gx * oy - gy * ox
            if cross > 0.0:          # obstacle sits to the left of the goal bearing
                tx, ty = gy, -gx     # so steer right
            else:
                tx, ty = -gy, gx
            rx += TANGENTIAL_WEIGHT * strength * tx
            ry += TANGENTIAL_WEIGHT * strength * ty
        return (rx, ry)

    def _diff_drive_command(self, target: Vec3) -> Command:
        spec = self.spec
        px, py = self._estimate[0], self._estimate[1]
        ex, ey = target[0] - px, target[1] - py
        distance = math.hypot(ex, ey)
        if distance <= self.scenario.goal.tolerance * 0.5 and self._is_final_target(target):
            return Command(linear=(0.0, 0.0, 0.0), yaw_rate=0.0)

        rx, ry = self._repulsion(self._estimate, self._last_t, target)
        weight = distance_scale(distance)
        dx, dy = ex + rx * weight, ey + ry * weight
        desired_yaw = math.atan2(dy, dx)
        heading_error = wrap_angle(desired_yaw - self._est_yaw)

        yaw_rate = clamp(spec.kp_angular * heading_error, -self.robot.max_yaw_rate, self.robot.max_yaw_rate)
        gate = max(0.0, math.cos(heading_error))
        speed = spec.kp_linear * distance * gate
        speed -= spec.kd_linear * self._last_command.forward
        speed = clamp(speed, 0.0, self.robot.max_speed)
        return Command(linear=(speed, 0.0, 0.0), yaw_rate=yaw_rate)

    def _quadrotor_command(self, target: Vec3) -> Command:
        spec = self.spec
        ex = target[0] - self._estimate[0]
        ey = target[1] - self._estimate[1]
        ez = target[2] - self._estimate[2]
        distance = math.sqrt(ex * ex + ey * ey + ez * ez)
        if distance <= self.scenario.goal.tolerance * 0.5 and self._is_final_target(target):
            return Command(linear=(0.0, 0.0, 0.0), yaw_rate=0.0)

        rx, ry = self._repulsion(self._estimate, self._last_t, target)
        vx = spec.kp_linear * ex + rx
        vy = spec.kp_linear * ey + ry
        vz = spec.kp_linear * ez
        vx -= spec.kd_linear * self._last_command.linear[0]
        vy -= spec.kd_linear * self._last_command.linear[1]
        vz -= spec.kd_linear * self._last_command.linear[2]
        mag = math.sqrt(vx * vx + vy * vy + vz * vz)
        if mag > self.robot.max_speed and mag > 0.0:
            scale = self.robot.max_speed / mag
            vx, vy, vz = vx * scale, vy * scale, vz * scale

        yaw_rate = 0.0
        if math.hypot(vx, vy) > 0.15:
            heading_error = wrap_angle(math.atan2(vy, vx) - self._est_yaw)
            yaw_rate = clamp(spec.kp_angular * heading_error, -self.robot.max_yaw_rate, self.robot.max_yaw_rate)
        return Command(linear=(vx, vy, vz), yaw_rate=yaw_rate)

    def _is_final_target(self, target: Vec3) -> bool:
        goal = self.scenario.goal_xyz()
        return math.dist(target, goal) < 1e-9

    # ``_last_t`` is set by :meth:`compute` via the runner so the repulsion term
    # sees moving obstacles where they actually are.
    _last_t: float = 0.0


def distance_scale(distance: float) -> float:
    """Weight the repulsion term relative to the attraction term.

    Attraction grows with distance to target; repulsion does not. Without this
    the avoidance term becomes irrelevant far from the goal and dominant near
    it, which is exactly backwards.
    """
    return clamp(distance, 0.5, 4.0)


class GotoGoalController(Controller):
    """Drive straight at the goal, pushing off obstacles on the way."""

    def target(self, state: SimState) -> Vec3:
        self._last_t = state.t
        return self.scenario.goal_xyz()


class WaypointMissionController(Controller):
    """Visit each waypoint in order, then the goal.

    A waypoint is retired when the *estimate* is within
    ``controller.waypoint_tolerance``. Retiring on the estimate rather than on
    truth is deliberate: a mission that skips a waypoint because of estimator
    drift is a real failure mode, and this reproduces it.
    """

    def __init__(self, scenario: Scenario) -> None:
        super().__init__(scenario)
        self._waypoints: List[Vec3] = [tuple(w) for w in scenario.controller.waypoints]  # type: ignore[misc]
        self._waypoints.append(scenario.goal_xyz())
        self._index = 0

    def reset(self) -> None:
        super().reset()
        self._index = 0

    def target(self, state: SimState) -> Vec3:
        self._last_t = state.t
        while self._index < len(self._waypoints) - 1:
            wp = self._waypoints[self._index]
            if math.dist(self._estimate, wp) <= self.spec.waypoint_tolerance:
                self._index += 1
            else:
                break
        return self._waypoints[self._index]

    def status(self) -> Dict[str, Any]:
        out = super().status()
        out.update({"waypoint_index": self._index, "waypoint_count": len(self._waypoints)})
        return out


def make_controller(scenario: Scenario) -> Controller:
    """Build the controller named by ``scenario.controller.type``."""
    if scenario.controller.type == "waypoint_mission":
        return WaypointMissionController(scenario)
    return GotoGoalController(scenario)
