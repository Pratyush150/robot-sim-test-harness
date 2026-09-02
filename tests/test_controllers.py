"""Controllers: steering, avoidance geometry, waypoints and dead reckoning."""

from __future__ import annotations

import math

import pytest

from conftest import minimal_scenario_dict
from simharness.controllers import (
    GotoGoalController,
    WaypointMissionController,
    TANGENTIAL_WEIGHT,
    make_controller,
)
from simharness.scenario import Scenario
from simharness.simulators.base import SimState


def _state(x=0.0, y=0.0, z=0.0, yaw=0.0, t=0.0, valid=True, vel=(0.0, 0.0, 0.0)) -> SimState:
    return SimState(
        t=t,
        position=(x, y, z),
        velocity=vel,
        yaw=yaw,
        est_position=(x, y, z),
        est_yaw=yaw,
        est_velocity=vel,
        sensor_valid=valid,
    )


def _scenario(**overrides) -> Scenario:
    return Scenario.from_dict(minimal_scenario_dict(**overrides))


def test_make_controller_selects_by_type():
    assert isinstance(make_controller(_scenario()), GotoGoalController)
    mission = _scenario(controller={"type": "waypoint_mission", "waypoints": [[1.0, 0.0]]})
    assert isinstance(make_controller(mission), WaypointMissionController)


def test_goto_goal_drives_forward_when_already_facing_the_goal():
    controller = make_controller(_scenario())
    command = controller.compute(_state(x=0.0, yaw=0.0), 0.1)
    assert command.forward > 0.0
    assert command.yaw_rate == pytest.approx(0.0, abs=1e-9)


def test_goto_goal_turns_toward_a_goal_that_is_off_to_the_side():
    scenario = _scenario(goal={"x": 0.0, "y": 4.0, "tolerance": 0.25})
    controller = make_controller(scenario)
    command = controller.compute(_state(yaw=0.0), 0.1)
    assert command.yaw_rate > 0.5, "goal is to the left, so yaw rate must be positive"


def test_forward_speed_is_gated_by_heading_error():
    scenario = _scenario()
    controller = make_controller(scenario)
    facing = controller.compute(_state(yaw=0.0), 0.1).forward
    controller.reset()
    backwards = controller.compute(_state(yaw=math.pi), 0.1).forward
    assert backwards == pytest.approx(0.0)
    assert facing > backwards


def test_controller_stops_inside_the_goal_deadband():
    controller = make_controller(_scenario())
    command = controller.compute(_state(x=10.0, y=0.0), 0.1)
    assert command.forward == pytest.approx(0.0)
    assert command.yaw_rate == pytest.approx(0.0)


def test_avoidance_pushes_around_a_head_on_obstacle():
    """The tangential term is what makes a head-on approach solvable."""
    scenario = _scenario(obstacles=[{"id": "pillar", "x": 3.0, "y": 0.0, "radius": 0.5}])
    controller = make_controller(scenario)
    command = controller.compute(_state(x=2.0, y=0.0, yaw=0.0), 0.1)
    assert abs(command.yaw_rate) > 0.1, "a purely radial push would leave yaw_rate at zero"
    assert TANGENTIAL_WEIGHT > 1.0


def test_avoidance_is_inert_when_disabled():
    scenario = _scenario(
        obstacles=[{"id": "pillar", "x": 3.0, "y": 0.0, "radius": 0.5}],
        controller={"avoid_gain": 0.0},
    )
    controller = make_controller(scenario)
    command = controller.compute(_state(x=2.0, y=0.0, yaw=0.0), 0.1)
    assert command.yaw_rate == pytest.approx(0.0, abs=1e-9)


def test_avoidance_ignores_obstacles_beyond_its_range():
    scenario = _scenario(
        obstacles=[{"id": "far", "x": 9.0, "y": 2.0, "radius": 0.3}],
        controller={"avoid_range": 1.0},
    )
    controller = make_controller(scenario)
    command = controller.compute(_state(x=0.0, y=0.0, yaw=0.0), 0.1)
    assert command.yaw_rate == pytest.approx(0.0, abs=1e-9)


def test_quadrotor_controller_produces_a_world_frame_velocity():
    data = minimal_scenario_dict()
    data["robot"].update({"type": "quadrotor", "max_speed": 3.0})
    data["robot"]["spawn"] = {"x": 0.0, "y": 0.0, "z": 2.0}
    data["world"]["max_z"] = 20.0
    data["goal"] = {"x": 3.0, "y": 4.0, "z": 4.0, "tolerance": 0.3}
    controller = make_controller(Scenario.from_dict(data))
    command = controller.compute(_state(x=0.0, y=0.0, z=2.0), 0.1)
    assert command.linear[0] > 0.0
    assert command.linear[1] > 0.0
    assert command.linear[2] > 0.0
    assert command.magnitude() <= 3.0 + 1e-9


def test_waypoints_are_retired_in_order():
    scenario = _scenario(
        controller={
            "type": "waypoint_mission",
            "waypoint_tolerance": 0.5,
            "waypoints": [[2.0, 0.0], [4.0, 0.0]],
        }
    )
    controller = make_controller(scenario)
    # target() reads the controller's own estimate, which compute() maintains.
    controller.compute(_state(x=0.0), 0.1)
    assert controller.target(_state(x=0.0)) == (2.0, 0.0, 0.0)
    controller.compute(_state(x=2.2), 0.1)
    assert controller.target(_state(x=2.2)) == (4.0, 0.0, 0.0)
    controller.compute(_state(x=4.1), 0.1)
    assert controller.target(_state(x=4.1)) == scenario.goal_xyz()
    assert controller.status()["waypoint_index"] == 2


def test_waypoint_controller_reset_returns_to_the_first_leg():
    scenario = _scenario(
        controller={"type": "waypoint_mission", "waypoint_tolerance": 0.5, "waypoints": [[2.0, 0.0]]}
    )
    controller = make_controller(scenario)
    controller.compute(_state(x=2.1), 0.1)
    assert controller.status()["waypoint_index"] == 1
    controller.reset()
    assert controller.status()["waypoint_index"] == 0


def test_dead_reckoning_advances_the_estimate_while_the_sensor_is_out():
    controller = make_controller(_scenario())
    controller.compute(_state(x=0.0, yaw=0.0, valid=True), 0.1)
    before = controller._estimate[0]
    for _ in range(5):
        controller.compute(_state(x=99.0, yaw=0.0, valid=False), 0.1)
    after = controller._estimate[0]
    assert after > before, "the estimate must keep moving on commanded velocity"
    assert after < 99.0, "and must not magically snap to ground truth"
    assert controller.status()["dead_reckoning_steps"] == 5


def test_valid_fix_resets_the_estimate_to_the_measurement():
    controller = make_controller(_scenario())
    controller.compute(_state(x=0.0, valid=False), 0.1)
    controller.compute(_state(x=7.0, valid=True), 0.1)
    assert controller._estimate[0] == pytest.approx(7.0)
