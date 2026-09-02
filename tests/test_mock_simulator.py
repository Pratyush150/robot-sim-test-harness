"""The built-in simulator: kinematics, collisions, wind, sensors, lifecycle."""

from __future__ import annotations

import math

import pytest

from conftest import minimal_scenario_dict
from simharness.scenario import Pose, Scenario
from simharness.simulators import (
    MockSimulator,
    SimulatorError,
    availability_report,
    create_simulator,
    get_simulator_class,
    select_simulator,
)
from simharness.simulators.base import Command


def _scenario(**overrides) -> Scenario:
    return Scenario.from_dict(minimal_scenario_dict(**overrides))


def _drive(scenario: Scenario, command: Command, steps: int) -> MockSimulator:
    sim = MockSimulator()
    sim.reset(scenario)
    sim.send_command(command)
    for _ in range(steps):
        sim.step(scenario.sim.dt)
    return sim


# -- differential drive ----------------------------------------------------


def test_diff_drive_travels_forward_at_the_commanded_speed():
    scenario = _scenario()
    sim = _drive(scenario, Command(linear=(1.0, 0.0, 0.0)), 10)
    state = sim.get_state()
    # Near-zero actuator lag plus a huge accel limit means v == v_cmd after one step.
    assert state.position[0] == pytest.approx(1.0, abs=1e-9)
    assert state.position[1] == pytest.approx(0.0, abs=1e-12)
    assert state.velocity[0] == pytest.approx(1.0)


def test_diff_drive_is_nonholonomic_and_ignores_lateral_commands():
    scenario = _scenario()
    sim = _drive(scenario, Command(linear=(0.0, 1.0, 0.0)), 10)
    assert sim.get_state().position == pytest.approx((0.0, 0.0, 0.0))


def test_diff_drive_yaw_integrates_the_yaw_rate():
    scenario = _scenario()
    sim = _drive(scenario, Command(linear=(0.0, 0.0, 0.0), yaw_rate=1.0), 10)
    assert sim.get_state().yaw == pytest.approx(1.0, abs=0.02)


def test_actuator_lag_prevents_an_instant_step_in_speed():
    data = minimal_scenario_dict()
    data["robot"]["actuator_tau"] = 0.5
    data["robot"]["max_accel"] = 10.0
    scenario = Scenario.from_dict(data)
    sim = _drive(scenario, Command(linear=(1.0, 0.0, 0.0)), 1)
    # Exact first-order lag: dv = (v_cmd - v) * (1 - exp(-dt / tau))
    expected = 1.0 - math.exp(-0.1 / 0.5)
    assert sim.get_state().velocity[0] == pytest.approx(expected, abs=1e-12)
    assert expected < 0.2, "the exact solution must undershoot the Euler estimate"


def test_acceleration_limit_clamps_the_lag_response():
    data = minimal_scenario_dict()
    data["robot"]["actuator_tau"] = 0.01
    data["robot"]["max_accel"] = 1.0
    scenario = Scenario.from_dict(data)
    sim = _drive(scenario, Command(linear=(1.0, 0.0, 0.0)), 1)
    assert sim.get_state().velocity[0] == pytest.approx(0.1, abs=1e-9)  # a_max * dt


def test_speed_command_is_saturated_at_the_vehicle_limit():
    scenario = _scenario()
    sim = _drive(scenario, Command(linear=(99.0, 0.0, 0.0)), 5)
    assert sim.get_state().velocity[0] == pytest.approx(scenario.robot.max_speed)


def test_wind_slips_a_ground_robot_sideways():
    data = minimal_scenario_dict()
    data["robot"]["drag"] = 0.5
    data["disturbance"] = {"wind": [0.0, 2.0, 0.0]}
    scenario = Scenario.from_dict(data)
    sim = _drive(scenario, Command(linear=(0.0, 0.0, 0.0)), 10)
    # slip = drag * wind = 1.0 m/s for 1.0 s
    assert sim.get_state().position[1] == pytest.approx(1.0, abs=1e-9)


# -- quadrotor -------------------------------------------------------------


def _air_scenario(**robot_overrides) -> Scenario:
    data = minimal_scenario_dict()
    data["robot"].update({"type": "quadrotor", "max_speed": 5.0, "max_accel": 20.0, "drag": 0.0})
    data["robot"].update(robot_overrides)
    data["robot"]["spawn"] = {"x": 0.0, "y": 0.0, "z": 2.0}
    data["world"]["max_z"] = 20.0
    data["goal"] = {"x": 10.0, "y": 0.0, "z": 2.0, "tolerance": 0.25}
    return Scenario.from_dict(data)


def test_quadrotor_tracks_a_world_frame_velocity_setpoint():
    scenario = _air_scenario(actuator_tau=0.2)
    sim = _drive(scenario, Command(linear=(1.0, 2.0, 0.0)), 40)
    state = sim.get_state()
    assert state.velocity[0] == pytest.approx(1.0, abs=0.02)
    assert state.velocity[1] == pytest.approx(2.0, abs=0.02)
    assert state.position[2] == pytest.approx(2.0, abs=1e-6)


def test_quadrotor_climbs_on_a_positive_z_command():
    scenario = _air_scenario(actuator_tau=0.2)
    sim = _drive(scenario, Command(linear=(0.0, 0.0, 1.0)), 20)
    assert sim.get_state().position[2] > 2.5


def test_steady_wind_produces_a_steady_state_drift_on_a_quadrotor():
    data = minimal_scenario_dict()
    data["robot"].update(
        {"type": "quadrotor", "max_speed": 5.0, "max_accel": 20.0, "drag": 1.0, "actuator_tau": 0.05}
    )
    data["robot"]["spawn"] = {"x": 0.0, "y": 0.0, "z": 2.0}
    data["world"]["max_z"] = 20.0
    data["disturbance"] = {"wind": [0.0, 3.0, 0.0]}
    scenario = Scenario.from_dict(data)
    sim = _drive(scenario, Command(linear=(0.0, 0.0, 0.0)), 100)
    # Commanding zero velocity against a 3 m/s wind leaves a residual: a
    # proportional velocity loop cannot null a constant disturbance.
    assert sim.get_state().velocity[1] > 0.05
    assert sim.get_state().position[1] > 0.1


# -- collisions ------------------------------------------------------------


def test_collision_flags_at_the_first_penetrating_step():
    """The robot advances exactly 0.1 m per step toward a pillar.

    Contact geometry: obstacle centre x=2.0, obstacle radius 0.5, robot radius
    0.25, so the hulls touch at x = 1.25. The robot is at x = 0.1 * k after
    step k, so the first penetrating step is k = 13 (x = 1.3) and step 12
    (x = 1.2) must still be clear.
    """
    scenario = _scenario(obstacles=[{"id": "pillar", "x": 2.0, "y": 0.0, "radius": 0.5}])
    sim = MockSimulator()
    sim.reset(scenario)
    sim.send_command(Command(linear=(1.0, 0.0, 0.0)))
    states = [sim.step(0.1) for _ in range(20)]

    first_hit = next(i for i, s in enumerate(states) if s.collided)
    assert first_hit == 12  # states[12] is step 13 of the run (states are 1-indexed steps)
    assert states[11].collided is False
    assert states[11].clearance > 0.0
    assert states[12].clearance < 0.0
    assert states[11].position[0] == pytest.approx(1.2, abs=1e-9)
    assert states[12].position[0] == pytest.approx(1.3, abs=1e-9)


def test_collision_latches_once_set():
    scenario = _scenario(obstacles=[{"id": "pillar", "x": 2.0, "y": 0.0, "radius": 0.5}])
    sim = _drive(scenario, Command(linear=(1.0, 0.0, 0.0)), 40)
    assert sim.get_state().collided is True
    assert len([e for e in sim.drain_events() if e[1] == "collision"]) == 1


def test_clearance_is_measured_to_the_hull_not_the_centre():
    scenario = _scenario(obstacles=[{"id": "pillar", "x": 2.0, "y": 0.0, "radius": 0.5}])
    sim = MockSimulator()
    state = sim.reset(scenario)
    # centre distance 2.0, minus obstacle 0.5, minus robot 0.25
    assert state.clearance == pytest.approx(1.25)


def test_flying_over_an_obstacle_reports_no_clearance_constraint():
    data = minimal_scenario_dict(obstacles=[{"id": "low_wall", "x": 2.0, "y": 0.0, "radius": 0.5, "height": 1.0}])
    data["robot"].update({"type": "quadrotor", "max_speed": 5.0})
    data["robot"]["spawn"] = {"x": 0.0, "y": 0.0, "z": 3.0}
    data["world"]["max_z"] = 20.0
    data["goal"] = {"x": 10.0, "y": 0.0, "z": 3.0, "tolerance": 0.25}
    sim = MockSimulator()
    state = sim.reset(Scenario.from_dict(data))
    assert math.isinf(state.clearance)


def test_moving_obstacle_clearance_changes_with_time():
    scenario = _scenario(
        obstacles=[{"id": "crosser", "x": 5.0, "y": -3.0, "radius": 0.5, "motion": {"type": "linear", "vy": 1.0}}]
    )
    sim = MockSimulator()
    start = sim.reset(scenario).clearance
    sim.send_command(Command())
    for _ in range(20):
        sim.step(0.1)
    assert sim.get_state().clearance < start


# -- sensors ---------------------------------------------------------------


def test_noise_free_profile_gives_a_perfect_estimate():
    scenario = _scenario()
    sim = _drive(scenario, Command(linear=(1.0, 0.0, 0.0)), 5)
    state = sim.get_state()
    assert state.est_position == pytest.approx(state.position)
    assert state.sensor_valid is True


def test_position_noise_perturbs_the_estimate_but_not_the_truth():
    data = minimal_scenario_dict()
    data["sensors"] = {"position_noise_std": 0.2}
    scenario = Scenario.from_dict(data)
    sim = _drive(scenario, Command(linear=(1.0, 0.0, 0.0)), 20)
    state = sim.get_state()
    assert state.position[0] == pytest.approx(2.0, abs=1e-9)
    assert state.est_position[0] != state.position[0]
    assert abs(state.est_position[0] - state.position[0]) < 2.0


def test_dropout_holds_the_last_fix_and_flags_invalid():
    data = minimal_scenario_dict()
    data["sensors"] = {"dropout_probability": 1.0, "dropout_duration": 0.5}
    scenario = Scenario.from_dict(data)
    sim = MockSimulator()
    sim.reset(scenario)
    sim.send_command(Command(linear=(1.0, 0.0, 0.0)))
    states = [sim.step(0.1) for _ in range(4)]
    assert all(not s.sensor_valid for s in states)
    held = {round(s.est_position[0], 9) for s in states}
    assert len(held) == 1, "a held fix must not move while the sensor is out"
    assert states[-1].position[0] > states[-1].est_position[0]


def test_position_bias_offsets_the_estimate_consistently():
    data = minimal_scenario_dict()
    data["sensors"] = {"position_bias": [0.5, -0.25, 0.0]}
    scenario = Scenario.from_dict(data)
    sim = _drive(scenario, Command(linear=(1.0, 0.0, 0.0)), 5)
    state = sim.get_state()
    assert state.est_position[0] - state.position[0] == pytest.approx(0.5)
    assert state.est_position[1] - state.position[1] == pytest.approx(-0.25)


# -- energy ----------------------------------------------------------------


def test_energy_accumulates_and_never_decreases():
    scenario = _scenario()
    sim = MockSimulator()
    sim.reset(scenario)
    sim.send_command(Command(linear=(1.0, 0.0, 0.0)))
    energies = [sim.step(0.1).energy_j for _ in range(10)]
    assert energies == sorted(energies)
    assert energies[-1] > 0.0


# -- lifecycle and registry ------------------------------------------------


def test_step_before_reset_raises():
    with pytest.raises(SimulatorError):
        MockSimulator().step(0.1)


def test_zero_dt_is_rejected():
    sim = MockSimulator()
    sim.reset(_scenario())
    with pytest.raises(SimulatorError):
        sim.step(0.0)


def test_close_is_idempotent_and_blocks_reset():
    sim = MockSimulator()
    sim.close()
    sim.close()
    with pytest.raises(SimulatorError):
        sim.reset(_scenario())


def test_context_manager_closes():
    with MockSimulator() as sim:
        sim.reset(_scenario())
    with pytest.raises(SimulatorError):
        sim.reset(_scenario())


def test_spawn_adds_an_obstacle_to_the_running_world():
    scenario = _scenario()
    sim = MockSimulator()
    sim.reset(scenario)
    assert sim.get_state().clearance == float("inf")
    sim.spawn("late_arrival", Pose(x=1.0, y=0.0), radius=0.4)
    sim.send_command(Command())
    state = sim.step(0.1)
    assert state.clearance == pytest.approx(1.0 - 0.4 - 0.25)
    assert [o.id for o in sim.obstacles] == ["late_arrival"]


def test_spawn_rejects_duplicate_ids_and_unknown_kwargs():
    sim = MockSimulator()
    sim.reset(_scenario())
    sim.spawn("thing", Pose(x=3.0, y=3.0))
    with pytest.raises(SimulatorError):
        sim.spawn("thing", Pose(x=4.0, y=4.0))
    with pytest.raises(SimulatorError):
        sim.spawn("other", Pose(x=4.0, y=4.0), colour="red")


def test_registry_reports_every_backend_with_a_reason():
    report = dict((name, (ok, why)) for name, ok, why in availability_report())
    assert set(report) >= {"mock", "gazebo", "airsim", "isaac"}
    assert report["mock"][0] is True
    for name, (ok, why) in report.items():
        assert why, f"backend '{name}' gave no availability reason"


def test_registry_defaults_to_mock_and_auto_falls_back_to_it():
    assert select_simulator()[1] == "mock"
    assert select_simulator("auto")[1] == "mock"
    assert isinstance(create_simulator(), MockSimulator)


def test_registry_rejects_an_unknown_backend():
    with pytest.raises(SimulatorError):
        get_simulator_class("unreal_engine_9")


def test_guarded_adapters_import_without_their_backends():
    from simharness.simulators import airsim, gazebo, isaac

    for module, cls_name in ((gazebo, "GazeboSimulator"), (airsim, "AirSimSimulator"), (isaac, "IsaacSimulator")):
        cls = getattr(module, cls_name)
        available, reason = cls.availability()
        assert available is False
        assert len(reason) > 20, "an unavailable backend must say why in a usable sentence"
        assert cls.connection, "every adapter must document its connection mechanism"


def test_actuator_lag_stays_stable_when_dt_far_exceeds_tau():
    """Explicit Euler would ring or diverge here. The exact form must not."""
    data = minimal_scenario_dict()
    data["robot"]["actuator_tau"] = 0.001
    data["robot"]["max_accel"] = 1000.0
    scenario = Scenario.from_dict(data)
    sim = MockSimulator()
    sim.reset(scenario)
    sim.send_command(Command(linear=(1.0, 0.0, 0.0)))
    speeds = [sim.step(0.1).velocity[0] for _ in range(10)]
    assert all(0.0 <= v <= 1.0 + 1e-12 for v in speeds)
    assert speeds[-1] == pytest.approx(1.0, abs=1e-12)
