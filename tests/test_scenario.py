"""Scenario schema: validation, error keys, inheritance and overrides."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from conftest import minimal_scenario_dict
from simharness.scenario import (
    Obstacle,
    Scenario,
    ScenarioError,
    deep_merge,
    get_by_path,
    load_scenario,
    load_scenario_dir,
    set_by_path,
)


def test_minimal_scenario_parses_with_defaults(minimal_scenario):
    assert minimal_scenario.name == "unit"
    assert minimal_scenario.robot.type == "diff_drive"
    assert minimal_scenario.sim.dt == pytest.approx(0.1)
    assert minimal_scenario.goal.tolerance == pytest.approx(0.25)
    assert minimal_scenario.expect_failure is False


def test_missing_name_names_the_key():
    data = minimal_scenario_dict()
    del data["name"]
    with pytest.raises(ScenarioError) as exc:
        Scenario.from_dict(data)
    assert exc.value.key == "name"


def test_unknown_top_level_key_is_reported_with_a_suggestion():
    data = minimal_scenario_dict(obstacle=[])
    with pytest.raises(ScenarioError) as exc:
        Scenario.from_dict(data)
    assert exc.value.key == "obstacle"
    assert "obstacles" in str(exc.value)


def test_unknown_nested_key_names_the_full_path():
    data = minimal_scenario_dict()
    data["robot"]["max_sped"] = 3.0
    with pytest.raises(ScenarioError) as exc:
        Scenario.from_dict(data)
    assert exc.value.key == "robot.max_sped"
    assert "max_speed" in str(exc.value)


def test_bad_obstacle_field_names_the_index():
    data = minimal_scenario_dict(
        obstacles=[
            {"id": "a", "shape": "circle", "x": 1.0, "y": 0.0, "radius": 0.4},
            {"id": "b", "shape": "circle", "x": 4.0, "y": 0.0, "radius": "wide"},
        ]
    )
    with pytest.raises(ScenarioError) as exc:
        Scenario.from_dict(data)
    assert exc.value.key == "obstacles[1].radius"


def test_negative_radius_is_rejected():
    data = minimal_scenario_dict(obstacles=[{"id": "a", "x": 4.0, "y": 0.0, "radius": -1.0}])
    with pytest.raises(ScenarioError) as exc:
        Scenario.from_dict(data)
    assert exc.value.key == "obstacles[0].radius"
    assert "> 0" in str(exc.value)


def test_duplicate_obstacle_ids_are_rejected():
    data = minimal_scenario_dict(
        obstacles=[{"id": "same", "x": 3.0, "y": 0.0}, {"id": "same", "x": 6.0, "y": 0.0}]
    )
    with pytest.raises(ScenarioError) as exc:
        Scenario.from_dict(data)
    assert "duplicate obstacle id" in str(exc.value)


def test_unknown_robot_type_lists_the_choices():
    data = minimal_scenario_dict()
    data["robot"]["type"] = "hexapod"
    with pytest.raises(ScenarioError) as exc:
        Scenario.from_dict(data)
    assert exc.value.key == "robot.type"
    assert "diff_drive" in str(exc.value)


def test_goal_deadline_beyond_time_limit_is_rejected():
    data = minimal_scenario_dict()
    data["goal"]["within"] = 999.0
    with pytest.raises(ScenarioError) as exc:
        Scenario.from_dict(data)
    assert exc.value.key == "goal.within"


def test_spawn_outside_world_bounds_is_rejected():
    data = minimal_scenario_dict()
    data["robot"]["spawn"] = {"x": 100.0, "y": 0.0}
    with pytest.raises(ScenarioError) as exc:
        Scenario.from_dict(data)
    assert exc.value.key == "robot.spawn"


def test_obstacle_on_top_of_spawn_is_rejected():
    data = minimal_scenario_dict(obstacles=[{"id": "onto_me", "x": 0.1, "y": 0.0, "radius": 0.5}])
    with pytest.raises(ScenarioError) as exc:
        Scenario.from_dict(data)
    assert exc.value.key == "obstacles[0]"
    assert "spawn" in str(exc.value)


def test_waypoint_mission_requires_waypoints():
    data = minimal_scenario_dict(controller={"type": "waypoint_mission"})
    with pytest.raises(ScenarioError) as exc:
        Scenario.from_dict(data)
    assert exc.value.key == "controller.waypoints"


def test_yaw_deg_is_converted_to_radians():
    data = minimal_scenario_dict()
    data["robot"]["spawn"] = {"x": 0.0, "y": 0.0, "yaw_deg": 90.0}
    scenario = Scenario.from_dict(data)
    assert scenario.spawn.yaw == pytest.approx(math.pi / 2)


def test_round_trip_through_to_dict(minimal_scenario):
    again = Scenario.from_dict(minimal_scenario.to_dict())
    assert again.to_dict() == minimal_scenario.to_dict()


# -- inheritance -----------------------------------------------------------


def test_extends_merges_base_and_overlay(tmp_path: Path):
    (tmp_path / "_base.yaml").write_text(
        "name: base\n"
        "robot: {type: diff_drive, max_speed: 1.0, spawn: {x: 0.0, y: 0.0}}\n"
        "goal: {x: 5.0, y: 0.0, tolerance: 0.25}\n"
        "sim: {dt: 0.05, time_limit: 20.0, seed: 1}\n",
        encoding="utf-8",
    )
    (tmp_path / "child.yaml").write_text(
        "extends: _base.yaml\nname: child\nrobot: {max_speed: 2.5}\nsim: {seed: 99}\n",
        encoding="utf-8",
    )
    scenario = load_scenario(tmp_path / "child.yaml")
    assert scenario.name == "child"
    assert scenario.robot.max_speed == pytest.approx(2.5)
    assert scenario.robot.type == "diff_drive"   # inherited
    assert scenario.sim.dt == pytest.approx(0.05)  # inherited
    assert scenario.sim.seed == 99                 # overridden


def test_extends_chain_of_two_levels(tmp_path: Path):
    (tmp_path / "_a.yaml").write_text(
        "name: a\nrobot: {max_speed: 1.0, spawn: {x: 0.0}}\ngoal: {x: 4.0, tolerance: 0.2}\n"
        "sim: {dt: 0.05, time_limit: 10.0}\n",
        encoding="utf-8",
    )
    (tmp_path / "_b.yaml").write_text("extends: _a.yaml\nname: b\nrobot: {max_speed: 2.0}\n", encoding="utf-8")
    (tmp_path / "c.yaml").write_text("extends: _b.yaml\nname: c\nrobot: {radius: 0.4}\n", encoding="utf-8")
    scenario = load_scenario(tmp_path / "c.yaml")
    assert scenario.robot.max_speed == pytest.approx(2.0)
    assert scenario.robot.radius == pytest.approx(0.4)


def test_circular_extends_is_detected(tmp_path: Path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\nname: a\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\nname: b\n", encoding="utf-8")
    with pytest.raises(ScenarioError) as exc:
        load_scenario(tmp_path / "a.yaml")
    assert "circular" in str(exc.value)


def test_missing_base_is_reported(tmp_path: Path):
    (tmp_path / "a.yaml").write_text("extends: nope.yaml\nname: a\n", encoding="utf-8")
    with pytest.raises(ScenarioError) as exc:
        load_scenario(tmp_path / "a.yaml")
    assert "not found" in str(exc.value)


def test_deep_merge_replaces_lists_and_merges_maps():
    merged = deep_merge({"a": {"x": 1, "y": 2}, "l": [1, 2, 3]}, {"a": {"y": 9}, "l": [7]})
    assert merged == {"a": {"x": 1, "y": 9}, "l": [7]}


# -- dotted paths ----------------------------------------------------------


def test_get_and_set_by_path_round_trip():
    data = {"disturbance": {"wind": [1.0, 2.0, 3.0]}, "sim": {"seed": 4}}
    assert get_by_path(data, "disturbance.wind[1]") == 2.0
    set_by_path(data, "disturbance.wind[1]", 8.0)
    assert data["disturbance"]["wind"][1] == 8.0
    set_by_path(data, "sim.seed", 11)
    assert data["sim"]["seed"] == 11


def test_set_by_path_refuses_to_invent_keys():
    data = {"sim": {"seed": 4}}
    with pytest.raises(ScenarioError):
        set_by_path(data, "sim.sed", 5)


def test_with_overrides_produces_a_new_validated_scenario(minimal_scenario):
    varied = minimal_scenario.with_overrides({"sim.seed": 4242, "robot.max_speed": 2.0})
    assert varied.sim.seed == 4242
    assert varied.robot.max_speed == pytest.approx(2.0)
    assert minimal_scenario.sim.seed == 7  # original untouched


def test_with_overrides_rejects_a_typo(minimal_scenario):
    with pytest.raises(ScenarioError):
        minimal_scenario.with_overrides({"disturbance.wnd[0]": 3.0})


# -- obstacle motion -------------------------------------------------------


def test_static_obstacle_does_not_move():
    obstacle = Obstacle.parse({"id": "s", "x": 2.0, "y": 1.0, "radius": 0.5}, "obstacles[0]", 0)
    assert obstacle.position_at(0.0) == obstacle.position_at(10.0)


def test_linear_obstacle_moves_at_constant_velocity():
    obstacle = Obstacle.parse(
        {"id": "m", "x": 0.0, "y": 0.0, "radius": 0.5, "motion": {"type": "linear", "vy": 2.0}},
        "obstacles[0]",
        0,
    )
    assert obstacle.position_at(3.0)[1] == pytest.approx(6.0)


def test_oscillating_obstacle_returns_to_start_each_period():
    obstacle = Obstacle.parse(
        {
            "id": "o",
            "x": 0.0,
            "y": 0.0,
            "radius": 0.5,
            "motion": {"type": "oscillate", "amplitude": 2.0, "period": 4.0, "axis": "y"},
        },
        "obstacles[0]",
        0,
    )
    assert obstacle.position_at(1.0)[1] == pytest.approx(2.0)
    assert obstacle.position_at(4.0)[1] == pytest.approx(0.0, abs=1e-9)


def test_box_clearance_is_negative_inside():
    obstacle = Obstacle.parse(
        {"id": "b", "shape": "box", "x": 0.0, "y": 0.0, "size_x": 2.0, "size_y": 2.0}, "obstacles[0]", 0
    )
    assert obstacle.clearance_from(0.0, 0.0, 0.0) == pytest.approx(-1.0)
    assert obstacle.clearance_from(3.0, 0.0, 0.0) == pytest.approx(2.0)


# -- directory loading -----------------------------------------------------


def test_shipped_scenarios_all_load(scenario_dir: Path):
    scenarios = load_scenario_dir(scenario_dir)
    names = {s.name for s in scenarios}
    assert len(scenarios) >= 8
    assert "straight_line" in names
    assert not any(n.startswith("_") for n in names), "base fragments must not be loaded as scenarios"


def test_exactly_one_shipped_scenario_expects_failure(scenario_dir: Path):
    expecting = [s.name for s in load_scenario_dir(scenario_dir) if s.expect_failure]
    assert expecting == ["expected_failure_no_avoidance"]
