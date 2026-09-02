"""Declarative scenario format.

A scenario is a YAML document describing *one test*: the world, where the robot
starts, where it must get to, what is in the way, how noisy its sensors are,
what the wind is doing, how long it has, and the behavioural assertions that
decide pass or fail.

Two properties matter more than anything else here:

1. **Validation errors name the offending key.** A typo in a 60-line scenario
   file should tell you ``obstacles[2].radius`` and not ``KeyError``.
2. **Scenarios compose.** ``extends:`` pulls in a base scenario and deep-merges
   the current document over it, so a family of tests can vary one field
   without copy-pasting the world.

Example
-------
.. code-block:: yaml

    name: straight_line
    extends: base_ground.yaml
    goal: {x: 8.0, y: 0.0, tolerance: 0.25, within: 25.0}
    assertions:
      - {type: reached_goal}
      - {type: no_collision}
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:  # PyYAML is a hard requirement for file loading, but not for the dataclasses.
    import yaml
except ImportError:  # pragma: no cover - exercised only on a broken install
    yaml = None  # type: ignore[assignment]

__all__ = [
    "ScenarioError",
    "Pose",
    "Goal",
    "ObstacleMotion",
    "Obstacle",
    "SensorProfile",
    "Disturbance",
    "RobotSpec",
    "WorldSpec",
    "SimSpec",
    "ControllerSpec",
    "AssertionSpec",
    "Scenario",
    "load_scenario",
    "load_scenario_dir",
    "get_by_path",
    "set_by_path",
    "deep_merge",
]

ROBOT_TYPES = ("diff_drive", "quadrotor")
OBSTACLE_SHAPES = ("circle", "box")
MOTION_TYPES = ("static", "linear", "oscillate", "circular")
CONTROLLER_TYPES = ("goto_goal", "waypoint_mission")


class ScenarioError(ValueError):
    """Raised when a scenario document is malformed.

    ``key`` holds the dotted path of the offending field, e.g.
    ``obstacles[2].radius``. The message always embeds it so that a bare
    ``print(exc)`` is still useful.
    """

    def __init__(self, key: str, message: str) -> None:
        self.key = key
        self.detail = message
        super().__init__(f"{key}: {message}" if key else message)


# --------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------


class _Node:
    """A mapping under validation, remembering where it came from."""

    def __init__(self, data: Any, path: str = "") -> None:
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise ScenarioError(path or "<root>", f"expected a mapping, got {_tname(data)}")
        self._data: Dict[str, Any] = dict(data)
        self.path = path

    def _child(self, key: str) -> str:
        return f"{self.path}.{key}" if self.path else key

    def unknown(self, allowed: Sequence[str]) -> None:
        """Reject keys that are not in ``allowed`` (typos are the common case)."""
        for key in self._data:
            if key not in allowed:
                near = _closest(str(key), allowed)
                hint = f"; did you mean '{near}'?" if near else ""
                raise ScenarioError(
                    self._child(str(key)),
                    f"unknown key{hint} (allowed: {', '.join(sorted(allowed))})",
                )

    def has(self, key: str) -> bool:
        return key in self._data and self._data[key] is not None

    def raw(self, key: str, default: Any = None) -> Any:
        value = self._data.get(key, default)
        return default if value is None else value

    def number(
        self,
        key: str,
        default: Optional[float] = None,
        *,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        positive: bool = False,
    ) -> float:
        path = self._child(key)
        if not self.has(key):
            if default is None:
                raise ScenarioError(path, "required numeric field is missing")
            return float(default)
        value = self._data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ScenarioError(path, f"expected a number, got {_tname(value)}")
        value = float(value)
        if not math.isfinite(value):
            raise ScenarioError(path, f"expected a finite number, got {value!r}")
        if positive and value <= 0.0:
            raise ScenarioError(path, f"must be > 0, got {value}")
        if minimum is not None and value < minimum:
            raise ScenarioError(path, f"must be >= {minimum}, got {value}")
        if maximum is not None and value > maximum:
            raise ScenarioError(path, f"must be <= {maximum}, got {value}")
        return value

    def integer(self, key: str, default: Optional[int] = None, *, minimum: Optional[int] = None) -> int:
        path = self._child(key)
        if not self.has(key):
            if default is None:
                raise ScenarioError(path, "required integer field is missing")
            return int(default)
        value = self._data[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ScenarioError(path, f"expected an integer, got {_tname(value)}")
        if minimum is not None and value < minimum:
            raise ScenarioError(path, f"must be >= {minimum}, got {value}")
        return int(value)

    def text(self, key: str, default: Optional[str] = None, *, choices: Optional[Sequence[str]] = None) -> str:
        path = self._child(key)
        if not self.has(key):
            if default is None:
                raise ScenarioError(path, "required string field is missing")
            value = default
        else:
            value = self._data[key]
        if not isinstance(value, str):
            raise ScenarioError(path, f"expected a string, got {_tname(value)}")
        if choices is not None and value not in choices:
            near = _closest(value, choices)
            hint = f"; did you mean '{near}'?" if near else ""
            raise ScenarioError(path, f"must be one of {', '.join(choices)}, got '{value}'{hint}")
        return value

    def flag(self, key: str, default: bool = False) -> bool:
        path = self._child(key)
        if key not in self._data or self._data[key] is None:
            return default
        value = self._data[key]
        if not isinstance(value, bool):
            raise ScenarioError(path, f"expected true/false, got {_tname(value)}")
        return value

    def mapping(self, key: str) -> "_Node":
        return _Node(self._data.get(key), self._child(key))

    def sequence(self, key: str) -> List[Tuple[Any, str]]:
        path = self._child(key)
        if not self.has(key):
            return []
        value = self._data[key]
        if not isinstance(value, (list, tuple)):
            raise ScenarioError(path, f"expected a list, got {_tname(value)}")
        return [(item, f"{path}[{i}]") for i, item in enumerate(value)]

    def point(self, key: str, default: Optional[Sequence[float]] = None, *, size: int = 2) -> Tuple[float, ...]:
        path = self._child(key)
        if not self.has(key):
            if default is None:
                raise ScenarioError(path, f"required {size}-element point is missing")
            return tuple(float(v) for v in default)
        value = self._data[key]
        if not isinstance(value, (list, tuple)) or len(value) != size:
            raise ScenarioError(path, f"expected a list of {size} numbers, got {value!r}")
        out = []
        for i, item in enumerate(value):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ScenarioError(f"{path}[{i}]", f"expected a number, got {_tname(item)}")
            out.append(float(item))
        return tuple(out)


def _tname(value: Any) -> str:
    return type(value).__name__


def _closest(word: str, candidates: Sequence[str]) -> Optional[str]:
    """Cheap edit-distance suggestion so validation errors are actionable."""
    best: Optional[str] = None
    best_score = 3
    for cand in candidates:
        score = _levenshtein(word.lower(), cand.lower())
        if score < best_score:
            best_score = score
            best = cand
    return best


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# --------------------------------------------------------------------------
# scenario pieces
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Pose:
    """A spawn or reference pose. ``yaw`` is radians, ENU, CCW-positive."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0

    @staticmethod
    def parse(node: _Node) -> "Pose":
        node.unknown(("x", "y", "z", "yaw", "yaw_deg"))
        yaw = node.number("yaw", 0.0)
        if node.has("yaw_deg"):
            yaw = math.radians(node.number("yaw_deg", 0.0))
        return Pose(node.number("x", 0.0), node.number("y", 0.0), node.number("z", 0.0), yaw)

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z, "yaw": self.yaw}


@dataclass(frozen=True)
class Goal:
    """Target position plus the acceptance radius and deadline."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    tolerance: float = 0.25
    within: Optional[float] = None

    @staticmethod
    def parse(node: _Node) -> "Goal":
        node.unknown(("x", "y", "z", "tolerance", "within"))
        within = node.number("within", -1.0, minimum=-1.0)
        return Goal(
            node.number("x", 0.0),
            node.number("y", 0.0),
            node.number("z", 0.0),
            node.number("tolerance", 0.25, positive=True),
            None if within < 0.0 else within,
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"x": self.x, "y": self.y, "z": self.z, "tolerance": self.tolerance}
        if self.within is not None:
            out["within"] = self.within
        return out


@dataclass(frozen=True)
class ObstacleMotion:
    """How a moving obstacle travels. All motion is a closed-form function of time.

    Closed-form (not integrated) is deliberate: obstacle position at time ``t``
    does not depend on the step history, so changing ``dt`` does not silently
    move the obstacles.
    """

    type: str = "static"
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    amplitude: float = 0.0
    period: float = 1.0
    axis: str = "y"
    radius: float = 1.0

    @staticmethod
    def parse(node: _Node) -> "ObstacleMotion":
        node.unknown(("type", "vx", "vy", "vz", "amplitude", "period", "axis", "radius"))
        mtype = node.text("type", "static", choices=MOTION_TYPES)
        return ObstacleMotion(
            type=mtype,
            vx=node.number("vx", 0.0),
            vy=node.number("vy", 0.0),
            vz=node.number("vz", 0.0),
            amplitude=node.number("amplitude", 0.0),
            period=node.number("period", 1.0, positive=True),
            axis=node.text("axis", "y", choices=("x", "y", "z")),
            radius=node.number("radius", 1.0, positive=True),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "vx": self.vx,
            "vy": self.vy,
            "vz": self.vz,
            "amplitude": self.amplitude,
            "period": self.period,
            "axis": self.axis,
            "radius": self.radius,
        }


@dataclass(frozen=True)
class Obstacle:
    """A circle (cylinder) or axis-aligned box obstacle, optionally moving."""

    id: str
    shape: str = "circle"
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    radius: float = 0.5
    size_x: float = 1.0
    size_y: float = 1.0
    height: float = 2.0
    motion: ObstacleMotion = field(default_factory=ObstacleMotion)

    @staticmethod
    def parse(data: Any, path: str, index: int) -> "Obstacle":
        node = _Node(data, path)
        node.unknown(("id", "shape", "x", "y", "z", "radius", "size_x", "size_y", "height", "motion"))
        shape = node.text("shape", "circle", choices=OBSTACLE_SHAPES)
        obstacle_id = node.text("id", f"obstacle_{index}")
        if shape == "circle":
            radius = node.number("radius", 0.5, positive=True)
            size_x = size_y = 2.0 * radius
        else:
            size_x = node.number("size_x", 1.0, positive=True)
            size_y = node.number("size_y", 1.0, positive=True)
            radius = 0.5 * math.hypot(size_x, size_y)
        motion = ObstacleMotion.parse(node.mapping("motion")) if node.has("motion") else ObstacleMotion()
        return Obstacle(
            id=obstacle_id,
            shape=shape,
            x=node.number("x", 0.0),
            y=node.number("y", 0.0),
            z=node.number("z", 0.0),
            radius=radius,
            size_x=size_x,
            size_y=size_y,
            height=node.number("height", 2.0, positive=True),
            motion=motion,
        )

    def position_at(self, t: float) -> Tuple[float, float, float]:
        """Closed-form obstacle centre at simulation time ``t``."""
        m = self.motion
        if m.type == "static":
            return (self.x, self.y, self.z)
        if m.type == "linear":
            return (self.x + m.vx * t, self.y + m.vy * t, self.z + m.vz * t)
        if m.type == "oscillate":
            offset = m.amplitude * math.sin(2.0 * math.pi * t / m.period)
            if m.axis == "x":
                return (self.x + offset, self.y, self.z)
            if m.axis == "y":
                return (self.x, self.y + offset, self.z)
            return (self.x, self.y, self.z + offset)
        # circular
        theta = 2.0 * math.pi * t / m.period
        return (self.x + m.radius * math.cos(theta), self.y + m.radius * math.sin(theta), self.z)

    def clearance_from(self, px: float, py: float, t: float) -> float:
        """Signed distance from point ``(px, py)`` to the obstacle *surface*.

        Negative means the point is inside the obstacle footprint.
        """
        cx, cy, _ = self.position_at(t)
        if self.shape == "circle":
            return math.hypot(px - cx, py - cy) - self.radius
        hx, hy = 0.5 * self.size_x, 0.5 * self.size_y
        return -_box_signed(px, py, cx - hx, cy - hy, cx + hx, cy + hy)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"id": self.id, "shape": self.shape, "x": self.x, "y": self.y, "z": self.z}
        if self.shape == "circle":
            out["radius"] = self.radius
        else:
            out["size_x"] = self.size_x
            out["size_y"] = self.size_y
        out["height"] = self.height
        if self.motion.type != "static":
            out["motion"] = self.motion.to_dict()
        return out


def _box_signed(px: float, py: float, minx: float, miny: float, maxx: float, maxy: float) -> float:
    """Positive inside the box, negative outside (distance to the boundary)."""
    left, right = px - minx, maxx - px
    bottom, top = py - miny, maxy - py
    if left >= 0.0 and right >= 0.0 and bottom >= 0.0 and top >= 0.0:
        return min(left, right, bottom, top)
    dx = max(minx - px, 0.0, px - maxx)
    dy = max(miny - py, 0.0, py - maxy)
    return -math.hypot(dx, dy)


@dataclass(frozen=True)
class SensorProfile:
    """Noise, bias and dropout applied to what the controller is allowed to see.

    The simulator always tracks perfect ground truth; the *estimate* handed to
    the controller is corrupted by this profile. That split is what makes a
    sensor-dropout scenario meaningful.
    """

    position_noise_std: float = 0.0
    yaw_noise_std: float = 0.0
    velocity_noise_std: float = 0.0
    position_bias: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dropout_probability: float = 0.0
    dropout_duration: float = 0.0
    latency: float = 0.0

    @staticmethod
    def parse(node: _Node) -> "SensorProfile":
        node.unknown(
            (
                "position_noise_std",
                "yaw_noise_std",
                "velocity_noise_std",
                "position_bias",
                "dropout_probability",
                "dropout_duration",
                "latency",
            )
        )
        return SensorProfile(
            position_noise_std=node.number("position_noise_std", 0.0, minimum=0.0),
            yaw_noise_std=node.number("yaw_noise_std", 0.0, minimum=0.0),
            velocity_noise_std=node.number("velocity_noise_std", 0.0, minimum=0.0),
            position_bias=node.point("position_bias", (0.0, 0.0, 0.0), size=3),  # type: ignore[arg-type]
            dropout_probability=node.number("dropout_probability", 0.0, minimum=0.0, maximum=1.0),
            dropout_duration=node.number("dropout_duration", 0.0, minimum=0.0),
            latency=node.number("latency", 0.0, minimum=0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_noise_std": self.position_noise_std,
            "yaw_noise_std": self.yaw_noise_std,
            "velocity_noise_std": self.velocity_noise_std,
            "position_bias": list(self.position_bias),
            "dropout_probability": self.dropout_probability,
            "dropout_duration": self.dropout_duration,
            "latency": self.latency,
        }


@dataclass(frozen=True)
class Disturbance:
    """Steady wind plus a sinusoidal gust and band-limited turbulence."""

    wind: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gust_amplitude: float = 0.0
    gust_period: float = 5.0
    gust_axis: str = "y"
    turbulence_std: float = 0.0

    @staticmethod
    def parse(node: _Node) -> "Disturbance":
        node.unknown(("wind", "gust_amplitude", "gust_period", "gust_axis", "turbulence_std"))
        return Disturbance(
            wind=node.point("wind", (0.0, 0.0, 0.0), size=3),  # type: ignore[arg-type]
            gust_amplitude=node.number("gust_amplitude", 0.0, minimum=0.0),
            gust_period=node.number("gust_period", 5.0, positive=True),
            gust_axis=node.text("gust_axis", "y", choices=("x", "y", "z")),
            turbulence_std=node.number("turbulence_std", 0.0, minimum=0.0),
        )

    def wind_at(self, t: float) -> Tuple[float, float, float]:
        """Deterministic wind vector at time ``t`` (turbulence is added by the sim)."""
        wx, wy, wz = self.wind
        if self.gust_amplitude != 0.0:
            gust = self.gust_amplitude * math.sin(2.0 * math.pi * t / self.gust_period)
            if self.gust_axis == "x":
                wx += gust
            elif self.gust_axis == "y":
                wy += gust
            else:
                wz += gust
        return (wx, wy, wz)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wind": list(self.wind),
            "gust_amplitude": self.gust_amplitude,
            "gust_period": self.gust_period,
            "gust_axis": self.gust_axis,
            "turbulence_std": self.turbulence_std,
        }


@dataclass(frozen=True)
class RobotSpec:
    """Vehicle limits and the first-order actuator model."""

    type: str = "diff_drive"
    radius: float = 0.25
    max_speed: float = 1.0
    max_accel: float = 2.0
    max_yaw_rate: float = 2.0
    actuator_tau: float = 0.15
    mass: float = 5.0
    battery_wh: float = 50.0
    idle_power_w: float = 2.0
    drag: float = 0.4

    @staticmethod
    def parse(node: _Node) -> "RobotSpec":
        node.unknown(
            (
                "type",
                "radius",
                "max_speed",
                "max_accel",
                "max_yaw_rate",
                "actuator_tau",
                "mass",
                "battery_wh",
                "idle_power_w",
                "drag",
                "spawn",
            )
        )
        return RobotSpec(
            type=node.text("type", "diff_drive", choices=ROBOT_TYPES),
            radius=node.number("radius", 0.25, positive=True),
            max_speed=node.number("max_speed", 1.0, positive=True),
            max_accel=node.number("max_accel", 2.0, positive=True),
            max_yaw_rate=node.number("max_yaw_rate", 2.0, positive=True),
            actuator_tau=node.number("actuator_tau", 0.15, minimum=0.0),
            mass=node.number("mass", 5.0, positive=True),
            battery_wh=node.number("battery_wh", 50.0, positive=True),
            idle_power_w=node.number("idle_power_w", 2.0, minimum=0.0),
            drag=node.number("drag", 0.4, minimum=0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "radius": self.radius,
            "max_speed": self.max_speed,
            "max_accel": self.max_accel,
            "max_yaw_rate": self.max_yaw_rate,
            "actuator_tau": self.actuator_tau,
            "mass": self.mass,
            "battery_wh": self.battery_wh,
            "idle_power_w": self.idle_power_w,
            "drag": self.drag,
        }


@dataclass(frozen=True)
class WorldSpec:
    """World name plus the axis-aligned bounds used as a default geofence."""

    name: str = "empty"
    min_xy: Tuple[float, float] = (-50.0, -50.0)
    max_xy: Tuple[float, float] = (50.0, 50.0)
    min_z: float = -1.0
    max_z: float = 30.0

    @staticmethod
    def parse(node: _Node) -> "WorldSpec":
        node.unknown(("name", "min_xy", "max_xy", "min_z", "max_z"))
        min_xy = node.point("min_xy", (-50.0, -50.0), size=2)
        max_xy = node.point("max_xy", (50.0, 50.0), size=2)
        for i, axis in enumerate("xy"):
            if max_xy[i] <= min_xy[i]:
                raise ScenarioError(
                    f"{node.path}.max_xy" if node.path else "max_xy",
                    f"max_xy[{i}] ({max_xy[i]}) must exceed min_xy[{i}] ({min_xy[i]}) on the {axis} axis",
                )
        return WorldSpec(
            name=node.text("name", "empty"),
            min_xy=min_xy,  # type: ignore[arg-type]
            max_xy=max_xy,  # type: ignore[arg-type]
            min_z=node.number("min_z", -1.0),
            max_z=node.number("max_z", 30.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "min_xy": list(self.min_xy),
            "max_xy": list(self.max_xy),
            "min_z": self.min_z,
            "max_z": self.max_z,
        }


@dataclass(frozen=True)
class SimSpec:
    """Integration step, wall/sim time limit and the master seed."""

    dt: float = 0.02
    time_limit: float = 30.0
    seed: int = 0
    stop_on_goal: bool = True
    settle_time: float = 0.0

    @staticmethod
    def parse(node: _Node) -> "SimSpec":
        node.unknown(("dt", "time_limit", "seed", "stop_on_goal", "settle_time"))
        return SimSpec(
            dt=node.number("dt", 0.02, positive=True, maximum=1.0),
            time_limit=node.number("time_limit", 30.0, positive=True),
            seed=node.integer("seed", 0, minimum=0),
            stop_on_goal=node.flag("stop_on_goal", True),
            settle_time=node.number("settle_time", 0.0, minimum=0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dt": self.dt,
            "time_limit": self.time_limit,
            "seed": self.seed,
            "stop_on_goal": self.stop_on_goal,
            "settle_time": self.settle_time,
        }


@dataclass(frozen=True)
class ControllerSpec:
    """The controller under test: which one, its gains, and its waypoints."""

    type: str = "goto_goal"
    kp_linear: float = 1.0
    kp_angular: float = 3.0
    kd_linear: float = 0.0
    avoid_gain: float = 1.2
    avoid_range: float = 1.5
    waypoint_tolerance: float = 0.35
    waypoints: Tuple[Tuple[float, float, float], ...] = ()

    @staticmethod
    def parse(node: _Node) -> "ControllerSpec":
        node.unknown(
            (
                "type",
                "kp_linear",
                "kp_angular",
                "kd_linear",
                "avoid_gain",
                "avoid_range",
                "waypoint_tolerance",
                "waypoints",
            )
        )
        ctype = node.text("type", "goto_goal", choices=CONTROLLER_TYPES)
        waypoints: List[Tuple[float, float, float]] = []
        for item, path in node.sequence("waypoints"):
            if not isinstance(item, (list, tuple)) or len(item) not in (2, 3):
                raise ScenarioError(path, f"waypoint must be [x, y] or [x, y, z], got {item!r}")
            for i, value in enumerate(item):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ScenarioError(f"{path}[{i}]", f"expected a number, got {_tname(value)}")
            wp = tuple(float(v) for v in item)
            waypoints.append((wp[0], wp[1], wp[2] if len(wp) > 2 else 0.0))
        if ctype == "waypoint_mission" and not waypoints:
            raise ScenarioError(
                f"{node.path}.waypoints" if node.path else "waypoints",
                "controller type 'waypoint_mission' requires at least one waypoint",
            )
        return ControllerSpec(
            type=ctype,
            kp_linear=node.number("kp_linear", 1.0, positive=True),
            kp_angular=node.number("kp_angular", 3.0, positive=True),
            kd_linear=node.number("kd_linear", 0.0, minimum=0.0),
            avoid_gain=node.number("avoid_gain", 1.2, minimum=0.0),
            avoid_range=node.number("avoid_range", 1.5, positive=True),
            waypoint_tolerance=node.number("waypoint_tolerance", 0.35, positive=True),
            waypoints=tuple(waypoints),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "kp_linear": self.kp_linear,
            "kp_angular": self.kp_angular,
            "kd_linear": self.kd_linear,
            "avoid_gain": self.avoid_gain,
            "avoid_range": self.avoid_range,
            "waypoint_tolerance": self.waypoint_tolerance,
            "waypoints": [list(w) for w in self.waypoints],
        }


@dataclass(frozen=True)
class AssertionSpec:
    """One behavioural assertion, kept as its validated parameter mapping.

    The concrete checking lives in :mod:`simharness.assertions`; keeping the
    spec as data means a scenario can round-trip through YAML unchanged.
    """

    type: str
    params: Mapping[str, Any] = field(default_factory=dict)
    name: Optional[str] = None
    path: str = ""

    @property
    def label(self) -> str:
        return self.name or self.type

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"type": self.type}
        if self.name:
            out["name"] = self.name
        out.update({k: v for k, v in self.params.items()})
        return out


# --------------------------------------------------------------------------
# the scenario itself
# --------------------------------------------------------------------------

_TOP_LEVEL_KEYS = (
    "name",
    "description",
    "extends",
    "tags",
    "world",
    "robot",
    "goal",
    "obstacles",
    "sensors",
    "disturbance",
    "controller",
    "sim",
    "assertions",
    "expect_failure",
    "metadata",
)


@dataclass(frozen=True)
class Scenario:
    """A fully validated, immutable scenario."""

    name: str
    description: str = ""
    tags: Tuple[str, ...] = ()
    world: WorldSpec = field(default_factory=WorldSpec)
    robot: RobotSpec = field(default_factory=RobotSpec)
    spawn: Pose = field(default_factory=Pose)
    goal: Goal = field(default_factory=Goal)
    obstacles: Tuple[Obstacle, ...] = ()
    sensors: SensorProfile = field(default_factory=SensorProfile)
    disturbance: Disturbance = field(default_factory=Disturbance)
    controller: ControllerSpec = field(default_factory=ControllerSpec)
    sim: SimSpec = field(default_factory=SimSpec)
    assertions: Tuple[AssertionSpec, ...] = ()
    expect_failure: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source: Optional[str] = None

    # -- construction ------------------------------------------------------

    @staticmethod
    def from_dict(data: Mapping[str, Any], *, source: Optional[str] = None) -> "Scenario":
        """Validate a raw mapping into a :class:`Scenario`.

        Raises :class:`ScenarioError` naming the offending key.
        """
        root = _Node(data, "")
        root.unknown(_TOP_LEVEL_KEYS)
        name = root.text("name")
        if not name.strip():
            raise ScenarioError("name", "must be a non-empty string")

        tags: List[str] = []
        for item, path in root.sequence("tags"):
            if not isinstance(item, str):
                raise ScenarioError(path, f"expected a string tag, got {_tname(item)}")
            tags.append(item)

        robot_node = root.mapping("robot")
        robot = RobotSpec.parse(robot_node)
        spawn = Pose.parse(robot_node.mapping("spawn"))

        obstacles = [
            Obstacle.parse(item, path, index)
            for index, (item, path) in enumerate(root.sequence("obstacles"))
        ]
        seen: Dict[str, str] = {}
        for index, (obstacle, (_, path)) in enumerate(zip(obstacles, root.sequence("obstacles"))):
            if obstacle.id in seen:
                raise ScenarioError(f"{path}.id", f"duplicate obstacle id '{obstacle.id}' (also at {seen[obstacle.id]})")
            seen[obstacle.id] = path

        assertions = [
            _parse_assertion(item, path) for item, path in root.sequence("assertions")
        ]

        metadata = root.raw("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ScenarioError("metadata", f"expected a mapping, got {_tname(metadata)}")

        scenario = Scenario(
            name=name,
            description=root.text("description", ""),
            tags=tuple(tags),
            world=WorldSpec.parse(root.mapping("world")),
            robot=robot,
            spawn=spawn,
            goal=Goal.parse(root.mapping("goal")),
            obstacles=tuple(obstacles),
            sensors=SensorProfile.parse(root.mapping("sensors")),
            disturbance=Disturbance.parse(root.mapping("disturbance")),
            controller=ControllerSpec.parse(root.mapping("controller")),
            sim=SimSpec.parse(root.mapping("sim")),
            assertions=tuple(assertions),
            expect_failure=root.flag("expect_failure", False),
            metadata=dict(metadata),
            source=source,
        )
        scenario._cross_validate()
        return scenario

    def _cross_validate(self) -> None:
        """Checks that need more than one field to be in hand."""
        if self.goal.within is not None and self.goal.within > self.sim.time_limit:
            raise ScenarioError(
                "goal.within",
                f"deadline {self.goal.within}s exceeds sim.time_limit {self.sim.time_limit}s, "
                "so the assertion can never pass",
            )
        if self.sim.dt >= self.sim.time_limit:
            raise ScenarioError("sim.dt", f"step {self.sim.dt}s is not smaller than time_limit {self.sim.time_limit}s")
        world = self.world
        if not (world.min_xy[0] <= self.spawn.x <= world.max_xy[0] and world.min_xy[1] <= self.spawn.y <= world.max_xy[1]):
            raise ScenarioError(
                "robot.spawn",
                f"spawn ({self.spawn.x}, {self.spawn.y}) lies outside world bounds "
                f"{list(world.min_xy)}..{list(world.max_xy)}",
            )
        for i, obstacle in enumerate(self.obstacles):
            if obstacle.clearance_from(self.spawn.x, self.spawn.y, 0.0) < self.robot.radius:
                raise ScenarioError(
                    f"obstacles[{i}]",
                    f"obstacle '{obstacle.id}' overlaps the spawn pose at t=0 "
                    f"(clearance {obstacle.clearance_from(self.spawn.x, self.spawn.y, 0.0):.3f} m "
                    f"< robot radius {self.robot.radius} m)",
                )

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Round-trippable mapping. ``Scenario.from_dict(s.to_dict()) == s``."""
        robot = self.robot.to_dict()
        robot["spawn"] = self.spawn.to_dict()
        out: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "world": self.world.to_dict(),
            "robot": robot,
            "goal": self.goal.to_dict(),
            "obstacles": [o.to_dict() for o in self.obstacles],
            "sensors": self.sensors.to_dict(),
            "disturbance": self.disturbance.to_dict(),
            "controller": self.controller.to_dict(),
            "sim": self.sim.to_dict(),
            "assertions": [a.to_dict() for a in self.assertions],
            "expect_failure": self.expect_failure,
        }
        if self.metadata:
            out["metadata"] = dict(self.metadata)
        return out

    def with_overrides(self, overrides: Mapping[str, Any]) -> "Scenario":
        """Return a new scenario with dotted-path ``overrides`` applied.

        This is what :mod:`simharness.sweep` uses to vary one parameter at a
        time. Unknown paths raise :class:`ScenarioError`, so a typo in a sweep
        definition fails loudly instead of silently sweeping nothing.
        """
        data = self.to_dict()
        for path, value in overrides.items():
            set_by_path(data, path, value)
        return Scenario.from_dict(data, source=self.source)

    def goal_xyz(self) -> Tuple[float, float, float]:
        return (self.goal.x, self.goal.y, self.goal.z)

    def spawn_xyz(self) -> Tuple[float, float, float]:
        return (self.spawn.x, self.spawn.y, self.spawn.z)

    def straight_line_distance(self) -> float:
        """Distance from spawn to goal, used as the optimal-path baseline."""
        return math.dist(self.spawn_xyz(), self.goal_xyz())


def _parse_assertion(data: Any, path: str) -> AssertionSpec:
    node = _Node(data, path)
    atype = node.text("type")
    params = {k: v for k, v in dict(data).items() if k not in ("type", "name")}
    name = node.text("name", "") or None
    return AssertionSpec(type=atype, params=params, name=name, path=path)


# --------------------------------------------------------------------------
# loading, inheritance and dotted-path access
# --------------------------------------------------------------------------


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overlay`` over ``base``.

    Mappings merge key-by-key. Lists **replace** wholesale: half-overriding a
    list of obstacles by index is the kind of cleverness that makes scenario
    files impossible to read.
    """
    out: Dict[str, Any] = dict(copy.deepcopy(dict(base)))
    for key, value in overlay.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def get_by_path(data: Mapping[str, Any], path: str) -> Any:
    """Read ``data`` at a dotted path, e.g. ``disturbance.wind[0]``."""
    node: Any = data
    for token, index in _tokenise(path):
        if not isinstance(node, Mapping) or token not in node:
            raise ScenarioError(path, f"no such key '{token}'")
        node = node[token]
        if index is not None:
            if not isinstance(node, list) or not (-len(node) <= index < len(node)):
                raise ScenarioError(path, f"index [{index}] out of range for '{token}'")
            node = node[index]
    return node


def set_by_path(data: MutableMapping[str, Any], path: str, value: Any) -> None:
    """Write ``value`` into ``data`` at a dotted path, creating nothing new.

    Refusing to create keys is deliberate: a sweep over ``disturbance.wnd.x``
    should fail, not quietly add a field nobody reads.
    """
    tokens = _tokenise(path)
    node: Any = data
    for i, (token, index) in enumerate(tokens):
        last = i == len(tokens) - 1
        if not isinstance(node, Mapping) or token not in node:
            raise ScenarioError(path, f"no such key '{token}'")
        if last and index is None:
            node[token] = value  # type: ignore[index]
            return
        node = node[token]
        if index is not None:
            if not isinstance(node, list) or not (-len(node) <= index < len(node)):
                raise ScenarioError(path, f"index [{index}] out of range for '{token}'")
            if last:
                node[index] = value
                return
            node = node[index]


def _tokenise(path: str) -> List[Tuple[str, Optional[int]]]:
    if not path:
        raise ScenarioError("<path>", "empty override path")
    tokens: List[Tuple[str, Optional[int]]] = []
    for part in path.split("."):
        if part.endswith("]") and "[" in part:
            head, _, tail = part.partition("[")
            try:
                index = int(tail[:-1])
            except ValueError as exc:
                raise ScenarioError(path, f"bad list index in '{part}'") from exc
            tokens.append((head, index))
        else:
            tokens.append((part, None))
    return tokens


def _read_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise ScenarioError(str(path), "PyYAML is not installed; run 'pip install pyyaml'")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScenarioError(str(path), f"cannot read scenario file: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:  # pragma: no cover - message varies by PyYAML build
        raise ScenarioError(str(path), f"invalid YAML: {exc}") from exc
    if data is None:
        raise ScenarioError(str(path), "scenario file is empty")
    if not isinstance(data, Mapping):
        raise ScenarioError(str(path), f"top level must be a mapping, got {_tname(data)}")
    return dict(data)


def _resolve_document(path: Path, seen: List[Path]) -> Dict[str, Any]:
    """Load one YAML document and fold in everything it ``extends``."""
    real = path.resolve()
    if real in seen:
        chain = " -> ".join(p.name for p in seen + [real])
        raise ScenarioError("extends", f"circular scenario inheritance: {chain}")
    seen = seen + [real]
    doc = _read_yaml(real)
    parents = doc.pop("extends", None)
    if parents is None:
        return doc
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or not all(isinstance(p, str) for p in parents):
        raise ScenarioError("extends", f"expected a path or list of paths, got {parents!r}")
    merged: Dict[str, Any] = {}
    for parent in parents:
        parent_path = (real.parent / parent).resolve()
        if not parent_path.exists():
            raise ScenarioError("extends", f"base scenario '{parent}' not found next to {real.name}")
        merged = deep_merge(merged, _resolve_document(parent_path, seen))
    return deep_merge(merged, doc)


def load_scenario(path: os.PathLike | str, *, overrides: Optional[Mapping[str, Any]] = None) -> Scenario:
    """Load, resolve inheritance for, and validate a scenario YAML file."""
    p = Path(path)
    if not p.exists():
        raise ScenarioError(str(p), "scenario file does not exist")
    data = _resolve_document(p, [])
    data.setdefault("name", p.stem)
    scenario = Scenario.from_dict(data, source=str(p))
    if overrides:
        scenario = scenario.with_overrides(overrides)
    return scenario


def load_scenario_dir(directory: os.PathLike | str, *, recursive: bool = True) -> List[Scenario]:
    """Load every ``*.yaml`` scenario in a directory, skipping ``_``-prefixed bases.

    A file whose name starts with ``_`` is treated as a base/overlay fragment
    and is only pulled in through ``extends``.
    """
    root = Path(directory)
    if not root.is_dir():
        raise ScenarioError(str(root), "scenario directory does not exist")
    pattern = "**/*.y*ml" if recursive else "*.y*ml"
    scenarios: List[Scenario] = []
    for path in sorted(root.glob(pattern)):
        if path.name.startswith("_"):
            continue
        scenarios.append(load_scenario(path))
    return scenarios
