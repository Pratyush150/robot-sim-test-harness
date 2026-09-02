"""The simulator abstraction every backend implements.

The harness never talks to Gazebo, AirSim or Isaac Sim directly. It talks to
:class:`Simulator`. That is the whole reason the assertion suite, the sweep
engine and the reports can be developed and tested on a laptop with no GPU and
no simulator installed, and then pointed at a real backend unchanged.

Six methods. That is the entire contract:

============== ==========================================================
``reset``      put the world in the scenario's initial state, return state
``send_command`` hand the vehicle a setpoint (latched until the next one)
``step``       advance simulation time by ``dt``, return the new state
``get_state``  current ground truth + the estimate the controller may see
``spawn``      add a model to the running world
``close``      release the backend (process, socket, GPU context)
============== ==========================================================
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from ..scenario import Pose, Scenario

__all__ = [
    "Command",
    "SimState",
    "SimulatorError",
    "SimulatorUnavailable",
    "Simulator",
]

Vec3 = Tuple[float, float, float]


class SimulatorError(RuntimeError):
    """A backend failed while running. Fatal for the scenario, not the suite."""


class SimulatorUnavailable(SimulatorError):
    """The backend is not installed or not reachable on this machine.

    The registry catches this and moves on to the next candidate, which is why
    a machine without Isaac Sim still runs the suite against the mock.
    """


@dataclass(frozen=True)
class Command:
    """A vehicle setpoint.

    The meaning of ``linear`` depends on the vehicle:

    * ``diff_drive`` -- ``linear[0]`` is forward body speed in m/s. ``linear[1]``
      and ``linear[2]`` are ignored, because a differential-drive base is
      nonholonomic and cannot honour them. They are kept in the type so one
      command struct serves both vehicles.
    * ``quadrotor`` -- world-frame velocity setpoint in m/s.

    ``yaw_rate`` is rad/s, CCW positive, for both.
    """

    linear: Vec3 = (0.0, 0.0, 0.0)
    yaw_rate: float = 0.0

    @property
    def forward(self) -> float:
        return self.linear[0]

    def magnitude(self) -> float:
        return math.sqrt(sum(c * c for c in self.linear))

    def clipped(self, max_speed: float, max_yaw_rate: float) -> "Command":
        """Saturate the command the way a real autopilot would."""
        mag = self.magnitude()
        linear = self.linear
        if mag > max_speed and mag > 0.0:
            scale = max_speed / mag
            linear = (linear[0] * scale, linear[1] * scale, linear[2] * scale)
        yaw_rate = max(-max_yaw_rate, min(max_yaw_rate, self.yaw_rate))
        return Command(linear=linear, yaw_rate=yaw_rate)


@dataclass
class SimState:
    """Ground truth plus what the sensors reported this step.

    Keeping ``position`` (truth) and ``est_position`` (measurement) separate is
    the difference between a harness that can test an estimator and one that
    quietly cheats by feeding perfect state to the controller.
    """

    t: float = 0.0
    position: Vec3 = (0.0, 0.0, 0.0)
    velocity: Vec3 = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    yaw_rate: float = 0.0
    est_position: Vec3 = (0.0, 0.0, 0.0)
    est_yaw: float = 0.0
    est_velocity: Vec3 = (0.0, 0.0, 0.0)
    sensor_valid: bool = True
    clearance: float = float("inf")
    collided: bool = False
    energy_j: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)

    def speed(self) -> float:
        return math.sqrt(sum(v * v for v in self.velocity))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "position": list(self.position),
            "velocity": list(self.velocity),
            "yaw": self.yaw,
            "yaw_rate": self.yaw_rate,
            "est_position": list(self.est_position),
            "est_yaw": self.est_yaw,
            "sensor_valid": self.sensor_valid,
            "clearance": self.clearance,
            "collided": self.collided,
            "energy_j": self.energy_j,
        }


class Simulator(abc.ABC):
    """Abstract simulator backend.

    Subclasses must be usable as a context manager and must tolerate
    ``close()`` being called more than once.
    """

    #: Short identifier used by the registry and printed in reports.
    name: str = "abstract"

    #: Human-readable description of how this backend connects to its engine.
    connection: str = ""

    @classmethod
    def availability(cls) -> Tuple[bool, str]:
        """Return ``(available, reason)``.

        The reason is logged either way, so a suite run on a laptop states
        plainly *why* it fell back to the mock instead of silently doing it.
        """
        return (False, "no availability check implemented")

    @classmethod
    def is_available(cls) -> bool:
        return cls.availability()[0]

    @abc.abstractmethod
    def reset(self, scenario: Scenario) -> SimState:
        """Load ``scenario``, place the robot at its spawn pose, zero the clock."""

    @abc.abstractmethod
    def step(self, dt: float) -> SimState:
        """Advance simulation time by ``dt`` seconds and return the new state."""

    @abc.abstractmethod
    def get_state(self) -> SimState:
        """Return the current state without advancing time."""

    @abc.abstractmethod
    def send_command(self, command: Command) -> None:
        """Latch a setpoint. It stays in effect until the next call."""

    @abc.abstractmethod
    def spawn(self, model_id: str, pose: Pose, **kwargs: Any) -> None:
        """Insert a model into the running world."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release the backend. Must be idempotent."""

    # -- convenience -------------------------------------------------------

    def __enter__(self) -> "Simulator":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def describe(self) -> Mapping[str, Any]:
        """Metadata recorded in the trace so a result is traceable to a backend."""
        return {"simulator": self.name, "connection": self.connection}
