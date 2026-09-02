"""A real, headless simulator that ships with the harness.

This is not a stub. It integrates differential-drive and quadrotor point-mass
dynamics with first-order actuator lag, applies wind and turbulence, detects
collisions against static and moving obstacles, and corrupts the state estimate
with configurable noise, bias, latency and dropout.

Why it exists: the value of this repo is the *harness*, and a harness that can
only be exercised on a workstation with Isaac Sim installed cannot be unit
tested, cannot run in CI, and cannot be reviewed by anyone in under a day.
``MockSimulator`` makes the whole pipeline runnable on a laptop in under a
second, so the scenario format, the temporal assertions, the sweep engine and
the reports are all covered by tests that need nothing but Python.

What it is not: an aerodynamic model. There is no rotor wake, no wheel slip
model, no contact solver, no sensor raytracing. Read ``docs/SIM_TESTING.md``
for what does and does not transfer.

Models
------

The actuator lag is integrated with the **exact** discrete solution of the
first-order ODE, ``alpha = 1 - exp(-dt / tau)``, rather than explicit Euler.
That matters: with explicit Euler, any ``dt`` larger than ``tau`` makes the
actuator model oscillate or blow up, so a scenario that merely lowered its
step rate would start producing nonsense. The exponential form is
unconditionally stable and exact, so ``dt`` and ``tau`` are independent knobs.

**Differential drive** (nonholonomic, planar)::

    alpha  = 1 - exp(-dt / tau)
    dv     = (v_cmd - v) * alpha
    v      += clamp(dv / dt, +/-max_accel) * dt
    omega  <- the same lag toward omega_cmd, clamped to max_yaw_rate
    yaw    += omega * dt
    p      += (v * [cos yaw, sin yaw, 0] + drag * wind) * dt

The wheels grip, so wind does not accelerate the base; it produces a slip
velocity proportional to ``robot.drag``.

**Quadrotor** (holonomic point mass)::

    a_cmd  = clamp((v_cmd - v) * alpha / dt, max_accel)
    a_wind = drag * (wind - v)           # relative-airspeed drag
    v      += (a_cmd + a_wind) * dt
    p      += v * dt

A steady wind therefore produces a real steady-state position error unless the
controller integrates it out, which is what the wind-rejection scenario tests.

**Energy**::

    P = idle_power_w + mass * |a| * |v| + mass * drag * |v|^2
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional, Tuple

from ..geometry import clamp, wrap_angle
from ..scenario import Disturbance, Obstacle, ObstacleMotion, Pose, Scenario, SensorProfile
from .base import Command, SimState, Simulator, SimulatorError

__all__ = ["MockSimulator"]

Vec3 = Tuple[float, float, float]


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Vec3, k: float) -> Vec3:
    return (a[0] * k, a[1] * k, a[2] * k)


def _norm(a: Vec3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _lag_alpha(dt: float, tau: float) -> float:
    """Exact first-order lag coefficient, ``1 - exp(-dt / tau)``.

    Unconditionally stable for any ``dt``, unlike ``dt / tau``, which is what
    an explicit-Euler actuator model uses and which diverges once ``dt`` gets
    within a factor of two of ``tau``.
    """
    if tau <= 0.0:
        return 1.0
    ratio = dt / tau
    if ratio > 40.0:  # exp(-40) is already below float resolution against 1.0
        return 1.0
    return 1.0 - math.exp(-ratio)


def _clamp_norm(a: Vec3, limit: float) -> Vec3:
    mag = _norm(a)
    if mag <= limit or mag == 0.0:
        return a
    return _scale(a, limit / mag)


@dataclass
class _SensorChannel:
    """Noise, bias, latency and dropout applied to one state estimate."""

    profile: SensorProfile
    rng: random.Random
    dt: float
    latency_steps: int = 0
    buffer: Deque[Tuple[Vec3, float, Vec3]] = field(default_factory=deque)
    dropout_remaining: float = 0.0
    last_valid: Optional[Tuple[Vec3, float, Vec3]] = None
    dropout_started: bool = False

    def measure(
        self, position: Vec3, yaw: float, velocity: Vec3
    ) -> Tuple[Vec3, float, Vec3, bool, bool]:
        """Return ``(est_pos, est_yaw, est_vel, valid, dropout_began)``."""
        p = self.profile
        noisy = (
            position[0] + p.position_bias[0] + self._gauss(p.position_noise_std),
            position[1] + p.position_bias[1] + self._gauss(p.position_noise_std),
            position[2] + p.position_bias[2] + self._gauss(p.position_noise_std),
        )
        noisy_yaw = wrap_angle(yaw + self._gauss(p.yaw_noise_std))
        noisy_vel = (
            velocity[0] + self._gauss(p.velocity_noise_std),
            velocity[1] + self._gauss(p.velocity_noise_std),
            velocity[2] + self._gauss(p.velocity_noise_std),
        )

        self.buffer.append((noisy, noisy_yaw, noisy_vel))
        while len(self.buffer) > self.latency_steps + 1:
            self.buffer.popleft()
        delayed = self.buffer[0]

        dropout_began = False
        if self.dropout_remaining > 0.0:
            self.dropout_remaining = max(0.0, self.dropout_remaining - self.dt)
        elif p.dropout_probability > 0.0 and self.rng.random() < p.dropout_probability:
            self.dropout_remaining = max(p.dropout_duration, self.dt)
            dropout_began = True

        if self.dropout_remaining > 0.0:
            if self.last_valid is None:
                self.last_valid = delayed
            held = self.last_valid
            return (held[0], held[1], held[2], False, dropout_began)

        self.last_valid = delayed
        return (delayed[0], delayed[1], delayed[2], True, False)

    def _gauss(self, std: float) -> float:
        if std <= 0.0:
            return 0.0
        return self.rng.gauss(0.0, std)


class MockSimulator(Simulator):
    """Headless kinematic simulator. Deterministic given a scenario seed."""

    name = "mock"
    connection = "in-process; no external engine, no sockets, no GPU"

    def __init__(self, *, seed_salt: str = "") -> None:
        self._scenario: Optional[Scenario] = None
        self._seed_salt = seed_salt
        self._closed = False

        self._t = 0.0
        self._position: Vec3 = (0.0, 0.0, 0.0)
        self._velocity: Vec3 = (0.0, 0.0, 0.0)
        self._accel: Vec3 = (0.0, 0.0, 0.0)
        self._yaw = 0.0
        self._yaw_rate = 0.0
        self._forward = 0.0
        self._energy_j = 0.0
        self._collided = False
        self._clearance = float("inf")
        self._command = Command()
        self._obstacles: List[Obstacle] = []
        self._sensors: Optional[_SensorChannel] = None
        self._turbulence: Optional[random.Random] = None
        self._events: List[Tuple[float, str, str]] = []
        self._state = SimState()
        self._steps = 0

    # -- availability ------------------------------------------------------

    @classmethod
    def availability(cls) -> Tuple[bool, str]:
        return (True, "built in; always available")

    # -- lifecycle ---------------------------------------------------------

    def reset(self, scenario: Scenario) -> SimState:
        """Load a scenario and place the robot at its spawn pose at t=0."""
        if self._closed:
            raise SimulatorError("MockSimulator.reset() called after close()")
        self._scenario = scenario
        self._t = 0.0
        self._steps = 0
        spawn = scenario.spawn
        self._position = (spawn.x, spawn.y, spawn.z)
        self._velocity = (0.0, 0.0, 0.0)
        self._accel = (0.0, 0.0, 0.0)
        self._yaw = wrap_angle(spawn.yaw)
        self._yaw_rate = 0.0
        self._forward = 0.0
        self._energy_j = 0.0
        self._collided = False
        self._command = Command()
        self._obstacles = list(scenario.obstacles)
        self._events = []

        seed = scenario.sim.seed
        salt = self._seed_salt
        self._sensors = _SensorChannel(
            profile=scenario.sensors,
            rng=random.Random(f"simharness:sensor:{seed}:{salt}"),
            dt=scenario.sim.dt,
            latency_steps=int(round(scenario.sensors.latency / scenario.sim.dt)),
        )
        self._turbulence = random.Random(f"simharness:turb:{seed}:{salt}")

        self._clearance = self._compute_clearance(self._position, 0.0)
        self._state = self._observe(dropout_check=False)
        return self._state

    def close(self) -> None:
        """Idempotent. Nothing to release, but the contract says it must exist."""
        self._closed = True

    # -- control -----------------------------------------------------------

    def send_command(self, command: Command) -> None:
        if self._scenario is None:
            raise SimulatorError("send_command() before reset()")
        robot = self._scenario.robot
        self._command = command.clipped(robot.max_speed, robot.max_yaw_rate)

    def spawn(self, model_id: str, pose: Pose, **kwargs: Any) -> None:
        """Insert an obstacle into the running world.

        ``kwargs`` mirror the scenario obstacle fields (``shape``, ``radius``,
        ``size_x``, ``size_y``, ``height``, ``motion``). Used by scenarios that
        need something to appear mid-run.
        """
        if self._scenario is None:
            raise SimulatorError("spawn() before reset()")
        if any(o.id == model_id for o in self._obstacles):
            raise SimulatorError(f"model id '{model_id}' already exists in the world")
        motion_kw = kwargs.pop("motion", None)
        motion = ObstacleMotion(**motion_kw) if isinstance(motion_kw, dict) else ObstacleMotion()
        shape = kwargs.pop("shape", "circle")
        radius = float(kwargs.pop("radius", 0.5))
        size_x = float(kwargs.pop("size_x", 2.0 * radius))
        size_y = float(kwargs.pop("size_y", 2.0 * radius))
        height = float(kwargs.pop("height", 2.0))
        if kwargs:
            raise SimulatorError(f"spawn() got unexpected keyword(s): {', '.join(sorted(kwargs))}")
        self._obstacles.append(
            Obstacle(
                id=model_id,
                shape=shape,
                x=pose.x,
                y=pose.y,
                z=pose.z,
                radius=radius,
                size_x=size_x,
                size_y=size_y,
                height=height,
                motion=motion,
            )
        )
        self._events.append((self._t, "spawn", f"spawned '{model_id}' at ({pose.x:.2f}, {pose.y:.2f})"))

    # -- stepping ----------------------------------------------------------

    def step(self, dt: float) -> SimState:
        """Advance the world by ``dt`` seconds using semi-implicit Euler."""
        if self._scenario is None:
            raise SimulatorError("step() before reset()")
        if dt <= 0.0:
            raise SimulatorError(f"step(dt={dt}) requires dt > 0")
        scenario = self._scenario
        robot = scenario.robot
        wind = self._wind_at(self._t, scenario.disturbance)

        if robot.type == "diff_drive":
            self._step_diff_drive(dt, wind)
        else:
            self._step_quadrotor(dt, wind)

        self._t += dt
        self._steps += 1
        self._accumulate_energy(dt)

        previous_collided = self._collided
        self._clearance = self._compute_clearance(self._position, self._t)
        if self._clearance < 0.0 and not previous_collided:
            self._collided = True
            self._events.append(
                (self._t, "collision", f"penetration {-self._clearance:.3f} m at t={self._t:.3f}s")
            )
        self._state = self._observe()
        return self._state

    def _step_diff_drive(self, dt: float, wind: Vec3) -> None:
        robot = self._scenario.robot  # type: ignore[union-attr]
        tau = max(robot.actuator_tau, 1e-9)
        v_cmd = clamp(self._command.forward, -robot.max_speed, robot.max_speed)

        alpha = _lag_alpha(dt, tau)
        accel = clamp((v_cmd - self._forward) * alpha / dt, -robot.max_accel, robot.max_accel)
        self._forward = clamp(self._forward + accel * dt, -robot.max_speed, robot.max_speed)

        # Yaw responds faster than translation on a differential-drive base:
        # spinning the wheels in opposite directions moves far less mass.
        yaw_alpha = _lag_alpha(dt, max(0.5 * tau, 1e-9))
        self._yaw_rate = clamp(
            self._yaw_rate + (self._command.yaw_rate - self._yaw_rate) * yaw_alpha,
            -robot.max_yaw_rate,
            robot.max_yaw_rate,
        )
        self._yaw = wrap_angle(self._yaw + self._yaw_rate * dt)

        slip = _scale(wind, robot.drag)
        world_v = (
            self._forward * math.cos(self._yaw) + slip[0],
            self._forward * math.sin(self._yaw) + slip[1],
            0.0,
        )
        self._accel = _scale(_sub(world_v, self._velocity), 1.0 / dt)
        self._velocity = world_v
        self._position = _add(self._position, _scale(world_v, dt))
        self._position = (self._position[0], self._position[1], self._scenario.spawn.z)  # type: ignore[union-attr]

    def _step_quadrotor(self, dt: float, wind: Vec3) -> None:
        robot = self._scenario.robot  # type: ignore[union-attr]
        tau = max(robot.actuator_tau, 1e-9)
        alpha = _lag_alpha(dt, tau)
        a_cmd = _clamp_norm(
            _scale(_sub(self._command.linear, self._velocity), alpha / dt), robot.max_accel
        )
        a_wind = _scale(_sub(wind, self._velocity), robot.drag)
        accel = _add(a_cmd, a_wind)
        self._accel = accel
        self._velocity = _clamp_norm(_add(self._velocity, _scale(accel, dt)), robot.max_speed * 1.5)
        self._position = _add(self._position, _scale(self._velocity, dt))

        yaw_alpha = _lag_alpha(dt, max(0.5 * tau, 1e-9))
        self._yaw_rate = clamp(
            self._yaw_rate + (self._command.yaw_rate - self._yaw_rate) * yaw_alpha,
            -robot.max_yaw_rate,
            robot.max_yaw_rate,
        )
        self._yaw = wrap_angle(self._yaw + self._yaw_rate * dt)
        self._forward = _norm(self._velocity)

    def _wind_at(self, t: float, disturbance: Disturbance) -> Vec3:
        wind = disturbance.wind_at(t)
        if disturbance.turbulence_std > 0.0 and self._turbulence is not None:
            wind = (
                wind[0] + self._turbulence.gauss(0.0, disturbance.turbulence_std),
                wind[1] + self._turbulence.gauss(0.0, disturbance.turbulence_std),
                wind[2] + self._turbulence.gauss(0.0, disturbance.turbulence_std),
            )
        return wind

    def _accumulate_energy(self, dt: float) -> None:
        robot = self._scenario.robot  # type: ignore[union-attr]
        speed = _norm(self._velocity)
        power = robot.idle_power_w + robot.mass * _norm(self._accel) * speed + robot.mass * robot.drag * speed * speed
        self._energy_j += power * dt

    def _compute_clearance(self, position: Vec3, t: float) -> float:
        """Smallest gap between the robot hull and any obstacle. Negative = contact."""
        robot = self._scenario.robot  # type: ignore[union-attr]
        best = float("inf")
        for obstacle in self._obstacles:
            if position[2] > obstacle.z + obstacle.height:
                continue  # flown over the top of it
            gap = obstacle.clearance_from(position[0], position[1], t) - robot.radius
            if gap < best:
                best = gap
        return best

    def _observe(self, *, dropout_check: bool = True) -> SimState:
        assert self._sensors is not None
        if dropout_check:
            est_p, est_yaw, est_v, valid, began = self._sensors.measure(
                self._position, self._yaw, self._velocity
            )
            if began:
                self._events.append((self._t, "sensor_dropout", f"position fix lost at t={self._t:.3f}s"))
        else:
            est_p, est_yaw, est_v, valid = self._position, self._yaw, self._velocity, True
        return SimState(
            t=self._t,
            position=self._position,
            velocity=self._velocity,
            yaw=self._yaw,
            yaw_rate=self._yaw_rate,
            est_position=est_p,
            est_yaw=est_yaw,
            est_velocity=est_v,
            sensor_valid=valid,
            clearance=self._clearance,
            collided=self._collided,
            energy_j=self._energy_j,
            extras={"steps": self._steps, "accel": self._accel},
        )

    def get_state(self) -> SimState:
        if self._scenario is None:
            raise SimulatorError("get_state() before reset()")
        return self._state

    # -- introspection used by the runner ----------------------------------

    def drain_events(self) -> List[Tuple[float, str, str]]:
        """Return and clear the backend's event list."""
        events = self._events
        self._events = []
        return events

    @property
    def obstacles(self) -> List[Obstacle]:
        return list(self._obstacles)
