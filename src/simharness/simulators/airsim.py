"""AirSim / Colosseum adapter, behind a guarded import.

Connection mechanism
--------------------
AirSim exposes an msgpack-RPC server inside the Unreal process, default
``127.0.0.1:41451``. The ``airsim`` Python package is a thin client over it::

    client = airsim.MultirotorClient(ip="127.0.0.1", port=41451)
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

Deterministic stepping is the important part and it is easy to get wrong::

    client.simPause(True)                 # take the sim off wall-clock
    client.simContinueForTime(dt)         # advance exactly dt seconds
    # NOT: time.sleep(dt) while unpaused

State and collisions::

    state = client.getMultirotorState()   # kinematics_estimated (EKF output)
    truth = client.simGetGroundTruthKinematics()
    hit   = client.simGetCollisionInfo()  # .has_collided, .time_stamp, .penetration_depth

Setup that bites people
-----------------------
* ``settings.json`` lives in ``~/Documents/AirSim/`` (Windows) or
  ``~/Documents/AirSim/`` created by the binary on Linux. ``"ClockType":
  "SteppableClock"`` is required for ``simContinueForTime`` to mean anything.
* ``"ClockSpeed"`` is not a stepping mechanism. It rescales wall-clock. If you
  are relying on it for repeatability you do not have repeatability.
* Wind is set globally with ``client.simSetWind(airsim.Vector3r(x, y, z))``.
* The upstream ``airsim`` pip package is unmaintained and pinned to old
  msgpack/numpy. Colosseum is the maintained fork; the RPC surface used here is
  identical.

This adapter is wiring, not a tested code path: the offline suite runs against
``MockSimulator``.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

from ..scenario import Pose, Scenario
from .base import Command, SimState, Simulator, SimulatorError, SimulatorUnavailable

__all__ = ["AirSimSimulator"]

_IMPORT_ERROR: Optional[str] = None
try:
    import airsim as _airsim  # type: ignore
except ImportError as exc:  # pragma: no cover - depends on host install
    _airsim = None  # type: ignore[assignment]
    _IMPORT_ERROR = (
        f"the 'airsim' Python package is not importable ({exc}). "
        "Install 'airsim' or 'colosseum' and start the Unreal binary first."
    )


class AirSimSimulator(Simulator):
    """Drives a paused AirSim multirotor over msgpack-RPC."""

    name = "airsim"
    connection = "msgpack-RPC to 127.0.0.1:41451; simPause + simContinueForTime for fixed stepping"

    def __init__(
        self,
        *,
        ip: str = "127.0.0.1",
        port: int = 41451,
        vehicle: str = "",
        use_ground_truth: bool = True,
    ) -> None:
        available, reason = self.availability()
        if not available:
            raise SimulatorUnavailable(reason)
        self._vehicle = vehicle
        self._use_ground_truth = use_ground_truth
        self._client = _airsim.MultirotorClient(ip=ip, port=port)
        try:
            self._client.confirmConnection()
        except Exception as exc:  # pragma: no cover - network dependent
            raise SimulatorUnavailable(
                f"AirSim RPC at {ip}:{port} did not answer ({exc}). Is the Unreal binary running?"
            ) from exc
        self._t = 0.0
        self._closed = False
        self._scenario: Optional[Scenario] = None
        self._collided = False

    @classmethod
    def availability(cls) -> Tuple[bool, str]:
        if _IMPORT_ERROR is not None:
            return (False, _IMPORT_ERROR)
        return (True, "airsim client library importable")

    # -- lifecycle ---------------------------------------------------------

    def reset(self, scenario: Scenario) -> SimState:
        self._require_open()
        self._scenario = scenario
        client = self._client
        client.reset()
        client.enableApiControl(True, self._vehicle)
        client.armDisarm(True, self._vehicle)
        client.simPause(True)

        spawn = scenario.spawn
        # AirSim is NED with z down; scenario altitudes are ENU with z up.
        pose = _airsim.Pose(
            _airsim.Vector3r(spawn.x, spawn.y, -spawn.z),
            _airsim.to_quaternion(0.0, 0.0, spawn.yaw),
        )
        client.simSetVehiclePose(pose, True, self._vehicle)
        wind = scenario.disturbance.wind
        client.simSetWind(_airsim.Vector3r(wind[0], wind[1], -wind[2]))
        self._t = 0.0
        self._collided = False
        return self.get_state()

    def step(self, dt: float) -> SimState:
        self._require_open()
        self._client.simContinueForTime(dt)
        self._t += dt
        return self.get_state()

    def send_command(self, command: Command) -> None:
        """Send a world-frame velocity setpoint (converted ENU -> NED)."""
        self._require_open()
        vx, vy, vz = command.linear
        yaw_mode = _airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(command.yaw_rate))
        self._client.moveByVelocityAsync(
            vx, vy, -vz, duration=1.0, yaw_mode=yaw_mode, vehicle_name=self._vehicle
        )

    def spawn(self, model_id: str, pose: Pose, **kwargs: Any) -> None:
        """Spawn a static mesh. ``asset`` names an Unreal asset in the level."""
        self._require_open()
        asset = kwargs.pop("asset", None)
        if asset is None:
            raise SimulatorError("spawn() needs asset=<unreal asset name> for the AirSim backend")
        scale = float(kwargs.pop("scale", 1.0))
        self._client.simSpawnObject(
            model_id,
            asset,
            _airsim.Pose(_airsim.Vector3r(pose.x, pose.y, -pose.z), _airsim.to_quaternion(0, 0, pose.yaw)),
            _airsim.Vector3r(scale, scale, scale),
        )

    def get_state(self) -> SimState:
        self._require_open()
        client = self._client
        estimated = client.getMultirotorState(self._vehicle).kinematics_estimated
        truth = client.simGetGroundTruthKinematics(self._vehicle) if self._use_ground_truth else estimated
        collision = client.simGetCollisionInfo(self._vehicle)
        if collision.has_collided:
            self._collided = True

        def _enu(vec: Any) -> Tuple[float, float, float]:
            return (float(vec.x_val), float(vec.y_val), -float(vec.z_val))

        _, _, yaw = _airsim.to_eularian_angles(truth.orientation)
        _, _, est_yaw = _airsim.to_eularian_angles(estimated.orientation)
        return SimState(
            t=self._t,
            position=_enu(truth.position),
            velocity=_enu(truth.linear_velocity),
            yaw=float(yaw),
            yaw_rate=float(truth.angular_velocity.z_val),
            est_position=_enu(estimated.position),
            est_yaw=float(est_yaw),
            est_velocity=_enu(estimated.linear_velocity),
            collided=self._collided,
            extras={"penetration_depth": float(getattr(collision, "penetration_depth", 0.0))},
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:  # pragma: no cover - network dependent
            self._client.simPause(False)
            self._client.armDisarm(False, self._vehicle)
            self._client.enableApiControl(False, self._vehicle)
        except Exception:
            pass

    def _require_open(self) -> None:
        if self._closed:
            raise SimulatorError("AirSimSimulator used after close()")
