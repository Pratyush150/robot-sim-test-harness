"""Gazebo Sim (Garden / Harmonic) adapter, behind a guarded import.

Connection mechanism
--------------------
Gazebo Sim does not expose a Python stepping API the way AirSim does. It speaks
Gazebo Transport (protobuf over its own pub/sub + service layer). This adapter
uses the Python bindings shipped with the simulator:

* ``gz.transport13`` (Harmonic) or ``gz.transport12`` (Garden) -- the node.
* ``gz.msgs10`` / ``gz.msgs9`` -- ``WorldControl``, ``Pose``, ``EntityFactory``,
  ``Boolean``, ``Clock``.

The three calls that matter:

======================================= ==================================================
``/world/<world>/control`` (service)    ``WorldControl{pause:true, multi_step:N}``
``/world/<world>/pose/info`` (topic)    ground-truth poses of every model
``/world/<world>/create`` (service)     ``EntityFactory`` with an SDF string, to spawn
======================================= ==================================================

**The world must be started paused** (``gz sim -r`` is exactly what you do not
want) and stepped with ``multi_step``. Otherwise Gazebo runs on wall-clock and
your "fixed step" harness is racing it, which is the single most common reason a
Gazebo regression suite is not reproducible.

Determinism caveats, honestly
-----------------------------
* Set ``<physics><max_step_size>`` in the SDF and make ``dt`` an integer
  multiple of it, or ``multi_step`` will quantise your step and the trace
  timestamps will drift from the ones the harness thinks it asked for.
* DART is deterministic for a fixed step size, fixed model set and fixed build.
  Change the plugin set or the Gazebo version and traces will diverge; that is a
  baseline refresh, not a regression.
* Sensor plugins that use their own noise seeds must be given an explicit
  ``<seed>`` or the ``sensors`` system will seed from the clock.

This adapter is wiring, not a tested code path: nothing in ``tests/`` exercises
it, because Gazebo is not installed in CI. The offline suite runs against
``MockSimulator``.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

from ..scenario import Pose, Scenario
from .base import Command, SimState, Simulator, SimulatorError, SimulatorUnavailable

__all__ = ["GazeboSimulator"]

_IMPORT_ERROR: Optional[str] = None
_transport: Any = None
_msgs: Any = None

try:  # Harmonic
    from gz import msgs10 as _msgs  # type: ignore[no-redef]
    from gz import transport13 as _transport  # type: ignore[no-redef]
except ImportError:
    try:  # Garden
        from gz import msgs9 as _msgs  # type: ignore[no-redef]
        from gz import transport12 as _transport  # type: ignore[no-redef]
    except ImportError as exc:  # pragma: no cover - depends on host install
        _IMPORT_ERROR = (
            f"gz.transport Python bindings not importable ({exc}). Install Gazebo "
            "Harmonic or Garden and source its setup.sh so python-gz-transport is on "
            "PYTHONPATH."
        )


class GazeboSimulator(Simulator):
    """Steps a paused Gazebo Sim world over Gazebo Transport."""

    name = "gazebo"
    connection = "gz-transport service /world/<world>/control with multi_step; poses from /world/<world>/pose/info"

    def __init__(self, *, world: str = "default", model: str = "robot", timeout_ms: int = 2000) -> None:
        available, reason = self.availability()
        if not available:
            raise SimulatorUnavailable(reason)
        self._world = world
        self._model = model
        self._timeout_ms = timeout_ms
        self._node = _transport.Node()
        self._scenario: Optional[Scenario] = None
        self._t = 0.0
        self._pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._yaw = 0.0
        self._velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._command = Command()
        self._closed = False
        self._cmd_topic = f"/model/{model}/cmd_vel"
        self._publisher = self._node.advertise(self._cmd_topic, _msgs.Twist)

    @classmethod
    def availability(cls) -> Tuple[bool, str]:
        if _IMPORT_ERROR is not None:
            return (False, _IMPORT_ERROR)
        return (True, "gz.transport bindings importable")

    # -- lifecycle ---------------------------------------------------------

    def reset(self, scenario: Scenario) -> SimState:
        """Reset the world, then place the robot at the scenario spawn pose.

        ``WorldControl.reset.all`` restores the SDF initial conditions; the
        follow-up ``set_pose`` puts the vehicle where the scenario wants it.
        """
        self._require_open()
        self._scenario = scenario
        control = _msgs.WorldControl()
        control.pause = True
        control.reset.all = True
        self._call(f"/world/{self._world}/control", control, _msgs.Boolean)

        pose = _msgs.Pose()
        pose.name = self._model
        pose.position.x = scenario.spawn.x
        pose.position.y = scenario.spawn.y
        pose.position.z = scenario.spawn.z
        pose.orientation.z = math.sin(scenario.spawn.yaw / 2.0)
        pose.orientation.w = math.cos(scenario.spawn.yaw / 2.0)
        self._call(f"/world/{self._world}/set_pose", pose, _msgs.Boolean)

        self._t = 0.0
        self._pose = (scenario.spawn.x, scenario.spawn.y, scenario.spawn.z)
        self._yaw = scenario.spawn.yaw
        self._velocity = (0.0, 0.0, 0.0)
        return self.get_state()

    def step(self, dt: float) -> SimState:
        """Advance by ``dt`` using ``multi_step``.

        ``dt`` must be an integer multiple of the world's ``max_step_size``;
        anything else is rejected here rather than silently rounded by Gazebo.
        """
        self._require_open()
        step_size = self._max_step_size()
        ratio = dt / step_size
        steps = int(round(ratio))
        if steps < 1 or abs(ratio - steps) > 1e-9:
            raise SimulatorError(
                f"dt={dt}s is not an integer multiple of the world max_step_size "
                f"({step_size}s); Gazebo would quantise the step and the trace "
                "timestamps would drift"
            )
        control = _msgs.WorldControl()
        control.pause = True
        control.multi_step = steps
        self._call(f"/world/{self._world}/control", control, _msgs.Boolean)
        self._t += dt
        self._refresh_pose()
        return self.get_state()

    def send_command(self, command: Command) -> None:
        self._require_open()
        self._command = command
        twist = _msgs.Twist()
        twist.linear.x = command.linear[0]
        twist.linear.y = command.linear[1]
        twist.linear.z = command.linear[2]
        twist.angular.z = command.yaw_rate
        self._publisher.publish(twist)

    def spawn(self, model_id: str, pose: Pose, **kwargs: Any) -> None:
        """Insert a model via ``EntityFactory``.

        Pass ``sdf=<string>`` for an inline SDF, or ``sdf_filename=<path>``.
        """
        self._require_open()
        factory = _msgs.EntityFactory()
        factory.name = model_id
        if "sdf" in kwargs:
            factory.sdf = kwargs["sdf"]
        elif "sdf_filename" in kwargs:
            factory.sdf_filename = kwargs["sdf_filename"]
        else:
            raise SimulatorError("spawn() needs sdf= or sdf_filename= for the Gazebo backend")
        factory.pose.position.x = pose.x
        factory.pose.position.y = pose.y
        factory.pose.position.z = pose.z
        self._call(f"/world/{self._world}/create", factory, _msgs.Boolean)

    def get_state(self) -> SimState:
        return SimState(
            t=self._t,
            position=self._pose,
            velocity=self._velocity,
            yaw=self._yaw,
            est_position=self._pose,
            est_yaw=self._yaw,
            est_velocity=self._velocity,
        )

    def close(self) -> None:
        self._closed = True
        self._node = None

    # -- internals ---------------------------------------------------------

    def _require_open(self) -> None:
        if self._closed:
            raise SimulatorError("GazeboSimulator used after close()")

    def _call(self, service: str, request: Any, reply_type: Any) -> Any:
        ok, reply = self._node.request(service, request, type(request), reply_type, self._timeout_ms)
        if not ok:
            raise SimulatorError(f"gz service call to {service} timed out after {self._timeout_ms} ms")
        return reply

    def _max_step_size(self) -> float:
        """World physics step. Override by setting ``self.step_size``."""
        return float(getattr(self, "step_size", 0.001))

    def _refresh_pose(self) -> None:
        """Pull the latest ground-truth pose off ``/world/<world>/pose/info``.

        Gazebo Transport has no synchronous "get latest message" call, so real
        use subscribes once in ``__init__`` and caches into ``self._pose``. The
        hook is kept separate so a ROS 2 deployment can override it to read
        ``/model/<name>/odometry`` instead.
        """
        raise SimulatorError(
            "GazeboSimulator._refresh_pose() must be wired to your pose source "
            "(subscribe to /world/<world>/pose/info, or override this method to "
            "read a ROS 2 odometry topic)"
        )
