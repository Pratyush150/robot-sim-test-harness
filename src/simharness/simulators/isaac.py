"""NVIDIA Isaac Sim adapter, behind a guarded import.

Connection mechanism
--------------------
Isaac Sim is not a server you connect to. Your process *is* the simulator: the
Omniverse Kit runtime is loaded into the Python interpreter, which is why this
script must be launched with Isaac's own interpreter::

    ~/.local/share/ov/pkg/isaac-sim-*/python.sh -m simharness.cli suite scenarios/

The strict rule that trips everyone up: ``SimulationApp`` must be constructed
**before any other omni import**, because constructing it is what loads the
extensions those modules live in::

    from isaacsim import SimulationApp          # 4.x
    app = SimulationApp({"headless": True})     # nothing omni.* above this line
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage

Stepping is genuinely fixed-step and deterministic::

    world = World(physics_dt=1/240, rendering_dt=1/60, stage_units_in_meters=1.0)
    world.reset()
    world.step(render=False)     # advances exactly physics_dt

Determinism caveats, honestly
-----------------------------
* PhysX is deterministic for a fixed ``physics_dt``, a fixed scene and a fixed
  GPU/driver pair. Change the driver and contact-rich scenes can diverge. For
  regression testing prefer scenarios that are not contact-dominated, or accept
  a tolerance rather than exact equality.
* ``world.step(render=True)`` couples physics to the render loop. Use
  ``render=False`` in CI.
* GPU dynamics (``enable_gpu_dynamics``) changes results versus CPU. Pin it.
* Isaac Sim is a large licensed download; it does not belong in a CI container
  unless you have a GPU runner. This is exactly why the harness abstracts it.

This adapter is wiring, not a tested code path: the offline suite runs against
``MockSimulator``.
"""

from __future__ import annotations

import importlib.util
import math
import os
from typing import Any, Optional, Tuple

from ..scenario import Pose, Scenario
from .base import Command, SimState, Simulator, SimulatorError, SimulatorUnavailable

__all__ = ["IsaacSimulator"]


def _isaac_present() -> Tuple[bool, str]:
    for module in ("isaacsim", "omni.isaac.kit"):
        try:
            if importlib.util.find_spec(module) is not None:
                return (True, f"'{module}' is importable in this interpreter")
        except (ImportError, ValueError):
            continue
    hint = os.environ.get("ISAAC_SIM_PATH")
    if hint:
        return (
            False,
            f"ISAAC_SIM_PATH is set to {hint} but Kit is not importable here; "
            "launch through that install's python.sh",
        )
    return (
        False,
        "Omniverse Kit is not importable. Isaac Sim only works under its own "
        "python.sh interpreter; a plain 'python3' can never load it.",
    )


class IsaacSimulator(Simulator):
    """Drives an Isaac Sim ``World`` in-process, headless, at a fixed physics dt."""

    name = "isaac"
    connection = "in-process Omniverse Kit (SimulationApp) driven by World.step(render=False)"

    def __init__(self, *, physics_dt: float = 1.0 / 240.0, headless: bool = True, prim_path: str = "/World/Robot") -> None:
        available, reason = self.availability()
        if not available:
            raise SimulatorUnavailable(reason)
        self._physics_dt = physics_dt
        self._prim_path = prim_path
        self._closed = False
        self._t = 0.0
        self._scenario: Optional[Scenario] = None
        self._command = Command()

        from isaacsim import SimulationApp  # type: ignore  # noqa: PLC0415

        self._app = SimulationApp({"headless": headless})
        from omni.isaac.core import World  # type: ignore  # noqa: PLC0415

        self._world = World(physics_dt=physics_dt, rendering_dt=physics_dt, stage_units_in_meters=1.0)
        self._robot: Any = None

    @classmethod
    def availability(cls) -> Tuple[bool, str]:
        return _isaac_present()

    # -- lifecycle ---------------------------------------------------------

    def reset(self, scenario: Scenario) -> SimState:
        self._require_open()
        self._scenario = scenario
        self._world.reset()
        self._t = 0.0
        if self._robot is None:
            from omni.isaac.core.utils.prims import get_prim_at_path  # type: ignore  # noqa: PLC0415
            from omni.isaac.core.prims import XFormPrim  # type: ignore  # noqa: PLC0415

            if get_prim_at_path(self._prim_path) is None:
                raise SimulatorError(
                    f"no prim at {self._prim_path}; add your robot USD to the stage before reset()"
                )
            self._robot = XFormPrim(self._prim_path)
        spawn = scenario.spawn
        self._robot.set_world_pose(
            position=(spawn.x, spawn.y, spawn.z),
            orientation=(math.cos(spawn.yaw / 2.0), 0.0, 0.0, math.sin(spawn.yaw / 2.0)),
        )
        return self.get_state()

    def step(self, dt: float) -> SimState:
        """Advance by ``dt``, which must be a whole number of physics steps."""
        self._require_open()
        ratio = dt / self._physics_dt
        substeps = int(round(ratio))
        if substeps < 1 or abs(ratio - substeps) > 1e-9:
            raise SimulatorError(
                f"dt={dt}s is not an integer multiple of physics_dt={self._physics_dt}s; "
                "Isaac would not land on your requested timestamps"
            )
        for _ in range(substeps):
            self._world.step(render=False)
        self._t += dt
        return self.get_state()

    def send_command(self, command: Command) -> None:
        """Apply a world-frame velocity setpoint to the articulation.

        Wire this to your own controller: a wheeled base takes wheel velocities
        via ``ArticulationController``, a drone takes rotor thrusts or a
        velocity controller. There is no single correct mapping, which is why
        it is left explicit.
        """
        self._require_open()
        self._command = command
        if self._robot is None:
            raise SimulatorError("send_command() before reset()")
        setter = getattr(self._robot, "set_linear_velocity", None)
        if setter is None:
            raise SimulatorError(
                f"prim {self._prim_path} is not a RigidPrim; override send_command() "
                "to drive your articulation"
            )
        setter(command.linear)

    def spawn(self, model_id: str, pose: Pose, **kwargs: Any) -> None:
        """Add a USD reference to the stage. Pass ``usd_path=<path or asset URL>``."""
        self._require_open()
        usd_path = kwargs.pop("usd_path", None)
        if usd_path is None:
            raise SimulatorError("spawn() needs usd_path= for the Isaac backend")
        from omni.isaac.core.prims import XFormPrim  # type: ignore  # noqa: PLC0415
        from omni.isaac.core.utils.stage import add_reference_to_stage  # type: ignore  # noqa: PLC0415

        prim_path = f"/World/{model_id}"
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        XFormPrim(prim_path).set_world_pose(position=(pose.x, pose.y, pose.z))

    def get_state(self) -> SimState:
        self._require_open()
        if self._robot is None:
            return SimState(t=self._t)
        position, orientation = self._robot.get_world_pose()
        velocity = getattr(self._robot, "get_linear_velocity", lambda: (0.0, 0.0, 0.0))()
        w, _, _, z = (float(v) for v in orientation)
        yaw = 2.0 * math.atan2(z, w)
        pos = (float(position[0]), float(position[1]), float(position[2]))
        vel = (float(velocity[0]), float(velocity[1]), float(velocity[2]))
        return SimState(
            t=self._t,
            position=pos,
            velocity=vel,
            yaw=yaw,
            est_position=pos,
            est_yaw=yaw,
            est_velocity=vel,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:  # pragma: no cover - requires Kit
            self._world.stop()
            self._app.close()
        except Exception:
            pass

    def _require_open(self) -> None:
        if self._closed:
            raise SimulatorError("IsaacSimulator used after close()")
