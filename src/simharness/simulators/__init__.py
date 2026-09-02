"""Simulator backends and the registry that picks one.

Selection policy, and why it is what it is
------------------------------------------
The default backend is ``mock``, not "whatever is installed". A regression
suite that silently changes physics engine because someone installed Gazebo on
the build agent is not a regression suite. You opt in to a native backend
explicitly::

    SIMHARNESS_SIMULATOR=gazebo simharness suite scenarios/
    simharness suite scenarios/ --simulator auto     # walk the preference list

Every selection decision is logged at INFO, and every rejection at DEBUG with
the exact reason (missing package, unreachable RPC port, wrong interpreter), so
"why did it use the mock" is always answerable from the log.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from .base import Command, SimState, Simulator, SimulatorError, SimulatorUnavailable
from .mock import MockSimulator

__all__ = [
    "Command",
    "SimState",
    "Simulator",
    "SimulatorError",
    "SimulatorUnavailable",
    "MockSimulator",
    "register",
    "registered_names",
    "get_simulator_class",
    "availability_report",
    "select_simulator",
    "create_simulator",
    "NATIVE_PREFERENCE",
    "ENV_VAR",
]

LOG = logging.getLogger("simharness.simulators")

ENV_VAR = "SIMHARNESS_SIMULATOR"
NATIVE_PREFERENCE: Tuple[str, ...] = ("gazebo", "isaac", "airsim", "mock")

_LOADERS: Dict[str, Callable[[], Type[Simulator]]] = {}


def register(name: str, loader: Callable[[], Type[Simulator]]) -> None:
    """Register a backend under ``name``.

    ``loader`` is called lazily and may raise ``ImportError``; that is how a
    backend whose Python bindings are missing stays out of the way instead of
    breaking ``import simharness``.
    """
    _LOADERS[name] = loader


def _load_gazebo() -> Type[Simulator]:
    from .gazebo import GazeboSimulator

    return GazeboSimulator


def _load_airsim() -> Type[Simulator]:
    from .airsim import AirSimSimulator

    return AirSimSimulator


def _load_isaac() -> Type[Simulator]:
    from .isaac import IsaacSimulator

    return IsaacSimulator


register("mock", lambda: MockSimulator)
register("gazebo", _load_gazebo)
register("airsim", _load_airsim)
register("isaac", _load_isaac)


def registered_names() -> List[str]:
    """Names of every registered backend, in registration order."""
    return list(_LOADERS)


def get_simulator_class(name: str) -> Type[Simulator]:
    """Import and return a backend class by name."""
    if name not in _LOADERS:
        raise SimulatorError(
            f"unknown simulator '{name}'; registered: {', '.join(registered_names())}"
        )
    try:
        return _LOADERS[name]()
    except ImportError as exc:
        raise SimulatorUnavailable(f"backend '{name}' could not be imported: {exc}") from exc


def availability_report() -> List[Tuple[str, bool, str]]:
    """``(name, available, reason)`` for every registered backend.

    Never raises. A backend that explodes on import is reported as unavailable
    with the exception text as its reason.
    """
    report: List[Tuple[str, bool, str]] = []
    for name in registered_names():
        try:
            cls = get_simulator_class(name)
            available, reason = cls.availability()
        except Exception as exc:  # defensive: a broken backend must not break the report
            available, reason = False, f"{type(exc).__name__}: {exc}"
        report.append((name, available, reason))
    return report


def select_simulator(preferred: Optional[str] = None) -> Tuple[Type[Simulator], str]:
    """Choose a backend class and log why.

    ``preferred`` may be a backend name, ``"auto"`` to walk
    :data:`NATIVE_PREFERENCE`, or ``None`` to read ``SIMHARNESS_SIMULATOR``
    and otherwise default to ``mock``.
    """
    if preferred is None:
        preferred = os.environ.get(ENV_VAR, "mock")

    if preferred != "auto":
        cls = get_simulator_class(preferred)
        available, reason = cls.availability()
        if not available:
            raise SimulatorUnavailable(f"simulator '{preferred}' was requested but is unavailable: {reason}")
        LOG.info("simulator '%s' selected explicitly (%s)", preferred, reason)
        return cls, preferred

    for name in NATIVE_PREFERENCE:
        try:
            cls = get_simulator_class(name)
            available, reason = cls.availability()
        except Exception as exc:
            LOG.debug("simulator '%s' skipped: %s", name, exc)
            continue
        if available:
            LOG.info("simulator '%s' selected by auto-detection (%s)", name, reason)
            return cls, name
        LOG.debug("simulator '%s' skipped: %s", name, reason)
    raise SimulatorUnavailable("no simulator backend is available, not even the built-in mock")


def create_simulator(preferred: Optional[str] = None, **kwargs: Any) -> Simulator:
    """Select a backend and instantiate it."""
    cls, _name = select_simulator(preferred)
    return cls(**kwargs)
