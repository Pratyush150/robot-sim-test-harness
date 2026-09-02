"""Shared fixtures. Everything here is offline and deterministic."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simharness.scenario import Scenario  # noqa: E402
from simharness.trace import Sample, Trace  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def scenario_dir(repo_root: Path) -> Path:
    return repo_root / "scenarios"


def minimal_scenario_dict(**overrides: Any) -> Dict[str, Any]:
    """A tiny, valid scenario document that tests can mutate."""
    data: Dict[str, Any] = {
        "name": "unit",
        "world": {"min_xy": [-5.0, -5.0], "max_xy": [15.0, 5.0], "min_z": -1.0, "max_z": 5.0},
        "robot": {
            "type": "diff_drive",
            "radius": 0.25,
            "max_speed": 1.0,
            "max_accel": 1000000.0,
            "actuator_tau": 0.000001,
            "spawn": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        },
        "goal": {"x": 10.0, "y": 0.0, "tolerance": 0.25},
        "sim": {"dt": 0.1, "time_limit": 10.0, "seed": 7},
        "assertions": [{"type": "no_collision"}],
    }
    data.update(overrides)
    return data


@pytest.fixture
def minimal_scenario() -> Scenario:
    return Scenario.from_dict(minimal_scenario_dict())


def build_trace(
    values: Dict[str, List[float]],
    *,
    dt: float = 0.1,
    scenario_name: str = "handmade",
    meta: Dict[str, Any] | None = None,
) -> Trace:
    """Build a trace from explicit per-field columns.

    Any field not supplied stays at the :class:`Sample` default, which is what
    lets a temporal-operator test say exactly what it means without simulating
    anything.
    """
    length = max(len(v) for v in values.values())
    trace = Trace(scenario=scenario_name, simulator="handmade", dt=dt, meta=meta or {})
    for i in range(length):
        kwargs = {"t": i * dt}
        for name, column in values.items():
            kwargs[name] = column[i] if i < len(column) else column[-1]
        trace.record(Sample(**kwargs))
    return trace


@pytest.fixture
def make_trace():
    return build_trace
