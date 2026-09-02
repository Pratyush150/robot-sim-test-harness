"""simharness -- scenario-driven regression testing for robots in simulation.

Anyone can launch a simulator. The hard part is asserting that the robot still
*behaves* the same way after you change a gain, a planner or a firmware
version. This package is that missing half: a declarative scenario format,
behavioural and temporal assertions over recorded runs, deterministic seeded
execution, parameter sweeps that find the failure boundary, and reports that
drop straight into CI.

Typical use::

    from simharness import load_scenario, run_scenario

    result = run_scenario(load_scenario("scenarios/obstacle_slalom.yaml"))
    print(result.status, result.metrics)
    for assertion in result.assertions:
        print(assertion)
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Pratyush Vatsa"
__license__ = "MIT"

from .assertions import AssertionResult, evaluate_all, summarise
from .scenario import Scenario, ScenarioError, load_scenario, load_scenario_dir
from .simulators import MockSimulator, Simulator, create_simulator, availability_report
from .trace import Sample, Trace, TraceDiff, diff_traces

__all__ = [
    "__version__",
    "AssertionResult",
    "MockSimulator",
    "Sample",
    "Scenario",
    "ScenarioError",
    "Simulator",
    "Trace",
    "TraceDiff",
    "availability_report",
    "create_simulator",
    "diff_traces",
    "evaluate_all",
    "load_scenario",
    "load_scenario_dir",
    "run_scenario",
    "run_suite",
    "summarise",
]


def __getattr__(name: str):
    """Lazily expose the runner so ``import simharness`` stays cheap."""
    if name in ("run_scenario", "run_suite", "RunResult", "SuiteResult"):
        from . import runner

        return getattr(runner, name)
    raise AttributeError(f"module 'simharness' has no attribute '{name}'")
