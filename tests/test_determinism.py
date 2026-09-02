"""Determinism: the property the whole harness rests on.

If the same seed does not produce the same trace, every other guarantee here
is decoration. These tests assert byte-identical JSON, not "close enough".
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from simharness.runner import run_scenario, run_suite
from simharness.scenario import load_scenario, load_scenario_dir
from simharness.trace import diff_traces


def _digest(trace) -> str:
    return hashlib.sha256(trace.to_json().encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def noisy_scenario(request):
    root = Path(request.config.rootdir)
    return load_scenario(root / "scenarios" / "sensor_dropout.yaml")


def test_same_seed_gives_a_byte_identical_trace(noisy_scenario):
    a = run_scenario(noisy_scenario).trace
    b = run_scenario(noisy_scenario).trace
    assert a.to_json() == b.to_json()
    assert _digest(a) == _digest(b)
    assert diff_traces(a, b).identical is True


def test_a_different_seed_gives_a_different_trace(noisy_scenario):
    a = run_scenario(noisy_scenario).trace
    b = run_scenario(noisy_scenario.with_overrides({"sim.seed": noisy_scenario.sim.seed + 1})).trace
    assert a.to_json() != b.to_json()
    difference = diff_traces(a, b)
    assert difference.identical is False
    assert difference.first_divergent_index is not None


def test_determinism_holds_for_every_shipped_scenario(scenario_dir: Path):
    for scenario in load_scenario_dir(scenario_dir):
        first = run_scenario(scenario).trace
        second = run_scenario(scenario).trace
        assert first.to_json() == second.to_json(), f"'{scenario.name}' is not reproducible"


def test_metrics_are_identical_across_repeat_runs(noisy_scenario):
    a = run_scenario(noisy_scenario)
    b = run_scenario(noisy_scenario)
    assert a.metrics == b.metrics
    assert [x.message for x in a.assertions] == [x.message for x in b.assertions]


def test_parallel_execution_matches_serial_execution(scenario_dir: Path):
    scenarios = load_scenario_dir(scenario_dir)
    serial = run_suite(scenarios, workers=1, keep_traces=True)
    parallel = run_suite(scenarios, workers=4, keep_traces=True)
    assert [r.scenario for r in serial] == [r.scenario for r in parallel], "result order must not depend on workers"
    for a, b in zip(serial, parallel):
        assert a.status == b.status
        assert a.metrics == b.metrics
        assert a.trace.to_json() == b.trace.to_json()


def test_noise_streams_are_independent_of_the_step_count(noisy_scenario):
    """Shortening the run must not change the samples that came before.

    A shared RNG consumed in a different order is the classic way a harness
    stops being reproducible, so this pins the ordering.
    """
    full = run_scenario(noisy_scenario).trace
    short = run_scenario(noisy_scenario.with_overrides({"sim.time_limit": 5.0, "goal.within": 4.0})).trace
    assert len(short) < len(full)
    for i in range(len(short)):
        assert full.samples[i].to_row() == short.samples[i].to_row()


def test_seed_is_recorded_in_the_trace(noisy_scenario):
    trace = run_scenario(noisy_scenario).trace
    assert trace.seed == noisy_scenario.sim.seed
    assert trace.dt == pytest.approx(noisy_scenario.sim.dt)
