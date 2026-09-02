"""Trace recording, serialisation and diffing."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from simharness.trace import PRECISION, SAMPLE_FIELDS, Sample, Trace, diff_traces


def _trace(n: int = 5, offset: float = 0.0) -> Trace:
    trace = Trace(scenario="t", simulator="mock", seed=3, dt=0.1)
    for i in range(n):
        trace.record(
            Sample(
                t=i * 0.1,
                x=float(i) + offset,
                y=0.5 * i,
                vx=1.0,
                clearance=2.0 - 0.1 * i,
                energy_j=10.0 * i,
                sensor_valid=i != 2,
                collided=i >= 4,
            )
        )
    return trace


def test_record_quantises_floats():
    trace = Trace()
    trace.record(Sample(t=0.1234567890123, x=1.0 / 3.0))
    sample = trace[0]
    assert sample.t == round(0.1234567890123, PRECISION)
    assert sample.x == round(1.0 / 3.0, PRECISION)


def test_negative_zero_is_normalised():
    trace = Trace()
    trace.record(Sample(x=-0.0))
    assert math.copysign(1.0, trace[0].x) > 0


def test_trace_basics():
    trace = _trace()
    assert len(trace) == 5
    assert trace.duration == pytest.approx(0.4)
    assert trace.times() == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    assert trace.column("x") == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])
    assert trace[2].sensor_valid is False


def test_column_rejects_unknown_field():
    with pytest.raises(KeyError):
        _trace().column("altitude")


def test_path_length_sums_segment_distances():
    trace = _trace()
    # Each step moves (1.0, 0.5, 0.0): four steps of sqrt(1.25).
    assert trace.path_length() == pytest.approx(4 * math.hypot(1.0, 0.5))


def test_index_at_finds_the_last_sample_at_or_before_t():
    trace = _trace()
    assert trace.index_at(0.0) == 0
    assert trace.index_at(0.25) == 2
    assert trace.index_at(99.0) == 4


def test_events_are_recorded_and_filtered():
    trace = _trace()
    trace.add_event(0.2, "collision", "hit the pillar", obstacle="pillar_a")
    trace.add_event(0.3, "goal_reached", "arrived")
    assert len(trace.events) == 2
    assert [e.kind for e in trace.events_of("collision")] == ["collision"]
    assert trace.events_of("collision")[0].data["obstacle"] == "pillar_a"


def test_json_round_trip_preserves_every_field(tmp_path: Path):
    trace = _trace()
    trace.add_event(0.2, "note", "hello")
    path = trace.save_json(tmp_path / "t.json")
    loaded = Trace.load_json(path)
    assert loaded.scenario == trace.scenario
    assert loaded.seed == trace.seed
    assert len(loaded) == len(trace)
    for a, b in zip(trace.samples, loaded.samples):
        for field in SAMPLE_FIELDS:
            assert getattr(a, field) == getattr(b, field)
    assert loaded.events[0].detail == "hello"


def test_json_is_a_valid_document_with_a_format_tag():
    data = json.loads(_trace().to_json())
    assert data["format"] == "simharness-trace/1"
    assert data["fields"] == list(SAMPLE_FIELDS)
    assert len(data["samples"]) == 5


def test_csv_round_trip(tmp_path: Path):
    trace = _trace()
    path = trace.save_csv(tmp_path / "t.csv")
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(SAMPLE_FIELDS)
    assert rows[2]["sensor_valid"] == "0"
    loaded = Trace.load_csv(path)
    assert len(loaded) == 5
    assert loaded.column("x") == pytest.approx(trace.column("x"))
    assert loaded.dt == pytest.approx(0.1)


def test_infinite_clearance_survives_json_round_trip(tmp_path: Path):
    trace = Trace(dt=0.1)
    trace.record(Sample(t=0.0, clearance=float("inf")))
    loaded = Trace.load_json(trace.save_json(tmp_path / "inf.json"))
    assert math.isinf(loaded[0].clearance)


# -- diffing ---------------------------------------------------------------


def test_identical_traces_diff_clean():
    result = diff_traces(_trace(), _trace())
    assert result.identical is True
    assert result.first_divergent_index is None
    assert result.compared_steps == 5
    assert "identical" in result.summary()


def test_diff_reports_the_first_divergent_step():
    a = _trace(8)
    b = _trace(8)
    b.samples[3].x += 0.25
    b.samples[6].x += 5.0
    result = diff_traces(a, b)
    assert result.identical is False
    assert result.first_divergent_index == 3
    assert result.first_divergent_time == pytest.approx(0.3)
    assert result.first_divergent_field == "x"
    assert result.first_divergent_values == pytest.approx((3.0, 3.25))
    # The scan continues past the first divergence, so the worst is the later one.
    assert result.max_abs_diff["x"] == pytest.approx(5.0)
    assert "step 3" in result.summary()


def test_diff_tolerance_suppresses_small_differences():
    a = _trace(6)
    b = _trace(6)
    b.samples[2].x += 1e-4
    assert diff_traces(a, b, tolerance=0.0).identical is False
    assert diff_traces(a, b, tolerance=1e-3).identical is True


def test_diff_can_be_restricted_to_a_field_subset():
    a = _trace(6)
    b = _trace(6)
    b.samples[2].y += 1.0
    assert diff_traces(a, b, fields=["x"]).identical is True
    assert diff_traces(a, b, fields=["y"]).identical is False


def test_diff_rejects_unknown_fields():
    with pytest.raises(KeyError):
        diff_traces(_trace(), _trace(), fields=["nope"])


def test_diff_reports_a_length_mismatch():
    result = diff_traces(_trace(5), _trace(8))
    assert result.identical is False
    assert result.length_a == 5
    assert result.length_b == 8
    assert "length differs" in result.note


def test_diff_worst_field_picks_the_largest_deviation():
    a = _trace(6)
    b = _trace(6)
    b.samples[1].x += 0.1
    b.samples[1].y += 3.0
    assert diff_traces(a, b).worst_field == "y"


def test_diff_treats_matching_infinities_as_equal():
    a = Trace(dt=0.1)
    b = Trace(dt=0.1)
    for trace in (a, b):
        trace.record(Sample(t=0.0, clearance=float("inf")))
    assert diff_traces(a, b).identical is True


def test_diff_to_dict_is_json_serialisable():
    payload = json.dumps(diff_traces(_trace(5), _trace(6)).to_dict())
    assert "first_divergent_index" in payload
