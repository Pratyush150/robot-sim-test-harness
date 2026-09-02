"""The run record: what actually happened, timestep by timestep.

A :class:`Trace` is the only thing assertions, reports and diffs ever look at.
The simulator produces one; everything downstream consumes one. That separation
is what lets the same assertion suite run against a mock simulator today and a
Gazebo bridge tomorrow.

Floats are quantised to :data:`PRECISION` decimal places the moment they are
recorded. That is what makes "same seed, byte-identical trace" a property you
can actually assert on rather than a hope about floating-point formatting.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "PRECISION",
    "SAMPLE_FIELDS",
    "Sample",
    "Event",
    "Trace",
    "TraceDiff",
    "diff_traces",
]

PRECISION = 9
"""Decimal places every recorded float is rounded to."""

SAMPLE_FIELDS: Tuple[str, ...] = (
    "t",
    "x",
    "y",
    "z",
    "yaw",
    "vx",
    "vy",
    "vz",
    "yaw_rate",
    "cmd_x",
    "cmd_y",
    "cmd_z",
    "cmd_yaw_rate",
    "est_x",
    "est_y",
    "est_z",
    "est_yaw",
    "clearance",
    "energy_j",
    "sensor_valid",
    "collided",
)

_BOOL_FIELDS = ("sensor_valid", "collided")

#: JSON has no literal for infinity that every parser accepts, and CSV has
#: none at all, so an unbounded clearance is written as this sentinel and
#: turned back into ``inf`` on load.
_INF_SENTINEL = 1e308


def q(value: float) -> float:
    """Quantise a float for recording. ``-0.0`` is normalised to ``0.0``."""
    rounded = round(float(value), PRECISION)
    return 0.0 if rounded == 0.0 else rounded


@dataclass
class Sample:
    """One simulation timestep.

    ``x/y/z/yaw`` are ground truth. ``est_*`` is what the controller was shown
    after sensor noise, bias and dropout. When the two diverge you are looking
    at an estimator problem, not a controller problem.
    """

    t: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0
    cmd_x: float = 0.0
    cmd_y: float = 0.0
    cmd_z: float = 0.0
    cmd_yaw_rate: float = 0.0
    est_x: float = 0.0
    est_y: float = 0.0
    est_z: float = 0.0
    est_yaw: float = 0.0
    clearance: float = float("inf")
    energy_j: float = 0.0
    sensor_valid: bool = True
    collided: bool = False

    def quantised(self) -> "Sample":
        """Return a copy with every float rounded to :data:`PRECISION`."""
        values = {}
        for name in SAMPLE_FIELDS:
            value = getattr(self, name)
            if name in _BOOL_FIELDS:
                values[name] = bool(value)
            elif isinstance(value, float) and math.isinf(value):
                values[name] = value
            else:
                values[name] = q(value)
        return Sample(**values)

    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def velocity(self) -> Tuple[float, float, float]:
        return (self.vx, self.vy, self.vz)

    def speed(self) -> float:
        return math.sqrt(self.vx * self.vx + self.vy * self.vy + self.vz * self.vz)

    def to_row(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in SAMPLE_FIELDS}


@dataclass
class Event:
    """A discrete thing that happened, e.g. a collision or a sensor dropout."""

    t: float
    kind: str
    detail: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"t": q(self.t), "kind": self.kind, "detail": self.detail}
        if self.data:
            out["data"] = dict(self.data)
        return out


@dataclass
class Trace:
    """A whole run: metadata, samples and events."""

    scenario: str = ""
    simulator: str = ""
    seed: int = 0
    dt: float = 0.0
    samples: List[Sample] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- building ----------------------------------------------------------

    def record(self, sample: Sample) -> None:
        """Append a sample, quantising it first."""
        self.samples.append(sample.quantised())

    def add_event(self, t: float, kind: str, detail: str = "", **data: Any) -> None:
        self.events.append(Event(t=q(t), kind=kind, detail=detail, data=data))

    # -- reading -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]

    @property
    def duration(self) -> float:
        """Simulated time covered by the trace, in seconds."""
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1].t - self.samples[0].t

    def times(self) -> List[float]:
        return [s.t for s in self.samples]

    def column(self, name: str) -> List[float]:
        """Extract one field as a list. Booleans come back as 0.0/1.0."""
        if name not in SAMPLE_FIELDS:
            raise KeyError(f"unknown trace field '{name}'; known fields: {', '.join(SAMPLE_FIELDS)}")
        return [float(getattr(s, name)) for s in self.samples]

    def events_of(self, kind: str) -> List[Event]:
        return [e for e in self.events if e.kind == kind]

    def index_at(self, t: float) -> int:
        """Index of the last sample at or before time ``t`` (0 if the trace is empty)."""
        best = 0
        for i, sample in enumerate(self.samples):
            if sample.t <= t + 1e-12:
                best = i
            else:
                break
        return best

    def path_length(self) -> float:
        """Total 3D distance travelled along the recorded path."""
        total = 0.0
        for a, b in zip(self.samples, self.samples[1:]):
            total += math.dist(a.position(), b.position())
        return total

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": "simharness-trace/1",
            "scenario": self.scenario,
            "simulator": self.simulator,
            "seed": self.seed,
            "dt": q(self.dt),
            "fields": list(SAMPLE_FIELDS),
            "meta": self.meta,
            "events": [e.to_dict() for e in self.events],
            "samples": [_encode_row(s) for s in self.samples],
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        """Serialise to JSON. Deterministic for a deterministic run."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False, allow_nan=True)

    def save_json(self, path: os.PathLike | str, *, indent: Optional[int] = 2) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=indent), encoding="utf-8")
        return target

    def save_csv(self, path: os.PathLike | str) -> Path:
        """Write the samples as CSV. Events are dropped; use JSON to keep them."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(SAMPLE_FIELDS))
            writer.writeheader()
            for sample in self.samples:
                row = _encode_row(sample)
                writer.writerow({name: _csv_value(row[i]) for i, name in enumerate(SAMPLE_FIELDS)})
        return target

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "Trace":
        fields = list(data.get("fields", SAMPLE_FIELDS))
        trace = Trace(
            scenario=str(data.get("scenario", "")),
            simulator=str(data.get("simulator", "")),
            seed=int(data.get("seed", 0)),
            dt=float(data.get("dt", 0.0)),
            meta=dict(data.get("meta", {})),
        )
        for row in data.get("samples", []):
            values: Dict[str, Any] = {}
            for name, value in zip(fields, row):
                if name in _BOOL_FIELDS:
                    values[name] = bool(value)
                else:
                    values[name] = _decode_float(value)
            trace.samples.append(Sample(**values))
        for item in data.get("events", []):
            trace.events.append(
                Event(
                    t=float(item.get("t", 0.0)),
                    kind=str(item.get("kind", "")),
                    detail=str(item.get("detail", "")),
                    data=dict(item.get("data", {})),
                )
            )
        return trace

    @staticmethod
    def load_json(path: os.PathLike | str) -> "Trace":
        return Trace.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @staticmethod
    def load_csv(path: os.PathLike | str) -> "Trace":
        trace = Trace()
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                values: Dict[str, Any] = {}
                for name in SAMPLE_FIELDS:
                    raw = row.get(name, "")
                    if name in _BOOL_FIELDS:
                        values[name] = raw.strip().lower() in ("1", "true", "yes")
                    else:
                        values[name] = _decode_float(raw)
                trace.samples.append(Sample(**values))
        if len(trace.samples) >= 2:
            trace.dt = q(trace.samples[1].t - trace.samples[0].t)
        return trace


def _decode_float(value: Any) -> float:
    """Undo :data:`_INF_SENTINEL` so a reloaded trace plots and diffs correctly."""
    number = float(value)
    if number >= _INF_SENTINEL:
        return float("inf")
    if number <= -_INF_SENTINEL:
        return float("-inf")
    return number


def _encode_row(sample: Sample) -> List[Any]:
    row: List[Any] = []
    for name in SAMPLE_FIELDS:
        value = getattr(sample, name)
        if name in _BOOL_FIELDS:
            row.append(bool(value))
        elif isinstance(value, float) and math.isinf(value):
            row.append(_INF_SENTINEL if value > 0 else -_INF_SENTINEL)
        else:
            row.append(q(value))
    return row


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    return value


# --------------------------------------------------------------------------
# diffing
# --------------------------------------------------------------------------


@dataclass
class TraceDiff:
    """The result of comparing two traces.

    ``first_divergent_index`` is the earliest timestep at which any compared
    field differs by more than the tolerance. That number is the whole point:
    when a controller change breaks a scenario, you want the step where the two
    runs stopped agreeing, not a 4000-line side-by-side.
    """

    identical: bool
    first_divergent_index: Optional[int] = None
    first_divergent_time: Optional[float] = None
    first_divergent_field: Optional[str] = None
    first_divergent_values: Optional[Tuple[float, float]] = None
    length_a: int = 0
    length_b: int = 0
    compared_steps: int = 0
    max_abs_diff: Dict[str, float] = field(default_factory=dict)
    tolerance: float = 0.0
    note: str = ""

    @property
    def worst_field(self) -> Optional[str]:
        """The field with the largest absolute difference across the run."""
        if not self.max_abs_diff:
            return None
        return max(self.max_abs_diff.items(), key=lambda kv: kv[1])[0]

    def summary(self) -> str:
        if self.identical:
            return f"traces identical over {self.compared_steps} steps (tolerance {self.tolerance:g})"
        if self.first_divergent_index is None:
            return f"traces differ: {self.note}"
        assert self.first_divergent_values is not None
        a, b = self.first_divergent_values
        return (
            f"first divergence at step {self.first_divergent_index} "
            f"(t={self.first_divergent_time:.6g}s) in '{self.first_divergent_field}': "
            f"{a:.9g} vs {b:.9g}"
            + (f"; {self.note}" if self.note else "")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identical": self.identical,
            "first_divergent_index": self.first_divergent_index,
            "first_divergent_time": self.first_divergent_time,
            "first_divergent_field": self.first_divergent_field,
            "first_divergent_values": list(self.first_divergent_values) if self.first_divergent_values else None,
            "length_a": self.length_a,
            "length_b": self.length_b,
            "compared_steps": self.compared_steps,
            "max_abs_diff": self.max_abs_diff,
            "tolerance": self.tolerance,
            "note": self.note,
            "summary": self.summary(),
        }


def diff_traces(
    a: Trace,
    b: Trace,
    *,
    tolerance: float = 0.0,
    fields: Optional[Sequence[str]] = None,
) -> TraceDiff:
    """Compare two traces field by field, timestep by timestep.

    ``tolerance`` is an absolute threshold. Pass ``0.0`` to demand exact
    equality (the determinism check); pass something like ``1e-3`` to ask
    "did this change actually move the robot" rather than "did the last bit
    of a float change".

    The scan does not stop at the first divergence: ``max_abs_diff`` covers the
    whole overlapping region so you can tell a one-step glitch from a run that
    walks away and never comes back.
    """
    compare = list(fields) if fields is not None else list(SAMPLE_FIELDS)
    for name in compare:
        if name not in SAMPLE_FIELDS:
            raise KeyError(f"unknown trace field '{name}'")
    n = min(len(a.samples), len(b.samples))
    max_abs: Dict[str, float] = {name: 0.0 for name in compare}
    first_index: Optional[int] = None
    first_field: Optional[str] = None
    first_values: Optional[Tuple[float, float]] = None

    for i in range(n):
        sa, sb = a.samples[i], b.samples[i]
        for name in compare:
            va = float(getattr(sa, name))
            vb = float(getattr(sb, name))
            if math.isinf(va) and math.isinf(vb) and (va > 0) == (vb > 0):
                delta = 0.0
            else:
                delta = abs(va - vb)
            if delta > max_abs[name]:
                max_abs[name] = delta
            if first_index is None and delta > tolerance:
                first_index = i
                first_field = name
                first_values = (va, vb)

    note = ""
    if len(a.samples) != len(b.samples):
        note = f"length differs: {len(a.samples)} vs {len(b.samples)} samples"

    identical = first_index is None and len(a.samples) == len(b.samples)
    return TraceDiff(
        identical=identical,
        first_divergent_index=first_index,
        first_divergent_time=a.samples[first_index].t if first_index is not None else None,
        first_divergent_field=first_field,
        first_divergent_values=first_values,
        length_a=len(a.samples),
        length_b=len(b.samples),
        compared_steps=n,
        max_abs_diff={k: q(v) for k, v in max_abs.items()},
        tolerance=tolerance,
        note=note,
    )
