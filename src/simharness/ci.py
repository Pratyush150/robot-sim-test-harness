"""CI integration: baselines, tolerances, and a build-failing verdict.

A scenario suite that only reports pass/fail catches the moment a robot starts
crashing into things. It does not catch the slow stuff: the path that got 8%
longer, the settling time that crept up, the energy budget that is now 95%
consumed instead of 70%. Those are the changes that turn into a field problem
three sprints later.

So the suite is compared against a **stored baseline** with **per-metric
tolerances**:

* a scenario that was green and is now red is a regression, and fails the build;
* a scenario that was red and is now green is reported, not celebrated silently;
* a metric that moved more than its tolerance is reported as drift, and fails
  the build only under ``strict_metrics`` -- because the honest default is that
  a 12% longer path is something a human should look at, not something that
  should block a merge at 2 a.m.

Baselines are JSON and are meant to be committed. Refresh them deliberately
with ``simharness ci --update-baseline`` and review the diff like any other
change.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .runner import GREEN_STATUSES, SuiteResult

__all__ = [
    "MetricTolerance",
    "DEFAULT_TOLERANCES",
    "Baseline",
    "MetricDrift",
    "Comparison",
    "build_baseline",
    "save_baseline",
    "load_baseline",
    "compare_to_baseline",
    "load_tolerances",
]

BASELINE_FORMAT = "simharness-baseline/1"


@dataclass(frozen=True)
class MetricTolerance:
    """How far a metric may move before it counts as drift.

    A value is within tolerance when it is inside *either* the absolute or the
    relative band. Small numbers need the absolute term (0.001 -> 0.002 is a
    100% change and means nothing); large ones need the relative term.
    """

    abs_tol: float = 0.0
    rel_tol: float = 0.0

    def within(self, baseline: float, current: float) -> bool:
        delta = abs(current - baseline)
        if delta <= self.abs_tol:
            return True
        if self.rel_tol > 0.0 and abs(baseline) > 0.0:
            return delta / abs(baseline) <= self.rel_tol
        return False


#: Sensible starting point. Override per project; these are not laws of physics.
DEFAULT_TOLERANCES: Dict[str, MetricTolerance] = {
    "path_length_m": MetricTolerance(abs_tol=0.05, rel_tol=0.05),
    "time_to_goal_s": MetricTolerance(abs_tol=0.10, rel_tol=0.05),
    "final_distance_to_goal_m": MetricTolerance(abs_tol=0.05, rel_tol=0.20),
    "min_clearance_m": MetricTolerance(abs_tol=0.02, rel_tol=0.10),
    "max_speed_mps": MetricTolerance(abs_tol=0.05, rel_tol=0.05),
    "energy_wh": MetricTolerance(abs_tol=0.001, rel_tol=0.10),
    "sim_time_s": MetricTolerance(abs_tol=0.10, rel_tol=0.05),
}


def load_tolerances(path: os.PathLike | str) -> Dict[str, MetricTolerance]:
    """Load tolerances from JSON or YAML.

    Format::

        path_length_m: {abs: 0.05, rel: 0.05}
        min_clearance_m: {abs: 0.02}
    """
    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith((".yaml", ".yml")):
        import yaml

        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: tolerance file must be a mapping of metric -> {{abs, rel}}")
    out: Dict[str, MetricTolerance] = dict(DEFAULT_TOLERANCES)
    for name, spec in raw.items():
        if not isinstance(spec, Mapping):
            raise ValueError(f"{path}: tolerance for '{name}' must be a mapping with 'abs' and/or 'rel'")
        out[str(name)] = MetricTolerance(
            abs_tol=float(spec.get("abs", spec.get("abs_tol", 0.0))),
            rel_tol=float(spec.get("rel", spec.get("rel_tol", 0.0))),
        )
    return out


@dataclass
class Baseline:
    """A recorded suite result, to be compared against later runs."""

    entries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    simulator: str = ""
    recorded_at: float = 0.0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": BASELINE_FORMAT,
            "simulator": self.simulator,
            "recorded_at": self.recorded_at,
            "note": self.note,
            "scenarios": self.entries,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "Baseline":
        fmt = data.get("format")
        if fmt != BASELINE_FORMAT:
            raise ValueError(f"unsupported baseline format '{fmt}', expected '{BASELINE_FORMAT}'")
        return Baseline(
            entries=dict(data.get("scenarios", {})),
            simulator=str(data.get("simulator", "")),
            recorded_at=float(data.get("recorded_at", 0.0)),
            note=str(data.get("note", "")),
        )


def build_baseline(suite: SuiteResult, *, note: str = "") -> Baseline:
    """Snapshot a suite result as a baseline."""
    entries: Dict[str, Dict[str, Any]] = {}
    for result in suite.results:
        entries[result.scenario] = {
            "status": result.status,
            "seed": result.seed,
            "expect_failure": result.expect_failure,
            "assertions": {a.name: a.passed for a in result.assertions},
            "metrics": dict(result.metrics),
        }
    simulator = suite.results[0].simulator if suite.results else ""
    return Baseline(entries=entries, simulator=simulator, recorded_at=time.time(), note=note)


def save_baseline(suite: SuiteResult, path: os.PathLike | str, *, note: str = "") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_baseline(suite, note=note).to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return target


def load_baseline(path: os.PathLike | str) -> Baseline:
    return Baseline.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class MetricDrift:
    """One metric that moved further than its tolerance allows."""

    scenario: str
    metric: str
    baseline: float
    current: float
    tolerance: MetricTolerance

    @property
    def delta(self) -> float:
        return self.current - self.baseline

    @property
    def percent(self) -> Optional[float]:
        if self.baseline == 0.0:
            return None
        return 100.0 * self.delta / abs(self.baseline)

    def describe(self) -> str:
        pct = "" if self.percent is None else f" ({self.percent:+.1f}%)"
        return (
            f"{self.scenario}.{self.metric}: {self.baseline:.6g} -> {self.current:.6g}"
            f"{pct}, tolerance abs {self.tolerance.abs_tol:g} / rel {self.tolerance.rel_tol:g}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "metric": self.metric,
            "baseline": self.baseline,
            "current": self.current,
            "delta": self.delta,
            "percent": self.percent,
            "abs_tol": self.tolerance.abs_tol,
            "rel_tol": self.tolerance.rel_tol,
        }


@dataclass
class Comparison:
    """The verdict for one CI run."""

    regressions: List[Tuple[str, str, str]] = field(default_factory=list)
    improvements: List[Tuple[str, str, str]] = field(default_factory=list)
    status_changes: List[Tuple[str, str, str]] = field(default_factory=list)
    new_scenarios: List[str] = field(default_factory=list)
    missing_scenarios: List[str] = field(default_factory=list)
    drift: List[MetricDrift] = field(default_factory=list)
    strict_metrics: bool = False
    suite_ok: bool = True

    @property
    def ok(self) -> bool:
        """Should the build pass?"""
        if not self.suite_ok or self.regressions or self.missing_scenarios:
            return False
        return not (self.strict_metrics and self.drift)

    def report(self) -> str:
        lines: List[str] = []
        if self.regressions:
            lines.append(f"Regressions ({len(self.regressions)}):")
            lines.extend(f"  - {name}: {was} -> {now}" for name, was, now in self.regressions)
        if self.missing_scenarios:
            lines.append(f"Missing from this run ({len(self.missing_scenarios)}):")
            lines.extend(f"  - {name} (in the baseline, not in the suite)" for name in self.missing_scenarios)
        if self.improvements:
            lines.append(f"Improvements ({len(self.improvements)}):")
            lines.extend(f"  - {name}: {was} -> {now}" for name, was, now in self.improvements)
        if self.status_changes:
            lines.append(f"Status changes within the same verdict ({len(self.status_changes)}):")
            lines.extend(f"  - {name}: {was} -> {now}" for name, was, now in self.status_changes)
        if self.new_scenarios:
            lines.append(f"New scenarios, not yet in the baseline ({len(self.new_scenarios)}):")
            lines.extend(f"  - {name}" for name in self.new_scenarios)
        if self.drift:
            severity = "failing" if self.strict_metrics else "advisory"
            lines.append(f"Metric drift beyond tolerance ({len(self.drift)}, {severity}):")
            lines.extend(f"  - {d.describe()}" for d in self.drift)
        if not lines:
            lines.append("No status changes and no metric drift against the baseline.")
        lines.append("CI verdict: " + ("PASS" if self.ok else "FAIL"))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "suite_ok": self.suite_ok,
            "strict_metrics": self.strict_metrics,
            "regressions": [{"scenario": s, "was": w, "now": n} for s, w, n in self.regressions],
            "improvements": [{"scenario": s, "was": w, "now": n} for s, w, n in self.improvements],
            "status_changes": [{"scenario": s, "was": w, "now": n} for s, w, n in self.status_changes],
            "new_scenarios": self.new_scenarios,
            "missing_scenarios": self.missing_scenarios,
            "drift": [d.to_dict() for d in self.drift],
        }


def compare_to_baseline(
    suite: SuiteResult,
    baseline: Baseline,
    *,
    tolerances: Optional[Mapping[str, MetricTolerance]] = None,
    strict_metrics: bool = False,
    ignore_missing: bool = False,
) -> Comparison:
    """Compare a suite result against a stored baseline."""
    tol = dict(DEFAULT_TOLERANCES)
    if tolerances:
        tol.update(tolerances)
    comparison = Comparison(strict_metrics=strict_metrics, suite_ok=suite.ok)
    seen: set[str] = set()

    for result in suite.results:
        seen.add(result.scenario)
        entry = baseline.entries.get(result.scenario)
        if entry is None:
            comparison.new_scenarios.append(result.scenario)
            continue
        was = str(entry.get("status", "unknown"))
        now = result.status
        was_green = was in GREEN_STATUSES
        now_green = now in GREEN_STATUSES
        if was_green and not now_green:
            comparison.regressions.append((result.scenario, was, now))
        elif not was_green and now_green:
            comparison.improvements.append((result.scenario, was, now))
        elif was != now:
            # Same colour, different flavour: pass -> expected_failure, or
            # fail -> timeout. Not a build failure, but you want to see it.
            comparison.status_changes.append((result.scenario, was, now))

        base_metrics = entry.get("metrics", {}) or {}
        for metric, current in result.metrics.items():
            if metric not in base_metrics:
                continue
            limit = tol.get(metric)
            if limit is None:
                continue
            baseline_value = float(base_metrics[metric])
            if not limit.within(baseline_value, float(current)):
                comparison.drift.append(
                    MetricDrift(
                        scenario=result.scenario,
                        metric=metric,
                        baseline=baseline_value,
                        current=float(current),
                        tolerance=limit,
                    )
                )

    if not ignore_missing:
        comparison.missing_scenarios = sorted(set(baseline.entries) - seen)
    return comparison
