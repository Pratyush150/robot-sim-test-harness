"""Results output: terminal, JUnit XML, JSON, and a self-contained HTML report.

Design constraints, chosen deliberately:

* **No plotting library.** The SVG in the HTML report is generated here, by
  hand, from the trace. Adding matplotlib to a CI container to draw four
  polylines is not a trade worth making.
* **No CDN, no external assets.** The HTML file is one file. It opens from a
  build artifact, from an email attachment, and from a machine with no network.
* **JUnit XML that real CI parses.** ``testsuite`` per scenario, ``testcase``
  per assertion, ``skipped`` for expected failures so a deliberately-failing
  scenario does not turn the build red.

The trajectory plot marks the failure point -- the timestamp of the worst
failing assertion -- because "it failed" and "it failed *there*" are different
amounts of information.
"""

from __future__ import annotations

import html
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from .runner import (
    STATUS_ERROR,
    STATUS_EXPECTED_FAILURE,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_TIMEOUT,
    STATUS_UNEXPECTED_PASS,
    RunResult,
    SuiteResult,
)
from .trace import Trace

__all__ = [
    "format_terminal",
    "junit_xml",
    "write_junit",
    "write_json",
    "html_report",
    "write_html",
    "trajectory_svg",
    "timeseries_svg",
]

_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
}

_STATUS_COLOUR = {
    STATUS_PASS: "green",
    STATUS_EXPECTED_FAILURE: "yellow",
    STATUS_FAIL: "red",
    STATUS_UNEXPECTED_PASS: "magenta",
    STATUS_ERROR: "red",
    STATUS_TIMEOUT: "red",
}

_STATUS_LABEL = {
    STATUS_PASS: "PASS",
    STATUS_EXPECTED_FAILURE: "XFAIL",
    STATUS_FAIL: "FAIL",
    STATUS_UNEXPECTED_PASS: "XPASS",
    STATUS_ERROR: "ERROR",
    STATUS_TIMEOUT: "TIMEOUT",
}


def _use_colour(explicit: Optional[bool]) -> bool:
    if explicit is not None:
        return explicit
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


def _paint(text: str, colour: str, enabled: bool) -> str:
    if not enabled or colour not in _ANSI:
        return text
    return f"{_ANSI[colour]}{text}{_ANSI['reset']}"


# --------------------------------------------------------------------------
# terminal
# --------------------------------------------------------------------------


def format_terminal(suite: SuiteResult, *, colour: Optional[bool] = None, verbose: bool = False) -> str:
    """Human-readable suite summary.

    Failing assertions are always shown with their worst value and timestamp.
    ``verbose`` also lists the passing ones.
    """
    on = _use_colour(colour)
    lines: List[str] = []
    lines.append(_paint("simharness scenario suite", "bold", on))
    lines.append("=" * 72)

    for result in suite.results:
        label = _STATUS_LABEL.get(result.status, result.status.upper())
        colour_name = _STATUS_COLOUR.get(result.status, "reset")
        head = f"{_paint(f'{label:<8}', colour_name, on)} {result.scenario}"
        detail = f"{len([a for a in result.assertions if a.passed])}/{len(result.assertions)} assertions"
        timing = f"{result.sim_time_s:.2f}s sim / {result.wall_time_s * 1000:.0f} ms wall"
        lines.append(f"{head}  {_paint(detail, 'dim', on)}  {_paint(timing, 'dim', on)}")

        if result.error:
            for line in result.error.strip().splitlines():
                lines.append(f"           {_paint(line, 'red', on)}")
        for assertion in result.assertions:
            if assertion.passed and not verbose:
                continue
            mark = "+" if assertion.passed else "-"
            tint = "green" if assertion.passed else ("yellow" if result.expect_failure else "red")
            lines.append(f"           {_paint(mark, tint, on)} {assertion.name}: {assertion.message}")
        if verbose and result.metrics:
            metrics = "  ".join(f"{k}={v:g}" for k, v in sorted(result.metrics.items()))
            lines.append(f"           {_paint(metrics, 'dim', on)}")

    lines.append("=" * 72)
    counts = suite.by_status()
    parts = []
    for status in (STATUS_PASS, STATUS_EXPECTED_FAILURE, STATUS_FAIL, STATUS_UNEXPECTED_PASS, STATUS_ERROR, STATUS_TIMEOUT):
        if counts.get(status):
            parts.append(_paint(f"{counts[status]} {_STATUS_LABEL[status].lower()}", _STATUS_COLOUR[status], on))
    verdict = _paint("SUITE PASSED", "green", on) if suite.ok else _paint("SUITE FAILED", "red", on)
    lines.append(f"{verdict}  " + ", ".join(parts) + f"  in {suite.wall_time_s:.2f}s across {suite.workers} worker(s)")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# JUnit XML
# --------------------------------------------------------------------------


def junit_xml(suite: SuiteResult, *, suite_name: str = "simharness") -> str:
    """Render the suite as JUnit XML.

    One ``testsuite`` per scenario, one ``testcase`` per assertion. An
    ``expect_failure`` scenario's failing assertions become ``skipped``
    entries, so CI shows them as deliberately-skipped rather than red. An
    ``unexpected_pass`` produces a real ``failure`` on a synthetic testcase,
    because a negative test that started passing is a broken negative test.
    """
    root = ET.Element("testsuites", name=suite_name)
    total_tests = total_failures = total_errors = total_skipped = 0

    for result in suite.results:
        ts = ET.SubElement(
            root,
            "testsuite",
            name=result.scenario,
            tests=str(max(len(result.assertions), 1)),
            time=f"{result.wall_time_s:.6f}",
        )
        props = ET.SubElement(ts, "properties")
        for key, value in (
            ("simulator", result.simulator),
            ("seed", str(result.seed)),
            ("status", result.status),
            ("expect_failure", str(result.expect_failure).lower()),
            ("source", result.source or ""),
        ):
            ET.SubElement(props, "property", name=key, value=value)
        for key, value in sorted(result.metrics.items()):
            ET.SubElement(props, "property", name=f"metric.{key}", value=f"{value:g}")

        failures = errors = skipped = 0
        if result.status in (STATUS_ERROR, STATUS_TIMEOUT) or not result.assertions:
            case = ET.SubElement(ts, "testcase", classname=result.scenario, name="run", time=f"{result.wall_time_s:.6f}")
            if result.status in (STATUS_ERROR, STATUS_TIMEOUT):
                node = ET.SubElement(case, "error", message=(result.error or result.status)[:400], type=result.status)
                node.text = result.traceback or result.error or result.status
                errors += 1
            total_tests += 1
        for assertion in result.assertions:
            case = ET.SubElement(
                ts,
                "testcase",
                classname=result.scenario,
                name=assertion.name,
                time=f"{result.wall_time_s / max(len(result.assertions), 1):.6f}",
            )
            total_tests += 1
            if assertion.passed:
                continue
            if result.expect_failure:
                node = ET.SubElement(case, "skipped", message=f"expected failure: {assertion.message}"[:400])
                node.text = assertion.message
                skipped += 1
            else:
                node = ET.SubElement(case, "failure", message=assertion.message[:400], type=assertion.type)
                node.text = _assertion_detail(assertion)
                failures += 1
        if result.status == STATUS_UNEXPECTED_PASS:
            case = ET.SubElement(ts, "testcase", classname=result.scenario, name="expected_failure_guard")
            node = ET.SubElement(
                case,
                "failure",
                message="scenario is marked expect_failure but every assertion passed",
                type="unexpected_pass",
            )
            node.text = (
                "This scenario exists to prove the harness catches a regression. "
                "It now passes, so either the bug was fixed (remove expect_failure) "
                "or the assertions no longer test what they used to."
            )
            failures += 1
            total_tests += 1

        ts.set("failures", str(failures))
        ts.set("errors", str(errors))
        ts.set("skipped", str(skipped))
        total_failures += failures
        total_errors += errors
        total_skipped += skipped

    root.set("tests", str(total_tests))
    root.set("failures", str(total_failures))
    root.set("errors", str(total_errors))
    root.set("skipped", str(total_skipped))
    root.set("time", f"{suite.wall_time_s:.6f}")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode")


def _assertion_detail(assertion: Any) -> str:
    parts = [assertion.message]
    if assertion.worst_value is not None:
        parts.append(f"worst value: {assertion.worst_value:g} {assertion.units}".strip())
    if assertion.worst_time is not None:
        parts.append(f"at t = {assertion.worst_time:g} s (step {assertion.worst_index})")
    if assertion.threshold is not None:
        parts.append(f"limit: {assertion.threshold:g} {assertion.units}".strip())
    return "\n".join(parts)


def write_junit(suite: SuiteResult, path: os.PathLike | str, *, suite_name: str = "simharness") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(junit_xml(suite, suite_name=suite_name), encoding="utf-8")
    return target


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------


def write_json(suite: SuiteResult, path: os.PathLike | str, *, include_traces: bool = False) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(suite.to_dict(include_traces=include_traces), indent=2), encoding="utf-8")
    return target


# --------------------------------------------------------------------------
# SVG
# --------------------------------------------------------------------------


@dataclass
class _Projection:
    """Maps world metres to SVG user units, y flipped, aspect preserved."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    width: float
    height: float
    pad: float = 28.0

    def __post_init__(self) -> None:
        span_x = max(self.max_x - self.min_x, 1e-6)
        span_y = max(self.max_y - self.min_y, 1e-6)
        inner_w = max(self.width - 2 * self.pad, 1.0)
        inner_h = max(self.height - 2 * self.pad, 1.0)
        self.scale = min(inner_w / span_x, inner_h / span_y)
        self.offset_x = self.pad + 0.5 * (inner_w - span_x * self.scale)
        self.offset_y = self.pad + 0.5 * (inner_h - span_y * self.scale)

    def px(self, x: float) -> float:
        return self.offset_x + (x - self.min_x) * self.scale

    def py(self, y: float) -> float:
        return self.offset_y + (self.max_y - y) * self.scale

    def r(self, metres: float) -> float:
        return metres * self.scale


def _f(value: float) -> str:
    return f"{value:.2f}"


def trajectory_svg(
    trace: Trace,
    *,
    width: int = 620,
    height: int = 380,
    failure_time: Optional[float] = None,
    title: str = "",
) -> str:
    """Top-down trajectory plot with obstacles, goal and the failure point.

    Everything it needs is in ``trace.meta``, which the runner fills in, so a
    saved trace can be re-plotted later without the scenario file.
    """
    meta = trace.meta or {}
    world = meta.get("world", {})
    goal = meta.get("goal", [0.0, 0.0, 0.0])
    spawn = meta.get("spawn", [0.0, 0.0, 0.0])
    tolerance = float(meta.get("goal_tolerance", 0.25))
    obstacles: List[Mapping[str, Any]] = list(meta.get("obstacles", []))

    xs = [s.x for s in trace.samples] or [0.0]
    ys = [s.y for s in trace.samples] or [0.0]
    min_x = min([*xs, goal[0], spawn[0], world.get("min_xy", [min(xs), 0])[0]])
    max_x = max([*xs, goal[0], spawn[0], world.get("max_xy", [max(xs), 0])[0]])
    min_y = min([*ys, goal[1], spawn[1], world.get("min_xy", [0, min(ys)])[1]])
    max_y = max([*ys, goal[1], spawn[1], world.get("max_xy", [0, max(ys)])[1]])
    margin = 0.05 * max(max_x - min_x, max_y - min_y, 1.0)
    proj = _Projection(min_x - margin, max_x + margin, min_y - margin, max_y + margin, width, height)

    out: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="trajectory plot">'
    ]
    out.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fbfbfd" stroke="#d8d8e0"/>')
    if title:
        out.append(f'<text x="10" y="18" font-size="12" font-family="monospace" fill="#333">{html.escape(title)}</text>')

    if "min_xy" in world and "max_xy" in world:
        wx0, wy0 = world["min_xy"]
        wx1, wy1 = world["max_xy"]
        out.append(
            f'<rect x="{_f(proj.px(wx0))}" y="{_f(proj.py(wy1))}" '
            f'width="{_f(proj.r(wx1 - wx0))}" height="{_f(proj.r(wy1 - wy0))}" '
            'fill="none" stroke="#9aa0b5" stroke-dasharray="6 4" stroke-width="1"/>'
        )

    for obstacle in obstacles:
        out.append(_obstacle_svg(obstacle, proj, trace))

    if len(trace.samples) > 1:
        points = " ".join(f"{_f(proj.px(s.x))},{_f(proj.py(s.y))}" for s in trace.samples)
        out.append(f'<polyline points="{points}" fill="none" stroke="#1f6feb" stroke-width="2"/>')

    out.append(
        f'<circle cx="{_f(proj.px(goal[0]))}" cy="{_f(proj.py(goal[1]))}" r="{_f(max(proj.r(tolerance), 3.0))}" '
        'fill="none" stroke="#1a7f37" stroke-width="1.5" stroke-dasharray="3 3"/>'
    )
    gx, gy = proj.px(goal[0]), proj.py(goal[1])
    out.append(
        f'<path d="M{_f(gx - 6)},{_f(gy)} L{_f(gx + 6)},{_f(gy)} M{_f(gx)},{_f(gy - 6)} L{_f(gx)},{_f(gy + 6)}" '
        'stroke="#1a7f37" stroke-width="2"/>'
    )
    out.append(
        f'<circle cx="{_f(proj.px(spawn[0]))}" cy="{_f(proj.py(spawn[1]))}" r="4" fill="#1f6feb"/>'
    )

    if failure_time is not None and trace.samples:
        index = trace.index_at(failure_time)
        sample = trace.samples[index]
        fx, fy = proj.px(sample.x), proj.py(sample.y)
        out.append(f'<circle cx="{_f(fx)}" cy="{_f(fy)}" r="7" fill="none" stroke="#cf222e" stroke-width="2.5"/>')
        out.append(
            f'<path d="M{_f(fx - 4)},{_f(fy - 4)} L{_f(fx + 4)},{_f(fy + 4)} '
            f'M{_f(fx + 4)},{_f(fy - 4)} L{_f(fx - 4)},{_f(fy + 4)}" stroke="#cf222e" stroke-width="2"/>'
        )
        out.append(
            f'<text x="{_f(min(fx + 10, width - 90))}" y="{_f(max(fy - 10, 14))}" font-size="11" '
            f'font-family="monospace" fill="#cf222e">t={sample.t:.2f}s</text>'
        )

    out.append(
        f'<text x="10" y="{height - 8}" font-size="10" font-family="monospace" fill="#666">'
        f'x {min_x:.1f}..{max_x:.1f} m, y {min_y:.1f}..{max_y:.1f} m</text>'
    )
    out.append("</svg>")
    return "".join(out)


def _obstacle_svg(obstacle: Mapping[str, Any], proj: _Projection, trace: Trace) -> str:
    """Draw one obstacle. Moving obstacles also get a faint swept path."""
    motion = obstacle.get("motion")
    parts: List[str] = []
    if motion and trace.samples:
        from .scenario import Obstacle  # local import keeps report.py importable standalone

        model = Obstacle.parse(dict(obstacle), "obstacle", 0)
        step = max(1, len(trace.samples) // 40)
        pts = []
        for sample in trace.samples[::step]:
            cx, cy, _ = model.position_at(sample.t)
            pts.append(f"{_f(proj.px(cx))},{_f(proj.py(cy))}")
        if len(pts) > 1:
            parts.append(
                f'<polyline points="{" ".join(pts)}" fill="none" stroke="#d1a24a" '
                'stroke-width="1" stroke-dasharray="2 3"/>'
            )
    if obstacle.get("shape", "circle") == "circle":
        parts.append(
            f'<circle cx="{_f(proj.px(obstacle["x"]))}" cy="{_f(proj.py(obstacle["y"]))}" '
            f'r="{_f(proj.r(float(obstacle.get("radius", 0.5))))}" fill="#f2d9a8" stroke="#b58b2d" stroke-width="1.2"/>'
        )
    else:
        sx = float(obstacle.get("size_x", 1.0))
        sy = float(obstacle.get("size_y", 1.0))
        parts.append(
            f'<rect x="{_f(proj.px(obstacle["x"] - sx / 2))}" y="{_f(proj.py(obstacle["y"] + sy / 2))}" '
            f'width="{_f(proj.r(sx))}" height="{_f(proj.r(sy))}" fill="#f2d9a8" stroke="#b58b2d" stroke-width="1.2"/>'
        )
    return "".join(parts)


def timeseries_svg(
    times: Sequence[float],
    values: Sequence[float],
    *,
    label: str,
    units: str = "",
    threshold: Optional[float] = None,
    marker_t: Optional[float] = None,
    width: int = 620,
    height: int = 150,
) -> str:
    """A single-signal line chart with axis ticks and an optional limit line."""
    pad_l, pad_r, pad_t, pad_b = 52.0, 12.0, 18.0, 22.0
    # 1e308 is the sentinel a reloaded trace uses for "no obstacle in range";
    # plotting it would flatten every real value onto the axis.
    finite = [v for v in values if math.isfinite(v) and abs(v) < 1e300]
    if not times or not finite:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
            f'height="{height}"><text x="10" y="20" font-size="11" font-family="monospace">'
            f"{html.escape(label)}: no data</text></svg>"
        )
    lo, hi = min(finite), max(finite)
    if threshold is not None and math.isfinite(threshold):
        lo, hi = min(lo, threshold), max(hi, threshold)
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    span = hi - lo
    if span <= 0.0 or not math.isfinite(span):  # degenerate after padding
        lo, hi, span = -1.0, 1.0, 2.0
    lo -= 0.08 * span
    hi += 0.08 * span
    t0, t1 = times[0], times[-1]
    if t1 - t0 < 1e-9:
        t1 = t0 + 1.0

    def px(t: float) -> float:
        return pad_l + (t - t0) / (t1 - t0) * (width - pad_l - pad_r)

    def py(v: float) -> float:
        return pad_t + (hi - v) / (hi - lo) * (height - pad_t - pad_b)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img" aria-label="{html.escape(label)} time series">'
    ]
    out.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" stroke="#e2e2ea"/>')
    for frac in (0.0, 0.5, 1.0):
        value = hi - frac * (hi - lo)
        y = py(value)
        out.append(
            f'<line x1="{_f(pad_l)}" y1="{_f(y)}" x2="{_f(width - pad_r)}" y2="{_f(y)}" '
            'stroke="#eef0f4" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{_f(pad_l - 6)}" y="{_f(y + 3)}" font-size="9" font-family="monospace" '
            f'fill="#888" text-anchor="end">{value:.3g}</text>'
        )
    out.append(
        f'<line x1="{_f(pad_l)}" y1="{_f(pad_t)}" x2="{_f(pad_l)}" y2="{_f(height - pad_b)}" stroke="#bbb"/>'
    )
    out.append(
        f'<line x1="{_f(pad_l)}" y1="{_f(height - pad_b)}" x2="{_f(width - pad_r)}" '
        f'y2="{_f(height - pad_b)}" stroke="#bbb"/>'
    )
    if threshold is not None and math.isfinite(threshold):
        out.append(
            f'<line x1="{_f(pad_l)}" y1="{_f(py(threshold))}" x2="{_f(width - pad_r)}" y2="{_f(py(threshold))}" '
            'stroke="#cf222e" stroke-width="1" stroke-dasharray="5 4"/>'
        )
    step = max(1, len(times) // 700)
    points = " ".join(
        f"{_f(px(times[i]))},{_f(py(values[i]))}"
        for i in range(0, len(times), step)
        if math.isfinite(values[i]) and abs(values[i]) < 1e300
    )
    out.append(f'<polyline points="{points}" fill="none" stroke="#1f6feb" stroke-width="1.6"/>')
    if marker_t is not None:
        out.append(
            f'<line x1="{_f(px(marker_t))}" y1="{_f(pad_t)}" x2="{_f(px(marker_t))}" '
            f'y2="{_f(height - pad_b)}" stroke="#cf222e" stroke-width="1.2" stroke-dasharray="3 3"/>'
        )
    caption = f"{label} [{units}]" if units else label
    out.append(
        f'<text x="{_f(pad_l)}" y="12" font-size="10" font-family="monospace" fill="#444">{html.escape(caption)}</text>'
    )
    out.append(
        f'<text x="{_f(width - pad_r)}" y="{height - 6}" font-size="9" font-family="monospace" fill="#888" '
        f'text-anchor="end">t = {t0:.1f} .. {t1:.1f} s</text>'
    )
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

_CSS = """
:root { color-scheme: light; }
body { margin: 0; padding: 24px; background: #f6f7f9; color: #16181d;
       font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0; }
.sub { color: #5b6270; margin-bottom: 18px; font-size: 13px; }
.card { background: #fff; border: 1px solid #e2e4ea; border-radius: 8px;
        padding: 16px 18px; margin-bottom: 16px; }
.badge { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 11px;
         font-weight: 700; letter-spacing: .04em; color: #fff; }
.pass { background: #1a7f37; } .fail { background: #cf222e; }
.expected_failure { background: #bf8700; } .unexpected_pass { background: #8250df; }
.error, .timeout { background: #82071e; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #eceef2; }
th { color: #5b6270; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
td.num { text-align: right; font-variant-numeric: tabular-nums; font-family: ui-monospace, monospace; }
.ok { color: #1a7f37; font-weight: 700; } .no { color: #cf222e; font-weight: 700; }
.msg { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.plots { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 12px; }
.metrics { color: #5b6270; font-family: ui-monospace, monospace; font-size: 12px; margin-top: 8px; }
pre { background: #f6f7f9; border: 1px solid #e2e4ea; border-radius: 6px; padding: 10px;
      overflow-x: auto; font-size: 12px; }
footer { color: #7a808c; font-size: 12px; margin-top: 24px; }
"""


def html_report(suite: SuiteResult, *, title: str = "simharness scenario suite") -> str:
    """One self-contained HTML page. No external CSS, JS, fonts or images."""
    counts = suite.by_status()
    verdict = "PASSED" if suite.ok else "FAILED"
    parts: List[str] = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html.escape(title)}</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>{html.escape(title)} &mdash; {verdict}</h1>",
        f'<div class="sub">{len(suite.results)} scenario(s), '
        + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
        + f", {suite.wall_time_s:.2f}s wall across {suite.workers} worker(s)</div>",
        '<div class="card"><h2>Summary</h2><table><tr><th>Scenario</th><th>Status</th>'
        "<th>Assertions</th><th>Sim time</th><th>Path</th><th>Min clearance</th></tr>",
    ]
    for result in suite.results:
        passed = sum(1 for a in result.assertions if a.passed)
        clearance = result.metrics.get("min_clearance_m")
        parts.append(
            f'<tr><td><a href="#{html.escape(result.scenario)}">{html.escape(result.scenario)}</a></td>'
            f'<td><span class="badge {result.status}">{_STATUS_LABEL.get(result.status, result.status)}</span></td>'
            f'<td class="num">{passed}/{len(result.assertions)}</td>'
            f'<td class="num">{result.sim_time_s:.2f} s</td>'
            f'<td class="num">{result.metrics.get("path_length_m", float("nan")):.2f} m</td>'
            f'<td class="num">{"-" if clearance is None else f"{clearance:.3f} m"}</td></tr>'
        )
    parts.append("</table></div>")

    for result in suite.results:
        parts.append(_scenario_card(result))

    parts.append(
        '<footer>Generated by simharness. Plots are inline SVG produced by '
        "simharness.report; no plotting library and no external assets.</footer>"
    )
    parts.append("</body></html>")
    return "\n".join(p for p in parts if p)


def _scenario_card(result: RunResult) -> str:
    label = _STATUS_LABEL.get(result.status, result.status)
    parts = [
        f'<div class="card" id="{html.escape(result.scenario)}">',
        f'<h2>{html.escape(result.scenario)} <span class="badge {result.status}">{label}</span></h2>',
        f'<div class="metrics">simulator={html.escape(result.simulator)} seed={result.seed} '
        f"steps={result.steps} wall={result.wall_time_s * 1000:.0f} ms</div>",
    ]
    if result.error:
        parts.append(f"<pre>{html.escape(result.traceback or result.error)}</pre>")
    if result.assertions:
        parts.append("<table><tr><th>Assertion</th><th>Result</th><th>Worst</th><th>At</th><th>Detail</th></tr>")
        for assertion in result.assertions:
            worst = "-" if assertion.worst_value is None else f"{assertion.worst_value:.4g} {assertion.units}".strip()
            at = "-" if assertion.worst_time is None else f"{assertion.worst_time:.2f} s"
            klass = "ok" if assertion.passed else "no"
            parts.append(
                f"<tr><td>{html.escape(assertion.name)}</td>"
                f'<td class="{klass}">{assertion.status}</td>'
                f'<td class="num">{html.escape(worst)}</td>'
                f'<td class="num">{at}</td>'
                f'<td class="msg">{html.escape(assertion.message)}</td></tr>'
            )
        parts.append("</table>")
    if result.metrics:
        parts.append(
            '<div class="metrics">' + "  ".join(f"{k}={v:g}" for k, v in sorted(result.metrics.items())) + "</div>"
        )
    if result.trace is not None and len(result.trace):
        failure_time = _failure_time(result)
        parts.append('<div class="plots">')
        parts.append(trajectory_svg(result.trace, failure_time=failure_time, title=f"{result.scenario} (top-down)"))
        parts.append("</div>")
        parts.append(_signal_plots(result, failure_time))
    parts.append("</div>")
    return "\n".join(parts)


def _failure_time(result: RunResult) -> Optional[float]:
    for assertion in result.assertions:
        if not assertion.passed and assertion.worst_time is not None:
            return assertion.worst_time
    return None


def _signal_plots(result: RunResult, failure_time: Optional[float]) -> str:
    trace = result.trace
    assert trace is not None
    times = trace.times()
    plots: List[Tuple[str, List[float], str, Optional[float]]] = [
        ("speed", [s.speed() for s in trace.samples], "m/s", _threshold_for(result, "max_velocity")),
        ("distance to goal", _distance_series(trace), "m", None),
    ]
    clearances = [s.clearance for s in trace.samples]
    if any(math.isfinite(c) for c in clearances):
        plots.append(("clearance", clearances, "m", _threshold_for(result, "min_clearance")))
    plots.append(("energy", [s.energy_j / 3600.0 for s in trace.samples], "Wh", _threshold_for(result, "energy_budget")))

    out = ['<div class="plots">']
    for label, values, units, threshold in plots:
        out.append(
            timeseries_svg(times, values, label=label, units=units, threshold=threshold, marker_t=failure_time)
        )
    out.append("</div>")
    return "\n".join(out)


def _distance_series(trace: Trace) -> List[float]:
    goal = trace.meta.get("goal", [0.0, 0.0, 0.0])
    return [math.dist((s.x, s.y, s.z), tuple(goal)) for s in trace.samples]


def _threshold_for(result: RunResult, assertion_type: str) -> Optional[float]:
    for assertion in result.assertions:
        if assertion.type == assertion_type and assertion.threshold is not None:
            return assertion.threshold
    return None


def write_html(suite: SuiteResult, path: os.PathLike | str, *, title: str = "simharness scenario suite") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html_report(suite, title=title), encoding="utf-8")
    return target
