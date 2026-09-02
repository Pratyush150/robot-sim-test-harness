"""Reports: terminal text, JUnit XML, JSON, and the self-contained HTML page."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from conftest import minimal_scenario_dict
from simharness.report import (
    format_terminal,
    html_report,
    junit_xml,
    timeseries_svg,
    trajectory_svg,
    write_html,
    write_json,
    write_junit,
)
from simharness.runner import SuiteResult, run_scenario, run_suite
from simharness.scenario import Scenario, load_scenario_dir

SVG_RE = re.compile(r"<svg\b.*?</svg>", re.DOTALL)


@pytest.fixture(scope="module")
def suite(request) -> SuiteResult:
    root = Path(request.config.rootdir)
    names = ("straight_line", "obstacle_slalom", "expected_failure_no_avoidance")
    scenarios = [s for s in load_scenario_dir(root / "scenarios") if s.name in names]
    return run_suite(scenarios, keep_traces=True)


@pytest.fixture(scope="module")
def broken_suite() -> SuiteResult:
    scenario = Scenario.from_dict(
        minimal_scenario_dict(
            assertions=[{"type": "min_clearance", "name": "impossible", "threshold": 99.0}],
            obstacles=[{"id": "blob", "x": 6.0, "y": 3.0, "radius": 0.5}],
        )
    )
    return SuiteResult(results=[run_scenario(scenario)], wall_time_s=0.1)


# -- terminal --------------------------------------------------------------


def test_terminal_summary_lists_every_scenario(suite):
    text = format_terminal(suite, colour=False)
    for result in suite.results:
        assert result.scenario in text
    assert "SUITE PASSED" in text
    assert "\x1b[" not in text, "colour must be suppressible"


def test_terminal_summary_shows_the_expected_failure_as_xfail(suite):
    text = format_terminal(suite, colour=False)
    assert "XFAIL" in text
    assert "expected_failure_no_avoidance" in text


def test_terminal_colour_can_be_forced(suite):
    assert "\x1b[" in format_terminal(suite, colour=True)


def test_terminal_verbose_shows_passing_assertions(suite):
    quiet = format_terminal(suite, colour=False, verbose=False)
    loud = format_terminal(suite, colour=False, verbose=True)
    assert len(loud) > len(quiet)
    assert "path_length_m=" in loud


def test_terminal_reports_a_failing_suite(broken_suite):
    text = format_terminal(broken_suite, colour=False)
    assert "SUITE FAILED" in text
    assert "impossible" in text


# -- JUnit -----------------------------------------------------------------


def test_junit_is_well_formed_and_parses_with_stdlib_elementtree(suite):
    root = ET.fromstring(junit_xml(suite))
    assert root.tag == "testsuites"
    assert int(root.get("tests")) > 0
    names = {ts.get("name") for ts in root.findall("testsuite")}
    assert names == {r.scenario for r in suite.results}


def test_junit_testcase_per_assertion(suite):
    root = ET.fromstring(junit_xml(suite))
    for result in suite.results:
        ts = root.find(f"./testsuite[@name='{result.scenario}']")
        cases = {tc.get("name") for tc in ts.findall("testcase")}
        for assertion in result.assertions:
            assert assertion.name in cases


def test_junit_marks_expected_failures_as_skipped_not_failed(suite):
    root = ET.fromstring(junit_xml(suite))
    ts = root.find("./testsuite[@name='expected_failure_no_avoidance']")
    assert int(ts.get("failures")) == 0
    assert int(ts.get("skipped")) >= 1
    skipped = ts.findall(".//skipped")
    assert any("expected failure" in node.get("message") for node in skipped)


def test_junit_records_a_real_failure(broken_suite):
    root = ET.fromstring(junit_xml(broken_suite))
    assert int(root.get("failures")) == 1
    failure = root.find(".//failure")
    assert failure is not None
    assert failure.get("type") == "min_clearance"
    assert "worst value" in failure.text


def test_junit_records_metrics_as_properties(suite):
    root = ET.fromstring(junit_xml(suite))
    ts = root.find("./testsuite[@name='straight_line']")
    props = {p.get("name") for p in ts.findall("./properties/property")}
    assert "simulator" in props
    assert "seed" in props
    assert any(name.startswith("metric.") for name in props)


def test_junit_flags_an_unexpected_pass_as_a_failure():
    data = minimal_scenario_dict(assertions=[{"type": "no_collision"}], expect_failure=True)
    result = run_scenario(Scenario.from_dict(data))
    root = ET.fromstring(junit_xml(SuiteResult(results=[result])))
    guard = root.find(".//testcase[@name='expected_failure_guard']/failure")
    assert guard is not None
    assert guard.get("type") == "unexpected_pass"


def test_junit_records_an_errored_run_as_an_error():
    from simharness.runner import RunResult, STATUS_ERROR

    result = RunResult(scenario="boom", status=STATUS_ERROR, error="SimulatorError: nope", traceback="trace")
    root = ET.fromstring(junit_xml(SuiteResult(results=[result])))
    assert int(root.get("errors")) == 1
    assert root.find(".//error").get("type") == STATUS_ERROR


def test_write_junit_produces_a_parseable_file(suite, tmp_path: Path):
    path = write_junit(suite, tmp_path / "nested" / "junit.xml")
    assert path.exists()
    ET.parse(path)


# -- JSON ------------------------------------------------------------------


def test_write_json_round_trips(suite, tmp_path: Path):
    path = write_json(suite, tmp_path / "results.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["ok"] is True
    assert len(data["results"]) == len(suite.results)
    restored = SuiteResult.from_dict(data)
    assert [r.scenario for r in restored] == [r.scenario for r in suite]
    assert restored.ok == suite.ok


# -- SVG -------------------------------------------------------------------


def _assert_valid_svg(markup: str) -> ET.Element:
    root = ET.fromstring(markup)
    assert root.tag.endswith("svg")
    assert root.get("viewBox")
    return root


def test_trajectory_svg_is_valid_xml_and_draws_the_path(suite):
    result = suite.get("obstacle_slalom")
    markup = trajectory_svg(result.trace, title="obstacle_slalom")
    root = _assert_valid_svg(markup)
    assert root.findall(".//{http://www.w3.org/2000/svg}polyline"), "the trajectory must be drawn"
    assert len(root.findall(".//{http://www.w3.org/2000/svg}circle")) >= 4, "obstacles, goal and start"


def test_trajectory_svg_marks_the_failure_point(suite):
    result = suite.get("expected_failure_no_avoidance")
    failure_time = next(a.worst_time for a in result.assertions if not a.passed and a.worst_time is not None)
    markup = trajectory_svg(result.trace, failure_time=failure_time)
    _assert_valid_svg(markup)
    assert f"t={failure_time:.2f}s" in markup


def test_timeseries_svg_is_valid_and_draws_a_threshold():
    times = [i * 0.1 for i in range(50)]
    values = [float(i) for i in range(50)]
    markup = timeseries_svg(times, values, label="speed", units="m/s", threshold=25.0, marker_t=2.0)
    root = _assert_valid_svg(markup)
    assert len(root.findall(".//{http://www.w3.org/2000/svg}line")) >= 4
    assert "speed [m/s]" in markup


def test_timeseries_svg_survives_empty_input():
    _assert_valid_svg(timeseries_svg([], [], label="nothing"))


def test_timeseries_svg_survives_a_constant_signal():
    times = [i * 0.1 for i in range(10)]
    _assert_valid_svg(timeseries_svg(times, [3.0] * 10, label="flat"))


# -- HTML ------------------------------------------------------------------


def test_html_report_is_self_contained(suite):
    page = html_report(suite)
    assert page.startswith("<!doctype html>")
    assert "<style>" in page
    assert "http://" not in page.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in page
    assert "<script" not in page


def test_html_report_contains_valid_svg_for_every_scenario(suite):
    page = html_report(suite)
    blocks = SVG_RE.findall(page)
    assert len(blocks) >= 4 * len(suite.results)
    for block in blocks:
        _assert_valid_svg(block)


def test_html_report_lists_assertions_and_statuses(suite):
    page = html_report(suite)
    assert "XFAIL" in page
    assert "keeps_its_distance" in page
    assert "min clearance" in page or "minimum clearance" in page


def test_html_report_escapes_scenario_text():
    from simharness.runner import RunResult

    result = RunResult(scenario="<script>alert(1)</script>", status="pass")
    page = html_report(SuiteResult(results=[result]))
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_write_html_creates_the_file(suite, tmp_path: Path):
    path = write_html(suite, tmp_path / "out" / "report.html")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "simharness" in text
    assert SVG_RE.search(text)
