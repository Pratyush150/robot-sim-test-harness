"""CLI smoke tests. Every subcommand is exercised end to end, offline."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from simharness.cli import EXIT_FAILED, EXIT_OK, EXIT_USAGE, build_parser, main


@pytest.fixture
def scenarios(scenario_dir: Path) -> str:
    return str(scenario_dir)


def test_parser_builds_and_lists_every_subcommand():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions, "the parser must expose subcommands"
    assert set(actions[0].choices) == {"run", "suite", "sweep", "replay", "diff", "report", "ci", "simulators"}


def test_no_arguments_prints_help_and_exits_with_usage(capsys):
    assert main([]) == EXIT_USAGE
    assert "simharness" in capsys.readouterr().out


def test_run_a_passing_scenario(capsys, scenario_dir: Path):
    code = main(["run", str(scenario_dir / "straight_line.yaml"), "--no-color"])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "PASS" in out
    assert "straight_line" in out


def test_run_the_expected_failure_scenario_is_green(capsys, scenario_dir: Path):
    code = main(["run", str(scenario_dir / "expected_failure_no_avoidance.yaml"), "--no-color"])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "XFAIL" in out


def test_run_with_a_set_override_changes_the_outcome(capsys, scenario_dir: Path):
    code = main(
        ["run", str(scenario_dir / "straight_line.yaml"), "--no-color", "--set", "sim.time_limit=0.5",
         "--set", "goal.within=0.4"]
    )
    assert code == EXIT_FAILED
    assert "FAIL" in capsys.readouterr().out


def test_run_with_a_bad_override_reports_a_scenario_error(capsys, scenario_dir: Path):
    code = main(["run", str(scenario_dir / "straight_line.yaml"), "--set", "sim.tme_limit=1"])
    assert code == EXIT_USAGE
    assert "scenario error" in capsys.readouterr().err


def test_run_writes_every_requested_artifact(tmp_path: Path, scenario_dir: Path, capsys):
    code = main(
        [
            "run",
            str(scenario_dir / "obstacle_slalom.yaml"),
            "--no-color",
            "--junit", str(tmp_path / "junit.xml"),
            "--html", str(tmp_path / "report.html"),
            "--json", str(tmp_path / "results.json"),
            "--trace-dir", str(tmp_path / "traces"),
        ]
    )
    capsys.readouterr()
    assert code == EXIT_OK
    ET.parse(tmp_path / "junit.xml")
    assert "<svg" in (tmp_path / "report.html").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))["ok"] is True
    assert (tmp_path / "traces" / "obstacle_slalom.trace.json").exists()


def test_suite_runs_everything_and_is_green(capsys, scenarios):
    code = main(["suite", scenarios, "--no-color", "--workers", "2"])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "SUITE PASSED" in out
    assert "xfail" in out


def test_suite_on_an_empty_directory_is_a_usage_error(tmp_path: Path, capsys):
    code = main(["suite", str(tmp_path), "--no-color"])
    assert code == EXIT_USAGE
    assert "no scenarios" in capsys.readouterr().err


def test_sweep_grid_prints_a_pass_rate_table(capsys, scenario_dir: Path):
    code = main(
        [
            "sweep",
            str(scenario_dir / "wind_disturbance.yaml"),
            "--param", "disturbance.wind[1]=0,4,8,12",
            "--min-pass-rate", "0.0",
            "--no-color",
        ]
    )
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "pass" in out.lower()
    assert "disturbance.wind[1]" in out
    assert "total" in out


def test_sweep_monte_carlo_is_repeatable(capsys, scenario_dir: Path, tmp_path: Path):
    args = [
        "sweep",
        str(scenario_dir / "wind_disturbance.yaml"),
        "--range", "disturbance.wind[1]=0:12",
        "--samples", "6",
        "--seed", "11",
        "--min-pass-rate", "0.0",
        "--json", str(tmp_path / "sweep.json"),
    ]
    main(args)
    capsys.readouterr()
    first = json.loads((tmp_path / "sweep.json").read_text(encoding="utf-8"))
    main(args)
    capsys.readouterr()
    second = json.loads((tmp_path / "sweep.json").read_text(encoding="utf-8"))
    assert [p["overrides"] for p in first["points"]] == [p["overrides"] for p in second["points"]]
    assert first["mode"] == "monte_carlo"


def test_sweep_boundary_finds_the_wind_edge(capsys, scenario_dir: Path):
    code = main(
        [
            "sweep",
            str(scenario_dir / "wind_disturbance.yaml"),
            "--boundary", "disturbance.wind[1]",
            "--low", "0",
            "--high", "14",
            "--tolerance", "0.5",
        ]
    )
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "boundary" in out
    assert "pass" in out and "fail" in out


def test_sweep_without_a_mode_is_a_usage_error(capsys, scenario_dir: Path):
    code = main(["sweep", str(scenario_dir / "straight_line.yaml")])
    assert code == EXIT_USAGE
    assert "--param" in capsys.readouterr().err


def test_replay_reevaluates_a_saved_trace(tmp_path: Path, scenario_dir: Path, capsys):
    main(["run", str(scenario_dir / "straight_line.yaml"), "--no-color", "--trace-dir", str(tmp_path)])
    capsys.readouterr()
    code = main(
        ["replay", str(tmp_path / "straight_line.trace.json"), "--scenario", str(scenario_dir / "straight_line.yaml")]
    )
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "replayed 'straight_line'" in out
    assert "PASS" in out


def test_replay_reports_an_expected_failure_as_xfail(tmp_path: Path, scenario_dir: Path, capsys):
    path = scenario_dir / "expected_failure_no_avoidance.yaml"
    main(["run", str(path), "--no-color", "--trace-dir", str(tmp_path)])
    capsys.readouterr()
    code = main(["replay", str(tmp_path / "expected_failure_no_avoidance.trace.json"), "--scenario", str(path)])
    assert code == EXIT_OK
    assert "XFAIL" in capsys.readouterr().out


def test_diff_of_identical_traces_exits_zero(tmp_path: Path, scenario_dir: Path, capsys):
    a = tmp_path / "a"
    b = tmp_path / "b"
    for target in (a, b):
        main(["run", str(scenario_dir / "straight_line.yaml"), "--no-color", "--trace-dir", str(target)])
    capsys.readouterr()
    code = main([
        "diff",
        str(a / "straight_line.trace.json"),
        str(b / "straight_line.trace.json"),
    ])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "identical" in out


def test_diff_of_different_seeds_reports_the_first_divergent_step(tmp_path: Path, scenario_dir: Path, capsys):
    a = tmp_path / "a"
    b = tmp_path / "b"
    main(["run", str(scenario_dir / "sensor_dropout.yaml"), "--no-color", "--trace-dir", str(a)])
    main(["run", str(scenario_dir / "sensor_dropout.yaml"), "--no-color", "--seed", "999", "--trace-dir", str(b)])
    capsys.readouterr()
    code = main([
        "diff",
        str(a / "sensor_dropout.trace.json"),
        str(b / "sensor_dropout.trace.json"),
        "--json", str(tmp_path / "diff.json"),
    ])
    out = capsys.readouterr().out
    assert code == EXIT_FAILED
    assert "first divergence at step" in out
    payload = json.loads((tmp_path / "diff.json").read_text(encoding="utf-8"))
    assert payload["identical"] is False
    assert payload["first_divergent_index"] is not None


def test_report_renders_a_stored_results_file(tmp_path: Path, scenario_dir: Path, capsys):
    main([
        "suite", str(scenario_dir), "--no-color",
        "--json", str(tmp_path / "results.json"),
        "--trace-dir", str(tmp_path / "traces"),
    ])
    capsys.readouterr()
    code = main([
        "report", str(tmp_path / "results.json"),
        "--no-color",
        "--html", str(tmp_path / "report.html"),
        "--junit", str(tmp_path / "junit.xml"),
    ])
    capsys.readouterr()
    assert code == EXIT_OK
    ET.parse(tmp_path / "junit.xml")
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "<svg" in html, "traces referenced by the results file must be re-loaded for plotting"


def test_ci_creates_then_compares_a_baseline(tmp_path: Path, scenarios, capsys):
    baseline = tmp_path / "baseline.json"
    assert main(["ci", scenarios, "--baseline", str(baseline), "--update-baseline", "--no-color"]) == EXIT_OK
    capsys.readouterr()
    assert baseline.exists()
    code = main([
        "ci", scenarios, "--baseline", str(baseline), "--no-color",
        "--comparison-json", str(tmp_path / "comparison.json"),
    ])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "CI verdict: PASS" in out
    payload = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert payload["regressions"] == []


def test_ci_detects_a_missing_scenario_against_the_baseline(tmp_path: Path, scenarios, capsys, scenario_dir: Path):
    baseline = tmp_path / "baseline.json"
    main(["ci", scenarios, "--baseline", str(baseline), "--update-baseline", "--no-color"])
    capsys.readouterr()
    subset = tmp_path / "subset"
    subset.mkdir()
    for name in ("_base_ground.yaml", "straight_line.yaml"):
        (subset / name).write_text((scenario_dir / name).read_text(encoding="utf-8"), encoding="utf-8")
    code = main(["ci", str(subset), "--baseline", str(baseline), "--no-color"])
    out = capsys.readouterr().out
    assert code == EXIT_FAILED
    assert "Missing from this run" in out


def test_simulators_command_lists_backends(capsys):
    assert main(["simulators"]) == EXIT_OK
    out = capsys.readouterr().out
    for name in ("mock", "gazebo", "airsim", "isaac"):
        assert name in out
    assert "yes" in out


def test_demo_runs_the_whole_pipeline(capsys):
    code = main(["--demo", "--no-color", "--workers", "2"])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "SUITE PASSED" in out
    assert "parameter sweep" in out
    assert "failure boundary" in out
