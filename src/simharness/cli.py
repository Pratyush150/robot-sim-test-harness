"""Command line interface.

``simharness --demo`` runs the bundled scenario suite end to end with no
arguments, no simulator install and no network. Everything else is a
subcommand::

    simharness run scenarios/straight_line.yaml
    simharness suite scenarios/ --workers 4 --junit out/junit.xml --html out/report.html
    simharness sweep scenarios/wind_rejection.yaml --range disturbance.wind[0]=0:8 --samples 40
    simharness sweep scenarios/wind_rejection.yaml --boundary disturbance.wind[0] --low 0 --high 8
    simharness replay out/traces/straight_line.trace.json --scenario scenarios/straight_line.yaml
    simharness diff a.trace.json b.trace.json
    simharness report out/results.json --html out/report.html
    simharness ci scenarios/ --baseline ci/baseline.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import __version__
from .assertions import evaluate_all
from .ci import compare_to_baseline, load_baseline, load_tolerances, save_baseline
from .report import format_terminal, write_html, write_json, write_junit
from .runner import SuiteResult, run_scenario, run_suite
from .scenario import ScenarioError, load_scenario, load_scenario_dir
from .simulators import availability_report
from .sweep import find_scenario_boundary, grid_sweep, monte_carlo
from .trace import Trace, diff_traces

LOG = logging.getLogger("simharness")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def _repo_root() -> Path:
    """Where the bundled scenarios live, whether installed or run from a checkout."""
    here = Path(__file__).resolve()
    for candidate in (here.parents[2], here.parents[1], Path.cwd()):
        if (candidate / "scenarios").is_dir():
            return candidate
    return Path.cwd()


def _parse_overrides(items: Sequence[str]) -> Dict[str, Any]:
    """Turn ``--set a.b=1.5`` pairs into an override mapping."""
    import yaml

    out: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got '{item}'")
        key, _, raw = item.partition("=")
        out[key.strip()] = yaml.safe_load(raw)
    return out


def _parse_grid(items: Sequence[str]) -> Dict[str, List[Any]]:
    import yaml

    out: Dict[str, List[Any]] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--param expects path=v1,v2,v3, got '{item}'")
        key, _, raw = item.partition("=")
        out[key.strip()] = [yaml.safe_load(v) for v in raw.split(",") if v != ""]
    return out


def _parse_ranges(items: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for item in items:
        if "=" not in item or ":" not in item:
            raise SystemExit(f"--range expects path=low:high, got '{item}'")
        key, _, raw = item.partition("=")
        low, _, high = raw.partition(":")
        out[key.strip()] = (float(low), float(high))
    return out


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    overrides = _parse_overrides(args.set or [])
    if args.seed is not None:
        overrides["sim.seed"] = args.seed
    scenario = load_scenario(args.scenario, overrides=overrides or None)
    result = run_scenario(
        scenario,
        simulator=args.simulator,
        trace_dir=args.trace_dir,
        wall_timeout_s=args.wall_timeout,
    )
    suite = SuiteResult(results=[result], wall_time_s=result.wall_time_s)
    print(format_terminal(suite, colour=args.color, verbose=args.verbose))
    _emit(suite, args)
    return EXIT_OK if suite.ok else EXIT_FAILED


def cmd_suite(args: argparse.Namespace) -> int:
    directory = args.directory or str(_repo_root() / "scenarios")
    scenarios = load_scenario_dir(directory)
    if not scenarios:
        print(f"no scenarios found under {directory}", file=sys.stderr)
        return EXIT_USAGE
    suite = run_suite(
        scenarios,
        simulator=args.simulator,
        workers=args.workers,
        keep_traces=bool(args.html or args.trace_dir),
        trace_dir=args.trace_dir,
        wall_timeout_s=args.wall_timeout,
    )
    print(format_terminal(suite, colour=args.color, verbose=args.verbose))
    _emit(suite, args)
    return EXIT_OK if suite.ok else EXIT_FAILED


def cmd_sweep(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario, overrides=_parse_overrides(args.set or []) or None)

    if args.boundary:
        if args.low is None or args.high is None:
            print("--boundary needs --low and --high", file=sys.stderr)
            return EXIT_USAGE
        boundary = find_scenario_boundary(
            scenario,
            args.boundary,
            low=args.low,
            high=args.high,
            tolerance=args.tolerance,
            simulator=args.simulator,
        )
        print(boundary.summary())
        for value, passed in boundary.history:
            print(f"  {args.boundary}={value:<12.6g} {'pass' if passed else 'fail'}")
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps(boundary.to_dict(), indent=2), encoding="utf-8")
            print(f"wrote {args.json}")
        return EXIT_OK if boundary.found else EXIT_FAILED

    if args.range:
        result = monte_carlo(
            scenario,
            _parse_ranges(args.range),
            samples=args.samples,
            seed=args.seed or 0,
            workers=args.workers,
            simulator=args.simulator,
        )
    elif args.param:
        result = grid_sweep(scenario, _parse_grid(args.param), workers=args.workers, simulator=args.simulator)
    else:
        print("sweep needs --param, --range or --boundary", file=sys.stderr)
        return EXIT_USAGE

    print(f"sweep '{result.scenario}' ({result.mode}): {result.passed}/{result.total} passed "
          f"({result.pass_rate * 100:.1f}%) in {result.wall_time_s:.2f}s")
    print()
    print(result.format_table(args.table_by))
    failures = result.failures()
    if failures:
        print()
        print(f"first {min(len(failures), 10)} failing point(s):")
        for point in failures[:10]:
            reason = point.error or ", ".join(point.failed_assertions) or point.status
            print(f"  {point.label()}  ->  {reason}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return EXIT_OK if result.pass_rate >= args.min_pass_rate else EXIT_FAILED


def cmd_replay(args: argparse.Namespace) -> int:
    """Re-evaluate assertions against a stored trace, without re-simulating."""
    trace = Trace.load_json(args.trace)
    source = args.scenario or trace.meta.get("source")
    if not source:
        print("replay needs --scenario (the trace does not name its scenario file)", file=sys.stderr)
        return EXIT_USAGE
    scenario = load_scenario(source)
    results = evaluate_all(trace, scenario)
    failed = [r for r in results if not r.passed]
    print(f"replayed '{trace.scenario}' from {args.trace}: {len(trace)} samples, "
          f"{trace.duration:.2f}s sim, seed {trace.seed}, simulator {trace.simulator}")
    for assertion in results:
        print(f"  {assertion}")
    verdict = "PASS" if not failed else "FAIL"
    if scenario.expect_failure:
        verdict = "XFAIL" if failed else "XPASS"
    print(f"{verdict}: {len(results) - len(failed)}/{len(results)} assertions")
    ok = (not failed) if not scenario.expect_failure else bool(failed)
    return EXIT_OK if ok else EXIT_FAILED


def cmd_diff(args: argparse.Namespace) -> int:
    a = Trace.load_json(args.a)
    b = Trace.load_json(args.b)
    fields = args.fields.split(",") if args.fields else None
    result = diff_traces(a, b, tolerance=args.tolerance, fields=fields)
    print(result.summary())
    if not result.identical:
        worst = result.worst_field
        print(f"  samples: {result.length_a} vs {result.length_b}, compared {result.compared_steps} steps")
        if worst:
            print(f"  largest deviation: '{worst}' up to {result.max_abs_diff[worst]:.9g}")
        for name, value in sorted(result.max_abs_diff.items(), key=lambda kv: -kv[1])[:8]:
            if value > 0.0:
                print(f"    {name:14s} {value:.9g}")
    if args.json:
        Path(args.json).write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return EXIT_OK if result.identical else EXIT_FAILED


def cmd_report(args: argparse.Namespace) -> int:
    """Render a stored results JSON into HTML, JUnit XML or the terminal."""
    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    suite = SuiteResult.from_dict(data)
    for result in suite.results:
        if result.trace is None and result.trace_path and Path(result.trace_path).exists():
            result.trace = Trace.load_json(result.trace_path)
    print(format_terminal(suite, colour=args.color, verbose=args.verbose))
    _emit(suite, args)
    return EXIT_OK if suite.ok else EXIT_FAILED


def cmd_ci(args: argparse.Namespace) -> int:
    directory = args.directory or str(_repo_root() / "scenarios")
    scenarios = load_scenario_dir(directory)
    suite = run_suite(
        scenarios,
        simulator=args.simulator,
        workers=args.workers,
        keep_traces=bool(args.html),
        trace_dir=args.trace_dir,
    )
    print(format_terminal(suite, colour=args.color, verbose=args.verbose))
    _emit(suite, args)

    baseline_path = Path(args.baseline)
    if args.update_baseline:
        save_baseline(suite, baseline_path, note=args.note or "")
        print(f"\nwrote baseline {baseline_path} ({len(suite.results)} scenarios)")
        return EXIT_OK if suite.ok else EXIT_FAILED
    if not baseline_path.exists():
        print(f"\nno baseline at {baseline_path}; create one with --update-baseline", file=sys.stderr)
        return EXIT_OK if suite.ok else EXIT_FAILED

    tolerances = load_tolerances(args.tolerances) if args.tolerances else None
    comparison = compare_to_baseline(
        suite,
        load_baseline(baseline_path),
        tolerances=tolerances,
        strict_metrics=args.strict_metrics,
    )
    print()
    print(comparison.report())
    if args.comparison_json:
        Path(args.comparison_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.comparison_json).write_text(json.dumps(comparison.to_dict(), indent=2), encoding="utf-8")
    return EXIT_OK if comparison.ok else EXIT_FAILED


def cmd_simulators(args: argparse.Namespace) -> int:
    print(f"{'backend':10s} {'available':10s} reason")
    print("-" * 78)
    for name, available, reason in availability_report():
        print(f"{name:10s} {'yes' if available else 'no':10s} {reason}")
    return EXIT_OK


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the bundled suite plus a small sweep. No arguments, no installs."""
    root = _repo_root()
    directory = root / "scenarios"
    print(f"simharness {__version__} demo")
    print(f"scenarios: {directory}")
    print()
    for name, available, reason in availability_report():
        print(f"  backend {name:8s} {'available' if available else 'unavailable'}: {reason[:88]}")
    print()
    scenarios = load_scenario_dir(directory)
    suite = run_suite(scenarios, keep_traces=False, workers=args.workers)
    print(format_terminal(suite, colour=args.color, verbose=False))

    sweep_target = next((s for s in scenarios if "sweepable" in s.tags), None)
    if sweep_target is not None:
        print()
        print(f"parameter sweep on '{sweep_target.name}':")
        result = grid_sweep(
            sweep_target,
            {"disturbance.wind[1]": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]},
            workers=args.workers,
        )
        print(result.format_table())
        print()
        print("failure boundary (bisection):")
        boundary = find_scenario_boundary(
            sweep_target, "disturbance.wind[1]", low=0.0, high=14.0, tolerance=0.1
        )
        print(f"  {boundary.summary()}")
    return EXIT_OK if suite.ok else EXIT_FAILED


def _emit(suite: SuiteResult, args: argparse.Namespace) -> None:
    """Write whichever machine-readable outputs were requested."""
    if getattr(args, "junit", None):
        print(f"wrote {write_junit(suite, args.junit)}")
    if getattr(args, "html", None):
        print(f"wrote {write_html(suite, args.html)}")
    if getattr(args, "json", None) and getattr(args, "command", "") != "sweep":
        print(f"wrote {write_json(suite, args.json)}")


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simharness",
        description="Scenario-driven regression testing for robots in simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"simharness {__version__}")
    parser.add_argument("--demo", action="store_true", help="run the bundled suite and a sweep, then exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="show passing assertions and metrics")
    parser.add_argument("--debug", action="store_true", help="enable debug logging")
    colour = parser.add_mutually_exclusive_group()
    colour.add_argument("--color", dest="color", action="store_true", default=None, help="force ANSI colour")
    colour.add_argument("--no-color", dest="color", action="store_false", help="disable ANSI colour")
    parser.add_argument("--workers", type=int, default=1, help="parallel worker processes (default 1)")

    # The same flags again, with SUPPRESS defaults, so they work either side of
    # the subcommand: "simharness -v suite" and "simharness suite -v" both work.
    common = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    common.add_argument("-v", "--verbose", action="store_true", help="show passing assertions and metrics")
    common.add_argument("--debug", action="store_true", help="enable debug logging")
    common_colour = common.add_mutually_exclusive_group()
    common_colour.add_argument("--color", dest="color", action="store_true", help="force ANSI colour")
    common_colour.add_argument("--no-color", dest="color", action="store_false", help="disable ANSI colour")
    common.add_argument("--workers", type=int, help="parallel worker processes")

    subs = parser.add_subparsers(dest="command", parser_class=argparse.ArgumentParser)

    def add_common(sub: argparse.ArgumentParser, *, outputs: bool = True) -> None:
        sub.add_argument("--simulator", default=None, help="backend name, or 'auto' to detect (default: mock)")
        sub.add_argument("--wall-timeout", type=float, default=None, help="abort a run after N wall-clock seconds")
        if outputs:
            sub.add_argument("--junit", default=None, help="write JUnit XML here")
            sub.add_argument("--html", default=None, help="write a self-contained HTML report here")
            sub.add_argument("--json", default=None, help="write results JSON here")
            sub.add_argument("--trace-dir", default=None, help="write per-scenario trace JSON into this directory")

    p_run = subs.add_parser("run", parents=[common], help="run a single scenario")
    p_run.add_argument("scenario")
    p_run.add_argument("--seed", type=int, default=None, help="override sim.seed")
    p_run.add_argument("--set", action="append", metavar="PATH=VALUE", help="override any scenario field")
    add_common(p_run)
    p_run.set_defaults(func=cmd_run)

    p_suite = subs.add_parser("suite", parents=[common], help="run every scenario in a directory")
    p_suite.add_argument("directory", nargs="?", default=None)
    add_common(p_suite)
    p_suite.set_defaults(func=cmd_suite)

    p_sweep = subs.add_parser("sweep", parents=[common], help="parameter sweep, Monte Carlo, or failure-boundary bisection")
    p_sweep.add_argument("scenario")
    p_sweep.add_argument("--param", action="append", metavar="PATH=V1,V2", help="grid values for a parameter")
    p_sweep.add_argument("--range", action="append", metavar="PATH=LOW:HIGH", help="Monte Carlo range")
    p_sweep.add_argument("--samples", type=int, default=40, help="Monte Carlo sample count")
    p_sweep.add_argument("--seed", type=int, default=0, help="sampling seed")
    p_sweep.add_argument("--boundary", default=None, metavar="PATH", help="bisect this parameter for the failure edge")
    p_sweep.add_argument("--low", type=float, default=None, help="boundary search lower bound (expected to pass)")
    p_sweep.add_argument("--high", type=float, default=None, help="boundary search upper bound (expected to fail)")
    p_sweep.add_argument("--tolerance", type=float, default=0.05, help="boundary bracket width")
    p_sweep.add_argument("--table-by", default=None, help="which parameter to tabulate")
    p_sweep.add_argument("--min-pass-rate", type=float, default=1.0, help="exit non-zero below this pass rate")
    p_sweep.add_argument("--set", action="append", metavar="PATH=VALUE", help="fixed overrides applied first")
    p_sweep.add_argument("--json", default=None, help="write the sweep result JSON here")
    p_sweep.add_argument("--simulator", default=None)
    p_sweep.set_defaults(func=cmd_sweep)

    p_replay = subs.add_parser("replay", parents=[common], help="re-evaluate assertions against a saved trace")
    p_replay.add_argument("trace")
    p_replay.add_argument("--scenario", default=None, help="scenario file the trace came from")
    p_replay.set_defaults(func=cmd_replay)

    p_diff = subs.add_parser("diff", parents=[common], help="compare two traces and report the first divergent step")
    p_diff.add_argument("a")
    p_diff.add_argument("b")
    p_diff.add_argument("--tolerance", type=float, default=0.0, help="absolute per-field tolerance (default: exact)")
    p_diff.add_argument("--fields", default=None, help="comma-separated subset of fields to compare")
    p_diff.add_argument("--json", default=None)
    p_diff.set_defaults(func=cmd_diff)

    p_report = subs.add_parser("report", parents=[common], help="render a stored results JSON")
    p_report.add_argument("results")
    p_report.add_argument("--junit", default=None)
    p_report.add_argument("--html", default=None)
    p_report.add_argument("--json", default=None)
    p_report.set_defaults(func=cmd_report)

    p_ci = subs.add_parser("ci", parents=[common], help="run the suite and compare against a stored baseline")
    p_ci.add_argument("directory", nargs="?", default=None)
    p_ci.add_argument("--baseline", default="ci/baseline.json")
    p_ci.add_argument("--update-baseline", action="store_true", help="overwrite the baseline with this run")
    p_ci.add_argument("--strict-metrics", action="store_true", help="fail the build on metric drift too")
    p_ci.add_argument("--tolerances", default=None, help="YAML/JSON tolerance overrides")
    p_ci.add_argument("--comparison-json", default=None, help="write the comparison verdict here")
    p_ci.add_argument("--note", default=None, help="note stored alongside a new baseline")
    add_common(p_ci)
    p_ci.set_defaults(func=cmd_ci)

    p_sim = subs.add_parser("simulators", parents=[common], help="list backends and why each is or is not available")
    p_sim.set_defaults(func=cmd_simulators)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.demo:
        return cmd_demo(args)
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_USAGE
    try:
        return int(args.func(args))
    except ScenarioError as exc:
        print(f"scenario error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except FileNotFoundError as exc:
        print(f"file not found: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
