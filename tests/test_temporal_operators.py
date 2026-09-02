"""Temporal operators, evaluated on hand-built boolean series and traces.

These are the assertions with real semantics to get wrong, so they are tested
twice: once as pure functions over boolean lists, and once end to end through
the scenario assertion machinery on a trace whose values are written by hand.
"""

from __future__ import annotations

import pytest

from simharness.assertions import (
    build_signals,
    compile_predicate,
    describe_predicate,
    evaluate_assertion,
    op_always,
    op_eventually,
    op_eventually_always,
    op_until,
)
from simharness.scenario import AssertionSpec, ScenarioError

T, F = True, False


# -- always ---------------------------------------------------------------


def test_always_true_series():
    assert op_always([T, T, T]) == (True, None)


def test_always_reports_the_first_violation_not_the_last():
    assert op_always([T, T, F, T, F]) == (False, 2)


def test_always_on_empty_series_is_vacuously_true():
    assert op_always([]) == (True, None)


# -- eventually -----------------------------------------------------------


def test_eventually_finds_the_first_satisfying_index():
    assert op_eventually([F, F, T, T]) == (True, 2)


def test_eventually_false_everywhere():
    assert op_eventually([F, F, F]) == (False, None)


def test_eventually_on_empty_series_is_false():
    assert op_eventually([]) == (False, None)


# -- eventually always ----------------------------------------------------


def test_eventually_always_finds_the_start_of_the_final_run():
    assert op_eventually_always([F, T, F, T, T, T]) == (True, 3)


def test_eventually_always_ignores_an_earlier_run_of_trues():
    # The signal settles, un-settles, then settles again: only the last run counts.
    assert op_eventually_always([T, T, F, F, T, T]) == (True, 4)


def test_eventually_always_fails_when_the_last_step_is_false():
    assert op_eventually_always([T, T, T, F]) == (False, None)


def test_eventually_always_all_true_stabilises_at_zero():
    assert op_eventually_always([T, T, T]) == (True, 0)


def test_eventually_always_single_true_at_the_end():
    assert op_eventually_always([F, F, T]) == (True, 2)


# -- until ----------------------------------------------------------------


def test_until_releases_at_the_first_true_of_b():
    holds, release, violation = op_until([T, T, T, T], [F, F, T, T])
    assert (holds, release, violation) == (True, 2, None)


def test_until_is_satisfied_immediately_when_b_holds_at_step_zero():
    assert op_until([F, F], [T, F]) == (True, 0, None)


def test_until_fails_when_a_breaks_before_b():
    holds, release, violation = op_until([T, F, T], [F, F, T])
    assert (holds, release, violation) == (False, None, 1)


def test_strong_until_fails_when_b_never_happens():
    holds, release, violation = op_until([T, T, T], [F, F, F])
    assert holds is False
    assert release is None
    assert violation is None


def test_until_stops_at_the_shorter_of_the_two_series():
    assert op_until([T, T], [F, F, T]) == (False, None, None)


# -- predicate compilation ------------------------------------------------


def _ctx(make_trace, **columns):
    from conftest import minimal_scenario_dict
    from simharness.scenario import Scenario

    trace = make_trace(columns)
    return build_signals(trace, Scenario.from_dict(minimal_scenario_dict()))


def test_atom_predicate_compares_a_signal(make_trace):
    ctx = _ctx(make_trace, x=[0.0, 1.0, 2.0, 3.0])
    series = compile_predicate({"signal": "x", "op": "<", "value": 2.0}, "p")(ctx)
    assert series == [True, True, False, False]


def test_all_predicate_is_a_conjunction(make_trace):
    ctx = _ctx(make_trace, x=[0.0, 1.0, 2.0], y=[5.0, 0.0, 0.0])
    spec = {"all": [{"signal": "x", "op": "<=", "value": 1.0}, {"signal": "y", "op": "<=", "value": 1.0}]}
    assert compile_predicate(spec, "p")(ctx) == [False, True, False]


def test_any_predicate_is_a_disjunction(make_trace):
    ctx = _ctx(make_trace, x=[0.0, 5.0, 5.0], y=[5.0, 5.0, 0.0])
    spec = {"any": [{"signal": "x", "op": "<=", "value": 1.0}, {"signal": "y", "op": "<=", "value": 1.0}]}
    assert compile_predicate(spec, "p")(ctx) == [True, False, True]


def test_not_predicate_inverts(make_trace):
    ctx = _ctx(make_trace, x=[0.0, 5.0])
    assert compile_predicate({"not": {"signal": "x", "op": "<=", "value": 1.0}}, "p")(ctx) == [False, True]


def test_unknown_signal_names_the_key(make_trace):
    ctx = _ctx(make_trace, x=[0.0])
    with pytest.raises(ScenarioError) as exc:
        compile_predicate({"signal": "altitude_agl", "op": "<", "value": 1.0}, "assertions[0].predicate")(ctx)
    assert "assertions[0].predicate.signal" in str(exc.value)


def test_unknown_operator_is_rejected():
    with pytest.raises(ScenarioError) as exc:
        compile_predicate({"signal": "x", "op": "=~", "value": 1.0}, "p")
    assert exc.value.key == "p.op"


def test_missing_value_is_rejected():
    with pytest.raises(ScenarioError) as exc:
        compile_predicate({"signal": "x", "op": "<"}, "p")
    assert exc.value.key == "p.value"


def test_describe_predicate_renders_nested_expressions():
    text = describe_predicate({"all": [{"signal": "x", "op": ">", "value": 1}, {"not": {"signal": "y", "op": "<", "value": 2}}]})
    assert text == "(x > 1 and not y < 2)"


# -- through the assertion layer ------------------------------------------


def _run(make_trace, spec_dict, **columns):
    from conftest import minimal_scenario_dict
    from simharness.scenario import Scenario

    ctx = build_signals(make_trace(columns), Scenario.from_dict(minimal_scenario_dict()))
    spec = AssertionSpec(
        type=spec_dict["type"],
        params={k: v for k, v in spec_dict.items() if k not in ("type", "name")},
        name=spec_dict.get("name"),
        path="assertions[0]",
    )
    return evaluate_assertion(spec, ctx)


def test_always_assertion_reports_the_violation_timestamp(make_trace):
    result = _run(
        make_trace,
        {"type": "always", "predicate": {"signal": "x", "op": "<=", "value": 1.5}},
        x=[0.0, 1.0, 2.0, 3.0],
    )
    assert result.passed is False
    assert result.worst_index == 2
    assert result.worst_time == pytest.approx(0.2)


def test_eventually_assertion_respects_its_deadline(make_trace):
    columns = {"x": [0.0, 0.0, 0.0, 9.0]}  # only satisfied at t = 0.3
    late = _run(make_trace, {"type": "eventually", "predicate": {"signal": "x", "op": ">=", "value": 5.0}, "within": 0.2}, **columns)
    on_time = _run(make_trace, {"type": "eventually", "predicate": {"signal": "x", "op": ">=", "value": 5.0}, "within": 0.4}, **columns)
    assert late.passed is False
    assert on_time.passed is True
    assert on_time.worst_time == pytest.approx(0.3)


def test_eventually_always_assertion_reports_the_stabilisation_time(make_trace):
    result = _run(
        make_trace,
        {"type": "eventually_always", "predicate": {"signal": "x", "op": "<=", "value": 1.0}, "settle_by": 0.5},
        x=[5.0, 0.5, 5.0, 0.5, 0.5, 0.5],
    )
    assert result.passed is True
    assert result.worst_index == 3
    assert result.worst_time == pytest.approx(0.3)


def test_eventually_always_assertion_fails_past_its_deadline(make_trace):
    result = _run(
        make_trace,
        {"type": "eventually_always", "predicate": {"signal": "x", "op": "<=", "value": 1.0}, "settle_by": 0.2},
        x=[5.0, 5.0, 5.0, 0.5, 0.5],
    )
    assert result.passed is False
    assert result.worst_time == pytest.approx(0.3)


def test_until_assertion_end_to_end(make_trace):
    result = _run(
        make_trace,
        {
            "type": "until",
            "condition": {"signal": "clearance", "op": ">=", "value": 0.2},
            "release": {"signal": "x", "op": ">=", "value": 3.0},
        },
        x=[0.0, 1.0, 2.0, 3.0, 4.0],
        clearance=[1.0, 1.0, 1.0, 1.0, 0.0],
    )
    assert result.passed is True
    assert result.worst_index == 3


def test_until_assertion_reports_where_the_condition_broke(make_trace):
    result = _run(
        make_trace,
        {
            "type": "until",
            "condition": {"signal": "clearance", "op": ">=", "value": 0.2},
            "release": {"signal": "x", "op": ">=", "value": 3.0},
        },
        x=[0.0, 1.0, 2.0, 3.0],
        clearance=[1.0, 0.1, 1.0, 1.0],
    )
    assert result.passed is False
    assert result.worst_index == 1
    assert "broke" in result.message


def test_never_assertion(make_trace):
    ok = _run(make_trace, {"type": "never", "predicate": {"signal": "x", "op": ">", "value": 9.0}}, x=[0.0, 1.0])
    bad = _run(make_trace, {"type": "never", "predicate": {"signal": "x", "op": ">", "value": 0.5}}, x=[0.0, 1.0])
    assert ok.passed is True
    assert bad.passed is False
    assert bad.worst_index == 1


def test_unknown_assertion_type_names_the_key(make_trace):
    with pytest.raises(ScenarioError) as exc:
        _run(make_trace, {"type": "vibes_check"}, x=[0.0])
    assert "assertions[0].type" in str(exc.value)
