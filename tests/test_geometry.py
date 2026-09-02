"""Geometry primitives. Small, but everything downstream sits on them."""

from __future__ import annotations

import math

import pytest

from simharness.geometry import (
    angle_diff,
    box_signed_distance,
    clamp,
    dist2,
    dist3,
    mean,
    point_in_polygon,
    point_segment_distance,
    polygon_signed_distance,
    wrap_angle,
)

SQUARE = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]


def test_clamp_bounds_and_passthrough():
    assert clamp(5.0, 0.0, 1.0) == 1.0
    assert clamp(-5.0, 0.0, 1.0) == 0.0
    assert clamp(0.5, 0.0, 1.0) == 0.5


def test_clamp_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        clamp(0.0, 1.0, -1.0)


@pytest.mark.parametrize(
    "angle,expected",
    [(0.0, 0.0), (math.pi, math.pi), (-math.pi, math.pi), (3 * math.pi, math.pi), (2 * math.pi, 0.0)],
)
def test_wrap_angle(angle, expected):
    assert wrap_angle(angle) == pytest.approx(expected, abs=1e-12)


def test_angle_diff_takes_the_short_way_round():
    assert angle_diff(math.radians(179), math.radians(-179)) == pytest.approx(math.radians(-2), abs=1e-9)


def test_distances():
    assert dist2((0.0, 0.0, 9.0), (3.0, 4.0, -9.0)) == pytest.approx(5.0)
    assert dist3((0.0, 0.0, 0.0), (1.0, 2.0, 2.0)) == pytest.approx(3.0)
    assert dist3((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


def test_point_segment_distance_endpoints_and_middle():
    assert point_segment_distance(0.0, 1.0, -1.0, 0.0, 1.0, 0.0) == pytest.approx(1.0)
    assert point_segment_distance(3.0, 0.0, -1.0, 0.0, 1.0, 0.0) == pytest.approx(2.0)
    assert point_segment_distance(0.0, 0.0, 2.0, 2.0, 2.0, 2.0) == pytest.approx(math.sqrt(8))


def test_point_in_polygon_inside_outside_and_on_edge():
    assert point_in_polygon(0.0, 0.0, SQUARE) is True
    assert point_in_polygon(2.0, 0.0, SQUARE) is False
    assert point_in_polygon(1.0, 0.0, SQUARE) is True  # on the boundary counts as inside


def test_point_in_polygon_rejects_degenerate_polygon():
    with pytest.raises(ValueError):
        point_in_polygon(0.0, 0.0, [(0.0, 0.0), (1.0, 1.0)])


def test_polygon_signed_distance_sign_convention():
    assert polygon_signed_distance(0.0, 0.0, SQUARE) == pytest.approx(1.0)
    assert polygon_signed_distance(3.0, 0.0, SQUARE) == pytest.approx(-2.0)


def test_box_signed_distance_sign_convention():
    assert box_signed_distance(0.0, 0.0, (-2.0, -1.0), (2.0, 1.0)) == pytest.approx(1.0)
    assert box_signed_distance(4.0, 0.0, (-2.0, -1.0), (2.0, 1.0)) == pytest.approx(-2.0)
    assert box_signed_distance(5.0, 4.0, (-2.0, -1.0), (2.0, 1.0)) == pytest.approx(-math.hypot(3.0, 3.0))


def test_mean_handles_empty():
    assert mean([]) == 0.0
    assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
