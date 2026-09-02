"""Small pure-Python geometry helpers used by the simulator and assertions.

No third-party dependencies. Every function here is deterministic and works on
plain floats and tuples so that a recorded trace is bit-reproducible across
runs on the same interpreter.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

Vec3 = Tuple[float, float, float]

__all__ = [
    "Vec3",
    "clamp",
    "wrap_angle",
    "angle_diff",
    "norm2",
    "norm3",
    "dist2",
    "dist3",
    "point_segment_distance",
    "point_in_polygon",
    "polygon_signed_distance",
    "box_signed_distance",
]


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    if low > high:
        raise ValueError(f"clamp() called with low={low} > high={high}")
    if value < low:
        return low
    if value > high:
        return high
    return value


def wrap_angle(angle: float) -> float:
    """Wrap an angle in radians to ``(-pi, pi]``."""
    wrapped = math.fmod(angle + math.pi, 2.0 * math.pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


def angle_diff(target: float, source: float) -> float:
    """Shortest signed angular difference ``target - source`` in ``(-pi, pi]``."""
    return wrap_angle(target - source)


def norm2(x: float, y: float) -> float:
    """Euclidean norm of a 2D vector."""
    return math.hypot(x, y)


def norm3(x: float, y: float, z: float) -> float:
    """Euclidean norm of a 3D vector."""
    return math.sqrt(x * x + y * y + z * z)


def dist2(a: Sequence[float], b: Sequence[float]) -> float:
    """Planar distance between two points, ignoring any z component."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dist3(a: Sequence[float], b: Sequence[float]) -> float:
    """3D distance between two points. Missing z components are treated as 0."""
    az = a[2] if len(a) > 2 else 0.0
    bz = b[2] if len(b) > 2 else 0.0
    return norm3(a[0] - b[0], a[1] - b[1], az - bz)


def point_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """Distance from point ``p`` to the segment ``ab`` in 2D."""
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = clamp(t, 0.0, 1.0)
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def point_in_polygon(px: float, py: float, polygon: Sequence[Sequence[float]]) -> bool:
    """Ray-casting point-in-polygon test for a simple (non self-intersecting) polygon.

    Points exactly on an edge are reported as inside, which is the behaviour a
    geofence check wants (a vehicle sitting on the fence is not yet outside).
    """
    n = len(polygon)
    if n < 3:
        raise ValueError(f"polygon needs at least 3 vertices, got {n}")
    for i in range(n):
        ax, ay = polygon[i][0], polygon[i][1]
        bx, by = polygon[(i + 1) % n][0], polygon[(i + 1) % n][1]
        if point_segment_distance(px, py, ax, ay, bx, by) <= 1e-12:
            return True
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if (yi > py) != (yj > py):
            x_cross = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_cross:
                inside = not inside
        j = i
    return inside


def polygon_signed_distance(
    px: float, py: float, polygon: Sequence[Sequence[float]]
) -> float:
    """Signed distance to a polygon boundary: positive inside, negative outside."""
    n = len(polygon)
    best = min(
        point_segment_distance(
            px,
            py,
            polygon[i][0],
            polygon[i][1],
            polygon[(i + 1) % n][0],
            polygon[(i + 1) % n][1],
        )
        for i in range(n)
    )
    return best if point_in_polygon(px, py, polygon) else -best


def box_signed_distance(
    px: float, py: float, min_xy: Sequence[float], max_xy: Sequence[float]
) -> float:
    """Signed distance to an axis-aligned box boundary: positive inside."""
    left = px - min_xy[0]
    right = max_xy[0] - px
    bottom = py - min_xy[1]
    top = max_xy[1] - py
    if left >= 0.0 and right >= 0.0 and bottom >= 0.0 and top >= 0.0:
        return min(left, right, bottom, top)
    dx = max(min_xy[0] - px, 0.0, px - max_xy[0])
    dy = max(min_xy[1] - py, 0.0, py - max_xy[1])
    return -math.hypot(dx, dy)


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean; returns 0.0 for an empty iterable."""
    items = list(values)
    if not items:
        return 0.0
    return math.fsum(items) / len(items)
