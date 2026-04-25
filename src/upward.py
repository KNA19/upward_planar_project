from __future__ import annotations

import math
from typing import Any, Sequence, Tuple

import numpy as np

from src.geometry import edges_for_order


Vector2 = Tuple[int, int]


# ------------------------------------------------------------
# Basic vector helpers
# ------------------------------------------------------------

def _is_zero_vec(v: Vector2) -> bool:
    """
    True iff the 2D vector is exactly (0, 0).
    """
    return int(v[0]) == 0 and int(v[1]) == 0


def _dot(u: Vector2, v: Vector2) -> int:
    """
    Integer dot product of two 2D vectors.
    """
    return int(u[0]) * int(v[0]) + int(u[1]) * int(v[1])


def _cross(u: Vector2, v: Vector2) -> int:
    """
    z-component of the 2D cross product:
    (u_x, u_y, 0) x (v_x, v_y, 0)
    """
    return int(u[0]) * int(v[1]) - int(u[1]) * int(v[0])


def _angle(v: Vector2) -> float:
    """
    Return the angle of vector v in [0, 2*pi).
    """
    if _is_zero_vec(v):
        raise ValueError("Zero vector has no direction.")

    a = math.atan2(int(v[1]), int(v[0]))

    if a < 0:
        a += 2.0 * math.pi

    return a


def _unique_angles(angles: list[float], eps: float = 1e-12) -> list[float]:
    """
    Remove duplicate angles caused by parallel edge directions.
    """
    if not angles:
        return []

    angles = sorted(angles)
    unique = [angles[0]]

    for a in angles[1:]:
        if abs(a - unique[-1]) > eps:
            unique.append(a)

    # Handle wrap-around duplicates.
    if len(unique) > 1:
        if abs((unique[0] + 2.0 * math.pi) - unique[-1]) <= eps:
            unique.pop()

    return unique


def _inside_open_semicircle(
    angle: float,
    start: float,
    eps: float = 1e-12,
) -> bool:
    """
    Return True iff 'angle' lies strictly inside the open interval

        (start, start + pi)

    measured counterclockwise modulo 2*pi.
    """
    two_pi = 2.0 * math.pi
    diff = angle - start

    if diff < 0:
        diff += two_pi

    return eps < diff < math.pi - eps


# ------------------------------------------------------------
# Legacy upward-sector test
# ------------------------------------------------------------

def is_in_acute_angular_sector(vectors: list[Vector2]) -> bool:
    """
    Return True iff all vectors lie inside an open wedge of width
    strictly less than pi.

    Equivalently, all vectors lie in one open half-plane whose boundary
    passes through the origin.

    This is the correct condition for upwardness after some rotation of
    the coordinate system.

    Important:
    - A zero vector is invalid and returns False.
    - This version is safer than the older wedge-maintenance version,
      because repeated parallel vectors in the same direction are handled
      correctly.
    """
    angles: list[float] = []

    for v in vectors:
        vv = (int(v[0]), int(v[1]))

        if _is_zero_vec(vv):
            return False

        angles.append(_angle(vv))

    if len(angles) <= 1:
        return True

    angles = _unique_angles(angles)

    if len(angles) <= 1:
        return True

    two_pi = 2.0 * math.pi
    max_gap = 0.0

    for i in range(len(angles)):
        if i + 1 < len(angles):
            gap = angles[i + 1] - angles[i]
        else:
            gap = angles[0] + two_pi - angles[i]

        max_gap = max(max_gap, gap)

    # The remaining directions fit in an open semicircle iff the
    # complementary empty gap is strictly larger than pi.
    return max_gap > math.pi + 1e-12


# ------------------------------------------------------------
# Directed edge-vector helper
# ------------------------------------------------------------

def _directed_edge_vectors(
    points: np.ndarray,
    order: Sequence[int],
    dirs: Sequence[int],
    cycle: bool = False,
) -> list[Vector2]:
    """
    Build directed edge vectors for the polyline defined by:

        points : coordinates array of shape (n, 2)
        order  : vertex order / permutation
        dirs   : per-edge directions, +1 forward and -1 backward

    Returns:
        list of integer (dx, dy) vectors.
    """
    p = np.asarray(points, dtype=np.int64)
    edges = edges_for_order(order, cycle=cycle)

    if len(edges) != len(dirs):
        raise ValueError(
            f"Length mismatch: got {len(edges)} edges but {len(dirs)} directions."
        )

    vecs: list[Vector2] = []

    for (u, v), d in zip(edges, dirs):
        tail, head = (u, v) if int(d) > 0 else (v, u)

        dx = int(p[head, 0]) - int(p[tail, 0])
        dy = int(p[head, 1]) - int(p[tail, 1])

        vec = (dx, dy)

        if _is_zero_vec(vec):
            raise ValueError("Zero-length edge found. This is not a valid embedding.")

        vecs.append(vec)

    return vecs


# ------------------------------------------------------------
# Chapter 6 aligned upward-orientation computation
# ------------------------------------------------------------

def upward_orientations_for_order(
    points: np.ndarray,
    order: Sequence[int],
    *,
    cycle: bool = False,
) -> list[dict[str, Any]]:
    """
    Given one undirected plane path order, compute all upward orientations
    induced by rotating the coordinate system.

    The returned direction string uses:

        + : edge is oriented order[i] -> order[i+1]
        - : edge is oriented order[i+1] -> order[i]

    Returns:
        [
            {
                "orientation_id": 1,
                "dirs": "+--+",
                "up_direction_angle": ...,
                "rotation_angle": ...
            },
            ...
        ]

    Meaning:
        up_direction_angle:
            A witness direction that can be treated as vertical upward.

        rotation_angle:
            Rotate the drawing by this angle to make the witness upward
            direction become vertical upward.
    """
    p = np.asarray(points, dtype=np.int64)
    edges = edges_for_order(order, cycle=cycle)
    m = len(edges)

    if m == 0:
        return [
            {
                "orientation_id": 1,
                "dirs": "",
                "up_direction_angle": math.pi / 2.0,
                "rotation_angle": 0.0,
            }
        ]

    base_angles: list[float] = []
    candidate_angles: list[float] = []

    for u, v in edges:
        dx = int(p[v, 0]) - int(p[u, 0])
        dy = int(p[v, 1]) - int(p[u, 1])

        vec = (dx, dy)

        if _is_zero_vec(vec):
            raise ValueError("Zero-length edge found. This is not a valid embedding.")

        a = _angle(vec)

        base_angles.append(a)
        candidate_angles.append(a)
        candidate_angles.append((a + math.pi) % (2.0 * math.pi))

    candidate_angles = _unique_angles(candidate_angles)

    results: list[dict[str, Any]] = []
    seen_dirs: set[str] = set()

    two_pi = 2.0 * math.pi

    for i in range(len(candidate_angles)):
        a1 = candidate_angles[i]
        a2 = candidate_angles[(i + 1) % len(candidate_angles)]

        if i == len(candidate_angles) - 1:
            a2 += two_pi

        if a2 - a1 <= 1e-12:
            continue

        # alpha is a representative lower boundary direction of an open
        # upward half-plane. We choose it between two consecutive event angles.
        alpha = ((a1 + a2) / 2.0) % two_pi

        dirs: list[str] = []
        valid = True

        for a in base_angles:
            forward_inside = _inside_open_semicircle(a, alpha)
            backward_inside = _inside_open_semicircle((a + math.pi) % two_pi, alpha)

            if forward_inside and not backward_inside:
                dirs.append("+")
            elif backward_inside and not forward_inside:
                dirs.append("-")
            else:
                valid = False
                break

        if not valid:
            continue

        dirs_str = "".join(dirs)

        # Avoid duplicate orientation records when parallel edge directions
        # create repeated rotation cells with the same orientation.
        if dirs_str in seen_dirs:
            continue

        seen_dirs.add(dirs_str)

        # The center of the open half-plane is a valid upward direction.
        up_direction_angle = (alpha + math.pi / 2.0) % two_pi

        # Rotate by this angle so that up_direction_angle becomes pi/2.
        rotation_angle = (math.pi / 2.0 - up_direction_angle) % two_pi

        results.append(
            {
                "orientation_id": len(results) + 1,
                "dirs": dirs_str,
                "up_direction_angle": up_direction_angle,
                "rotation_angle": rotation_angle,
            }
        )

    return results


# ------------------------------------------------------------
# Fixed-coordinate upwardness, useful for debugging/display
# ------------------------------------------------------------

def is_upward_in_current_coordinates(vectors: list[Vector2]) -> bool:
    """
    Return True iff every directed edge vector has positive y-component.

    This checks upwardness in the current coordinate system only.
    It does not allow rotation.
    """
    for dx, dy in vectors:
        if int(dx) == 0 and int(dy) == 0:
            return False

        if int(dy) <= 0:
            return False

    return True