from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from src.geometry import edges_for_order


Vector2 = Tuple[int, int]


def _is_zero_vec(v: Vector2) -> bool:
    """
    True iff the 2D vector is exactly (0, 0).
    """
    return v[0] == 0 and v[1] == 0


def _dot(u: Vector2, v: Vector2) -> int:
    """
    Integer dot product of two 2D vectors.
    """
    return u[0] * v[0] + u[1] * v[1]


def _cross(u: Vector2, v: Vector2) -> int:
    """
    z-component of the 2D cross product:
    (u_x, u_y, 0) x (v_x, v_y, 0)
    """
    return u[0] * v[1] - u[1] * v[0]


def is_in_acute_angular_sector(vectors: List[Vector2]) -> bool:
    """
    Return True iff all non-zero vectors lie inside an open wedge
    of width strictly less than pi.

    Notes:
    - Zero vectors are ignored.
    - Integer arithmetic is used throughout.
    """
    # Remove zero vectors and keep exact 2D tuple type
    w: List[Vector2] = [(int(v[0]), int(v[1])) for v in vectors if not _is_zero_vec(v)]

    if len(w) <= 1:
        return True

    # Pick two non-zero seeds
    i = 0
    while i < len(w) and _is_zero_vec(w[i]):
        i += 1
    if i == len(w):
        return True

    j = i + 1
    while j < len(w) and _is_zero_vec(w[j]):
        j += 1
    if j == len(w):
        return True

    u = w[i]
    v = w[j]
    c = _cross(u, v)

    if c == 0:
        # Collinear case: opposite directions would require width = pi
        if _dot(u, v) < 0:
            return False
        vmin, vmax = u, v
    elif c > 0:
        # v is counterclockwise from u
        vmin, vmax = u, v
    else:
        # v is clockwise from u, so swap
        vmin, vmax = v, u

    # Maintain a counterclockwise wedge (vmin -> vmax) with width < pi
    for k in range(j + 1, len(w)):
        wk = w[k]
        cmin = _cross(vmin, wk)
        cmax = _cross(vmax, wk)

        # Outside on both sides -> requires wedge >= pi
        if cmin <= 0 and cmax >= 0:
            return False

        # Strictly counterclockwise of both -> expand vmax
        if cmin > 0 and cmax > 0:
            vmax = wk
            continue

        # Strictly clockwise of both -> expand vmin backward
        if cmin < 0 and cmax < 0:
            vmin = wk
            continue

        # Otherwise already inside current wedge
    return True


def _directed_edge_vectors(
    points: np.ndarray,
    order: Sequence[int],
    dirs: Sequence[int],
    cycle: bool = False,
) -> List[Vector2]:
    """
    Build directed edge vectors (tail -> head) for the polyline defined by:
    - points: coordinates array of shape (n, 2)
    - order: vertex order / permutation
    - dirs: per-edge directions (+1 forward, -1 backward)

    Returns:
        list of integer (dx, dy) vectors
    """
    p = np.asarray(points, dtype=np.int64)
    edges = edges_for_order(order, cycle=cycle)

    if len(edges) != len(dirs):
        raise ValueError(
            f"Length mismatch: got {len(edges)} edges but {len(dirs)} directions."
        )

    vecs: List[Vector2] = []

    for (u, v), d in zip(edges, dirs):
        tail, head = (u, v) if int(d) > 0 else (v, u)
        dx = int(p[head, 0]) - int(p[tail, 0])
        dy = int(p[head, 1]) - int(p[tail, 1])
        vecs.append((dx, dy))

    return vecs