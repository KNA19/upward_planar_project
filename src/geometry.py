from __future__ import annotations

from itertools import product
from typing import Iterable, Sequence

import numpy as np


def orient(a: Sequence[int], b: Sequence[int], c: Sequence[int]) -> int:
    """
    Signed orientation / 2D cross product:
    > 0 : counterclockwise
    < 0 : clockwise
    = 0 : collinear
    """
    return (int(b[0]) - int(a[0])) * (int(c[1]) - int(a[1])) - (
        int(b[1]) - int(a[1])
    ) * (int(c[0]) - int(a[0]))


def segments_properly_intersect(
    p: Sequence[int],
    q: Sequence[int],
    r: Sequence[int],
    s: Sequence[int],
) -> bool:
    """
    True iff segments pq and rs cross properly in their interiors.
    Endpoint touching / collinear overlap is not counted as a proper crossing.
    """
    o1 = orient(p, q, r)
    o2 = orient(p, q, s)
    o3 = orient(r, s, p)
    o4 = orient(r, s, q)

    return ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0))


def edges_for_order(order: Sequence[int], cycle: bool = False) -> list[tuple[int, int]]:
    """
    Given a vertex order, return the edge list:

    Open path:
        (v0,v1), (v1,v2), ..., (v_{n-2}, v_{n-1})

    Cycle:
        same as above plus (v_{n-1}, v0)
    """
    m = len(order)
    edges = [(int(order[i]), int(order[i + 1])) for i in range(m - 1)]

    if cycle and m >= 2:
        edges.append((int(order[-1]), int(order[0])))

    return edges


def oriented_paths(n: int, cycle: bool = False) -> list[tuple[int, ...]]:
    """
    All orientation patterns for the edges of the order.

    Open path:
        number of edges = n - 1
        number of patterns = 2^(n-1)

    Cycle:
        number of edges = n
        number of patterns = 2^n

    Each pattern is a tuple of +1 / -1.
    """
    m = n if cycle else (n - 1)
    return list(product((+1, -1), repeat=m))


def embedding_is_planar(
    points: np.ndarray,
    order: Sequence[int],
    cycle: bool = False,
) -> bool:
    """
    Geometric planarity test for a polyline defined by 'order' over 'points'.

    Adjacent edges are ignored since they meet at a shared endpoint.
    For cycles, the first and last edges are also treated as adjacent.
    """
    p = np.asarray(points, dtype=np.int64)
    edges = edges_for_order(order, cycle=cycle)
    m = len(edges)

    for i in range(m):
        for j in range(i + 1, m):
            # Adjacent edges in a path/cycle are allowed to meet
            if j == i + 1:
                continue

            # In a cycle, first and last edges are adjacent
            if cycle and i == 0 and j == m - 1:
                continue

            a, b = edges[i]
            c, d = edges[j]

            # Shared endpoint is not a proper crossing
            if len({a, b, c, d}) < 4:
                continue

            if segments_properly_intersect(p[a], p[b], p[c], p[d]):
                return False

    return True