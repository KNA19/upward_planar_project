from __future__ import annotations

from itertools import permutations, product
from typing import Iterator, Sequence

import numpy as np


def canonical_path_orders(n: int) -> Iterator[tuple[int, ...]]:
    """
    Generate one representative for each undirected embedded path.

    The path order

        (v0, v1, ..., v_{n-1})

    and its reverse

        (v_{n-1}, ..., v1, v0)

    represent the same undirected path. We keep only the representative
    whose first endpoint index is smaller than its last endpoint index.

    For n = 0 or n = 1, there is only one path order.
    """
    if n <= 1:
        yield tuple(range(n))
        return

    for order in permutations(range(n)):
        if order[0] < order[-1]:
            yield order


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


def on_segment(a: Sequence[int], b: Sequence[int], c: Sequence[int]) -> bool:
    """
    Return True iff point b lies on the closed segment ac.

    Assumes a, b, c are collinear.
    """
    return (
        min(int(a[0]), int(c[0])) <= int(b[0]) <= max(int(a[0]), int(c[0]))
        and min(int(a[1]), int(c[1])) <= int(b[1]) <= max(int(a[1]), int(c[1]))
    )


def segments_properly_intersect(
    p: Sequence[int],
    q: Sequence[int],
    r: Sequence[int],
    s: Sequence[int],
) -> bool:
    """
    Return True iff segments pq and rs cross properly in their interiors.

    Endpoint touching and collinear overlap are not counted as proper crossings.
    """
    o1 = orient(p, q, r)
    o2 = orient(p, q, s)
    o3 = orient(r, s, p)
    o4 = orient(r, s, q)

    return (o1 * o2 < 0) and (o3 * o4 < 0)


def segments_intersect(
    p: Sequence[int],
    q: Sequence[int],
    r: Sequence[int],
    s: Sequence[int],
) -> bool:
    """
    Return True iff the closed segments pq and rs intersect.

    This includes:
    - proper crossings,
    - endpoint touching,
    - collinear overlap.

    For a straight-line planar path, non-adjacent edges should not intersect.
    """
    o1 = orient(p, q, r)
    o2 = orient(p, q, s)
    o3 = orient(r, s, p)
    o4 = orient(r, s, q)

    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    if o1 == 0 and on_segment(p, r, q):
        return True

    if o2 == 0 and on_segment(p, s, q):
        return True

    if o3 == 0 and on_segment(r, p, s):
        return True

    if o4 == 0 and on_segment(r, q, s):
        return True

    return False


def edges_for_order(order: Sequence[int], cycle: bool = False) -> list[tuple[int, int]]:
    """
    Given a vertex order, return the undirected edge list.

    Open path:
        (v0, v1), (v1, v2), ..., (v_{n-2}, v_{n-1})

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
    Legacy helper.

    All orientation patterns for the edges of an order.

    Open path:
        number of edges = n - 1
        number of patterns = 2^(n-1)

    Cycle:
        number of edges = n
        number of patterns = 2^n

    Each pattern is a tuple of +1 / -1.

    NOTE:
        The Chapter 6-aligned dataset builder should not use this function.
        It should instead generate undirected plane paths first and then compute
        upward orientations for each plane path.
    """
    m = n if cycle else max(0, n - 1)
    return list(product((+1, -1), repeat=m))


def embedding_is_planar(
    points: np.ndarray,
    order: Sequence[int],
    cycle: bool = False,
) -> bool:
    """
    Geometric planarity test for a polyline defined by 'order' over 'points'.

    Adjacent edges are ignored because they meet at a shared endpoint.

    For cycles, the first and last edges are also treated as adjacent.

    For non-adjacent edges, any intersection is rejected:
    - proper crossing,
    - endpoint touching,
    - collinear overlap.

    This is slightly stricter and safer than checking only proper crossings.
    For general-position order-type data, it gives the same result as the
    proper-crossing test.
    """
    p = np.asarray(points, dtype=np.int64)
    edges = edges_for_order(order, cycle=cycle)
    m = len(edges)

    for i in range(m):
        for j in range(i + 1, m):
            # Adjacent edges in a path/cycle are allowed to meet.
            if j == i + 1:
                continue

            # In a cycle, first and last edges are adjacent.
            if cycle and i == 0 and j == m - 1:
                continue

            a, b = edges[i]
            c, d = edges[j]

            if segments_intersect(p[a], p[b], p[c], p[d]):
                return False

    return True