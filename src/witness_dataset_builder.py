from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any, Sequence

from src.reader import _default_path, _infer_bits, iter_otypes_one_by_one
from src.geometry import canonical_path_orders, embedding_is_planar, edges_for_order
from src.upward import upward_orientations_for_order


def orientation_patterns(n: int, *, cycle: bool = False) -> list[str]:
    """
    Generate all edge-orientation patterns.

    Open path:
        number of edges = n - 1
        number of patterns = 2^(n-1)

    Example for n = 4:
        +++, ++-, +-+, +--, -++, -+-, --+, ---
    """
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    m = n if cycle else max(0, n - 1)

    return ["".join(chars) for chars in product("+-", repeat=m)]


def directed_edges_for_dirs(
    order: Sequence[int],
    dirs: str,
    *,
    cycle: bool = False,
) -> list[list[int]]:
    """
    Convert a path order and a direction string into directed edges.

    Meaning:
        + : order[i]   -> order[i+1]
        - : order[i+1] -> order[i]
    """
    edges = edges_for_order(order, cycle=cycle)

    if len(edges) != len(dirs):
        raise ValueError(
            f"Length mismatch: {len(edges)} edges but {len(dirs)} direction symbols."
        )

    directed_edges: list[list[int]] = []

    for (u, v), d in zip(edges, dirs):
        if d == "+":
            directed_edges.append([int(u), int(v)])
        elif d == "-":
            directed_edges.append([int(v), int(u)])
        else:
            raise ValueError(f"Invalid direction symbol: {d}")

    return directed_edges


def build_witness_dataset(
    n: int,
    *,
    path: str | None = None,
    bits: int | None = None,
    root_dir: str | Path = ".",
    cycle: bool = False,
    start_set: int = 0,
    stop_set: int | None = None,
    limit_perms: int | None = None,
) -> dict[str, Any]:
    """
    Build a full orientation-witness dataset.

    Goal:
        For every orientation pattern of a path with n vertices,
        store only one upward plane-path witness.

    This is not the full dataset. It does not store all sets, all plane paths,
    or all upward embeddings. It stores the first valid witness found for each
    orientation pattern.

    Search flow:
        1. Generate all orientation patterns of length n - 1.
        2. Iterate through representative point sets.
        3. Iterate through canonical undirected path orders.
        4. Keep only plane path embeddings.
        5. Compute upward orientations under rotation.
        6. Save the first witness for each orientation pattern.
        7. Stop when all orientation patterns have witnesses.
    """
    actual_bits = _infer_bits(n, path, bits)

    if path is None:
        actual_path = _default_path(n, actual_bits, root_dir)
    else:
        actual_path = path

    required_patterns = orientation_patterns(n, cycle=cycle)
    required_set = set(required_patterns)

    found: dict[str, dict[str, Any]] = {}

    sets_checked = 0
    path_orders_checked = 0
    plane_paths_checked = 0

    for local_idx, pts in enumerate(
        iter_otypes_one_by_one(
            n=n,
            path=actual_path,
            bits=actual_bits,
            root_dir=root_dir,
            mmap=True,
            start_set=start_set,
            stop_set=stop_set,
            cast_int64_for_geometry=True,
        )
    ):
        set_id = start_set + local_idx
        sets_checked += 1

        for perm_idx, order in enumerate(canonical_path_orders(n)):
            if limit_perms is not None and perm_idx >= limit_perms:
                break

            path_orders_checked += 1

            if not embedding_is_planar(pts, order, cycle=cycle):
                continue

            plane_paths_checked += 1

            upward_orients = upward_orientations_for_order(
                pts,
                order,
                cycle=cycle,
            )

            for upward_record in upward_orients:
                dirs = str(upward_record["dirs"])

                if dirs not in required_set:
                    continue

                if dirs in found:
                    continue

                found[dirs] = {
                    "orientation_id": required_patterns.index(dirs) + 1,
                    "dirs": dirs,
                    "set_id": int(set_id),
                    "points": pts.tolist(),
                    "order_id": int(perm_idx + 1),
                    "order": [int(x) for x in order],
                    "undirected_edges": [
                        [int(u), int(v)] for u, v in edges_for_order(order, cycle=cycle)
                    ],
                    "directed_edges": directed_edges_for_dirs(
                        order,
                        dirs,
                        cycle=cycle,
                    ),
                    "up_direction_angle": float(upward_record["up_direction_angle"]),
                    "rotation_angle": float(upward_record["rotation_angle"]),
                    "orientation_id_within_plane_path": int(
                        upward_record["orientation_id"]
                    ),
                }

                print(
                    f"n={n}: found {len(found)}/{len(required_patterns)} "
                    f"orientation witnesses; latest dirs={dirs}, set_id={set_id}"
                )

                if len(found) == len(required_patterns):
                    break

            if len(found) == len(required_patterns):
                break

        if len(found) == len(required_patterns):
            break

    witnesses = [found[dirs] for dirs in required_patterns if dirs in found]
    missing = [dirs for dirs in required_patterns if dirs not in found]

    dataset: dict[str, Any] = {
        "n": int(n),
        "bits": int(actual_bits),
        "cycle": bool(cycle),
        "dataset_type": "full orientation-witness dataset",
        "orientation_model": "all binary edge-direction patterns of length n-1",
        "upwardness_model": "upwardness under rotation of the coordinate system",
        "total_required_orientations": len(required_patterns),
        "total_found_orientations": len(witnesses),
        "complete": len(missing) == 0,
        "required_orientations": required_patterns,
        "missing_orientations": missing,
        "search_statistics": {
            "sets_checked": int(sets_checked),
            "path_orders_checked": int(path_orders_checked),
            "plane_paths_checked": int(plane_paths_checked),
        },
        "witnesses": witnesses,
    }

    return dataset


def save_witness_dataset_json(dataset: dict[str, Any], outfile: str | Path) -> None:
    out_path = Path(outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)


def load_witness_dataset_json(infile: str | Path) -> dict[str, Any]:
    in_path = Path(infile)

    with in_path.open("r", encoding="utf-8") as f:
        return json.load(f)
