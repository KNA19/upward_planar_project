from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.reader import _default_path, _infer_bits, iter_otypes_one_by_one
from src.geometry import canonical_path_orders, embedding_is_planar
from src.upward import upward_orientations_for_order


def build_upward_dataset(
    n: int,
    *,
    path: str | None = None,
    bits: int | None = None,
    root_dir: str | Path = ".",
    cycle: bool = False,
    start_set: int = 0,
    stop_set: int | None = None,
    limit_perms: int | None = None,
    omit_empty_paths: bool = False,
) -> dict[str, Any]:
    """
    Build the upward-embedding dataset following the Chapter 6 improved
    brute-force approach.

    Chapter 6 flow:
        1. Read one representative point set from the order-type database.
        2. Enumerate undirected path embeddings, one representative up to reversal.
        3. Keep only the plane undirected path embeddings.
        4. For each plane undirected path, compute the upward orientations
           induced by rotating the coordinate system.
    """
    actual_bits = _infer_bits(n, path, bits)

    if path is None:
        actual_path = _default_path(n, actual_bits, root_dir)
    else:
        actual_path = path

    dataset: dict[str, Any] = {
        "n": n,
        "source_file": str(actual_path),
        "bits": actual_bits,
        "cycle": cycle,
        "algorithm": "Brute Force on Samll Order Types",
        "path_model": "undirected plane paths, one representative up to reversal",
        "num_sets": 0,
        "sets": [],
    }

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

        plane_paths: list[dict[str, Any]] = []
        num_plane_undirected_paths = 0
        total_upward_orientations = 0

        for perm_idx, order in enumerate(canonical_path_orders(n)):
            if limit_perms is not None and perm_idx >= limit_perms:
                break

            if not embedding_is_planar(pts, order, cycle=cycle):
                continue

            num_plane_undirected_paths += 1

            upward_orients = upward_orientations_for_order(
                pts,
                order,
                cycle=cycle,
            )

            if omit_empty_paths and not upward_orients:
                continue

            path_record: dict[str, Any] = {
                "path_id": len(plane_paths) + 1,
                "order_id": perm_idx + 1,
                "order": list(order),
                "upward_count": len(upward_orients),
                "upward_orientations": upward_orients,
            }

            plane_paths.append(path_record)
            total_upward_orientations += len(upward_orients)

        set_record: dict[str, Any] = {
            "set_id": set_id,
            "points": pts.tolist(),
            "num_plane_undirected_paths": num_plane_undirected_paths,
            "total_upward_orientations": total_upward_orientations,
            "plane_paths": plane_paths,
        }

        dataset["sets"].append(set_record)

    dataset["num_sets"] = len(dataset["sets"])
    return dataset


def save_dataset_json(dataset: dict[str, Any], outfile: str | Path) -> None:
    out_path = Path(outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)


def load_dataset_json(infile: str | Path) -> dict[str, Any]:
    in_path = Path(infile)

    with in_path.open("r", encoding="utf-8") as f:
        return json.load(f)