from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path
from typing import Any

from src.reader import _default_path, _infer_bits, iter_otypes_one_by_one
from src.geometry import embedding_is_planar, oriented_paths
from src.upward import _directed_edge_vectors, is_in_acute_angular_sector


def dirs_to_string(dirs: tuple[int, ...]) -> str:
    return "".join("+" if d > 0 else "-" for d in dirs)


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
    actual_bits = _infer_bits(n, path, bits)

    if path is None:
        actual_path = _default_path(n, actual_bits, root_dir)
    else:
        actual_path = path

    patterns = oriented_paths(n, cycle=cycle)

    dataset: dict[str, Any] = {
        "n": n,
        "source_file": str(actual_path),
        "bits": actual_bits,
        "cycle": cycle,
        "num_orientation_patterns": len(patterns),
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

        planar_perms: list[tuple[int, ...]] = []
        for perm_idx, order in enumerate(permutations(range(n))):
            if limit_perms is not None and perm_idx >= limit_perms:
                break
            if embedding_is_planar(pts, order, cycle=cycle):
                planar_perms.append(order)

        set_record: dict[str, Any] = {
            "set_id": set_id,
            "points": pts.tolist(),
            "num_planar_perms": len(planar_perms),
            "num_paths": len(patterns),
            "total_upward_embeddings": 0,
            "paths": [],
        }

        for path_id, dirs in enumerate(patterns, start=1):
            upward_perms: list[list[int]] = []

            for order in planar_perms:
                vecs = _directed_edge_vectors(pts, order, dirs, cycle=cycle)
                if is_in_acute_angular_sector(vecs):
                    upward_perms.append(list(order))

            if omit_empty_paths and not upward_perms:
                continue

            path_record: dict[str, Any] = {
                "path_id": path_id,
                "dirs": dirs_to_string(dirs),
                "upward_count": len(upward_perms),
                "upward_perms": upward_perms,
            }

            set_record["paths"].append(path_record)
            set_record["total_upward_embeddings"] += len(upward_perms)

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