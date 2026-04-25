from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
DOCS_DATA_DIR = BASE_DIR / "docs" / "data"


N_VALUES = [3, 4, 5, 6, 7]


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 3)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def summarize_dataset(dataset: dict[str, Any], source_path: Path) -> dict[str, Any]:
    sets = dataset.get("sets", [])

    total_plane_undirected_paths = 0
    total_upward_orientations = 0

    per_set_summary: list[dict[str, Any]] = []

    for set_record in sets:
        set_id = int(set_record.get("set_id", len(per_set_summary)))

        plane_paths = set_record.get("plane_paths", [])

        num_plane_paths = int(
            set_record.get("num_plane_undirected_paths", len(plane_paths))
        )

        upward_count_for_set = int(
            set_record.get(
                "total_upward_orientations",
                sum(int(path.get("upward_count", 0)) for path in plane_paths),
            )
        )

        total_plane_undirected_paths += num_plane_paths
        total_upward_orientations += upward_count_for_set

        per_set_summary.append(
            {
                "set_id": set_id,
                "num_plane_undirected_paths": num_plane_paths,
                "total_upward_orientations": upward_count_for_set,
            }
        )

    summary: dict[str, Any] = {
        "n": int(dataset["n"]),
        "status": "complete",
        "source_dataset_file": source_path.name,
        "source_dataset_size_mb": file_size_mb(source_path),
        "num_sets": int(dataset.get("num_sets", len(sets))),
        "cycle": bool(dataset.get("cycle", False)),
        "algorithm": str(dataset.get("algorithm", "")),
        "path_model": str(dataset.get("path_model", "")),
        "total_plane_undirected_paths": total_plane_undirected_paths,
        "total_upward_orientations": total_upward_orientations,
        "sets": per_set_summary,
    }

    return summary


def main() -> None:
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for n in N_VALUES:
        source_path = OUTPUTS_DIR / f"n{n}_upward.json"
        summary_path = DOCS_DATA_DIR / f"n{n}_summary.json"

        if not source_path.exists():
            print(f"Skipping n={n}: missing {source_path}")
            continue

        print(f"Creating summary for n={n}...")

        dataset = load_json(source_path)
        summary = summarize_dataset(dataset, source_path)

        save_json(summary, summary_path)

        print(f"Saved {summary_path}")

    print("Done.")


if __name__ == "__main__":
    main()