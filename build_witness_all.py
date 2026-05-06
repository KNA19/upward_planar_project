from __future__ import annotations

from pathlib import Path

from src.witness_dataset_builder import (
    build_witness_dataset,
    save_witness_dataset_json,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_WITNESS_DIR = BASE_DIR / "outputs_witness"


configs = [
    {"n": 3, "path": DATA_DIR / "otypes03.b08", "bits": 8},
    {"n": 4, "path": DATA_DIR / "otypes04.b08", "bits": 8},
    {"n": 5, "path": DATA_DIR / "otypes05.b08", "bits": 8},
    {"n": 6, "path": DATA_DIR / "otypes06.b08", "bits": 8},
    {"n": 7, "path": DATA_DIR / "otypes07.b08", "bits": 8},
    {"n": 8, "path": DATA_DIR / "otypes08.b08", "bits": 8},
    {"n": 9, "path": DATA_DIR / "otypes09.b16", "bits": 16},
    {"n": 10, "path": DATA_DIR / "otypes10.b16", "bits": 16},
]


def main() -> None:
    OUTPUTS_WITNESS_DIR.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        n = int(cfg["n"])

        print("=" * 70)
        print(f"Building full witness dataset for n={n}...")
        print("=" * 70)

        dataset = build_witness_dataset(
            n=n,
            path=str(cfg["path"]),
            bits=int(cfg["bits"]),
            cycle=False,
        )

        outfile = OUTPUTS_WITNESS_DIR / f"n{n}_witness.json"
        save_witness_dataset_json(dataset, outfile)

        print(f"Saved {outfile}")
        print(
            f"Found {dataset['total_found_orientations']}/"
            f"{dataset['total_required_orientations']} orientations"
        )
        print(f"Complete: {dataset['complete']}")


if __name__ == "__main__":
    main()
