from __future__ import annotations

from pathlib import Path

from src.dataset_builder import build_upward_dataset, save_dataset_json


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"


configs = [
    {"n": 3, "path": DATA_DIR / "otypes03.b08", "bits": 8},
    {"n": 4, "path": DATA_DIR / "otypes04.b08", "bits": 8},
    {"n": 5, "path": DATA_DIR / "otypes05.b08", "bits": 8},
    {"n": 6, "path": DATA_DIR / "otypes06.b08", "bits": 8},
    {"n": 7, "path": DATA_DIR / "otypes07.b08", "bits": 8}
]


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        n = int(cfg["n"])

        print(f"Building dataset for n={n}...")

        dataset = build_upward_dataset(
            n=n,
            path=str(cfg["path"]),
            bits=int(cfg["bits"]),
            cycle=False,
            omit_empty_paths=False,
        )

        outfile = OUTPUTS_DIR / f"n{n}_upward.json"
        save_dataset_json(dataset, outfile)

        print(f"Saved {outfile}")


if __name__ == "__main__":
    main()