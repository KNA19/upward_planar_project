from src.dataset_builder import build_upward_dataset, save_dataset_json

configs = [
    {"n": 3, "path": "data/otypes03.b08", "bits": 8},
    {"n": 4, "path": "data/otypes04.b08", "bits": 8},
    {"n": 5, "path": "data/otypes05.b08", "bits": 8},
    {"n": 6, "path": "data/otypes06.b08", "bits": 8},
    {"n": 7, "path": "data/otypes07.b08", "bits": 8},
]

for cfg in configs:
    n = cfg["n"]
    dataset = build_upward_dataset(
        n=n,
        path=cfg["path"],
        bits=cfg["bits"],
        cycle=False,
        omit_empty_paths=False,
    )
    outfile = f"outputs/n{n}_upward.json"
    save_dataset_json(dataset, outfile)
    print(f"Saved {outfile}")