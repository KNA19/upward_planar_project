from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import streamlit as st


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"


# =========================================================
# Data loading helpers
# =========================================================

@st.cache_data
def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_output_files(outputs_dir: Path) -> dict[int, Path]:
    """
    Find files like:
      outputs/n4_upward.json
      outputs/n5_upward.json
      outputs/n6_upward.json
      outputs/n7_upward.json
    """
    result: dict[int, Path] = {}
    pattern = re.compile(r"^n(\d+)_upward\.json$")

    if not outputs_dir.exists():
        return result

    for file_path in outputs_dir.iterdir():
        if not file_path.is_file():
            continue

        match = pattern.match(file_path.name)
        if match:
            n_value = int(match.group(1))
            result[n_value] = file_path

    return dict(sorted(result.items()))


def get_set_by_id(dataset: dict[str, Any], set_id: int) -> dict[str, Any]:
    for set_record in dataset["sets"]:
        if int(set_record["set_id"]) == int(set_id):
            return set_record
    raise ValueError(f"Set ID {set_id} not found.")


def get_path_by_id(set_record: dict[str, Any], path_id: int) -> dict[str, Any]:
    for path_record in set_record["paths"]:
        if int(path_record["path_id"]) == int(path_id):
            return path_record
    raise ValueError(f"Path ID {path_id} not found.")


# =========================================================
# Drawing helpers
# =========================================================

def compute_directed_edges(
    perm: list[int] | tuple[int, ...],
    dirs: str,
) -> list[tuple[int, int]]:
    """
    Given:
      perm = [v0, v1, v2, ...]
      dirs = '++-+-'
    return directed edges as (tail, head) pairs.

    If dirs[i] == '+', edge goes perm[i] -> perm[i+1]
    If dirs[i] == '-', edge goes perm[i+1] -> perm[i]
    """
    edges: list[tuple[int, int]] = []

    for i, sign in enumerate(dirs):
        u = int(perm[i])
        v = int(perm[i + 1])

        if sign == "+":
            edges.append((u, v))
        else:
            edges.append((v, u))

    return edges


def draw_embedding(
    points: list[list[int]],
    perm: list[int] | tuple[int, ...],
    dirs: str,
    set_id: int,
    path_id: int,
) -> Figure:
    """
    Draw the point set and the selected directed path embedding.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    ax.scatter(xs, ys, s=80)

    for idx, (x, y) in enumerate(points):
        ax.text(x + 2, y + 2, str(idx), fontsize=10)

    poly_x = [points[i][0] for i in perm]
    poly_y = [points[i][1] for i in perm]
    ax.plot(poly_x, poly_y, linestyle="--", linewidth=1)

    directed_edges = compute_directed_edges(perm, dirs)
    for step_idx, (tail, head) in enumerate(directed_edges, start=1):
        x1, y1 = points[tail]
        x2, y2 = points[head]

        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=2),
        )

        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        ax.text(mx, my, f"e{step_idx}", fontsize=9)

    ax.set_title(
        f"Set #{set_id} | Path {path_id:02d} [{dirs}] | Permutation {tuple(perm)}"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    return fig


# =========================================================
# Streamlit app
# =========================================================

st.set_page_config(page_title="Upward Planar Embedding Explorer", layout="wide")

st.title("Upward Planar Embedding Explorer")

available_files = discover_output_files(OUTPUTS_DIR)

if not available_files:
    st.error(
        "No dataset JSON files found in the 'outputs' folder. "
        "Expected files like outputs/n4_upward.json, outputs/n5_upward.json, etc."
    )
    st.stop()

st.sidebar.header("Controls")

available_n = list(available_files.keys())
selected_n = st.sidebar.selectbox("Choose n", available_n)

dataset_path = available_files[selected_n]
dataset = load_json(str(dataset_path))

set_ids = [int(s["set_id"]) for s in dataset["sets"]]
selected_set_id = st.sidebar.selectbox("Choose set ID", set_ids)

selected_set = get_set_by_id(dataset, selected_set_id)

path_options = [
    (
        int(path_record["path_id"]),
        f'Path {int(path_record["path_id"]):02d} [{path_record["dirs"]}] '
        f'| upward_count={int(path_record.get("upward_count", len(path_record["upward_perms"])))}'
    )
    for path_record in selected_set["paths"]
]

selected_path_display = st.sidebar.selectbox(
    "Choose path",
    options=path_options,
    format_func=lambda x: x[1],
)

selected_path_id = selected_path_display[0]
selected_path = get_path_by_id(selected_set, selected_path_id)

upward_perms = selected_path["upward_perms"]

if upward_perms:
    perm_options = list(range(len(upward_perms)))
    selected_perm_index = st.sidebar.selectbox(
        "Choose upward permutation index",
        perm_options,
        format_func=lambda idx: f"{idx}: {tuple(upward_perms[idx])}",
    )
    selected_perm = upward_perms[selected_perm_index]
else:
    selected_perm_index = None
    selected_perm = None

st.subheader("Dataset Summary")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("n", int(dataset["n"]))
col2.metric("Number of sets", int(dataset.get("num_sets", len(dataset["sets"]))))
col3.metric("Cycle", str(bool(dataset.get("cycle", False))))
col4.metric("Orientation patterns", int(dataset.get("num_orientation_patterns", 0)))
col5.metric("Source file", str(dataset.get("source_file", "")))

st.divider()

left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader(f"Set #{selected_set_id}")

    st.write("**Points**")
    st.write(selected_set["points"])

    st.write("**Set summary**")
    st.write(
        {
            "num_planar_perms": int(selected_set.get("num_planar_perms", 0)),
            "num_paths": int(selected_set.get("num_paths", len(selected_set["paths"]))),
            "total_upward_embeddings": int(selected_set.get("total_upward_embeddings", 0)),
        }
    )

    st.write("**Path summary**")
    summary_rows = []
    for path_record in selected_set["paths"]:
        summary_rows.append(
            {
                "path_id": int(path_record["path_id"]),
                "dirs": path_record["dirs"],
                "upward_count": int(
                    path_record.get("upward_count", len(path_record["upward_perms"]))
                ),
            }
        )
    st.dataframe(summary_rows, use_container_width=True, hide_index=True)

with right_col:
    st.subheader("Embedding View")

    if selected_perm is None:
        st.warning("This path has no upward permutations stored.")
    else:
        fig = draw_embedding(
            points=selected_set["points"],
            perm=selected_perm,
            dirs=selected_path["dirs"],
            set_id=selected_set_id,
            path_id=selected_path_id,
        )
        st.pyplot(fig)

        st.write("**Selected path**")
        st.write(
            {
                "path_id": int(selected_path["path_id"]),
                "dirs": selected_path["dirs"],
                "upward_count": int(
                    selected_path.get("upward_count", len(selected_path["upward_perms"]))
                ),
            }
        )

        st.write("**Selected upward permutation**")
        st.code(str(tuple(selected_perm)))

st.divider()

st.subheader("All Upward Permutations for Selected Path")

if not upward_perms:
    st.info("No upward permutations for this path.")
else:
    perm_rows = [{"index": idx, "perm": tuple(perm)} for idx, perm in enumerate(upward_perms)]
    st.dataframe(perm_rows, use_container_width=True, hide_index=True)