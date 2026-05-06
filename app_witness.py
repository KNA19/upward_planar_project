from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_WITNESS_DIR = BASE_DIR / "outputs_witness"


def discover_witness_files() -> dict[int, Path]:
    files: dict[int, Path] = {}

    if not OUTPUTS_WITNESS_DIR.exists():
        return files

    for path in OUTPUTS_WITNESS_DIR.glob("n*_witness.json"):
        match = re.match(r"n(\d+)_witness\.json$", path.name)

        if match:
            files[int(match.group(1))] = path

    return dict(sorted(files.items()))


@st.cache_data
def load_json(path_str: str, mtime: float) -> dict[str, Any]:
    path = Path(path_str)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rotate_points(points: np.ndarray, angle: float) -> np.ndarray:
    center = points.mean(axis=0)
    shifted = points - center

    c = math.cos(angle)
    s = math.sin(angle)

    rotation = np.array([[c, -s], [s, c]], dtype=float)
    return shifted @ rotation.T + center


def point_label_offset(points: np.ndarray) -> float:
    if points.size == 0:
        return 0.1

    x_span = float(points[:, 0].max() - points[:, 0].min())
    y_span = float(points[:, 1].max() - points[:, 1].min())
    return 0.025 * max(x_span, y_span, 1.0)


def draw_witness(
    points: list[list[float]],
    directed_edges: list[list[int]],
    undirected_edges: list[list[int]],
    *,
    display_mode: str,
    rotation_angle: float,
    up_direction_angle: float,
) -> None:
    pts = np.asarray(points, dtype=float)

    if display_mode == "Rotated upward witness view":
        draw_pts = rotate_points(pts, rotation_angle)
    else:
        draw_pts = pts.copy()

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(draw_pts[:, 0], draw_pts[:, 1], s=70)

    offset = point_label_offset(draw_pts)

    for idx, (x, y) in enumerate(draw_pts):
        ax.text(x + offset, y + offset, str(idx), fontsize=11)

    for u, v in undirected_edges:
        x0, y0 = draw_pts[u]
        x1, y1 = draw_pts[v]
        ax.plot([x0, x1], [y0, y1], linestyle="--", linewidth=1, color="0.55")

    for tail, head in directed_edges:
        x0, y0 = draw_pts[tail]
        x1, y1 = draw_pts[head]

        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "->",
                "lw": 2,
                "connectionstyle": "arc3,rad=0",
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )

    xmin, xmax = float(draw_pts[:, 0].min()), float(draw_pts[:, 0].max())
    ymin, ymax = float(draw_pts[:, 1].min()), float(draw_pts[:, 1].max())

    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)

    pad_x = 0.20 * dx
    pad_y = 0.20 * dy

    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    if display_mode == "Original coordinate view with witness-up arrow":
        base_x = xmin - 0.10 * dx
        base_y = ymin - 0.10 * dy
        length = 0.25 * max(dx, dy)

        ux = math.cos(up_direction_angle)
        uy = math.sin(up_direction_angle)
        end_x = base_x + length * ux
        end_y = base_y + length * uy

        ax.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(base_x, base_y),
            arrowprops={"arrowstyle": "->", "lw": 2},
        )
        ax.text(end_x, end_y, "witness up", fontsize=10)
    else:
        base_x = xmin - 0.10 * dx
        base_y = ymin - 0.10 * dy
        length = 0.25 * max(dx, dy)
        end_x = base_x
        end_y = base_y + length

        ax.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(base_x, base_y),
            arrowprops={"arrowstyle": "->", "lw": 2},
        )
        ax.text(end_x, end_y, "up", fontsize=10)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)


def main() -> None:
    st.set_page_config(
        page_title="UPSE Full Witness Dataset",
        layout="wide",
    )

    st.title("UPSE Full Witness Dataset")

    st.markdown(
        """
        This app displays the full witness dataset.

        For each orientation pattern, the dataset stores only one upward
        plane-path witness.
        """
    )

    files = discover_witness_files()

    if not files:
        st.error(
            "No witness dataset files found. Run `python build_witness_all.py` first."
        )
        return

    n_values = list(files.keys())

    selected_n = st.sidebar.selectbox(
        "Choose n",
        n_values,
        format_func=lambda x: f"n = {x}",
    )

    path = files[selected_n]
    dataset = load_json(str(path), path.stat().st_mtime)

    st.sidebar.caption(f"Loaded: {path.name}")

    st.subheader(f"Dataset for n = {selected_n}")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Required orientations",
        dataset.get("total_required_orientations", 0),
    )
    col2.metric(
        "Found orientations",
        dataset.get("total_found_orientations", 0),
    )
    col3.metric(
        "Complete",
        "Yes" if dataset.get("complete") else "No",
    )
    col4.metric(
        "Sets checked",
        dataset.get("search_statistics", {}).get("sets_checked", 0),
    )

    witnesses = dataset.get("witnesses", [])

    if not witnesses:
        st.warning("No witnesses found in this dataset.")
        return

    witness_labels = [
        f"{w['orientation_id']}: dirs = {w['dirs']}, set_id = {w['set_id']}, order = {w['order']}"
        for w in witnesses
    ]

    selected_label = st.sidebar.selectbox("Choose orientation witness", witness_labels)
    selected_index = witness_labels.index(selected_label)
    witness = witnesses[selected_index]

    display_mode = st.sidebar.radio(
        "Display mode",
        [
            "Rotated upward witness view",
            "Original coordinate view with witness-up arrow",
        ],
    )

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("### Drawing")

        draw_witness(
            witness["points"],
            witness["directed_edges"],
            witness["undirected_edges"],
            display_mode=display_mode,
            rotation_angle=float(witness["rotation_angle"]),
            up_direction_angle=float(witness["up_direction_angle"]),
        )

    with right:
        st.markdown("### Witness details")

        st.write(f"**Orientation ID:** {witness['orientation_id']}")
        st.write(f"**Direction pattern:** `{witness['dirs']}`")
        st.write(f"**Set ID:** `{witness['set_id']}`")
        st.write(f"**Order ID:** `{witness['order_id']}`")
        st.write(f"**Path order:** `{witness['order']}`")
        st.write(f"**Directed edges:** `{witness['directed_edges']}`")
        st.write(f"**Rotation angle:** `{witness['rotation_angle']}`")
        st.write(f"**Witness up-direction angle:** `{witness['up_direction_angle']}`")

        with st.expander("Raw witness JSON"):
            st.json(witness)

    missing = dataset.get("missing_orientations", [])

    if missing:
        st.warning(f"Missing orientations: {missing}")

    with st.expander("Dataset metadata"):
        metadata = {
            key: value
            for key, value in dataset.items()
            if key not in {"witnesses"}
        }
        st.json(metadata)


if __name__ == "__main__":
    main()
