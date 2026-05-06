from __future__ import annotations

import json
import math
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
def load_json(path: str, file_mtime: float) -> dict[str, Any]:
    """
    Load JSON with file_mtime included in the cache key so Streamlit reloads
    after outputs are rebuilt.
    """
    _ = file_mtime

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


def get_plane_path_by_id(set_record: dict[str, Any], path_id: int) -> dict[str, Any]:
    for path_record in set_record["plane_paths"]:
        if int(path_record["path_id"]) == int(path_id):
            return path_record

    raise ValueError(f"Plane path ID {path_id} not found.")


def get_orientation_by_id(
    path_record: dict[str, Any],
    orientation_id: int,
) -> dict[str, Any]:
    for orientation_record in path_record["upward_orientations"]:
        if int(orientation_record["orientation_id"]) == int(orientation_id):
            return orientation_record

    raise ValueError(f"Orientation ID {orientation_id} not found.")


# =========================================================
# Drawing helpers
# =========================================================

def compute_directed_edges(
    order: list[int] | tuple[int, ...],
    dirs: str,
    *,
    cycle: bool = False,
) -> list[tuple[int, int]]:
    """
    Given:
      order = [v0, v1, v2, ...]
      dirs  = '++-+-'

    return directed edges as (tail, head) pairs.

    If dirs[i] == '+', edge goes order[i] -> order[i+1].
    If dirs[i] == '-', edge goes order[i+1] -> order[i].
    """
    order_tuple = tuple(int(v) for v in order)

    expected_edges = len(order_tuple) if cycle else max(0, len(order_tuple) - 1)

    if len(dirs) != expected_edges:
        raise ValueError(
            f"Direction string length mismatch: got {len(dirs)}, "
            f"expected {expected_edges}."
        )

    edges: list[tuple[int, int]] = []

    for i, sign in enumerate(dirs):
        u = order_tuple[i]
        v = order_tuple[(i + 1) % len(order_tuple)]

        if sign == "+":
            edges.append((u, v))
        elif sign == "-":
            edges.append((v, u))
        else:
            raise ValueError(f"Invalid direction character: {sign}")

    return edges


def rotate_points(
    points: list[list[int]] | list[list[float]],
    angle: float,
) -> list[tuple[float, float]]:
    """
    Rotate all points around their centroid.
    """
    if not points:
        return []

    cx = sum(float(p[0]) for p in points) / len(points)
    cy = sum(float(p[1]) for p in points) / len(points)

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    rotated: list[tuple[float, float]] = []

    for x_raw, y_raw in points:
        x = float(x_raw) - cx
        y = float(y_raw) - cy

        xr = x * cos_a - y * sin_a
        yr = x * sin_a + y * cos_a

        rotated.append((xr + cx, yr + cy))

    return rotated


def point_label_offset(points: list[tuple[float, float]]) -> float:
    """
    Compute a small label offset based on the drawing size.
    """
    if not points:
        return 0.1

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    return 0.025 * span


def draw_up_direction_arrow(
    ax: Any,
    points: list[tuple[float, float]],
    angle: float,
) -> None:
    """
    Draw a small witness-up direction arrow.
    """
    if not points:
        return

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    span = max(max_x - min_x, max_y - min_y, 1.0)

    start_x = min_x + 0.12 * span
    start_y = min_y + 0.12 * span

    length = 0.18 * span
    end_x = start_x + length * math.cos(angle)
    end_y = start_y + length * math.sin(angle)

    ax.annotate(
        "",
        xy=(end_x, end_y),
        xytext=(start_x, start_y),
        arrowprops=dict(arrowstyle="->", lw=2),
    )

    ax.text(end_x, end_y, "up", fontsize=10)


def draw_embedding(
    points: list[list[int]],
    order: list[int] | tuple[int, ...],
    dirs: str,
    *,
    set_id: int,
    path_id: int,
    orientation_id: int,
    cycle: bool = False,
    display_mode: str = "rotated",
    rotation_angle: float = 0.0,
    up_direction_angle: float = math.pi / 2.0,
) -> Figure:
    """
    Draw the point set and selected directed path.

    display_mode:
        "rotated"  -> rotate drawing so the witness direction becomes vertical upward.
        "original" -> keep original order-type coordinates and draw witness-up arrow.
    """
    rotate_for_display = display_mode == "rotated"

    if rotate_for_display:
        display_points = rotate_points(points, rotation_angle)
        displayed_up_angle = math.pi / 2.0
    else:
        display_points = [(float(p[0]), float(p[1])) for p in points]
        displayed_up_angle = up_direction_angle

    fig, ax = plt.subplots(figsize=(7, 7))

    xs = [p[0] for p in display_points]
    ys = [p[1] for p in display_points]

    ax.scatter(xs, ys, s=80)

    offset = point_label_offset(display_points)

    for idx, (x, y) in enumerate(display_points):
        ax.text(x + offset, y + offset, str(idx), fontsize=10)

    order_tuple = tuple(int(v) for v in order)

    poly_x = [display_points[i][0] for i in order_tuple]
    poly_y = [display_points[i][1] for i in order_tuple]

    if cycle and len(order_tuple) >= 2:
        poly_x.append(display_points[order_tuple[0]][0])
        poly_y.append(display_points[order_tuple[0]][1])

    ax.plot(poly_x, poly_y, linestyle="--", linewidth=1)

    directed_edges = compute_directed_edges(order_tuple, dirs, cycle=cycle)

    for step_idx, (tail, head) in enumerate(directed_edges, start=1):
        x1, y1 = display_points[tail]
        x2, y2 = display_points[head]

        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=2),
        )

        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        ax.text(mx, my, f"e{step_idx}", fontsize=9)

    draw_up_direction_arrow(ax, display_points, displayed_up_angle)

    view_text = (
        "rotated upward witness view"
        if rotate_for_display
        else "original coordinate view"
    )

    ax.set_title(
        f"Set #{set_id} | Plane path {path_id:02d} | "
        f"Orientation {orientation_id:02d} [{dirs}] | {view_text}"
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
dataset = load_json(str(dataset_path), dataset_path.stat().st_mtime)

if "sets" not in dataset:
    st.error("Invalid dataset file: missing 'sets'.")
    st.stop()

set_ids = [int(s["set_id"]) for s in dataset["sets"]]
selected_set_id = st.sidebar.selectbox("Choose set ID", set_ids)

selected_set = get_set_by_id(dataset, selected_set_id)

if "plane_paths" not in selected_set:
    st.error(
        "This output JSON uses the old structure with 'paths' and 'upward_perms'. "
        "Please rebuild the outputs using the updated Chapter 6-aligned dataset_builder.py."
    )
    st.stop()

plane_paths = selected_set["plane_paths"]

if not plane_paths:
    st.warning("This set has no plane paths stored.")
    st.stop()

path_options = [
    (
        int(path_record["path_id"]),
        f'Plane path {int(path_record["path_id"]):02d} '
        f'| upward_count={int(path_record.get("upward_count", 0))} '
        f'| order={tuple(path_record["order"])}'
    )
    for path_record in plane_paths
]

selected_path_display = st.sidebar.selectbox(
    "Choose plane path",
    options=path_options,
    format_func=lambda x: x[1],
)

selected_path_id = selected_path_display[0]
selected_path = get_plane_path_by_id(selected_set, selected_path_id)

upward_orientations = selected_path.get("upward_orientations", [])

if upward_orientations:
    orientation_options = [
        (
            int(orientation_record["orientation_id"]),
            f'Orientation {int(orientation_record["orientation_id"]):02d} '
            f'[{orientation_record["dirs"]}]'
        )
        for orientation_record in upward_orientations
    ]

    selected_orientation_display = st.sidebar.selectbox(
        "Choose upward orientation",
        options=orientation_options,
        format_func=lambda x: x[1],
    )

    selected_orientation_id = selected_orientation_display[0]
    selected_orientation = get_orientation_by_id(selected_path, selected_orientation_id)
else:
    selected_orientation_id = None
    selected_orientation = None

display_choice = st.sidebar.radio(
    "Display mode",
    options=[
        "Rotated upward witness view",
        "Original coordinate view with witness-up arrow",
    ],
    index=0,
)

display_mode = "rotated" if display_choice.startswith("Rotated") else "original"


# =========================================================
# Dataset summary
# =========================================================

st.subheader("Dataset Summary")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("n", int(dataset["n"]))
col2.metric("Number of sets", int(dataset.get("num_sets", len(dataset["sets"]))))
col3.metric("Cycle", str(bool(dataset.get("cycle", False))))
col4.metric(
    "Plane paths in set",
    int(selected_set.get("num_plane_undirected_paths", len(plane_paths))),
)
col5.metric(
    "Upward orientations",
    int(selected_set.get("total_upward_orientations", 0)),
)

# st.caption(f"Algorithm: {dataset.get('algorithm', '')}")
# st.caption(f"Path model: {dataset.get('path_model', '')}")
# st.caption(f"Source file: {dataset.get('source_file', '')}")

st.divider()

left_col, right_col = st.columns([1, 2])


# =========================================================
# Left column
# =========================================================

with left_col:
    st.subheader(f"Set #{selected_set_id}")

    st.write("**Points**")
    st.write(selected_set["points"])

    st.write("**Set summary**")
    st.write(
        {
            "num_plane_undirected_paths": int(
                selected_set.get("num_plane_undirected_paths", len(plane_paths))
            ),
            "total_upward_orientations": int(
                selected_set.get("total_upward_orientations", 0)
            ),
        }
    )

    st.write("**Plane path summary**")

    path_rows = []

    for path_record in plane_paths:
        path_rows.append(
            {
                "path_id": int(path_record["path_id"]),
                "order": tuple(path_record["order"]),
                "upward_count": int(path_record.get("upward_count", 0)),
            }
        )

    st.dataframe(path_rows, width="stretch", hide_index=True)

    if selected_orientation is not None:
        st.write("**Orientations for selected plane path**")

        orientation_rows = []

        for orientation_record in upward_orientations:
            orientation_rows.append(
                {
                    "orientation_id": int(orientation_record["orientation_id"]),
                    "dirs": orientation_record["dirs"],
                    "up_direction_angle": float(
                        orientation_record.get("up_direction_angle", 0.0)
                    ),
                    "rotation_angle": float(
                        orientation_record.get("rotation_angle", 0.0)
                    ),
                }
            )

        st.dataframe(orientation_rows, width="stretch", hide_index=True)


# =========================================================
# Right column
# =========================================================

with right_col:
    st.subheader("Embedding View")

    if selected_orientation is None:
        st.warning("This plane path has no upward orientations stored.")
    else:
        dirs = str(selected_orientation["dirs"])
        rotation_angle = float(selected_orientation.get("rotation_angle", 0.0))
        up_direction_angle = float(
            selected_orientation.get("up_direction_angle", math.pi / 2.0)
        )

        fig = draw_embedding(
            points=selected_set["points"],
            order=selected_path["order"],
            dirs=dirs,
            set_id=selected_set_id,
            path_id=selected_path_id,
            orientation_id=int(selected_orientation["orientation_id"]),
            cycle=bool(dataset.get("cycle", False)),
            display_mode=display_mode,
            rotation_angle=rotation_angle,
            up_direction_angle=up_direction_angle,
        )

        st.pyplot(fig)

        st.write("**Selected plane path**")
        st.write(
            {
                "path_id": int(selected_path["path_id"]),
                "order": tuple(selected_path["order"]),
                "upward_count": int(selected_path.get("upward_count", 0)),
            }
        )

        st.write("**Selected upward orientation**")
        st.write(
            {
                "orientation_id": int(selected_orientation["orientation_id"]),
                "dirs": selected_orientation["dirs"],
                "up_direction_angle": up_direction_angle,
                "rotation_angle": rotation_angle,
            }
        )

        st.write("**Direction string**")
        st.code(str(selected_orientation["dirs"]))

        st.write("**Plane path order**")
        st.code(str(tuple(selected_path["order"])))

st.divider()

st.subheader("All Upward Orientations for Selected Plane Path")

if not upward_orientations:
    st.info("No upward orientations for this plane path.")
else:
    all_orientation_rows = [
        {
            "orientation_id": int(orientation_record["orientation_id"]),
            "dirs": orientation_record["dirs"],
            "up_direction_angle": float(
                orientation_record.get("up_direction_angle", 0.0)
            ),
            "rotation_angle": float(
                orientation_record.get("rotation_angle", 0.0)
            ),
        }
        for orientation_record in upward_orientations
    ]

    st.dataframe(all_orientation_rows, width="stretch", hide_index=True)