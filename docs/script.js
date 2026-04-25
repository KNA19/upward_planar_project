"use strict";

// =========================================================
// Global state
// =========================================================

const datasetCache = new Map();

let currentDataset = null;
let currentSet = null;
let currentPath = null;
let currentOrientation = null;

// =========================================================
// DOM helpers
// =========================================================

function $(id) {
  return document.getElementById(id);
}

function setText(id, value) {
  const el = $(id);
  if (el) {
    el.textContent = value;
  }
}

function clearSelect(selectEl) {
  selectEl.innerHTML = "";
}

function addOption(selectEl, value, label) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  selectEl.appendChild(option);
}

// =========================================================
// Data loading
// =========================================================

async function loadDataset(n) {
  const key = String(n);

  if (datasetCache.has(key)) {
    return datasetCache.get(key);
  }

  const url = `data/n${n}_upward.json`;

  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Could not load ${url}`);
  }

  const dataset = await response.json();

  datasetCache.set(key, dataset);

  return dataset;
}

// =========================================================
// Selection population
// =========================================================

function populateSetSelect(dataset) {
  const setSelect = $("setSelect");
  clearSelect(setSelect);

  dataset.sets.forEach((setRecord) => {
    const setId = Number(setRecord.set_id);
    const numPaths = Number(
      setRecord.num_plane_undirected_paths ?? setRecord.plane_paths?.length ?? 0
    );
    const upwardTotal = Number(setRecord.total_upward_orientations ?? 0);

    addOption(
      setSelect,
      String(setId),
      `Set ${setId} | plane paths=${numPaths} | upward orientations=${upwardTotal}`
    );
  });
}

function populatePathSelect(setRecord) {
  const pathSelect = $("pathSelect");
  clearSelect(pathSelect);

  const planePaths = setRecord.plane_paths ?? [];

  planePaths.forEach((pathRecord) => {
    const pathId = Number(pathRecord.path_id);
    const upwardCount = Number(pathRecord.upward_count ?? 0);
    const order = pathRecord.order ?? [];

    addOption(
      pathSelect,
      String(pathId),
      `Path ${pathId} | upward=${upwardCount} | order=(${order.join(", ")})`
    );
  });
}

function populateOrientationSelect(pathRecord) {
  const orientationSelect = $("orientationSelect");
  clearSelect(orientationSelect);

  const orientations = pathRecord.upward_orientations ?? [];

  orientations.forEach((orientationRecord) => {
    const orientationId = Number(orientationRecord.orientation_id);
    const dirs = orientationRecord.dirs ?? "";

    addOption(
      orientationSelect,
      String(orientationId),
      `Orientation ${orientationId} [${dirs}]`
    );
  });
}

// =========================================================
// Lookup helpers
// =========================================================

function getSetById(dataset, setId) {
  return dataset.sets.find((setRecord) => Number(setRecord.set_id) === Number(setId));
}

function getPathById(setRecord, pathId) {
  return setRecord.plane_paths.find(
    (pathRecord) => Number(pathRecord.path_id) === Number(pathId)
  );
}

function getOrientationById(pathRecord, orientationId) {
  return pathRecord.upward_orientations.find(
    (orientationRecord) =>
      Number(orientationRecord.orientation_id) === Number(orientationId)
  );
}

// =========================================================
// Geometry helpers
// =========================================================

function rotatePoints(points, angle) {
  if (!points || points.length === 0) {
    return [];
  }

  const cx =
    points.reduce((sum, p) => sum + Number(p[0]), 0) / points.length;
  const cy =
    points.reduce((sum, p) => sum + Number(p[1]), 0) / points.length;

  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);

  return points.map((p) => {
    const x = Number(p[0]) - cx;
    const y = Number(p[1]) - cy;

    const xr = x * cosA - y * sinA;
    const yr = x * sinA + y * cosA;

    return [xr + cx, yr + cy];
  });
}

function computeDirectedEdges(order, dirs, cycle = false) {
  const edges = [];

  const expected = cycle ? order.length : Math.max(0, order.length - 1);

  if (dirs.length !== expected) {
    throw new Error(
      `Direction string length mismatch: got ${dirs.length}, expected ${expected}`
    );
  }

  for (let i = 0; i < dirs.length; i++) {
    const u = Number(order[i]);
    const v = Number(order[(i + 1) % order.length]);

    if (dirs[i] === "+") {
      edges.push([u, v]);
    } else if (dirs[i] === "-") {
      edges.push([v, u]);
    } else {
      throw new Error(`Invalid direction character: ${dirs[i]}`);
    }
  }

  return edges;
}

// =========================================================
// Canvas drawing helpers
// =========================================================

function getBounds(points) {
  const xs = points.map((p) => Number(p[0]));
  const ys = points.map((p) => Number(p[1]));

  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys),
  };
}

function makeTransform(points, canvasWidth, canvasHeight) {
  const bounds = getBounds(points);

  const padding = 55;

  const width = Math.max(bounds.maxX - bounds.minX, 1);
  const height = Math.max(bounds.maxY - bounds.minY, 1);

  const scale = Math.min(
    (canvasWidth - 2 * padding) / width,
    (canvasHeight - 2 * padding) / height
  );

  const centerX = (bounds.minX + bounds.maxX) / 2;
  const centerY = (bounds.minY + bounds.maxY) / 2;

  return function transform(point) {
    const x = Number(point[0]);
    const y = Number(point[1]);

    const screenX = canvasWidth / 2 + (x - centerX) * scale;

    // Canvas y-axis points downward, so invert the mathematical y-axis.
    const screenY = canvasHeight / 2 - (y - centerY) * scale;

    return [screenX, screenY];
  };
}

function drawArrow(ctx, x1, y1, x2, y2, color = "#1f2937") {
  const headLength = 12;
  const angle = Math.atan2(y2 - y1, x2 - x1);

  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2.3;

  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(
    x2 - headLength * Math.cos(angle - Math.PI / 6),
    y2 - headLength * Math.sin(angle - Math.PI / 6)
  );
  ctx.lineTo(
    x2 - headLength * Math.cos(angle + Math.PI / 6),
    y2 - headLength * Math.sin(angle + Math.PI / 6)
  );
  ctx.closePath();
  ctx.fill();
}

function drawGrid(ctx, width, height) {
  ctx.strokeStyle = "#eef2f7";
  ctx.lineWidth = 1;

  const step = 40;

  for (let x = 0; x <= width; x += step) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }

  for (let y = 0; y <= height; y += step) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
}

function drawUpArrow(ctx, points, transform, angle, label) {
  if (!points || points.length === 0) {
    return;
  }

  const bounds = getBounds(points);
  const span = Math.max(
    bounds.maxX - bounds.minX,
    bounds.maxY - bounds.minY,
    1
  );

  const start = [
    bounds.minX + 0.12 * span,
    bounds.minY + 0.12 * span,
  ];

  const length = 0.18 * span;

  const end = [
    start[0] + length * Math.cos(angle),
    start[1] + length * Math.sin(angle),
  ];

  const [x1, y1] = transform(start);
  const [x2, y2] = transform(end);

  drawArrow(ctx, x1, y1, x2, y2, "#2563eb");

  ctx.fillStyle = "#1e40af";
  ctx.font = "bold 13px system-ui, sans-serif";
  ctx.fillText(label, x2 + 6, y2 - 6);
}

function drawEmbedding() {
  const canvas = $("embeddingCanvas");
  const ctx = canvas.getContext("2d");

  const width = canvas.width;
  const height = canvas.height;

  ctx.clearRect(0, 0, width, height);

  drawGrid(ctx, width, height);

  if (!currentSet || !currentPath || !currentOrientation) {
    ctx.fillStyle = "#64748b";
    ctx.font = "16px system-ui, sans-serif";
    ctx.fillText("No embedding selected.", 32, 48);
    return;
  }

  const originalPoints = currentSet.points;
  const order = currentPath.order;
  const dirs = currentOrientation.dirs;

  const displayMode = $("displayMode").value;

  const rotationAngle = Number(currentOrientation.rotation_angle ?? 0);
  const upDirectionAngle = Number(
    currentOrientation.up_direction_angle ?? Math.PI / 2
  );

  const cycle = Boolean(currentDataset?.cycle ?? false);

  let displayPoints;
  let displayedUpAngle;
  let upLabel;

  if (displayMode === "rotated") {
    displayPoints = rotatePoints(originalPoints, rotationAngle);
    displayedUpAngle = Math.PI / 2;
    upLabel = "up";
  } else {
    displayPoints = originalPoints.map((p) => [Number(p[0]), Number(p[1])]);
    displayedUpAngle = upDirectionAngle;
    upLabel = "witness up";
  }

  const transform = makeTransform(displayPoints, width, height);

  // Draw undirected polyline first.
  ctx.strokeStyle = "#94a3b8";
  ctx.lineWidth = 1.4;
  ctx.setLineDash([6, 5]);

  ctx.beginPath();

  order.forEach((pointIndex, i) => {
    const [x, y] = transform(displayPoints[Number(pointIndex)]);

    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });

  if (cycle && order.length >= 2) {
    const [x0, y0] = transform(displayPoints[Number(order[0])]);
    ctx.lineTo(x0, y0);
  }

  ctx.stroke();
  ctx.setLineDash([]);

  // Draw directed edges.
  const directedEdges = computeDirectedEdges(order, dirs, cycle);

  directedEdges.forEach(([tail, head], i) => {
    const [x1, y1] = transform(displayPoints[tail]);
    const [x2, y2] = transform(displayPoints[head]);

    drawArrow(ctx, x1, y1, x2, y2, "#111827");

    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2;

    ctx.fillStyle = "#334155";
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText(`e${i + 1}`, mx + 4, my - 4);
  });

  // Draw points and labels.
  displayPoints.forEach((p, idx) => {
    const [x, y] = transform(p);

    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "#0f172a";
    ctx.lineWidth = 2;

    ctx.beginPath();
    ctx.arc(x, y, 6, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#0f172a";
    ctx.font = "bold 12px system-ui, sans-serif";
    ctx.fillText(String(idx), x + 8, y - 8);
  });

  drawUpArrow(ctx, displayPoints, transform, displayedUpAngle, upLabel);
}

// =========================================================
// Details panel
// =========================================================

function updateDetailsPanel() {
  if (!currentPath || !currentOrientation) {
    setText("orderOutput", "No path selected.");
    setText("dirsOutput", "No orientation selected.");
    setText("rotationOutput", "No orientation selected.");
    setText("upDirectionOutput", "No orientation selected.");
    return;
  }

  setText("orderOutput", JSON.stringify(currentPath.order));
  setText("dirsOutput", currentOrientation.dirs);
  setText(
    "rotationOutput",
    Number(currentOrientation.rotation_angle ?? 0).toFixed(8)
  );
  setText(
    "upDirectionOutput",
    Number(currentOrientation.up_direction_angle ?? 0).toFixed(8)
  );
}

// =========================================================
// State update flow
// =========================================================

async function handleDatasetChange() {
  const n = $("nSelect").value;

  setText("orderOutput", "Loading dataset...");
  setText("dirsOutput", "Loading dataset...");
  setText("rotationOutput", "Loading dataset...");
  setText("upDirectionOutput", "Loading dataset...");

  currentDataset = await loadDataset(n);

  if (!currentDataset.sets || !Array.isArray(currentDataset.sets)) {
    throw new Error("Invalid dataset: missing sets array.");
  }

  if (
    currentDataset.sets.length > 0 &&
    !("plane_paths" in currentDataset.sets[0])
  ) {
    throw new Error(
      "This dataset uses the old JSON structure. Rebuild outputs using the updated dataset_builder.py."
    );
  }

  populateSetSelect(currentDataset);

  handleSetChange();
}

function handleSetChange() {
  const setId = $("setSelect").value;

  currentSet = getSetById(currentDataset, setId);

  populatePathSelect(currentSet);

  handlePathChange();
}

function handlePathChange() {
  const pathId = $("pathSelect").value;

  currentPath = getPathById(currentSet, pathId);

  populateOrientationSelect(currentPath);

  handleOrientationChange();
}

function handleOrientationChange() {
  const orientationId = $("orientationSelect").value;

  if (!orientationId) {
    currentOrientation = null;
  } else {
    currentOrientation = getOrientationById(currentPath, orientationId);
  }

  updateDetailsPanel();
  drawEmbedding();
}

function handleDisplayModeChange() {
  drawEmbedding();
}

// =========================================================
// Initialization
// =========================================================

async function init() {
  $("nSelect").addEventListener("change", () => {
    handleDatasetChange().catch(showError);
  });

  $("setSelect").addEventListener("change", () => {
    try {
      handleSetChange();
    } catch (err) {
      showError(err);
    }
  });

  $("pathSelect").addEventListener("change", () => {
    try {
      handlePathChange();
    } catch (err) {
      showError(err);
    }
  });

  $("orientationSelect").addEventListener("change", () => {
    try {
      handleOrientationChange();
    } catch (err) {
      showError(err);
    }
  });

  $("displayMode").addEventListener("change", () => {
    try {
      handleDisplayModeChange();
    } catch (err) {
      showError(err);
    }
  });

  await handleDatasetChange();
}

function showError(err) {
  console.error(err);

  const message = err instanceof Error ? err.message : String(err);

  setText("orderOutput", `Error: ${message}`);
  setText("dirsOutput", "");
  setText("rotationOutput", "");
  setText("upDirectionOutput", "");

  const canvas = $("embeddingCanvas");
  const ctx = canvas.getContext("2d");

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "#b91c1c";
  ctx.font = "16px system-ui, sans-serif";
  ctx.fillText("Error", 32, 48);

  ctx.fillStyle = "#7f1d1d";
  ctx.font = "13px system-ui, sans-serif";
  wrapCanvasText(ctx, message, 32, 78, canvas.width - 64, 20);
}

function wrapCanvasText(ctx, text, x, y, maxWidth, lineHeight) {
  const words = text.split(" ");
  let line = "";

  for (const word of words) {
    const testLine = line + word + " ";
    const metrics = ctx.measureText(testLine);

    if (metrics.width > maxWidth && line !== "") {
      ctx.fillText(line, x, y);
      line = word + " ";
      y += lineHeight;
    } else {
      line = testLine;
    }
  }

  ctx.fillText(line, x, y);
}

window.addEventListener("DOMContentLoaded", () => {
  init().catch(showError);
});