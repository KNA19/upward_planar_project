"use strict";

const datasetCache = new Map();

let currentDataset = null;
let currentWitness = null;

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

async function loadDataset(n) {
  const key = String(n);

  if (datasetCache.has(key)) {
    return datasetCache.get(key);
  }

  const url = `data/n${n}_witness.json`;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Could not load ${url}`);
  }

  const dataset = await response.json();
  datasetCache.set(key, dataset);

  return dataset;
}

function populateWitnessSelect(dataset) {
  const witnessSelect = $("witnessSelect");
  clearSelect(witnessSelect);

  const witnesses = Array.isArray(dataset.witnesses) ? dataset.witnesses : [];

  witnesses.forEach((witness, index) => {
    const order = Array.isArray(witness.order) ? witness.order : [];

    addOption(
      witnessSelect,
      String(index),
      `${witness.orientation_id}: dirs=${witness.dirs}, set=${witness.set_id}, order=(${order.join(", ")})`
    );
  });
}

function rotatePoints(points, angle) {
  if (!points || points.length === 0) {
    return [];
  }

  const cx = points.reduce((sum, p) => sum + Number(p[0]), 0) / points.length;
  const cy = points.reduce((sum, p) => sum + Number(p[1]), 0) / points.length;

  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);

  return points.map((p) => {
    const x = Number(p[0]) - cx;
    const y = Number(p[1]) - cy;

    return [
      x * cosA - y * sinA + cx,
      x * sinA + y * cosA + cy,
    ];
  });
}

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

    return [
      canvasWidth / 2 + (x - centerX) * scale,
      canvasHeight / 2 - (y - centerY) * scale,
    ];
  };
}

function drawArrow(ctx, x1, y1, x2, y2, color = "#111827") {
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

  for (let x = 0; x <= width; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }

  for (let y = 0; y <= height; y += 40) {
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
  const span = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY, 1);
  const start = [bounds.minX + 0.12 * span, bounds.minY + 0.12 * span];
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

  if (!currentWitness) {
    ctx.fillStyle = "#64748b";
    ctx.font = "16px system-ui, sans-serif";
    ctx.fillText("No witness selected.", 32, 48);
    return;
  }

  const displayMode = $("displayMode").value;
  const originalPoints = currentWitness.points;
  const rotationAngle = Number(currentWitness.rotation_angle ?? 0);
  const upDirectionAngle = Number(currentWitness.up_direction_angle ?? Math.PI / 2);

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

  ctx.strokeStyle = "#94a3b8";
  ctx.lineWidth = 1.4;
  ctx.setLineDash([6, 5]);

  const undirectedEdges = Array.isArray(currentWitness.undirected_edges)
    ? currentWitness.undirected_edges
    : [];

  for (const [u, v] of undirectedEdges) {
    const [x1, y1] = transform(displayPoints[Number(u)]);
    const [x2, y2] = transform(displayPoints[Number(v)]);

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }

  ctx.setLineDash([]);

  const directedEdges = Array.isArray(currentWitness.directed_edges)
    ? currentWitness.directed_edges
    : [];

  for (const [tail, head] of directedEdges) {
    const [x1, y1] = transform(displayPoints[Number(tail)]);
    const [x2, y2] = transform(displayPoints[Number(head)]);
    drawArrow(ctx, x1, y1, x2, y2);
  }

  displayPoints.forEach((point, index) => {
    const [x, y] = transform(point);

    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "#0f172a";
    ctx.lineWidth = 2;

    ctx.beginPath();
    ctx.arc(x, y, 6, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#0f172a";
    ctx.font = "bold 12px system-ui, sans-serif";
    ctx.fillText(String(index), x + 8, y - 8);
  });

  drawUpArrow(ctx, displayPoints, transform, displayedUpAngle, upLabel);
}

function updateDetailsPanel() {
  if (!currentWitness) {
    setText("orderOutput", "No witness selected.");
    setText("dirsOutput", "No witness selected.");
    setText("rotationOutput", "No witness selected.");
    setText("upDirectionOutput", "No witness selected.");
    return;
  }

  setText("orderOutput", JSON.stringify(currentWitness.order));
  setText("dirsOutput", currentWitness.dirs);
  setText("rotationOutput", Number(currentWitness.rotation_angle ?? 0).toFixed(8));
  setText(
    "upDirectionOutput",
    Number(currentWitness.up_direction_angle ?? 0).toFixed(8)
  );
}

async function handleDatasetChange() {
  const n = $("nSelect").value;

  setText("orderOutput", "Loading dataset...");
  setText("dirsOutput", "Loading dataset...");
  setText("rotationOutput", "Loading dataset...");
  setText("upDirectionOutput", "Loading dataset...");

  currentDataset = await loadDataset(n);

  if (!Array.isArray(currentDataset.witnesses)) {
    throw new Error("Invalid witness dataset: missing witnesses array.");
  }

  populateWitnessSelect(currentDataset);
  handleWitnessChange();
}

function handleWitnessChange() {
  const index = Number($("witnessSelect").value);
  const witnesses = currentDataset && Array.isArray(currentDataset.witnesses)
    ? currentDataset.witnesses
    : [];
  currentWitness = witnesses[index] || null;

  updateDetailsPanel();
  drawEmbedding();
}

async function init() {
  $("nSelect").addEventListener("change", () => {
    handleDatasetChange().catch(showError);
  });

  $("witnessSelect").addEventListener("change", () => {
    try {
      handleWitnessChange();
    } catch (err) {
      showError(err);
    }
  });

  $("displayMode").addEventListener("change", () => {
    try {
      drawEmbedding();
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
