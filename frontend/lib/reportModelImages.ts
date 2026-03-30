import "server-only";

import { classifyWall } from "@/lib/materialEngine";
import { buildColumns, buildStructuralNodes } from "@/lib/sceneGraph";
import type { SceneData } from "@/lib/types";

type PreviewImage = {
  title: string;
  dataUri: string;
};

function toDataUri(svg: string) {
  return `data:image/svg+xml;base64,${Buffer.from(svg, "utf-8").toString("base64")}`;
}

function esc(value: string) {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function viewBox(scene: SceneData) {
  const pad = 0.6;
  const minX = scene.bounds.minX - pad;
  const minZ = scene.bounds.minZ - pad;
  const width = scene.bounds.width + pad * 2;
  const height = scene.bounds.depth + pad * 2;
  return { minX, minZ, width: Math.max(width, 1), height: Math.max(height, 1) };
}

function line(x1: number, y1: number, x2: number, y2: number, color: string, stroke = 0.08) {
  return `<line x1="${x1.toFixed(3)}" y1="${y1.toFixed(3)}" x2="${x2.toFixed(3)}" y2="${y2.toFixed(3)}" stroke="${color}" stroke-width="${stroke}" stroke-linecap="round" />`;
}

function circle(x: number, y: number, r: number, color: string) {
  return `<circle cx="${x.toFixed(3)}" cy="${y.toFixed(3)}" r="${r}" fill="${color}" />`;
}

function baseSvg(scene: SceneData, title: string, body: string) {
  const vb = viewBox(scene);
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${vb.minX} ${vb.minZ} ${vb.width} ${vb.height}" width="800" height="460">
    <rect x="${vb.minX}" y="${vb.minZ}" width="${vb.width}" height="${vb.height}" fill="#f8f7f5" />
    <g transform="scale(1,-1) translate(0,${-(vb.minZ * 2 + vb.height).toFixed(3)})">${body}</g>
    <text x="${(vb.minX + 0.2).toFixed(3)}" y="${(vb.minZ + 0.35).toFixed(3)}" font-size="0.24" fill="#4b5563">${esc(title)}</text>
  </svg>`;
}

function planImage(scene: SceneData): PreviewImage {
  const roomPolygons = scene.rooms
    .map((room) => {
      const points = room.polygon2D.map((point) => `${point[0].toFixed(3)},${point[1].toFixed(3)}`).join(" ");
      return `<polygon points="${points}" fill="#eceff3" stroke="#d7dde4" stroke-width="0.04" />`;
    })
    .join("");

  const wallLines = scene.walls.map((wall) => line(wall.start[0], wall.start[1], wall.end[0], wall.end[1], "#1f2937", 0.1)).join("");
  return {
    title: "Top View Plan",
    dataUri: toDataUri(baseSvg(scene, "Top View Plan", `${roomPolygons}${wallLines}`)),
  };
}

function classificationImage(scene: SceneData): PreviewImage {
  const wallLines = scene.walls
    .map((wall) => {
      const role = classifyWall(wall);
      const color = role === "load_bearing" ? "#0f766e" : role === "partition" ? "#92400e" : "#4338ca";
      return line(wall.start[0], wall.start[1], wall.end[0], wall.end[1], color, 0.12);
    })
    .join("");

  const openingMarks = scene.openings
    .map((opening) => circle(opening.position[0], opening.position[2], 0.05, opening.kind === "door" ? "#be123c" : "#2563eb"))
    .join("");
  return {
    title: "Wall Classification + Openings",
    dataUri: toDataUri(baseSvg(scene, "Wall Classification + Openings", `${wallLines}${openingMarks}`)),
  };
}

function structureImage(scene: SceneData): PreviewImage {
  const nodes = buildStructuralNodes(scene);
  const columns = buildColumns(scene, nodes);
  const walls = scene.walls.map((wall) => line(wall.start[0], wall.start[1], wall.end[0], wall.end[1], "#9ca3af", 0.07)).join("");
  const nodeMarks = nodes.map((node) => circle(node.position[0], node.position[2], 0.04, "#111827")).join("");
  const columnMarks = columns
    .map(
      (column) =>
        `<rect x="${(column.position[0] - column.width / 2).toFixed(3)}" y="${(column.position[2] - column.depth / 2).toFixed(3)}" width="${column.width.toFixed(3)}" height="${column.depth.toFixed(3)}" fill="#16a34a" fill-opacity="0.65" stroke="#14532d" stroke-width="0.03" />`,
    )
    .join("");
  return {
    title: "Structural Nodes + Columns",
    dataUri: toDataUri(baseSvg(scene, "Structural Nodes + Columns", `${walls}${nodeMarks}${columnMarks}`)),
  };
}

function roomSpanImage(scene: SceneData): PreviewImage {
  const roomBodies = scene.rooms
    .map((room) => {
      const points = room.polygon2D.map((point) => `${point[0].toFixed(3)},${point[1].toFixed(3)}`).join(" ");
      return `<polygon points="${points}" fill="#eef2ff" stroke="#c7d2fe" stroke-width="0.04" />`;
    })
    .join("");
  const spanMarkers = scene.rooms
    .map(
      (room) =>
        `<circle cx="${room.centroid[0].toFixed(3)}" cy="${room.centroid[2].toFixed(3)}" r="0.08" fill="#4338ca" fill-opacity="0.85" />` +
        `<text x="${(room.centroid[0] + 0.12).toFixed(3)}" y="${(room.centroid[2] + 0.04).toFixed(3)}" font-size="0.18" fill="#1f2937">${room.span.toFixed(2)}m</text>`,
    )
    .join("");
  return {
    title: "Room Span Snapshot",
    dataUri: toDataUri(baseSvg(scene, "Room Span Snapshot", `${roomBodies}${spanMarkers}`)),
  };
}

export function buildReportModelImages(scene: SceneData): PreviewImage[] {
  return [planImage(scene), classificationImage(scene), structureImage(scene), roomSpanImage(scene)];
}
