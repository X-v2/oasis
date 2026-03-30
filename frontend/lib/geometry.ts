import type { ConfidenceLevel, NormalizedOpening, NormalizedRoom, NormalizedWall, Point2D, Point3D, SceneBounds } from "@/lib/types";

export function distance(a: Point2D, b: Point2D) {
  return Math.hypot(b[0] - a[0], b[1] - a[1]);
}

export function midpoint(a: Point2D, b: Point2D): Point2D {
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
}

export function angle(a: Point2D, b: Point2D) {
  return Math.atan2(b[1] - a[1], b[0] - a[0]);
}

export function to3DPoint(point: Point2D, y = 0): Point3D {
  return [point[0], y, point[1]];
}

export function getWallLength(wall: Pick<NormalizedWall, "start" | "end">) {
  return distance(wall.start, wall.end);
}

export function pointAlongWall(start: Point2D, end: Point2D, dist: number) {
  const dx = end[0] - start[0];
  const dz = end[1] - start[1];
  const len = Math.hypot(dx, dz) || 1;

  return [start[0] + (dx / len) * dist, start[1] + (dz / len) * dist] as Point2D;
}

export function getWallDirection(wall: Pick<NormalizedWall, "start" | "end">): Point2D {
  const length = getWallLength(wall) || 1;
  return [(wall.end[0] - wall.start[0]) / length, (wall.end[1] - wall.start[1]) / length];
}

export function getWallNormal(wall: Pick<NormalizedWall, "start" | "end">): Point2D {
  const dir = getWallDirection(wall);
  return [-dir[1], dir[0]];
}

export function offsetWall(start: Point2D, end: Point2D, thickness: number) {
  const dx = end[0] - start[0];
  const dz = end[1] - start[1];
  const len = Math.hypot(dx, dz) || 1;
  const nx = -dz / len;
  const nz = dx / len;
  const half = thickness / 2;

  return {
    outerStart: [start[0] + nx * half, start[1] + nz * half] as Point2D,
    outerEnd: [end[0] + nx * half, end[1] + nz * half] as Point2D,
    innerStart: [start[0] - nx * half, start[1] - nz * half] as Point2D,
    innerEnd: [end[0] - nx * half, end[1] - nz * half] as Point2D,
  };
}

export function getOpeningIntervals(wall: NormalizedWall, openings: NormalizedOpening[]) {
  return openings
    .filter((opening) => opening.wallId === wall.id)
    .map((opening) => ({
      start: Math.max(0, Math.min(wall.length, opening.offset)),
      end: Math.max(0, Math.min(wall.length, opening.offset + opening.width)),
      data: opening,
    }))
    .sort((a, b) => a.start - b.start);
}

export function splitWall(wall: NormalizedWall, openings: NormalizedOpening[]) {
  const intervals = getOpeningIntervals(wall, openings);
  const wallLength = getWallLength(wall);
  let cursor = 0;
  const segments: Array<{ start: number; end: number }> = [];

  intervals.forEach((interval) => {
    if (interval.start > cursor) {
      segments.push({ start: cursor, end: interval.start });
    }

    cursor = Math.max(cursor, interval.end);
  });

  if (cursor < wallLength) {
    segments.push({ start: cursor, end: wallLength });
  }

  return segments;
}

export function buildWallMeshes(wall: NormalizedWall, openings: NormalizedOpening[]) {
  return splitWall(wall, openings).map((segment) => ({
    start: pointAlongWall(wall.start, wall.end, segment.start),
    end: pointAlongWall(wall.start, wall.end, segment.end),
    thickness: wall.thickness,
    type: wall.type,
  }));
}

export function polygonArea(points: Point2D[]) {
  let sum = 0;
  for (let i = 0; i < points.length; i += 1) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[(i + 1) % points.length];
    sum += x1 * y2 - x2 * y1;
  }

  return Math.abs(sum) / 2;
}

export function polygonCentroid(points: Point2D[]): Point2D {
  const areaFactor = polygonArea(points) * 6 || 1;
  let cx = 0;
  let cy = 0;

  for (let i = 0; i < points.length; i += 1) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[(i + 1) % points.length];
    const cross = x1 * y2 - x2 * y1;
    cx += (x1 + x2) * cross;
    cy += (y1 + y2) * cross;
  }

  return [cx / areaFactor, cy / areaFactor];
}

export function computePolygonSpan(points: Point2D[]) {
  return computeRoomSpan(points).span;
}

function sampleValues(min: number, max: number, values: number[]) {
  const seeded = new Set<number>([min, max, (min + max) / 2, ...values]);
  return [...seeded].sort((a, b) => a - b);
}

function uniqueSortedIntersections(values: number[], tolerance = 1e-6) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted.filter((value, index) => index === 0 || Math.abs(value - sorted[index - 1]) > tolerance);
}

function axisAlignedSpanIntervals(points: Point2D[], axis: "horizontal" | "vertical") {
  const xs = points.map(([x]) => x);
  const zs = points.map(([, z]) => z);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minZ = Math.min(...zs);
  const maxZ = Math.max(...zs);

  const scanValues =
    axis === "horizontal" ? sampleValues(minZ, maxZ, zs) : sampleValues(minX, maxX, xs);

  let best = {
    span: 0,
    line: [to3DPoint([minX, minZ], 0.03), to3DPoint([minX, minZ], 0.03)] as [Point3D, Point3D],
  };

  for (const scan of scanValues) {
    const intersections: number[] = [];

    for (let i = 0; i < points.length; i += 1) {
      const start = points[i];
      const end = points[(i + 1) % points.length];

      if (axis === "horizontal") {
        const [x1, z1] = start;
        const [x2, z2] = end;
        const minEdge = Math.min(z1, z2);
        const maxEdge = Math.max(z1, z2);
        if (scan < minEdge || scan >= maxEdge || z1 === z2) {
          continue;
        }
        const t = (scan - z1) / (z2 - z1);
        intersections.push(x1 + t * (x2 - x1));
      } else {
        const [x1, z1] = start;
        const [x2, z2] = end;
        const minEdge = Math.min(x1, x2);
        const maxEdge = Math.max(x1, x2);
        if (scan < minEdge || scan >= maxEdge || x1 === x2) {
          continue;
        }
        const t = (scan - x1) / (x2 - x1);
        intersections.push(z1 + t * (z2 - z1));
      }
    }

    const unique = uniqueSortedIntersections(intersections);
    for (let i = 0; i + 1 < unique.length; i += 2) {
      const start = unique[i];
      const end = unique[i + 1];
      const span = Math.abs(end - start);

      if (span > best.span) {
        best =
          axis === "horizontal"
            ? {
                span,
                line: [to3DPoint([start, scan], 0.03), to3DPoint([end, scan], 0.03)],
              }
            : {
                span,
                line: [to3DPoint([scan, start], 0.03), to3DPoint([scan, end], 0.03)],
              };
      }
    }
  }

  return best;
}

function orthogonalityConfidence(points: Point2D[]): ConfidenceLevel {
  let orthogonalEdges = 0;

  for (let i = 0; i < points.length; i += 1) {
    const current = points[i];
    const next = points[(i + 1) % points.length];
    const edgeAngle = Math.abs((angle(current, next) * 180) / Math.PI) % 180;
    const distanceToAxis = Math.min(
      Math.abs(edgeAngle),
      Math.abs(edgeAngle - 90),
      Math.abs(edgeAngle - 180),
    );

    if (distanceToAxis < 8) {
      orthogonalEdges += 1;
    }
  }

  const ratio = orthogonalEdges / Math.max(points.length, 1);
  if (ratio > 0.85) {
    return "high";
  }

  if (ratio > 0.55) {
    return "medium";
  }

  return "low";
}

export function computeRoomSpan(points: Point2D[]) {
  const horizontal = axisAlignedSpanIntervals(points, "horizontal");
  const vertical = axisAlignedSpanIntervals(points, "vertical");
  const dominant = horizontal.span >= vertical.span ? horizontal : vertical;
  const confidence = orthogonalityConfidence(points);

  return {
    span: dominant.span,
    spanLine: dominant.line,
    inferredSpan: confidence !== "high",
    confidence,
  };
}

export function pointInPolygon(point: Point2D, polygon: Point2D[]) {
  let inside = false;

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0];
    const yi = polygon[i][1];
    const xj = polygon[j][0];
    const yj = polygon[j][1];

    const intersect =
      yi > point[1] !== yj > point[1] &&
      point[0] < ((xj - xi) * (point[1] - yi)) / ((yj - yi) || Number.EPSILON) + xi;

    if (intersect) {
      inside = !inside;
    }
  }

  return inside;
}

export function distanceToSegment(point: Point2D, start: Point2D, end: Point2D) {
  const [px, py] = point;
  const [x1, y1] = start;
  const [x2, y2] = end;
  const dx = x2 - x1;
  const dy = y2 - y1;

  if (dx === 0 && dy === 0) {
    return distance(point, start);
  }

  const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)));
  const projection: Point2D = [x1 + t * dx, y1 + t * dy];
  return distance(point, projection);
}

export function createBounds(points: Point2D[]): SceneBounds {
  if (points.length === 0) {
    return {
      minX: -1,
      maxX: 1,
      minZ: -1,
      maxZ: 1,
      width: 2,
      depth: 2,
      center: [0, 0, 0],
    };
  }

  const xs = points.map((point) => point[0]);
  const zs = points.map((point) => point[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minZ = Math.min(...zs);
  const maxZ = Math.max(...zs);

  return {
    minX,
    maxX,
    minZ,
    maxZ,
    width: Math.max(maxX - minX, 1),
    depth: Math.max(maxZ - minZ, 1),
    center: [(minX + maxX) / 2, 0, (minZ + maxZ) / 2],
  };
}

export function getRoomContext(room?: NormalizedRoom) {
  if (!room) {
    return { span: 0, inferredSpan: true, confidence: "low" as ConfidenceLevel };
  }

  return {
    room,
    span: room.span,
    inferredSpan: room.inferredSpan,
    confidence: room.confidence,
    spanLine: room.spanLine,
  };
}

export function buildPointLabels(points: Point3D[]) {
  return points.map((point, index) => ({
    id: `pt-${index}-${point[0]}-${point[2]}`,
    label: `P${index + 1}`,
    point,
  }));
}
