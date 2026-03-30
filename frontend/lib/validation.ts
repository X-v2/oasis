import {
  angle,
  computeRoomSpan,
  createBounds,
  distance,
  midpoint,
  pointAlongWall,
  polygonArea,
  polygonCentroid,
  to3DPoint,
} from "@/lib/geometry";
import type {
  ColumnInput,
  ConfidenceLevel,
  DoorInput,
  DoorSwing,
  GraphNodeInput,
  LabelInput,
  MetaInput,
  NormalizedColumn,
  NormalizedGraphNode,
  NormalizedLabel,
  NormalizedOpening,
  NormalizedRoom,
  NormalizedWall,
  OpeningInput,
  Point2D,
  Point3D,
  RawSceneInput,
  SlabInput,
  SceneData,
  SceneMeta,
  ValidationIssue,
  ValidationResult,
  WallInput,
  WallType,
  WindowPanelType,
  WindowInput,
} from "@/lib/types";

const DEFAULT_META: SceneMeta = {
  unit: "meter",
  wallHeight: 3,
  defaultWallThickness: 0.2,
};

const MIN_WALL_LENGTH = 0.2;
const ENDPOINT_SNAP_TOLERANCE = 0.05;
const DUPLICATE_WALL_TOLERANCE = 0.06;

function createIssue(issues: ValidationIssue[], issue: Omit<ValidationIssue, "id">) {
  issues.push({
    id: `issue-${issues.length + 1}`,
    ...issue,
  });
}

function asObject(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }

  return value as Record<string, unknown>;
}

function isPoint2D(value: unknown): value is Point2D {
  return (
    Array.isArray(value) &&
    value.length === 2 &&
    value.every((entry) => typeof entry === "number" && Number.isFinite(entry))
  );
}

function isPoint3D(value: unknown): value is Point3D {
  return (
    Array.isArray(value) &&
    value.length === 3 &&
    value.every((entry) => typeof entry === "number" && Number.isFinite(entry))
  );
}

function normalizePoint(value: unknown) {
  return isPoint2D(value) ? value : undefined;
}

function normalizePoint3D(value: unknown) {
  if (isPoint3D(value)) {
    return value;
  }
  if (isPoint2D(value)) {
    return to3DPoint(value, 0.2);
  }
  return undefined;
}

function normalizePolygon2D(value: unknown) {
  if (!Array.isArray(value) || value.length < 3) {
    return undefined;
  }
  const points: Point2D[] = [];
  for (const entry of value) {
    if (isPoint2D(entry)) {
      points.push(entry);
      continue;
    }
    if (isPoint3D(entry)) {
      points.push([entry[0], entry[2]]);
      continue;
    }
    return undefined;
  }
  return points;
}

function emptyScene(readiness: SceneData["readiness"] = "invalid"): SceneData {
  return {
    meta: DEFAULT_META,
    walls: [],
    rooms: [],
    labels: [],
    openings: [],
    graphNodes: [],
    columns: [],
    bounds: createBounds([]),
    readiness,
  };
}

function parseMeta(input: MetaInput | undefined, issues: ValidationIssue[]) {
  const meta = asObject(input);

  const unit = meta?.unit === "meter" ? "meter" : DEFAULT_META.unit;
  if (meta?.unit && meta.unit !== "meter") {
    createIssue(issues, {
      severity: "warning",
      message: 'Only "meter" is supported right now, so the upload was normalized to meters.',
      path: "meta.unit",
    });
  }

  const wallHeight =
    typeof meta?.wallHeight === "number" && Number.isFinite(meta.wallHeight) && meta.wallHeight > 0
      ? meta.wallHeight
      : DEFAULT_META.wallHeight;

  const defaultWallThickness =
    typeof meta?.defaultWallThickness === "number" &&
    Number.isFinite(meta.defaultWallThickness) &&
    meta.defaultWallThickness > 0
      ? meta.defaultWallThickness
      : DEFAULT_META.defaultWallThickness;

  return {
    unit,
    wallHeight,
    defaultWallThickness,
  } satisfies SceneMeta;
}

function buildFootprintBounds(wallsInput: WallInput[]) {
  const wallPoints = wallsInput
    .map((entry) => asObject(entry))
    .flatMap((wall) => {
      const start = normalizePoint(wall?.start);
      const end = normalizePoint(wall?.end);
      return start && end ? [start, end] : [];
    });

  return createBounds(wallPoints);
}

function inferWallType(
  start: Point2D,
  end: Point2D,
  bounds: SceneData["bounds"],
): { type: WallType; confidence: ConfidenceLevel } {
  const dx = Math.abs(end[0] - start[0]);
  const dz = Math.abs(end[1] - start[1]);
  const axisTolerance = 0.14;
  const nearEnvelope =
    (Math.abs(start[0] - bounds.minX) < axisTolerance && Math.abs(end[0] - bounds.minX) < axisTolerance) ||
    (Math.abs(start[0] - bounds.maxX) < axisTolerance && Math.abs(end[0] - bounds.maxX) < axisTolerance) ||
    (Math.abs(start[1] - bounds.minZ) < axisTolerance && Math.abs(end[1] - bounds.minZ) < axisTolerance) ||
    (Math.abs(start[1] - bounds.maxZ) < axisTolerance && Math.abs(end[1] - bounds.maxZ) < axisTolerance);

  const coverage = dx > dz ? dx / Math.max(bounds.width, 1) : dz / Math.max(bounds.depth, 1);

  if (nearEnvelope && coverage > 0.35) {
    return { type: "outer", confidence: coverage > 0.55 ? "high" : "medium" };
  }

  return { type: "partition", confidence: "low" };
}

function parseWalls(
  input: WallInput[],
  meta: SceneMeta,
  bounds: SceneData["bounds"],
  issues: ValidationIssue[],
) {
  const walls: NormalizedWall[] = [];
  const seenIds = new Set<string>();

  input.forEach((wallEntry, index) => {
    const wall = asObject(wallEntry);
    if (!wall) {
      createIssue(issues, {
        severity: "error",
        message: "Wall entry is not an object and was skipped.",
        path: `walls[${index}]`,
      });
      return;
    }

    const id = typeof wall.id === "string" && wall.id.trim() ? wall.id : `wall-${index + 1}`;
    const finalId = seenIds.has(id) ? `${id}-${index + 1}` : id;
    seenIds.add(finalId);

    const start = normalizePoint(wall.start);
    const end = normalizePoint(wall.end);
    if (!start || !end) {
      createIssue(issues, {
        severity: "error",
        message: `Wall ${finalId} is missing a valid start/end pair and was skipped.`,
        path: `walls[${index}]`,
        wallId: finalId,
      });
      return;
    }

    const length = distance(start, end);
    if (length < MIN_WALL_LENGTH) {
      createIssue(issues, {
        severity: "error",
        message: `Wall ${finalId} is too short to render and was skipped.`,
        path: `walls[${index}]`,
        wallId: finalId,
      });
      return;
    }

    const rawType = typeof wall.type === "string" ? wall.type.trim().toLowerCase() : undefined;
    const inferred = inferWallType(start, end, bounds);
    const type: WallType =
      rawType === "outer" || rawType === "partition" || rawType === "semi_structural" ? rawType : inferred.type;

    const thickness =
      typeof wall.thickness === "number" && Number.isFinite(wall.thickness) && wall.thickness > 0
        ? wall.thickness
        : type === "outer"
          ? meta.defaultWallThickness
          : Math.max(0.12, meta.defaultWallThickness * 0.75);

    const height =
      typeof wall.height === "number" && Number.isFinite(wall.height) && wall.height > 0
        ? wall.height
        : meta.wallHeight;

    const center2D = midpoint(start, end);

    walls.push({
      id: finalId,
      start,
      end,
      start3D: to3DPoint(start),
      end3D: to3DPoint(end),
      thickness,
      height,
      type,
      inferredType: !rawType,
      length,
      midpoint: to3DPoint(center2D, height / 2),
      angle: angle(start, end),
      confidence: rawType ? "high" : inferred.confidence,
    });

    if (!rawType) {
      createIssue(issues, {
        severity: "info",
        message: `Wall ${finalId} type was inferred as ${type} with ${inferred.confidence} confidence.`,
        path: `walls[${index}].type`,
        wallId: finalId,
      });
    }
  });

  return sanitizeWallsForNoise(walls, issues);
}

function nearestClusterIndex(clusters: Point2D[], point: Point2D, tolerance: number) {
  for (let index = 0; index < clusters.length; index += 1) {
    if (distance(clusters[index], point) <= tolerance) {
      return index;
    }
  }
  return -1;
}

function clusterWallEndpoints(walls: NormalizedWall[]) {
  const clusters: Array<{ center: Point2D; count: number }> = [];
  const endpointMap = new Map<string, Point2D>();

  walls.forEach((wall) => {
    [wall.start, wall.end].forEach((point, pointIndex) => {
      const index = nearestClusterIndex(
        clusters.map((entry) => entry.center),
        point,
        ENDPOINT_SNAP_TOLERANCE,
      );
      if (index === -1) {
        clusters.push({ center: [point[0], point[1]], count: 1 });
        endpointMap.set(`${wall.id}:${pointIndex}`, [point[0], point[1]]);
      } else {
        const cluster = clusters[index];
        const nextCount = cluster.count + 1;
        cluster.center = [
          (cluster.center[0] * cluster.count + point[0]) / nextCount,
          (cluster.center[1] * cluster.count + point[1]) / nextCount,
        ];
        cluster.count = nextCount;
        endpointMap.set(`${wall.id}:${pointIndex}`, [cluster.center[0], cluster.center[1]]);
      }
    });
  });

  return endpointMap;
}

function normalizeWallGeometry(wall: NormalizedWall, start: Point2D, end: Point2D) {
  const length = distance(start, end);
  const center2D = midpoint(start, end);
  return {
    ...wall,
    start,
    end,
    start3D: to3DPoint(start),
    end3D: to3DPoint(end),
    length,
    midpoint: to3DPoint(center2D, wall.height / 2),
    angle: angle(start, end),
    roomId: undefined,
  };
}

function canonicalWallKey(start: Point2D, end: Point2D) {
  const a = `${start[0].toFixed(3)}:${start[1].toFixed(3)}`;
  const b = `${end[0].toFixed(3)}:${end[1].toFixed(3)}`;
  return a < b ? `${a}|${b}` : `${b}|${a}`;
}

function sanitizeWallsForNoise(walls: NormalizedWall[], issues: ValidationIssue[]) {
  if (walls.length === 0) {
    return walls;
  }

  const endpointMap = clusterWallEndpoints(walls);
  const snapped = walls
    .map((wall) => {
      const start = endpointMap.get(`${wall.id}:0`) ?? wall.start;
      const end = endpointMap.get(`${wall.id}:1`) ?? wall.end;
      const normalized = normalizeWallGeometry(wall, start, end);
      return normalized;
    })
    .filter((wall) => wall.length >= MIN_WALL_LENGTH);

  const seen = new Map<string, NormalizedWall>();
  const deduped: NormalizedWall[] = [];

  snapped.forEach((wall) => {
    const key = canonicalWallKey(wall.start, wall.end);
    const existing = seen.get(key);
    const directDuplicate =
      existing &&
      distance(existing.start, wall.start) <= DUPLICATE_WALL_TOLERANCE &&
      distance(existing.end, wall.end) <= DUPLICATE_WALL_TOLERANCE;
    const reverseDuplicate =
      existing &&
      distance(existing.start, wall.end) <= DUPLICATE_WALL_TOLERANCE &&
      distance(existing.end, wall.start) <= DUPLICATE_WALL_TOLERANCE;
    if (directDuplicate || reverseDuplicate) {
      createIssue(issues, {
        severity: "warning",
        message: `Wall ${wall.id} was removed as a near-duplicate of ${existing.id}.`,
        wallId: wall.id,
        path: "walls",
      });
      return;
    }
    seen.set(key, wall);
    deduped.push(wall);
  });

  if (deduped.length !== walls.length) {
    createIssue(issues, {
      severity: "info",
      message: "Input wall endpoints were snapped/cleaned to stabilize noisy or slightly skewed geometry.",
      path: "walls",
    });
  }

  return deduped;
}

function parseLabels(input: LabelInput[], issues: ValidationIssue[]) {
  const labels: NormalizedLabel[] = [];
  const seenIds = new Set<string>();

  input.forEach((entry, index) => {
    const label = asObject(entry);
    if (!label) {
      createIssue(issues, {
        severity: "error",
        message: "Label entry is not an object and was skipped.",
        path: `labels[${index}]`,
      });
      return;
    }

    const text = typeof label.text === "string" && label.text.trim() ? label.text.trim() : undefined;
    if (!text) {
      createIssue(issues, {
        severity: "error",
        message: `Label ${index + 1} is missing text and was skipped.`,
        path: `labels[${index}].text`,
      });
      return;
    }

    const position = normalizePoint3D(label.position);
    if (!position) {
      createIssue(issues, {
        severity: "error",
        message: `Label ${text} is missing a valid position and was skipped.`,
        path: `labels[${index}].position`,
      });
      return;
    }

    const id = typeof label.id === "string" && label.id.trim() ? label.id.trim() : `label-${index + 1}`;
    const finalId = seenIds.has(id) ? `${id}-${index + 1}` : id;
    seenIds.add(finalId);

    labels.push({
      id: finalId,
      text,
      position,
    });
  });

  return labels;
}

function parseSlabs(input: SlabInput[], issues: ValidationIssue[]) {
  const slabs: NormalizedRoom[] = [];
  const seenIds = new Set<string>();

  input.forEach((entry, index) => {
    const slab = asObject(entry);
    if (!slab) {
      createIssue(issues, {
        severity: "error",
        message: "Slab entry is not an object and was skipped.",
        path: `slabs[${index}]`,
      });
      return;
    }

    const id = typeof slab.id === "string" && slab.id.trim() ? slab.id : `slab-${index + 1}`;
    const finalId = seenIds.has(id) ? `${id}-${index + 1}` : id;
    seenIds.add(finalId);

    const polygon = normalizePolygon2D(slab.polygon ?? slab.coordinates);
    if (!polygon) {
      createIssue(issues, {
        severity: "error",
        message: `Slab ${finalId} needs polygon/coordinates with at least 3 points and was skipped.`,
        path: `slabs[${index}].polygon`,
        roomId: finalId,
      });
      return;
    }

    const area = polygonArea(polygon);
    if (area <= 0.01) {
      createIssue(issues, {
        severity: "error",
        message: `Slab ${finalId} has near-zero area and was skipped.`,
        path: `slabs[${index}].polygon`,
        roomId: finalId,
      });
      return;
    }

    const spanData = computeRoomSpan(polygon);
    const centroidInput = normalizePoint3D(slab.centroid);
    const centroid2D = centroidInput ? ([centroidInput[0], centroidInput[2]] as Point2D) : polygonCentroid(polygon);
    const name =
      typeof slab.name === "string" && slab.name.trim()
        ? slab.name.trim()
        : finalId.replace(/[-_]/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());

    slabs.push({
      id: finalId,
      name,
      polygon2D: polygon,
      polygon3D: polygon.map((point) => to3DPoint(point)),
      span: spanData.span,
      spanLine: spanData.spanLine,
      centroid: centroidInput ?? to3DPoint(centroid2D, 0.02),
      area,
      inferredSpan: spanData.inferredSpan,
      confidence: spanData.confidence,
    });
  });

  return slabs;
}

function projectOffsetOnWall(position: Point3D, wall: NormalizedWall) {
  const vx = wall.end[0] - wall.start[0];
  const vz = wall.end[1] - wall.start[1];
  const lenSq = vx * vx + vz * vz;
  if (lenSq <= 0.000001) {
    return 0;
  }
  const px = position[0] - wall.start[0];
  const pz = position[2] - wall.start[1];
  const t = Math.max(0, Math.min(1, (px * vx + pz * vz) / lenSq));
  return t * wall.length;
}

function parseDoors(input: DoorInput[], walls: NormalizedWall[], issues: ValidationIssue[]) {
  const openings: NormalizedOpening[] = [];
  const wallMap = new Map(walls.map((wall) => [wall.id, wall]));

  input.forEach((doorEntry, index) => {
    const door = asObject(doorEntry);
    if (!door) {
      createIssue(issues, {
        severity: "error",
        message: "Door entry is not an object and was skipped.",
        path: `doors[${index}]`,
      });
      return;
    }

    const id = typeof door.id === "string" && door.id.trim() ? door.id : `door-${index + 1}`;
    const wallId = typeof door.wallId === "string" ? door.wallId : undefined;
    const wall = wallId ? wallMap.get(wallId) : undefined;

    if (!wall) {
      createIssue(issues, {
        severity: "warning",
        message: `Door ${id} references a missing wall and was skipped.`,
        path: `doors[${index}].wallId`,
        openingId: id,
      });
      return;
    }

    const rawPosition = normalizePoint3D(door.position);
    const offsetFromPosition = rawPosition ? projectOffsetOnWall(rawPosition, wall) : undefined;
    const offset =
      typeof door.offset === "number" && Number.isFinite(door.offset)
        ? Math.max(0, Math.min(wall.length, door.offset))
        : offsetFromPosition ?? wall.length * 0.35;
    const width =
      typeof door.width === "number" && Number.isFinite(door.width) && door.width > 0
        ? Math.min(door.width, Math.max(0.1, wall.length - offset))
        : 0.9;
    const height =
      typeof door.height === "number" && Number.isFinite(door.height) && door.height > 0
        ? Math.min(door.height, wall.height)
        : 2.1;
    const swing: DoorSwing = door.swing === "right" ? "right" : "left";
    const point = pointAlongWall(wall.start, wall.end, offset);
    const position = rawPosition ?? to3DPoint(point, 0);

    openings.push({
      id,
      wallId: wall.id,
      kind: "door",
      offset,
      width,
      height,
      position,
      angle: wall.angle,
      swing,
    });
  });

  return openings;
}

function parseWindows(input: WindowInput[], walls: NormalizedWall[], issues: ValidationIssue[]) {
  const openings: NormalizedOpening[] = [];
  const wallMap = new Map(walls.map((wall) => [wall.id, wall]));

  input.forEach((windowEntry, index) => {
    const windowData = asObject(windowEntry);
    if (!windowData) {
      createIssue(issues, {
        severity: "error",
        message: "Window entry is not an object and was skipped.",
        path: `windows[${index}]`,
      });
      return;
    }

    const id = typeof windowData.id === "string" && windowData.id.trim() ? windowData.id : `window-${index + 1}`;
    const wallId = typeof windowData.wallId === "string" ? windowData.wallId : undefined;
    const wall = wallId ? wallMap.get(wallId) : undefined;

    if (!wall) {
      createIssue(issues, {
        severity: "warning",
        message: `Window ${id} references a missing wall and was skipped.`,
        path: `windows[${index}].wallId`,
        openingId: id,
      });
      return;
    }

    const rawPosition = normalizePoint3D(windowData.position);
    const offsetFromPosition = rawPosition ? projectOffsetOnWall(rawPosition, wall) : undefined;
    const offset =
      typeof windowData.offset === "number" && Number.isFinite(windowData.offset)
        ? Math.max(0, Math.min(wall.length, windowData.offset))
        : offsetFromPosition ?? wall.length * 0.4;
    const width =
      typeof windowData.width === "number" && Number.isFinite(windowData.width) && windowData.width > 0
        ? Math.min(windowData.width, Math.max(0.1, wall.length - offset))
        : 1.2;
    const height =
      typeof windowData.height === "number" && Number.isFinite(windowData.height) && windowData.height > 0
        ? Math.min(windowData.height, wall.height * 0.8)
        : 1.2;
    const sillHeight =
      typeof windowData.sillHeight === "number" &&
      Number.isFinite(windowData.sillHeight) &&
      windowData.sillHeight >= 0
        ? Math.min(windowData.sillHeight, Math.max(0, wall.height - height))
        : 1;
    const rawWindowType =
      typeof windowData.windowType === "string"
        ? windowData.windowType
        : typeof windowData.type === "string"
          ? windowData.type
          : "double";
    const panelType: WindowPanelType = rawWindowType.toLowerCase() === "single" ? "single" : "double";
    const point = pointAlongWall(wall.start, wall.end, offset);
    const position = rawPosition ?? to3DPoint(point, sillHeight + height / 2);

    openings.push({
      id,
      wallId: wall.id,
      kind: "window",
      offset,
      width,
      height,
      position,
      angle: wall.angle,
      sillHeight,
      panelType,
    });
  });

  return openings;
}

function parseOpenings(input: OpeningInput[], walls: NormalizedWall[], issues: ValidationIssue[]) {
  const openings: NormalizedOpening[] = [];
  const wallMap = new Map(walls.map((wall) => [wall.id, wall]));
  const seenIds = new Set<string>();

  input.forEach((entry, index) => {
    const openingData = asObject(entry);
    if (!openingData) {
      createIssue(issues, {
        severity: "error",
        message: "Opening entry is not an object and was skipped.",
        path: `openings[${index}]`,
      });
      return;
    }

    const kindRaw = typeof openingData.kind === "string" ? openingData.kind.toLowerCase() : undefined;
    const kind = kindRaw === "door" || kindRaw === "window" ? kindRaw : undefined;
    if (!kind) {
      createIssue(issues, {
        severity: "error",
        message: `Opening ${index + 1} must specify kind as "door" or "window".`,
        path: `openings[${index}].kind`,
      });
      return;
    }

    const id = typeof openingData.id === "string" && openingData.id.trim() ? openingData.id : `${kind}-${index + 1}`;
    if (seenIds.has(id)) {
      createIssue(issues, {
        severity: "warning",
        message: `Opening ${id} is duplicated in openings[] and later entry was skipped.`,
        path: `openings[${index}].id`,
        openingId: id,
      });
      return;
    }
    seenIds.add(id);

    const wallId = typeof openingData.wallId === "string" ? openingData.wallId : undefined;
    const wall = wallId ? wallMap.get(wallId) : undefined;
    if (!wall) {
      createIssue(issues, {
        severity: "warning",
        message: `Opening ${id} references a missing wall and was skipped.`,
        path: `openings[${index}].wallId`,
        openingId: id,
      });
      return;
    }

    const rawPosition = normalizePoint3D(openingData.position);
    const offsetFromPosition = rawPosition ? projectOffsetOnWall(rawPosition, wall) : undefined;
    const offset =
      typeof openingData.offset === "number" && Number.isFinite(openingData.offset)
        ? Math.max(0, Math.min(wall.length, openingData.offset))
        : offsetFromPosition ?? wall.length * (kind === "door" ? 0.35 : 0.4);
    const width =
      typeof openingData.width === "number" && Number.isFinite(openingData.width) && openingData.width > 0
        ? Math.min(openingData.width, Math.max(0.1, wall.length - offset))
        : kind === "door"
          ? 0.9
          : 1.2;
    const height =
      typeof openingData.height === "number" && Number.isFinite(openingData.height) && openingData.height > 0
        ? Math.min(openingData.height, wall.height)
        : kind === "door"
          ? 2.1
          : 1.2;
    const point = pointAlongWall(wall.start, wall.end, offset);

    if (kind === "door") {
      const swing: DoorSwing = openingData.swing === "right" ? "right" : "left";
      openings.push({
        id,
        wallId: wall.id,
        kind,
        offset,
        width,
        height,
        position: rawPosition ?? to3DPoint(point, 0),
        angle: wall.angle,
        swing,
      });
      return;
    }

    const sillHeight =
      typeof openingData.sillHeight === "number" &&
      Number.isFinite(openingData.sillHeight) &&
      openingData.sillHeight >= 0
        ? Math.min(openingData.sillHeight, Math.max(0, wall.height - height))
        : 1;
    const rawWindowType =
      typeof openingData.windowType === "string"
        ? openingData.windowType
        : typeof openingData.type === "string"
          ? openingData.type
          : "double";
    const panelType: WindowPanelType = rawWindowType.toLowerCase() === "single" ? "single" : "double";
    openings.push({
      id,
      wallId: wall.id,
      kind,
      offset,
      width,
      height,
      position: rawPosition ?? to3DPoint(point, sillHeight + height / 2),
      angle: wall.angle,
      sillHeight,
      panelType,
    });
  });

  return openings;
}

function dedupeOpenings(openings: NormalizedOpening[], issues: ValidationIssue[]) {
  const unique: NormalizedOpening[] = [];
  const seenById = new Map<string, NormalizedOpening>();
  const seenByShape = new Set<string>();

  function shapeKey(opening: NormalizedOpening) {
    const sill = opening.kind === "window" ? (opening.sillHeight ?? 0).toFixed(3) : "0.000";
    const panel = opening.kind === "window" ? (opening.panelType ?? "double") : "na";
    const swing = opening.kind === "door" ? (opening.swing ?? "left") : "na";
    return [
      opening.kind,
      opening.wallId,
      opening.offset.toFixed(3),
      opening.width.toFixed(3),
      opening.height.toFixed(3),
      sill,
      panel,
      swing,
    ].join("|");
  }

  openings.forEach((opening) => {
    const byId = seenById.get(opening.id);
    if (byId) {
      createIssue(issues, {
        severity: "warning",
        message: `Opening ${opening.id} appears multiple times across doors/windows/openings and duplicate entries were removed.`,
        path: "openings",
        openingId: opening.id,
        wallId: opening.wallId,
      });
      return;
    }

    const key = shapeKey(opening);
    if (seenByShape.has(key)) {
      createIssue(issues, {
        severity: "warning",
        message: `Detected repeated ${opening.kind} geometry on wall ${opening.wallId}; duplicate entry ${opening.id} was removed.`,
        path: "openings",
        openingId: opening.id,
        wallId: opening.wallId,
      });
      return;
    }

    seenById.set(opening.id, opening);
    seenByShape.add(key);
    unique.push(opening);
  });

  return unique;
}

function parseGraphNodes(input: GraphNodeInput[], walls: NormalizedWall[], issues: ValidationIssue[]) {
  const nodes: NormalizedGraphNode[] = [];
  const validWallIds = new Set(walls.map((wall) => wall.id));
  const seenIds = new Set<string>();

  input.forEach((entry, index) => {
    const node = asObject(entry);
    if (!node) {
      createIssue(issues, {
        severity: "error",
        message: "Graph node entry is not an object and was skipped.",
        path: `graphNodes[${index}]`,
      });
      return;
    }

    const position = normalizePoint3D(node.position);
    if (!position) {
      createIssue(issues, {
        severity: "error",
        message: `Graph node ${index + 1} is missing valid position and was skipped.`,
        path: `graphNodes[${index}].position`,
      });
      return;
    }

    const id = typeof node.id === "string" && node.id.trim() ? node.id : `node-${index + 1}`;
    const finalId = seenIds.has(id) ? `${id}-${index + 1}` : id;
    seenIds.add(finalId);

    const connectedWallIds = Array.isArray(node.connectedWallIds)
      ? node.connectedWallIds.filter((wallId): wallId is string => typeof wallId === "string" && validWallIds.has(wallId))
      : [];
    const loadBearingWallIds = Array.isArray(node.loadBearingWallIds)
      ? node.loadBearingWallIds.filter((wallId): wallId is string => typeof wallId === "string" && validWallIds.has(wallId))
      : [];
    const degree =
      typeof node.degree === "number" && Number.isFinite(node.degree) && node.degree >= 0
        ? Math.floor(node.degree)
        : connectedWallIds.length;
    const rawType = typeof node.type === "string" ? node.type.toLowerCase() : "";
    const type: NormalizedGraphNode["type"] =
      rawType === "terminal" || rawType === "corner" || rawType === "junction" || rawType === "inline"
        ? rawType
        : degree <= 1
          ? "terminal"
          : degree >= 3
            ? "junction"
            : "corner";
    const likelyColumn = typeof node.likelyColumn === "boolean" ? node.likelyColumn : degree >= 3 || loadBearingWallIds.length >= 2;
    const color = typeof node.color === "string" && node.color.trim() ? node.color : likelyColumn ? "#4f6678" : "#8091a0";

    nodes.push({
      id: finalId,
      position,
      degree,
      type,
      connectedWallIds,
      loadBearingWallIds,
      likelyColumn,
      color,
    });
  });

  return nodes;
}

function parseColumns(input: ColumnInput[], graphNodes: NormalizedGraphNode[], meta: SceneMeta, issues: ValidationIssue[]) {
  const columns: NormalizedColumn[] = [];
  const validNodeIds = new Set(graphNodes.map((node) => node.id));
  const seenIds = new Set<string>();

  input.forEach((entry, index) => {
    const column = asObject(entry);
    if (!column) {
      createIssue(issues, {
        severity: "error",
        message: "Column entry is not an object and was skipped.",
        path: `columns[${index}]`,
      });
      return;
    }

    const position = normalizePoint3D(column.position);
    if (!position) {
      createIssue(issues, {
        severity: "error",
        message: `Column ${index + 1} is missing valid position and was skipped.`,
        path: `columns[${index}].position`,
      });
      return;
    }

    const id = typeof column.id === "string" && column.id.trim() ? column.id : `column-${index + 1}`;
    const finalId = seenIds.has(id) ? `${id}-${index + 1}` : id;
    seenIds.add(finalId);

    const nodeIdRaw = typeof column.nodeId === "string" ? column.nodeId : "";
    const nodeId = validNodeIds.has(nodeIdRaw) ? nodeIdRaw : graphNodes[index]?.id ?? `node-${index + 1}`;
    const width =
      typeof column.width === "number" && Number.isFinite(column.width) && column.width > 0 ? column.width : 0.3;
    const depth =
      typeof column.depth === "number" && Number.isFinite(column.depth) && column.depth > 0 ? column.depth : width;
    const height =
      typeof column.height === "number" && Number.isFinite(column.height) && column.height > 0
        ? column.height
        : Math.max(2.4, meta.wallHeight);
    const degree =
      typeof column.degree === "number" && Number.isFinite(column.degree) && column.degree >= 0
        ? Math.floor(column.degree)
        : graphNodes.find((node) => node.id === nodeId)?.degree ?? 0;
    const connectedWallIds = Array.isArray(column.connectedWallIds)
      ? column.connectedWallIds.filter((wallId): wallId is string => typeof wallId === "string")
      : graphNodes.find((node) => node.id === nodeId)?.connectedWallIds ?? [];
    const color = typeof column.color === "string" && column.color.trim() ? column.color : "#667786";

    columns.push({
      id: finalId,
      nodeId,
      position,
      width,
      depth,
      height,
      color,
      degree,
      connectedWallIds,
    });
  });

  return columns;
}

function resolveOpeningCollisions(openings: NormalizedOpening[], walls: NormalizedWall[], issues: ValidationIssue[]) {
  const wallMap = new Map(walls.map((wall) => [wall.id, wall]));
  const junctionOffsetsByWall = buildWallJunctionOffsets(walls);
  const groupedByWall = new Map<string, Array<{ opening: NormalizedOpening; sourceIndex: number }>>();

  openings.forEach((opening, index) => {
    const list = groupedByWall.get(opening.wallId) ?? [];
    list.push({ opening: { ...opening }, sourceIndex: index });
    groupedByWall.set(opening.wallId, list);
  });

  const resolved: Array<{ opening: NormalizedOpening; sourceIndex: number }> = [];
  const minimumWidth = 0.1;
  const endClearance = 0.2;
  const junctionClearance = 0.3;

  function pairClearance(
    previousKind: NormalizedOpening["kind"],
    nextKind: NormalizedOpening["kind"],
  ) {
    if (previousKind === "door" && nextKind === "door") {
      return 0.22;
    }
    if (previousKind === "window" && nextKind === "window") {
      return 0.15;
    }
    return 0.24;
  }

  groupedByWall.forEach((entries, wallId) => {
    const wall = wallMap.get(wallId);
    if (!wall) {
      return;
    }
    const junctionOffsets = junctionOffsetsByWall.get(wallId) ?? [];

    const ordered = [...entries].sort((a, b) => {
      const delta = a.opening.offset - b.opening.offset;
      return Math.abs(delta) < 0.0001 ? a.sourceIndex - b.sourceIndex : delta;
    });
    let previousEnd = endClearance;
    let previousKind: NormalizedOpening["kind"] | null = null;

    ordered.forEach((entry) => {
      const current = entry.opening;
      const desiredStart = Math.max(endClearance, Math.min(wall.length - endClearance - minimumWidth, current.offset));
      const desiredWidth = Math.max(minimumWidth, Math.min(current.width, wall.length - endClearance - desiredStart));
      const desiredEnd = desiredStart + desiredWidth;
      const clearanceToPrevious = previousKind ? pairClearance(previousKind, current.kind) : 0;
      let start = Math.max(desiredStart, previousEnd + clearanceToPrevious);
      let adjustedForJunction = false;
      let guard = 0;

      while (guard < junctionOffsets.length + 3) {
        guard += 1;
        const end = start + desiredWidth;
        const blockingOffset = junctionOffsets.find(
          (junctionOffset) =>
            end > junctionOffset - junctionClearance && start < junctionOffset + junctionClearance,
        );
        if (blockingOffset === undefined) {
          break;
        }
        adjustedForJunction = true;
        start = Math.max(start, blockingOffset + junctionClearance);
      }
      const maxWidth = wall.length - endClearance - start;

      if (maxWidth < minimumWidth) {
        createIssue(issues, {
          severity: "error",
          message: `${current.kind === "door" ? "Door" : "Window"} ${current.id} has unresolved clash on wall ${wall.id} and was removed.`,
          path: `${current.kind}s`,
          wallId: wall.id,
          openingId: current.id,
        });
        return;
      }

      const width = Math.max(minimumWidth, Math.min(desiredWidth, maxWidth));
      const end = start + width;

      if (Math.abs(start - desiredStart) > 0.001 || Math.abs(end - desiredEnd) > 0.001) {
        createIssue(issues, {
          severity: "warning",
          message: `${current.kind === "door" ? "Door" : "Window"} ${current.id} was shifted/resized to keep clearance on wall ${wall.id}.`,
          path: `${current.kind}s`,
          wallId: wall.id,
          openingId: current.id,
        });
      }
      if (adjustedForJunction) {
        createIssue(issues, {
          severity: "warning",
          message: `${current.kind === "door" ? "Door" : "Window"} ${current.id} was moved to avoid wall-node/junction clearance on wall ${wall.id}.`,
          path: `${current.kind}s`,
          wallId: wall.id,
          openingId: current.id,
        });
      }

      if (width < current.width * 0.6) {
        createIssue(issues, {
          severity: "error",
          message: `${current.kind === "door" ? "Door" : "Window"} ${current.id} had major size reduction from clash constraints on wall ${wall.id}.`,
          path: `${current.kind}s`,
          wallId: wall.id,
          openingId: current.id,
        });
      }

      const point = pointAlongWall(wall.start, wall.end, start);
      const yCenter = current.kind === "window" ? (current.sillHeight ?? 1) + current.height / 2 : 0;

      resolved.push({
        sourceIndex: entry.sourceIndex,
        opening: {
          ...current,
          offset: start,
          width,
          position: to3DPoint(point, yCenter),
          angle: wall.angle,
        },
      });
      previousEnd = end;
      previousKind = current.kind;
    });
  });

  return resolved.sort((a, b) => a.sourceIndex - b.sourceIndex).map((entry) => entry.opening);
}

function detectOpeningClashes(openings: NormalizedOpening[], walls: NormalizedWall[], issues: ValidationIssue[]) {
  const wallMap = new Map(walls.map((wall) => [wall.id, wall]));
  const junctionOffsetsByWall = buildWallJunctionOffsets(walls);
  const groupedByWall = new Map<string, NormalizedOpening[]>();
  const edgeClearance = 0.2;
  const junctionClearance = 0.3;
  const tolerance = 0.001;

  function pairClearance(
    previousKind: NormalizedOpening["kind"],
    nextKind: NormalizedOpening["kind"],
  ) {
    if (previousKind === "door" && nextKind === "door") {
      return 0.22;
    }
    if (previousKind === "window" && nextKind === "window") {
      return 0.15;
    }
    return 0.24;
  }

  openings.forEach((opening) => {
    const list = groupedByWall.get(opening.wallId) ?? [];
    list.push(opening);
    groupedByWall.set(opening.wallId, list);
  });

  groupedByWall.forEach((entries, wallId) => {
    const wall = wallMap.get(wallId);
    if (!wall) {
      return;
    }
    const junctionOffsets = junctionOffsetsByWall.get(wallId) ?? [];

    const ordered = [...entries].sort((a, b) => a.offset - b.offset);
    ordered.forEach((opening, index) => {
      if (opening.offset < edgeClearance - tolerance) {
        createIssue(issues, {
          severity: "error",
          message: `${opening.kind === "door" ? "Door" : "Window"} ${opening.id} clashes with wall ${wall.id} start edge.`,
          path: `${opening.kind}s`,
          wallId: wall.id,
          openingId: opening.id,
          fixType: "opening_clash",
        });
      }

      if (opening.offset + opening.width > wall.length - edgeClearance + tolerance) {
        createIssue(issues, {
          severity: "error",
          message: `${opening.kind === "door" ? "Door" : "Window"} ${opening.id} clashes with wall ${wall.id} end edge.`,
          path: `${opening.kind}s`,
          wallId: wall.id,
          openingId: opening.id,
          fixType: "opening_clash",
        });
      }

      if (index === 0) {
        // continue to junction checks
      } else {
        const previous = ordered[index - 1];
        const gap = opening.offset - (previous.offset + previous.width);
        const required = pairClearance(previous.kind, opening.kind);
        if (gap + tolerance < required) {
          createIssue(issues, {
            severity: "error",
            message: `Clash between ${previous.kind} ${previous.id} and ${opening.kind} ${opening.id} on wall ${wall.id}.`,
            path: `${opening.kind}s`,
            wallId: wall.id,
            openingId: opening.id,
            fixType: "opening_clash",
          });
        }
      }

      const openingStart = opening.offset;
      const openingEnd = opening.offset + opening.width;
      const junctionClash = junctionOffsets.find(
        (junctionOffset) =>
          openingEnd > junctionOffset - junctionClearance && openingStart < junctionOffset + junctionClearance,
      );
      if (junctionClash !== undefined) {
        createIssue(issues, {
          severity: "error",
          message: `${opening.kind === "door" ? "Door" : "Window"} ${opening.id} clashes with wall-node/junction zone on wall ${wall.id}.`,
          path: `${opening.kind}s`,
          wallId: wall.id,
          openingId: opening.id,
          fixType: "opening_clash",
        });
      }
    });
  });
}

function buildWallJunctionOffsets(walls: NormalizedWall[]) {
  const map = new Map<string, number[]>();
  const projectionTolerance = 0.08;
  const endFractionTolerance = 0.03;

  walls.forEach((wall) => {
    const offsets: number[] = [];
    const vx = wall.end[0] - wall.start[0];
    const vz = wall.end[1] - wall.start[1];
    const lenSq = vx * vx + vz * vz;
    if (lenSq < 0.0001 || wall.length < MIN_WALL_LENGTH) {
      map.set(wall.id, offsets);
      return;
    }

    walls.forEach((candidate) => {
      if (candidate.id === wall.id) {
        return;
      }

      [candidate.start, candidate.end].forEach((point) => {
        const px = point[0] - wall.start[0];
        const pz = point[1] - wall.start[1];
        const t = (px * vx + pz * vz) / lenSq;
        if (t <= endFractionTolerance || t >= 1 - endFractionTolerance) {
          return;
        }

        const cx = wall.start[0] + t * vx;
        const cz = wall.start[1] + t * vz;
        const perpendicularDistance = Math.hypot(point[0] - cx, point[1] - cz);
        if (perpendicularDistance > projectionTolerance) {
          return;
        }

        const offset = t * wall.length;
        const hasNearDuplicate = offsets.some((existingOffset) => Math.abs(existingOffset - offset) < 0.08);
        if (!hasNearDuplicate) {
          offsets.push(offset);
        }
      });
    });

    offsets.sort((a, b) => a - b);
    map.set(wall.id, offsets);
  });

  return map;
}

function normalizeInput(input: RawSceneInput, issues: ValidationIssue[]) {
  const meta = parseMeta(asObject(input.meta) as MetaInput | undefined, issues);
  const wallsArray = Array.isArray(input.walls) ? (input.walls as WallInput[]) : [];
  const slabsArray = Array.isArray(input.slabs) ? (input.slabs as SlabInput[]) : [];
  const labelsArray = Array.isArray(input.labels) ? (input.labels as LabelInput[]) : [];
  const doorsArray = Array.isArray(input.doors) ? (input.doors as DoorInput[]) : [];
  const windowsArray = Array.isArray(input.windows) ? (input.windows as WindowInput[]) : [];
  const openingsArray = Array.isArray(input.openings) ? (input.openings as OpeningInput[]) : [];
  const graphNodesArray = Array.isArray(input.graphNodes) ? (input.graphNodes as GraphNodeInput[]) : [];
  const columnsArray = Array.isArray(input.columns) ? (input.columns as ColumnInput[]) : [];

  if (Array.isArray(input.rooms) && input.rooms.length > 0) {
    createIssue(issues, {
      severity: "info",
      message: "The rooms array is deprecated. Use slabs[] for slab coordinates and labels[] for labels.",
      path: "rooms",
    });
  }

  const footprintBounds = buildFootprintBounds(wallsArray);
  const walls = parseWalls(wallsArray, meta, footprintBounds, issues);
  const rooms = parseSlabs(slabsArray, issues);
  const labels = parseLabels(labelsArray, issues);
  const mergedOpenings = [
    ...parseDoors(doorsArray, walls, issues),
    ...parseWindows(windowsArray, walls, issues),
    ...parseOpenings(openingsArray, walls, issues),
  ];
  const openings = dedupeOpenings(mergedOpenings, issues);
  const graphNodes = parseGraphNodes(graphNodesArray, walls, issues);
  const columns = parseColumns(columnsArray, graphNodes, meta, issues);
  detectOpeningClashes(openings, walls, issues);

  if (walls.length === 0) {
    return emptyScene("invalid");
  }

  return {
    meta,
    walls,
    rooms,
    labels,
    openings,
    graphNodes,
    columns,
    bounds: createBounds(walls.flatMap((wall) => [wall.start, wall.end])),
    readiness: issues.length > 0 ? "partial" : "valid",
  } satisfies SceneData;
}

export function validateAndNormalize(input: unknown): ValidationResult {
  const issues: ValidationIssue[] = [];
  const payload = asObject(input) as RawSceneInput | undefined;

  if (!payload) {
    createIssue(issues, {
      severity: "warning",
      message:
        "Upload a JSON model using meta, walls, slabs, labels, doors/windows/openings, graphNodes, and columns.",
      path: "root",
    });

    return {
      data: emptyScene("invalid"),
      issues,
      usedFallbackDataset: false,
    };
  }

  const normalized = normalizeInput(payload, issues);
  return {
    data: normalized,
    issues,
    usedFallbackDataset: false,
  };
}

export function applyOpeningClashFix(rawInput: RawSceneInput, issue: ValidationIssue): RawSceneInput | null {
  if (issue.fixType !== "opening_clash" || !issue.openingId || !issue.wallId) {
    return null;
  }

  const payload = asObject(rawInput) as RawSceneInput | undefined;
  if (!payload) {
    return null;
  }

  const cloned = JSON.parse(JSON.stringify(payload)) as RawSceneInput;
  const scratchIssues: ValidationIssue[] = [];

  const meta = parseMeta(asObject(cloned.meta) as MetaInput | undefined, scratchIssues);
  const wallsArray = Array.isArray(cloned.walls) ? (cloned.walls as WallInput[]) : [];
  const doorsArray = Array.isArray(cloned.doors) ? (cloned.doors as DoorInput[]) : [];
  const windowsArray = Array.isArray(cloned.windows) ? (cloned.windows as WindowInput[]) : [];
  const openingsArray = Array.isArray(cloned.openings) ? (cloned.openings as OpeningInput[]) : [];
  const footprintBounds = buildFootprintBounds(wallsArray);
  const walls = parseWalls(wallsArray, meta, footprintBounds, scratchIssues);
  const openings = [
    ...parseDoors(doorsArray, walls, scratchIssues),
    ...parseWindows(windowsArray, walls, scratchIssues),
    ...parseOpenings(openingsArray, walls, scratchIssues),
  ];
  const collisionIssues: ValidationIssue[] = [];
  const resolvedOpenings = resolveOpeningCollisions(openings, walls, collisionIssues);
  const target = resolvedOpenings.find(
    (opening) => opening.id === issue.openingId && opening.wallId === issue.wallId,
  );

  if (!target) {
    return null;
  }

  cloned.doors = (Array.isArray(cloned.doors) ? cloned.doors : []).map((doorEntry) => {
    const door = asObject(doorEntry);
    if (!door || door.id !== target.id || target.kind !== "door") {
      return doorEntry;
    }
    return {
      ...doorEntry,
      wallId: target.wallId,
      offset: Number(target.offset.toFixed(4)),
      width: Number(target.width.toFixed(4)),
      height: Number(target.height.toFixed(4)),
      swing: target.swing ?? door.swing,
    };
  });

  cloned.windows = (Array.isArray(cloned.windows) ? cloned.windows : []).map((windowEntry) => {
    const windowData = asObject(windowEntry);
    if (!windowData || windowData.id !== target.id || target.kind !== "window") {
      return windowEntry;
    }
    return {
      ...windowEntry,
      wallId: target.wallId,
      offset: Number(target.offset.toFixed(4)),
      width: Number(target.width.toFixed(4)),
      height: Number(target.height.toFixed(4)),
      sillHeight: Number((target.sillHeight ?? 1).toFixed(4)),
      windowType: target.panelType ?? "double",
    };
  });

  cloned.openings = (Array.isArray(cloned.openings) ? cloned.openings : []).map((openingEntry) => {
    const openingData = asObject(openingEntry);
    if (!openingData || openingData.id !== target.id) {
      return openingEntry;
    }
    const base = {
      ...openingEntry,
      wallId: target.wallId,
      offset: Number(target.offset.toFixed(4)),
      width: Number(target.width.toFixed(4)),
      height: Number(target.height.toFixed(4)),
      kind: target.kind,
    };
    if (target.kind === "window") {
      return {
        ...base,
        sillHeight: Number((target.sillHeight ?? 1).toFixed(4)),
        windowType: target.panelType ?? "double",
      };
    }
    return {
      ...base,
      swing: target.swing ?? openingData.swing,
    };
  });

  return cloned;
}
