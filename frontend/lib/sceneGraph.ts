import {
  buildWallMeshes,
  distance,
  getOpeningIntervals,
  getWallDirection,
  getWallNormal,
  pointAlongWall,
  to3DPoint,
} from "@/lib/geometry";
import { classifyWall } from "@/lib/materialEngine";
import type { NormalizedOpening, NormalizedWall, Point2D, Point3D, SceneData } from "@/lib/types";

export type RoomRenderNode = {
  id: string;
  name: string;
  polygon2D: Point2D[];
  fillColor: string;
  edgeColor: string;
  elevation: number;
  thickness: number;
  area: number;
  span: number;
  perimeter: number;
  closedLoop: boolean;
  connectedWallIds: string[];
  supportNodeIds: string[];
  labelPosition: Point3D;
};

export type WallRenderNode = {
  id: string;
  wall: NormalizedWall;
  bodyColor: string;
  edgeColor: string;
  topColor: string;
  position: Point3D;
  length: number;
  height: number;
  thickness: number;
};

export type OpeningRenderNode = {
  id: string;
  opening: NormalizedOpening;
  wall: NormalizedWall;
  bodyColor: string;
  normal: Point2D;
  dir: Point2D;
  angle: number;
  framePosition: Point3D;
  panelDepthOffset: number;
  hinge3D?: Point3D;
  hingeLeft3D?: Point3D;
  hingeRight3D?: Point3D;
};

export type StructuralNodeType = "terminal" | "corner" | "junction" | "inline";

export type StructuralNodeRenderNode = {
  id: string;
  position: Point3D;
  degree: number;
  type: StructuralNodeType;
  connectedWallIds: string[];
  loadBearingWallIds: string[];
  likelyColumn: boolean;
  color: string;
};

export type ColumnRenderNode = {
  id: string;
  nodeId: string;
  position: Point3D;
  width: number;
  depth: number;
  height: number;
  color: string;
  degree: number;
  connectedWallIds: string[];
};

export type SceneGraph = {
  rooms: RoomRenderNode[];
  walls: WallRenderNode[];
  openings: OpeningRenderNode[];
  nodes: StructuralNodeRenderNode[];
  columns: ColumnRenderNode[];
};

const roomPalette = [
  { fill: "#eae7df", edge: "#c8b89d" },
  { fill: "#f2efe8", edge: "#c6b08f" },
  { fill: "#ddd8cf", edge: "#a8a49e" },
  { fill: "#d9d9d9", edge: "#a7a7a7" },
];

function roomPaletteIndex(name: string, index: number) {
  const normalized = name.toLowerCase();
  if (normalized.includes("bath") || normalized.includes("toilet")) {
    return 3;
  }

  if (normalized.includes("kitchen")) {
    return 2;
  }

  if (normalized.includes("bed")) {
    return 1;
  }

  return index % 2;
}

function paletteForWall(wall: NormalizedWall, structuralView: boolean) {
  const classification = classifyWall(wall);
  if (!structuralView) {
    return { body: "#e0e0dd", edge: "#a4a4a0", top: "#d2d2ce" };
  }

  if (classification === "load_bearing") {
    return { body: "#8d9198", edge: "#4f5863", top: "#7b818a" };
  }

  if (classification === "partition") {
    return { body: "#ede9e1", edge: "#c5beb2", top: "#ddd7cd" };
  }

  return { body: "#c6b8a3", edge: "#918371", top: "#b9ab95" };
}

function renderedWallThickness(wall: NormalizedWall, structuralView: boolean) {
  if (!structuralView) {
    return wall.thickness;
  }

  const classification = classifyWall(wall);
  if (classification === "load_bearing") {
    return Math.max(wall.thickness + 0.04, 0.3);
  }

  if (classification === "partition") {
    return Math.min(wall.thickness, 0.12);
  }

  return Math.max(wall.thickness, 0.2);
}

function createWallNode(
  wall: NormalizedWall,
  id: string,
  start2D: Point2D,
  end2D: Point2D,
  centerY: number,
  height: number,
  thickness: number,
  palette: { body: string; edge: string; top: string },
): WallRenderNode | null {
  const dx = end2D[0] - start2D[0];
  const dz = end2D[1] - start2D[1];
  const length = Math.hypot(dx, dz);

  if (length <= 0.03 || height <= 0.03) {
    return null;
  }

  return {
    id,
    wall,
    bodyColor: palette.body,
    edgeColor: palette.edge,
    topColor: palette.top,
    position: [(start2D[0] + end2D[0]) / 2, centerY, (start2D[1] + end2D[1]) / 2],
    length,
    height,
    thickness,
  };
}

function buildWallNodes(wall: NormalizedWall, openings: NormalizedOpening[], structuralView: boolean) {
  const palette = paletteForWall(wall, structuralView);
  const thickness = renderedWallThickness(wall, structuralView);
  const baseSegments = buildWallMeshes(wall, openings).flatMap((segment, index) => {
    const node = createWallNode(
      wall,
      `${wall.id}-segment-${index}`,
      segment.start,
      segment.end,
      wall.height / 2,
      wall.height,
      thickness,
      palette,
    );
    return node ? [node] : [];
  });

  const intervalNodes = getOpeningIntervals(wall, openings).flatMap((interval, index) => {
    const sillHeight = interval.data.kind === "window" ? interval.data.sillHeight ?? 1 : 0;
    const start2D = pointAlongWall(wall.start, wall.end, interval.start);
    const end2D = pointAlongWall(wall.start, wall.end, interval.end);
    const nodes: WallRenderNode[] = [];

    if (sillHeight > 0.03) {
      const bottom = createWallNode(
        wall,
        `${wall.id}-bottom-${index}`,
        start2D,
        end2D,
        sillHeight / 2,
        sillHeight,
        thickness,
        palette,
      );
      if (bottom) {
        nodes.push(bottom);
      }
    }

    const openingTop = sillHeight + interval.data.height;
    const topHeight = wall.height - openingTop;
    if (topHeight > 0.03) {
      const top = createWallNode(
        wall,
        `${wall.id}-top-${index}`,
        start2D,
        end2D,
        openingTop + topHeight / 2,
        topHeight,
        thickness,
        palette,
      );
      if (top) {
        nodes.push(top);
      }
    }

    return nodes;
  });

  return [...baseSegments, ...intervalNodes];
}

function computeDoorTransform(wall: NormalizedWall, opening: NormalizedOpening) {
  const dir = getWallDirection(wall);
  const normal = getWallNormal(wall);
  const basePoint = pointAlongWall(wall.start, wall.end, opening.offset);
  const hinge = opening.swing === "right"
    ? [basePoint[0] + dir[0] * opening.width, basePoint[1] + dir[1] * opening.width] as Point2D
    : basePoint;
  const frameCenter = pointAlongWall(wall.start, wall.end, opening.offset + opening.width / 2);

  return {
    dir,
    normal,
    angle: wall.angle,
    hinge3D: to3DPoint(hinge, 0),
    frameCenter3D: to3DPoint(frameCenter, opening.height / 2),
  };
}

function pointKey(point: Point2D) {
  return `${point[0].toFixed(3)}:${point[1].toFixed(3)}`;
}

function roomPerimeter(polygon: Point2D[]) {
  if (polygon.length < 2) {
    return 0;
  }

  let perimeter = 0;
  for (let index = 0; index < polygon.length; index += 1) {
    const start = polygon[index];
    const end = polygon[(index + 1) % polygon.length];
    perimeter += Math.hypot(end[0] - start[0], end[1] - start[1]);
  }

  return perimeter;
}

function slabThicknessFromSpan(span: number) {
  return Math.max(0.12, Math.min(0.24, 0.12 + Math.max(span - 3, 0) * 0.018));
}

function buildNodeKeyMap(nodes: StructuralNodeRenderNode[]) {
  const map = new Map<string, StructuralNodeRenderNode[]>();

  nodes.forEach((node) => {
    const key = `${node.position[0].toFixed(3)}:${node.position[2].toFixed(3)}`;
    const list = map.get(key) ?? [];
    list.push(node);
    map.set(key, list);
  });

  return map;
}

function closestNodeId(nodeMap: Map<string, StructuralNodeRenderNode[]>, point: Point2D) {
  const direct = nodeMap.get(pointKey(point));
  if (direct?.[0]) {
    return direct[0].id;
  }

  let bestNodeId: string | undefined;
  let bestDistance = Number.POSITIVE_INFINITY;

  nodeMap.forEach((nodes) => {
    nodes.forEach((node) => {
      const dx = node.position[0] - point[0];
      const dz = node.position[2] - point[1];
      const candidate = Math.hypot(dx, dz);
      if (candidate < bestDistance) {
        bestDistance = candidate;
        bestNodeId = node.id;
      }
    });
  });

  return bestDistance <= 0.08 ? bestNodeId : undefined;
}

function roomWalls(scene: SceneData, roomPolygon: Point2D[]) {
  const tolerance = 0.08;

  return scene.walls
    .filter((wall) => {
      for (let index = 0; index < roomPolygon.length; index += 1) {
        const start = roomPolygon[index];
        const end = roomPolygon[(index + 1) % roomPolygon.length];
        const directMatch =
          Math.hypot(wall.start[0] - start[0], wall.start[1] - start[1]) < tolerance &&
          Math.hypot(wall.end[0] - end[0], wall.end[1] - end[1]) < tolerance;
        const reverseMatch =
          Math.hypot(wall.start[0] - end[0], wall.start[1] - end[1]) < tolerance &&
          Math.hypot(wall.end[0] - start[0], wall.end[1] - start[1]) < tolerance;
        if (directMatch || reverseMatch) {
          return true;
        }
      }

      return false;
    })
    .map((wall) => wall.id);
}

function classifyNodeType(directions: Point2D[]): StructuralNodeType {
  if (directions.length <= 1) {
    return "terminal";
  }

  if (directions.length >= 3) {
    return "junction";
  }

  const [first, second] = directions;
  const dot = first[0] * second[0] + first[1] * second[1];

  return Math.abs(dot) > 0.92 ? "inline" : "corner";
}

const NODE_MERGE_TOLERANCE = 0.08;
const NODE_DIRECTION_ALIGNMENT = 0.985;

function directionDot(a: Point2D, b: Point2D) {
  return a[0] * b[0] + a[1] * b[1];
}

function appendUniqueDirection(directions: Point2D[], direction: Point2D) {
  const duplicate = directions.some((entry) => Math.abs(directionDot(entry, direction)) >= NODE_DIRECTION_ALIGNMENT);
  if (!duplicate) {
    directions.push(direction);
  }
}

export function buildStructuralNodes(scene: SceneData): StructuralNodeRenderNode[] {
  if (scene.graphNodes.length > 0) {
    return scene.graphNodes.map((node) => ({
      id: node.id,
      position: node.position,
      degree: node.degree,
      type: node.type,
      connectedWallIds: node.connectedWallIds,
      loadBearingWallIds: node.loadBearingWallIds,
      likelyColumn: node.likelyColumn,
      color: node.color,
    }));
  }

  const groups: Array<{
    point: Point2D;
    pointCount: number;
    wallIds: Set<string>;
    loadBearingWallIds: Set<string>;
    directions: Point2D[];
  }> = [];

  scene.walls.forEach((wall) => {
    const direction = getWallDirection(wall);
    const loadBearing = classifyWall(wall) === "load_bearing";

    [
      { point: wall.start, direction: [-direction[0], -direction[1]] as Point2D },
      { point: wall.end, direction },
    ].forEach(({ point, direction: nodeDirection }) => {
      let entry = groups.find((candidate) => distance(candidate.point, point) <= NODE_MERGE_TOLERANCE);
      if (!entry) {
        entry = {
          point: [point[0], point[1]],
          pointCount: 1,
          wallIds: new Set<string>(),
          loadBearingWallIds: new Set<string>(),
          directions: [],
        };
        groups.push(entry);
      } else {
        const nextCount = entry.pointCount + 1;
        entry.point = [
          (entry.point[0] * entry.pointCount + point[0]) / nextCount,
          (entry.point[1] * entry.pointCount + point[1]) / nextCount,
        ];
        entry.pointCount = nextCount;
      }

      entry.wallIds.add(wall.id);
      if (loadBearing) {
        entry.loadBearingWallIds.add(wall.id);
      }
      appendUniqueDirection(entry.directions, nodeDirection);
    });
  });

  const orderedGroups = [...groups].sort((a, b) => {
    if (Math.abs(a.point[1] - b.point[1]) > 0.001) {
      return a.point[1] - b.point[1];
    }

    return a.point[0] - b.point[0];
  });

  return orderedGroups.map((entry, index) => {
    const connectedWallIds = Array.from(entry.wallIds);
    const loadBearingWallIds = Array.from(entry.loadBearingWallIds);
    const degree = connectedWallIds.length;
    const type = classifyNodeType(entry.directions);
    const likelyColumn = degree >= 3 || loadBearingWallIds.length >= 2;
    const color = likelyColumn ? "#4f6678" : type === "junction" ? "#5f7484" : "#8091a0";

    return {
      id: `node-${index + 1}`,
      position: [entry.point[0], 0.2, entry.point[1]],
      degree,
      type,
      connectedWallIds,
      loadBearingWallIds,
      likelyColumn,
      color,
    };
  });
}

export function buildColumns(scene: SceneData, nodes: StructuralNodeRenderNode[]): ColumnRenderNode[] {
  if (scene.columns.length > 0) {
    return scene.columns.map((column) => ({
      id: column.id,
      nodeId: column.nodeId,
      position: column.position,
      width: column.width,
      depth: column.depth,
      height: column.height,
      color: column.color,
      degree: column.degree,
      connectedWallIds: column.connectedWallIds,
    }));
  }

  return nodes
    .filter((node) => node.likelyColumn)
    .map((node, index) => {
      const section = node.loadBearingWallIds.length >= 2 ? 0.34 : node.degree >= 4 ? 0.32 : 0.28;

      return {
        id: `column-${index + 1}`,
        nodeId: node.id,
        position: [node.position[0], scene.meta.wallHeight / 2, node.position[2]],
        width: section,
        depth: section,
        height: Math.max(2.4, scene.meta.wallHeight),
        color: "#667786",
        degree: node.degree,
        connectedWallIds: node.connectedWallIds,
      };
    });
}

export function buildSceneGraph(scene: SceneData, structuralView: boolean): SceneGraph {
  const wallMap = new Map(scene.walls.map((wall) => [wall.id, wall]));
  const nodes = buildStructuralNodes(scene);
  const columns = buildColumns(scene, nodes);
  const nodeMap = buildNodeKeyMap(nodes);

  const rooms: RoomRenderNode[] = scene.rooms.map((room, index) => {
    const palette = roomPalette[roomPaletteIndex(room.name, index)];
    const connectedWallIds = roomWalls(scene, room.polygon2D);
    const supportNodeIds = Array.from(
      new Set(
        room.polygon2D
          .map((point) => closestNodeId(nodeMap, point))
          .filter((entry): entry is string => Boolean(entry)),
      ),
    );

    return {
      id: room.id,
      name: room.name,
      polygon2D: room.polygon2D,
      fillColor: palette.fill,
      edgeColor: palette.edge,
      elevation: 0.008 + (index % 3) * 0.003,
      thickness: slabThicknessFromSpan(room.span),
      area: room.area,
      span: room.span,
      perimeter: roomPerimeter(room.polygon2D),
      closedLoop: room.polygon2D.length >= 3 && connectedWallIds.length >= 3,
      connectedWallIds,
      supportNodeIds,
      labelPosition: [room.centroid[0], 0.06, room.centroid[2]],
    };
  });

  const openingsByWall = new Map<string, NormalizedOpening[]>();
  scene.openings.forEach((opening) => {
    const list = openingsByWall.get(opening.wallId) ?? [];
    list.push(opening);
    openingsByWall.set(opening.wallId, list);
  });

  const walls = scene.walls.flatMap((wall) => buildWallNodes(wall, openingsByWall.get(wall.id) ?? [], structuralView));

  const openings: OpeningRenderNode[] = scene.openings.flatMap<OpeningRenderNode>((opening) => {
    const wall = wallMap.get(opening.wallId);
    if (!wall) {
      return [];
    }

    const dir = getWallDirection(wall);
    const normal = getWallNormal(wall);
    const panelDepthOffset = Math.max(0.02, wall.thickness / 2 - 0.02);

    if (opening.kind === "door") {
      const transform = computeDoorTransform(wall, opening);
      const framePosition: Point3D = [
        transform.frameCenter3D[0] + normal[0] * panelDepthOffset,
        transform.frameCenter3D[1],
        transform.frameCenter3D[2] + normal[1] * panelDepthOffset,
      ];
      const hinge3D: Point3D = [
        transform.hinge3D[0] + normal[0] * panelDepthOffset,
        0,
        transform.hinge3D[2] + normal[1] * panelDepthOffset,
      ];

      return [{
        id: opening.id,
        opening,
        wall,
        bodyColor: paletteForWall(wall, structuralView).body,
        normal,
        dir,
        angle: transform.angle,
        framePosition,
        panelDepthOffset,
        hinge3D,
      }];
    }

    const leftBase = pointAlongWall(wall.start, wall.end, opening.offset);
    const rightBase = pointAlongWall(wall.start, wall.end, opening.offset + opening.width);
    const base = pointAlongWall(wall.start, wall.end, opening.offset + opening.width / 2);
    const y = (opening.sillHeight ?? 1) + opening.height / 2;
    const framePosition: Point3D = [
      base[0] + normal[0] * panelDepthOffset,
      y,
      base[1] + normal[1] * panelDepthOffset,
    ];
    const hingeLeft3D: Point3D = [
      leftBase[0] + normal[0] * panelDepthOffset,
      y - opening.height / 2,
      leftBase[1] + normal[1] * panelDepthOffset,
    ];
    const hingeRight3D: Point3D = [
      rightBase[0] + normal[0] * panelDepthOffset,
      y - opening.height / 2,
      rightBase[1] + normal[1] * panelDepthOffset,
    ];

    return [{
      id: opening.id,
      opening,
      wall,
      bodyColor: paletteForWall(wall, structuralView).body,
      normal,
      dir,
      angle: wall.angle,
      framePosition,
      panelDepthOffset,
      hingeLeft3D,
      hingeRight3D,
    }];
  });

  return { rooms, walls, openings, nodes, columns };
}
