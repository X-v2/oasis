import "server-only";

import type {
  HeuristicConfidence,
  HeuristicReport,
  HeuristicSeverity,
  HeuristicSuggestion,
  NormalizedRoom,
  NormalizedWall,
  Point2D,
  SceneData,
  StructuralRole,
  WallStructuralProfile,
} from "@/lib/types";
import { distanceToSegment, pointInPolygon } from "@/lib/geometry";
import { buildColumns, buildStructuralNodes } from "@/lib/sceneGraph";

const THICK_SUPPORT_M = 0.18;
const SUPPORT_SPAN_LIMIT_M = 5.0;
const MASONRY_FALLBACK_SPAN_LIMIT_M = 4.5;
const OPENING_RATIO_LIMIT = 0.25;

const AXIS_CHAIN_TOLERANCE_M = 0.18;
const AXIS_CHAIN_GAP_TOLERANCE_M = 0.35;
const MAJOR_CHAIN_MIN_COVERAGE_M = 4.0;
const MAJOR_CHAIN_MIN_WALLS = 3;
const ALIGN_TOLERANCE_M = 0.15;

const LOAD_PATH_CRITICAL_OFFSET_M = 0.5;
const MAX_PLAUSIBLE_LOAD_PATH_OFFSET_M = 1.5;
const MAX_CHAIN_ENDPOINT_GAP_M = 2.5;

const EDGE_MATCH_TOLERANCE_M = 0.1;
const NODE_MATCH_TOLERANCE_M = 0.08;
const SMALL_ROOM_AREA_M2 = 3.5;
const COLUMN_GAP_ROOM_SPAN_LIMIT_M = 5.2;

type WallOrientation = "horizontal" | "vertical";

type AxisInterval = {
  min: number;
  max: number;
};

type AxisGroup = {
  orientation: WallOrientation;
  axis: number;
  wallIds: string[];
  intervals: AxisInterval[];
};

type RoomSupportSpan = {
  roomId: string;
  supportSpan: number;
  supportWallIds: string[];
  axis: WallOrientation | "none";
  inferred: boolean;
  confidence: HeuristicConfidence;
  fallbackReason?: string;
};

function wallOrientation(wall: NormalizedWall): WallOrientation {
  const dx = Math.abs(wall.end[0] - wall.start[0]);
  const dz = Math.abs(wall.end[1] - wall.start[1]);
  return dx >= dz ? "horizontal" : "vertical";
}

function wallAxisCoordinate(wall: NormalizedWall, orientation: WallOrientation) {
  return orientation === "horizontal" ? (wall.start[1] + wall.end[1]) / 2 : (wall.start[0] + wall.end[0]) / 2;
}

function wallRunInterval(wall: NormalizedWall, orientation: WallOrientation): AxisInterval {
  if (orientation === "horizontal") {
    return {
      min: Math.min(wall.start[0], wall.end[0]),
      max: Math.max(wall.start[0], wall.end[0]),
    };
  }
  return {
    min: Math.min(wall.start[1], wall.end[1]),
    max: Math.max(wall.start[1], wall.end[1]),
  };
}

function overlapLength(a: AxisInterval, b: AxisInterval) {
  return Math.max(0, Math.min(a.max, b.max) - Math.max(a.min, b.min));
}

function mergeCoverage(intervals: AxisInterval[]) {
  if (intervals.length === 0) {
    return 0;
  }

  const ordered = [...intervals].sort((a, b) => a.min - b.min);
  let coverage = 0;
  let currentMin = ordered[0].min;
  let currentMax = ordered[0].max;

  for (let index = 1; index < ordered.length; index += 1) {
    const next = ordered[index];
    if (next.min <= currentMax + AXIS_CHAIN_GAP_TOLERANCE_M) {
      currentMax = Math.max(currentMax, next.max);
      continue;
    }
    coverage += currentMax - currentMin;
    currentMin = next.min;
    currentMax = next.max;
  }

  coverage += currentMax - currentMin;
  return coverage;
}

function endpointDistance(a: Point2D, b: Point2D) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function approximatePointMatch(a: Point2D, b: Point2D, tolerance = EDGE_MATCH_TOLERANCE_M) {
  return endpointDistance(a, b) <= tolerance;
}

function wallMatchesEdge(wall: NormalizedWall, edgeStart: Point2D, edgeEnd: Point2D) {
  const direct = approximatePointMatch(wall.start, edgeStart) && approximatePointMatch(wall.end, edgeEnd);
  const reverse = approximatePointMatch(wall.start, edgeEnd) && approximatePointMatch(wall.end, edgeStart);
  return direct || reverse;
}

function wallDistanceToRoomEdges(wall: NormalizedWall, room: NormalizedRoom) {
  const points: Point2D[] = [wall.start, wall.end, [wall.midpoint[0], wall.midpoint[2]]];
  let best = Number.POSITIVE_INFINITY;

  points.forEach((point) => {
    for (let index = 0; index < room.polygon2D.length; index += 1) {
      const start = room.polygon2D[index];
      const end = room.polygon2D[(index + 1) % room.polygon2D.length];
      const candidate = distanceToSegment(point, start, end);
      if (candidate < best) {
        best = candidate;
      }
    }
  });

  return best;
}

function minEndpointDistanceBetweenWalls(first: NormalizedWall, second: NormalizedWall) {
  const pairs: Array<[Point2D, Point2D]> = [
    [first.start, second.start],
    [first.start, second.end],
    [first.end, second.start],
    [first.end, second.end],
  ];
  return Math.min(...pairs.map(([a, b]) => endpointDistance(a, b)));
}

function collectAxisGroups(scene: SceneData) {
  const groups: AxisGroup[] = [];

  scene.walls.forEach((wall) => {
    if (wall.length < 1.2) {
      return;
    }
    const orientation = wallOrientation(wall);
    const axis = wallAxisCoordinate(wall, orientation);
    const interval = wallRunInterval(wall, orientation);
    let group = groups.find(
      (entry) => entry.orientation === orientation && Math.abs(entry.axis - axis) <= AXIS_CHAIN_TOLERANCE_M,
    );
    if (!group) {
      group = { orientation, axis, wallIds: [], intervals: [] };
      groups.push(group);
    } else {
      group.axis = (group.axis * group.wallIds.length + axis) / (group.wallIds.length + 1);
    }
    group.wallIds.push(wall.id);
    group.intervals.push(interval);
  });

  return groups;
}

function roleWeight(role: StructuralRole) {
  if (role === "primary_support") {
    return 3;
  }
  if (role === "secondary_support") {
    return 2;
  }
  return 1;
}

function severityWeight(severity: HeuristicSeverity) {
  if (severity === "high") {
    return 3;
  }
  if (severity === "medium") {
    return 2;
  }
  return 1;
}

function buildWallProfiles(scene: SceneData) {
  const axisGroups = collectAxisGroups(scene);
  const partOfAxisChainIds = new Set<string>();
  const partOfMajorAxisChainIds = new Set<string>();

  axisGroups.forEach((group) => {
    const coverage = mergeCoverage(group.intervals);
    const isChain = group.wallIds.length >= 2 && coverage >= 2.4;
    const isMajor = isChain && group.wallIds.length >= MAJOR_CHAIN_MIN_WALLS && coverage >= MAJOR_CHAIN_MIN_COVERAGE_M;
    if (isChain) {
      group.wallIds.forEach((wallId) => partOfAxisChainIds.add(wallId));
    }
    if (isMajor) {
      group.wallIds.forEach((wallId) => partOfMajorAxisChainIds.add(wallId));
    }
  });

  const profiles: WallStructuralProfile[] = scene.walls.map((wall) => {
    const evidence: string[] = [];
    const typeOuter = wall.type === "outer";
    const thicknessSupport = wall.thickness >= THICK_SUPPORT_M;
    const partOfAxisChain = partOfAxisChainIds.has(wall.id);
    const partOfMajorAxisChain = partOfMajorAxisChainIds.has(wall.id);

    if (typeOuter) {
      evidence.push(`outer wall (${wall.type})`);
    }
    if (thicknessSupport) {
      evidence.push(`thickness ${wall.thickness.toFixed(2)}m >= ${THICK_SUPPORT_M.toFixed(2)}m`);
    }
    if (partOfAxisChain) {
      evidence.push("part of continuous axis chain");
    }

    const isLoadBearing = typeOuter || thicknessSupport || partOfAxisChain;
    const role: StructuralRole = isLoadBearing
      ? partOfMajorAxisChain
        ? "primary_support"
        : "secondary_support"
      : "partition";

    if (!isLoadBearing) {
      evidence.push("no load-bearing trigger matched");
    }

    return {
      wallId: wall.id,
      role,
      isLoadBearing,
      partOfAxisChain,
      partOfMajorAxisChain,
      evidence,
    };
  });

  return { profiles, axisGroups };
}

function roomSupportSpan(room: NormalizedRoom, scene: SceneData, profileMap: Map<string, WallStructuralProfile>): RoomSupportSpan {
  const edgeWalls: NormalizedWall[] = [];
  for (let index = 0; index < room.polygon2D.length; index += 1) {
    const edgeStart = room.polygon2D[index];
    const edgeEnd = room.polygon2D[(index + 1) % room.polygon2D.length];
    const matched = scene.walls.find((wall) => wallMatchesEdge(wall, edgeStart, edgeEnd));
    if (matched) {
      edgeWalls.push(matched);
    }
  }

  const supports = edgeWalls.filter((wall) => profileMap.get(wall.id)?.isLoadBearing);
  const nearbySupports = scene.walls
    .filter((wall) => profileMap.get(wall.id)?.isLoadBearing)
    .filter((wall) => {
      const midpoint: Point2D = [wall.midpoint[0], wall.midpoint[2]];
      return pointInPolygon(midpoint, room.polygon2D) || wallDistanceToRoomEdges(wall, room) <= 0.22;
    });
  const supportPool = supports.length >= 2 ? supports : nearbySupports;
  if (supportPool.length < 2) {
    return {
      roomId: room.id,
      supportSpan: room.span,
      supportWallIds: supportPool.map((wall) => wall.id),
      axis: "none",
      inferred: true,
      confidence: "low",
      fallbackReason: "No sufficient load-bearing supports detected; conservative geometric span assumption used.",
    };
  }

  let bestSpan = 0;
  let bestPair: [string, string] | null = null;
  let bestAxis: WallOrientation | "none" = "none";

  const byOrientation = {
    horizontal: supportPool.filter((wall) => wallOrientation(wall) === "horizontal"),
    vertical: supportPool.filter((wall) => wallOrientation(wall) === "vertical"),
  };

  (["horizontal", "vertical"] as const).forEach((orientation) => {
    const list = byOrientation[orientation];
    for (let i = 0; i < list.length; i += 1) {
      for (let j = i + 1; j < list.length; j += 1) {
        const first = list[i];
        const second = list[j];
        const firstAxis = wallAxisCoordinate(first, orientation);
        const secondAxis = wallAxisCoordinate(second, orientation);
        const span = Math.abs(firstAxis - secondAxis);
        if (span > bestSpan) {
          bestSpan = span;
          bestPair = [first.id, second.id];
          bestAxis = orientation;
        }
      }
    }
  });

  if (!bestPair) {
    return {
      roomId: room.id,
      supportSpan: room.span,
      supportWallIds: supportPool.map((wall) => wall.id),
      axis: "none",
      inferred: true,
      confidence: "medium",
      fallbackReason: `Detected candidate supports (${supportPool.map((wall) => wall.id).join(", ")}), but no parallel aligned pair forming a valid span. Conservative geometric span assumption used.`,
    };
  }

  return {
    roomId: room.id,
    supportSpan: bestSpan,
    supportWallIds: [...bestPair],
    axis: bestAxis,
    inferred: false,
    confidence: "high",
  };
}

function endpointDegrees(scene: SceneData) {
  const points: Point2D[] = [];
  const assignments = new Map<string, { start: number; end: number }>();

  function findPointIndex(point: Point2D) {
    for (let index = 0; index < points.length; index += 1) {
      if (endpointDistance(point, points[index]) <= NODE_MATCH_TOLERANCE_M) {
        return index;
      }
    }
    points.push([point[0], point[1]]);
    return points.length - 1;
  }

  scene.walls.forEach((wall) => {
    const start = findPointIndex(wall.start);
    const end = findPointIndex(wall.end);
    assignments.set(wall.id, { start, end });
  });

  const degreeCounts = new Map<number, number>();
  assignments.forEach((entry) => {
    degreeCounts.set(entry.start, (degreeCounts.get(entry.start) ?? 0) + 1);
    degreeCounts.set(entry.end, (degreeCounts.get(entry.end) ?? 0) + 1);
  });

  const wallEndpointDegree = new Map<string, { startDegree: number; endDegree: number }>();
  assignments.forEach((entry, wallId) => {
    wallEndpointDegree.set(wallId, {
      startDegree: degreeCounts.get(entry.start) ?? 1,
      endDegree: degreeCounts.get(entry.end) ?? 1,
    });
  });

  return wallEndpointDegree;
}

function suggestionComparator(a: HeuristicSuggestion, b: HeuristicSuggestion) {
  const severityDelta = severityWeight(b.severity) - severityWeight(a.severity);
  if (severityDelta !== 0) {
    return severityDelta;
  }
  return a.id.localeCompare(b.id);
}

export function buildLayoutHeuristicReport(scene: SceneData): HeuristicReport {
  const suggestions: HeuristicSuggestion[] = [];
  const { profiles } = buildWallProfiles(scene);
  const profileMap = new Map(profiles.map((profile) => [profile.wallId, profile]));
  const openingsByWall = new Map<string, number>();
  scene.openings.forEach((opening) => {
    openingsByWall.set(opening.wallId, (openingsByWall.get(opening.wallId) ?? 0) + opening.width);
  });

  const roomSupportMap = new Map<string, RoomSupportSpan>();
  scene.rooms.forEach((room) => {
    roomSupportMap.set(room.id, roomSupportSpan(room, scene, profileMap));
  });
  const nodes = buildStructuralNodes(scene);
  const columns = buildColumns(scene, nodes);

  let counter = 1;
  const nextId = () => `heur-${counter++}`;

  scene.rooms.forEach((room) => {
    const support = roomSupportMap.get(room.id);
    if (!support || support.supportSpan <= SUPPORT_SPAN_LIMIT_M) {
      return;
    }
    const severity: HeuristicSeverity = support.supportSpan > 6 ? "high" : "medium";
    const nearestPartition = scene.walls.find(
      (wall) => wall.roomId === room.id && profileMap.get(wall.id)?.role === "partition",
    );
    const adjustmentHint = nearestPartition ? `, or shifting wall ${nearestPartition.id} by up to 0.8m` : "";
    const axisText = support.axis === "vertical" ? "X-axis" : support.axis === "horizontal" ? "Z-axis" : "room span axis";
    const midSpanDistance = (support.supportSpan / 2).toFixed(2);
    const supportPair = support.supportWallIds.join(" and ");
    suggestions.push({
      id: nextId(),
      type: "SPAN_FIX",
      severity,
      confidence: support.confidence,
      location: room.name,
      issue: `Support span ${support.supportSpan.toFixed(2)}m exceeds ${SUPPORT_SPAN_LIMIT_M.toFixed(1)}m comfort limit.`,
      suggestion: `Add a beam/support line between ${supportPair || "detected supports"} at mid-span (~${midSpanDistance}m from ${support.supportWallIds[0] ?? "nearest support"} along ${axisText})${adjustmentHint}.`,
      impact: "Can reduce need for thicker slab or long-span system escalation.",
      impactBasis: "Span correction may avoid slab thickening/long-span upgrade in low-rise layouts.",
      evidence: [
        support.inferred
          ? support.fallbackReason ?? "No sufficient load-bearing supports detected; conservative geometric span assumption used."
          : `Measured between load-bearing supports (${support.axis} axis).`,
        `Support walls: ${support.supportWallIds.join(", ") || "none"}.`,
      ],
      assumptions: ["Single-floor heuristic model; no full frame analysis."],
      relatedWallIds: support.supportWallIds,
      relatedRoomId: room.id,
      metrics: {
        supportSpanM: Number(support.supportSpan.toFixed(3)),
        spanLimitM: SUPPORT_SPAN_LIMIT_M,
      },
    });
  });

  scene.rooms.forEach((room) => {
    const support = roomSupportMap.get(room.id);
    if (!support || support.supportSpan < COLUMN_GAP_ROOM_SPAN_LIMIT_M) {
      return;
    }

    const roomColumns = columns.filter((column) => pointInPolygon([column.position[0], column.position[2]], room.polygon2D));
    if (roomColumns.length > 0) {
      return;
    }

    suggestions.push({
      id: nextId(),
      type: "COLUMN_GAP",
      severity: support.supportSpan > 6 ? "high" : "medium",
      confidence: support.confidence === "high" ? "medium" : "low",
      location: room.name,
      issue: `Large supported span (${support.supportSpan.toFixed(2)}m) has no inferred column position inside the room.`,
      suggestion: "Introduce at least one intermediate support/column position or beam transfer line near room center.",
      impact: "Improves load distribution and reduces slab flexural demand over long clear spans.",
      impactBasis: "Long spans without intermediate support elevate bending demand and deflection risk.",
      evidence: [
        `support_span=${support.supportSpan.toFixed(2)}m`,
        `in-room inferred columns=${roomColumns.length}`,
        `support walls: ${support.supportWallIds.join(", ") || "none"}`,
      ],
      assumptions: ["Column inference is graph-based and may under-detect unconventional support systems."],
      relatedWallIds: support.supportWallIds,
      relatedRoomId: room.id,
      metrics: {
        supportSpanM: Number(support.supportSpan.toFixed(3)),
        inferredColumnCount: roomColumns.length,
      },
    });
  });

  const loadBearingWalls = scene.walls.filter((wall) => profileMap.get(wall.id)?.isLoadBearing);
  for (let i = 0; i < loadBearingWalls.length; i += 1) {
    for (let j = i + 1; j < loadBearingWalls.length; j += 1) {
      const first = loadBearingWalls[i];
      const second = loadBearingWalls[j];
      const orientation = wallOrientation(first);
      if (wallOrientation(second) !== orientation) {
        continue;
      }
      const firstInterval = wallRunInterval(first, orientation);
      const secondInterval = wallRunInterval(second, orientation);
      const overlap = overlapLength(firstInterval, secondInterval);
      if (overlap < 0.6) {
        continue;
      }

      const offset = Math.abs(wallAxisCoordinate(first, orientation) - wallAxisCoordinate(second, orientation));
      if (offset <= 0.01) {
        continue;
      }
      if (offset > MAX_PLAUSIBLE_LOAD_PATH_OFFSET_M) {
        continue;
      }
      const endpointGap = minEndpointDistanceBetweenWalls(first, second);
      if (endpointGap > MAX_CHAIN_ENDPOINT_GAP_M) {
        continue;
      }

      const firstRole = profileMap.get(first.id)?.role ?? "partition";
      const secondRole = profileMap.get(second.id)?.role ?? "partition";

      if (offset < ALIGN_TOLERANCE_M) {
        suggestions.push({
          id: nextId(),
          type: "ALIGNMENT_FIX",
          severity: offset > 0.1 ? "medium" : "low",
          confidence: "high",
          location: `${first.id} / ${second.id}`,
          issue: `Near-aligned ${orientation} supports have ${offset.toFixed(2)}m axis offset.`,
          suggestion: `Snap ${first.id} and ${second.id} to a shared axis to create continuous vertical load path from slab to foundation.`,
          impact: "Improves load transfer continuity and reduces local eccentricity risk.",
          impactBasis: "Axis alignment strengthens load-path continuity through support chain.",
          evidence: [
            `${first.id} role: ${firstRole.replace("_", " ")}`,
            `${second.id} role: ${secondRole.replace("_", " ")}`,
            `Overlap along run axis: ${overlap.toFixed(2)}m.`,
            `Endpoint chain gap: ${endpointGap.toFixed(2)}m.`,
          ],
          assumptions: ["Single-level plan interpreted as vertical load-path proxy."],
          relatedWallIds: [first.id, second.id],
          metrics: {
            axisOffsetM: Number(offset.toFixed(3)),
            overlapM: Number(overlap.toFixed(3)),
          },
        });
      }

      if (offset >= 0.03) {
        const severity: HeuristicSeverity = offset > LOAD_PATH_CRITICAL_OFFSET_M ? "high" : "medium";
        suggestions.push({
          id: nextId(),
          type: "LOAD_PATH_GAP",
          severity,
          confidence: "high",
          location: `${first.id} / ${second.id}`,
          issue: `Discontinuous load path detected with ${offset.toFixed(2)}m support-axis offset.`,
          suggestion:
            "Realign the support line or introduce transfer element so wall-to-support load transfer remains continuous.",
          impact: "Reduces inefficient load transfer and localized stress concentration risk.",
          impactBasis: "Severity is scaled by alignment offset distance and assumed affected level count.",
          evidence: [
            `Offset: ${offset.toFixed(3)}m.`,
            `Endpoint chain gap: ${endpointGap.toFixed(3)}m.`,
            "Affected levels assumed = 1 (single-floor default).",
            `Roles: ${(profileMap.get(first.id)?.role ?? "partition").replace("_", " ")} and ${(profileMap.get(second.id)?.role ?? "partition").replace("_", " ")}.`,
          ],
          assumptions: ["No floor-stack metadata supplied, so levels_affected defaults to 1."],
          relatedWallIds: [first.id, second.id],
          metrics: {
            axisOffsetM: Number(offset.toFixed(3)),
            endpointGapM: Number(endpointGap.toFixed(3)),
            levelsAffected: 1,
          },
        });
      }
    }
  }

  const partitionCandidates = scene.walls
    .filter((wall) => profileMap.get(wall.id)?.role === "partition")
    .sort((a, b) => b.length - a.length)
    .slice(0, 4);

  partitionCandidates.forEach((wall) => {
    const openingRatio = (openingsByWall.get(wall.id) ?? 0) / Math.max(wall.length, 0.001);
    suggestions.push({
      id: nextId(),
      type: "MATERIAL_OPT",
      severity: "low",
      confidence: "high",
      location: `Wall ${wall.id}`,
      issue: `Partition wall ${wall.id} is a non-primary structural element (opening ratio ${openingRatio.toFixed(2)}).`,
      suggestion: `Use AAC Blocks as a viable alternative to conventional brick for wall ${wall.id}.`,
      impact: "Typically lighter and often lower installed cost for partition use.",
      impactBasis: "AAC vs brick is typically ~15-25% lighter and often cheaper for partitions.",
      evidence: [
        `Role tag: ${(profileMap.get(wall.id)?.role ?? "partition").replace("_", " ")}.`,
        `Load-bearing rule not triggered for wall ${wall.id}.`,
      ],
      assumptions: [
        "Assumes uniformly distributed openings; localized stress effects are not modeled.",
        "Suggestion is a viable alternative, not an automatic replacement.",
      ],
      relatedWallIds: [wall.id],
      metrics: {
        openingRatio: Number(openingRatio.toFixed(3)),
      },
    });
  });

  const loadBearingAlternatives = scene.walls.filter((wall) => {
    const profile = profileMap.get(wall.id);
    return Boolean(profile?.isLoadBearing);
  });

  loadBearingAlternatives.forEach((wall) => {
    const openingRatio = (openingsByWall.get(wall.id) ?? 0) / Math.max(wall.length, 0.001);
    const supportSpan = wall.roomId ? roomSupportMap.get(wall.roomId)?.supportSpan ?? wall.length : wall.length;

    if (supportSpan > MASONRY_FALLBACK_SPAN_LIMIT_M || openingRatio > OPENING_RATIO_LIMIT) {
      return;
    }

    suggestions.push({
      id: nextId(),
      type: "MATERIAL_OPT",
      severity: "low",
      confidence: "medium",
      location: `Wall ${wall.id}`,
      issue: `Load-bearing wall ${wall.id} may be over-specified for support span ${supportSpan.toFixed(2)}m.`,
      suggestion: "Confined masonry or red brick is a viable alternative where detailing and code checks permit.",
      impact: "Can reduce cost while keeping adequate wall behavior for moderate spans.",
      impactBasis: "Guarded fallback enabled only when support span and opening ratio remain within conservative thresholds.",
      evidence: [
        `support_span=${supportSpan.toFixed(2)}m <= ${MASONRY_FALLBACK_SPAN_LIMIT_M.toFixed(2)}m`,
        `opening_ratio=${openingRatio.toFixed(2)} <= ${OPENING_RATIO_LIMIT.toFixed(2)}`,
        `Role tag: ${(profileMap.get(wall.id)?.role ?? "secondary_support").replace("_", " ")}`,
      ],
      assumptions: [
        "Assumes uniformly distributed openings; localized stress effects are not modeled.",
        "Marked as viable alternative, not mandatory replacement.",
      ],
      relatedWallIds: [wall.id],
      relatedRoomId: wall.roomId,
      metrics: {
        supportSpanM: Number(supportSpan.toFixed(3)),
        openingRatio: Number(openingRatio.toFixed(3)),
      },
    });
  });

  const degreeMap = endpointDegrees(scene);
  scene.walls.forEach((wall) => {
    const profile = profileMap.get(wall.id);
    if (!profile || profile.role !== "partition") {
      return;
    }
    const openingWidth = openingsByWall.get(wall.id) ?? 0;
    if (openingWidth > 0) {
      return;
    }
    if (wall.length < 1.2 || wall.length > 2.4) {
      return;
    }
    const room = wall.roomId ? scene.rooms.find((entry) => entry.id === wall.roomId) : undefined;
    if (!room || room.area > SMALL_ROOM_AREA_M2) {
      return;
    }
    const degree = degreeMap.get(wall.id);
    if (!degree || degree.startDegree > 2 || degree.endDegree > 2) {
      return;
    }

    suggestions.push({
      id: nextId(),
      type: "WALL_REMOVAL",
      severity: "low",
      confidence: "medium",
      location: room.name,
      issue: `Short partition wall ${wall.id} segments a small room (${room.area.toFixed(2)}m2).`,
      suggestion: `Evaluate removing wall ${wall.id} to merge usable space, while preserving circulation and privacy intent.`,
      impact: "Can reduce wall material quantity and finishing labor.",
      impactBasis: "Removing redundant partitions lowers wall material, plaster, and paint scope.",
      evidence: [
        `Role tag: ${profile.role.replace("_", " ")}.`,
        `Length ${wall.length.toFixed(2)}m with no hosted openings.`,
        `Endpoint degree pattern ${degree.startDegree}/${degree.endDegree} indicates non-primary branch wall.`,
      ],
      assumptions: ["Functional zoning and privacy constraints must be validated by designer."],
      relatedWallIds: [wall.id],
      relatedRoomId: room.id,
      metrics: {
        wallLengthM: Number(wall.length.toFixed(3)),
        roomAreaM2: Number(room.area.toFixed(3)),
      },
    });
  });

  suggestions.sort(suggestionComparator);

  return {
    generatedAt: new Date().toISOString(),
    assumptions: [
      "Heuristic structural reasoning from 2D plan + wall/opening metadata; not full code-compliance analysis.",
      "Suggestions are advisory only and do not mutate geometry.",
      "Single-floor default is used when level metadata is not provided.",
    ],
    defaults: {
      thickSupportM: THICK_SUPPORT_M,
      supportSpanLimitM: SUPPORT_SPAN_LIMIT_M,
      masonryFallbackSpanLimitM: MASONRY_FALLBACK_SPAN_LIMIT_M,
      openingRatioLimit: OPENING_RATIO_LIMIT,
    },
    wallProfiles: [...profiles].sort((a, b) => {
      const roleDelta = roleWeight(b.role) - roleWeight(a.role);
      if (roleDelta !== 0) {
        return roleDelta;
      }
      return a.wallId.localeCompare(b.wallId);
    }),
    suggestions,
  };
}
