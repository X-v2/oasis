import { distanceToSegment, getRoomContext } from "@/lib/geometry";
import { MATERIALS } from "@/lib/materialData";
import type {
  ConfidenceLevel,
  NormalizedRoom,
  NormalizedWall,
  Recommendation,
  RoomContext,
  SceneData,
  WallClassification,
} from "@/lib/types";

export function classifyWall(wall: NormalizedWall): WallClassification {
  if (wall.type === "outer") {
    return "load_bearing";
  }

  if (wall.type === "partition") {
    return "partition";
  }

  if (wall.thickness < 0.14) {
    return "partition";
  }

  return "semi_structural";
}

export function findRoomForWall(wall: NormalizedWall, rooms: NormalizedRoom[]) {
  if (wall.roomId) {
    return rooms.find((room) => room.id === wall.roomId);
  }

  const midpoint2D: [number, number] = [wall.midpoint[0], wall.midpoint[2]];
  let best: NormalizedRoom | undefined;
  let bestDistance = Number.POSITIVE_INFINITY;

  for (const room of rooms) {
    for (let i = 0; i < room.polygon2D.length; i += 1) {
      const start = room.polygon2D[i];
      const end = room.polygon2D[(i + 1) % room.polygon2D.length];
      const candidate = distanceToSegment(midpoint2D, start, end);
      if (candidate < bestDistance) {
        best = room;
        bestDistance = candidate;
      }
    }
  }

  return best;
}

export function getRoomContextForWall(wall: NormalizedWall, scene: SceneData): RoomContext {
  return getRoomContext(findRoomForWall(wall, scene.rooms));
}

function confidenceRank(level: ConfidenceLevel) {
  if (level === "high") {
    return 3;
  }

  if (level === "medium") {
    return 2;
  }

  return 1;
}

function lowerConfidence(a: ConfidenceLevel, b: ConfidenceLevel): ConfidenceLevel {
  return confidenceRank(a) <= confidenceRank(b) ? a : b;
}

export function scoreMaterial(
  material: (typeof MATERIALS)[number],
  classification: WallClassification,
  context: RoomContext,
) {
  const longSpanBoost =
    context.span > 5 && (material.use === "long_span" || material.use === "load_bearing") ? 1.2 : 0;

  if (classification === "load_bearing") {
    return (
      0.45 * material.strength +
      0.3 * material.durability +
      0.1 * material.ease -
      0.2 * material.cost +
      longSpanBoost
    );
  }

  if (classification === "partition") {
    return (
      0.2 * material.strength +
      0.25 * material.durability +
      0.35 * material.ease -
      0.25 * material.cost +
      (material.use === "partition" ? 0.8 : 0)
    );
  }

  return (
    0.35 * material.strength +
    0.3 * material.durability +
    0.15 * material.ease -
    0.2 * material.cost +
    (material.use === "semi_structural" ? 0.5 : 0) +
    longSpanBoost * 0.7
  );
}

function buildRationale(
  materialName: string,
  classification: WallClassification,
  context: RoomContext,
  confidence: ConfidenceLevel,
  comparisonTarget?: string,
) {
  const reasons = [];

  if (classification === "load_bearing") {
    reasons.push("high strength for outer structural duty");
  } else if (classification === "partition") {
    reasons.push("better cost-to-build speed for internal separation");
  } else {
    reasons.push("balanced strength and durability for uncertain load case");
  }

  if (context.span > 5) {
    reasons.push(`handles the ${context.span.toFixed(2)}m span more safely`);
  }

  const comparison =
    comparisonTarget && classification === "load_bearing" && context.span > 4.5
      ? `${materialName} is preferred over ${comparisonTarget} because the span pushes the wall into a more structural use case.`
      : comparisonTarget && classification === "partition"
        ? `${materialName} edges past ${comparisonTarget} because this wall behaves more like a lightweight divider than a primary support.`
        : comparisonTarget
          ? `${materialName} is kept ahead of ${comparisonTarget} because it balances this uncertain structural role more safely.`
          : `${materialName} ranks well in the current context.`;

  return `${comparison} It offers ${reasons.join(" and ")} with ${confidence} confidence.`;
}

export function recommendMaterials(wall: NormalizedWall, scene: SceneData): Recommendation[] {
  const classification = classifyWall(wall);
  const context = getRoomContextForWall(wall, scene);
  const recommendationConfidence = lowerConfidence(wall.confidence, context.confidence);
  const scored = MATERIALS.map((material) => ({
    name: material.name,
    score: scoreMaterial(material, classification, context),
  })).sort((a, b) => b.score - a.score);

  return scored.slice(0, 3).map((recommendation, index) => ({
    ...recommendation,
    rationale: buildRationale(
      recommendation.name,
      classification,
      context,
      recommendationConfidence,
      scored[index + 1]?.name,
    ),
    confidence: recommendationConfidence,
  }));
}
