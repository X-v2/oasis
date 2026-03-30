import { classifyWall, getRoomContextForWall } from "@/lib/materialEngine";
import type { NormalizedWall, Recommendation, SceneData, ValidationIssue } from "@/lib/types";

export function generateExplanation(
  wall: NormalizedWall,
  recommendations: Recommendation[],
  scene: SceneData,
  issues: ValidationIssue[],
) {
  const topRecommendation = recommendations[0];
  const context = getRoomContextForWall(wall, scene);
  const classification = classifyWall(wall);
  const wallIssues = issues.filter((issue) => issue.wallId === wall.id);

  const parts = [
    `Wall ${wall.id} is treated as ${classification.replace("_", "-")} based on ${
      wall.inferredType ? "fallback inference" : "the provided type"
    } and a thickness of ${wall.thickness.toFixed(2)}m, with ${wall.confidence} confidence.`,
    `Its modeled length is ${wall.length.toFixed(2)}m and height is ${wall.height.toFixed(2)}m.`,
  ];

  if (context.room) {
    parts.push(
      `The nearest room span is ${context.span.toFixed(2)}m in ${context.room.name},${
        context.inferredSpan ? " inferred from repaired geometry." : " taken from valid room geometry."
      } Room context confidence is ${context.confidence}.`,
    );
  } else {
    parts.push("No reliable room polygon was available, so the span context is treated as uncertain.");
  }

  if (topRecommendation) {
    parts.push(`Recommended material: ${topRecommendation.rationale}`);
  }

  if (wallIssues.length > 0) {
    parts.push(
      `This conclusion includes ${wallIssues.length} validation ${
        wallIssues.length === 1 ? "adjustment" : "adjustments"
      }, so the demo flags it as partially inferred.`,
    );
  }

  return parts.join(" ");
}
