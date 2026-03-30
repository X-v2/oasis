import type { DatasetReadiness, SceneData, ValidationIssue } from "@/lib/types";

export function getReadinessLabel(readiness: DatasetReadiness) {
  if (readiness === "valid") {
    return "Valid";
  }

  if (readiness === "partial") {
    return "Partial";
  }

  return "Invalid";
}

export function buildProjectQuestions(scene: SceneData, issues: ValidationIssue[]) {
  const lowConfidenceWalls = scene.walls.filter((wall) => wall.confidence === "low").length;
  const inferredWalls = scene.walls.filter((wall) => wall.inferredType).length;
  const inferredRooms = scene.rooms.filter((room) => room.inferredSpan).length;
  const hasOpenings = scene.openings.length > 0;
  const genericRooms = scene.rooms.filter((room) => /^room-\d+$/i.test(room.id)).length;

  const questions = [];

  questions.push({
    id: "floors",
    title: "Floors",
    prompt: "How many floors should the model represent, and which floor is this plan describing?",
  });

  if (genericRooms > 0 || scene.rooms.length === 0) {
    questions.push({
      id: "rooms",
      title: "Rooms",
      prompt: "What are the actual room names and uses so we can explain why each span matters structurally?",
    });
  } else {
    questions.push({
      id: "room-priority",
      title: "Room Priority",
      prompt: "Which rooms are public, private, or service spaces so the recommendations can mention functional tradeoffs?",
    });
  }

  if (inferredWalls > 0 || lowConfidenceWalls > 0) {
    questions.push({
      id: "structure",
      title: "Structure",
      prompt: "Which walls are definitely load-bearing, and is there a column or beam grid we should respect?",
    });
  }

  if (inferredRooms > 0) {
    questions.push({
      id: "span-direction",
      title: "Span Direction",
      prompt: "Should any room span in a preferred direction, or should we keep using the geometric approximation?",
    });
  }

  if (hasOpenings) {
    questions.push({
      id: "openings",
      title: "Openings",
      prompt: "Are doors and windows only visual markers, or should they affect structural reasoning in this demo?",
    });
  }

  if (issues.some((issue) => issue.severity !== "info")) {
    questions.push({
      id: "source-quality",
      title: "Input Quality",
      prompt: "Do you want the demo to keep repaired geometry, or should uncertain elements be explicitly marked before judging?",
    });
  }

  return questions.slice(0, 5);
}
