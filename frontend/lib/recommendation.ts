import "server-only";

import { MATERIALS } from "@/lib/materialData";
import { classifyWall } from "@/lib/materialEngine";
import { buildColumns, buildStructuralNodes } from "@/lib/sceneGraph";
import type {
  ElementRecommendation,
  MaterialOption,
  MaterialRecommendationTable,
  SceneData,
  SceneSelection,
  SceneSelectionType,
  StructuralElementType,
} from "@/lib/types";

type ScoreWeights = {
  cost: number;
  constructability: number;
  availability: number;
  durability: number;
  safetyMargin: number;
};

type StructuralSystemClass = {
  id: "option_a" | "option_b" | "option_c";
  label: string;
  reason: string;
  maxSpan: number;
  outerWalls: number;
  internalWalls: number;
};

const SCORE_WEIGHTS: ScoreWeights = {
  cost: 0.35,
  constructability: 0.25,
  availability: 0.15,
  durability: 0.15,
  safetyMargin: 0.1,
};

type FocusContext = {
  selectedType: SceneSelectionType | "none";
  selectedId: string;
  focusedElementType: StructuralElementType | "none";
  sizeSummary: string;
};

function normalize(value: number, min: number, max: number) {
  if (max <= min) {
    return 0.5;
  }
  return (value - min) / (max - min);
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function focusFromSelection(scene: SceneData, selection: SceneSelection | null): FocusContext {
  if (!selection) {
    return {
      selectedType: "none" as const,
      selectedId: "-",
      focusedElementType: "none" as const,
      sizeSummary: "No active selection. Click a wall, slab, column, node, door, or window.",
    };
  }

  if (selection.type === "wall") {
    const wall = scene.walls.find((entry) => entry.id === selection.id);
    if (!wall) {
      return {
        selectedType: selection.type,
        selectedId: selection.id,
        focusedElementType: "none" as const,
        sizeSummary: "Selected wall was not found in the normalized scene.",
      };
    }
    const wallType = classifyWall(wall);
    return {
      selectedType: selection.type,
      selectedId: selection.id,
      focusedElementType: wallType === "partition" ? ("partition_wall" as const) : ("load_bearing_wall" as const),
      sizeSummary: `Wall ${wall.id}: length ${wall.length.toFixed(2)}m, height ${wall.height.toFixed(2)}m, thickness ${wall.thickness.toFixed(2)}m.`,
    };
  }

  if (selection.type === "slab") {
    const slab = scene.rooms.find((room) => room.id === selection.id);
    if (!slab) {
      return {
        selectedType: selection.type,
        selectedId: selection.id,
        focusedElementType: "none" as const,
        sizeSummary: "Selected slab was not found in the normalized scene.",
      };
    }
    return {
      selectedType: selection.type,
      selectedId: selection.id,
      focusedElementType: "slab" as const,
      sizeSummary: `Slab ${slab.id}: span ${slab.span.toFixed(2)}m, area ${slab.area.toFixed(2)}m2.`,
    };
  }

  if (selection.type === "column" || selection.type === "node") {
    const nodes = buildStructuralNodes(scene);
    const columns = buildColumns(scene, nodes);
    const column =
      selection.type === "column"
        ? columns.find((entry) => entry.id === selection.id)
        : columns.find((entry) => entry.nodeId === selection.id);
    if (column) {
      return {
        selectedType: selection.type,
        selectedId: selection.id,
        focusedElementType: "column" as const,
        sizeSummary: `Column ${column.id}: ${column.width.toFixed(2)}m x ${column.depth.toFixed(2)}m x ${column.height.toFixed(2)}m.`,
      };
    }

    if (selection.type === "node") {
      const node = nodes.find((entry) => entry.id === selection.id);
      const firstWallId = node?.connectedWallIds[0];
      const wall = firstWallId ? scene.walls.find((entry) => entry.id === firstWallId) : undefined;
      if (node && wall) {
        const wallType = classifyWall(wall);
        return {
          selectedType: selection.type,
          selectedId: selection.id,
          focusedElementType: wallType === "partition" ? ("partition_wall" as const) : ("load_bearing_wall" as const),
          sizeSummary: `Node ${node.id}: degree ${node.degree}, nearest wall ${wall.id} length ${wall.length.toFixed(2)}m.`,
        };
      }
    }

    return {
      selectedType: selection.type,
      selectedId: selection.id,
      focusedElementType: "none" as const,
      sizeSummary: "Selected node/column does not map to a structural element.",
    };
  }

  if (selection.type === "door" || selection.type === "window") {
    const opening = scene.openings.find((entry) => entry.id === selection.id);
    const hostWall = opening ? scene.walls.find((wall) => wall.id === opening.wallId) : undefined;
    if (!opening || !hostWall) {
      return {
        selectedType: selection.type,
        selectedId: selection.id,
        focusedElementType: "none" as const,
        sizeSummary: "Selected opening or host wall was not found.",
      };
    }

    const hostType = classifyWall(hostWall);
    return {
      selectedType: selection.type,
      selectedId: selection.id,
      focusedElementType: hostType === "partition" ? ("partition_wall" as const) : ("load_bearing_wall" as const),
      sizeSummary: `${opening.kind} ${opening.id}: width ${opening.width.toFixed(2)}m, height ${opening.height.toFixed(2)}m on wall ${hostWall.id}.`,
    };
  }

  return {
    selectedType: selection.type,
    selectedId: selection.id,
    focusedElementType: "none" as const,
    sizeSummary: `Selection ${selection.type} ${selection.id} is not mapped to a structural formula.`,
  };
}

function classifyStructuralSystem(scene: SceneData): StructuralSystemClass {
  const maxSpan = scene.rooms.length > 0 ? Math.max(...scene.rooms.map((room) => room.span)) : 0;
  const outerWalls = scene.walls.filter((wall) => wall.type === "outer" || classifyWall(wall) === "load_bearing").length;
  const internalWalls = scene.walls.filter((wall) => classifyWall(wall) === "partition").length;

  if (maxSpan > 5) {
    return {
      id: "option_c",
      label: "Option C: large-span localized steel/prestressed support",
      reason: `Max span ${maxSpan.toFixed(2)}m exceeds 5m, so long-span structural members are needed in critical locations only.`,
      maxSpan,
      outerWalls,
      internalWalls,
    };
  }

  if (outerWalls >= Math.max(3, Math.ceil(scene.walls.length * 0.75)) && internalWalls <= 1) {
    return {
      id: "option_b",
      label: "Option B: heavy wall system (RCC load-bearing fallback)",
      reason: "Most walls behave as primary supports with few partitions, so heavy wall behavior is dominant.",
      maxSpan,
      outerWalls,
      internalWalls,
    };
  }

  return {
    id: "option_a",
    label: "Option A: low-rise RCC frame + masonry infill",
    reason: "Typical low-rise behavior detected: mixed outer/internal walls with moderate spans.",
    maxSpan,
    outerWalls,
    internalWalls,
  };
}

function requiredCapacity(
  elementType: StructuralElementType,
  system: StructuralSystemClass,
  focusType: SceneSelectionType | "none",
) {
  const spanPenalty = Math.max(0, system.maxSpan - 3) * 10;

  if (elementType === "column") {
    return 88 + spanPenalty;
  }

  if (elementType === "slab") {
    return 76 + spanPenalty;
  }

  if (elementType === "load_bearing_wall") {
    const openingPenalty = focusType === "door" || focusType === "window" ? 6 : 0;
    const systemPenalty = system.id === "option_b" ? 8 : 0;
    return 58 + spanPenalty * 0.7 + openingPenalty + systemPenalty;
  }

  return 24;
}

function materialAvailability(material: (typeof MATERIALS)[number]) {
  const name = material.name.toLowerCase();
  let availability = material.use === "partition" || material.use === "general" ? 0.84 : 0.74;

  if (name.includes("rcc") || name.includes("brick") || name.includes("aac") || name.includes("fly ash")) {
    availability += 0.08;
  }
  if (name.includes("prestressed") || name.includes("composite steel") || name.includes("tube column")) {
    availability -= 0.1;
  }

  return clamp(availability, 0.35, 0.98);
}

function materialCapacity(material: (typeof MATERIALS)[number]) {
  const name = material.name.toLowerCase();
  let capacity = material.strength * 24 + material.durability * 4;

  if (material.use === "load_bearing") {
    capacity += 18;
  }
  if (material.use === "long_span") {
    capacity += 24;
  }
  if (material.use === "partition") {
    capacity -= 8;
  }

  if (name.includes("rcc") || name.includes("concrete") || name.includes("rc ")) {
    capacity += 8;
  }
  if (name.includes("brick") || name.includes("block") || name.includes("masonry")) {
    capacity -= 2;
  }
  if (name.includes("gypsum")) {
    capacity -= 18;
  }

  return clamp(capacity, 10, 180);
}

function materialAllowedForElement(
  material: (typeof MATERIALS)[number],
  elementType: StructuralElementType,
  system: StructuralSystemClass,
) {
  const name = material.name.toLowerCase();
  const isConcreteLike =
    name.includes("rcc") || name.includes("concrete") || name.includes("rc ") || name.includes("precast");
  const isMasonryLike =
    name.includes("brick") || name.includes("block") || name.includes("masonry") || name.includes("aac");
  const isLongSpanSpecial = material.use === "long_span" || name.includes("prestressed") || name.includes("steel");

  if (elementType === "partition_wall") {
    if (isLongSpanSpecial || name.includes("column") || name.includes("shear wall")) {
      return false;
    }
    return material.use === "partition" || material.use === "general" || isMasonryLike;
  }

  if (elementType === "load_bearing_wall") {
    if (name.includes("prestressed") || name.includes("column") || name.includes("deck slab")) {
      return false;
    }
    return material.use === "load_bearing" || material.use === "general" || isMasonryLike || isConcreteLike;
  }

  if (elementType === "slab") {
    if (name.includes("column") || isMasonryLike || material.use === "partition") {
      return false;
    }
    if (system.id === "option_c") {
      return isConcreteLike || material.use === "long_span" || name.includes("steel");
    }
    return isConcreteLike;
  }

  if (elementType === "column") {
    if (name.includes("panel") || name.includes("block") || name.includes("brick") || name.includes("slab")) {
      return false;
    }
    if (system.id === "option_c") {
      return isConcreteLike || material.use === "long_span" || name.includes("steel");
    }
    return isConcreteLike;
  }

  return true;
}

function spanBasedReasoning(scene: SceneData) {
  const sorted = [...scene.rooms].sort((a, b) => b.span - a.span);
  const messages: string[] = [];

  sorted.slice(0, 3).forEach((room) => {
    if (room.span > 5) {
      messages.push(`${room.name} span = ${room.span.toFixed(2)}m exceeds masonry comfort zone, so RCC beam/frame support is required.`);
    } else {
      messages.push(`${room.name} span = ${room.span.toFixed(2)}m is within typical low-rise range; masonry infill + slab is generally viable.`);
    }
  });

  return messages;
}

function rowConcerns(
  elementType: StructuralElementType,
  scene: SceneData,
  system: StructuralSystemClass,
  required: number,
  validCount: number,
  rejectedCount: number,
  focusType: SceneSelectionType | "none",
) {
  const concerns: string[] = [
    `Structural system classification: ${system.label}. ${system.reason}`,
    `Adequacy gate enabled: materials with capacity below ${required.toFixed(1)} are rejected before scoring.`,
  ];

  spanBasedReasoning(scene).forEach((message) => concerns.push(message));

  if (elementType === "column") {
    concerns.push("Decision map: columns -> RCC-first, use steel/composite only where spans demand long-span behavior.");
  }
  if (elementType === "slab") {
    concerns.push("Decision map: slabs -> RCC-first, with prestressed/steel deck only for large spans.");
  }
  if (elementType === "load_bearing_wall") {
    concerns.push("Decision map: load-bearing walls -> brick/masonry or RCC case-by-case; avoid long-span special systems as generic wall picks.");
  }
  if (elementType === "partition_wall") {
    concerns.push("Decision map: partitions -> AAC/hollow block/light systems prioritized over heavy structural materials.");
  }

  if ((focusType === "door" || focusType === "window") && elementType.includes("wall")) {
    concerns.push("Opening-hosted wall context: maintain edge safety and crack control around openings after adequacy is met.");
  }

  concerns.push(`Valid options after adequacy gate: ${validCount}. Rejected for inadequate capacity: ${rejectedCount}.`);

  if (validCount === 0) {
    concerns.push("No material passed adequacy gate with current constraints; review geometry assumptions or widen system options.");
  }

  return concerns;
}

function scoreForElement(
  elementType: StructuralElementType,
  scene: SceneData,
  count: number,
  focusType: SceneSelectionType | "none",
): ElementRecommendation {
  const system = classifyStructuralSystem(scene);
  const required = requiredCapacity(elementType, system, focusType);
  const candidates = MATERIALS.filter((material) => materialAllowedForElement(material, elementType, system));
  const rankedSet = candidates.length > 0 ? candidates : MATERIALS;

  const costs = rankedSet.map((m) => m.cost);
  const durabilities = rankedSet.map((m) => m.durability);
  const eases = rankedSet.map((m) => m.ease);
  const minCost = Math.min(...costs);
  const maxCost = Math.max(...costs);
  const minDurability = Math.min(...durabilities);
  const maxDurability = Math.max(...durabilities);
  const minEase = Math.min(...eases);
  const maxEase = Math.max(...eases);

  const computed = rankedSet.map((material) => {
    const capacity = materialCapacity(material);
    const adequate = capacity >= required;
    const costEfficiency = 1 - normalize(material.cost, minCost, maxCost);
    const constructability = normalize(material.ease, minEase, maxEase);
    const availability = materialAvailability(material);
    const durability = normalize(material.durability, minDurability, maxDurability);
    const safetyMargin = adequate ? clamp((capacity - required) / Math.max(required, 1), 0, 1) : 0;

    const score =
      100 *
      (SCORE_WEIGHTS.cost * costEfficiency +
        SCORE_WEIGHTS.constructability * constructability +
        SCORE_WEIGHTS.availability * availability +
        SCORE_WEIGHTS.durability * durability +
        SCORE_WEIGHTS.safetyMargin * safetyMargin);

    return {
      material,
      adequate,
      capacity,
      safetyMargin,
      score: Number(score.toFixed(2)),
      rationale: adequate
        ? `${material.name} passes adequacy gate (capacity ${capacity.toFixed(1)} >= required ${required.toFixed(1)}) and is ranked on cost/buildability/availability/durability.`
        : `${material.name} rejected by adequacy gate (capacity ${capacity.toFixed(1)} < required ${required.toFixed(1)}).`,
    };
  });

  const valid = computed.filter((entry) => entry.adequate).sort((a, b) => b.score - a.score);
  const rejectedCount = computed.length - valid.length;
  const fallback = [...computed].sort((a, b) => b.capacity - a.capacity);
  const shortlisted = (valid.length > 0 ? valid : fallback).slice(0, count);

  const options: MaterialOption[] = shortlisted.map((entry, index) => ({
    material: entry.material.name,
    rank: index + 1,
    tradeoffScore: entry.score,
    cost: entry.material.cost,
    strength: entry.material.strength,
    durability: entry.material.durability,
    ease: entry.material.ease,
    rationale: entry.rationale,
  }));

  const formula =
    "score = 100 * (0.35*cost_eff + 0.25*constructability + 0.15*availability + 0.15*durability + 0.10*safety_margin), applied only after adequacy gate";

  return {
    elementType,
    weightJustification:
      "Strength is handled as a hard adequacy constraint (gate), not a weighted preference. Among safe options, the system minimizes total build burden with explicit cost-first weighting.",
    formula,
    structuralConcerns: rowConcerns(elementType, scene, system, required, valid.length, rejectedCount, focusType),
    options,
  };
}

type GeminiReason = {
  elementType: StructuralElementType;
  optionReasons: Array<{ material: string; rationale: string }>;
  concerns?: string[];
};

function extractJson(text: string) {
  const fenced = text.match(/```json\s*([\s\S]*?)```/i);
  if (fenced?.[1]) {
    return fenced[1];
  }
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start !== -1 && end > start) {
    return text.slice(start, end + 1);
  }
  return "";
}

async function buildGeminiRationales(
  rows: ElementRecommendation[],
  model: string,
): Promise<GeminiReason[] | null> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return null;
  }

  const payload = rows.map((row) => ({
    elementType: row.elementType,
    formula: row.formula,
    options: row.options.map((option) => ({
      material: option.material,
      tradeoffScore: option.tradeoffScore,
      cost: option.cost,
      strength: option.strength,
      durability: option.durability,
    })),
    concerns: row.structuralConcerns,
  }));

  const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [
        {
          role: "user",
          parts: [
            {
              text:
                'Return ONLY JSON with schema {"rows":[{"elementType":"load_bearing_wall|partition_wall|slab|column","optionReasons":[{"material":"string","rationale":"string"}],"concerns":["string"]}]}. Keep rationale concise, plain language, and reference adequacy-gate-first logic. Input: ' +
                JSON.stringify(payload),
            },
          ],
        },
      ],
    }),
  });

  if (!response.ok) {
    return null;
  }

  const data = (await response.json()) as {
    candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>;
  };
  const text = data.candidates?.[0]?.content?.parts?.map((part) => part.text ?? "").join("\n") ?? "";
  if (!text) {
    return null;
  }

  try {
    const parsed = JSON.parse(extractJson(text)) as { rows?: GeminiReason[] };
    return Array.isArray(parsed.rows) ? parsed.rows : null;
  } catch {
    return null;
  }
}

export async function generateMaterialRecommendationTable(
  scene: SceneData,
  selection: SceneSelection | null,
  selectionMetrics?: string,
): Promise<MaterialRecommendationTable> {
  const countRaw = Number(process.env.RECOMMENDATION_COUNT ?? 3);
  const recommendationCount = Math.max(2, Math.min(3, Number.isFinite(countRaw) ? countRaw : 3));
  const model = process.env.GEMINI_MODEL ?? "gemini-2.5-flash";
  const focus = focusFromSelection(scene, selection);

  if (focus.focusedElementType === "none") {
    return {
      source: "deterministic",
      model,
      generatedAt: new Date().toISOString(),
      recommendationCount,
      focus: {
        ...focus,
        sizeSummary: selectionMetrics || focus.sizeSummary,
      },
      rows: [],
    };
  }

  const rows: ElementRecommendation[] = [
    scoreForElement(focus.focusedElementType, scene, recommendationCount, focus.selectedType),
  ];

  const geminiRows = await buildGeminiRationales(rows, model);
  if (!geminiRows) {
    return {
      source: "deterministic",
      model,
      generatedAt: new Date().toISOString(),
      recommendationCount,
      focus: {
        ...focus,
        sizeSummary: selectionMetrics || focus.sizeSummary,
      },
      rows,
    };
  }

  const merged = rows.map((row) => {
    const match = geminiRows.find((entry) => entry.elementType === row.elementType);
    if (!match) {
      return row;
    }
    return {
      ...row,
      structuralConcerns: match.concerns?.length ? match.concerns : row.structuralConcerns,
      options: row.options.map((option) => {
        const reason = match.optionReasons.find((entry) => entry.material === option.material);
        return {
          ...option,
          rationale: reason?.rationale ?? option.rationale,
        };
      }),
    };
  });

  return {
    source: "gemini",
    model,
    generatedAt: new Date().toISOString(),
    recommendationCount,
    focus: {
      ...focus,
      sizeSummary: selectionMetrics || focus.sizeSummary,
    },
    rows: merged,
  };
}
