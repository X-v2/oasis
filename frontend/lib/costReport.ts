import "server-only";

import { MATERIALS } from "@/lib/materialData";
import { classifyWall } from "@/lib/materialEngine";
import { buildColumns, buildStructuralNodes } from "@/lib/sceneGraph";
import type {
  CostLineItem,
  CostReport,
  CostReportElementType,
  CostScenarioOption,
  CostUnit,
  MaterialPricingEntry,
  NormalizedRoom,
  SceneData,
} from "@/lib/types";

type CostReportOptions = {
  projectName?: string;
  currency?: string;
  pricing?: Record<string, MaterialPricingEntry>;
  wastageOverridePercent?: number;
  includeBeamCandidates?: boolean;
  materialByElement?: Partial<Record<CostReportElementType, string>>;
  elementMaterialOverrides?: Record<string, string>;
  scenarioId?: string;
  scenarioLabel?: string;
  scenarioDescription?: string;
};

type CostScenarioPreset = {
  id: string;
  label: string;
  description: string;
  materialByElement: Partial<Record<CostReportElementType, string>>;
  pricing: Record<string, MaterialPricingEntry>;
  wastageDeltaPercent: number;
};

const DEFAULT_PRICING: Record<string, MaterialPricingEntry> = {
  "Red Brick": { unit: "m2", rate: 1750 },
  "AAC Blocks": { unit: "m2", rate: 1350 },
  "Fly Ash Brick": { unit: "m2", rate: 1580 },
  "Solid Concrete Block": { unit: "m2", rate: 1980 },
  "RCC (Slab)": { unit: "m2", rate: 1850 },
  "RCC (Column)": { unit: "nos", rate: 18500 },
  "RCC (Beam)": { unit: "m3", rate: 12800 },
  "Steel Frame": { unit: "m3", rate: 16500 },
};

export const COST_SCENARIOS: CostScenarioPreset[] = [
  {
    id: "economy",
    label: "Economy",
    description: "Lower-cost mix optimized for budget-sensitive delivery.",
    materialByElement: {
      load_bearing_wall: "Fly Ash Brick",
      partition_wall: "AAC Blocks",
      slab: "RCC (Slab)",
      column: "RCC (Column)",
      beam: "RCC (Beam)",
    },
    pricing: {
      "AAC Blocks": { unit: "m2", rate: 1225 },
      "Fly Ash Brick": { unit: "m2", rate: 1460 },
      "RCC (Slab)": { unit: "m2", rate: 1700 },
      "RCC (Column)": { unit: "nos", rate: 17100 },
      "RCC (Beam)": { unit: "m3", rate: 11900 },
    },
    wastageDeltaPercent: 0.5,
  },
  {
    id: "standard",
    label: "Standard",
    description: "Balanced baseline for typical execution quality and rates.",
    materialByElement: {
      load_bearing_wall: "Red Brick",
      partition_wall: "AAC Blocks",
      slab: "RCC (Slab)",
      column: "RCC (Column)",
      beam: "RCC (Beam)",
    },
    pricing: {
      "Red Brick": { unit: "m2", rate: 1750 },
      "AAC Blocks": { unit: "m2", rate: 1350 },
      "RCC (Slab)": { unit: "m2", rate: 1850 },
      "RCC (Column)": { unit: "nos", rate: 18500 },
      "RCC (Beam)": { unit: "m3", rate: 12800 },
    },
    wastageDeltaPercent: 0,
  },
  {
    id: "premium",
    label: "Premium",
    description: "Higher-grade materials and market rates for premium delivery.",
    materialByElement: {
      load_bearing_wall: "Solid Concrete Block",
      partition_wall: "AAC Blocks",
      slab: "RCC (Slab)",
      column: "RCC (Column)",
      beam: "Steel Frame",
    },
    pricing: {
      "Solid Concrete Block": { unit: "m2", rate: 2120 },
      "AAC Blocks": { unit: "m2", rate: 1490 },
      "RCC (Slab)": { unit: "m2", rate: 2050 },
      "RCC (Column)": { unit: "nos", rate: 20100 },
      "Steel Frame": { unit: "m3", rate: 17800 },
    },
    wastageDeltaPercent: 1,
  },
];

function buildMaterialCatalog(pricing: Record<string, MaterialPricingEntry>) {
  const byName = new Map(MATERIALS.map((entry) => [entry.name, entry] as const));
  const catalog: CostReport["materialCatalog"] = {};

  Object.entries(pricing).forEach(([material, rateEntry]) => {
    const canonical = material.includes("(") ? material.split("(")[0].trim() : material;
    const profile = byName.get(material) ?? byName.get(canonical);
    catalog[material] = {
      unit: rateEntry.unit,
      rate: rateEntry.rate,
      strength: profile?.strength ?? 3,
      durability: profile?.durability ?? 3,
    };
  });

  return catalog;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function round3(value: number) {
  return Math.round(value * 1000) / 1000;
}

function confidenceToScore(value: "high" | "medium" | "low") {
  if (value === "high") return 0.9;
  if (value === "medium") return 0.75;
  return 0.55;
}

function slabThicknessFromSpan(span: number) {
  return Math.max(0.12, Math.min(0.24, 0.12 + Math.max(span - 3, 0) * 0.018));
}

function materialForElement(
  elementType: CostReportElementType,
  materialByElement?: Partial<Record<CostReportElementType, string>>,
) {
  const override = materialByElement?.[elementType];
  if (override) {
    return override;
  }
  if (elementType === "load_bearing_wall") return "Red Brick";
  if (elementType === "partition_wall") return "AAC Blocks";
  if (elementType === "beam") return "RCC (Beam)";
  if (elementType === "slab") return "RCC (Slab)";
  return "RCC (Column)";
}

function defaultWastageForElement(elementType: CostReportElementType) {
  if (elementType === "load_bearing_wall" || elementType === "partition_wall") return 0.06;
  if (elementType === "column") return 0.03;
  if (elementType === "beam") return 0.03;
  return 0.04;
}

function elementJustification(elementType: CostReportElementType, material: string, spanM?: number) {
  if (elementType === "load_bearing_wall") {
    return `${material} selected for load-bearing duty to prioritize compressive strength and predictable behavior.`;
  }
  if (elementType === "partition_wall") {
    return `${material} selected for partition use to reduce dead load while keeping build speed and cost efficient.`;
  }
  if (elementType === "slab") {
    return `${material} selected for slab continuity and familiar site execution for typical low-rise spans.`;
  }
  if (elementType === "column") {
    return `${material} selected for vertical support where stiffness and durability are primary requirements.`;
  }
  return spanM && spanM > 5.5
    ? `${material} selected for beam candidate due to longer span demand and structural continuity.`
    : `${material} selected for beam candidate as a practical baseline for frame action.`;
}

function withCost(
  partial: Omit<
    CostLineItem,
    "unit" | "unitRate" | "baseCost" | "wastageAllowance" | "subtotal" | "quantity" | "wastageFactor"
  > & {
    quantityByUnit: Partial<Record<CostUnit, number>>;
    wastageFactorRaw: number;
  },
  pricing: Record<string, MaterialPricingEntry>,
  warnings: string[],
) {
  let pricingEntry = pricing[partial.material];
  if (!pricingEntry) {
    const canonical = partial.material.includes("(") ? partial.material.split("(")[0].trim() : partial.material;
    const profile = MATERIALS.find((entry) => entry.name === partial.material || entry.name === canonical);
    if (profile) {
      const inferredRate = round3(900 + profile.cost * 2200);
      const inferredUnit: CostUnit = profile.use === "partition" ? "m2" : "m3";
      pricingEntry = { unit: inferredUnit, rate: inferredRate };
      pricing[partial.material] = pricingEntry;
      warnings.push(`Rate for "${partial.material}" inferred from material profile. Verify local market rate.`);
    } else {
      warnings.push(`No pricing found for material "${partial.material}". Rate set to 0.`);
    }
  }

  const unit: CostUnit = pricingEntry?.unit ?? "m3";
  const unitRate = pricingEntry?.rate ?? 0;
  const quantity = round3(partial.quantityByUnit[unit] ?? 0);
  const wastageFactor = round3(partial.wastageFactorRaw);
  const baseCost = round3(quantity * unitRate);
  const wastageAllowance = round3(baseCost * wastageFactor);
  const subtotal = round3(baseCost + wastageAllowance);

  const item: CostLineItem = {
    elementId: partial.elementId,
    elementType: partial.elementType,
    material: partial.material,
    quantity,
    unit,
    unitRate,
    baseCost,
    wastageFactor,
    wastageAllowance,
    subtotal,
    confidence: partial.confidence,
    justification: partial.justification,
    assumptions: partial.assumptions,
  };

  return item;
}

function beamCandidatesFromRooms(rooms: NormalizedRoom[]) {
  return rooms
    .filter((room) => room.span >= 4.5)
    .map((room) => {
      const width = room.span > 5.5 ? 0.23 : 0.2;
      const depth = room.span > 5.5 ? 0.45 : 0.4;
      return {
        id: `beam_${room.id}`,
        room,
        length: room.span,
        width,
        depth,
        volume: room.span * width * depth,
      };
    });
}

export function generateCostReport(scene: SceneData, options?: CostReportOptions): CostReport {
  const pricing = options?.pricing ? { ...DEFAULT_PRICING, ...options.pricing } : { ...DEFAULT_PRICING };
  const overrideWastage = Number.isFinite(options?.wastageOverridePercent)
    ? clamp((options?.wastageOverridePercent ?? 0) / 100, 0, 0.25)
    : 0;
  const includeBeamCandidates = options?.includeBeamCandidates ?? true;

  const warnings: string[] = [];
  const assumptions: string[] = [
    `Standard floor height assumed from scene meta (${scene.meta.wallHeight.toFixed(2)}m).`,
    "Openings are deducted from wall volume using opening area x wall thickness.",
    "Starter rate card is used unless overridden in API payload.",
    "Wastage is applied by element type, with optional global override/scenario delta.",
  ];

  const items: CostLineItem[] = [];
  const openingsByWall = new Map<string, number>();
  scene.openings.forEach((opening) => {
    const current = openingsByWall.get(opening.wallId) ?? 0;
    openingsByWall.set(opening.wallId, current + opening.width * opening.height);
  });

  scene.walls.forEach((wall) => {
    if (wall.length <= 0 || wall.height <= 0 || wall.thickness <= 0) {
      warnings.push(`Wall "${wall.id}" has non-positive dimensions and was skipped.`);
      return;
    }

    const elementType: CostReportElementType =
      classifyWall(wall) === "partition" ? "partition_wall" : "load_bearing_wall";
    const material =
      options?.elementMaterialOverrides?.[wall.id] ?? materialForElement(elementType, options?.materialByElement);
    const grossVolume = wall.length * wall.height * wall.thickness;
    const grossArea = wall.length * wall.height;
    const openingArea = openingsByWall.get(wall.id) ?? 0;
    const deductedVolume = Math.min(grossVolume, openingArea * wall.thickness);
    const netVolume = Math.max(0, grossVolume - deductedVolume);
    const netArea = Math.max(0, grossArea - openingArea);
    const wastage = defaultWastageForElement(elementType) + overrideWastage;

    items.push(
      withCost(
        {
          elementId: wall.id,
          elementType,
          material,
          quantityByUnit: {
            m3: netVolume,
            m2: netArea,
            nos: 1,
          },
          wastageFactorRaw: wastage,
          confidence: confidenceToScore(wall.confidence),
          justification: elementJustification(elementType, material),
          assumptions: [
            `Gross wall volume: ${grossVolume.toFixed(3)} m3.`,
            `Gross wall area: ${grossArea.toFixed(3)} m2.`,
            `Opening area deduction: ${openingArea.toFixed(3)} m2.`,
            `Opening deduction: ${deductedVolume.toFixed(3)} m3.`,
          ],
        },
        pricing,
        warnings,
      ),
    );
  });

  scene.rooms.forEach((room) => {
    if (room.area <= 0) {
      warnings.push(`Slab/room "${room.id}" has non-positive area and was skipped.`);
      return;
    }

    const thickness = slabThicknessFromSpan(room.span);
    const volume = room.area * thickness;
    const material =
      options?.elementMaterialOverrides?.[room.id] ?? materialForElement("slab", options?.materialByElement);
    const wastage = defaultWastageForElement("slab") + overrideWastage;
    items.push(
      withCost(
        {
          elementId: room.id,
          elementType: "slab",
          material,
          quantityByUnit: {
            m3: volume,
            m2: room.area,
            nos: 1,
          },
          wastageFactorRaw: wastage,
          confidence: confidenceToScore(room.confidence),
          justification: elementJustification("slab", material),
          assumptions: [
            `Slab volume = area (${room.area.toFixed(3)} m2) x thickness (${thickness.toFixed(3)} m).`,
          ],
        },
        pricing,
        warnings,
      ),
    );
  });

  const nodes = buildStructuralNodes(scene);
  const columns = buildColumns(scene, nodes);
  columns.forEach((column) => {
    if (column.width <= 0 || column.depth <= 0 || column.height <= 0) {
      warnings.push(`Column "${column.id}" has non-positive dimensions and was skipped.`);
      return;
    }

    const volume = column.width * column.depth * column.height;
    const areaProjection = column.width * column.depth;
    const material =
      options?.elementMaterialOverrides?.[column.id] ?? materialForElement("column", options?.materialByElement);
    const wastage = defaultWastageForElement("column") + overrideWastage;
    items.push(
      withCost(
        {
          elementId: column.id,
          elementType: "column",
          material,
          quantityByUnit: {
            m3: volume,
            m2: areaProjection,
            nos: 1,
          },
          wastageFactorRaw: wastage,
          confidence: 0.78,
          justification: elementJustification("column", material),
          assumptions: [
            `Column volume = width (${column.width.toFixed(3)}) x depth (${column.depth.toFixed(3)}) x height (${column.height.toFixed(3)}).`,
          ],
        },
        pricing,
        warnings,
      ),
    );
  });

  if (includeBeamCandidates) {
    const beamCandidates = beamCandidatesFromRooms(scene.rooms);
    beamCandidates.forEach((beam) => {
      const material =
        options?.elementMaterialOverrides?.[beam.id] ?? materialForElement("beam", options?.materialByElement);
      const wastage = defaultWastageForElement("beam") + overrideWastage;
      items.push(
        withCost(
        {
          elementId: beam.id,
          elementType: "beam",
          material,
          quantityByUnit: {
            m3: beam.volume,
            m2: beam.length * beam.width,
            nos: 1,
          },
          wastageFactorRaw: wastage,
            confidence: confidenceToScore(beam.room.confidence) - 0.06,
            justification: elementJustification("beam", material, beam.length),
            assumptions: [
              `Beam candidate inferred from room span ${beam.room.span.toFixed(2)}m.`,
              `Beam volume = length (${beam.length.toFixed(3)}) x width (${beam.width.toFixed(3)}) x depth (${beam.depth.toFixed(3)}).`,
            ],
          },
          pricing,
          warnings,
        ),
      );
    });
    if (beamCandidates.length === 0) {
      warnings.push("No beam candidates detected (no room span >= 4.5m).");
    }
  }

  const totalArea = scene.rooms.reduce((acc, room) => acc + Math.max(0, room.area), 0);
  const totalCost = round3(items.reduce((acc, item) => acc + item.subtotal, 0));
  const materialTotals = items.reduce<Record<string, number>>((acc, item) => {
    acc[item.material] = round3((acc[item.material] ?? 0) + item.subtotal);
    return acc;
  }, {});
  const materialCatalog = buildMaterialCatalog(pricing);

  const report: CostReport = {
    projectName: options?.projectName?.trim() || "SIS Structural Estimate",
    currency: options?.currency?.trim() || "INR",
    generatedAt: new Date().toISOString(),
    scenarioId: options?.scenarioId,
    scenarioLabel: options?.scenarioLabel,
    scenarioDescription: options?.scenarioDescription,
    summary: {
      totalCost,
      totalArea: round3(totalArea),
      costPerSqm: totalArea > 0 ? round3(totalCost / totalArea) : undefined,
    },
    items,
    materialTotals,
    materialCatalog,
    assumptions,
    warnings,
  };

  return report;
}

export function generateCostReportOptions(
  scene: SceneData,
  options?: Omit<CostReportOptions, "materialByElement" | "pricing" | "scenarioId" | "scenarioLabel" | "scenarioDescription">,
): CostScenarioOption[] {
  return COST_SCENARIOS.map((scenario) => {
    const report = generateCostReport(scene, {
      ...options,
      scenarioId: scenario.id,
      scenarioLabel: scenario.label,
      scenarioDescription: scenario.description,
      materialByElement: scenario.materialByElement,
      pricing: scenario.pricing,
      wastageOverridePercent: (options?.wastageOverridePercent ?? 0) + scenario.wastageDeltaPercent,
    });

    return {
      id: scenario.id,
      label: scenario.label,
      description: scenario.description,
      report,
    };
  });
}
