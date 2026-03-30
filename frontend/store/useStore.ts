"use client";

import { create } from "zustand";

import { generateExplanation } from "@/lib/explainer";
import { recommendMaterials } from "@/lib/materialEngine";
import { buildColumns, buildStructuralNodes } from "@/lib/sceneGraph";
import type { RawSceneInput, Recommendation, SceneData, SceneSelection, ValidationIssue, ValidationResult } from "@/lib/types";
import { validateAndNormalize } from "@/lib/validation";

type ViewerState = {
  rawInput: RawSceneInput | null;
  scene: SceneData;
  issues: ValidationIssue[];
  usedFallbackDataset: boolean;
  selectedEntity: SceneSelection | null;
  structuralView: boolean;
  debugOverlay: boolean;
  recommendations: Recommendation[];
  explanation: string;
  loadRawInput: (rawInput: RawSceneInput | unknown) => void;
  selectEntity: (selection: SceneSelection | null) => void;
  toggleStructuralView: () => void;
  toggleDebugOverlay: () => void;
  clearScene: () => void;
};

function deriveSelection(scene: SceneData, selectedEntity: SceneSelection | null): SceneSelection | null {
  if (!selectedEntity) {
    return null;
  }

  if (selectedEntity.type === "wall" && scene.walls.some((wall) => wall.id === selectedEntity.id)) {
    return selectedEntity;
  }

  if (
    (selectedEntity.type === "door" || selectedEntity.type === "window") &&
    scene.openings.some((opening) => opening.id === selectedEntity.id)
  ) {
    return selectedEntity;
  }

  if (selectedEntity.type === "slab" && scene.rooms.some((room) => room.id === selectedEntity.id)) {
    return selectedEntity;
  }

  const nodes = buildStructuralNodes(scene);
  if (selectedEntity.type === "node" && nodes.some((node) => node.id === selectedEntity.id)) {
    return selectedEntity;
  }

  const columns = buildColumns(scene, nodes);
  if (selectedEntity.type === "column" && columns.some((column) => column.id === selectedEntity.id)) {
    return selectedEntity;
  }

  return null;
}

function deriveOutputs(scene: SceneData, issues: ValidationIssue[], selectedEntity: SceneSelection | null) {
  const resolvedSelection = deriveSelection(scene, selectedEntity);
  if (!resolvedSelection) {
    return {
      selectedEntity: null,
      recommendations: [],
      explanation: "Upload a schema-compliant JSON file and select a wall, slab, node, column, door, or window.",
    };
  }

  if (resolvedSelection.type === "wall") {
    const selectedWall = scene.walls.find((wall) => wall.id === resolvedSelection.id);
    if (!selectedWall) {
      return {
        selectedEntity: null,
        recommendations: [],
        explanation: "Upload a schema-compliant JSON file and select a wall, slab, node, column, door, or window.",
      };
    }

    const recommendations = recommendMaterials(selectedWall, scene);
    const explanation = generateExplanation(selectedWall, recommendations, scene, issues);

    return {
      selectedEntity: resolvedSelection,
      recommendations,
      explanation,
    };
  }

  if (resolvedSelection.type === "node") {
    const node = buildStructuralNodes(scene).find((entry) => entry.id === resolvedSelection.id);
    return {
      selectedEntity: resolvedSelection,
      recommendations: [],
      explanation: node
        ? `Node ${node.id} is a ${node.type} with degree ${node.degree}, connecting ${node.connectedWallIds.join(", ")}. ${
            node.likelyColumn
              ? "Its connectivity pattern makes it a likely structural support location, so it is promoted into the column layer."
              : "It stays in the interaction layer as a semantic junction marker rather than wall geometry."
          }`
        : "The selected node could not be resolved from the current wall graph.",
    };
  }

  if (resolvedSelection.type === "column") {
    const nodes = buildStructuralNodes(scene);
    const column = buildColumns(scene, nodes).find((entry) => entry.id === resolvedSelection.id);
    return {
      selectedEntity: resolvedSelection,
      recommendations: [],
      explanation: column
        ? `Column ${column.id} is inferred from ${column.nodeId} at a high-connectivity junction. It anchors ${column.connectedWallIds.join(", ")} and gives the demo a separate render and interaction target for likely support points.`
        : "The selected column could not be resolved from the current wall graph.",
    };
  }

  if (resolvedSelection.type === "slab") {
    const room = scene.rooms.find((entry) => entry.id === resolvedSelection.id);
    return {
      selectedEntity: resolvedSelection,
      recommendations: [],
      explanation: room
        ? `Slab ${room.id} maps to room "${room.name}" with a ${room.span.toFixed(2)}m dominant span across ${room.area.toFixed(2)}m2. Its closed polygon loop is used to keep floor geometry deterministic and structurally aligned with surrounding walls.`
        : "The selected slab could not be resolved from the current room graph.",
    };
  }

  const selectedOpening = scene.openings.find((opening) => opening.id === resolvedSelection.id);
  const hostWall = selectedOpening ? scene.walls.find((wall) => wall.id === selectedOpening.wallId) : null;

  return {
    selectedEntity: resolvedSelection,
    recommendations: [],
    explanation: selectedOpening
      ? `${selectedOpening.kind === "door" ? "Door" : "Window"} ${selectedOpening.id} is attached to wall ${selectedOpening.wallId} at ${selectedOpening.offset.toFixed(2)}m. ${
          hostWall
            ? `The host wall is classified as ${hostWall.type.replace("_", " ")}, so the opening stays separate from the wall mesh but keeps its semantic link.`
            : "Its host wall is missing from the normalized scene."
        }`
      : "The selected opening could not be resolved from the current scene.",
  };
}

const initialValidation = validateAndNormalize(null);
const initialDerived = deriveOutputs(initialValidation.data, initialValidation.issues, null);

export const useStore = create<ViewerState>((set) => ({
  rawInput: null,
  scene: initialValidation.data,
  issues: initialValidation.issues,
  usedFallbackDataset: false,
  selectedEntity: initialDerived.selectedEntity,
  structuralView: true,
  debugOverlay: false,
  recommendations: initialDerived.recommendations,
  explanation: initialDerived.explanation,
  loadRawInput: (rawInput) =>
    set(() => {
      const result: ValidationResult = validateAndNormalize(rawInput);
      const derived = deriveOutputs(result.data, result.issues, null);

      return {
        rawInput: rawInput as RawSceneInput,
        scene: result.data,
        issues: result.issues,
        usedFallbackDataset: false,
        selectedEntity: derived.selectedEntity,
        recommendations: derived.recommendations,
        explanation: derived.explanation,
      };
    }),
  selectEntity: (selection) =>
    set((state) => {
      const derived = deriveOutputs(state.scene, state.issues, selection);

      return {
        selectedEntity: derived.selectedEntity,
        recommendations: derived.recommendations,
        explanation: derived.explanation,
      };
    }),
  toggleStructuralView: () =>
    set((state) => ({
      structuralView: !state.structuralView,
    })),
  toggleDebugOverlay: () =>
    set((state) => ({
      debugOverlay: !state.debugOverlay,
    })),
  clearScene: () =>
    set(() => {
      const result = validateAndNormalize(null);
      const derived = deriveOutputs(result.data, result.issues, null);

      return {
        rawInput: null,
        scene: result.data,
        issues: result.issues,
        usedFallbackDataset: false,
        selectedEntity: derived.selectedEntity,
        recommendations: derived.recommendations,
        explanation: derived.explanation,
      };
    }),
}));
