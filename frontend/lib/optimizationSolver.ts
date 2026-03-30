import type {
  HeuristicReport,
  OptimizationAction,
  OptimizationPreviewLine,
  RawSceneInput,
  SceneData,
  ValidationIssue,
} from "@/lib/types";
import { applyOpeningClashFix } from "@/lib/validation";

function asObject(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }
  return value as Record<string, unknown>;
}

function asPoint2D(value: unknown): [number, number] | undefined {
  if (!Array.isArray(value) || value.length !== 2) {
    return undefined;
  }
  const [x, z] = value;
  if (typeof x !== "number" || typeof z !== "number" || !Number.isFinite(x) || !Number.isFinite(z)) {
    return undefined;
  }
  return [x, z];
}

function deepClone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function overlap(listA: string[], listB: string[]) {
  const setB = new Set(listB);
  return listA.some((entry) => setB.has(entry));
}

type OpeningFixMeta = {
  openingId: string;
  wallId: string;
  issue: ValidationIssue;
};

type AlignmentMeta = {
  anchorWallId: string;
  movingWallId: string;
};

type WallRemovalMeta = {
  wallId: string;
};

type ActionMeta = {
  openingFix?: OpeningFixMeta;
  alignment?: AlignmentMeta;
  wallRemoval?: WallRemovalMeta;
};

export function buildOptimizationActions(
  scene: SceneData,
  heuristics: HeuristicReport | null,
  issues: ValidationIssue[],
) {
  const actions: OptimizationAction[] = [];
  const actionMeta = new Map<string, ActionMeta>();

  const clashByOpening = new Map<string, ValidationIssue>();
  issues.forEach((issue) => {
    if (issue.fixType !== "opening_clash" || !issue.openingId || !issue.wallId) {
      return;
    }
    const key = `${issue.openingId}|${issue.wallId}`;
    if (!clashByOpening.has(key)) {
      clashByOpening.set(key, issue);
    }
  });

  clashByOpening.forEach((issue) => {
    const openingId = issue.openingId as string;
    const wallId = issue.wallId as string;
    const id = `opt-opening-${openingId}-${wallId}`;
    actions.push({
      id,
      kind: "opening_clash_fix",
      state: "idle",
      safeToApply: true,
      conflictGroupId: `wall:${wallId}`,
      relatedWallIds: [wallId],
      relatedOpeningId: openingId,
      sourceIssueId: issue.id,
      confidence: "high",
      severity: "high",
      title: `Opening clash fix (${openingId})`,
      issue: issue.message,
      suggestion: `Apply deterministic clash repair for opening ${openingId} on wall ${wallId}.`,
      impact: "Resolves geometric clash and keeps minimum clearances consistent.",
      impactBasis: "Uses existing deterministic opening clash fix routine.",
      evidence: [`Issue source: ${issue.id}.`],
      assumptions: ["Opening clash fix is deterministic and schema-safe."],
    });
    actionMeta.set(id, { openingFix: { openingId, wallId, issue } });
  });

  (heuristics?.suggestions ?? []).forEach((suggestion) => {
    if (suggestion.type === "ALIGNMENT_FIX" && suggestion.relatedWallIds.length >= 2) {
      const [anchorWallId, movingWallId] = suggestion.relatedWallIds;
      const id = `opt-align-${anchorWallId}-${movingWallId}`;
      actions.push({
        id,
        kind: "alignment_snap",
        state: "idle",
        safeToApply: true,
        conflictGroupId: `walls:${[...suggestion.relatedWallIds].sort().join("|")}`,
        relatedWallIds: suggestion.relatedWallIds,
        relatedRoomId: suggestion.relatedRoomId,
        sourceSuggestionId: suggestion.id,
        confidence: suggestion.confidence,
        severity: suggestion.severity,
        title: "Alignment snap",
        issue: suggestion.issue,
        suggestion: suggestion.suggestion,
        impact: suggestion.impact,
        impactBasis: suggestion.impactBasis,
        evidence: suggestion.evidence,
        assumptions: suggestion.assumptions,
      });
      actionMeta.set(id, { alignment: { anchorWallId, movingWallId } });
      return;
    }

    if (suggestion.type === "WALL_REMOVAL" && suggestion.relatedWallIds[0]) {
      const wallId = suggestion.relatedWallIds[0];
      const id = `opt-remove-${wallId}`;
      actions.push({
        id,
        kind: "wall_removal",
        state: "idle",
        safeToApply: true,
        conflictGroupId: `wall:${wallId}`,
        relatedWallIds: [wallId],
        relatedRoomId: suggestion.relatedRoomId,
        sourceSuggestionId: suggestion.id,
        confidence: suggestion.confidence,
        severity: suggestion.severity,
        title: "Guarded wall removal",
        issue: suggestion.issue,
        suggestion: suggestion.suggestion,
        impact: suggestion.impact,
        impactBasis: suggestion.impactBasis,
        evidence: suggestion.evidence,
        assumptions: suggestion.assumptions,
      });
      actionMeta.set(id, { wallRemoval: { wallId } });
      return;
    }

    const id = `opt-adv-${suggestion.id}`;
    actions.push({
      id,
      kind: "advisory",
      state: "idle",
      safeToApply: false,
      conflictGroupId: suggestion.relatedWallIds[0] ? `wall:${suggestion.relatedWallIds[0]}` : undefined,
      relatedWallIds: suggestion.relatedWallIds,
      relatedRoomId: suggestion.relatedRoomId,
      sourceSuggestionId: suggestion.id,
      confidence: suggestion.confidence,
      severity: suggestion.severity,
      title: suggestion.type.replace(/_/g, " "),
      issue: suggestion.issue,
      suggestion: suggestion.suggestion,
      impact: suggestion.impact,
      impactBasis: suggestion.impactBasis,
      evidence: suggestion.evidence,
      assumptions: suggestion.assumptions,
    });
  });

  const ordered = [...actions].sort((a, b) => {
    if (a.safeToApply !== b.safeToApply) {
      return a.safeToApply ? -1 : 1;
    }
    const severityRank = { high: 3, medium: 2, low: 1 };
    const severityDelta = severityRank[b.severity] - severityRank[a.severity];
    if (severityDelta !== 0) {
      return severityDelta;
    }
    return a.id.localeCompare(b.id);
  });

  return { actions: ordered, actionMeta };
}

export function deriveActionStates(actions: OptimizationAction[], pendingIds: Set<string>) {
  return actions.map((action) => {
    if (!action.safeToApply) {
      return action;
    }
    if (pendingIds.has(action.id)) {
      return { ...action, state: "pending" as const };
    }
    const hasConflictingPending = actions.some((candidate) => {
      if (!pendingIds.has(candidate.id) || candidate.id === action.id) {
        return false;
      }
      if (candidate.conflictGroupId && action.conflictGroupId && candidate.conflictGroupId === action.conflictGroupId) {
        return true;
      }
      if (candidate.relatedOpeningId && action.relatedOpeningId && candidate.relatedOpeningId === action.relatedOpeningId) {
        return true;
      }
      return overlap(candidate.relatedWallIds, action.relatedWallIds);
    });
    if (hasConflictingPending) {
      return { ...action, state: "blocked" as const };
    }
    return { ...action, state: "idle" as const };
  });
}

function applyAlignment(rawInput: RawSceneInput, anchorWallId: string, movingWallId: string) {
  const payload = deepClone(rawInput);
  const wallEntries = Array.isArray(payload.walls) ? payload.walls : [];
  const anchor = wallEntries.find((entry) => asObject(entry)?.id === anchorWallId);
  const moving = wallEntries.find((entry) => asObject(entry)?.id === movingWallId);
  const anchorObj = asObject(anchor);
  const movingObj = asObject(moving);

  if (!anchorObj || !movingObj) {
    return null;
  }

  const anchorStart = asPoint2D(anchorObj.start);
  const anchorEnd = asPoint2D(anchorObj.end);
  const movingStart = asPoint2D(movingObj.start);
  const movingEnd = asPoint2D(movingObj.end);
  if (!anchorStart || !anchorEnd || !movingStart || !movingEnd) {
    return null;
  }

  const anchorDx = Math.abs(anchorEnd[0] - anchorStart[0]);
  const anchorDz = Math.abs(anchorEnd[1] - anchorStart[1]);
  const anchorAngleDeg = (Math.atan2(anchorEnd[1] - anchorStart[1], anchorEnd[0] - anchorStart[0]) * 180) / Math.PI;
  const normalized = Math.abs(anchorAngleDeg) % 180;
  const distanceToOrthogonal = Math.min(
    Math.abs(normalized),
    Math.abs(normalized - 90),
    Math.abs(normalized - 180),
  );
  if (distanceToOrthogonal > 8) {
    return null;
  }
  const horizontal = anchorDx >= anchorDz;
  if (horizontal) {
    const targetAxis = (anchorStart[1] + anchorEnd[1]) / 2;
    movingObj.start = [movingStart[0], targetAxis];
    movingObj.end = [movingEnd[0], targetAxis];
  } else {
    const targetAxis = (anchorStart[0] + anchorEnd[0]) / 2;
    movingObj.start = [targetAxis, movingStart[1]];
    movingObj.end = [targetAxis, movingEnd[1]];
  }

  return payload;
}

function applyWallRemoval(rawInput: RawSceneInput, wallId: string) {
  const payload = deepClone(rawInput);
  const wallsArray = Array.isArray(payload.walls) ? payload.walls : null;
  if (!wallsArray) {
    return null;
  }
  const nextWalls = wallsArray.filter((entry) => asObject(entry)?.id !== wallId);
  if (nextWalls.length === wallsArray.length) {
    return null;
  }
  payload.walls = nextWalls;
  if (Array.isArray(payload.doors)) {
    payload.doors = payload.doors.filter((entry) => asObject(entry)?.wallId !== wallId);
  }
  if (Array.isArray(payload.windows)) {
    payload.windows = payload.windows.filter((entry) => asObject(entry)?.wallId !== wallId);
  }
  return payload;
}

export function applyOptimizationAction(
  rawInput: RawSceneInput,
  action: OptimizationAction,
  actionMeta: Map<string, ActionMeta>,
) {
  if (!action.safeToApply) {
    return null;
  }

  const meta = actionMeta.get(action.id);
  if (!meta) {
    return null;
  }

  if (action.kind === "opening_clash_fix" && meta.openingFix) {
    return applyOpeningClashFix(rawInput, meta.openingFix.issue);
  }

  if (action.kind === "alignment_snap" && meta.alignment) {
    return applyAlignment(rawInput, meta.alignment.anchorWallId, meta.alignment.movingWallId);
  }

  if (action.kind === "wall_removal" && meta.wallRemoval) {
    return applyWallRemoval(rawInput, meta.wallRemoval.wallId);
  }

  return null;
}

export function buildOptimizationPreviewLines(scene: SceneData, actions: OptimizationAction[], pendingIds: Set<string>) {
  const previews: OptimizationPreviewLine[] = [];
  const wallMap = new Map(scene.walls.map((wall) => [wall.id, wall]));

  actions.forEach((action) => {
    if (!pendingIds.has(action.id) || !action.safeToApply) {
      return;
    }

    if (action.kind === "alignment_snap" && action.relatedWallIds.length >= 2) {
      const anchor = wallMap.get(action.relatedWallIds[0]);
      const moving = wallMap.get(action.relatedWallIds[1]);
      if (!anchor || !moving) {
        return;
      }
      const anchorDx = Math.abs(anchor.end[0] - anchor.start[0]);
      const anchorDz = Math.abs(anchor.end[1] - anchor.start[1]);
      const horizontal = anchorDx >= anchorDz;
      const from: [number, number, number] = [moving.midpoint[0], Math.max(moving.height + 0.2, 2.6), moving.midpoint[2]];
      const to: [number, number, number] = horizontal
        ? [moving.midpoint[0], Math.max(moving.height + 0.2, 2.6), (anchor.start[1] + anchor.end[1]) / 2]
        : [(anchor.start[0] + anchor.end[0]) / 2, Math.max(moving.height + 0.2, 2.6), moving.midpoint[2]];
      previews.push({ id: `${action.id}-align`, from, to, color: "#22d3ee", label: "alignment preview" });
      return;
    }

    if (action.kind === "wall_removal" && action.relatedWallIds[0]) {
      const wall = wallMap.get(action.relatedWallIds[0]);
      if (!wall) {
        return;
      }
      previews.push({
        id: `${action.id}-remove`,
        from: [wall.start[0], 0.14, wall.start[1]],
        to: [wall.end[0], 0.14, wall.end[1]],
        color: "#fb7185",
        label: "removal preview",
      });
      return;
    }

    if (action.kind === "opening_clash_fix" && action.relatedWallIds[0]) {
      const wall = wallMap.get(action.relatedWallIds[0]);
      if (!wall) {
        return;
      }
      previews.push({
        id: `${action.id}-opening`,
        from: [wall.midpoint[0], 0.12, wall.midpoint[2]],
        to: [wall.midpoint[0], Math.max(2.4, wall.height - 0.3), wall.midpoint[2]],
        color: "#f59e0b",
        label: "opening fix preview",
      });
    }
  });

  return previews;
}
