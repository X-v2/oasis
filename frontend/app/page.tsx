"use client";

import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";

import MaterialPanel from "@/components/MaterialPanel";
import Viewer3D from "@/components/Viewer3D";
import { buildColumns, buildStructuralNodes } from "@/lib/sceneGraph";
import { classifyWall, getRoomContextForWall } from "@/lib/materialEngine";
import {
  applyOptimizationAction,
  buildOptimizationActions,
  buildOptimizationPreviewLines,
  deriveActionStates,
} from "@/lib/optimizationSolver";
import { getReadinessLabel } from "@/lib/sceneInsights";
import type {
  HeuristicConfidence,
  HeuristicReport,
  HeuristicSeverity,
  MaterialRecommendationTable,
  NormalizedRoom,
  NormalizedWall,
  OptimizationAction,
  Point2D,
  Point3D,
  RawSceneInput,
  RecommendationApiResponse,
  SpanPreviewLine,
  ValidationIssue,
} from "@/lib/types";
import { applyOpeningClashFix } from "@/lib/validation";
import { useStore } from "@/store/useStore";

function severityClass(severity: "error" | "warning" | "info") {
  if (severity === "error") return "error";
  if (severity === "warning") return "warning";
  return "info";
}

function heuristicSeverityBadge(severity: HeuristicSeverity) {
  if (severity === "high") return "badge-danger";
  if (severity === "medium") return "badge-warning";
  return "badge-info";
}

function heuristicConfidenceBadge(confidence: "low" | "medium" | "high") {
  if (confidence === "high") return "badge-primary";
  if (confidence === "medium") return "badge-warning";
  return "badge-neutral";
}

function heuristicSeverityMeaning(severity: HeuristicSeverity) {
  if (severity === "high") return "High impact risk if ignored.";
  if (severity === "medium") return "Useful improvement, moderate impact.";
  return "Minor refinement, low impact.";
}

function heuristicConfidenceMeaning(confidence: HeuristicConfidence) {
  if (confidence === "high") return "Strong evidence from geometry and rules.";
  if (confidence === "medium") return "Reasonable evidence, some assumptions.";
  return "Weak evidence, treat as exploratory.";
}

function optimizationInterpretation(severity: HeuristicSeverity, confidence: HeuristicConfidence) {
  if (severity === "high" && confidence === "high") {
    return "Priority: act early. Big downside if ignored, and evidence is strong.";
  }
  if (severity === "high" && confidence === "low") {
    return "Investigate first: potentially serious issue, but evidence is uncertain.";
  }
  if (severity === "low" && confidence === "high") {
    return "Safe polish item: low risk, but recommendation is reliable.";
  }
  if (severity === "low" && confidence === "low") {
    return "Optional idea: low risk and low certainty.";
  }
  return "Moderate priority: validate context, then apply if it aligns with intent.";
}

function formatHeuristicType(value: string) {
  return value.replace(/_/g, " ");
}

function formatPoint3D(point: Point3D) {
  return `${point[0].toFixed(2)}, ${point[1].toFixed(2)}, ${point[2].toFixed(2)}`;
}

function formatPoint2D(point: Point2D) {
  return `${point[0].toFixed(2)}, ${point[1].toFixed(2)}`;
}

function approximateMatch(a: Point2D, b: Point2D, tolerance = 0.08) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]) <= tolerance;
}

function roomWallIds(room: NormalizedRoom, walls: NormalizedWall[]) {
  return walls
    .filter((wall) =>
      room.polygon2D.some((start, index) => {
        const end = room.polygon2D[(index + 1) % room.polygon2D.length];
        const direct = approximateMatch(start, wall.start) && approximateMatch(end, wall.end);
        const reverse = approximateMatch(start, wall.end) && approximateMatch(end, wall.start);
        return direct || reverse;
      }),
    )
    .map((wall) => wall.id);
}

type UploadState = "idle" | "uploading" | "success" | "error";

function isErrorResponse(value: unknown): value is { error: string } {
  if (!value || typeof value !== "object") {
    return false;
  }
  const maybeError = (value as { error?: unknown }).error;
  return typeof maybeError === "string";
}

export default function Home() {
  const [darkMode, setDarkMode] = useState(true);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("theme-dark");
    } else {
      document.documentElement.classList.remove("theme-dark");
    }
  }, [darkMode]);
  const [recommendationTable, setRecommendationTable] = useState<MaterialRecommendationTable | null>(null);
  const [heuristicReport, setHeuristicReport] = useState<HeuristicReport | null>(null);
  const [recommendationLoading, setRecommendationLoading] = useState(false);
  const [issueFocusPoint, setIssueFocusPoint] = useState<Point3D | null>(null);
  const [issueFocusToken, setIssueFocusToken] = useState(0);
  const [pendingFixIssueKeys, setPendingFixIssueKeys] = useState<Record<string, boolean>>({});
  const [pendingOptimizationActionIds, setPendingOptimizationActionIds] = useState<Record<string, boolean>>({});
  const recommendationCacheRef = useRef<Map<string, RecommendationApiResponse>>(new Map());
  const [uploadMessage, setUploadMessage] = useState(
    "Upload a JSON model with coordinates for walls, slabs, labels, openings, graphNodes, and columns.",
  );
  const [uploadState, setUploadState] = useState<UploadState>("idle");
  const [uploadResultText, setUploadResultText] = useState<string | null>(null);
  const {
    scene,
    issues,
    selectedEntity,
    structuralView,
    debugOverlay,
    loadRawInput,
    selectEntity,
    toggleStructuralView,
    toggleDebugOverlay,
    clearScene,
    rawInput,
  } = useStore();
  const derivedNodes = useMemo(() => buildStructuralNodes(scene), [scene]);
  const derivedColumns = useMemo(() => buildColumns(scene, derivedNodes), [derivedNodes, scene]);
  const selectedWall = useMemo(
    () => (selectedEntity?.type === "wall" ? scene.walls.find((wall) => wall.id === selectedEntity.id) ?? null : null),
    [scene.walls, selectedEntity],
  );
  const selectedNode = useMemo(
    () => (selectedEntity?.type === "node" ? derivedNodes.find((node) => node.id === selectedEntity.id) ?? null : null),
    [derivedNodes, selectedEntity],
  );
  const selectedColumn = useMemo(
    () =>
      selectedEntity?.type === "column" ? derivedColumns.find((column) => column.id === selectedEntity.id) ?? null : null,
    [derivedColumns, selectedEntity],
  );
  const selectedSlab = useMemo(
    () => (selectedEntity?.type === "slab" ? scene.rooms.find((room) => room.id === selectedEntity.id) ?? null : null),
    [scene.rooms, selectedEntity],
  );
  const selectedOpening = useMemo(
    () =>
      selectedEntity?.type === "door" || selectedEntity?.type === "window"
        ? scene.openings.find((opening) => opening.id === selectedEntity.id) ?? null
        : null,
    [scene.openings, selectedEntity],
  );
  const selectedEntityType = selectedEntity?.type ?? "none";
  const selectedEntityId = selectedEntity?.id ?? "-";
  const selectedMetrics = useMemo(() => {
    if (!selectedEntity) {
      return "";
    }

    if (selectedWall) {
      return `Wall ${selectedWall.id}: length ${selectedWall.length.toFixed(2)}m, height ${selectedWall.height.toFixed(2)}m, thickness ${selectedWall.thickness.toFixed(2)}m.`;
    }

    if (selectedSlab) {
      return `Slab ${selectedSlab.id}: span ${selectedSlab.span.toFixed(2)}m, area ${selectedSlab.area.toFixed(2)}m2.`;
    }

    if (selectedColumn) {
      return `Column ${selectedColumn.id}: ${selectedColumn.width.toFixed(2)}m x ${selectedColumn.depth.toFixed(2)}m x ${selectedColumn.height.toFixed(2)}m.`;
    }

    if (selectedNode) {
      return `Node ${selectedNode.id}: degree ${selectedNode.degree}, connected walls ${selectedNode.connectedWallIds.join(", ") || "none"}.`;
    }

    if (selectedOpening) {
      const windowTypeText =
        selectedOpening.kind === "window" ? `, type ${selectedOpening.panelType ?? "double"}` : "";
      return `${selectedOpening.kind} ${selectedOpening.id}: width ${selectedOpening.width.toFixed(2)}m, height ${selectedOpening.height.toFixed(2)}m, offset ${selectedOpening.offset.toFixed(2)}m on wall ${selectedOpening.wallId}${windowTypeText}.`;
    }

    return "";
  }, [selectedColumn, selectedEntity, selectedNode, selectedOpening, selectedSlab, selectedWall]);

  useEffect(() => {
    recommendationCacheRef.current.clear();
  }, [scene]);

  useEffect(() => {
    const issueKeySet = new Set(issues.map((issue) => issueKey(issue)));
    setPendingFixIssueKeys((current) => {
      const next: Record<string, boolean> = {};
      Object.entries(current).forEach(([key, value]) => {
        if (value && issueKeySet.has(key)) {
          next[key] = true;
        }
      });
      return next;
    });
  }, [issues]);

  useEffect(() => {
    if (scene.walls.length === 0) {
      setRecommendationTable(null);
      setHeuristicReport(null);
      setRecommendationLoading(false);
      return;
    }

    const requestKey = `${selectedEntityType}:${selectedEntityId}:${selectedMetrics || "-"}`;
    const cached = recommendationCacheRef.current.get(requestKey);
    if (cached) {
      setRecommendationTable(cached.materialTable);
      setHeuristicReport(cached.heuristics);
      setRecommendationLoading(false);
      return;
    }

    const controller = new AbortController();

    async function loadRecommendationTable() {
      setRecommendationLoading(true);
      try {
        const response = await fetch("/api/recommendations", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ scene, selectedEntity, selectionMetrics: selectedMetrics }),
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error("Recommendation request failed");
        }

        const payload = (await response.json()) as RecommendationApiResponse;
        recommendationCacheRef.current.set(requestKey, payload);
        setRecommendationTable(payload.materialTable);
        setHeuristicReport(payload.heuristics);
      } catch {
        if (!controller.signal.aborted) {
          setRecommendationTable(null);
          setHeuristicReport(null);
        }
      } finally {
        if (!controller.signal.aborted) {
          setRecommendationLoading(false);
        }
      }
    }

    loadRecommendationTable();
    return () => controller.abort();
  }, [scene, selectedEntity, selectedEntityId, selectedEntityType, selectedMetrics]);

  const selectedWallContext = useMemo(
    () => (selectedWall ? getRoomContextForWall(selectedWall, scene) : null),
    [scene, selectedWall],
  );
  const selectedWallNodeIds = useMemo(
    () =>
      selectedWall
        ? derivedNodes.filter((node) => node.connectedWallIds.includes(selectedWall.id)).map((node) => node.id)
        : [],
    [derivedNodes, selectedWall],
  );
  const selectedWallOpeningIds = useMemo(
    () => (selectedWall ? scene.openings.filter((opening) => opening.wallId === selectedWall.id).map((opening) => opening.id) : []),
    [scene.openings, selectedWall],
  );
  const selectedSlabWallIds = useMemo(
    () => (selectedSlab ? roomWallIds(selectedSlab, scene.walls) : []),
    [scene.walls, selectedSlab],
  );
  const selectedSlabNodeIds = useMemo(
    () =>
      selectedSlab
        ? Array.from(
            new Set(
              selectedSlab.polygon2D
                .map((point) =>
                  derivedNodes.find((node) => Math.hypot(node.position[0] - point[0], node.position[2] - point[1]) < 0.08)?.id,
                )
                .filter((entry): entry is string => Boolean(entry)),
            ),
          )
        : [],
    [derivedNodes, selectedSlab],
  );
  const selectedSlabThickness = useMemo(
    () => (selectedSlab ? Math.max(0.12, Math.min(0.24, 0.12 + Math.max(selectedSlab.span - 3, 0) * 0.018)) : 0),
    [selectedSlab],
  );
  const openingTypeById = useMemo(
    () => new Map(scene.openings.map((opening) => [opening.id, opening.kind] as const)),
    [scene.openings],
  );

  const issueCounts = useMemo(
    () => ({
      errors: issues.filter((issue) => issue.severity === "error").length,
      warnings: issues.filter((issue) => issue.severity === "warning").length,
      infos: issues.filter((issue) => issue.severity === "info").length,
    }),
    [issues],
  );
  const openingCounts = useMemo(
    () => ({
      doors: scene.openings.filter((opening) => opening.kind === "door").length,
      windows: scene.openings.filter((opening) => opening.kind === "window").length,
    }),
    [scene.openings],
  );
  const elementCounts = useMemo(
    () => ({
      rooms: scene.rooms.length,
      slabs: scene.rooms.length,
      walls: scene.walls.length,
      doors: openingCounts.doors,
      windows: openingCounts.windows,
      openings: scene.openings.length,
      graphNodes: derivedNodes.length,
      columns: derivedColumns.length,
      errors: issueCounts.errors,
      warnings: issueCounts.warnings,
      infos: issueCounts.infos,
      issueEntries: issueCounts.errors + issueCounts.warnings + issueCounts.infos,
    }),
    [derivedColumns.length, derivedNodes.length, issueCounts.errors, issueCounts.infos, issueCounts.warnings, openingCounts.doors, openingCounts.windows, scene.openings.length, scene.rooms.length, scene.walls.length],
  );

  const pendingOptimizationIdSet = useMemo(
    () => new Set(Object.keys(pendingOptimizationActionIds).filter((id) => pendingOptimizationActionIds[id])),
    [pendingOptimizationActionIds],
  );
  const optimizationBase = useMemo(
    () => buildOptimizationActions(scene, heuristicReport, issues),
    [heuristicReport, issues, scene],
  );
  const optimizationActions = useMemo(
    () => deriveActionStates(optimizationBase.actions, pendingOptimizationIdSet),
    [optimizationBase.actions, pendingOptimizationIdSet],
  );
  const spanPreviewLines = useMemo<SpanPreviewLine[]>(() => {
    if (!heuristicReport) {
      return [];
    }

    return heuristicReport.suggestions
      .filter((entry) => entry.type === "SPAN_FIX")
      .flatMap((entry) => {
        const relatedWalls = entry.relatedWallIds
          .map((wallId) => scene.walls.find((wall) => wall.id === wallId))
          .filter((wall): wall is NormalizedWall => Boolean(wall));
        if (relatedWalls.length >= 2) {
          const first = relatedWalls[0];
          const second = relatedWalls[1];
          return [
            {
              id: entry.id,
              roomId: entry.relatedRoomId,
              from: [first.midpoint[0], Math.max(first.height + 0.2, 2.6), first.midpoint[2]],
              to: [second.midpoint[0], Math.max(second.height + 0.2, 2.6), second.midpoint[2]],
              confidence: entry.confidence,
            },
          ];
        }

        if (entry.relatedRoomId) {
          const room = scene.rooms.find((candidate) => candidate.id === entry.relatedRoomId);
          if (room?.spanLine) {
            return [
              {
                id: entry.id,
                roomId: room.id,
                from: [room.spanLine[0][0], 2.6, room.spanLine[0][2]],
                to: [room.spanLine[1][0], 2.6, room.spanLine[1][2]],
                confidence: entry.confidence,
              },
            ];
          }
        }

        return [];
      });
  }, [heuristicReport, scene.rooms, scene.walls]);
  const optimizationPreviewLines = useMemo(
    () => buildOptimizationPreviewLines(scene, optimizationActions, pendingOptimizationIdSet),
    [optimizationActions, pendingOptimizationIdSet, scene],
  );
  useEffect(() => {
    const validIds = new Set(optimizationBase.actions.map((action) => action.id));
    setPendingOptimizationActionIds((current) => {
      const next: Record<string, boolean> = {};
      Object.entries(current).forEach(([id, pending]) => {
        if (pending && validIds.has(id)) {
          next[id] = true;
        }
      });
      return next;
    });
  }, [optimizationBase.actions]);
  const readinessLabel = getReadinessLabel(scene.readiness);
  const readinessMessage =
    scene.readiness === "partial"
      ? "Rendered with validation fixes and inferred attributes."
      : scene.readiness === "valid"
        ? "Geometry comes directly from the uploaded schema."
        : "No active model. Upload JSON to generate the scene.";

  const readinessBadge = useMemo(() => {
    if (scene.readiness === "valid") return "badge-primary";
    if (scene.readiness === "partial") return "badge-warning";
    return "badge-danger";
  }, [scene.readiness]);

  const readinessInfoClass = useMemo(() => {
    if (scene.readiness === "valid") return "info";
    if (scene.readiness === "partial") return "warning";
    return "error";
  }, [scene.readiness]);

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    try {
      const content = await file.text();
      const parsed = JSON.parse(content) as unknown;
      setPendingFixIssueKeys({});
      setPendingOptimizationActionIds({});
      loadRawInput(parsed);
      setUploadState("success");
      setUploadResultText(JSON.stringify(parsed, null, 2));
      setUploadMessage(`Loaded ${file.name}. Coordinate-aware walls/slabs/openings/labels/nodes/columns were parsed.`);
    } catch {
      setPendingFixIssueKeys({});
      setPendingOptimizationActionIds({});
      clearScene();
      setUploadState("error");
      setUploadResultText(null);
      setUploadMessage(`Could not parse ${file.name}. The scene was cleared.`);
    } finally {
      event.target.value = "";
    }
  }

  async function handleImageUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setUploadState("uploading");
    setUploadResultText(null);
    setUploadMessage(`Uploading ${file.name} to parser API...`);

    try {
      const formData = new FormData();
      formData.append("image", file, file.name);

      const response = await fetch("/api/parse-image", {
        method: "POST",
        body: formData,
      });

      let payload: unknown = null;
      try {
        payload = await response.json();
      } catch {
        payload = { error: "Invalid JSON response from parse API." };
      }

      console.log("Image parse API response:", payload);
      setUploadResultText(JSON.stringify(payload, null, 2));

      if (!response.ok) {
        const message =
          isErrorResponse(payload) ? payload.error : `Parse API request failed with status ${response.status}.`;
        setUploadState("error");
        setUploadMessage(message);
        return;
      }

      if (isErrorResponse(payload)) {
        setUploadState("error");
        setUploadMessage(payload.error);
        return;
      }

      setPendingFixIssueKeys({});
      setPendingOptimizationActionIds({});
      loadRawInput(payload as RawSceneInput);
      setUploadState("success");
      setUploadMessage(`Image parsed successfully from ${file.name}. 3D view updated from API JSON.`);
    } catch (error) {
      console.error("Image parse API request failed:", error);
      setUploadState("error");
      setUploadResultText(
        JSON.stringify(
          {
            error: error instanceof Error ? error.message : "Unexpected API request failure.",
          },
          null,
          2,
        ),
      );
      setUploadMessage("Could not upload image to parse API. Check network/CORS/API availability.");
    } finally {
      event.target.value = "";
    }
  }

  function issueKey(issue: ValidationIssue) {
    return [issue.fixType ?? "-", issue.openingId ?? "-", issue.wallId ?? "-", issue.message].join("|");
  }

  function selectionFromIssue(issue: (typeof issues)[number]) {
    if (issue.openingId) {
      const openingType = openingTypeById.get(issue.openingId);
      if (openingType === "door" || openingType === "window") {
        return { type: openingType, id: issue.openingId } as const;
      }
    }

    if (issue.wallId && scene.walls.some((wall) => wall.id === issue.wallId)) {
      return { type: "wall" as const, id: issue.wallId };
    }

    if (issue.roomId && scene.rooms.some((room) => room.id === issue.roomId)) {
      return { type: "slab" as const, id: issue.roomId };
    }

    return null;
  }

  function focusPointFromSelection(selection: { type: "wall" | "door" | "window" | "slab"; id: string }) {
    if (selection.type === "wall") {
      const wall = scene.walls.find((entry) => entry.id === selection.id);
      return wall?.midpoint ?? null;
    }

    if (selection.type === "door" || selection.type === "window") {
      const opening = scene.openings.find((entry) => entry.id === selection.id);
      return opening?.position ?? null;
    }

    if (selection.type === "slab") {
      const room = scene.rooms.find((entry) => entry.id === selection.id);
      return room?.centroid ?? null;
    }

    return null;
  }

  function handleIssueSelection(selection: { type: "wall" | "door" | "window" | "slab"; id: string }) {
    selectEntity(selection);
    const target = focusPointFromSelection(selection);
    if (target) {
      setIssueFocusPoint(target);
      setIssueFocusToken((current) => current + 1);
    }
  }

  function handleViewerSelection(selection: Parameters<typeof selectEntity>[0]) {
    selectEntity(selection);
  }

  function freeIssueAngle() {
    setIssueFocusPoint(null);
    setIssueFocusToken((current) => current + 1);
  }

  function handleFixIssue(issue: ValidationIssue, selection: ReturnType<typeof selectionFromIssue>) {
    if (selection) {
      handleIssueSelection(selection);
    }
    const key = issueKey(issue);
    setPendingFixIssueKeys((current) => ({ ...current, [key]: true }));
  }

  function handleConfirmIssueFix(issue: ValidationIssue) {
    if (!rawInput) {
      return;
    }

    const patched = applyOpeningClashFix(rawInput, issue);
    const key = issueKey(issue);
    setPendingFixIssueKeys((current) => {
      const next = { ...current };
      delete next[key];
      return next;
    });
    if (patched) {
      loadRawInput(patched);
    }
    freeIssueAngle();
  }

  function handleCancelPendingFix(issue: ValidationIssue) {
    const key = issueKey(issue);
    setPendingFixIssueKeys((current) => {
      const next = { ...current };
      delete next[key];
      return next;
    });
    freeIssueAngle();
  }

  function handleFixOptimizationAction(action: OptimizationAction) {
    if (!action.safeToApply || action.state === "blocked") {
      return;
    }
    setPendingOptimizationActionIds((current) => ({ ...current, [action.id]: true }));
  }

  function handleRevertOptimizationAction(action: OptimizationAction) {
    setPendingOptimizationActionIds((current) => {
      const next = { ...current };
      delete next[action.id];
      return next;
    });
  }

  function handleConfirmOptimizationAction(action: OptimizationAction) {
    if (!rawInput || !action.safeToApply) {
      return;
    }
    const patched = applyOptimizationAction(rawInput, action, optimizationBase.actionMeta);
    setPendingOptimizationActionIds((current) => {
      const next = { ...current };
      delete next[action.id];
      return next;
    });
    if (patched) {
      loadRawInput(patched);
    }
  }

  return (
    <main
      className={darkMode ? "theme-dark" : ""}
      style={{
        minHeight: "100vh",
        background: "var(--bg-base)",
        color: "var(--fg-primary)",
        padding: "24px 20px 48px",
        transition: "background 0.25s ease, color 0.25s ease",
      }}
    >
      <div style={{ maxWidth: 1680, margin: "0 auto", display: "flex", flexDirection: "column", gap: 20 }}>

        {/* ── HEADER ── */}
        <header className="sis-surface-overlay" style={{ padding: "24px 28px" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            {/* Title row */}
            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 20, flexWrap: "wrap" }}>
              <div style={{ flex: "1 1 400px", minWidth: 0 }}>
                <p className="sis-eyebrow" style={{ marginBottom: 10 }}>Parametric Architectural Engine</p>
                <h1 style={{
                  fontSize: "clamp(20px, 3vw, 30px)",
                  fontWeight: 700,
                  letterSpacing: "-0.025em",
                  lineHeight: 1.2,
                  color: "var(--fg-primary)",
                  margin: 0,
                }}>
                  Structural Reasoning Viewer
                </h1>
                <p style={{
                  marginTop: 8,
                  fontSize: 13,
                  lineHeight: 1.7,
                  color: "var(--fg-secondary)",
                  maxWidth: 560,
                }}>
                  Upload the final schema, validate it visibly, and render a clean interactive 3D model
                  without hardcoded plan geometry.
                </p>
              </div>

              {/* Controls */}
              <div className="header-controls">
                <label className="btn btn-default" style={{ cursor: "pointer" }}>
                  <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
                    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  Upload Image
                  <input type="file" accept=".png,image/png" className="hidden" onChange={handleImageUpload} />
                </label>

                <button
                  type="button"
                  onClick={() => {
                    setPendingFixIssueKeys({});
                    setPendingOptimizationActionIds({});
                    clearScene();
                  }}
                  className="btn btn-default"
                >
                  <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
                    <path d="M3 6h18M19 6l-1 14H6L5 6M9 6V4h6v2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  Clear Scene
                </button>

                <button
                  type="button"
                  onClick={() => setDarkMode((current) => !current)}
                  className="btn btn-default"
                  style={{ minWidth: 44 }}
                  title="Toggle theme"
                >
                  {darkMode ? (
                    <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                      <circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" strokeLinecap="round"/>
                    </svg>
                  ) : (
                    <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                      <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  )}
                  {darkMode ? "Light" : "Dark"}
                </button>
              </div>
            </div>

            {/* Status strip */}
            <div className="status-strip">
              <div className="sis-surface-sunken" style={{ padding: "12px 16px", display: "flex", flexDirection: "column", gap: 4 }}>
                <p className="sis-label" style={{ marginBottom: 4 }}>Upload Status</p>
                <p style={{ fontSize: 13, color: "var(--fg-secondary)", lineHeight: 1.6, margin: 0 }}>{uploadMessage}</p>
                <span className={`api-status ${uploadState}`}>
                  {uploadState === "uploading" && (
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <path d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" opacity="0.3"/>
                      <path d="M12 3a9 9 0 019 9" strokeLinecap="round"><animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="0.8s" repeatCount="indefinite"/></path>
                    </svg>
                  )}
                  {uploadState !== "idle" && uploadState.toUpperCase()}
                </span>
                {uploadResultText ? (
                  <pre className="upload-result">{uploadResultText}</pre>
                ) : null}
              </div>
              <div
                className={`issue-entry ${readinessInfoClass}`}
                style={{ padding: "12px 16px", minWidth: 200, display: "flex", flexDirection: "column", gap: 4 }}
              >
                <p className="sis-label">Model Readiness</p>
                <p style={{ fontSize: 13, fontWeight: 600, margin: 0 }}>{readinessLabel}</p>
                <p style={{ fontSize: 12, lineHeight: 1.6, margin: 0, opacity: 0.85 }}>{readinessMessage}</p>
              </div>
            </div>
          </div>
        </header>

        {/* ── MAIN CONTENT AREA ── */}
        <div className="layout-main">
          {/* 3D Viewer */}
          <div>
            <Viewer3D
              data={scene}
              selectedEntity={selectedEntity}
              structuralView={structuralView}
              debugOverlay={debugOverlay}
              darkMode={darkMode}
              spanPreviewLines={spanPreviewLines}
              optimizationPreviewLines={optimizationPreviewLines}
              onSelectEntity={handleViewerSelection}
              focusPoint={issueFocusPoint}
              focusToken={issueFocusToken}
            />
          </div>

          {/* Right sidebar */}
          <aside style={{ display: "flex", flexDirection: "column", gap: 16 }}>

            {/* Validation Summary */}
            <section className="sis-surface" style={{ padding: "20px 22px" }}>
              <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12, marginBottom: 16 }}>
                <div>
                  <h2 style={{ fontSize: 15, fontWeight: 700, color: "var(--fg-primary)", margin: 0, letterSpacing: "-0.01em" }}>
                    Validation Summary
                  </h2>
                  <p style={{ fontSize: 12, color: "var(--fg-muted)", marginTop: 4, lineHeight: 1.5 }}>
                    Quality snapshot for current geometry.
                  </p>
                </div>
                <span className={`badge ${readinessBadge}`}>{readinessLabel}</span>
              </div>

              {/* Element counts grid */}
              <div className="sis-surface-sunken" style={{ padding: "14px 16px", marginBottom: 14 }}>
                <p className="sis-label" style={{ marginBottom: 10 }}>Element Counts</p>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "6px 12px" }}>
                  {[
                    ["Rooms", elementCounts.rooms],
                    ["Slabs", elementCounts.slabs],
                    ["Walls", elementCounts.walls],
                    ["Doors", elementCounts.doors],
                    ["Windows", elementCounts.windows],
                    ["Openings", elementCounts.openings],
                    ["Nodes", elementCounts.graphNodes],
                    ["Columns", elementCounts.columns],
                    ["Issues", elementCounts.issueEntries],
                  ].map(([label, count]) => (
                    <div key={label as string} style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
                      <span style={{ fontSize: 13, fontWeight: 700, color: "var(--fg-primary)", fontFamily: "var(--font-mono)" }}>
                        {count}
                      </span>
                      <span style={{ fontSize: 11, color: "var(--fg-muted)" }}>{label}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Issue stat pills */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
                <div className="stat-pill" style={{ background: "var(--danger-bg)", borderColor: "var(--danger-border)" }}>
                  <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.16em", textTransform: "uppercase", color: "var(--danger-fg)" }}>Errors</span>
                  <span style={{ fontSize: 20, fontWeight: 700, color: "var(--danger-fg)", fontFamily: "var(--font-mono)" }}>{elementCounts.errors}</span>
                </div>
                <div className="stat-pill" style={{ background: "var(--warning-bg)", borderColor: "var(--warning-border)" }}>
                  <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.16em", textTransform: "uppercase", color: "var(--warning-fg)" }}>Warnings</span>
                  <span style={{ fontSize: 20, fontWeight: 700, color: "var(--warning-fg)", fontFamily: "var(--font-mono)" }}>{elementCounts.warnings}</span>
                </div>
                <div className="stat-pill" style={{ background: "var(--info-bg)", borderColor: "var(--info-border)" }}>
                  <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.16em", textTransform: "uppercase", color: "var(--info-fg)" }}>Info</span>
                  <span style={{ fontSize: 20, fontWeight: 700, color: "var(--info-fg)", fontFamily: "var(--font-mono)" }}>{elementCounts.infos}</span>
                </div>
              </div>

              {/* Issue log */}
              <div style={{ marginTop: 16 }}>
                <hr className="section-rule" style={{ marginBottom: 14 }} />
                <p className="sis-label" style={{ marginBottom: 10 }}>Issue Log</p>
                <div style={{ maxHeight: 220, overflowY: "auto", display: "flex", flexDirection: "column", gap: 6, paddingRight: 2 }}>
                  {issues.length === 0 ? (
                    <div className="issue-entry info" style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
                        <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                      No validation issues. Geometry is ready for demo.
                    </div>
                  ) : (
                    issues.map((issue) => {
                      const selection = selectionFromIssue(issue);
                      return (
                        <article
                          key={issue.id}
                          className={`issue-entry ${severityClass(issue.severity)} ${selection ? "clickable" : ""}`}
                          onClick={() => {
                            if (selection) {
                              handleIssueSelection(selection);
                            }
                          }}
                        >
                          <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 10 }}>
                            <p style={{ margin: 0, fontSize: 13, lineHeight: 1.55 }}>{issue.message}</p>
                            <span className={`badge badge-${severityClass(issue.severity) === "error" ? "danger" : severityClass(issue.severity) === "warning" ? "warning" : "info"}`} style={{ flexShrink: 0 }}>
                              {issue.severity}
                            </span>
                          </div>
                          {issue.fixType === "opening_clash" && (
                            <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
                              {pendingFixIssueKeys[issueKey(issue)] ? (
                                <>
                                  <button
                                    type="button"
                                    className="action-btn action-confirm"
                                    onClick={(event) => {
                                      event.stopPropagation();
                                      handleConfirmIssueFix(issue);
                                    }}
                                  >
                                    Confirm
                                  </button>
                                  <button
                                    type="button"
                                    className="action-btn action-revert"
                                    onClick={(event) => {
                                      event.stopPropagation();
                                      handleCancelPendingFix(issue);
                                    }}
                                  >
                                    Revert
                                  </button>
                                </>
                              ) : (
                                <button
                                  type="button"
                                  className="action-btn action-fix"
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    handleFixIssue(issue, selection);
                                  }}
                                >
                                  Fix
                                </button>
                              )}
                            </div>
                          )}
                        </article>
                      );
                    })
                  )}
                </div>
              </div>
            </section>

            {/* Element Inspector */}
            <section className="sis-surface" style={{ padding: "20px 22px" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, marginBottom: 16 }}>
                <h2 style={{ fontSize: 15, fontWeight: 700, color: "var(--fg-primary)", margin: 0, letterSpacing: "-0.01em" }}>
                  Element Inspector
                </h2>
                <span className="badge badge-neutral" style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}>
                  {selectedEntity ? `${selectedEntity.type} · ${selectedEntity.id}` : "No selection"}
                </span>
              </div>

              {selectedWall ? (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }} className="animate-enter">
                  {[
                    ["Type / ID", `wall / ${selectedWall.id}`],
                    ["Classification", classifyWall(selectedWall).replace("_", " ")],
                    ["Coordinates", `start (${formatPoint2D(selectedWall.start)})\nend (${formatPoint2D(selectedWall.end)})`],
                    ["Dimensions", `L ${selectedWall.length.toFixed(2)}m\nH ${selectedWall.height.toFixed(2)}m · T ${selectedWall.thickness.toFixed(2)}m`],
                    ["Span / Room", selectedWallContext?.room ? `${selectedWallContext.room.id} · span ${selectedWallContext.span.toFixed(2)}m` : "No room association"],
                    ["Connectivity", [
                      selectedWallNodeIds.length > 0 ? `nodes ${selectedWallNodeIds.join(", ")}` : "No linked nodes",
                      selectedWallOpeningIds.length > 0 ? `openings ${selectedWallOpeningIds.join(", ")}` : "No hosted openings",
                    ].join("\n")],
                  ].map(([label, value]) => (
                    <div key={label as string} className="data-cell">
                      <p className="data-cell-label">{label}</p>
                      <p className="data-cell-value" style={{ whiteSpace: "pre-line" }}>{value}</p>
                    </div>
                  ))}
                </div>
              ) : selectedNode ? (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }} className="animate-enter">
                  {[
                    ["Type / ID", `node / ${selectedNode.id}`],
                    ["Classification", selectedNode.likelyColumn ? "structural support node" : selectedNode.type],
                    ["Coordinates", formatPoint3D(selectedNode.position)],
                    ["Dimensions / Degree", `Anchor radius 0.11m · degree ${selectedNode.degree}`],
                    ["Connected Walls", selectedNode.connectedWallIds.join(", ") || "None"],
                    ["Load-Bearing Links", selectedNode.loadBearingWallIds.join(", ") || "None"],
                  ].map(([label, value]) => (
                    <div key={label as string} className="data-cell">
                      <p className="data-cell-label">{label}</p>
                      <p className="data-cell-value">{value}</p>
                    </div>
                  ))}
                </div>
              ) : selectedColumn ? (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }} className="animate-enter">
                  {[
                    ["Type / ID", `column / ${selectedColumn.id}`],
                    ["Classification", "vertical structural support"],
                    ["Coordinates", formatPoint3D(selectedColumn.position)],
                    ["Dimensions", `${selectedColumn.width.toFixed(2)}m × ${selectedColumn.depth.toFixed(2)}m\nheight ${selectedColumn.height.toFixed(2)}m`],
                    ["Source Node", selectedColumn.nodeId],
                    ["Connected Walls", selectedColumn.connectedWallIds.join(", ") || "None"],
                  ].map(([label, value]) => (
                    <div key={label as string} className="data-cell">
                      <p className="data-cell-label">{label}</p>
                      <p className="data-cell-value" style={{ whiteSpace: "pre-line" }}>{value}</p>
                    </div>
                  ))}
                </div>
              ) : selectedSlab ? (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }} className="animate-enter">
                  {[
                    ["Type / ID", `slab / ${selectedSlab.id}`],
                    ["Classification", "horizontal structural surface"],
                    ["Coordinates", `centroid ${formatPoint3D(selectedSlab.centroid)}`],
                    ["Dimensions", `area ${selectedSlab.area.toFixed(2)}m²\nspan ${selectedSlab.span.toFixed(2)}m · t ${selectedSlabThickness.toFixed(2)}m`],
                    ["Loop / Geometry", `${selectedSlab.polygon2D.length} vertices\nclosed loop ${selectedSlab.polygon2D.length >= 3 ? "yes" : "no"}`],
                    ["Connectivity", [
                      selectedSlabWallIds.length > 0 ? `walls ${selectedSlabWallIds.join(", ")}` : "No edge-matched walls",
                      selectedSlabNodeIds.length > 0 ? `nodes ${selectedSlabNodeIds.join(", ")}` : "No matched nodes",
                    ].join("\n")],
                  ].map(([label, value]) => (
                    <div key={label as string} className="data-cell">
                      <p className="data-cell-label">{label}</p>
                      <p className="data-cell-value" style={{ whiteSpace: "pre-line" }}>{value}</p>
                    </div>
                  ))}
                </div>
              ) : selectedOpening ? (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }} className="animate-enter">
                  {[
                    ["Type / ID", `${selectedOpening.kind} / ${selectedOpening.id}`],
                    ["Classification", selectedOpening.kind === "door" ? "structural opening" : `${selectedOpening.panelType ?? "double"} facade window`],
                    ["Coordinates", formatPoint3D(selectedOpening.position)],
                    ["Dimensions", [
                      `W ${selectedOpening.width.toFixed(2)}m · H ${selectedOpening.height.toFixed(2)}m`,
                      `offset ${selectedOpening.offset.toFixed(2)}m`,
                      selectedOpening.kind === "window" ? `type ${selectedOpening.panelType ?? "double"}` : "",
                    ].filter(Boolean).join("\n")],
                    ["Host Wall", selectedOpening.wallId],
                    ["Room Association", scene.walls.find((wall) => wall.id === selectedOpening.wallId)?.roomId ?? "Unknown"],
                  ].map(([label, value]) => (
                    <div key={label as string} className="data-cell">
                      <p className="data-cell-label">{label}</p>
                      <p className="data-cell-value" style={{ whiteSpace: "pre-line" }}>{value}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="sis-surface-sunken" style={{ padding: "16px", textAlign: "center" }}>
                  <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24" style={{ margin: "0 auto 8px", color: "var(--fg-faint)", display: "block" }}>
                    <path d="M15 15l5.196 5.196M10.5 19a8.5 8.5 0 100-17 8.5 8.5 0 000 17z" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  <p style={{ fontSize: 13, color: "var(--fg-muted)", lineHeight: 1.6 }}>
                    Upload a plan and select a wall, node, column, slab, door, or window in the viewer.
                  </p>
                </div>
              )}
            </section>

          </aside>
        </div>

        {/* ── MATERIAL PANEL ── */}
        <MaterialPanel
          selectedWall={selectedWall}
          recommendationTable={recommendationTable}
          loading={recommendationLoading}
          darkMode={darkMode}
        />

        {/* ── LAYOUT OPTIMISATION ENGINE ── */}
        <section className="sis-surface" style={{ padding: "20px 22px" }}>
          <div className="calm-section-header">
            <div>
              <h2 className="calm-section-title">Layout Optimisation</h2>
              <p className="calm-section-subtitle">Heuristic-driven structural improvement suggestions.</p>
              <p style={{ marginTop: 6, marginBottom: 0, fontSize: 12, lineHeight: 1.6, color: "var(--fg-muted)" }}>
                `Severity` = impact if not fixed. `Confidence` = certainty of the suggestion.
                High severity + low confidence means "important but verify first".
                Low severity + high confidence means "safe, reliable polish".
              </p>
            </div>
            {optimizationActions.length > 0 && (
              <span className="badge badge-neutral" style={{ marginTop: 2 }}>{optimizationActions.length} suggestion{optimizationActions.length !== 1 ? "s" : ""}</span>
            )}
          </div>

          <div className="analysis-grid">
            {optimizationActions.length === 0 ? (
              <div className="opt-card" style={{ gridColumn: "1 / -1" }}>
                <p style={{ fontSize: 13, color: "var(--fg-muted)", lineHeight: 1.6, margin: 0 }}>
                  No structural inefficiency suggestions were generated for this scene.
                </p>
              </div>
            ) : (
              optimizationActions.map((entry) => (
                <article key={entry.id} className="opt-card">
                  <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 5, marginBottom: 8 }}>
                    <span className="sis-eyebrow">{entry.kind === "advisory" ? "advisory" : formatHeuristicType(entry.kind)}</span>
                    <span className={`badge ${heuristicSeverityBadge(entry.severity)}`} title={heuristicSeverityMeaning(entry.severity)}>
                      Severity: {entry.severity}
                    </span>
                    <span className={`badge ${heuristicConfidenceBadge(entry.confidence)}`} title={heuristicConfidenceMeaning(entry.confidence)}>
                      Confidence: {entry.confidence}
                    </span>
                    {entry.state === "blocked" && <span className="badge badge-warning">blocked</span>}
                  </div>
                  <p style={{ fontSize: 13, fontWeight: 600, color: "var(--fg-primary)", margin: "0 0 5px", letterSpacing: "-0.01em" }}>{entry.title}</p>
                  <p style={{ fontSize: 12.5, color: "var(--fg-secondary)", lineHeight: 1.6, margin: "0 0 5px" }}>{entry.issue}</p>
                  <p style={{ fontSize: 12, color: "var(--info-fg)", lineHeight: 1.55, margin: "0 0 6px" }}>
                    Interpretation: {optimizationInterpretation(entry.severity, entry.confidence)}
                  </p>
                  <p style={{ fontSize: 12.5, color: "var(--fg-secondary)", lineHeight: 1.6, margin: "0 0 3px" }}>
                    <span style={{ fontWeight: 600, color: "var(--fg-primary)" }}>Action:</span> {entry.suggestion}
                  </p>
                  <p style={{ fontSize: 12, color: "var(--fg-muted)", lineHeight: 1.5, margin: "0 0 2px" }}>
                    <span style={{ fontWeight: 600 }}>Impact:</span> {entry.impact}
                  </p>
                  <p style={{ fontSize: 11.5, color: "var(--fg-muted)", lineHeight: 1.5, margin: "0 0 2px" }}>Basis: {entry.impactBasis}</p>
                  {entry.evidence.length > 0 && (
                    <p style={{ fontSize: 11, color: "var(--fg-muted)", lineHeight: 1.5, margin: "0 0 2px" }}>
                      Evidence: {entry.evidence.join(" ")}
                    </p>
                  )}
                  {entry.assumptions.length > 0 && (
                    <p style={{ fontSize: 11, color: "var(--warning-fg)", lineHeight: 1.5, margin: 0 }}>
                      Assumptions: {entry.assumptions.join(" ")}
                    </p>
                  )}

                  {/* Action buttons */}
                  <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
                    {!entry.safeToApply ? (
                      <span className="action-btn action-advisory">Advisory only</span>
                    ) : entry.state === "pending" ? (
                      <>
                        <button
                          type="button"
                          className="action-btn action-confirm"
                          onClick={() => handleConfirmOptimizationAction(entry)}
                        >
                          Confirm
                        </button>
                        <button
                          type="button"
                          className="action-btn action-revert"
                          onClick={() => handleRevertOptimizationAction(entry)}
                        >
                          Revert
                        </button>
                      </>
                    ) : (
                      <button
                        type="button"
                        disabled={entry.state === "blocked"}
                        className={`action-btn ${entry.state === "blocked" ? "action-advisory" : "action-fix"}`}
                        style={entry.state === "blocked" ? { cursor: "not-allowed", opacity: 0.6 } : {}}
                        onClick={() => handleFixOptimizationAction(entry)}
                      >
                        Fix
                      </button>
                    )}
                  </div>
                </article>
              ))
            )}
          </div>
        </section>

      </div>
    </main>
  );
}
