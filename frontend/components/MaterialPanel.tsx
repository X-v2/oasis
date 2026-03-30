"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { classifyWall } from "@/lib/materialEngine";
import type {
  CostReport,
  CostReportElementType,
  CostReportOptionsResponse,
  CostScenarioOption,
  MaterialRecommendationTable,
  NormalizedWall,
  SceneData,
} from "@/lib/types";

declare global {
  interface Window {
    __sisCaptureModelScreenshots?: () => Promise<Array<{ title: string; dataUri: string }>>;
  }
}

type MaterialPanelProps = {
  selectedWall: NormalizedWall | null;
  recommendationTable: MaterialRecommendationTable | null;
  loading: boolean;
  scene: SceneData;
  darkMode?: boolean;
  readOnlyMode?: boolean;
  onFocusElement?: (elementId: string, elementType: CostReportElementType) => void;
};

type MaterialChoice = {
  material: string;
  unit: string;
  rate: number;
  strength: number;
  durability: number;
  rationale: string;
};

function formatElementType(value: string) {
  return value.replace(/_/g, " ");
}

function formatMoney(value: number, currency: string) {
  const symbol = currency.toUpperCase() === "INR" ? "\u20B9" : currency;
  return `${symbol} ${value.toLocaleString("en-IN", { maximumFractionDigits: 2 })}`;
}

export default function MaterialPanel({
  selectedWall,
  recommendationTable,
  loading,
  scene,
  darkMode = false,
  readOnlyMode = false,
}: MaterialPanelProps) {
  const [isHydrated, setIsHydrated] = useState(false);
  const [scenarioOptions, setScenarioOptions] = useState<CostScenarioOption[]>([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>("standard");
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState<string | null>(null);
  const [analysisSaving, setAnalysisSaving] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [analysisStatus, setAnalysisStatus] = useState<string | null>(null);
  const [wastageOverride, setWastageOverride] = useState("0");
  const [includeBeamCandidates, setIncludeBeamCandidates] = useState(true);
  const [materialOverrides, setMaterialOverrides] = useState<Record<string, string>>({});
  const [pickerItemId, setPickerItemId] = useState<string | null>(null);
  const [pdfPreviewHtml, setPdfPreviewHtml] = useState<string | null>(null);
  const [pdfPreviewOpen, setPdfPreviewOpen] = useState(false);
  const previewFrameRef = useRef<HTMLIFrameElement | null>(null);

  useEffect(() => {
    setIsHydrated(true);
  }, []);

  const hasScene = scene.walls.length > 0 || scene.rooms.length > 0;
  const hasSceneAfterHydration = isHydrated && hasScene;
  const parsedWastage = useMemo(() => {
    const parsed = Number(wastageOverride);
    return Number.isFinite(parsed) ? parsed : 0;
  }, [wastageOverride]);
  const selectedReport: CostReport | null = useMemo(
    () => scenarioOptions.find((option) => option.id === selectedScenarioId)?.report ?? scenarioOptions[0]?.report ?? null,
    [scenarioOptions, selectedScenarioId],
  );
  const pickerItem = useMemo(
    () => selectedReport?.items.find((item) => item.elementId === pickerItemId) ?? null,
    [pickerItemId, selectedReport],
  );

  const pickerOptions = useMemo<MaterialChoice[]>(() => {
    if (!pickerItem || !selectedReport) {
      return [];
    }

    const byMaterial = new Map<string, MaterialChoice>();
    const rowMatches =
      recommendationTable?.rows.filter(
        (row) =>
          row.elementType === pickerItem.elementType ||
          (pickerItem.elementType === "beam" && (row.elementType === "slab" || row.elementType === "column")),
      ) ?? [];
    const sourceRows = rowMatches.length > 0 ? rowMatches : recommendationTable?.rows ?? [];

    sourceRows.forEach((row) => {
      row.options.forEach((option) => {
        const catalog = selectedReport.materialCatalog[option.material];
        const inferredRate = Math.round((900 + option.cost * 2200) * 100) / 100;
        byMaterial.set(option.material, {
          material: option.material,
          unit: catalog?.unit ?? "m3",
          rate: catalog?.rate ?? inferredRate,
          strength: option.strength,
          durability: option.durability,
          rationale: option.rationale,
        });
      });
    });

    if (!byMaterial.has(pickerItem.material)) {
      const catalog = selectedReport.materialCatalog[pickerItem.material];
      byMaterial.set(pickerItem.material, {
        material: pickerItem.material,
        unit: catalog?.unit ?? pickerItem.unit,
        rate: catalog?.rate ?? pickerItem.unitRate,
        strength: catalog?.strength ?? 0,
        durability: catalog?.durability ?? 0,
        rationale: "Current selected material in this report.",
      });
    }

    return Array.from(byMaterial.values()).sort((a, b) => a.rate - b.rate);
  }, [pickerItem, recommendationTable, selectedReport]);

  async function generateCostReport(overrides: Record<string, string> = materialOverrides) {
    if (readOnlyMode) {
      setReportError("Read-only workspace: report generation is disabled.");
      return;
    }
    if (!hasScene) {
      return;
    }
    setReportLoading(true);
    setReportError(null);
    try {
      const response = await fetch("/api/cost-report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scene,
          projectName: "SIS Structural Estimate",
          currency: "INR",
          wastageOverridePercent: parsedWastage,
          includeBeamCandidates,
          includeScenarioOptions: true,
          elementMaterialOverrides: overrides,
        }),
      });

      if (!response.ok) {
        throw new Error("Cost report request failed");
      }
      const payload = (await response.json()) as CostReportOptionsResponse;
      setScenarioOptions(payload.options);
      setSelectedScenarioId(payload.options.find((entry) => entry.id === "standard")?.id ?? payload.options[0]?.id ?? "standard");
    } catch {
      setReportError("Could not generate the material report. Please try again.");
      setScenarioOptions([]);
    } finally {
      setReportLoading(false);
    }
  }

  async function exportCostReportPdf() {
    if (readOnlyMode) {
      setReportError("Read-only workspace: export is disabled.");
      return;
    }
    if (!hasScene) {
      return;
    }
    setExporting(true);
    setReportError(null);
    try {
      await saveAnalysisToBackend({ source: "export" });
      const modelScreenshots = window.__sisCaptureModelScreenshots ? await window.__sisCaptureModelScreenshots() : [];
      const response = await fetch("/api/cost-report/pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scene,
          projectName: "SIS Structural Estimate",
          currency: "INR",
          wastageOverridePercent: parsedWastage,
          includeBeamCandidates,
          scenarioId: selectedScenarioId,
          modelScreenshots,
          elementMaterialOverrides: materialOverrides,
        }),
      });

      if (!response.ok) {
        throw new Error("PDF preview request failed");
      }

      const html = await response.text();
      setPdfPreviewHtml(html);
      setPdfPreviewOpen(true);
    } catch {
      setReportError("Could not create PDF preview.");
    } finally {
      setExporting(false);
    }
  }

  async function saveAnalysisToBackend(options?: { source?: "manual" | "export" }) {
    if (readOnlyMode) {
      setAnalysisStatus("Read-only workspace: saving analysis is disabled.");
      return false;
    }
    const source = options?.source ?? "manual";
    if (!selectedReport) {
      setAnalysisStatus("Generate a material report before saving analysis.");
      return false;
    }

    setAnalysisSaving(true);
    setAnalysisStatus(null);

    try {
      const payload = {
        totalCost: selectedReport.summary.totalCost,
        totalArea: selectedReport.summary.totalArea ?? 0,
        costPerM2: selectedReport.summary.costPerSqm ?? 0,
        lineItems: selectedReport.items.map((item) => ({
          itemId: item.elementId,
          elementType: item.elementType,
          material: item.material,
          quantity: item.quantity,
          unit: item.unit,
          unitRate: item.unitRate,
          subtotal: item.subtotal,
          justification: item.justification,
        })),
        modelJson: scene,
      };

      const response = await fetch("/api/analyses", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const result = (await response.json()) as { analysisId?: string; error?: string };
      if (!response.ok) {
        throw new Error(result.error || "Failed to save analysis.");
      }
      setAnalysisStatus(
        source === "export"
          ? `Auto-saved analysis. ID: ${result.analysisId ?? "created"}`
          : `Analysis saved successfully. ID: ${result.analysisId ?? "created"}`,
      );
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to save analysis.";
      setAnalysisStatus(message);
      return false;
    } finally {
      setAnalysisSaving(false);
    }
  }

  function downloadPdfFromPreview() {
    const frameWindow = previewFrameRef.current?.contentWindow;
    if (!frameWindow) {
      setReportError("Preview not ready. Please reopen PDF preview and try again.");
      return;
    }
    frameWindow.focus();
    frameWindow.print();
  }

  async function applyMaterialOverride(elementId: string, material: string) {
    if (readOnlyMode) {
      setReportError("Read-only workspace: material override is disabled.");
      return;
    }
    const next = { ...materialOverrides, [elementId]: material };
    setMaterialOverrides(next);
    await generateCostReport(next);
    setPickerItemId(null);
  }

  return (
    <div className="material-panel-stack">
      <section className={`sis-surface material-panel-section${darkMode ? " theme-dark" : ""}`}>
        <div className="calm-section-header">
          <div>
            <h2 className="calm-section-title">Material Analysis</h2>
            <p className="calm-section-subtitle">Element-specific cost-strength tradeoff with ranked options.</p>
          </div>
          {selectedWall && (
            <span className="badge badge-neutral" style={{ fontFamily: "var(--font-mono)", marginTop: 2 }}>
              {classifyWall(selectedWall).replace("_", " ")}
            </span>
          )}
        </div>

        {loading ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <p style={{ fontSize: 13, color: "var(--fg-muted)", marginBottom: 12 }}>Generating recommendation table...</p>
            {[140, 100, 120].map((w, i) => (
              <div key={i} className="shimmer" style={{ height: 14, width: `${w}px` }} />
            ))}
          </div>
        ) : !recommendationTable ? (
          <div className="sis-surface-sunken" style={{ padding: "20px", textAlign: "center" }}>
            <p style={{ fontSize: 13, color: "var(--fg-muted)", lineHeight: 1.6 }}>
              Upload a model and click an element to generate recommendation tables.
            </p>
          </div>
        ) : recommendationTable.rows.length === 0 ? (
          <p style={{ fontSize: 13, color: "var(--fg-secondary)", lineHeight: 1.6 }}>{recommendationTable.focus.sizeSummary}</p>
        ) : (
          <div className="material-analysis-group">
            <p style={{ fontSize: 13, color: "var(--fg-secondary)", lineHeight: 1.6 }}>{recommendationTable.focus.sizeSummary}</p>
            {recommendationTable.rows.map((row) => (
              <article
                key={row.elementType}
                className="sis-surface-sunken"
                style={{ overflow: "hidden", borderRadius: "var(--radius-lg)", border: "1px solid var(--border-default)" }}
              >
                <div
                  style={{
                    padding: "14px 18px",
                    borderBottom: "1px solid var(--border-subtle)",
                    background: "var(--bg-sunken)",
                  }}
                >
                  <p
                    style={{
                      fontSize: 12,
                      fontWeight: 700,
                      textTransform: "uppercase",
                      letterSpacing: "0.12em",
                      color: "var(--fg-primary)",
                      margin: "0 0 6px",
                    }}
                  >
                    {formatElementType(row.elementType)}
                  </p>
                  <p style={{ fontSize: 13, color: "var(--fg-secondary)", margin: "0 0 4px", lineHeight: 1.5 }}>
                    {row.weightJustification}
                  </p>
                  <p className="sis-mono" style={{ color: "var(--fg-muted)", margin: "0 0 4px" }}>
                    Formula: {row.formula}
                  </p>
                  <p className="sis-mono" style={{ color: "var(--warning-fg)", margin: 0 }}>
                    Concern: {row.structuralConcerns.join(" ")}
                  </p>
                </div>
                <div className="table-scroll">
                  <table className="sis-table" style={{ minWidth: 900 }}>
                    <thead>
                      <tr>
                        <th>Material</th>
                        <th>Tradeoff Score</th>
                        <th>Cost</th>
                        <th>Strength</th>
                        <th>Durability</th>
                        <th>Ease</th>
                        <th>Reasoning</th>
                      </tr>
                    </thead>
                    <tbody>
                      {row.options.map((option, index) => (
                        <tr key={`${row.elementType}-${option.material}`} className={index === 0 ? "row-highlight" : ""}>
                          <td style={{ fontWeight: 600 }}>{option.material}</td>
                          <td style={{ fontWeight: 700, fontFamily: "var(--font-mono)" }}>{option.tradeoffScore.toFixed(2)}</td>
                          <td style={{ fontFamily: "var(--font-mono)" }}>{option.cost.toFixed(1)}</td>
                          <td style={{ fontFamily: "var(--font-mono)" }}>{option.strength.toFixed(1)}</td>
                          <td style={{ fontFamily: "var(--font-mono)" }}>{option.durability.toFixed(1)}</td>
                          <td style={{ fontFamily: "var(--font-mono)" }}>{option.ease.toFixed(1)}</td>
                          <td style={{ lineHeight: 1.55, maxWidth: 280 }}>{option.rationale}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>

      <section className={`sis-surface material-panel-section${darkMode ? " theme-dark" : ""}`}>
        <div className="calm-section-header">
          <div>
            <h2 className="calm-section-title">Material Report (Whole Model)</h2>
            <p className="calm-section-subtitle">Complete model costing with scenario options and deterministic quantities.</p>
          </div>
        </div>

        <div className="material-report-toolbar">
          <div className="material-report-fields">
            <div className="material-control">
              <label className="sis-label" htmlFor="wastage-input" style={{ margin: 0 }}>
                Optional Wastage Override (%)
              </label>
              <input
                id="wastage-input"
                type="number"
                min={0}
                max={25}
                step={0.5}
                value={wastageOverride}
                onChange={(event) => setWastageOverride(event.target.value)}
                className="material-number-input"
                disabled={readOnlyMode}
              />
            </div>
            <label className="material-checkbox">
              <input
                type="checkbox"
                checked={includeBeamCandidates}
                onChange={(event) => setIncludeBeamCandidates(event.target.checked)}
                disabled={readOnlyMode}
              />
              Include beam candidates
            </label>
          </div>
          <div className="material-report-actions">
            <button
              type="button"
              className="btn btn-primary material-primary-cta"
              onClick={() => generateCostReport()}
              disabled={readOnlyMode || reportLoading || !hasSceneAfterHydration}
            >
              {reportLoading ? (
                <>
                  <span className="btn-spinner" aria-hidden="true" />
                  Generating...
                </>
              ) : (
                "Generate Material Report"
              )}
            </button>
            <button
              type="button"
              className="btn btn-default material-primary-cta"
              onClick={exportCostReportPdf}
              disabled={readOnlyMode || !hasSceneAfterHydration || !selectedReport || analysisSaving || exporting}
              aria-busy={analysisSaving || exporting}
            >
              {analysisSaving || exporting ? (
                <>
                  <span className="btn-spinner" aria-hidden="true" />
                  Saving & Exporting...
                </>
              ) : (
                "Save & Export"
              )}
            </button>
          </div>
        </div>

        {readOnlyMode && (
          <div className="issue-entry info material-status">
            Read-only workspace mode is active. Report mutation actions are disabled.
          </div>
        )}

        {reportError && (
          <div className="issue-entry error material-status">
            {reportError}
          </div>
        )}
        {analysisStatus && (
          <div className="issue-entry material-status">
            {analysisStatus}
          </div>
        )}

        {!selectedReport ? (
          <p style={{ fontSize: 13, color: "var(--fg-muted)" }}>
            Generate report to preview complete-model costing options and line items.
          </p>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div className="scenario-grid">
              {scenarioOptions.map((option) => {
                const isActive = option.id === selectedScenarioId;
                return (
                  <button
                    key={option.id}
                    type="button"
                    onClick={() => setSelectedScenarioId(option.id)}
                    className={`${isActive ? "btn btn-active" : "btn btn-default"} scenario-card`}
                  >
                    <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 2 }}>{option.label}</div>
                    <div style={{ fontSize: 11, color: "var(--fg-secondary)", marginBottom: 4 }}>{option.description}</div>
                    <div style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--fg-primary)" }}>
                      {formatMoney(option.report.summary.totalCost, option.report.currency)}
                    </div>
                  </button>
                );
              })}
            </div>

            <div className="material-overview-grid">
              <div className="data-cell">
                <p className="data-cell-label">Project</p>
                <p className="data-cell-value">{selectedReport.projectName}</p>
              </div>
              <div className="data-cell">
                <p className="data-cell-label">Total Cost</p>
                <p className="data-cell-value">{formatMoney(selectedReport.summary.totalCost, selectedReport.currency)}</p>
              </div>
              <div className="data-cell">
                <p className="data-cell-label">Built-up Area</p>
                <p className="data-cell-value">{(selectedReport.summary.totalArea ?? 0).toFixed(2)} m2</p>
              </div>
              <div className="data-cell">
                <p className="data-cell-label">Cost per m2</p>
                <p className="data-cell-value">{formatMoney(selectedReport.summary.costPerSqm ?? 0, selectedReport.currency)}</p>
              </div>
            </div>

            <div
              className="table-scroll material-table-wrap"
              style={{
                overflowY: selectedReport.items.length > 5 ? "auto" : "visible",
                maxHeight: selectedReport.items.length > 5 ? 316 : undefined,
              }}
            >
              <table className="sis-table" style={{ minWidth: 980 }}>
                <thead>
                  <tr>
                    <th>Element ID</th>
                    <th>Element Type</th>
                    <th>Material</th>
                    <th>Qty</th>
                    <th>Unit</th>
                    <th>Unit Rate</th>
                    <th>Base</th>
                    <th>Wastage</th>
                    <th>Subtotal</th>
                    <th>Justification</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedReport.items.map((item) => (
                    <tr key={`${item.elementType}-${item.elementId}`}>
                      <td style={{ maxWidth: 160, overflowWrap: "anywhere" }}>{item.elementId}</td>
                      <td>{formatElementType(item.elementType)}</td>
                      <td>
                        <button
                          type="button"
                          className="btn btn-default material-cell-button"
                          style={{ padding: "4px 8px", fontSize: 11, maxWidth: 180 }}
                          title={item.material}
                          disabled={readOnlyMode}
                          onClick={() => setPickerItemId(item.elementId)}
                        >
                          {item.material}
                        </button>
                      </td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{item.quantity.toFixed(3)}</td>
                      <td>{item.unit}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{formatMoney(item.unitRate, selectedReport.currency)}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{formatMoney(item.baseCost, selectedReport.currency)}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{(item.wastageFactor * 100).toFixed(1)}%</td>
                      <td style={{ fontFamily: "var(--font-mono)", fontWeight: 700 }}>
                        {formatMoney(item.subtotal, selectedReport.currency)}
                      </td>
                      <td style={{ maxWidth: 280, lineHeight: 1.5, overflowWrap: "anywhere", whiteSpace: "normal" }}>{item.justification}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="material-summary-grid">
              <div className="sis-surface-sunken" style={{ padding: 12 }}>
                <p className="sis-label" style={{ marginBottom: 8 }}>
                  Material Summary
                </p>
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {Object.entries(selectedReport.materialTotals).map(([material, total]) => (
                    <div key={material} style={{ display: "flex", justifyContent: "space-between", gap: 8, fontSize: 12 }}>
                      <span>{material}</span>
                      <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}>
                        {formatMoney(total, selectedReport.currency)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="sis-surface-sunken" style={{ padding: 12 }}>
                <p className="sis-label" style={{ marginBottom: 8 }}>
                  Assumptions
                </p>
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {selectedReport.assumptions.map((entry, index) => (
                    <p key={`assumption-${index}`} style={{ margin: 0, fontSize: 12, lineHeight: 1.5, color: "var(--fg-secondary)" }}>
                      {entry}
                    </p>
                  ))}
                </div>
              </div>

              <div className="sis-surface-sunken" style={{ padding: 12 }}>
                <p className="sis-label" style={{ marginBottom: 8 }}>
                  Notes and Warnings
                </p>
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {(selectedReport.warnings.length ? selectedReport.warnings : ["No critical warnings."]).map((entry, index) => (
                    <p
                      key={`warning-${index}`}
                      style={{
                        margin: 0,
                        fontSize: 12,
                        lineHeight: 1.5,
                        color: selectedReport.warnings.length ? "var(--warning-fg)" : "var(--fg-secondary)",
                      }}
                    >
                      {entry}
                    </p>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </section>

      {pickerItem && selectedReport && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.45)",
            zIndex: 60,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: "clamp(10px, 2.5vw, 20px)",
          }}
          onClick={() => setPickerItemId(null)}
        >
          <div
            className={`sis-surface material-picker-modal${darkMode ? " theme-dark" : ""}`}
            style={{ width: "min(880px, 100%)", maxHeight: "82vh", overflowY: "auto", padding: 16 }}
            onClick={(event) => event.stopPropagation()}
          >
            <div className="calm-section-header">
              <div>
                <h2 className="calm-section-title">Material Options for {pickerItem.elementId}</h2>
                <p className="calm-section-subtitle">Rate, strength, and durability from eligible products.</p>
              </div>
              <button type="button" className="btn btn-default" onClick={() => setPickerItemId(null)}>
                Close
              </button>
            </div>
            <div className="table-scroll">
              <table className="sis-table" style={{ minWidth: 780 }}>
                <thead>
                  <tr>
                    <th>Material</th>
                    <th>Rate</th>
                    <th>Unit</th>
                    <th>Strength</th>
                    <th>Durability</th>
                    <th>Why</th>
                    <th>Apply</th>
                  </tr>
                </thead>
                <tbody>
                  {pickerOptions.map((option) => (
                    <tr key={`${pickerItem.elementId}-${option.material}`}>
                      <td>{option.material}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{formatMoney(option.rate, selectedReport.currency)}</td>
                      <td>{option.unit}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{option.strength.toFixed(1)}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{option.durability.toFixed(1)}</td>
                      <td style={{ maxWidth: 260, lineHeight: 1.45 }}>{option.rationale}</td>
                      <td>
                        <button
                          type="button"
                          className="btn btn-primary"
                          style={{ padding: "4px 8px", fontSize: 11 }}
                          disabled={readOnlyMode || reportLoading}
                          onClick={() => applyMaterialOverride(pickerItem.elementId, option.material)}
                        >
                          Select
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {pdfPreviewOpen && pdfPreviewHtml && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.45)",
            zIndex: 70,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: "clamp(10px, 2.5vw, 20px)",
          }}
          onClick={() => setPdfPreviewOpen(false)}
        >
          <div
            className={`sis-surface material-picker-modal${darkMode ? " theme-dark" : ""}`}
            style={{
              width: "min(1200px, 96vw)",
              height: "min(860px, 92vh)",
              display: "flex",
              flexDirection: "column",
              gap: 10,
              padding: 14,
            }}
            onClick={(event) => event.stopPropagation()}
          >
            <div className="calm-section-header" style={{ marginBottom: 4 }}>
              <div>
                <h2 className="calm-section-title">PDF Preview</h2>
                <p className="calm-section-subtitle">Review report layout before downloading.</p>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <button type="button" className="btn btn-primary" onClick={downloadPdfFromPreview}>
                  Download PDF
                </button>
                <button type="button" className="btn btn-default" onClick={() => setPdfPreviewOpen(false)}>
                  Close
                </button>
              </div>
            </div>
            <div className="sis-surface-sunken" style={{ flex: 1, padding: 8, overflow: "hidden" }}>
              <iframe
                ref={previewFrameRef}
                title="Cost Report PDF Preview"
                srcDoc={pdfPreviewHtml}
                style={{ width: "100%", height: "100%", border: "1px solid var(--border-subtle)", borderRadius: 10, background: "#fff" }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

