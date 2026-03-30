"use client";

import { classifyWall } from "@/lib/materialEngine";
import type { MaterialRecommendationTable, NormalizedWall } from "@/lib/types";

type MaterialPanelProps = {
  selectedWall: NormalizedWall | null;
  recommendationTable: MaterialRecommendationTable | null;
  loading: boolean;
  darkMode?: boolean;
};

function formatElementType(value: string) {
  return value.replace(/_/g, " ");
}

export default function MaterialPanel({
  selectedWall,
  recommendationTable,
  loading,
  darkMode = false,
}: MaterialPanelProps) {
  return (
    <section
      className={`sis-surface${darkMode ? " theme-dark" : ""}`}
      style={{ padding: "22px 24px" }}
    >
      {/* Header */}
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

      {/* Body states */}
      {loading ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <p style={{ fontSize: 13, color: "var(--fg-muted)", marginBottom: 12 }}>
            Generating recommendation table…
          </p>
          {[140, 100, 120].map((w, i) => (
            <div key={i} className="shimmer" style={{ height: 14, width: `${w}px` }} />
          ))}
        </div>
      ) : !recommendationTable ? (
        <div className="sis-surface-sunken" style={{ padding: "20px", textAlign: "center" }}>
          <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24" style={{ margin: "0 auto 8px", color: "var(--fg-faint)", display: "block" }}>
            <path d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <p style={{ fontSize: 13, color: "var(--fg-muted)", lineHeight: 1.6 }}>
            Upload a model and click an element to generate recommendation tables.
          </p>
        </div>
      ) : recommendationTable.rows.length === 0 ? (
        <p style={{ fontSize: 13, color: "var(--fg-secondary)", lineHeight: 1.6 }}>
          {recommendationTable.focus.sizeSummary}
        </p>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <p style={{ fontSize: 13, color: "var(--fg-secondary)", lineHeight: 1.6 }}>
            {recommendationTable.focus.sizeSummary}
          </p>

          {recommendationTable.rows.map((row) => (
            <article
              key={row.elementType}
              className="sis-surface-sunken"
              style={{ overflow: "hidden", borderRadius: "var(--radius-lg)", border: "1px solid var(--border-default)" }}
            >
              {/* Row header */}
              <div style={{
                padding: "14px 18px",
                borderBottom: "1px solid var(--border-subtle)",
                background: "var(--bg-sunken)",
              }}>
                <p style={{ fontSize: 12, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.12em", color: "var(--fg-primary)", margin: "0 0 6px" }}>
                  {formatElementType(row.elementType)}
                </p>
                <p style={{ fontSize: 13, color: "var(--fg-secondary)", margin: "0 0 4px", lineHeight: 1.5 }}>
                  {row.weightJustification}
                </p>
                <p className="sis-mono" style={{ color: "var(--fg-muted)", margin: "0 0 4px" }}>
                  Formula: {row.formula}
                </p>
                <p className="sis-mono" style={{ color: "var(--warning-fg)", margin: 0 }}>
                  ⚠ {row.structuralConcerns.join(" ")}
                </p>
              </div>

              {/* Table */}
              <div style={{ overflowX: "auto" }}>
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
                      <tr
                        key={`${row.elementType}-${option.material}`}
                        className={index === 0 ? "row-highlight" : ""}
                      >
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
  );
}
