"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { useTheme } from "@/components/theme/useTheme";

type AnalysisLineItem = {
  itemId?: string | null;
  elementType: string;
  material: string;
  quantity: number;
  unit: string;
  unitRate: number;
  subtotal?: number | null;
  justification?: string;
};

type AnalysisRecord = {
  analysisId: string;
  createdAt?: string;
  totalCost?: number;
  totalArea?: number;
  costPerM2?: number;
  verificationStatus?: string;
  status?: string;
  lastVerifiedAt?: string;
  lineItems?: AnalysisLineItem[];
  modelJson?: unknown;
  dataHash?: string;
};

type VerifyResponse = {
  analysisId: string;
  valid?: boolean;
  status?: string;
  lastVerifiedAt?: string;
  message?: string;
  dbHash?: string;
  chainHash?: string;
};

function normalizeAnalyses(payload: unknown): AnalysisRecord[] {
  if (Array.isArray(payload)) {
    return payload as AnalysisRecord[];
  }
  if (!payload || typeof payload !== "object") {
    return [];
  }

  const maybeAnalyses = (payload as { analyses?: unknown }).analyses;
  if (Array.isArray(maybeAnalyses)) {
    return maybeAnalyses as AnalysisRecord[];
  }

  return [];
}

function formatMoney(value: number | undefined) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "N/A";
  }
  return `? ${value.toLocaleString("en-IN", { maximumFractionDigits: 2 })}`;
}

function analysisStatus(analysis: AnalysisRecord) {
  return (analysis.verificationStatus ?? analysis.status ?? "unverified").toLowerCase();
}

function statusLabel(value: string) {
  if (value === "verified") return "Verified";
  if (value === "tampered") return "Tampered";
  if (value === "error") return "Error";
  return "Unverified";
}

export default function AnalysisClient() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [analyses, setAnalyses] = useState<AnalysisRecord[]>([]);
  const [verifyingMap, setVerifyingMap] = useState<Record<string, boolean>>({});
  const [verifyErrorMap, setVerifyErrorMap] = useState<Record<string, string>>({});
  const { isDark: darkMode, toggleTheme } = useTheme();

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch("/api/analyses", { cache: "no-store" });
        const payload = (await response.json()) as unknown;
        if (!response.ok) {
          const message =
            payload && typeof payload === "object" && "error" in payload
              ? String((payload as { error?: unknown }).error ?? "Failed to fetch analyses.")
              : "Failed to fetch analyses.";
          throw new Error(message);
        }
        if (!cancelled) {
          setAnalyses(normalizeAnalyses(payload));
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "Failed to fetch analyses.");
          setAnalyses([]);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const sortedAnalyses = useMemo(
    () =>
      [...analyses].sort((a, b) => {
        const aTime = a.createdAt ? new Date(a.createdAt).getTime() : 0;
        const bTime = b.createdAt ? new Date(b.createdAt).getTime() : 0;
        return bTime - aTime;
      }),
    [analyses],
  );

  async function verifyAnalysis(analysisId: string) {
    setVerifyingMap((current) => ({ ...current, [analysisId]: true }));
    setVerifyErrorMap((current) => {
      const next = { ...current };
      delete next[analysisId];
      return next;
    });

    try {
      const response = await fetch(`/api/analyses/${encodeURIComponent(analysisId)}/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      const payload = (await response.json()) as VerifyResponse | { error?: string };
      if (!response.ok) {
        throw new Error(
          payload && typeof payload === "object" && "error" in payload
            ? String(payload.error ?? "Verification failed.")
            : "Verification failed.",
        );
      }

      const verifiedPayload = payload as VerifyResponse;
      const status = (verifiedPayload.status ?? "unverified").toLowerCase();
      const lastVerifiedAt = verifiedPayload.lastVerifiedAt;

      setAnalyses((current) =>
        current.map((analysis) =>
          analysis.analysisId === analysisId
            ? {
                ...analysis,
                verificationStatus: status,
                status,
                lastVerifiedAt: lastVerifiedAt ?? analysis.lastVerifiedAt,
              }
            : analysis,
        ),
      );
    } catch (verifyError) {
      setVerifyErrorMap((current) => ({
        ...current,
        [analysisId]: verifyError instanceof Error ? verifyError.message : "Verification failed.",
      }));
    } finally {
      setVerifyingMap((current) => {
        const next = { ...current };
        delete next[analysisId];
        return next;
      });
    }
  }

  return (
    <main
      className="app-shell"
      style={{
        minHeight: "100vh",
        background: "var(--bg-base)",
        color: "var(--fg-primary)",
        padding: "24px 20px 48px",
        transition: "background 0.25s ease, color 0.25s ease",
      }}
    >
      <div className="app-inner" style={{ maxWidth: 1280, margin: "0 auto", display: "flex", flexDirection: "column", gap: 20 }}>
        <header className="sis-surface-overlay" style={{ padding: "24px 28px" }}>
          <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 20, flexWrap: "wrap" }}>
            <div style={{ flex: "1 1 260px", minWidth: 0 }}>
              <p className="sis-eyebrow" style={{ marginBottom: 10 }}>oasis structures</p>
              <h1
                style={{
                  fontSize: "clamp(20px, 3vw, 30px)",
                  fontWeight: 700,
                  letterSpacing: "-0.025em",
                  lineHeight: 1.2,
                  color: "var(--fg-primary)",
                  margin: 0,
                }}
              >
                Analysis
              </h1>
              <p
                style={{
                  marginTop: 8,
                  fontSize: 13,
                  lineHeight: 1.7,
                  color: "var(--fg-secondary)",
                  maxWidth: 560,
                }}
              >
                Saved analyses from backend API. Open any record in read-only workspace mode with model data preloaded.
              </p>
            </div>

            <div className="header-quick-links" style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
              <Link href="/" className="btn btn-default">Home</Link>
              <Link href="/workspace" className="btn btn-default">Workspace</Link>
              <button
                type="button"
                onClick={toggleTheme}
                className="btn btn-default"
                title={darkMode ? "Switch to light theme" : "Switch to dark theme"}
                aria-label={darkMode ? "Switch to light theme" : "Switch to dark theme"}
                style={{
                  minWidth: 40,
                  width: 40,
                  height: 34,
                  padding: 0,
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexShrink: 0,
                }}
              >
                {darkMode ? (
                  <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                    <circle cx="12" cy="12" r="5" />
                    <path
                      d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"
                      strokeLinecap="round"
                    />
                  </svg>
                ) : (
                  <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                    <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                )}
              </button>
            </div>
          </div>
        </header>

        {loading ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 8, padding: 20 }}>
            <p className="sis-label" style={{ marginBottom: 12 }}>Loading analyses...</p>
            {[140, 100, 120].map((w, i) => (
              <div key={i} className="shimmer" style={{ height: 14, width: `${w}px` }} />
            ))}
          </div>
        ) : null}

        {error ? (
          <div className="issue-entry error" style={{ margin: "10px 0" }}>
            {error}
          </div>
        ) : null}

        {!loading && !error && sortedAnalyses.length === 0 ? (
          <div className="sis-surface-sunken" style={{ padding: "40px 20px", textAlign: "center" }}>
            <p style={{ fontSize: 14, color: "var(--fg-muted)", margin: 0 }}>No analyses found.</p>
          </div>
        ) : null}

        <div className="analysis-grid">
          {sortedAnalyses.map((analysis) => (
            <article key={analysis.analysisId} className="sis-surface" style={{ display: "flex", flexDirection: "column", padding: "20px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center", marginBottom: 12 }}>
                <p style={{ margin: 0, fontSize: 14, fontFamily: "var(--font-mono)", fontWeight: 700, overflowWrap: "anywhere", color: "var(--fg-primary)" }}>
                  {analysis.analysisId}
                </p>
                <span className="badge badge-neutral" style={{ flexShrink: 0 }}>
                  {analysis.lineItems?.length ?? 0} items
                </span>
              </div>

              <p style={{ margin: "0 0 14px", fontSize: 13, color: "var(--fg-muted)" }}>
                {analysis.createdAt ? new Date(analysis.createdAt).toLocaleString("en-IN") : "Unknown creation time"}
              </p>

              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, flexWrap: "wrap", marginBottom: 20 }}>
                <span
                  className={`badge ${
                    analysisStatus(analysis) === "verified"
                      ? "badge-primary"
                      : analysisStatus(analysis) === "tampered"
                        ? "badge-danger"
                        : "badge-warning"
                  }`}
                >
                  {statusLabel(analysisStatus(analysis))}
                </span>
                <span style={{ fontSize: 11, color: "var(--fg-muted)" }}>
                  {analysis.lastVerifiedAt
                    ? `Last verified: ${new Date(analysis.lastVerifiedAt).toLocaleString("en-IN")}`
                    : "Last verified: Never"}
                </span>
              </div>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
                  gap: 8,
                  padding: 12,
                  borderRadius: "var(--radius-md)",
                  border: "1px solid var(--border-subtle)",
                  background: "var(--bg-sunken)",
                  marginBottom: 20,
                }}
              >
                <div>
                  <p className="sis-label" style={{ marginBottom: 4 }}>Total Cost</p>
                  <p style={{ margin: 0, fontSize: 13, fontFamily: "var(--font-mono)", fontWeight: 600 }}>{formatMoney(analysis.totalCost)}</p>
                </div>
                <div>
                  <p className="sis-label" style={{ marginBottom: 4 }}>Total Area</p>
                  <p style={{ margin: 0, fontSize: 13, fontFamily: "var(--font-mono)", fontWeight: 600 }}>
                    {typeof analysis.totalArea === "number" ? `${analysis.totalArea.toFixed(2)} m²` : "N/A"}
                  </p>
                </div>
                <div>
                  <p className="sis-label" style={{ marginBottom: 4 }}>Cost per m²</p>
                  <p style={{ margin: 0, fontSize: 13, fontFamily: "var(--font-mono)", fontWeight: 600 }}>{formatMoney(analysis.costPerM2)}</p>
                </div>
              </div>

              {verifyErrorMap[analysis.analysisId] ? (
                <p style={{ margin: "-8px 0 10px", color: "var(--danger-fg)", fontSize: 12 }}>
                  {verifyErrorMap[analysis.analysisId]}
                </p>
              ) : null}

              <div style={{ marginTop: "auto", display: "flex", gap: 8, flexWrap: "wrap" }}>
                <button
                  type="button"
                  className="btn btn-accent"
                  onClick={() => verifyAnalysis(analysis.analysisId)}
                  disabled={Boolean(verifyingMap[analysis.analysisId])}
                  style={{ flex: 1, minWidth: 160, opacity: verifyingMap[analysis.analysisId] ? 0.74 : 1 }}
                >
                  {verifyingMap[analysis.analysisId] ? "Verifying..." : "Verify Again"}
                </button>
                <Link
                  href={`/workspace?analysisId=${encodeURIComponent(analysis.analysisId)}&readonly=1`}
                  className="btn btn-primary"
                  style={{ flex: 1, minWidth: 220 }}
                >
                  Open Read-Only Workspace
                </Link>
              </div>
            </article>
          ))}
        </div>
      </div>
    </main>
  );
}

