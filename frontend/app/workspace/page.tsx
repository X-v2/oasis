import type { Metadata } from "next";

import WorkspaceApp from "@/components/WorkspaceApp";

export const metadata: Metadata = {
  title: "Workspace",
  description: "Upload plans, inspect structural geometry in 3D, and review material and optimisation guidance.",
};

type WorkspacePageProps = {
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
};

export default async function WorkspacePage({ searchParams }: WorkspacePageProps) {
  const resolved = (await searchParams) ?? {};
  const analysisParam = resolved.analysisId;
  const readonlyParam = resolved.readonly;
  const initialAnalysisId = Array.isArray(analysisParam) ? analysisParam[0] : analysisParam ?? null;
  const readonlyValue = Array.isArray(readonlyParam) ? readonlyParam[0] : readonlyParam;
  const initialReadOnlyMode = readonlyValue === "1" || readonlyValue === "true";

  return <WorkspaceApp initialAnalysisId={initialAnalysisId} initialReadOnlyMode={initialReadOnlyMode} />;
}
