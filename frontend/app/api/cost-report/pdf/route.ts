import { COST_SCENARIOS, generateCostReport } from "@/lib/costReport";
import { renderCostReportHtml } from "@/lib/costReportHtml";
import { buildReportModelImages } from "@/lib/reportModelImages";
import type { MaterialPricingEntry, SceneData } from "@/lib/types";

type RequestBody = {
  scene?: SceneData;
  projectName?: string;
  currency?: string;
  wastageOverridePercent?: number;
  includeBeamCandidates?: boolean;
  pricing?: Record<string, MaterialPricingEntry>;
  scenarioId?: string;
  modelScreenshots?: Array<{ title?: string; dataUri: string }>;
  elementMaterialOverrides?: Record<string, string>;
};

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as RequestBody;
    if (!body.scene) {
      return Response.json({ error: "Missing scene payload." }, { status: 400 });
    }

    const selectedScenario = body.scenarioId
      ? COST_SCENARIOS.find((entry) => entry.id === body.scenarioId)
      : undefined;
    const report = generateCostReport(body.scene, {
      projectName: body.projectName,
      currency: body.currency,
      wastageOverridePercent:
        (body.wastageOverridePercent ?? 0) + (selectedScenario?.wastageDeltaPercent ?? 0),
      includeBeamCandidates: body.includeBeamCandidates,
      pricing: selectedScenario ? { ...selectedScenario.pricing, ...(body.pricing ?? {}) } : body.pricing,
      materialByElement: selectedScenario?.materialByElement,
      elementMaterialOverrides: body.elementMaterialOverrides,
      scenarioId: selectedScenario?.id,
      scenarioLabel: selectedScenario?.label,
      scenarioDescription: selectedScenario?.description,
    });
    const fallbackImages = buildReportModelImages(body.scene);
    const inputImages =
      body.modelScreenshots?.filter((entry) => typeof entry.dataUri === "string" && entry.dataUri.startsWith("data:image/")) ??
      [];
    const images =
      inputImages.length > 0
        ? inputImages.slice(0, 4).map((entry, index) => ({
            title: entry.title?.trim() || `Model Screenshot ${index + 1}`,
            dataUri: entry.dataUri,
          }))
        : fallbackImages;

    const html = renderCostReportHtml(report, images);

    return new Response(html, {
      headers: {
        "Content-Type": "text/html; charset=utf-8",
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return Response.json({ error: "Failed to generate PDF preview." }, { status: 500 });
  }
}
