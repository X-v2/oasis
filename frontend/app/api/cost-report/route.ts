import { generateCostReport, generateCostReportOptions } from "@/lib/costReport";
import type { CostReportOptionsResponse, MaterialPricingEntry, SceneData } from "@/lib/types";

type RequestBody = {
  scene?: SceneData;
  projectName?: string;
  currency?: string;
  wastageOverridePercent?: number;
  includeBeamCandidates?: boolean;
  pricing?: Record<string, MaterialPricingEntry>;
  includeScenarioOptions?: boolean;
  elementMaterialOverrides?: Record<string, string>;
};

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as RequestBody;
    if (!body.scene) {
      return Response.json({ error: "Missing scene payload." }, { status: 400 });
    }

    if (body.includeScenarioOptions) {
      const optionsPayload: CostReportOptionsResponse = {
        generatedAt: new Date().toISOString(),
        options: generateCostReportOptions(body.scene, {
          projectName: body.projectName,
          currency: body.currency,
          wastageOverridePercent: body.wastageOverridePercent,
          includeBeamCandidates: body.includeBeamCandidates,
          elementMaterialOverrides: body.elementMaterialOverrides,
        }),
      };
      return Response.json(optionsPayload);
    }

    return Response.json(
      generateCostReport(body.scene, {
        projectName: body.projectName,
        currency: body.currency,
        wastageOverridePercent: body.wastageOverridePercent,
        includeBeamCandidates: body.includeBeamCandidates,
        pricing: body.pricing,
        elementMaterialOverrides: body.elementMaterialOverrides,
      }),
    );
  } catch {
    return Response.json({ error: "Failed to generate cost report." }, { status: 500 });
  }
}
