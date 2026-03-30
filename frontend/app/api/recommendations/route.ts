import { buildLayoutHeuristicReport } from "@/lib/layoutHeuristics";
import { generateMaterialRecommendationTable } from "@/lib/recommendation";
import type { RecommendationApiResponse, SceneData, SceneSelection } from "@/lib/types";

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as {
      scene?: SceneData;
      selectedEntity?: SceneSelection | null;
      selectionMetrics?: string;
    };
    if (!body.scene) {
      return Response.json({ error: "Missing scene payload." }, { status: 400 });
    }

    const materialTable = await generateMaterialRecommendationTable(
      body.scene,
      body.selectedEntity ?? null,
      body.selectionMetrics,
    );
    const heuristics = buildLayoutHeuristicReport(body.scene);
    const payload: RecommendationApiResponse = { materialTable, heuristics };
    return Response.json(payload);
  } catch {
    return Response.json({ error: "Failed to generate recommendations." }, { status: 500 });
  }
}
