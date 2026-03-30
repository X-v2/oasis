const ANALYSIS_API_BASE = process.env.ANALYSIS_API_BASE_URL?.replace(/\/+$/, "");

type Params = {
  params: Promise<{ analysisId: string }>;
};

export async function POST(_: Request, { params }: Params) {
  if (!ANALYSIS_API_BASE) {
    return Response.json({ error: "ANALYSIS_API_BASE_URL is not configured." }, { status: 500 });
  }

  try {
    const { analysisId } = await params;
    const upstream = await fetch(`${ANALYSIS_API_BASE}/verify/${encodeURIComponent(analysisId)}`, {
      method: "GET",
      headers: { Accept: "application/json" },
      cache: "no-store",
    });

    let body: unknown = null;
    try {
      body = await upstream.json();
    } catch {
      body = { error: "Invalid JSON response from analysis backend." };
    }

    return Response.json(body, { status: upstream.status });
  } catch {
    return Response.json({ error: "Failed to reach analysis backend." }, { status: 502 });
  }
}

