const ANALYSIS_API_BASE = process.env.ANALYSIS_API_BASE_URL?.replace(/\/+$/, "");

export async function POST(request: Request) {
  if (!ANALYSIS_API_BASE) {
    return Response.json({ error: "ANALYSIS_API_BASE_URL is not configured." }, { status: 500 });
  }

  try {
    const payload = await request.json();
    const upstream = await fetch(`${ANALYSIS_API_BASE}/analyses`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
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

export async function GET() {
  if (!ANALYSIS_API_BASE) {
    return Response.json({ error: "ANALYSIS_API_BASE_URL is not configured." }, { status: 500 });
  }

  try {
    const upstream = await fetch(`${ANALYSIS_API_BASE}/analyses`, {
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
