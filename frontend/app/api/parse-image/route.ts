const PARSE_API_URL = "https://pulled-budapest-mind-peter.trycloudflare.com/parse";

export async function POST(request: Request) {
  try {
    const incoming = await request.formData();
    const image = incoming.get("image");

    if (!(image instanceof File)) {
      return Response.json({ error: "Missing image file in field 'image'." }, { status: 400 });
    }

    const outbound = new FormData();
    outbound.append("image", image, image.name);

    const upstream = await fetch(PARSE_API_URL, {
      method: "POST",
      body: outbound,
    });

    let payload: unknown = null;
    try {
      payload = await upstream.json();
    } catch {
      payload = { error: "Invalid JSON response from parser backend." };
    }

    return Response.json(payload, { status: upstream.status });
  } catch {
    return Response.json({ error: "Failed to reach parser backend." }, { status: 502 });
  }
}
