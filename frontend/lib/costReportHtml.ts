import "server-only";

import type { CostReport } from "@/lib/types";

type ReportImage = {
  title: string;
  dataUri: string;
};

function escapeHtml(value: string) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function money(value: number, currency: string) {
  const symbol = currency.toUpperCase() === "INR" ? "₹" : currency;
  return `${symbol} ${value.toLocaleString("en-IN", { maximumFractionDigits: 2 })}`;
}

function formatUnit(unit: string) {
  if (unit === "m2") return "m²";
  if (unit === "m3") return "m³";
  if (unit === "nos") return "Nos";
  return unit;
}

export function renderCostReportHtml(report: CostReport, images: ReportImage[] = []) {
  const rows = report.items
    .map(
      (item) => `
        <tr>
          <td>${escapeHtml(item.elementType.replaceAll("_", " "))}</td>
          <td>${escapeHtml(item.material)}</td>
          <td>${item.quantity.toFixed(2)} ${escapeHtml(formatUnit(item.unit))}</td>
          <td>${money(item.unitRate, report.currency)}</td>
          <td>${money(item.subtotal, report.currency)}</td>
        </tr>
      `,
    )
    .join("");

  const materialRows = Object.entries(report.materialTotals)
    .map(
      ([material, subtotal]) =>
        `<tr><td>${escapeHtml(material)}</td><td>${money(subtotal, report.currency)}</td></tr>`,
    )
    .join("");

  const assumptions = report.assumptions.slice(0, 3).map((entry) => `<li>${escapeHtml(entry)}</li>`).join("");
  const warnings = report.warnings.slice(0, 3).map((entry) => `<li>${escapeHtml(entry)}</li>`).join("");
  const gallery = images
    .map(
      (image) => `
      <div class="img-card">
        <img src="${image.dataUri}" alt="${escapeHtml(image.title)}" />
        <p>${escapeHtml(image.title)}</p>
      </div>`,
    )
    .join("");

  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>${escapeHtml(report.projectName)} - Cost Report</title>
    <style>
      :root { color-scheme: light; }
      @page { size: A4 portrait; margin: 10mm; }
      body { font-family: "Helvetica Neue", Arial, sans-serif; margin: 0 auto; max-width: 188mm; color: #1a1a1a; padding: 3mm 2mm; line-height: 1.45; }
      h1 { margin: 0 0 4px; font-size: 22px; letter-spacing: 0.02em; }
      h2 { margin: 12px 0 6px; font-size: 14px; }
      .meta { margin: 0 0 8px; color: #4a4a4a; font-size: 11px; }
      .project-line { margin: 0 0 8px; font-size: 16px; font-weight: 700; color: #1f2937; }
      table { width: 100%; border-collapse: collapse; margin-top: 8px; }
      th, td { border: 1px solid #ddd; padding: 6px 7px; font-size: 10.5px; vertical-align: top; word-break: break-word; }
      th { background: #f8f8f8; text-align: left; }
      tfoot td { font-weight: 700; background: #fafafa; }
      .image-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin: 8px 0 12px; }
      .img-card { border: 1px solid #ddd; border-radius: 6px; overflow: hidden; background: #fff; }
      .img-card img { width: 100%; height: 110px; object-fit: cover; display: block; }
      .img-card p { margin: 4px 6px 6px; font-size: 10px; color: #4b5563; }
      .totals { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }
      .card { border: 1px solid #ddd; border-radius: 6px; padding: 7px; }
      .label { font-size: 9px; text-transform: uppercase; color: #666; letter-spacing: 0.06em; }
      .value { font-size: 12px; font-weight: 700; margin-top: 3px; }
      ul { margin: 6px 0 0; padding-left: 16px; }
      li { font-size: 10px; margin: 3px 0; color: #525252; }
      .legal { font-size: 9.5px; color: #5b5b5b; }
      @media print {
        body { max-width: none; padding: 0; }
      }
    </style>
  </head>
  <body>
    <h1>Oasis Structures</h1>
    <p class="project-line">Project: ${escapeHtml(report.projectName)}</p>
    <p class="meta">Generated: ${escapeHtml(new Date(report.generatedAt).toLocaleString("en-IN"))}</p>

    <h2>Model Visuals</h2>
    <div class="image-grid">${gallery}</div>

    <h2>Material Summary</h2>
    <table>
      <thead><tr><th>Material</th><th>Total</th></tr></thead>
      <tbody>${materialRows}</tbody>
    </table>

    <h2>Line Items (${report.items.length})</h2>
    <table>
      <thead>
        <tr>
          <th>Element Type</th>
          <th>Material</th>
          <th>Qty</th>
          <th>Unit Rate</th>
          <th>Amount</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
      <tfoot>
        <tr>
          <td colspan="4">Line Items Total</td>
          <td>${money(report.summary.totalCost, report.currency)}</td>
        </tr>
      </tfoot>
    </table>

    <div class="totals">
      <div class="card"><div class="label">Total Cost</div><div class="value">${money(report.summary.totalCost, report.currency)}</div></div>
      <div class="card"><div class="label">Total Area</div><div class="value">${(report.summary.totalArea ?? 0).toFixed(2)} m²</div></div>
      <div class="card"><div class="label">Cost per m²</div><div class="value">${money(report.summary.costPerSqm ?? 0, report.currency)}</div></div>
      <div class="card"><div class="label">Line Items</div><div class="value">${report.items.length}</div></div>
    </div>

    <h2>Terms and Assumptions</h2>
    <ul class="legal">
      ${assumptions}
      ${
        report.warnings.length > 0
          ? warnings
          : "<li>No critical warnings observed for this estimate run.</li>"
      }
      <li>This is a preliminary estimate for planning and budgeting purposes only.</li>
      <li>Final payable quantities shall be based on approved drawings and site measurements.</li>
      <li>Rates are indicative and may vary by location, market movement, vendor, and execution timeline.</li>
      <li>Taxes, statutory fees, transport, and contractor overheads are excluded unless stated otherwise.</li>
      <li>Confirm structural design adequacy and local code compliance before procurement or construction.</li>
    </ul>
  </body>
</html>`;
}
