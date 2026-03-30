from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.storage import get_analysis, init_db, insert_analysis, list_analyses, update_verification_status
from backend.stellar import compute_analysis_hash, record_analysis_hash, get_onchain_hash
from parser.config import ParserConfig
from parser.main import parse_floorplan_bytes


app = FastAPI(title="Floorplan Parser API")
init_db()


class AnalysisLineItemInput(BaseModel):
    itemId: str | None = None
    elementType: str
    material: str
    quantity: float
    unit: str
    unitRate: float
    subtotal: float | None = None
    justification: str = ""


class AnalysisCreateRequest(BaseModel):
    totalCost: float | None = None
    totalArea: float
    costPerM2: float | None = None
    lineItems: list[AnalysisLineItemInput] = Field(default_factory=list)
    modelJson: dict


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/parse")
async def parse_floorplan_api(image: UploadFile = File(...)) -> JSONResponse:
    if not image.filename:
        raise HTTPException(status_code=400, detail="Missing image filename")
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        data = await image.read()
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded image is empty")

        config = ParserConfig(debug_enabled=False)
        payload = parse_floorplan_bytes(data, config=config)
        return JSONResponse(content=payload, status_code=200)
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(
            content={"error": str(exc)},
            status_code=500,
        )


@app.post("/analyses")
def create_analysis(request: AnalysisCreateRequest) -> JSONResponse:
    created_at = datetime.now(UTC).isoformat()
    analysis_id = f"analysis_{uuid4().hex[:12]}"

    line_items: list[dict] = []
    for index, item in enumerate(request.lineItems, start=1):
        subtotal = float(item.subtotal if item.subtotal is not None else item.quantity * item.unitRate)
        raw_item_id = item.itemId or f"item_{index}"
        line_items.append(
            {
                "itemId": f"{analysis_id}_{raw_item_id}",
                "elementType": item.elementType,
                "material": item.material,
                "quantity": float(item.quantity),
                "unit": item.unit,
                "unitRate": float(item.unitRate),
                "subtotal": subtotal,
                "justification": item.justification,
            }
        )

    total_cost = float(request.totalCost if request.totalCost is not None else sum(item["subtotal"] for item in line_items))
    total_area = float(request.totalArea)
    cost_per_m2 = float(request.costPerM2 if request.costPerM2 is not None else (total_cost / total_area if total_area > 0 else 0.0))

    hash_payload = {
        "analysisId": analysis_id,
        "createdAt": created_at,
        "totalCost": total_cost,
        "totalArea": total_area,
        "costPerM2": cost_per_m2,
        "lineItems": line_items,
        "modelJson": request.modelJson,
    }
    data_hash = compute_analysis_hash(hash_payload)
    stellar = record_analysis_hash(analysis_id=analysis_id, data_hash=data_hash, created_at=created_at)

    analysis_record = {
        "analysisId": analysis_id,
        "createdAt": created_at,
        "totalCost": total_cost,
        "totalArea": total_area,
        "costPerM2": cost_per_m2,
        "lineItems": line_items,
        "modelJson": request.modelJson,
        "dataHash": data_hash,
        "stellar": stellar,
    }
    insert_analysis(analysis_record, line_items)
    return JSONResponse(content=analysis_record, status_code=201)


@app.get("/analyses")
def list_analyses_api() -> JSONResponse:
    return JSONResponse(content={"analyses": list_analyses()}, status_code=200)


@app.get("/analyses/{analysis_id}")
def get_analysis_api(analysis_id: str) -> JSONResponse:
    analysis = get_analysis(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return JSONResponse(content=analysis, status_code=200)


@app.get("/verify/{analysis_id}")
def verify_analysis(analysis_id: str) -> JSONResponse:
    analysis = get_analysis(analysis_id)

    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # ✅ USE STORED HASH (FIXED)
    db_hash = analysis["dataHash"]

    # get blockchain hash
    chain_hash = get_onchain_hash(analysis_id)

    if not chain_hash:
        return JSONResponse(
            content={"status": "error", "message": "Could not fetch on-chain hash"},
            status_code=500,
        )

    is_valid = db_hash == chain_hash

    # update DB
    status = "verified" if is_valid else "tampered"
    verified_at = datetime.now(UTC).isoformat()

    update_verification_status(analysis_id, status, verified_at)

    # return updated state
    return JSONResponse(
        content={
            "analysisId": analysis_id,
            "valid": is_valid,
            "status": status,
            "lastVerifiedAt": verified_at,
            "dbHash": db_hash,
            "chainHash": chain_hash,
        },
        status_code=200,
    )