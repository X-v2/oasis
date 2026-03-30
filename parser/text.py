from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import easyocr
import numpy as np
import torch

from parser.config import ParserConfig


@dataclass(slots=True)
class TextDetection:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    kind: str


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).upper()


def _looks_like_scale_text(text: str) -> bool:
    return bool(re.search(r"(?:^|[^A-Z0-9])~?\s*\d+(?:\.\d+)?\s*M(?:$|[^A-Z])", text.upper()))


@lru_cache(maxsize=1)
def _reader(languages: tuple[str, ...], use_gpu: bool) -> easyocr.Reader:
    return easyocr.Reader(list(languages), gpu=use_gpu, verbose=False)


def detect_text(gray_image: np.ndarray, config: ParserConfig) -> list[TextDetection]:
    reader = _reader(config.easyocr_languages, torch.cuda.is_available())
    raw_results = reader.readtext(gray_image, detail=1, paragraph=False)
    detections: list[TextDetection] = []
    for points, text, confidence in raw_results:
        xs = [int(round(point[0])) for point in points]
        ys = [int(round(point[1])) for point in points]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        normalized = _normalize_text(text)
        kind = classify_text(normalized, config)
        detections.append(
            TextDetection(text=normalized, confidence=float(confidence), bbox=bbox, kind=kind)
        )
    return detections


def classify_text(text: str, config: ParserConfig) -> str:
    if text in config.room_name_allowlist or any(name in text for name in config.room_name_allowlist):
        return "room"
    if any(keyword in text for keyword in config.ignored_text_keywords):
        return "ignore"
    if _looks_like_scale_text(text):
        return "scale"
    return "other"


def build_text_mask(
    shape: tuple[int, int], detections: list[TextDetection], config: ParserConfig
) -> np.ndarray:
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        pad = config.text_mask_padding_px
        cv2.rectangle(
            mask,
            (max(0, x1 - pad), max(0, y1 - pad)),
            (min(shape[1] - 1, x2 + pad), min(shape[0] - 1, y2 + pad)),
            255,
            thickness=-1,
        )
    return mask


def serialize_text_detections(detections: list[TextDetection]) -> list[dict[str, Any]]:
    return [
        {
            "text": detection.text,
            "confidence": round(detection.confidence, 3),
            "bbox": list(detection.bbox),
            "kind": detection.kind,
        }
        for detection in detections
    ]
