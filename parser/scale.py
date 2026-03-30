from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from parser.config import ParserConfig
from parser.text import TextDetection


@dataclass(slots=True)
class ScaleResult:
    pixels_per_meter: float
    scale_meters: float
    source: str


def detect_scale(
    gray_image: np.ndarray,
    plan_bbox: tuple[int, int, int, int],
    texts: list[TextDetection],
    config: ParserConfig,
) -> ScaleResult:
    del plan_bbox
    fallback = ScaleResult(
        pixels_per_meter=config.default_pixels_per_meter,
        scale_meters=config.default_scale_m,
        source="fallback",
    )
    height, width = gray_image.shape
    roi_top = int(height * 0.82)
    roi_right = int(width * 0.4)
    roi = gray_image[roi_top:, :roi_right]
    if roi.size == 0:
        return fallback

    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horizontal = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1)),
    )
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scale_length_px = 0
    for contour in contours:
        x, _, w, h = cv2.boundingRect(contour)
        if w < 40 or h > 8:
            continue
        if x > width * 0.15:
            continue
        scale_length_px = max(scale_length_px, int(w))

    if scale_length_px == 0:
        return fallback

    scale_value = _bottom_left_scale_value(texts, width, height)
    if scale_value is None:
        return fallback
    return ScaleResult(
        pixels_per_meter=scale_length_px / scale_value,
        scale_meters=scale_value,
        source="scale_bar",
    )


def _parse_scale_value(text: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*M", text.upper())
    if not match:
        return None
    return float(match.group(1))


def _bottom_left_scale_value(
    texts: list[TextDetection],
    width: int,
    height: int,
) -> float | None:
    best_value: float | None = None
    best_score = -1.0
    for text in texts:
        x1, y1, x2, y2 = text.bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        if cy < int(height * 0.88):
            continue
        if cx > int(width * 0.65):
            continue
        value = _parse_scale_value(text.text)
        if value is None:
            value = _parse_numeric_value(text.text)
        if value is None:
            continue
        score = text.confidence
        if "M" in text.text.upper():
            score += 1.0
        if score > best_score:
            best_score = score
            best_value = value
    return best_value


def _parse_numeric_value(text: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text.upper())
    if not match:
        return None
    return float(match.group(1))
