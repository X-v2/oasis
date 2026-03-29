import cv2

from parser.config import FALLBACK_METERS_PER_PIXEL
from parser.ocr import parse_numeric_value


def extract_scale_layer(gray, ocr_items):
    height, width = gray.shape
    roi = gray[int(height * 0.82) :, : int(width * 0.4)]
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horizontal = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1)),
    )
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_bbox = None
    best_length = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 40 or h > 8:
            continue
        if x > width * 0.15:
            continue
        if w > best_length:
            best_length = w
            line_bbox = [x, y + int(height * 0.82), x + w - 1, y + int(height * 0.82) + h - 1]

    line_pixels = best_length if line_bbox else 0
    scale_value = None

    if line_bbox:
        for item in ocr_items:
            if item["center"][1] < int(height * 0.88):
                continue
            if item["center"][0] > int(width * 0.65):
                continue
            numeric = parse_numeric_value(item["text"])
            if numeric is not None:
                scale_value = numeric
                break

    meters_per_pixel = FALLBACK_METERS_PER_PIXEL
    scale_source = "fallback"
    if scale_value and line_pixels:
        meters_per_pixel = scale_value / float(line_pixels)
        scale_source = "scale_bar"

    return {
        "meters_per_pixel": meters_per_pixel,
        "scale_source": scale_source,
        "line_bbox": line_bbox,
        "line_pixels": line_pixels,
        "scale_value_meters": scale_value,
    }
