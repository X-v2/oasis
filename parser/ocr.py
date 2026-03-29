import re

import cv2
import easyocr
import numpy as np

from parser.config import LABEL_CONFIDENCE_THRESHOLD
from parser.geometry import bbox_center, bbox_contains_point, clean_text

reader = easyocr.Reader(["en"], gpu=False)


def parse_numeric_value(text):
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def run_ocr(image):
    results = reader.readtext(image)
    items = []

    for bbox, text, conf in results:
        clean = clean_text(text)
        xs = [int(point[0]) for point in bbox]
        ys = [int(point[1]) for point in bbox]
        bbox_xyxy = [min(xs), min(ys), max(xs), max(ys)]
        center = bbox_center(bbox_xyxy)
        items.append({"text": clean, "confidence": float(conf), "bbox": bbox_xyxy, "center": center})

    return items


def build_text_mask(shape, ocr_items, padding=4):
    mask = np.zeros(shape[:2], dtype=np.uint8)

    for item in ocr_items:
        x1, y1, x2, y2 = item["bbox"]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(shape[1] - 1, x2 + padding)
        y2 = min(shape[0] - 1, y2 + padding)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    return mask


def filter_room_labels(ocr_items, building_bbox):
    x1, y1, x2, y2 = building_bbox
    labels = []

    for item in ocr_items:
        if item["confidence"] < LABEL_CONFIDENCE_THRESHOLD:
            continue
        if not bbox_contains_point((x1, y1, x2, y2), item["center"], padding=6):
            continue
        if not re.search(r"[A-Za-z]", item["text"]):
            continue
        labels.append(item)

    return labels


def point_near_any_label(point, ocr_items, padding=28):
    for item in ocr_items:
        if bbox_contains_point(item["bbox"], point, padding=padding):
            return True
    return False
