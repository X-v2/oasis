import cv2
import numpy as np

from parser.config import (
    ALIGNMENT_TOLERANCE,
    DOOR_GAP_TOLERANCE,
    EXPORT_GAP_TOLERANCE,
    HORIZONTAL_WALL_KERNEL,
    LINE_LENGTH_THRESHOLD,
    SHORT_WALL_BUMP_MAX_LENGTH,
    SHORT_WALL_CONNECTION_TOLERANCE,
    SHORT_WALL_KERNEL,
    SHORT_WALL_MIN_LENGTH,
    THIN_HORIZONTAL_KERNEL,
    THIN_VERTICAL_KERNEL,
    VERTICAL_WALL_KERNEL,
)
from parser.geometry import bbox_contains_point, scale_point, segment_length, segment_start_end


def extract_line_segments(mask, orientation, min_length=LINE_LENGTH_THRESHOLD, max_thickness=None):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        length = max(w, h)
        thickness = h if orientation == "horizontal" else w

        if length < min_length:
            continue
        if max_thickness is not None and thickness > max_thickness:
            continue

        if orientation == "horizontal":
            segments.append({"orientation": "horizontal", "x1": x, "y": y + h // 2, "x2": x + w - 1, "thickness_px": h})
        else:
            segments.append({"orientation": "vertical", "x": x + w // 2, "y1": y, "y2": y + h - 1, "thickness_px": w})

    return segments


def merge_horizontal_segments(segments, gap_tolerance):
    segments = sorted(segments, key=lambda segment: (segment["y"], segment["x1"]))
    merged = []

    for segment in segments:
        if merged and abs(segment["y"] - merged[-1]["y"]) <= ALIGNMENT_TOLERANCE and segment["x1"] <= merged[-1]["x2"] + gap_tolerance:
            current = merged[-1]
            current["y"] = int(round((current["y"] + segment["y"]) / 2))
            current["x1"] = min(current["x1"], segment["x1"])
            current["x2"] = max(current["x2"], segment["x2"])
            current["thickness_px"] = max(current["thickness_px"], segment["thickness_px"])
        else:
            merged.append(segment.copy())

    return merged


def merge_vertical_segments(segments, gap_tolerance):
    segments = sorted(segments, key=lambda segment: (segment["x"], segment["y1"]))
    merged = []

    for segment in segments:
        if merged and abs(segment["x"] - merged[-1]["x"]) <= ALIGNMENT_TOLERANCE and segment["y1"] <= merged[-1]["y2"] + gap_tolerance:
            current = merged[-1]
            current["x"] = int(round((current["x"] + segment["x"]) / 2))
            current["y1"] = min(current["y1"], segment["y1"])
            current["y2"] = max(current["y2"], segment["y2"])
            current["thickness_px"] = max(current["thickness_px"], segment["thickness_px"])
        else:
            merged.append(segment.copy())

    return merged


def _point_near_segment(point, segment, tolerance):
    start, end = segment_start_end(segment)
    if segment["orientation"] == "horizontal":
        x1, x2 = sorted((start[0], end[0]))
        return x1 - tolerance <= point[0] <= x2 + tolerance and abs(point[1] - start[1]) <= tolerance
    y1, y2 = sorted((start[1], end[1]))
    return y1 - tolerance <= point[1] <= y2 + tolerance and abs(point[0] - start[0]) <= tolerance


def _segment_overlaps_existing(candidate, existing_segments, tolerance):
    for existing in existing_segments:
        if existing["orientation"] != candidate["orientation"]:
            continue
        if candidate["orientation"] == "horizontal":
            if abs(candidate["y"] - existing["y"]) > tolerance:
                continue
            if candidate["x1"] <= existing["x2"] + tolerance and candidate["x2"] >= existing["x1"] - tolerance:
                return True
        else:
            if abs(candidate["x"] - existing["x"]) > tolerance:
                continue
            if candidate["y1"] <= existing["y2"] + tolerance and candidate["y2"] >= existing["y1"] - tolerance:
                return True
    return False


def _anchored_short_segment(candidate, existing_segments, tolerance):
    start, end = segment_start_end(candidate)
    anchors = 0
    for point in (start, end):
        if any(_point_near_segment(point, existing, tolerance) for existing in existing_segments):
            anchors += 1
    return anchors >= 2


def _segment_anchor_points(candidate, existing_segments, tolerance):
    anchors = []
    for point in segment_start_end(candidate):
        matches = [existing for existing in existing_segments if _point_near_segment(point, existing, tolerance)]
        anchors.append((point, matches))
    return anchors


def _is_perpendicular_bump(candidate, existing_segments, tolerance):
    length = segment_length(candidate)
    if length > SHORT_WALL_BUMP_MAX_LENGTH:
        return False

    anchor_points = _segment_anchor_points(candidate, existing_segments, tolerance)
    anchored = [(point, matches) for point, matches in anchor_points if matches]
    free = [(point, matches) for point, matches in anchor_points if not matches]
    if len(anchored) != 1 or len(free) != 1:
        return False

    _, matches = anchored[0]
    for existing in matches:
        if existing["orientation"] != candidate["orientation"]:
            return True
    return False


def recover_short_wall_segments(clean, building_bbox, existing_segments):
    horizontal_mask = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (SHORT_WALL_KERNEL, 1)))
    vertical_mask = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, SHORT_WALL_KERNEL)))

    short_horizontal = filter_segments_to_bbox(
        merge_horizontal_segments(
            extract_line_segments(horizontal_mask, "horizontal", min_length=SHORT_WALL_MIN_LENGTH),
            gap_tolerance=2,
        ),
        building_bbox,
    )
    short_vertical = filter_segments_to_bbox(
        merge_vertical_segments(
            extract_line_segments(vertical_mask, "vertical", min_length=SHORT_WALL_MIN_LENGTH),
            gap_tolerance=2,
        ),
        building_bbox,
    )

    recovered = []
    for candidate in short_horizontal + short_vertical:
        if segment_length(candidate) >= LINE_LENGTH_THRESHOLD:
            continue
        if _segment_overlaps_existing(candidate, existing_segments + recovered, SHORT_WALL_CONNECTION_TOLERANCE):
            continue
        if not _anchored_short_segment(candidate, existing_segments, SHORT_WALL_CONNECTION_TOLERANCE) and not _is_perpendicular_bump(candidate, existing_segments, SHORT_WALL_CONNECTION_TOLERANCE):
            continue
        recovered.append(candidate)

    return recovered


def draw_wall_segments(shape, wall_segments):
    canvas = np.zeros(shape[:2], dtype=np.uint8)

    for segment in wall_segments:
        start, end = segment_start_end(segment)
        cv2.line(canvas, tuple(start), tuple(end), 255, max(3, int(segment["thickness_px"])))

    return cv2.dilate(canvas, np.ones((3, 3), np.uint8), iterations=1)


def largest_building_bbox(combined_mask, image_shape):
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image_shape[:2]
    best_bbox = [0, 0, width - 1, height - 1]
    best_area = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < best_area:
            continue
        if w < width * 0.4 or h < height * 0.4:
            continue
        if y > height * 0.25:
            continue
        best_area = area
        best_bbox = [x, y, x + w - 1, y + h - 1]

    return best_bbox


def filter_segments_to_bbox(segments, bbox, padding=10):
    filtered = []
    x1, y1, x2, y2 = bbox

    for segment in segments:
        start, end = segment_start_end(segment)
        center = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        if bbox_contains_point((x1, y1, x2, y2), center, padding=padding):
            filtered.append(segment)

    return filtered


def classify_walls(segments, building_bbox, starting_index=1):
    x1, y1, x2, y2 = building_bbox
    classified = []
    edge_tolerance = 18

    for index, segment in enumerate(segments, start=starting_index):
        start, end = segment_start_end(segment)
        wall_type = "partition"
        length = segment_length(segment)

        if segment["orientation"] == "horizontal":
            if (abs(segment["y"] - y1) <= edge_tolerance or abs(segment["y"] - y2) <= edge_tolerance) and length >= (x2 - x1) * 0.35:
                wall_type = "outer"
        else:
            if (abs(segment["x"] - x1) <= edge_tolerance or abs(segment["x"] - x2) <= edge_tolerance) and length >= (y2 - y1) * 0.35:
                wall_type = "outer"

        classified.append(
            {
                "id": f"w{index}",
                "orientation": segment["orientation"],
                "start_px": start,
                "end_px": end,
                "thickness_px": segment["thickness_px"],
                "type": wall_type,
            }
        )

    return classified


def prepare_clean_binary(gray, text_mask, symbol_mask=None):
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    clean = thresh.copy()
    clean[text_mask > 0] = 0
    if symbol_mask is not None:
        clean[symbol_mask > 0] = 0
    return clean


def _extract_segment_group(clean, building_bbox, h_kernel, v_kernel, gap_tolerance):
    horizontal_mask = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel, 1)))
    vertical_mask = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel)))

    horizontal_segments = filter_segments_to_bbox(
        merge_horizontal_segments(extract_line_segments(horizontal_mask, "horizontal"), gap_tolerance),
        building_bbox,
    )
    vertical_segments = filter_segments_to_bbox(
        merge_vertical_segments(extract_line_segments(vertical_mask, "vertical"), gap_tolerance),
        building_bbox,
    )
    wall_segments = horizontal_segments + vertical_segments
    wall_segments.extend(recover_short_wall_segments(clean, building_bbox, wall_segments))
    wall_mask = draw_wall_segments(clean.shape, wall_segments)

    return {
        "horizontal_mask": horizontal_mask,
        "vertical_mask": vertical_mask,
        "segments": wall_segments,
        "wall_mask": wall_mask,
    }


def extract_thin_line_masks(clean):
    horizontal = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (THIN_HORIZONTAL_KERNEL, 1)))
    vertical = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, THIN_VERTICAL_KERNEL)))
    return {"horizontal": horizontal, "vertical": vertical}


def extract_wall_layer(gray, text_mask, symbol_mask=None):
    base_clean = prepare_clean_binary(gray, text_mask)
    structural_combined = cv2.bitwise_or(
        cv2.morphologyEx(base_clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (HORIZONTAL_WALL_KERNEL, 1))),
        cv2.morphologyEx(base_clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, VERTICAL_WALL_KERNEL))),
    )
    structural_combined = cv2.dilate(structural_combined, np.ones((3, 3), np.uint8), iterations=1)
    building_bbox = largest_building_bbox(structural_combined, gray.shape)

    structural = _extract_segment_group(base_clean, building_bbox, HORIZONTAL_WALL_KERNEL, VERTICAL_WALL_KERNEL, DOOR_GAP_TOLERANCE)

    export_clean = prepare_clean_binary(gray, text_mask, symbol_mask=symbol_mask)
    export_group = _extract_segment_group(export_clean, building_bbox, HORIZONTAL_WALL_KERNEL, VERTICAL_WALL_KERNEL, EXPORT_GAP_TOLERANCE)
    thin_masks = extract_thin_line_masks(export_clean)

    return {
        "building_bbox": building_bbox,
        "clean_binary": base_clean,
        "export_clean_binary": export_clean,
        "structural_wall_mask": structural["wall_mask"],
        "structural_walls": classify_walls(structural["segments"], building_bbox),
        "structural_horizontal_mask": structural["horizontal_mask"],
        "structural_vertical_mask": structural["vertical_mask"],
        "export_wall_mask": export_group["wall_mask"],
        "export_walls": classify_walls(export_group["segments"], building_bbox),
        "export_horizontal_mask": export_group["horizontal_mask"],
        "export_vertical_mask": export_group["vertical_mask"],
        "thin_horizontal_mask": thin_masks["horizontal"],
        "thin_vertical_mask": thin_masks["vertical"],
        "symbol_mask": symbol_mask if symbol_mask is not None else np.zeros_like(base_clean),
    }


def format_walls_for_frontend(walls, meters_per_pixel, default_wall_thickness, wall_height):
    formatted = []
    for wall in walls:
        formatted.append(
            {
                "id": wall["id"],
                "start": scale_point(wall["start_px"], meters_per_pixel),
                "end": scale_point(wall["end_px"], meters_per_pixel),
                "thickness": default_wall_thickness,
                "height": wall_height,
                "type": wall["type"],
            }
        )
    return formatted
