import math

import cv2
import numpy as np

from parser.config import (
    DOOR_ARC_MIN_LENGTH,
    DOOR_ARC_SEARCH_PADDING,
    DEFAULT_DOOR_HEIGHT,
    DEFAULT_SILL_HEIGHT,
    DEFAULT_WINDOW_HEIGHT,
    DOOR_DUPLICATE_OFFSET_TOLERANCE_M,
    DOOR_LEAF_MAX_LENGTH,
    DOOR_LEAF_MIN_LENGTH,
    DOOR_LEAF_WALL_DISTANCE,
    DOOR_MAX_RADIUS,
    DOOR_MAX_WIDTH_M,
    DOOR_MIN_RADIUS,
    DOOR_MIN_WIDTH_M,
    DOOR_SUPPRESSION_PADDING,
    WINDOW_MAX_LENGTH,
    WINDOW_MASK_ORTHO_PADDING,
    WINDOW_MAX_THICKNESS,
    WINDOW_MIN_LENGTH,
    WINDOW_SUPPRESSION_PADDING,
    WINDOW_WALL_DISTANCE,
)
from parser.geometry import (
    entry_zone,
    interval_overlap,
    nearest_wall,
    project_offset_on_wall,
    wall_axis_bounds,
    wall_near_perimeter,
)
from parser.ocr import point_near_any_label


def _candidate_swing(point, wall):
    if wall["orientation"] == "horizontal":
        return "left" if point[0] <= (wall["start_px"][0] + wall["end_px"][0]) / 2.0 else "right"
    return "left" if point[1] <= (wall["start_px"][1] + wall["end_px"][1]) / 2.0 else "right"


def _door_width_m(radius_px, meters_per_pixel):
    # A floorplan door arc radius is usually close to the clear opening width.
    return round(radius_px * meters_per_pixel, 3)


def _line_orientation(line):
    x1, y1, x2, y2 = line
    return "horizontal" if abs(x2 - x1) >= abs(y2 - y1) else "vertical"


def _line_length(line):
    x1, y1, x2, y2 = line
    return math.hypot(x2 - x1, y2 - y1)


def _endpoints(line):
    return (line[0], line[1]), (line[2], line[3])


def _bbox_from_line(line, padding=0):
    x1, y1 = line[0], line[1]
    x2, y2 = line[2], line[3]
    return [min(x1, x2) - padding, min(y1, y2) - padding, max(x1, x2) + padding, max(y1, y2) + padding]


def _point_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _endpoint_wall_distance(point, wall):
    if wall["orientation"] == "horizontal":
        return abs(point[1] - wall["start_px"][1])
    return abs(point[0] - wall["start_px"][0])


def _line_center(line):
    return ((line[0] + line[2]) // 2, (line[1] + line[3]) // 2)


def _window_like_boxes(windows):
    return [{"bbox_px": window["bbox_px"]} for window in windows]


def _build_door_symbol_mask(wall_layer, text_mask, windows):
    symbol = wall_layer["clean_binary"].copy()
    symbol[text_mask > 0] = 0
    wall_mask = cv2.dilate(wall_layer["structural_wall_mask"], np.ones((5, 5), np.uint8), iterations=1)
    symbol[wall_mask > 0] = 0
    for window in windows:
        x1, y1, x2, y2 = window["bbox_px"]
        symbol[max(0, y1 - 8) : min(symbol.shape[0], y2 + 9), max(0, x1 - 8) : min(symbol.shape[1], x2 + 9)] = 0
    symbol = cv2.morphologyEx(symbol, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return symbol


def _build_door_edge_mask(gray, text_mask, wall_layer, windows):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 35, 120)
    edges[text_mask > 0] = 0
    wall_mask = cv2.dilate(wall_layer["structural_wall_mask"], np.ones((5, 5), np.uint8), iterations=1)
    edges[wall_mask > 0] = 0
    for window in windows:
        x1, y1, x2, y2 = window["bbox_px"]
        edges[max(0, y1 - 8) : min(edges.shape[0], y2 + 9), max(0, x1 - 8) : min(edges.shape[1], x2 + 9)] = 0
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return edges


def _prepare_door_component_mask(symbol_mask, edge_mask=None):
    prepared = symbol_mask.copy()
    if edge_mask is not None:
        prepared = cv2.bitwise_or(prepared, edge_mask)
    prepared = cv2.morphologyEx(prepared, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    prepared = cv2.dilate(prepared, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    prepared = cv2.morphologyEx(prepared, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    return prepared


def _door_symbol_support_mask(wall_layer, windows):
    symbol = wall_layer["clean_binary"].copy()
    wall_mask = cv2.dilate(wall_layer["structural_wall_mask"], np.ones((5, 5), np.uint8), iterations=1)
    symbol[wall_mask > 0] = 0
    for window in windows:
        x1, y1, x2, y2 = window["bbox_px"]
        symbol[max(0, y1 - 8) : min(symbol.shape[0], y2 + 9), max(0, x1 - 8) : min(symbol.shape[1], x2 + 9)] = 0
    return symbol


def _roi_nonzero(mask, bbox):
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(mask.shape[1] - 1, x2)
    y2 = min(mask.shape[0] - 1, y2)
    if x2 < x1 or y2 < y1:
        return 0
    return int(np.count_nonzero(mask[y1 : y2 + 1, x1 : x2 + 1]))


def _crop_mask(mask, bbox):
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(mask.shape[1] - 1, x2)
    y2 = min(mask.shape[0] - 1, y2)
    if x2 < x1 or y2 < y1:
        return None, [x1, y1, x1, y1]
    return mask[y1 : y2 + 1, x1 : x2 + 1], [x1, y1, x2, y2]


def _zone_bbox(center, radius, image_shape):
    return [
        max(0, int(center[0] - radius)),
        max(0, int(center[1] - radius)),
        min(image_shape[1] - 1, int(center[0] + radius)),
        min(image_shape[0] - 1, int(center[1] + radius)),
    ]


def _detect_door_lines(symbol_mask, walls, ocr_items):
    lines = cv2.HoughLinesP(symbol_mask, 1, np.pi / 180.0, threshold=10, minLineLength=DOOR_LEAF_MIN_LENGTH, maxLineGap=6)
    if lines is None:
        return []

    candidates = []
    seen = set()
    for raw in lines[:, 0, :]:
        line = [int(value) for value in raw]
        orientation = _line_orientation(line)
        length = _line_length(line)
        if length < DOOR_LEAF_MIN_LENGTH or length > DOOR_LEAF_MAX_LENGTH * 1.6:
            continue

        center = _line_center(line)
        if point_near_any_label(center, ocr_items, padding=16):
            continue

        wall = nearest_wall(center, walls, allowed_types={"outer", "partition"}, max_distance=DOOR_LEAF_WALL_DISTANCE + 10)
        if wall is None or orientation == wall["orientation"]:
            continue

        end_a, end_b = _endpoints(line)
        dist_a = _endpoint_wall_distance(end_a, wall)
        dist_b = _endpoint_wall_distance(end_b, wall)
        hinge = end_a if dist_a <= dist_b else end_b
        free_end = end_b if hinge == end_a else end_a
        hinge_distance = min(dist_a, dist_b)
        if hinge_distance > DOOR_LEAF_WALL_DISTANCE:
            continue

        key = (wall["id"], round(center[0] / 8), round(center[1] / 8), orientation)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "line_px": line,
                "wall": wall,
                "hinge_px": hinge,
                "free_end_px": free_end,
                "length_px": length,
                "center_px": [center[0], center[1]],
                "bbox_px": _bbox_from_line(line, padding=4),
            }
        )

    return candidates


def _detect_door_arcs(symbol_mask, ocr_items):
    contours, _ = cv2.findContours(symbol_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    arcs = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, False)
        if perimeter < DOOR_ARC_MIN_LENGTH:
            continue
        points = contour[:, 0, :]
        if len(points) < 8:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if max(w, h) < DOOR_MIN_RADIUS or max(w, h) > DOOR_MAX_RADIUS * 3:
            continue
        bbox = [x, y, x + w - 1, y + h - 1]
        center = [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)]
        if point_near_any_label(center, ocr_items, padding=16):
            continue

        start = tuple(points[0])
        end = tuple(points[-1])
        chord = max(1.0, _point_distance(start, end))
        curvature_ratio = perimeter / chord
        if curvature_ratio < 1.15:
            continue

        arcs.append(
            {
                "points": points,
                "bbox_px": bbox,
                "center_px": center,
                "perimeter_px": float(perimeter),
                "curvature_ratio": float(curvature_ratio),
            }
        )
    return arcs


def _arc_matches_line(arc, line_candidate):
    hinge = np.array(line_candidate["hinge_px"])
    free_end = np.array(line_candidate["free_end_px"])
    points = arc["points"]
    if len(points) < 6:
        return None

    distances_to_hinge = np.sqrt(((points - hinge) ** 2).sum(axis=1))
    near_hinge = points[distances_to_hinge <= max(10.0, line_candidate["length_px"] * 0.35)]
    if len(near_hinge) == 0:
        return None

    distances_to_free = np.sqrt(((points - free_end) ** 2).sum(axis=1))
    near_free = points[distances_to_free <= max(14.0, line_candidate["length_px"] * 0.55)]
    if len(near_free) == 0:
        return None

    radius_px = float(np.median(distances_to_hinge))
    if radius_px < DOOR_MIN_RADIUS or radius_px > DOOR_MAX_RADIUS * 1.6:
        return None

    angles = np.degrees(np.arctan2(points[:, 1] - hinge[1], points[:, 0] - hinge[0]))
    angle_span = float(angles.max() - angles.min())
    if angle_span < 20.0:
        return None

    return {
        "radius_px": radius_px,
        "angle_span_deg": angle_span,
        "score": angle_span + len(near_free) + len(near_hinge),
    }


def _arc_wall_contacts(arc, wall):
    points = arc["points"]
    tolerance = max(8, DOOR_LEAF_WALL_DISTANCE + 10)
    if wall["orientation"] == "horizontal":
        wall_axis = wall["start_px"][1]
        near_wall = points[np.abs(points[:, 1] - wall_axis) <= tolerance]
        axis_values = near_wall[:, 0] if len(near_wall) else np.array([])
    else:
        wall_axis = wall["start_px"][0]
        near_wall = points[np.abs(points[:, 0] - wall_axis) <= tolerance]
        axis_values = near_wall[:, 1] if len(near_wall) else np.array([])

    if len(axis_values) < 2:
        return None

    return {
        "points": near_wall,
        "axis_min": int(axis_values.min()),
        "axis_max": int(axis_values.max()),
    }


def _build_direct_door(line_candidate, arc_match, meters_per_pixel):
    wall = line_candidate["wall"]
    hinge = tuple(line_candidate["hinge_px"])
    free_end = tuple(line_candidate["free_end_px"])
    center = [int(round((hinge[0] + free_end[0]) / 2.0)), int(round((hinge[1] + free_end[1]) / 2.0))]
    width_px = max(line_candidate["length_px"], arc_match["radius_px"])
    if arc_match.get("wall_span_px"):
        width_px = max(width_px, arc_match["wall_span_px"])
    width_m = round(width_px * meters_per_pixel, 3)
    if width_m < max(0.35, DOOR_MIN_WIDTH_M * 0.55) or width_m > DOOR_MAX_WIDTH_M:
        return None

    bbox = _bbox_from_line(line_candidate["line_px"], padding=DOOR_ARC_SEARCH_PADDING)
    return {
        "center_px": center,
        "radius_px": int(round(arc_match["radius_px"])),
        "wallId": wall["id"],
        "offset": round(project_offset_on_wall(hinge, wall) * meters_per_pixel, 3),
        "width": width_m,
        "height": DEFAULT_DOOR_HEIGHT,
        "swing": _candidate_swing(free_end, wall),
        "bbox_px": bbox,
        "leaf_line_px": line_candidate["line_px"],
        "hinge_px": [int(hinge[0]), int(hinge[1])],
        "score": arc_match["score"],
        "detector": "direct_symbol",
    }


def _extract_leaf_candidates(gray, text_mask, wall_layer, windows):
    masked = gray.copy()
    masked[text_mask > 0] = 255
    edges = cv2.Canny(masked, 50, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=18, minLineLength=DOOR_LEAF_MIN_LENGTH, maxLineGap=6)
    window_boxes = _window_like_boxes(windows)
    candidates = []
    seen = []

    raw_lines = []
    if lines is not None:
        raw_lines.extend([[int(value) for value in raw] for raw in lines[:, 0, :]])

    for orientation, mask in (("horizontal", wall_layer["thin_horizontal_mask"]), ("vertical", wall_layer["thin_vertical_mask"])):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if orientation == "horizontal":
                raw_lines.append([x, y + h // 2, x + w - 1, y + h // 2])
            else:
                raw_lines.append([x + w // 2, y, x + w // 2, y + h - 1])

    for line in raw_lines:
        orientation = _line_orientation(line)
        length = _line_length(line)
        if length < DOOR_LEAF_MIN_LENGTH or length > DOOR_LEAF_MAX_LENGTH:
            continue
        bbox = _bbox_from_line(line, padding=4)
        if _overlaps_any_window(bbox, window_boxes):
            continue

        center = ((line[0] + line[2]) // 2, (line[1] + line[3]) // 2)
        nearest = nearest_wall(center, wall_layer["structural_walls"], allowed_types={"outer", "partition"}, max_distance=DOOR_LEAF_WALL_DISTANCE)
        if nearest is None:
            continue
        if orientation == nearest["orientation"]:
            continue
        if point_near_any_label(center, wall_layer.get("labels_for_debug", []), padding=14):
            continue

        end_a, end_b = _endpoints(line)
        wall_distance_a = abs(project_offset_on_wall(end_a, nearest) - project_offset_on_wall(center, nearest))
        wall_distance_b = abs(project_offset_on_wall(end_b, nearest) - project_offset_on_wall(center, nearest))
        hinge_point = end_a if wall_distance_a <= wall_distance_b else end_b
        free_end = end_b if hinge_point == end_a else end_a

        if nearest["orientation"] == "horizontal":
            wall_distance = abs(hinge_point[1] - nearest["start_px"][1])
        else:
            wall_distance = abs(hinge_point[0] - nearest["start_px"][0])
        if wall_distance > DOOR_LEAF_WALL_DISTANCE:
            continue

        key = (nearest["id"], round(center[0] / 8), round(center[1] / 8))
        if key in seen:
            continue
        seen.append(key)
        candidates.append(
            {
                "line_px": line,
                "bbox_px": bbox,
                "length_px": length,
                "orientation": orientation,
                "wall": nearest,
                "hinge_px": hinge_point,
                "free_end_px": free_end,
                "center_px": [center[0], center[1]],
            }
        )

    return candidates


def _extract_leaf_candidates_in_zone(symbol_mask, zone, wall):
    roi, roi_bbox = _crop_mask(symbol_mask, zone["bbox_px"])
    if roi is None or roi.size == 0:
        return []

    lines = cv2.HoughLinesP(roi, 1, np.pi / 180.0, threshold=10, minLineLength=DOOR_LEAF_MIN_LENGTH, maxLineGap=6)
    candidates = []
    seen = set()

    if lines is not None:
        raw_lines = [[int(value) for value in raw] for raw in lines[:, 0, :]]
    else:
        raw_lines = []

    for raw_line in raw_lines:
        line = [
            raw_line[0] + roi_bbox[0],
            raw_line[1] + roi_bbox[1],
            raw_line[2] + roi_bbox[0],
            raw_line[3] + roi_bbox[1],
        ]
        orientation = _line_orientation(line)
        length = _line_length(line)
        if length < DOOR_LEAF_MIN_LENGTH or length > DOOR_LEAF_MAX_LENGTH:
            continue
        if orientation == wall["orientation"]:
            continue

        center = ((line[0] + line[2]) // 2, (line[1] + line[3]) // 2)
        if _point_distance(center, zone["center_px"]) > max(28, zone["width_px"] * 0.8):
            continue

        end_a, end_b = _endpoints(line)
        if wall["orientation"] == "horizontal":
            dist_a = abs(end_a[1] - wall["start_px"][1])
            dist_b = abs(end_b[1] - wall["start_px"][1])
        else:
            dist_a = abs(end_a[0] - wall["start_px"][0])
            dist_b = abs(end_b[0] - wall["start_px"][0])

        hinge_point = end_a if dist_a <= dist_b else end_b
        free_end = end_b if hinge_point == end_a else end_a
        wall_distance = min(dist_a, dist_b)
        if wall_distance > DOOR_LEAF_WALL_DISTANCE:
            continue

        key = (round(center[0] / 6), round(center[1] / 6), orientation)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "line_px": line,
                "bbox_px": _bbox_from_line(line, padding=4),
                "length_px": length,
                "orientation": orientation,
                "wall": wall,
                "hinge_px": hinge_point,
                "free_end_px": free_end,
                "center_px": [center[0], center[1]],
                "zone_kind": zone["kind"],
                "zone_width_px": zone["width_px"],
                "debug_zone_bbox": zone["bbox_px"],
            }
        )

    candidates.sort(key=lambda item: (-item["length_px"], _point_distance(item["center_px"], zone["center_px"])))
    return candidates


def _has_arc_support(gray, text_mask, candidate):
    x1, y1, x2, y2 = candidate["bbox_px"]
    pad = DOOR_ARC_SEARCH_PADDING
    roi_bbox = [
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(gray.shape[1] - 1, x2 + pad),
        min(gray.shape[0] - 1, y2 + pad),
    ]
    roi = gray[roi_bbox[1] : roi_bbox[3] + 1, roi_bbox[0] : roi_bbox[2] + 1].copy()
    roi_text = text_mask[roi_bbox[1] : roi_bbox[3] + 1, roi_bbox[0] : roi_bbox[2] + 1]
    roi[roi_text > 0] = 255
    edges = cv2.Canny(roi, 50, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    hinge = (candidate["hinge_px"][0] - roi_bbox[0], candidate["hinge_px"][1] - roi_bbox[1])
    free_end = (candidate["free_end_px"][0] - roi_bbox[0], candidate["free_end_px"][1] - roi_bbox[1])
    leaf_bbox = [
        candidate["bbox_px"][0] - roi_bbox[0],
        candidate["bbox_px"][1] - roi_bbox[1],
        candidate["bbox_px"][2] - roi_bbox[0],
        candidate["bbox_px"][3] - roi_bbox[1],
    ]

    for contour in contours:
        arc_length = cv2.arcLength(contour, False)
        if arc_length < DOOR_ARC_MIN_LENGTH:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        contour_bbox = [bx, by, bx + bw - 1, by + bh - 1]
        if interval_overlap(contour_bbox[0], contour_bbox[2], leaf_bbox[0], leaf_bbox[2]) > bw * 0.7 and interval_overlap(contour_bbox[1], contour_bbox[3], leaf_bbox[1], leaf_bbox[3]) > bh * 0.7:
            continue

        points = contour[:, 0, :]
        if len(points) < 8:
            continue
        distances_to_hinge = np.sqrt((points[:, 0] - hinge[0]) ** 2 + (points[:, 1] - hinge[1]) ** 2)
        close_to_hinge = np.count_nonzero(distances_to_hinge <= max(8, candidate["length_px"] * 0.25))
        if close_to_hinge == 0:
            continue
        distances_to_free = np.sqrt((points[:, 0] - free_end[0]) ** 2 + (points[:, 1] - free_end[1]) ** 2)
        if np.count_nonzero(distances_to_free <= max(10, candidate["length_px"] * 0.35)) == 0:
            continue

        return True

    return False


def _has_arc_support_in_zone(symbol_mask, candidate, zone):
    roi, roi_bbox = _crop_mask(symbol_mask, zone["bbox_px"])
    if roi is None or roi.size == 0:
        return False

    contours, _ = cv2.findContours(roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    hinge = (candidate["hinge_px"][0] - roi_bbox[0], candidate["hinge_px"][1] - roi_bbox[1])
    free_end = (candidate["free_end_px"][0] - roi_bbox[0], candidate["free_end_px"][1] - roi_bbox[1])
    leaf_bbox = [
        candidate["bbox_px"][0] - roi_bbox[0],
        candidate["bbox_px"][1] - roi_bbox[1],
        candidate["bbox_px"][2] - roi_bbox[0],
        candidate["bbox_px"][3] - roi_bbox[1],
    ]

    for contour in contours:
        arc_length = cv2.arcLength(contour, False)
        if arc_length < DOOR_ARC_MIN_LENGTH:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        contour_bbox = [bx, by, bx + bw - 1, by + bh - 1]
        if interval_overlap(contour_bbox[0], contour_bbox[2], leaf_bbox[0], leaf_bbox[2]) > bw * 0.7 and interval_overlap(contour_bbox[1], contour_bbox[3], leaf_bbox[1], leaf_bbox[3]) > bh * 0.7:
            continue
        points = contour[:, 0, :]
        if len(points) < 8:
            continue
        dist_hinge = np.sqrt((points[:, 0] - hinge[0]) ** 2 + (points[:, 1] - hinge[1]) ** 2)
        dist_free = np.sqrt((points[:, 0] - free_end[0]) ** 2 + (points[:, 1] - free_end[1]) ** 2)
        if np.count_nonzero(dist_hinge <= max(8, candidate["length_px"] * 0.25)) == 0:
            continue
        if np.count_nonzero(dist_free <= max(10, candidate["length_px"] * 0.45)) == 0:
            continue
        return True

    return False


def _root_distance_to_wall(point, wall):
    if wall["orientation"] == "horizontal":
        return abs(point[1] - wall["start_px"][1])
    return abs(point[0] - wall["start_px"][0])


def _collect_component_pixels(symbol_mask, bbox):
    roi, roi_bbox = _crop_mask(symbol_mask, bbox)
    if roi is None or roi.size == 0:
        return None, None, None
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    return roi, roi_bbox, (component_count, labels, stats)


def _component_points(component_labels, component_id, roi_bbox):
    ys, xs = np.where(component_labels == component_id)
    if len(xs) == 0:
        return None
    points = np.stack([xs + roi_bbox[0], ys + roi_bbox[1]], axis=1)
    return points


def _fit_fin_line(points):
    if points is None or len(points) < 8:
        return None
    pts = points.astype(np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    direction = np.array([float(vx[0]), float(vy[0])])
    direction = direction / (np.linalg.norm(direction) + 1e-6)
    center = np.array([float(x0[0]), float(y0[0])])
    projections = np.dot(points - center, direction)
    start = center + direction * projections.min()
    end = center + direction * projections.max()
    line = [int(round(start[0])), int(round(start[1])), int(round(end[0])), int(round(end[1]))]
    return line


def _curve_support_from_root(points, hinge_point, fin_length_px):
    if points is None or len(points) < 8:
        return None
    deltas = points - np.array(hinge_point)
    distances = np.sqrt((deltas[:, 0] ** 2) + (deltas[:, 1] ** 2))
    if len(distances) == 0:
        return None
    outer = distances >= max(8.0, fin_length_px * 0.45)
    if np.count_nonzero(outer) < 6:
        return None
    outer_points = points[outer]
    outer_distances = distances[outer]
    radius_px = float(np.median(outer_distances))
    if radius_px < DOOR_MIN_RADIUS or radius_px > DOOR_MAX_RADIUS * 1.5:
        return None
    radius_band = np.abs(outer_distances - radius_px) <= max(6.0, fin_length_px * 0.18)
    arc_points = outer_points[radius_band]
    if len(arc_points) < 6:
        return None
    angles = np.degrees(np.arctan2(arc_points[:, 1] - hinge_point[1], arc_points[:, 0] - hinge_point[0]))
    angle_span = float(angles.max() - angles.min())
    if angle_span < 18.0:
        return None
    return {
        "radius_px": radius_px,
        "angle_span_deg": angle_span,
        "point_count": int(len(arc_points)),
        "arc_points": arc_points,
    }


def _component_bbox(points, padding=0):
    x1 = int(points[:, 0].min()) - padding
    y1 = int(points[:, 1].min()) - padding
    x2 = int(points[:, 0].max()) + padding
    y2 = int(points[:, 1].max()) + padding
    return [x1, y1, x2, y2]


def _component_axis_span(points, orientation):
    if orientation == "horizontal":
        return float(points[:, 0].max() - points[:, 0].min())
    return float(points[:, 1].max() - points[:, 1].min())


def _component_perp_span(points, orientation):
    if orientation == "horizontal":
        return float(points[:, 1].max() - points[:, 1].min())
    return float(points[:, 0].max() - points[:, 0].min())


def _extract_component_line_candidates(component_mask, bbox_px, wall):
    roi, roi_bbox = _crop_mask(component_mask, bbox_px)
    if roi is None or roi.size == 0:
        return []

    lines = cv2.HoughLinesP(roi, 1, np.pi / 180.0, threshold=8, minLineLength=DOOR_LEAF_MIN_LENGTH - 4, maxLineGap=8)
    if lines is None:
        return []

    candidates = []
    seen = set()
    for raw in lines[:, 0, :]:
        line = [
            int(raw[0] + roi_bbox[0]),
            int(raw[1] + roi_bbox[1]),
            int(raw[2] + roi_bbox[0]),
            int(raw[3] + roi_bbox[1]),
        ]
        orientation = _line_orientation(line)
        if orientation == wall["orientation"]:
            continue
        length = _line_length(line)
        if length < DOOR_LEAF_MIN_LENGTH - 4 or length > DOOR_LEAF_MAX_LENGTH * 1.5:
            continue
        end_a, end_b = _endpoints(line)
        dist_a = _root_distance_to_wall(end_a, wall)
        dist_b = _root_distance_to_wall(end_b, wall)
        hinge = end_a if dist_a <= dist_b else end_b
        free_end = end_b if hinge == end_a else end_a
        hinge_distance = min(dist_a, dist_b)
        if hinge_distance > DOOR_LEAF_WALL_DISTANCE + 8:
            continue
        key = (round((line[0] + line[2]) / 12), round((line[1] + line[3]) / 12), orientation)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "line_px": line,
                "orientation": orientation,
                "length_px": float(length),
                "hinge_px": hinge,
                "free_end_px": free_end,
                "hinge_distance_px": float(hinge_distance),
                "bbox_px": _bbox_from_line(line, padding=4),
            }
        )

    candidates.sort(key=lambda item: (item["hinge_distance_px"], -item["length_px"]))
    return candidates


def _collect_wall_adjacent_components(symbol_mask, walls, ocr_items, edge_mask=None):
    prepared = _prepare_door_component_mask(symbol_mask, edge_mask=edge_mask)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(prepared, connectivity=8)
    components = []

    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < 14:
            continue

        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        bbox = [x, y, x + w - 1, y + h - 1]
        center = [x + w // 2, y + h // 2]
        if point_near_any_label(center, ocr_items, padding=18):
            continue

        wall = nearest_wall(center, walls, allowed_types={"outer", "partition"}, max_distance=DOOR_LEAF_WALL_DISTANCE + 28)
        if wall is None:
            continue

        points = _component_points(labels, component_id, [0, 0, prepared.shape[1] - 1, prepared.shape[0] - 1])
        if points is None or len(points) < 10:
            continue

        span_along_wall = _component_axis_span(points, wall["orientation"])
        if span_along_wall < DOOR_LEAF_MIN_LENGTH - 4:
            continue

        components.append(
            {
                "bbox_px": bbox,
                "center_px": center,
                "wall": wall,
                "points": points,
                "area": area,
            }
        )

    return prepared, components


def _build_component_door(component, component_mask, meters_per_pixel):
    wall = component["wall"]
    line_candidates = _extract_component_line_candidates(component_mask, component["bbox_px"], wall)
    if not line_candidates:
        return None

    best = None
    for line_candidate in line_candidates:
        curve = _curve_support_from_root(component["points"], line_candidate["hinge_px"], line_candidate["length_px"])
        if curve is None:
            continue

        wall_span_px = _component_axis_span(component["points"], wall["orientation"])
        width_px = max(line_candidate["length_px"], curve["radius_px"], wall_span_px * 0.9)
        width_m = round(float(width_px) * meters_per_pixel, 3)
        if width_m < max(0.4, DOOR_MIN_WIDTH_M * 0.5) or width_m > DOOR_MAX_WIDTH_M:
            continue

        center = [
            int(round((line_candidate["hinge_px"][0] + line_candidate["free_end_px"][0]) / 2.0)),
            int(round((line_candidate["hinge_px"][1] + line_candidate["free_end_px"][1]) / 2.0)),
        ]
        score = curve["angle_span_deg"] + curve["point_count"] + max(0.0, 32.0 - line_candidate["hinge_distance_px"])
        candidate = {
            "center_px": center,
            "radius_px": int(round(curve["radius_px"])),
            "wallId": wall["id"],
            "offset": round(project_offset_on_wall(tuple(line_candidate["hinge_px"]), wall) * meters_per_pixel, 3),
            "width": width_m,
            "height": DEFAULT_DOOR_HEIGHT,
            "swing": _candidate_swing(tuple(line_candidate["free_end_px"]), wall),
            "bbox_px": _component_bbox(component["points"], padding=6),
            "leaf_line_px": line_candidate["line_px"],
            "hinge_px": [int(line_candidate["hinge_px"][0]), int(line_candidate["hinge_px"][1])],
            "score": score,
            "debug_zone_bbox": component["bbox_px"],
            "detector": "component_symbol",
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def _wall_contact_points(points, wall, tolerance):
    if wall["orientation"] == "horizontal":
        mask = np.abs(points[:, 1] - wall["start_px"][1]) <= tolerance
        near = points[mask]
        axis_values = near[:, 0] if len(near) else np.array([])
    else:
        mask = np.abs(points[:, 0] - wall["start_px"][0]) <= tolerance
        near = points[mask]
        axis_values = near[:, 1] if len(near) else np.array([])
    return near, axis_values


def _sample_hinge_candidates(contact_points, wall):
    if len(contact_points) == 0:
        return []
    if wall["orientation"] == "horizontal":
        order = np.argsort(contact_points[:, 0])
    else:
        order = np.argsort(contact_points[:, 1])
    ordered = contact_points[order]
    picks = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    unique = []
    seen = set()
    for point in picks:
        key = (int(point[0]), int(point[1]))
        if key not in seen:
            seen.add(key)
            unique.append((int(point[0]), int(point[1])))
    return unique


def _build_arc_only_door(component, meters_per_pixel):
    wall = component["wall"]
    points = component["points"]
    contact_points, axis_values = _wall_contact_points(points, wall, DOOR_LEAF_WALL_DISTANCE + 10)
    if len(contact_points) < 3:
        return None

    best = None
    for hinge_point in _sample_hinge_candidates(contact_points, wall):
        curve = _curve_support_from_root(points, hinge_point, max(DOOR_LEAF_MIN_LENGTH, _component_axis_span(points, wall["orientation"]) * 0.7))
        if curve is None:
            continue

        width_m = round(float(curve["radius_px"]) * meters_per_pixel, 3)
        if width_m < max(0.4, DOOR_MIN_WIDTH_M * 0.5) or width_m > DOOR_MAX_WIDTH_M:
            continue

        arc_points = curve["arc_points"]
        distances = np.sqrt(((arc_points - np.array(hinge_point)) ** 2).sum(axis=1))
        tip = arc_points[int(np.argmax(distances))]
        wall_span_px = float(axis_values.max() - axis_values.min()) if len(axis_values) else 0.0
        perp_span_px = _component_perp_span(points, wall["orientation"])
        score = curve["angle_span_deg"] + curve["point_count"] + perp_span_px - wall_span_px * 0.2

        candidate = {
            "center_px": [int(round((hinge_point[0] + tip[0]) / 2.0)), int(round((hinge_point[1] + tip[1]) / 2.0))],
            "radius_px": int(round(curve["radius_px"])),
            "wallId": wall["id"],
            "offset": round(project_offset_on_wall(hinge_point, wall) * meters_per_pixel, 3),
            "width": width_m,
            "height": DEFAULT_DOOR_HEIGHT,
            "swing": _candidate_swing((int(tip[0]), int(tip[1])), wall),
            "bbox_px": _component_bbox(points, padding=6),
            "leaf_line_px": None,
            "hinge_px": [int(hinge_point[0]), int(hinge_point[1])],
            "score": score,
            "debug_zone_bbox": component["bbox_px"],
            "detector": "arc_only",
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def _build_fin_door_from_component(points, wall, zone, meters_per_pixel):
    fin_line = _fit_fin_line(points)
    if fin_line is None:
        return None
    fin_orientation = _line_orientation(fin_line)
    if fin_orientation == wall["orientation"]:
        return None

    end_a, end_b = _endpoints(fin_line)
    dist_a = _root_distance_to_wall(end_a, wall)
    dist_b = _root_distance_to_wall(end_b, wall)
    hinge_point = end_a if dist_a <= dist_b else end_b
    free_end = end_b if hinge_point == end_a else end_a
    fin_length_px = _line_length(fin_line)
    if fin_length_px < DOOR_LEAF_MIN_LENGTH or fin_length_px > DOOR_LEAF_MAX_LENGTH * 1.4:
        return None
    if min(dist_a, dist_b) > DOOR_LEAF_WALL_DISTANCE:
        return None

    curve = _curve_support_from_root(points, hinge_point, fin_length_px)
    if curve is None:
        return None

    estimated_width_px = max(fin_length_px, curve["radius_px"], zone["width_px"])
    width_m = round(float(estimated_width_px) * meters_per_pixel, 3)
    if width_m < DOOR_MIN_WIDTH_M or width_m > DOOR_MAX_WIDTH_M:
        return None

    center = [int(round((hinge_point[0] + free_end[0]) / 2.0)), int(round((hinge_point[1] + free_end[1]) / 2.0))]
    bbox = _component_bbox(points, padding=8)
    return {
        "center_px": center,
        "radius_px": int(round(curve["radius_px"])),
        "wallId": wall["id"],
        "offset": round(project_offset_on_wall(tuple(center), wall) * meters_per_pixel, 3),
        "width": width_m,
        "height": DEFAULT_DOOR_HEIGHT,
        "swing": _candidate_swing(tuple(free_end), wall),
        "bbox_px": bbox,
        "leaf_line_px": fin_line,
        "hinge_px": [int(hinge_point[0]), int(hinge_point[1])],
        "score": curve["angle_span_deg"] + curve["point_count"],
        "debug_zone_bbox": zone["bbox_px"],
        "detector": "fin",
    }


def _detect_fin_door_in_zone(symbol_mask, zone, wall, meters_per_pixel):
    roi, roi_bbox, component_data = _collect_component_pixels(symbol_mask, zone["bbox_px"])
    if roi is None or component_data is None:
        return None
    component_count, labels, stats = component_data
    best = None

    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < 12:
            continue
        points = _component_points(labels, component_id, roi_bbox)
        if points is None:
            continue
        comp_bbox = _component_bbox(points, padding=0)
        if _point_distance(
            [(comp_bbox[0] + comp_bbox[2]) / 2.0, (comp_bbox[1] + comp_bbox[3]) / 2.0],
            zone["center_px"],
        ) > max(40, zone["width_px"]):
            continue
        candidate = _build_fin_door_from_component(points, wall, zone, meters_per_pixel)
        if candidate is None:
            continue
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def _leaf_to_door(candidate, meters_per_pixel):
    wall = candidate["wall"]
    center = tuple(candidate["center_px"])
    estimated_width_px = candidate["length_px"]
    if candidate.get("zone_kind") == "endpoint":
        estimated_width_px = max(estimated_width_px, candidate.get("zone_width_px", estimated_width_px), candidate["length_px"] * 1.35)
    elif candidate.get("zone_width_px"):
        estimated_width_px = max(estimated_width_px, candidate["zone_width_px"] * 0.75)
    width_m = round(estimated_width_px * meters_per_pixel, 3)
    if width_m < DOOR_MIN_WIDTH_M or width_m > DOOR_MAX_WIDTH_M:
        return None
    return {
        "center_px": candidate["center_px"],
        "radius_px": max(DOOR_MIN_RADIUS, int(round(candidate["length_px"] / 2.0))),
        "wallId": wall["id"],
        "offset": round(project_offset_on_wall(center, wall) * meters_per_pixel, 3),
        "width": width_m,
        "height": DEFAULT_DOOR_HEIGHT,
        "swing": _candidate_swing(center, wall),
        "bbox_px": _bbox_from_line(candidate["line_px"], padding=DOOR_ARC_SEARCH_PADDING // 2),
        "leaf_line_px": candidate["line_px"],
        "hinge_px": [candidate["hinge_px"][0], candidate["hinge_px"][1]],
        "score": 2,
        "debug_zone_bbox": candidate.get("debug_zone_bbox"),
        "detector": "blade_curve",
    }


def _collect_gap_zones(wall_layer, meters_per_pixel, windows, ocr_items):
    zones = []
    window_boxes = _window_like_boxes(windows)

    for wall in wall_layer["structural_walls"]:
        if wall["orientation"] == "horizontal":
            line_y = wall["start_px"][1]
            x1 = min(wall["start_px"][0], wall["end_px"][0])
            x2 = max(wall["start_px"][0], wall["end_px"][0])
            band = wall_layer["structural_horizontal_mask"][max(0, line_y - 2) : min(wall_layer["structural_horizontal_mask"].shape[0], line_y + 3), x1 : x2 + 1]
            occupancy = band.max(axis=0) > 0
            gaps = find_gaps(occupancy)
            for gap_start, gap_end in gaps:
                width_px = gap_end - gap_start + 1
                width_m = round(width_px * meters_per_pixel, 3)
                center = [x1 + int((gap_start + gap_end) / 2), line_y]
                bbox = [center[0] - 36, center[1] - 36, center[0] + 36, center[1] + 36]
                if width_m < DOOR_MIN_WIDTH_M or width_m > DOOR_MAX_WIDTH_M:
                    continue
                if wall_near_perimeter(wall, wall_layer["building_bbox"], tolerance=65) and not entry_zone(center, wall_layer["building_bbox"]):
                    continue
                if point_near_any_label(center, ocr_items, padding=18):
                    continue
                if _overlaps_any_window([center[0] - width_px // 2, center[1] - 18, center[0] + width_px // 2, center[1] + 18], window_boxes):
                    continue
                zones.append({"kind": "gap", "wall": wall, "center_px": center, "bbox_px": bbox, "width_px": width_px})
        else:
            line_x = wall["start_px"][0]
            y1 = min(wall["start_px"][1], wall["end_px"][1])
            y2 = max(wall["start_px"][1], wall["end_px"][1])
            band = wall_layer["structural_vertical_mask"][y1 : y2 + 1, max(0, line_x - 2) : min(wall_layer["structural_vertical_mask"].shape[1], line_x + 3)]
            occupancy = band.max(axis=1) > 0
            gaps = find_gaps(occupancy)
            for gap_start, gap_end in gaps:
                width_px = gap_end - gap_start + 1
                width_m = round(width_px * meters_per_pixel, 3)
                center = [line_x, y1 + int((gap_start + gap_end) / 2)]
                bbox = [center[0] - 36, center[1] - 36, center[0] + 36, center[1] + 36]
                if width_m < DOOR_MIN_WIDTH_M or width_m > DOOR_MAX_WIDTH_M:
                    continue
                if wall_near_perimeter(wall, wall_layer["building_bbox"], tolerance=65) and not entry_zone(center, wall_layer["building_bbox"]):
                    continue
                if point_near_any_label(center, ocr_items, padding=18):
                    continue
                if _overlaps_any_window([center[0] - 18, center[1] - width_px // 2, center[0] + 18, center[1] + width_px // 2], window_boxes):
                    continue
                zones.append({"kind": "gap", "wall": wall, "center_px": center, "bbox_px": bbox, "width_px": width_px})

    return zones


def _collect_endpoint_zones(wall_layer, meters_per_pixel, windows, ocr_items):
    zones = []
    window_boxes = _window_like_boxes(windows)
    for wall in wall_layer["structural_walls"]:
        for endpoint in (wall["start_px"], wall["end_px"]):
            center = [int(endpoint[0]), int(endpoint[1])]
            bbox = _zone_bbox(center, 44, wall_layer["clean_binary"].shape)
            if point_near_any_label(center, ocr_items, padding=18):
                continue
            if _overlaps_any_window(bbox, window_boxes):
                continue
            zones.append({"kind": "endpoint", "wall": wall, "center_px": center, "bbox_px": bbox, "width_px": int(round(0.9 / max(meters_per_pixel, 1e-6)))})
    return zones


def _collect_intersection_zones(wall_layer, meters_per_pixel, windows, ocr_items):
    zones = []
    window_boxes = _window_like_boxes(windows)
    walls = wall_layer["structural_walls"]

    for wall in walls:
        for other in walls:
            if wall["id"] >= other["id"]:
                continue
            if wall["orientation"] == other["orientation"]:
                continue

            if wall["orientation"] == "horizontal":
                hx1, hx2 = sorted((wall["start_px"][0], wall["end_px"][0]))
                vy1, vy2 = sorted((other["start_px"][1], other["end_px"][1]))
                intersection = [other["start_px"][0], wall["start_px"][1]]
                primary_wall = wall
            else:
                hx1, hx2 = sorted((other["start_px"][0], other["end_px"][0]))
                vy1, vy2 = sorted((wall["start_px"][1], wall["end_px"][1]))
                intersection = [wall["start_px"][0], other["start_px"][1]]
                primary_wall = other

            if not (hx1 - 12 <= intersection[0] <= hx2 + 12 and vy1 - 12 <= intersection[1] <= vy2 + 12):
                continue
            bbox = _zone_bbox(intersection, 52, wall_layer["clean_binary"].shape)
            if point_near_any_label(intersection, ocr_items, padding=18):
                continue
            if _overlaps_any_window(bbox, window_boxes):
                continue
            zones.append(
                {
                    "kind": "intersection",
                    "wall": primary_wall,
                    "center_px": intersection,
                    "bbox_px": bbox,
                    "width_px": int(round(0.95 / max(meters_per_pixel, 1e-6))),
                }
            )

    return zones


def _build_door_detection(circle, wall, meters_per_pixel, leaf_line=None):
    cx, cy, radius = circle
    width_m = _door_width_m(radius, meters_per_pixel)
    if width_m < DOOR_MIN_WIDTH_M or width_m > DOOR_MAX_WIDTH_M:
        return None

    center = (cx, cy)
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    if leaf_line is not None:
        bbox = [
            min(bbox[0], leaf_line[0], leaf_line[2]),
            min(bbox[1], leaf_line[1], leaf_line[3]),
            max(bbox[2], leaf_line[0], leaf_line[2]),
            max(bbox[3], leaf_line[1], leaf_line[3]),
        ]

    return {
        "center_px": [cx, cy],
        "radius_px": radius,
        "wallId": wall["id"],
        "offset": round(project_offset_on_wall(center, wall) * meters_per_pixel, 3),
        "width": width_m,
        "height": DEFAULT_DOOR_HEIGHT,
        "swing": _candidate_swing(center, wall),
        "bbox_px": bbox,
        "leaf_line_px": leaf_line,
    }


def find_gaps(occupancy, min_gap=20, max_gap=120):
    gaps = []
    start = None

    for index, value in enumerate(occupancy):
        if not value and start is None:
            start = index
        if value and start is not None:
            gap_length = index - start
            if min_gap <= gap_length <= max_gap:
                gaps.append((start, index - 1))
            start = None

    if start is not None:
        gap_length = len(occupancy) - start
        if min_gap <= gap_length <= max_gap:
            gaps.append((start, len(occupancy) - 1))

    return gaps


def _overlaps_any_window(candidate_bbox, windows):
    for window in windows:
        if interval_overlap(candidate_bbox[0], candidate_bbox[2], window["bbox_px"][0], window["bbox_px"][2]) > 8 and interval_overlap(candidate_bbox[1], candidate_bbox[3], window["bbox_px"][1], window["bbox_px"][3]) > 8:
            return True
    return False


def detect_gap_support_doors(wall_layer, meters_per_pixel, windows, ocr_items):
    detections = []
    support_mask = _door_symbol_support_mask(wall_layer, windows)

    for wall in wall_layer["structural_walls"]:
        if wall["orientation"] == "horizontal":
            line_y = wall["start_px"][1]
            x1 = min(wall["start_px"][0], wall["end_px"][0])
            x2 = max(wall["start_px"][0], wall["end_px"][0])
            band = wall_layer["structural_horizontal_mask"][max(0, line_y - 2) : min(wall_layer["structural_horizontal_mask"].shape[0], line_y + 3), x1 : x2 + 1]
            occupancy = band.max(axis=0) > 0
            gaps = find_gaps(occupancy)

            for gap_start, gap_end in gaps:
                gap_center_x = x1 + int((gap_start + gap_end) / 2)
                gap_width_px = gap_end - gap_start + 1
                center = (gap_center_x, line_y)
                bbox = [gap_center_x - gap_width_px // 2, line_y - 18, gap_center_x + gap_width_px // 2, line_y + 18]

                if wall_near_perimeter(wall, wall_layer["building_bbox"], tolerance=65) and not entry_zone(center, wall_layer["building_bbox"]):
                    continue
                if point_near_any_label(center, ocr_items, padding=18):
                    continue
                if _overlaps_any_window(bbox, windows):
                    continue
                if _roi_nonzero(support_mask, [bbox[0] - 14, bbox[1] - 14, bbox[2] + 14, bbox[3] + 14]) < 18:
                    continue

                width_m = round(gap_width_px * meters_per_pixel, 3)
                if width_m < DOOR_MIN_WIDTH_M or width_m > DOOR_MAX_WIDTH_M:
                    continue

                detections.append(
                    {
                        "center_px": [gap_center_x, line_y],
                        "radius_px": int(gap_width_px / 2),
                        "wallId": wall["id"],
                        "offset": round(project_offset_on_wall(center, wall) * meters_per_pixel, 3),
                        "width": width_m,
                        "height": DEFAULT_DOOR_HEIGHT,
                        "swing": _candidate_swing(center, wall),
                        "bbox_px": bbox,
                        "leaf_line_px": None,
                    }
                )
        else:
            line_x = wall["start_px"][0]
            y1 = min(wall["start_px"][1], wall["end_px"][1])
            y2 = max(wall["start_px"][1], wall["end_px"][1])
            band = wall_layer["structural_vertical_mask"][y1 : y2 + 1, max(0, line_x - 2) : min(wall_layer["structural_vertical_mask"].shape[1], line_x + 3)]
            occupancy = band.max(axis=1) > 0
            gaps = find_gaps(occupancy)

            for gap_start, gap_end in gaps:
                gap_center_y = y1 + int((gap_start + gap_end) / 2)
                gap_width_px = gap_end - gap_start + 1
                center = (line_x, gap_center_y)
                bbox = [line_x - 18, gap_center_y - gap_width_px // 2, line_x + 18, gap_center_y + gap_width_px // 2]

                if wall_near_perimeter(wall, wall_layer["building_bbox"], tolerance=65) and not entry_zone(center, wall_layer["building_bbox"]):
                    continue
                if point_near_any_label(center, ocr_items, padding=18):
                    continue
                if _overlaps_any_window(bbox, windows):
                    continue
                if _roi_nonzero(support_mask, [bbox[0] - 14, bbox[1] - 14, bbox[2] + 14, bbox[3] + 14]) < 18:
                    continue

                width_m = round(gap_width_px * meters_per_pixel, 3)
                if width_m < DOOR_MIN_WIDTH_M or width_m > DOOR_MAX_WIDTH_M:
                    continue

                detections.append(
                    {
                        "center_px": [line_x, gap_center_y],
                        "radius_px": int(gap_width_px / 2),
                        "wallId": wall["id"],
                        "offset": round(project_offset_on_wall(center, wall) * meters_per_pixel, 3),
                        "width": width_m,
                        "height": DEFAULT_DOOR_HEIGHT,
                        "swing": _candidate_swing(center, wall),
                        "bbox_px": bbox,
                        "leaf_line_px": None,
                    }
                )

    return detections


def _build_arc_seed_mask(gray, text_mask, wall_layer, windows):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 110)
    edges[text_mask > 0] = 0
    wall_mask = cv2.dilate(wall_layer["structural_wall_mask"], np.ones((5, 5), np.uint8), iterations=1)
    edges[wall_mask > 0] = 0
    for window in windows:
        x1, y1, x2, y2 = window["bbox_px"]
        edges[max(0, y1 - 10) : min(edges.shape[0], y2 + 11), max(0, x1 - 10) : min(edges.shape[1], x2 + 11)] = 0
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return edges


def _wall_contact_band(points, wall, tolerance):
    if wall["orientation"] == "horizontal":
        mask = np.abs(points[:, 1] - wall["start_px"][1]) <= tolerance
        near = points[mask]
        axis = near[:, 0] if len(near) else np.array([])
    else:
        mask = np.abs(points[:, 0] - wall["start_px"][0]) <= tolerance
        near = points[mask]
        axis = near[:, 1] if len(near) else np.array([])
    return near, axis


def _estimate_arc_from_component(points, wall, meters_per_pixel):
    contact_points, axis_values = _wall_contact_band(points, wall, DOOR_LEAF_WALL_DISTANCE + 10)
    if len(contact_points) < 3:
        return None

    if wall["orientation"] == "horizontal":
        order = np.argsort(contact_points[:, 0])
    else:
        order = np.argsort(contact_points[:, 1])
    contact_points = contact_points[order]
    hinge = contact_points[len(contact_points) // 2]

    distances = np.sqrt(((points - hinge) ** 2).sum(axis=1))
    if len(distances) < 8:
        return None
    radius_px = float(np.percentile(distances, 80))
    if radius_px < DOOR_MIN_RADIUS * 0.75 or radius_px > DOOR_MAX_RADIUS * 1.6:
        return None

    ring = np.abs(distances - radius_px) <= max(5.0, radius_px * 0.2)
    arc_points = points[ring]
    if len(arc_points) < 8:
        return None

    angles = np.degrees(np.arctan2(arc_points[:, 1] - hinge[1], arc_points[:, 0] - hinge[0]))
    angle_span = float(angles.max() - angles.min())
    if angle_span < 28.0:
        return None

    width_m = round(radius_px * meters_per_pixel, 3)
    if width_m < 0.45 or width_m > DOOR_MAX_WIDTH_M:
        return None

    tip = arc_points[int(np.argmax(np.sqrt(((arc_points - hinge) ** 2).sum(axis=1))))]
    bbox = _component_bbox(points, padding=6)
    score = angle_span + len(arc_points) + min(20.0, float(len(contact_points)))
    return {
        "center_px": [int(round((hinge[0] + tip[0]) / 2.0)), int(round((hinge[1] + tip[1]) / 2.0))],
        "radius_px": int(round(radius_px)),
        "wallId": wall["id"],
        "offset": round(project_offset_on_wall((int(hinge[0]), int(hinge[1])), wall) * meters_per_pixel, 3),
        "width": width_m,
        "height": DEFAULT_DOOR_HEIGHT,
        "swing": _candidate_swing((int(tip[0]), int(tip[1])), wall),
        "bbox_px": bbox,
        "leaf_line_px": None,
        "hinge_px": [int(hinge[0]), int(hinge[1])],
        "score": score,
        "arc_bbox_px": _component_bbox(arc_points, padding=2),
        "detector": "scratch_arc",
    }


def _detect_arc_components(gray, text_mask, wall_layer, windows, ocr_items):
    arc_mask = _build_arc_seed_mask(gray, text_mask, wall_layer, windows)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(arc_mask, connectivity=8)
    components = []

    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < 18:
            continue
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        bbox = [x, y, x + w - 1, y + h - 1]
        center = [x + w // 2, y + h // 2]
        if point_near_any_label(center, ocr_items, padding=18):
            continue

        points = _component_points(labels, component_id, [0, 0, arc_mask.shape[1] - 1, arc_mask.shape[0] - 1])
        if points is None or len(points) < 10:
            continue

        wall = nearest_wall(center, wall_layer["structural_walls"], allowed_types={"outer", "partition"}, max_distance=DOOR_MAX_RADIUS + 28)
        if wall is None:
            continue

        contact_points, axis_values = _wall_contact_band(points, wall, DOOR_LEAF_WALL_DISTANCE + 10)
        if len(contact_points) < 3:
            continue
        wall_span = float(axis_values.max() - axis_values.min()) if len(axis_values) else 0.0
        if wall_span > DOOR_MAX_RADIUS * 1.8:
            continue

        components.append(
            {
                "bbox_px": bbox,
                "center_px": center,
                "wall": wall,
                "points": points,
                "arc_bbox_px": bbox,
            }
        )

    return arc_mask, components


def detect_symbol_doors(gray, text_mask, wall_layer, meters_per_pixel, windows, ocr_items):
    arc_mask, components = _detect_arc_components(gray, text_mask, wall_layer, windows, ocr_items)
    detections = []

    for component in components:
        door = _estimate_arc_from_component(component["points"], component["wall"], meters_per_pixel)
        if door is not None:
            door["debug_zone_bbox"] = component["bbox_px"]
            detections.append(door)

    detections.sort(key=lambda item: (-item["score"], -item["width"], item["wallId"], item["offset"]))
    filtered = []
    for candidate in detections:
        duplicate = False
        for existing in filtered:
            if existing["wallId"] != candidate["wallId"]:
                continue
            if abs(existing["offset"] - candidate["offset"]) <= DOOR_DUPLICATE_OFFSET_TOLERANCE_M:
                duplicate = True
                break
            if "hinge_px" in existing and _point_distance(existing["hinge_px"], candidate["hinge_px"]) <= 20:
                duplicate = True
                break
            if interval_overlap(candidate["bbox_px"][0], candidate["bbox_px"][2], existing["bbox_px"][0], existing["bbox_px"][2]) > 12 and interval_overlap(candidate["bbox_px"][1], candidate["bbox_px"][3], existing["bbox_px"][1], existing["bbox_px"][3]) > 12:
                duplicate = True
                break
        if not duplicate:
            filtered.append(candidate)

    return filtered


def inspect_door_detector(gray, text_mask, wall_layer, ocr_items, windows=None):
    windows = windows or []
    arc_mask, components = _detect_arc_components(gray, text_mask, wall_layer, windows, ocr_items)
    arc_candidates = []
    for component in components:
        arc_candidates.append({"bbox_px": component["bbox_px"], "center_px": component["center_px"]})
    return {
        "symbol_mask": arc_mask,
        "edge_mask": arc_mask,
        "component_mask": arc_mask,
        "component_candidates": components,
        "line_candidates": [],
        "arc_candidates": arc_candidates,
    }


def detect_doors(gray, text_mask, wall_layer, ocr_items, meters_per_pixel, windows=None):
    windows = windows or []
    symbol_doors = detect_symbol_doors(gray, text_mask, wall_layer, meters_per_pixel, windows, ocr_items)
    combined = []

    for candidate in symbol_doors:
        duplicate = False
        for existing in combined:
            if existing["wallId"] != candidate["wallId"]:
                continue
            if abs(existing["offset"] - candidate["offset"]) <= DOOR_DUPLICATE_OFFSET_TOLERANCE_M:
                duplicate = True
                break
        if not duplicate:
            combined.append(candidate)

    combined.sort(key=lambda item: (item["wallId"], item["offset"]))
    formatted = []
    for index, door in enumerate(combined, start=1):
        item = door.copy()
        item["id"] = f"d{index}"
        formatted.append(item)
    return formatted


def detect_windows(walls, building_bbox, meters_per_pixel):
    outer_walls = [wall for wall in walls["structural_walls"] if wall["type"] == "outer"]
    candidates = []
    for orientation, mask in (("horizontal", walls["thin_horizontal_mask"]), ("vertical", walls["thin_vertical_mask"])):
        segments = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            length = max(w, h)
            thickness = min(w, h)
            if thickness > WINDOW_MAX_THICKNESS:
                continue
            if length < WINDOW_MIN_LENGTH or length > WINDOW_MAX_LENGTH:
                continue
            if orientation == "horizontal":
                segments.append({"orientation": orientation, "start_px": [x, y + h // 2], "end_px": [x + w - 1, y + h // 2], "bbox": [x, y, x + w - 1, y + h - 1]})
            else:
                segments.append({"orientation": orientation, "start_px": [x + w // 2, y], "end_px": [x + w // 2, y + h - 1], "bbox": [x, y, x + w - 1, y + h - 1]})
        candidates.extend(segments)

    grouped = []

    for candidate in candidates:
        candidate_length = math.dist(candidate["start_px"], candidate["end_px"])
        center = (int((candidate["start_px"][0] + candidate["end_px"][0]) / 2), int((candidate["start_px"][1] + candidate["end_px"][1]) / 2))
        nearest = nearest_wall(center, outer_walls, allowed_types={"outer"}, max_distance=WINDOW_WALL_DISTANCE)
        if nearest is None or nearest["orientation"] != candidate["orientation"]:
            continue
        axis_start, axis_end = wall_axis_bounds(nearest)
        candidate_axis_start, candidate_axis_end = (
            sorted((candidate["start_px"][0], candidate["end_px"][0])) if candidate["orientation"] == "horizontal" else sorted((candidate["start_px"][1], candidate["end_px"][1]))
        )
        if interval_overlap(axis_start, axis_end, candidate_axis_start, candidate_axis_end) < candidate_length * 0.3:
            continue
        offset_px = project_offset_on_wall(center, nearest)
        merged = False

        for group in grouped:
            if group["wallId"] != nearest["id"]:
                continue
            if abs(group["offset_px"] - offset_px) <= 20:
                group["offset_px"] = int(round((group["offset_px"] + offset_px) / 2))
                group["length_px"] = max(group["length_px"], candidate_length)
                x1, y1, x2, y2 = group["bbox"]
                cx1, cy1, cx2, cy2 = candidate["bbox"]
                group["bbox"] = [min(x1, cx1), min(y1, cy1), max(x2, cx2), max(y2, cy2)]
                group["segment_count"] += 1
                merged = True
                break

        if not merged:
            grouped.append(
                {
                    "wallId": nearest["id"],
                    "offset_px": offset_px,
                    "length_px": candidate_length,
                    "bbox": candidate["bbox"][:],
                    "segment_count": 1,
                    "wall_orientation": nearest["orientation"],
                    "wall_axis_px": nearest["start_px"][1] if nearest["orientation"] == "horizontal" else nearest["start_px"][0],
                }
            )

    windows = []
    for index, group in enumerate(grouped, start=1):
        width_m = round(group["length_px"] * meters_per_pixel, 3)
        if width_m < 0.4 or width_m > 2.5:
            continue
        center_px = [
            int((group["bbox"][0] + group["bbox"][2]) / 2),
            int((group["bbox"][1] + group["bbox"][3]) / 2),
        ]
        windows.append(
            {
                "id": f"win{index}",
                "wallId": group["wallId"],
                "offset": round(group["offset_px"] * meters_per_pixel, 3),
                "width": width_m,
                "height": DEFAULT_WINDOW_HEIGHT,
                "sillHeight": DEFAULT_SILL_HEIGHT,
                "bbox_px": group["bbox"],
                "center_px": center_px,
                "segment_count": group["segment_count"],
                "wall_orientation": group["wall_orientation"],
                "wall_axis_px": group["wall_axis_px"],
            }
        )

    deduped = []
    windows.sort(key=lambda item: (-item["width"], -item["segment_count"]))
    for window in windows:
        duplicate = False
        for existing in deduped:
            if existing["wallId"] != window["wallId"]:
                continue
            if abs(existing["offset"] - window["offset"]) > 0.35:
                continue
            overlap_x = interval_overlap(window["bbox_px"][0], window["bbox_px"][2], existing["bbox_px"][0], existing["bbox_px"][2])
            overlap_y = interval_overlap(window["bbox_px"][1], window["bbox_px"][3], existing["bbox_px"][1], existing["bbox_px"][3])
            if overlap_x <= 0 or overlap_y <= 0:
                continue
            duplicate = True
            break
        if not duplicate:
            deduped.append(window)

    for index, window in enumerate(deduped, start=1):
        window["id"] = f"win{index}"

    return deduped


def build_symbol_mask(image_shape, doors, windows):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for door in doors:
        x1, y1, x2, y2 = door["bbox_px"]
        cv2.rectangle(
            mask,
            (max(0, x1 - DOOR_SUPPRESSION_PADDING), max(0, y1 - DOOR_SUPPRESSION_PADDING)),
            (min(image_shape[1] - 1, x2 + DOOR_SUPPRESSION_PADDING), min(image_shape[0] - 1, y2 + DOOR_SUPPRESSION_PADDING)),
            255,
            -1,
        )
        if door.get("leaf_line_px"):
            sx, sy, ex, ey = door["leaf_line_px"]
            cv2.line(mask, (sx, sy), (ex, ey), 255, 6)

    for window in windows:
        x1, y1, x2, y2 = window["bbox_px"]
        if window.get("wall_orientation") == "horizontal":
            wall_y = int(window.get("wall_axis_px", (y1 + y2) // 2))
            x1 -= WINDOW_SUPPRESSION_PADDING
            x2 += WINDOW_SUPPRESSION_PADDING
            y1 = min(y1, wall_y) - WINDOW_MASK_ORTHO_PADDING
            y2 = max(y2, wall_y) + WINDOW_MASK_ORTHO_PADDING
        elif window.get("wall_orientation") == "vertical":
            wall_x = int(window.get("wall_axis_px", (x1 + x2) // 2))
            x1 = min(x1, wall_x) - WINDOW_MASK_ORTHO_PADDING
            x2 = max(x2, wall_x) + WINDOW_MASK_ORTHO_PADDING
            y1 -= WINDOW_SUPPRESSION_PADDING
            y2 += WINDOW_SUPPRESSION_PADDING
        elif (x2 - x1) >= (y2 - y1):
            x1 -= WINDOW_SUPPRESSION_PADDING
            x2 += WINDOW_SUPPRESSION_PADDING
            y1 -= WINDOW_MASK_ORTHO_PADDING
            y2 += WINDOW_MASK_ORTHO_PADDING
        else:
            x1 -= WINDOW_MASK_ORTHO_PADDING
            x2 += WINDOW_MASK_ORTHO_PADDING
            y1 -= WINDOW_SUPPRESSION_PADDING
            y2 += WINDOW_SUPPRESSION_PADDING
        cv2.rectangle(
            mask,
            (max(0, x1), max(0, y1)),
            (min(image_shape[1] - 1, x2), min(image_shape[0] - 1, y2)),
            255,
            -1,
        )

    return mask


def prune_window_parallel_wall_segments(walls, windows):
    pruned = []

    for wall in walls:
        keep = True
        wall_axis = wall["start_px"][1] if wall["orientation"] == "horizontal" else wall["start_px"][0]
        wall_axis_start, wall_axis_end = wall_axis_bounds(wall)

        for window in windows:
            if wall["orientation"] != window.get("wall_orientation"):
                continue

            host_axis = int(window.get("wall_axis_px", wall_axis))
            if abs(wall_axis - host_axis) <= 2:
                continue
            if abs(wall_axis - host_axis) > WINDOW_MASK_ORTHO_PADDING + WINDOW_SUPPRESSION_PADDING + 6:
                continue

            if wall["orientation"] == "horizontal":
                win_start, win_end = sorted((window["bbox_px"][0], window["bbox_px"][2]))
            else:
                win_start, win_end = sorted((window["bbox_px"][1], window["bbox_px"][3]))

            if interval_overlap(wall_axis_start, wall_axis_end, win_start - WINDOW_SUPPRESSION_PADDING, win_end + WINDOW_SUPPRESSION_PADDING) < 12:
                continue

            keep = False
            break

        if keep:
            pruned.append(wall)

    return pruned


def reassign_openings_to_walls(openings, walls, meters_per_pixel):
    reassigned = []
    for opening in openings:
        center = tuple(opening.get("center_px") or [(opening["bbox_px"][0] + opening["bbox_px"][2]) // 2, (opening["bbox_px"][1] + opening["bbox_px"][3]) // 2])
        nearest = nearest_wall(center, walls, allowed_types={"outer", "partition"}, max_distance=80)
        if nearest is None:
            nearest = nearest_wall(center, walls, max_distance=120)
        if nearest is None:
            continue
        updated = opening.copy()
        updated["wallId"] = nearest["id"]
        updated["offset"] = round(project_offset_on_wall(center, nearest) * meters_per_pixel, 3)
        reassigned.append(updated)
    return reassigned


def dedupe_windows_after_reassign(windows):
    ordered = sorted(windows, key=lambda item: (-item["width"], item["wallId"], item["offset"]))
    deduped = []
    for window in ordered:
        duplicate = False
        for existing in deduped:
            if existing["wallId"] != window["wallId"]:
                continue
            if abs(existing["offset"] - window["offset"]) > 0.3:
                continue
            duplicate = True
            break
        if not duplicate:
            deduped.append(window)

    for index, window in enumerate(deduped, start=1):
        window["id"] = f"win{index}"
    return deduped
