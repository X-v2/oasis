from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from parser.config import ParserConfig
from parser.doors import Door
from parser.scale import ScaleResult
from parser.text import TextDetection
from parser.walls import WallSegment, render_wall_mask
from parser.windows import Window


def build_schema(
    config: ParserConfig,
    scale: ScaleResult,
    plan_bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    walls: list[WallSegment],
    doors: list[Door],
    windows: list[Window],
    texts: list[TextDetection],
) -> dict[str, Any]:
    x0, y0, _, _ = plan_bbox

    def to_point_2d(x_px: float, y_px: float) -> list[float]:
        x = round((x_px - x0) / scale.pixels_per_meter, 3)
        z = round((y_px - y0) / scale.pixels_per_meter, 3)
        return [x, z]

    def to_point_3d(x_px: float, y_m: float, y_px: float) -> list[float]:
        x, z = to_point_2d(x_px, y_px)
        return [x, round(y_m, 3), z]

    payload: dict[str, Any] = {
        "meta": {
            "unit": "meter",
            "wallHeight": config.default_wall_height_m,
            "defaultWallThickness": config.default_wall_thickness_m,
        },
        "walls": [],
        "slabs": [],
        "labels": [],
        "doors": [],
        "windows": [],
        "openings": [],
        "graphNodes": [],
        "columns": [],
    }

    label_index = 1
    room_labels: list[dict[str, Any]] = []
    for text in texts:
        if text.kind != "room":
            continue
        tx1, ty1, tx2, ty2 = text.bbox
        cx = (tx1 + tx2) // 2
        cy = (ty1 + ty2) // 2
        if not _point_in_plan_bbox((cx, cy), plan_bbox):
            continue
        title = text.text.title()
        payload["labels"].append(
            {
                "id": f"l{label_index}",
                "text": title,
                "position": to_point_3d(cx, 0.22, cy),
            }
        )
        if _usable_room_label(text.text, config):
            room_labels.append(
                {
                    "id": f"l{label_index}",
                    "name": title,
                    "center": (cx, cy),
                    "bbox": text.bbox,
                }
            )
        label_index += 1

    for wall in walls:
        if wall.orientation == "horizontal":
            start = to_point_2d(wall.span_start, wall.center)
            end = to_point_2d(wall.span_end, wall.center)
        else:
            start = to_point_2d(wall.center, wall.span_start)
            end = to_point_2d(wall.center, wall.span_end)
        payload["walls"].append(
            {
                "id": wall.id,
                "start": start,
                "end": end,
                "thickness": round(max(1.0, wall.thickness_px) / scale.pixels_per_meter, 3),
                "height": config.default_wall_height_m,
                "type": wall.kind,
            }
        )

    wall_lookup = {wall.id: wall for wall in walls}
    for door in doors:
        wall = wall_lookup[door.wall_id]
        offset = _offset_on_wall(door.center, wall, scale, door.width_px)
        door_x_px, door_y_px = _opening_anchor_on_wall(door.center, door.width_px, wall)
        position = to_point_3d(door_x_px, 0.0, door_y_px)
        payload["doors"].append(
            {
                "id": door.id,
                "wallId": door.wall_id,
                "offset": round(offset, 3),
                "position": position,
                "width": round(max(0.6, door.width_px / scale.pixels_per_meter), 3),
                "height": config.default_door_height_m,
                "swing": door.swing,
            }
        )
        payload["openings"].append(
            {
                "id": f"o_{door.id}",
                "kind": "door",
                "wallId": door.wall_id,
                "offset": round(offset, 3),
                "position": position,
                "width": round(max(0.6, door.width_px / scale.pixels_per_meter), 3),
                "height": config.default_door_height_m,
                "swing": door.swing,
            }
        )

    for window in windows:
        wall = wall_lookup[window.wall_id]
        offset = _offset_on_wall(window.center, wall, scale, window.width_px)
        window_x_px, window_y_px = _opening_anchor_on_wall(window.center, window.width_px, wall)
        position = to_point_3d(
            window_x_px,
            config.default_window_sill_m + config.default_window_height_m / 2.0,
            window_y_px,
        )
        payload["windows"].append(
            {
                "id": window.id,
                "wallId": window.wall_id,
                "offset": round(offset, 3),
                "position": position,
                "width": round(max(0.6, window.width_px / scale.pixels_per_meter), 3),
                "height": config.default_window_height_m,
                "sillHeight": config.default_window_sill_m,
                "windowType": "double",
            }
        )
        payload["openings"].append(
            {
                "id": f"o_{window.id}",
                "kind": "window",
                "wallId": window.wall_id,
                "offset": round(offset, 3),
                "position": position,
                "width": round(max(0.6, window.width_px / scale.pixels_per_meter), 3),
                "height": config.default_window_height_m,
                "sillHeight": config.default_window_sill_m,
                "windowType": "double",
            }
        )

    slabs = _derive_slabs(room_labels, walls, plan_bbox, image_shape, config, to_point_2d, to_point_3d)
    payload["slabs"].extend(slabs)
    for slab in slabs:
        if slab["name"] != "":
            continue
        label_index += 1
        payload["labels"].append(
            {
                "id": f"l{label_index}",
                "text": "",
                "position": slab["centroid"],
            }
        )
    graph_nodes = _derive_graph_nodes(walls, to_point_3d)
    payload["graphNodes"].extend(graph_nodes)
    payload["columns"].extend(_derive_columns(graph_nodes, config.default_wall_height_m))
    return payload


def _offset_on_wall(center: tuple[int, int], wall: WallSegment, scale: ScaleResult, width_px: float = 0.0) -> float:
    anchor_x, anchor_y = _opening_anchor_on_wall(center, width_px, wall)
    cx, cy = anchor_x, anchor_y
    if wall.orientation == "horizontal":
        return max(0.0, (cx - wall.span_start) / scale.pixels_per_meter)
    return max(0.0, (cy - wall.span_start) / scale.pixels_per_meter)


def _opening_anchor_on_wall(center: tuple[int, int], width_px: float, wall: WallSegment) -> tuple[int, int]:
    cx, cy = center
    half_width = max(0.0, width_px / 2.0)
    if wall.orientation == "horizontal":
        x = int(round(min(max(cx - half_width, wall.span_start), wall.span_end)))
        return x, wall.center
    y = int(round(min(max(cy - half_width, wall.span_start), wall.span_end)))
    return wall.center, y


def _usable_room_label(text: str, config: ParserConfig) -> bool:
    normalized = text.upper()
    if "/" in normalized:
        return False
    return not any(keyword in normalized for keyword in config.ignored_text_keywords)


def _point_in_plan_bbox(point: tuple[int, int], plan_bbox: tuple[int, int, int, int]) -> bool:
    x, y = point
    x0, y0, w, h = plan_bbox
    return x0 <= x <= x0 + w - 1 and y0 <= y <= y0 + h - 1


def _derive_slabs(
    room_labels: list[dict[str, Any]],
    walls: list[WallSegment],
    plan_bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    config: ParserConfig,
    to_point_2d,
    to_point_3d,
) -> list[dict[str, Any]]:
    if not walls:
        return []

    x0, y0, w, h = plan_bbox
    wall_mask = render_wall_mask(image_shape, walls)
    plan_mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.rectangle(plan_mask, (x0, y0), (x0 + w - 1, y0 + h - 1), 255, thickness=-1)
    free_mask = cv2.bitwise_and(plan_mask, cv2.bitwise_not(wall_mask))
    free_mask = cv2.morphologyEx(free_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    region_labels = _connected_region_labels(free_mask)
    if region_labels is None:
        return []

    label_image, component_ids = region_labels
    min_region_area = max(250, config.room_min_area_px // 8)
    slabs: list[dict[str, Any]] = []
    slab_index = 1

    for component_id in component_ids:
        component_mask = np.where(label_image == component_id, 255, 0).astype(np.uint8)
        if cv2.countNonZero(component_mask) < min_region_area:
            continue
        labels_in_component = [label for label in room_labels if _label_in_region(label["center"], component_mask)]
        if not labels_in_component:
            if not _region_has_wall_support(component_mask, wall_mask):
                continue
            polygon_px = _region_to_polygon(component_mask)
            if len(polygon_px) < 3:
                continue
            centroid_px = _region_centroid(component_mask, None)
            slabs.append(
                {
                    "id": f"s{slab_index}",
                    "name": "",
                    "polygon": [to_point_2d(x, y) for x, y in polygon_px],
                    "centroid": to_point_3d(centroid_px[0], 0.02, centroid_px[1]),
                }
            )
            slab_index += 1
            continue
        split_regions = _split_region_by_labels(component_mask, labels_in_component, min_region_area)
        split_regions = _fill_region_gaps_by_relevance(component_mask, split_regions, labels_in_component)
        for region_mask, label in split_regions:
            polygon_px = _region_to_polygon(region_mask)
            if len(polygon_px) < 3:
                continue
            centroid_px = _region_centroid(region_mask, label["center"])
            slabs.append(
                {
                    "id": f"s{slab_index}",
                    "name": label["name"],
                    "polygon": [to_point_2d(x, y) for x, y in polygon_px],
                    "centroid": to_point_3d(centroid_px[0], 0.02, centroid_px[1]),
                }
            )
            slab_index += 1
    return slabs


def _connected_region_labels(free_mask: np.ndarray) -> tuple[np.ndarray, list[int]] | None:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(free_mask, 8)
    if num_labels <= 1:
        return None

    component_ids: list[int] = []
    height, width = free_mask.shape
    for index in range(1, num_labels):
        x, y, w, h, area = stats[index]
        if area <= 0:
            continue
        touches_border = x == 0 or y == 0 or (x + w) >= width or (y + h) >= height
        if touches_border:
            continue
        component_ids.append(index)
    return labels, component_ids


def _label_in_region(center: tuple[int, int], region_mask: np.ndarray) -> bool:
    x, y = center
    if x < 0 or y < 0 or y >= region_mask.shape[0] or x >= region_mask.shape[1]:
        return False
    return bool(region_mask[y, x] > 0)


def _region_has_wall_support(region_mask: np.ndarray, wall_mask: np.ndarray) -> bool:
    ys, xs = np.where(region_mask > 0)
    if len(xs) == 0:
        return False

    min_x = int(xs.min())
    max_x = int(xs.max())
    min_y = int(ys.min())
    max_y = int(ys.max())
    support_depth = 3
    padded_walls = cv2.dilate(wall_mask, np.ones((3, 3), np.uint8), iterations=1)
    min_support = 4
    supported_sides = 0

    left_y = ys[xs == min_x]
    if len(left_y):
        x1 = max(0, min_x - support_depth)
        strip = padded_walls[left_y.min() : left_y.max() + 1, x1:min_x]
        if cv2.countNonZero(strip) >= min_support:
            supported_sides += 1

    right_y = ys[xs == max_x]
    if len(right_y):
        x2 = min(padded_walls.shape[1], max_x + support_depth + 1)
        strip = padded_walls[right_y.min() : right_y.max() + 1, max_x + 1 : x2]
        if cv2.countNonZero(strip) >= min_support:
            supported_sides += 1

    top_x = xs[ys == min_y]
    if len(top_x):
        y1 = max(0, min_y - support_depth)
        strip = padded_walls[y1:min_y, top_x.min() : top_x.max() + 1]
        if cv2.countNonZero(strip) >= min_support:
            supported_sides += 1

    bottom_x = xs[ys == max_y]
    if len(bottom_x):
        y2 = min(padded_walls.shape[0], max_y + support_depth + 1)
        strip = padded_walls[max_y + 1 : y2, bottom_x.min() : bottom_x.max() + 1]
        if cv2.countNonZero(strip) >= min_support:
            supported_sides += 1

    return supported_sides >= 2


def _split_region_by_labels(
    region_mask: np.ndarray,
    labels_in_component: list[dict[str, Any]],
    min_region_area: int,
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    if len(labels_in_component) == 1:
        return [(region_mask, labels_in_component[0])]

    ys, xs = np.where(region_mask > 0)
    if len(xs) == 0:
        return []

    coords = np.column_stack((ys, xs)).astype(np.int32)
    seeds = np.array([(label["center"][1], label["center"][0]) for label in labels_in_component], dtype=np.int32)
    distances = ((coords[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2)
    ownership = distances.argmin(axis=1)

    split_regions: list[tuple[np.ndarray, dict[str, Any]]] = []
    for index, label in enumerate(labels_in_component):
        local_mask = np.zeros_like(region_mask)
        owned = coords[ownership == index]
        if len(owned) == 0:
            continue
        local_mask[owned[:, 0], owned[:, 1]] = 255
        local_mask = _keep_component_containing_seed(local_mask, label["center"])
        if cv2.countNonZero(local_mask) < min_region_area:
            continue
        split_regions.append((local_mask, label))

    if split_regions:
        return split_regions
    return [(region_mask, labels_in_component[0])]


def _fill_region_gaps_by_relevance(
    component_mask: np.ndarray,
    split_regions: list[tuple[np.ndarray, dict[str, Any]]],
    labels_in_component: list[dict[str, Any]],
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    if len(split_regions) <= 1:
        return split_regions

    combined = np.zeros_like(component_mask)
    for region_mask, _ in split_regions:
        combined = cv2.bitwise_or(combined, region_mask)

    gap_mask = cv2.bitwise_and(component_mask, cv2.bitwise_not(combined))
    if cv2.countNonZero(gap_mask) == 0:
        return split_regions

    ys, xs = np.where(gap_mask > 0)
    if len(xs) == 0:
        return split_regions

    seeds = np.array([(label["center"][1], label["center"][0]) for label in labels_in_component], dtype=np.float32)
    coords = np.column_stack((ys, xs)).astype(np.float32)
    distances = ((coords[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2)
    ownership = distances.argmin(axis=1)

    updated: list[tuple[np.ndarray, dict[str, Any]]] = []
    for index, (region_mask, label) in enumerate(split_regions):
        expanded = region_mask.copy()
        owned = coords[ownership == index]
        if len(owned):
            expanded[owned[:, 0].astype(np.int32), owned[:, 1].astype(np.int32)] = 255
        updated.append((expanded, label))
    return updated


def _keep_component_containing_seed(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    seed_x, seed_y = seed
    if seed_x < 0 or seed_y < 0 or seed_y >= mask.shape[0] or seed_x >= mask.shape[1]:
        return mask
    if mask[seed_y, seed_x] == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return mask
    seed_label = labels[seed_y, seed_x]
    if seed_label == 0:
        return mask
    kept = np.zeros_like(mask)
    kept[labels == seed_label] = 255
    return kept


def _region_to_polygon(region_mask: np.ndarray) -> list[tuple[int, int]]:
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea)
    points = [(int(point[0][0]), int(point[0][1])) for point in contour]
    if len(points) < 3:
        return []
    points = _orthogonalize_points(points, region_mask)
    points = _remove_collinear_points(points)
    if len(points) < 3:
        return []
    if _polygon_expands_outside_mask(points, region_mask):
        mask_safe = _mask_safe_polygon(region_mask)
        if len(mask_safe) >= 3:
            return mask_safe
    return points


def _orthogonalize_points(
    points: list[tuple[int, int]],
    region_mask: np.ndarray,
) -> list[tuple[int, int]]:
    if not points:
        return []

    orthogonal: list[tuple[int, int]] = [points[0]]
    for point in points[1:]:
        prev_x, prev_y = orthogonal[-1]
        x, y = point
        if x == prev_x or y == prev_y:
            orthogonal.append((x, y))
            continue
        elbow_a = (x, prev_y)
        elbow_b = (prev_x, y)
        if _l_path_inside_mask((prev_x, prev_y), elbow_a, (x, y), region_mask):
            orthogonal.append(elbow_a)
            orthogonal.append((x, y))
            continue
        if _l_path_inside_mask((prev_x, prev_y), elbow_b, (x, y), region_mask):
            orthogonal.append(elbow_b)
            orthogonal.append((x, y))
            continue
        if _l_path_penalty((prev_x, prev_y), elbow_a, (x, y), region_mask) <= _l_path_penalty((prev_x, prev_y), elbow_b, (x, y), region_mask):
            orthogonal.append(elbow_a)
        else:
            orthogonal.append(elbow_b)
        orthogonal.append((x, y))

    if orthogonal[0] != orthogonal[-1]:
        orthogonal.append(orthogonal[0])
    return orthogonal


def _l_path_inside_mask(
    start: tuple[int, int],
    elbow: tuple[int, int],
    end: tuple[int, int],
    region_mask: np.ndarray,
) -> bool:
    return _segment_inside_mask(start, elbow, region_mask) and _segment_inside_mask(elbow, end, region_mask)


def _segment_inside_mask(
    start: tuple[int, int],
    end: tuple[int, int],
    region_mask: np.ndarray,
) -> bool:
    x1, y1 = start
    x2, y2 = end
    if x1 == x2:
        low, high = sorted((y1, y2))
        for y in range(low, high + 1):
            if not _mask_on(region_mask, x1, y):
                return False
        return True
    if y1 == y2:
        low, high = sorted((x1, x2))
        for x in range(low, high + 1):
            if not _mask_on(region_mask, x, y1):
                return False
        return True
    return False


def _l_path_penalty(
    start: tuple[int, int],
    elbow: tuple[int, int],
    end: tuple[int, int],
    region_mask: np.ndarray,
) -> int:
    return _segment_penalty(start, elbow, region_mask) + _segment_penalty(elbow, end, region_mask)


def _segment_penalty(
    start: tuple[int, int],
    end: tuple[int, int],
    region_mask: np.ndarray,
) -> int:
    x1, y1 = start
    x2, y2 = end
    misses = 0
    if x1 == x2:
        low, high = sorted((y1, y2))
        for y in range(low, high + 1):
            if not _mask_on(region_mask, x1, y):
                misses += 1
    elif y1 == y2:
        low, high = sorted((x1, x2))
        for x in range(low, high + 1):
            if not _mask_on(region_mask, x, y1):
                misses += 1
    else:
        misses += max(abs(x2 - x1), abs(y2 - y1))
    return misses


def _mask_on(region_mask: np.ndarray, x: int, y: int) -> bool:
    if x < 0 or y < 0 or y >= region_mask.shape[0] or x >= region_mask.shape[1]:
        return False
    return bool(region_mask[y, x] > 0)


def _polygon_expands_outside_mask(points: list[tuple[int, int]], region_mask: np.ndarray) -> bool:
    polygon_mask = np.zeros_like(region_mask)
    contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(polygon_mask, [contour], 255)
    outside = cv2.bitwise_and(polygon_mask, cv2.bitwise_not(region_mask))
    return cv2.countNonZero(outside) > 0


def _mask_safe_polygon(region_mask: np.ndarray) -> list[tuple[int, int]]:
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    points = [(int(point[0][0]), int(point[0][1])) for point in contour]
    if len(points) < 3:
        return []
    simplified = cv2.approxPolyDP(contour, 2.0, True)
    points = [(int(point[0][0]), int(point[0][1])) for point in simplified]
    if len(points) < 3:
        return []
    points = _orthogonalize_points(points, region_mask)
    points = _remove_collinear_points(points)
    if len(points) < 3 or _polygon_expands_outside_mask(points, region_mask):
        return []
    return points


def _remove_collinear_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    simplified: list[tuple[int, int]] = []
    for point in points:
        if simplified and point == simplified[-1]:
            continue
        simplified.append(point)
    if len(simplified) > 1 and simplified[0] == simplified[-1]:
        simplified.pop()
    if len(simplified) < 3:
        return simplified

    changed = True
    while changed and len(simplified) >= 3:
        changed = False
        result: list[tuple[int, int]] = []
        count = len(simplified)
        for index, point in enumerate(simplified):
            prev_point = simplified[(index - 1) % count]
            next_point = simplified[(index + 1) % count]
            if _is_collinear(prev_point, point, next_point):
                changed = True
                continue
            result.append(point)
        if len(result) < 3:
            return simplified
        simplified = result
    return simplified


def _is_collinear(left: tuple[int, int], middle: tuple[int, int], right: tuple[int, int]) -> bool:
    return (left[0] == middle[0] == right[0]) or (left[1] == middle[1] == right[1])


def _region_centroid(region_mask: np.ndarray, preferred: tuple[int, int] | None) -> tuple[int, int]:
    if preferred is not None:
        px, py = preferred
        if 0 <= px < region_mask.shape[1] and 0 <= py < region_mask.shape[0] and region_mask[py, px] > 0:
            return preferred

    moments = cv2.moments(region_mask)
    if moments["m00"] > 0:
        cx = int(round(moments["m10"] / moments["m00"]))
        cy = int(round(moments["m01"] / moments["m00"]))
        return cx, cy

    ys, xs = np.where(region_mask > 0)
    if len(xs) == 0:
        return preferred if preferred is not None else (0, 0)
    return int(np.mean(xs)), int(np.mean(ys))


def _derive_graph_nodes(
    walls: list[WallSegment],
    to_point_3d,
) -> list[dict[str, Any]]:
    tolerance = 10
    clusters: list[dict[str, Any]] = []
    for wall in walls:
        endpoints = (
            (wall.span_start, wall.center) if wall.orientation == "horizontal" else (wall.center, wall.span_start),
            (wall.span_end, wall.center) if wall.orientation == "horizontal" else (wall.center, wall.span_end),
        )
        for point in endpoints:
            cluster = next((item for item in clusters if _distance(point, item["point"]) <= tolerance), None)
            if cluster is None:
                clusters.append({"point": point, "wall_ids": {wall.id}})
            else:
                cluster["wall_ids"].add(wall.id)

    nodes: list[dict[str, Any]] = []
    for index, cluster in enumerate(clusters, start=1):
        wall_ids = sorted(cluster["wall_ids"])
        degree = len(wall_ids)
        node_type = _node_type_for_degree(degree)
        likely_column = degree >= 3 or _has_perpendicular_walls(wall_ids, walls)
        x_px, y_px = cluster["point"]
        nodes.append(
            {
                "id": f"n{index}",
                "position": to_point_3d(x_px, 0.2, y_px),
                "connectedWallIds": wall_ids,
                "degree": degree,
                "type": node_type,
                "likelyColumn": likely_column,
            }
        )
    return nodes


def _derive_columns(
    graph_nodes: list[dict[str, Any]],
    wall_height_m: float,
) -> list[dict[str, Any]]:
    columns: list[dict[str, Any]] = []
    column_index = 1
    for node in graph_nodes:
        if not node["likelyColumn"]:
            continue
        x, _, z = node["position"]
        columns.append(
            {
                "id": f"c{column_index}",
                "nodeId": node["id"],
                "position": [x, round(wall_height_m / 2.0, 3), z],
                "width": 0.3,
                "depth": 0.3,
                "height": wall_height_m,
            }
        )
        column_index += 1
    return columns


def _distance(left: tuple[int, int], right: tuple[int, int]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _node_type_for_degree(degree: int) -> str:
    if degree <= 1:
        return "terminal"
    if degree == 2:
        return "corner"
    if degree == 3:
        return "junction"
    return "junction"


def _has_perpendicular_walls(wall_ids: list[str], walls: list[WallSegment]) -> bool:
    wall_lookup = {wall.id: wall for wall in walls}
    orientations = {wall_lookup[wall_id].orientation for wall_id in wall_ids if wall_id in wall_lookup}
    return len(orientations) > 1
