from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from parser.config import ParserConfig


@dataclass(slots=True)
class WallSegment:
    id: str
    orientation: str
    center: int
    span_start: int
    span_end: int
    thickness_px: float
    kind: str


@dataclass(slots=True)
class WallDetectionResult:
    plan_bbox: tuple[int, int, int, int]
    plan_mask: np.ndarray
    structure_mask: np.ndarray
    wall_mask: np.ndarray
    walls: list[WallSegment]


def detect_plan_and_walls(
    binary_inv: np.ndarray,
    text_mask: np.ndarray,
    config: ParserConfig,
) -> WallDetectionResult:
    wall_seed = cv2.bitwise_and(binary_inv, cv2.bitwise_not(text_mask))
    plan_bbox = _largest_component_bbox(wall_seed)
    x, y, w, h = plan_bbox
    pad = config.plan_padding_px
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(binary_inv.shape[1], x + w + pad)
    y2 = min(binary_inv.shape[0], y + h + pad)
    plan_bbox = (x1, y1, x2 - x1, y2 - y1)

    plan_mask = np.zeros_like(binary_inv)
    plan_mask[y1:y2, x1:x2] = 255
    clean = cv2.bitwise_and(wall_seed, plan_mask)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    thick_clean = _keep_thick_components(clean, config)

    structure = _extract_segment_group(
        clean=thick_clean,
        plan_bbox=plan_bbox,
        horizontal_kernel=config.horizontal_wall_kernel_px,
        vertical_kernel=config.vertical_wall_kernel_px,
        gap_tolerance=config.wall_merge_gap_px,
        config=config,
    )
    walls = _classify_walls(structure["segments"], plan_bbox, config)
    wall_mask = render_wall_mask(binary_inv.shape, walls)
    structure_mask = cv2.bitwise_and(thick_clean, wall_mask)
    return WallDetectionResult(
        plan_bbox=plan_bbox,
        plan_mask=plan_mask,
        structure_mask=structure_mask,
        wall_mask=wall_mask,
        walls=walls,
    )


def _largest_component_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return (0, 0, mask.shape[1], mask.shape[0])
    best_index = 1
    best_area = 0
    best_bbox = (0, 0, mask.shape[1], mask.shape[0])
    height, width = mask.shape
    for index in range(1, num_labels):
        x, y, w, h, area = stats[index]
        if area < best_area:
            continue
        if w < width * 0.4 or h < height * 0.4:
            continue
        best_area = int(area)
        best_bbox = (int(x), int(y), int(w), int(h))
    return best_bbox


def _keep_thick_components(mask: np.ndarray, config: ParserConfig) -> np.ndarray:
    if np.count_nonzero(mask) == 0:
        return mask.copy()
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    thick_core = np.zeros_like(mask)
    thick_core[distance >= config.wall_min_stroke_radius_px] = 255
    if np.count_nonzero(thick_core) == 0:
        return thick_core

    # Expand only from genuinely thick pixels so thin attached strokes do not survive.
    thick_mask = cv2.dilate(thick_core, np.ones((5, 5), np.uint8), iterations=1)
    thick_mask = cv2.bitwise_and(thick_mask, mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thick_mask, 8)
    filtered = np.zeros_like(mask)
    for index in range(1, num_labels):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < config.wall_min_component_area_px:
            continue
        filtered[labels == index] = 255
    return filtered


def _extract_segment_group(
    clean: np.ndarray,
    plan_bbox: tuple[int, int, int, int],
    horizontal_kernel: int,
    vertical_kernel: int,
    gap_tolerance: int,
    config: ParserConfig,
) -> dict[str, object]:
    horizontal_mask = cv2.morphologyEx(
        clean,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel, 1)),
    )
    vertical_mask = cv2.morphologyEx(
        clean,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel)),
    )

    horizontal_segments = _filter_segments_to_bbox(
        _merge_horizontal_segments(
            _extract_line_segments(horizontal_mask, "horizontal", config.min_wall_length_px),
            config.wall_merge_gap_px,
            config.alignment_tolerance_px,
        ),
        plan_bbox,
    )
    vertical_segments = _filter_segments_to_bbox(
        _merge_vertical_segments(
            _extract_line_segments(vertical_mask, "vertical", config.min_wall_length_px),
            config.wall_merge_gap_px,
            config.alignment_tolerance_px,
        ),
        plan_bbox,
    )

    segments = horizontal_segments + vertical_segments
    segments.extend(_recover_short_wall_segments(clean, plan_bbox, segments, config))
    segments = _bridge_wall_segments(segments, config)
    return {
        "horizontal_mask": horizontal_mask,
        "vertical_mask": vertical_mask,
        "segments": segments,
    }


def _extract_line_segments(
    mask: np.ndarray,
    orientation: str,
    min_length: int,
) -> list[dict[str, int | float | str]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments: list[dict[str, int | float | str]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        length = max(w, h)
        thickness = h if orientation == "horizontal" else w
        if length < min_length:
            continue
        if orientation == "horizontal":
            segments.append(
                {
                    "orientation": orientation,
                    "x1": x,
                    "y": y + h // 2,
                    "x2": x + w - 1,
                    "thickness_px": h,
                }
            )
        else:
            segments.append(
                {
                    "orientation": orientation,
                    "x": x + w // 2,
                    "y1": y,
                    "y2": y + h - 1,
                    "thickness_px": w,
                }
            )
    return segments


def _merge_horizontal_segments(
    segments: list[dict[str, int | float | str]],
    gap_tolerance: int,
    alignment_tolerance: int,
) -> list[dict[str, int | float | str]]:
    segments = sorted(segments, key=lambda item: (int(item["y"]), int(item["x1"])))
    merged: list[dict[str, int | float | str]] = []
    for segment in segments:
        if merged:
            current = merged[-1]
            aligned = abs(int(segment["y"]) - int(current["y"])) <= alignment_tolerance
            touching = int(segment["x1"]) <= int(current["x2"]) + gap_tolerance
            if aligned and touching:
                current["y"] = int(round((int(current["y"]) + int(segment["y"])) / 2))
                current["x1"] = min(int(current["x1"]), int(segment["x1"]))
                current["x2"] = max(int(current["x2"]), int(segment["x2"]))
                current["thickness_px"] = max(float(current["thickness_px"]), float(segment["thickness_px"]))
                continue
        merged.append(segment.copy())
    return merged


def _merge_vertical_segments(
    segments: list[dict[str, int | float | str]],
    gap_tolerance: int,
    alignment_tolerance: int,
) -> list[dict[str, int | float | str]]:
    segments = sorted(segments, key=lambda item: (int(item["x"]), int(item["y1"])))
    merged: list[dict[str, int | float | str]] = []
    for segment in segments:
        if merged:
            current = merged[-1]
            aligned = abs(int(segment["x"]) - int(current["x"])) <= alignment_tolerance
            touching = int(segment["y1"]) <= int(current["y2"]) + gap_tolerance
            if aligned and touching:
                current["x"] = int(round((int(current["x"]) + int(segment["x"])) / 2))
                current["y1"] = min(int(current["y1"]), int(segment["y1"]))
                current["y2"] = max(int(current["y2"]), int(segment["y2"]))
                current["thickness_px"] = max(float(current["thickness_px"]), float(segment["thickness_px"]))
                continue
        merged.append(segment.copy())
    return merged


def _recover_short_wall_segments(
    clean: np.ndarray,
    plan_bbox: tuple[int, int, int, int],
    existing_segments: list[dict[str, int | float | str]],
    config: ParserConfig,
) -> list[dict[str, int | float | str]]:
    horizontal_mask = cv2.morphologyEx(
        clean,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (config.short_wall_kernel_px, 1)),
    )
    vertical_mask = cv2.morphologyEx(
        clean,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, config.short_wall_kernel_px)),
    )
    short_horizontal = _filter_segments_to_bbox(
        _merge_horizontal_segments(
            _extract_line_segments(horizontal_mask, "horizontal", config.short_wall_min_length_px),
            2,
            config.alignment_tolerance_px,
        ),
        plan_bbox,
    )
    short_vertical = _filter_segments_to_bbox(
        _merge_vertical_segments(
            _extract_line_segments(vertical_mask, "vertical", config.short_wall_min_length_px),
            2,
            config.alignment_tolerance_px,
        ),
        plan_bbox,
    )

    recovered: list[dict[str, int | float | str]] = []
    for candidate in short_horizontal + short_vertical:
        if _segment_length(candidate) >= config.min_wall_length_px:
            continue
        if _segment_overlaps_existing(candidate, existing_segments + recovered, config.short_wall_connection_tolerance_px):
            continue
        anchored = _anchored_short_segment(candidate, existing_segments, config.short_wall_connection_tolerance_px)
        bump = _is_perpendicular_bump(candidate, existing_segments, config)
        if not anchored and not bump:
            continue
        recovered.append(candidate)
    return recovered


def _filter_segments_to_bbox(
    segments: list[dict[str, int | float | str]],
    bbox: tuple[int, int, int, int],
    padding: int = 10,
) -> list[dict[str, int | float | str]]:
    x, y, w, h = bbox
    x1, y1, x2, y2 = x, y, x + w - 1, y + h - 1
    filtered: list[dict[str, int | float | str]] = []
    for segment in segments:
        start, end = _segment_start_end(segment)
        center = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        if x1 - padding <= center[0] <= x2 + padding and y1 - padding <= center[1] <= y2 + padding:
            filtered.append(segment)
    return filtered


def _segment_start_end(segment: dict[str, int | float | str]) -> tuple[list[int], list[int]]:
    if segment["orientation"] == "horizontal":
        return [int(segment["x1"]), int(segment["y"])], [int(segment["x2"]), int(segment["y"])]
    return [int(segment["x"]), int(segment["y1"])], [int(segment["x"]), int(segment["y2"])]


def _segment_length(segment: dict[str, int | float | str]) -> int:
    start, end = _segment_start_end(segment)
    if segment["orientation"] == "horizontal":
        return abs(end[0] - start[0])
    return abs(end[1] - start[1])


def _point_near_segment(point: list[int], segment: dict[str, int | float | str], tolerance: int) -> bool:
    start, end = _segment_start_end(segment)
    if segment["orientation"] == "horizontal":
        left, right = sorted((start[0], end[0]))
        return left - tolerance <= point[0] <= right + tolerance and abs(point[1] - start[1]) <= tolerance
    top, bottom = sorted((start[1], end[1]))
    return top - tolerance <= point[1] <= bottom + tolerance and abs(point[0] - start[0]) <= tolerance


def _segment_overlaps_existing(
    candidate: dict[str, int | float | str],
    existing_segments: list[dict[str, int | float | str]],
    tolerance: int,
) -> bool:
    for existing in existing_segments:
        if existing["orientation"] != candidate["orientation"]:
            continue
        if candidate["orientation"] == "horizontal":
            if abs(int(candidate["y"]) - int(existing["y"])) > tolerance:
                continue
            if int(candidate["x1"]) <= int(existing["x2"]) + tolerance and int(candidate["x2"]) >= int(existing["x1"]) - tolerance:
                return True
        else:
            if abs(int(candidate["x"]) - int(existing["x"])) > tolerance:
                continue
            if int(candidate["y1"]) <= int(existing["y2"]) + tolerance and int(candidate["y2"]) >= int(existing["y1"]) - tolerance:
                return True
    return False


def _anchored_short_segment(
    candidate: dict[str, int | float | str],
    existing_segments: list[dict[str, int | float | str]],
    tolerance: int,
) -> bool:
    anchors = 0
    for point in _segment_start_end(candidate):
        if any(_point_near_segment(point, existing, tolerance) for existing in existing_segments):
            anchors += 1
    return anchors >= 2


def _is_perpendicular_bump(
    candidate: dict[str, int | float | str],
    existing_segments: list[dict[str, int | float | str]],
    config: ParserConfig,
) -> bool:
    if _segment_length(candidate) > config.short_wall_bump_max_length_px:
        return False
    anchor_points = []
    for point in _segment_start_end(candidate):
        matches = [existing for existing in existing_segments if _point_near_segment(point, existing, config.short_wall_connection_tolerance_px)]
        anchor_points.append((point, matches))
    anchored = [(point, matches) for point, matches in anchor_points if matches]
    free = [(point, matches) for point, matches in anchor_points if not matches]
    if len(anchored) != 1 or len(free) != 1:
        return False
    return any(existing["orientation"] != candidate["orientation"] for existing in anchored[0][1])


def _classify_walls(
    segments: list[dict[str, int | float | str]],
    plan_bbox: tuple[int, int, int, int],
    config: ParserConfig,
) -> list[WallSegment]:
    x, y, w, h = plan_bbox
    x1, y1, x2, y2 = x, y, x + w - 1, y + h - 1
    classified: list[WallSegment] = []
    for index, segment in enumerate(segments, start=1):
        start, end = _segment_start_end(segment)
        length = _segment_length(segment)
        kind = "partition"
        if segment["orientation"] == "horizontal":
            line_y = start[1]
            if (abs(line_y - y1) <= config.outer_wall_margin_px or abs(line_y - y2) <= config.outer_wall_margin_px) and length >= w * 0.35:
                kind = "outer"
            classified.append(
                WallSegment(
                    id=f"w{index}",
                    orientation="horizontal",
                    center=line_y,
                    span_start=start[0],
                    span_end=end[0],
                    thickness_px=float(segment["thickness_px"]),
                    kind=kind,
                )
            )
        else:
            line_x = start[0]
            if (abs(line_x - x1) <= config.outer_wall_margin_px or abs(line_x - x2) <= config.outer_wall_margin_px) and length >= h * 0.35:
                kind = "outer"
            classified.append(
                WallSegment(
                    id=f"w{index}",
                    orientation="vertical",
                    center=line_x,
                    span_start=start[1],
                    span_end=end[1],
                    thickness_px=float(segment["thickness_px"]),
                    kind=kind,
                )
            )
    return reclassify_outer_walls(classified, config)


def reclassify_outer_walls(
    walls: list[WallSegment],
    config: ParserConfig,
) -> list[WallSegment]:
    if not walls:
        return []

    horizontal = [wall for wall in walls if wall.orientation == "horizontal"]
    vertical = [wall for wall in walls if wall.orientation == "vertical"]
    if not horizontal or not vertical:
        return walls

    min_y = min(wall.center for wall in horizontal)
    max_y = max(wall.center for wall in horizontal)
    min_x = min(wall.center for wall in vertical)
    max_x = max(wall.center for wall in vertical)
    overall_width = max(wall.span_end for wall in horizontal) - min(wall.span_start for wall in horizontal)
    overall_height = max(wall.span_end for wall in vertical) - min(wall.span_start for wall in vertical)
    perimeter_margin = config.outer_wall_margin_px + 8
    horizontal_outer_lines = _outer_line_centers(
        horizontal,
        overall_width,
        perimeter_margin,
        min_y,
        max_y,
        config.alignment_tolerance_px + 2,
    )
    vertical_outer_lines = _outer_line_centers(
        vertical,
        overall_height,
        perimeter_margin,
        min_x,
        max_x,
        config.alignment_tolerance_px + 2,
    )

    reclassified: list[WallSegment] = []
    for wall in walls:
        kind = "partition"
        if wall.orientation == "horizontal":
            if any(abs(wall.center - line_center) <= config.alignment_tolerance_px + 2 for line_center in horizontal_outer_lines):
                kind = "outer"
        else:
            if any(abs(wall.center - line_center) <= config.alignment_tolerance_px + 2 for line_center in vertical_outer_lines):
                kind = "outer"
        reclassified.append(
            WallSegment(
                id=wall.id,
                orientation=wall.orientation,
                center=wall.center,
                span_start=wall.span_start,
                span_end=wall.span_end,
                thickness_px=wall.thickness_px,
                kind=kind,
            )
        )
    return reclassified


def segment_walls_by_outside_adjacency(
    shape: tuple[int, int],
    walls: list[WallSegment],
    config: ParserConfig,
) -> list[WallSegment]:
    if not walls:
        return []

    wall_mask = render_wall_mask(shape, walls)
    outside_mask = _build_outside_mask(wall_mask)
    segmented: list[WallSegment] = []
    for wall in walls:
        segments = _split_wall_by_outside_mask(wall, outside_mask, config)
        if segments:
            segmented.extend(segments)
        else:
            segmented.append(wall)
    segmented = _promote_envelope_connectors(segmented, config)
    segmented = _prune_contained_walls(segmented, config)
    return _collapse_redundant_base_splits(segmented, config)


def _outer_line_centers(
    walls: list[WallSegment],
    overall_span: int,
    perimeter_margin: int,
    min_center: int,
    max_center: int,
    alignment_tolerance: int,
) -> set[int]:
    eligible = [
        wall
        for wall in walls
        if abs(wall.center - min_center) <= perimeter_margin or abs(wall.center - max_center) <= perimeter_margin
    ]
    line_groups: list[dict[str, object]] = []
    for wall in sorted(eligible, key=lambda item: (item.center, item.span_start)):
        group = next(
            (item for item in line_groups if abs(int(item["center"]) - wall.center) <= alignment_tolerance),
            None,
        )
        if group is None:
            line_groups.append({"center": wall.center, "walls": [wall]})
        else:
            group["walls"].append(wall)
            group["center"] = int(round((int(group["center"]) + wall.center) / 2.0))

    outer_lines: set[int] = set()
    for group in line_groups:
        grouped_walls = list(group["walls"])
        total_span = sum(abs(wall.span_end - wall.span_start) for wall in grouped_walls)
        if total_span >= overall_span * 0.35:
            outer_lines.add(int(group["center"]))
    return outer_lines


def split_walls_at_intersections(
    walls: list[WallSegment],
    config: ParserConfig,
) -> list[WallSegment]:
    if not walls:
        return []

    tolerance = config.alignment_tolerance_px + 2
    split_walls: list[WallSegment] = []
    for wall in walls:
        cut_points = {wall.span_start, wall.span_end}
        for other in walls:
            if other.id == wall.id or other.orientation == wall.orientation:
                continue
            if wall.orientation == "horizontal":
                intersects = (
                    wall.span_start + 2 < other.center < wall.span_end - 2
                    and other.span_start - tolerance <= wall.center <= other.span_end + tolerance
                )
                if intersects:
                    cut_points.add(other.center)
            else:
                intersects = (
                    wall.span_start + 2 < other.center < wall.span_end - 2
                    and other.span_start - tolerance <= wall.center <= other.span_end + tolerance
                )
                if intersects:
                    cut_points.add(other.center)

        ordered = sorted(cut_points)
        if len(ordered) <= 2:
            split_walls.append(wall)
            continue

        parts: list[tuple[int, int]] = []
        for start, end in zip(ordered[:-1], ordered[1:]):
            if end - start < max(2, config.min_wall_stub_px):
                continue
            parts.append((start, end))

        if len(parts) <= 1:
            split_walls.append(wall)
            continue

        for index, (start, end) in enumerate(parts, start=1):
            split_walls.append(
                WallSegment(
                    id=f"{wall.id}_{index}",
                    orientation=wall.orientation,
                    center=wall.center,
                    span_start=start,
                    span_end=end,
                    thickness_px=wall.thickness_px,
                    kind=wall.kind,
                )
            )
    return split_walls


def _build_outside_mask(wall_mask: np.ndarray) -> np.ndarray:
    free = np.where(wall_mask == 0, 255, 0).astype(np.uint8)
    flood = free.copy()
    h, w = flood.shape
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 128)
    outside = np.zeros_like(flood)
    outside[flood == 128] = 255
    return outside


def _split_wall_by_outside_mask(
    wall: WallSegment,
    outside_mask: np.ndarray,
    config: ParserConfig,
) -> list[WallSegment]:
    sample_step = max(2, config.alignment_tolerance_px)
    normal_offset = max(4, int(round(wall.thickness_px / 2.0)) + 2)
    samples: list[tuple[int, bool]] = []

    if wall.orientation == "horizontal":
        start = min(wall.span_start, wall.span_end)
        end = max(wall.span_start, wall.span_end)
        for x in range(start, end + 1, sample_step):
            above = _outside_at(outside_mask, x, wall.center - normal_offset)
            below = _outside_at(outside_mask, x, wall.center + normal_offset)
            samples.append((x, above or below))
        if not samples or samples[-1][0] != end:
            above = _outside_at(outside_mask, end, wall.center - normal_offset)
            below = _outside_at(outside_mask, end, wall.center + normal_offset)
            samples.append((end, above or below))
    else:
        start = min(wall.span_start, wall.span_end)
        end = max(wall.span_start, wall.span_end)
        for y in range(start, end + 1, sample_step):
            left = _outside_at(outside_mask, wall.center - normal_offset, y)
            right = _outside_at(outside_mask, wall.center + normal_offset, y)
            samples.append((y, left or right))
        if not samples or samples[-1][0] != end:
            left = _outside_at(outside_mask, wall.center - normal_offset, end)
            right = _outside_at(outside_mask, wall.center + normal_offset, end)
            samples.append((end, left or right))

    if not samples:
        return []

    runs: list[tuple[int, int, bool]] = []
    run_start = samples[0][0]
    run_state = samples[0][1]
    prev_coord = samples[0][0]
    for coord, state in samples[1:]:
        if state != run_state:
            runs.append((run_start, prev_coord, run_state))
            run_start = coord
            run_state = state
        prev_coord = coord
    runs.append((run_start, samples[-1][0], run_state))
    runs = _merge_short_state_flips(runs, max(sample_step * 2, config.min_wall_stub_px))

    expanded_runs: list[tuple[int, int, bool]] = []
    for index, (run_start, run_end, state) in enumerate(runs):
        next_start = runs[index + 1][0] if index + 1 < len(runs) else run_end
        segment_end = next_start if index + 1 < len(runs) else run_end
        expanded_runs.append((run_start, segment_end, state))

    min_length = max(2, config.min_wall_stub_px)
    valid_runs = [(run_start, run_end, state) for run_start, run_end, state in expanded_runs if run_end - run_start >= min_length]
    valid_runs = _absorb_short_edge_runs(valid_runs, max(sample_step * 2, config.min_wall_stub_px * 2))
    if len(valid_runs) <= 1:
        has_outside = any(state for _, _, state in runs)
        kind = "outer" if has_outside else "partition"
        return [
            WallSegment(
                id=wall.id,
                orientation=wall.orientation,
                center=wall.center,
                span_start=wall.span_start,
                span_end=wall.span_end,
                thickness_px=wall.thickness_px,
                kind=kind,
            )
        ]

    split_segments: list[WallSegment] = []
    for index, (run_start, run_end, state) in enumerate(valid_runs, start=1):
        split_segments.append(
            WallSegment(
                id=f"{wall.id}_{index}",
                orientation=wall.orientation,
                center=wall.center,
                span_start=int(run_start),
                span_end=int(run_end),
                thickness_px=wall.thickness_px,
                kind="outer" if state else "partition",
            )
        )
    return _merge_adjacent_wall_segments(split_segments, config)


def _merge_short_state_flips(
    runs: list[tuple[int, int, bool]],
    min_run_length: int,
) -> list[tuple[int, int, bool]]:
    if len(runs) < 3:
        return runs

    smoothed = list(runs)
    changed = True
    while changed and len(smoothed) >= 3:
        changed = False
        merged: list[tuple[int, int, bool]] = []
        index = 0
        while index < len(smoothed):
            if 0 < index < len(smoothed) - 1:
                start, end, state = smoothed[index]
                prev_start, _, prev_state = smoothed[index - 1]
                _, next_end, next_state = smoothed[index + 1]
                if prev_state == next_state and state != prev_state and end - start < min_run_length:
                    merged.pop()
                    merged.append((prev_start, next_end, prev_state))
                    index += 2
                    changed = True
                    continue
            merged.append(smoothed[index])
            index += 1
        smoothed = merged
    return smoothed


def _merge_adjacent_wall_segments(
    segments: list[WallSegment],
    config: ParserConfig,
) -> list[WallSegment]:
    if not segments:
        return []

    merged: list[WallSegment] = [segments[0]]
    for segment in segments[1:]:
        current = merged[-1]
        touching = abs(segment.span_start - current.span_end) <= max(1, config.alignment_tolerance_px)
        same_line = segment.orientation == current.orientation and segment.center == current.center
        same_kind = segment.kind == current.kind
        if same_line and same_kind and touching:
            merged[-1] = WallSegment(
                id=current.id,
                orientation=current.orientation,
                center=current.center,
                span_start=current.span_start,
                span_end=segment.span_end,
                thickness_px=max(current.thickness_px, segment.thickness_px),
                kind=current.kind,
            )
            continue
        merged.append(segment)

    if len(merged) == 1:
        single = merged[0]
        return [
            WallSegment(
                id=single.id.split("_")[0],
                orientation=single.orientation,
                center=single.center,
                span_start=single.span_start,
                span_end=single.span_end,
                thickness_px=single.thickness_px,
                kind=single.kind,
            )
        ]

    normalized: list[WallSegment] = []
    base_id = merged[0].id.split("_")[0]
    for index, segment in enumerate(merged, start=1):
        normalized.append(
            WallSegment(
                id=f"{base_id}_{index}",
                orientation=segment.orientation,
                center=segment.center,
                span_start=segment.span_start,
                span_end=segment.span_end,
                thickness_px=segment.thickness_px,
                kind=segment.kind,
            )
        )
    return normalized


def _absorb_short_edge_runs(
    runs: list[tuple[int, int, bool]],
    edge_threshold: int,
) -> list[tuple[int, int, bool]]:
    if len(runs) <= 1:
        return runs

    adjusted = list(runs)
    if len(adjusted) >= 2 and adjusted[0][1] - adjusted[0][0] < edge_threshold:
        _, _, next_state = adjusted[1]
        if not adjusted[0][2] and next_state:
            adjusted[0] = (adjusted[0][0], adjusted[0][1], next_state)
    if len(adjusted) >= 2 and adjusted[-1][1] - adjusted[-1][0] < edge_threshold:
        prev_state = adjusted[-2][2]
        if not adjusted[-1][2] and prev_state:
            adjusted[-1] = (adjusted[-1][0], adjusted[-1][1], prev_state)

    collapsed: list[tuple[int, int, bool]] = []
    for start, end, state in adjusted:
        if collapsed and collapsed[-1][2] == state and start <= collapsed[-1][1]:
            prev_start, prev_end, prev_state = collapsed[-1]
            collapsed[-1] = (prev_start, max(prev_end, end), prev_state)
            continue
        if collapsed and collapsed[-1][2] == state:
            prev_start, prev_end, prev_state = collapsed[-1]
            collapsed[-1] = (prev_start, end, prev_state)
            continue
        collapsed.append((start, end, state))
    return collapsed


def _promote_envelope_connectors(
    walls: list[WallSegment],
    config: ParserConfig,
) -> list[WallSegment]:
    if not walls:
        return []

    tolerance = config.alignment_tolerance_px + 4
    promoted: list[WallSegment] = []
    outer_walls = [wall for wall in walls if wall.kind == "outer"]
    base_counts: dict[str, int] = {}
    for wall in walls:
        base_id = wall.id.split("_")[0]
        base_counts[base_id] = base_counts.get(base_id, 0) + 1
    for wall in walls:
        kind = wall.kind
        base_id = wall.id.split("_")[0]
        if kind != "outer" and base_counts.get(base_id, 0) == 1 and _touches_outer_at_both_ends(wall, outer_walls, tolerance):
            kind = "outer"
        promoted.append(
            WallSegment(
                id=wall.id,
                orientation=wall.orientation,
                center=wall.center,
                span_start=wall.span_start,
                span_end=wall.span_end,
                thickness_px=wall.thickness_px,
                kind=kind,
            )
        )
    return promoted


def _touches_outer_at_both_ends(
    wall: WallSegment,
    outer_walls: list[WallSegment],
    tolerance: int,
) -> bool:
    endpoints = (
        (wall.span_start, wall.center) if wall.orientation == "horizontal" else (wall.center, wall.span_start),
        (wall.span_end, wall.center) if wall.orientation == "horizontal" else (wall.center, wall.span_end),
    )
    connected_ends = 0
    for point in endpoints:
        if any(_point_on_wall(point, outer, tolerance) for outer in outer_walls if outer.id != wall.id):
            connected_ends += 1
    return connected_ends == 2


def _point_on_wall(point: tuple[int, int], wall: WallSegment, tolerance: int) -> bool:
    x, y = point
    if wall.orientation == "horizontal":
        return wall.span_start - tolerance <= x <= wall.span_end + tolerance and abs(y - wall.center) <= tolerance
    return wall.span_start - tolerance <= y <= wall.span_end + tolerance and abs(x - wall.center) <= tolerance


def _prune_contained_walls(
    walls: list[WallSegment],
    config: ParserConfig,
) -> list[WallSegment]:
    kept: list[WallSegment] = []
    for wall in walls:
        if _is_contained_in_other_wall(wall, walls, config):
            continue
        kept.append(wall)
    return kept


def _collapse_redundant_base_splits(
    walls: list[WallSegment],
    config: ParserConfig,
) -> list[WallSegment]:
    grouped: dict[str, list[WallSegment]] = {}
    passthrough: list[WallSegment] = []
    for wall in walls:
        if "_" not in wall.id:
            passthrough.append(wall)
            continue
        base_id = wall.id.split("_")[0]
        grouped.setdefault(base_id, []).append(wall)

    collapsed: list[WallSegment] = []
    gap_tolerance = max(config.alignment_tolerance_px + 2, config.min_wall_stub_px * 2)
    for base_id, segments in grouped.items():
        if len(segments) <= 1:
            collapsed.extend(segments)
            continue
        same_orientation = len({segment.orientation for segment in segments}) == 1
        same_center = len({segment.center for segment in segments}) == 1
        same_kind = len({segment.kind for segment in segments}) == 1
        if not (same_orientation and same_center and same_kind):
            collapsed.extend(_renumber_base_segments(base_id, segments))
            continue

        ordered = sorted(segments, key=lambda segment: segment.span_start)
        gaps = [ordered[index + 1].span_start - ordered[index].span_end for index in range(len(ordered) - 1)]
        if gaps and max(gaps) > gap_tolerance:
            collapsed.extend(_renumber_base_segments(base_id, ordered))
            continue

        collapsed.append(
            WallSegment(
                id=base_id,
                orientation=ordered[0].orientation,
                center=ordered[0].center,
                span_start=min(segment.span_start for segment in ordered),
                span_end=max(segment.span_end for segment in ordered),
                thickness_px=max(segment.thickness_px for segment in ordered),
                kind=ordered[0].kind,
            )
        )

    result = passthrough + collapsed
    result.sort(key=lambda wall: (wall.orientation, wall.center, wall.span_start, wall.id))
    return result


def _renumber_base_segments(base_id: str, segments: list[WallSegment]) -> list[WallSegment]:
    ordered = sorted(segments, key=lambda segment: segment.span_start)
    renumbered: list[WallSegment] = []
    for index, segment in enumerate(ordered, start=1):
        renumbered.append(
            WallSegment(
                id=f"{base_id}_{index}",
                orientation=segment.orientation,
                center=segment.center,
                span_start=segment.span_start,
                span_end=segment.span_end,
                thickness_px=segment.thickness_px,
                kind=segment.kind,
            )
        )
    return renumbered


def _is_contained_in_other_wall(
    wall: WallSegment,
    walls: list[WallSegment],
    config: ParserConfig,
) -> bool:
    tolerance = config.alignment_tolerance_px + 2
    wall_box = _wall_bounds(wall, tolerance)
    wall_length = abs(wall.span_end - wall.span_start)
    for other in walls:
        if other.id == wall.id:
            continue
        other_length = abs(other.span_end - other.span_start)
        if other_length <= wall_length + tolerance:
            continue
        other_box = _wall_bounds(other, tolerance)
        if _bounds_contains(other_box, wall_box) or _wall_endpoints_contained_by_box(wall, other_box):
            return True
    return False


def _wall_bounds(
    wall: WallSegment,
    tolerance: int,
) -> tuple[int, int, int, int]:
    half_thickness = max(2, int(round(wall.thickness_px / 2.0))) + tolerance
    if wall.orientation == "horizontal":
        x1 = min(wall.span_start, wall.span_end) - tolerance
        x2 = max(wall.span_start, wall.span_end) + tolerance
        y1 = wall.center - half_thickness
        y2 = wall.center + half_thickness
    else:
        x1 = wall.center - half_thickness
        x2 = wall.center + half_thickness
        y1 = min(wall.span_start, wall.span_end) - tolerance
        y2 = max(wall.span_start, wall.span_end) + tolerance
    return x1, y1, x2, y2


def _bounds_contains(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> bool:
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    return ox1 <= ix1 and oy1 <= iy1 and ix2 <= ox2 and iy2 <= oy2


def _wall_endpoints_contained_by_box(
    wall: WallSegment,
    bounds: tuple[int, int, int, int],
) -> bool:
    if wall.orientation == "horizontal":
        endpoints = ((wall.span_start, wall.center), (wall.span_end, wall.center))
    else:
        endpoints = ((wall.center, wall.span_start), (wall.center, wall.span_end))
    return all(_point_in_bounds(point, bounds) for point in endpoints)


def _point_in_bounds(
    point: tuple[int, int],
    bounds: tuple[int, int, int, int],
) -> bool:
    x, y = point
    x1, y1, x2, y2 = bounds
    return x1 <= x <= x2 and y1 <= y <= y2


def _outside_at(mask: np.ndarray, x: int, y: int) -> bool:
    if x < 0 or y < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
        return True
    return bool(mask[y, x] > 0)


def _bridge_wall_segments(
    segments: list[dict[str, int | float | str]],
    config: ParserConfig,
) -> list[dict[str, int | float | str]]:
    horizontal = [segment.copy() for segment in segments if segment["orientation"] == "horizontal"]
    vertical = [segment.copy() for segment in segments if segment["orientation"] == "vertical"]
    bridged_horizontal = _bridge_orientation_segments(horizontal, "horizontal", config)
    bridged_vertical = _bridge_orientation_segments(vertical, "vertical", config)
    combined = bridged_horizontal + bridged_vertical
    combined.sort(
        key=lambda segment: (
            str(segment["orientation"]),
            int(segment["y"]) if segment["orientation"] == "horizontal" else int(segment["x"]),
            int(segment["x1"]) if segment["orientation"] == "horizontal" else int(segment["y1"]),
        )
    )
    return combined


def _bridge_orientation_segments(
    segments: list[dict[str, int | float | str]],
    orientation: str,
    config: ParserConfig,
) -> list[dict[str, int | float | str]]:
    if orientation == "horizontal":
        segments.sort(key=lambda s: (int(s["y"]), int(s["x1"])))
    else:
        segments.sort(key=lambda s: (int(s["x"]), int(s["y1"])))

    changed = True
    while changed:
        changed = False
        result: list[dict[str, int | float | str]] = []
        index = 0
        while index < len(segments):
            current = segments[index].copy()
            compare_index = index + 1
            while compare_index < len(segments):
                other = segments[compare_index]
                if not _can_bridge(current, other, orientation, config):
                    compare_index += 1
                    continue
                current = _merge_bridged_pair(current, other, orientation)
                changed = True
                segments.pop(compare_index)
                break
            result.append(current)
            index += 1
        segments = result
        if orientation == "horizontal":
            segments.sort(key=lambda s: (int(s["y"]), int(s["x1"])))
        else:
            segments.sort(key=lambda s: (int(s["x"]), int(s["y1"])))
    return segments


def _can_bridge(
    left: dict[str, int | float | str],
    right: dict[str, int | float | str],
    orientation: str,
    config: ParserConfig,
) -> bool:
    if orientation == "horizontal":
        if abs(int(left["y"]) - int(right["y"])) > config.wall_bridge_alignment_px:
            return False
        thickness_delta = abs(float(left["thickness_px"]) - float(right["thickness_px"]))
        if thickness_delta > config.wall_bridge_thickness_delta_px:
            return False
        left_start, left_end = sorted((int(left["x1"]), int(left["x2"])))
        right_start, right_end = sorted((int(right["x1"]), int(right["x2"])))
        gap = right_start - left_end
    else:
        if abs(int(left["x"]) - int(right["x"])) > config.wall_bridge_alignment_px:
            return False
        thickness_delta = abs(float(left["thickness_px"]) - float(right["thickness_px"]))
        if thickness_delta > config.wall_bridge_thickness_delta_px:
            return False
        left_start, left_end = sorted((int(left["y1"]), int(left["y2"])))
        right_start, right_end = sorted((int(right["y1"]), int(right["y2"])))
        gap = right_start - left_end

    return config.wall_bridge_gap_min_px <= gap <= config.wall_bridge_gap_max_px


def _merge_bridged_pair(
    first: dict[str, int | float | str],
    second: dict[str, int | float | str],
    orientation: str,
) -> dict[str, int | float | str]:
    merged = first.copy()
    if orientation == "horizontal":
        merged["y"] = int(round((int(first["y"]) + int(second["y"])) / 2))
        merged["x1"] = min(int(first["x1"]), int(second["x1"]))
        merged["x2"] = max(int(first["x2"]), int(second["x2"]))
    else:
        merged["x"] = int(round((int(first["x"]) + int(second["x"])) / 2))
        merged["y1"] = min(int(first["y1"]), int(second["y1"]))
        merged["y2"] = max(int(first["y2"]), int(second["y2"]))
    merged["thickness_px"] = max(float(first["thickness_px"]), float(second["thickness_px"]))
    return merged


def render_wall_mask(shape: tuple[int, int], walls: Iterable[WallSegment]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for wall in walls:
        thickness = max(3, int(round(wall.thickness_px)))
        if wall.orientation == "horizontal":
            cv2.line(mask, (wall.span_start, wall.center), (wall.span_end, wall.center), 255, thickness)
        else:
            cv2.line(mask, (wall.center, wall.span_start), (wall.center, wall.span_end), 255, thickness)
    return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
