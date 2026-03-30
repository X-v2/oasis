from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from parser.config import ParserConfig
from parser.walls import WallSegment


@dataclass(slots=True)
class Door:
    id: str
    wall_id: str
    center: tuple[int, int]
    width_px: float
    swing: str


@dataclass(slots=True)
class DoorCandidate:
    wall_id: str
    center: tuple[int, int]
    width_px: float
    radius_px: int
    wall_kind: str
    width_ok: bool
    opening_ok: bool


@dataclass(slots=True)
class DoorScanBand:
    wall_id: str
    orientation: str
    rect: tuple[int, int, int, int]


def detect_doors(
    gray_image: np.ndarray,
    binary_inv: np.ndarray,
    walls: list[WallSegment],
    plan_bbox: tuple[int, int, int, int],
    config: ParserConfig,
) -> tuple[list[Door], np.ndarray]:
    del gray_image, plan_bbox
    candidates = inspect_door_candidates(binary_inv, walls, config)
    doors: list[Door] = []
    door_mask = np.zeros_like(binary_inv)
    door_index = 1

    for candidate in candidates["accepted"]:
        wall = next(w for w in walls if w.id == candidate.wall_id)
        door = Door(
            id=f"d{door_index}",
            wall_id=wall.id,
            center=candidate.center,
            width_px=float(candidate.width_px),
            swing=_infer_swing(candidate.center, wall),
        )
        doors.append(door)
        cv2.circle(door_mask, candidate.center, candidate.radius_px, 255, thickness=2)
        door_index += 1

    return doors, door_mask


def inspect_door_candidates(
    binary_inv: np.ndarray,
    walls: list[WallSegment],
    config: ParserConfig,
) -> dict[str, list]:
    all_candidates: list[DoorCandidate] = []
    width_pass: list[DoorCandidate] = []
    opening_pass: list[DoorCandidate] = []
    accepted: list[DoorCandidate] = []
    scan_bands: list[DoorScanBand] = []
    accepted_wall_ids: set[str] = set()

    for wall in walls:
        scan_bands.append(_scan_band_for_wall(binary_inv.shape, wall))
        raw_gaps = _find_wall_gaps(binary_inv, wall)
        merged_gaps = _merge_nearby_gaps(raw_gaps, config.door_gap_cluster_merge_px)
        for gap_start, gap_end in merged_gaps:
            gap_width = gap_end - gap_start
            center = _gap_center(wall, gap_start, gap_end)
            radius = max(8, int(round(gap_width / 2.0)))
            width_ok = True
            opening_ok = False
            if width_ok:
                opening_ok = _gap_has_clear_opening(binary_inv, wall, gap_start, gap_end)
            dominant_ok = gap_width >= config.door_min_merged_gap_px
            candidate = DoorCandidate(
                wall_id=wall.id,
                center=center,
                width_px=float(gap_width),
                radius_px=radius,
                wall_kind=wall.kind,
                width_ok=width_ok and dominant_ok,
                opening_ok=opening_ok and dominant_ok,
            )
            all_candidates.append(candidate)
            if width_ok and dominant_ok:
                width_pass.append(candidate)
            if width_ok and opening_ok and dominant_ok:
                opening_pass.append(candidate)
                accepted.append(candidate)
                accepted_wall_ids.add(wall.id)

    thin_mask = _build_thin_symbol_mask(binary_inv)
    for wall in walls:
        if wall.id in accepted_wall_ids:
            continue
        fallback = _symbol_candidate_for_wall(thin_mask, wall, walls, config)
        if fallback is None:
            continue
        all_candidates.append(fallback)
        width_pass.append(fallback)
        opening_pass.append(fallback)
        accepted.append(fallback)

    return {
        "scan_bands": scan_bands,
        "all_candidates": all_candidates,
        "width_pass": width_pass,
        "opening_pass": opening_pass,
        "accepted": accepted,
    }


def _scan_band_for_wall(shape: tuple[int, int], wall: WallSegment) -> DoorScanBand:
    band_half = max(3, int(round(wall.thickness_px / 2.0)))
    if wall.orientation == "horizontal":
        x1 = max(0, wall.span_start)
        x2 = min(shape[1] - 1, wall.span_end)
        y1 = max(0, wall.center - band_half)
        y2 = min(shape[0] - 1, wall.center + band_half)
    else:
        x1 = max(0, wall.center - band_half)
        x2 = min(shape[1] - 1, wall.center + band_half)
        y1 = max(0, wall.span_start)
        y2 = min(shape[0] - 1, wall.span_end)
    return DoorScanBand(
        wall_id=wall.id,
        orientation=wall.orientation,
        rect=(x1, y1, x2, y2),
    )


def _find_wall_gaps(binary_inv: np.ndarray, wall: WallSegment) -> list[tuple[int, int]]:
    band_half = max(3, int(round(wall.thickness_px / 2.0)))
    if wall.orientation == "horizontal":
        y1 = max(0, wall.center - band_half)
        y2 = min(binary_inv.shape[0], wall.center + band_half + 1)
        x1 = max(0, wall.span_start)
        x2 = min(binary_inv.shape[1], wall.span_end)
        band = binary_inv[y1:y2, x1:x2]
        projection = band.mean(axis=0)
    else:
        x1 = max(0, wall.center - band_half)
        x2 = min(binary_inv.shape[1], wall.center + band_half + 1)
        y1 = max(0, wall.span_start)
        y2 = min(binary_inv.shape[0], wall.span_end)
        band = binary_inv[y1:y2, x1:x2]
        projection = band.mean(axis=1)
    return _low_ink_spans(projection)


def _low_ink_spans(projection: np.ndarray) -> list[tuple[int, int]]:
    if projection.size == 0:
        return []
    max_value = float(np.max(projection))
    threshold = max(28.0, max_value * 0.22)
    is_gap = projection <= threshold
    spans: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(is_gap):
        if value and start is None:
            start = index
        elif not value and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(is_gap)))
    return spans


def _merge_nearby_gaps(gaps: list[tuple[int, int]], merge_distance: int) -> list[tuple[int, int]]:
    if not gaps:
        return []
    ordered = sorted(gaps, key=lambda item: item[0])
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= merge_distance:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _build_thin_symbol_mask(mask: np.ndarray) -> np.ndarray:
    if np.count_nonzero(mask) == 0:
        return mask.copy()
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    thin = np.zeros_like(mask)
    thin[(mask > 0) & (distance <= 2.5)] = 255
    thin = cv2.morphologyEx(thin, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return thin


def _symbol_candidate_for_wall(
    thin_mask: np.ndarray,
    wall: WallSegment,
    walls: list[WallSegment],
    config: ParserConfig,
) -> DoorCandidate | None:
    if wall.orientation == "horizontal":
        x1 = max(0, wall.span_start)
        x2 = min(thin_mask.shape[1], wall.span_end)
        y1 = max(0, wall.center - 70)
        y2 = min(thin_mask.shape[0], wall.center + 10)
    else:
        x1 = max(0, wall.center - 10)
        x2 = min(thin_mask.shape[1], wall.center + 70)
        y1 = max(0, wall.span_start)
        y2 = min(thin_mask.shape[0], wall.span_end)
    roi = thin_mask[y1:y2, x1:x2]
    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180.0,
        threshold=10,
        minLineLength=config.door_symbol_line_min_px,
        maxLineGap=6,
    )
    if lines is None:
        return None

    best: DoorCandidate | None = None
    best_length = -1.0
    for raw in lines[:, 0, :]:
        line = [int(raw[0]) + x1, int(raw[1]) + y1, int(raw[2]) + x1, int(raw[3]) + y1]
        length = float(np.hypot(line[2] - line[0], line[3] - line[1]))
        if length < config.door_symbol_line_min_px or length > config.door_symbol_line_max_px:
            continue
        if _line_orientation(line) == wall.orientation:
            continue
        hinge_point, free_end, hinge_distance = _hinge_and_free_end(line, wall)
        if hinge_distance > config.door_symbol_wall_distance_px:
            continue
        line_center = ((line[0] + line[2]) // 2, (line[1] + line[3]) // 2)
        if _distance_to_wall_end(line_center, wall) < config.door_symbol_end_margin_px:
            continue
        if _near_perpendicular_crossing(line_center, wall, walls, config.door_symbol_crossing_clearance_px):
            continue
        if length > best_length:
            width_px = length * 2.0
            if width_px < config.door_symbol_min_width_px:
                continue
            center = _symbol_opening_center(wall, hinge_point, free_end, width_px)
            if _distance_to_wall_end(center, wall) < config.door_symbol_end_margin_px:
                continue
            if _near_perpendicular_crossing(center, wall, walls, config.door_symbol_crossing_clearance_px):
                continue
            best_length = length
            best = DoorCandidate(
                wall_id=wall.id,
                center=center,
                width_px=width_px,
                radius_px=max(8, int(round(length))),
                wall_kind=wall.kind,
                width_ok=True,
                opening_ok=True,
            )
    return best


def _line_orientation(line: list[int]) -> str:
    return "horizontal" if abs(line[2] - line[0]) >= abs(line[3] - line[1]) else "vertical"


def _hinge_and_free_end(
    line: list[int], wall: WallSegment
) -> tuple[tuple[int, int], tuple[int, int], float]:
    p1 = (line[0], line[1])
    p2 = (line[2], line[3])
    d1 = _endpoint_wall_distance(p1, wall)
    d2 = _endpoint_wall_distance(p2, wall)
    if d1 <= d2:
        return p1, p2, d1
    return p2, p1, d2


def _endpoint_wall_distance(point: tuple[int, int], wall: WallSegment) -> float:
    if wall.orientation == "horizontal":
        return float(abs(point[1] - wall.center))
    return float(abs(point[0] - wall.center))


def _symbol_opening_center(
    wall: WallSegment,
    hinge_point: tuple[int, int],
    free_end: tuple[int, int],
    width_px: float,
) -> tuple[int, int]:
    half_width = int(round(width_px / 2.0))
    if wall.orientation == "horizontal":
        delta = free_end[0] - hinge_point[0]
        if delta == 0:
            center_x = hinge_point[0]
        else:
            center_x = hinge_point[0] + int(np.sign(delta) * half_width)
        return (center_x, wall.center)

    delta = free_end[1] - hinge_point[1]
    if delta == 0:
        center_y = hinge_point[1]
    else:
        center_y = hinge_point[1] + int(np.sign(delta) * half_width)
    return (wall.center, center_y)


def _distance_to_wall_end(center: tuple[int, int], wall: WallSegment) -> float:
    if wall.orientation == "horizontal":
        return float(min(abs(center[0] - wall.span_start), abs(center[0] - wall.span_end)))
    return float(min(abs(center[1] - wall.span_start), abs(center[1] - wall.span_end)))


def _near_perpendicular_crossing(
    center: tuple[int, int],
    wall: WallSegment,
    walls: list[WallSegment],
    clearance: int,
) -> bool:
    for other in walls:
        if other.id == wall.id or other.orientation == wall.orientation:
            continue
        if wall.orientation == "horizontal":
            if abs(other.center - center[0]) > clearance:
                continue
            if other.span_start - clearance <= wall.center <= other.span_end + clearance:
                return True
        else:
            if abs(other.center - center[1]) > clearance:
                continue
            if other.span_start - clearance <= wall.center <= other.span_end + clearance:
                return True
    return False


def _gap_center(wall: WallSegment, gap_start: int, gap_end: int) -> tuple[int, int]:
    middle = int(round((gap_start + gap_end) / 2.0))
    if wall.orientation == "horizontal":
        return (wall.span_start + middle, wall.center)
    return (wall.center, wall.span_start + middle)


def _gap_has_clear_opening(
    binary_inv: np.ndarray, wall: WallSegment, gap_start: int, gap_end: int
) -> bool:
    depth = 16
    if wall.orientation == "horizontal":
        x1 = max(0, wall.span_start + gap_start)
        x2 = min(binary_inv.shape[1], wall.span_start + gap_end)
        upper = binary_inv[max(0, wall.center - depth) : max(0, wall.center - 2), x1:x2]
        lower = binary_inv[min(binary_inv.shape[0], wall.center + 2) : min(binary_inv.shape[0], wall.center + depth), x1:x2]
        if upper.size == 0 or lower.size == 0:
            return False
        return upper.mean() < 95 or lower.mean() < 95
    y1 = max(0, wall.span_start + gap_start)
    y2 = min(binary_inv.shape[0], wall.span_start + gap_end)
    left = binary_inv[y1:y2, max(0, wall.center - depth) : max(0, wall.center - 2)]
    right = binary_inv[y1:y2, min(binary_inv.shape[1], wall.center + 2) : min(binary_inv.shape[1], wall.center + depth)]
    if left.size == 0 or right.size == 0:
        return False
    return left.mean() < 95 or right.mean() < 95


def _infer_swing(center: tuple[int, int], wall: WallSegment) -> str:
    cx, cy = center
    if wall.orientation == "horizontal":
        midpoint = (wall.span_start + wall.span_end) / 2.0
        return "left" if cx <= midpoint else "right"
    midpoint = (wall.span_start + wall.span_end) / 2.0
    return "left" if cy <= midpoint else "right"
