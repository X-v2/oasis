from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from parser.config import ParserConfig
from parser.walls import WallSegment
from parser.doors import Door


@dataclass(slots=True)
class Window:
    id: str
    wall_id: str
    center: tuple[int, int]
    width_px: float
    bbox: tuple[int, int, int, int]
    wall_orientation: str


@dataclass(slots=True)
class WindowCandidate:
    wall_id: str
    center: tuple[int, int]
    width_px: float
    bbox: tuple[int, int, int, int]
    orientation: str
    attached_ok: bool
    overlap_ok: bool


@dataclass(slots=True)
class WindowHost:
    orientation: str
    center: int
    span_start: int
    span_end: int
    wall_ids: tuple[str, ...]


def detect_windows(
    binary_inv: np.ndarray,
    walls: list[WallSegment],
    door_mask: np.ndarray,
    config: ParserConfig,
) -> list[Window]:
    checks = inspect_window_candidates(binary_inv, walls, door_mask, config)
    windows: list[Window] = []
    for index, group in enumerate(checks["accepted_groups"], start=1):
        bbox = group["bbox"]
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        windows.append(
            Window(
                id=f"win{index}",
                wall_id=str(group["wall_id"]),
                center=center,
                width_px=float(group["length_px"]),
                bbox=bbox,
                wall_orientation=str(group["wall_orientation"]),
            )
        )
    return windows


def merge_window_host_walls(
    walls: list[WallSegment],
    doors: list[Door],
    windows: list[Window],
    config: ParserConfig,
) -> tuple[list[WallSegment], list[Door], list[Window]]:
    hosts = _build_window_hosts((0, 0), walls, config)
    host_by_wall_id: dict[str, WindowHost] = {}
    for host in hosts:
        if len(host.wall_ids) < 2:
            continue
        for wall_id in host.wall_ids:
            host_by_wall_id[wall_id] = host

    if not host_by_wall_id:
        return walls, doors, windows

    wall_lookup = {wall.id: wall for wall in walls}
    merged_walls: list[WallSegment] = []
    consumed_wall_ids: set[str] = set()
    id_remap: dict[str, str] = {}

    for wall in walls:
        host = host_by_wall_id.get(wall.id)
        if host is None:
            merged_walls.append(wall)
            continue
        if wall.id in consumed_wall_ids:
            continue

        host_walls = [wall_lookup[wall_id] for wall_id in host.wall_ids if wall_id in wall_lookup]
        if not host_walls:
            merged_walls.append(wall)
            continue

        primary_id = host.wall_ids[0]
        merged_walls.append(
            WallSegment(
                id=primary_id,
                orientation=host.orientation,
                center=int(round(sum(item.center for item in host_walls) / len(host_walls))),
                span_start=min(item.span_start for item in host_walls),
                span_end=max(item.span_end for item in host_walls),
                thickness_px=max(item.thickness_px for item in host_walls),
                kind="outer" if any(item.kind == "outer" for item in host_walls) else host_walls[0].kind,
            )
        )
        for item in host_walls:
            consumed_wall_ids.add(item.id)
            id_remap[item.id] = primary_id

    remapped_doors = [
        Door(
            id=door.id,
            wall_id=id_remap.get(door.wall_id, door.wall_id),
            center=door.center,
            width_px=door.width_px,
            swing=door.swing,
        )
        for door in doors
    ]
    remapped_windows = [
        Window(
            id=window.id,
            wall_id=id_remap.get(window.wall_id, window.wall_id),
            center=window.center,
            width_px=window.width_px,
            bbox=window.bbox,
            wall_orientation=window.wall_orientation,
        )
        for window in windows
    ]
    return merged_walls, remapped_doors, remapped_windows


def inspect_window_candidates(
    binary_inv: np.ndarray,
    walls: list[WallSegment],
    door_mask: np.ndarray,
    config: ParserConfig,
) -> dict[str, object]:
    cleaned = cv2.bitwise_and(binary_inv, cv2.bitwise_not(door_mask))
    thin_horizontal = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (config.thin_horizontal_kernel_px, 1)),
    )
    thin_vertical = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, config.thin_vertical_kernel_px)),
    )

    outer_hosts = _build_window_hosts(binary_inv.shape, walls, config)
    raw_candidates: list[WindowCandidate] = []
    attached_pass: list[WindowCandidate] = []
    overlap_pass: list[WindowCandidate] = []
    crossing_pass: list[WindowCandidate] = []
    door_filtered: list[WindowCandidate] = []
    grouped: list[dict[str, object]] = []
    for orientation, mask in (("horizontal", thin_horizontal), ("vertical", thin_vertical)):
        for candidate in _extract_window_candidates(mask, orientation, config):
            nearest = _nearest_outer_host(
                candidate["center"],
                outer_hosts,
                candidate["orientation"],
                config.window_wall_distance_px,
            )
            window_candidate = WindowCandidate(
                wall_id=_host_label(nearest),
                center=candidate["center"],
                width_px=float(candidate["length_px"]),
                bbox=candidate["bbox"],
                orientation=str(candidate["orientation"]),
                attached_ok=False,
                overlap_ok=False,
            )
            raw_candidates.append(window_candidate)
            if nearest is None:
                continue
            attached_pass.append(
                WindowCandidate(
                    wall_id=_host_label(nearest),
                    center=candidate["center"],
                    width_px=float(candidate["length_px"]),
                    bbox=candidate["bbox"],
                    orientation=str(candidate["orientation"]),
                    attached_ok=True,
                    overlap_ok=False,
                )
            )
            if _axis_overlap(candidate, nearest) < candidate["length_px"] * 0.3:
                continue
            overlap_pass.append(
                WindowCandidate(
                    wall_id=_host_label(nearest),
                    center=candidate["center"],
                    width_px=float(candidate["length_px"]),
                    bbox=candidate["bbox"],
                    orientation=str(candidate["orientation"]),
                    attached_ok=True,
                    overlap_ok=True,
                )
            )
            if _bbox_hits_mask(candidate["bbox"], door_mask):
                door_filtered.append(
                    WindowCandidate(
                        wall_id=_host_label(nearest),
                        center=candidate["center"],
                        width_px=float(candidate["length_px"]),
                        bbox=candidate["bbox"],
                        orientation=str(candidate["orientation"]),
                        attached_ok=True,
                        overlap_ok=True,
                    )
                )
                continue
            offset_px = _project_offset_on_wall(candidate["center"], nearest)
            merged = False
            for group in grouped:
                if group["host"].wall_ids != nearest.wall_ids:
                    continue
                if abs(int(group["offset_px"]) - offset_px) > config.window_group_merge_px:
                    continue
                group["offset_px"] = int(round((int(group["offset_px"]) + offset_px) / 2.0))
                group["length_px"] = max(float(group["length_px"]), float(candidate["length_px"]))
                group["bbox"] = _merge_bbox(group["bbox"], candidate["bbox"])
                group["segment_count"] = int(group["segment_count"]) + 1
                merged = True
                break
            if not merged:
                grouped.append(
                    {
                        "wall_id": _host_primary_wall_id(nearest),
                        "offset_px": offset_px,
                        "length_px": float(candidate["length_px"]),
                        "bbox": candidate["bbox"],
                        "segment_count": 1,
                        "wall_orientation": nearest.orientation,
                        "host": nearest,
                    }
                )

    accepted_groups: list[dict[str, object]] = []
    ordered = sorted(grouped, key=lambda item: (-float(item["length_px"]), -int(item["segment_count"])))
    for group in ordered:
        duplicate = False
        for existing in accepted_groups:
            if existing["wall_id"] != _host_primary_wall_id(group["host"]):
                continue
            if abs(int(existing["offset_px"]) - int(group["offset_px"])) > config.window_group_merge_px:
                continue
            if _bbox_overlap(existing["bbox"], group["bbox"]):
                duplicate = True
                break
        if duplicate:
            continue
        accepted_groups.append(group)
    accepted = [
        WindowCandidate(
            wall_id=str(group["wall_id"]),
            center=((group["bbox"][0] + group["bbox"][2]) // 2, (group["bbox"][1] + group["bbox"][3]) // 2),
            width_px=float(group["length_px"]),
            bbox=group["bbox"],
            orientation=str(group["wall_orientation"]),
            attached_ok=True,
            overlap_ok=True,
        )
        for group in accepted_groups
    ]
    return {
        "cleaned": cleaned,
        "thin_horizontal": thin_horizontal,
        "thin_vertical": thin_vertical,
        "hosts": outer_hosts,
        "raw_candidates": raw_candidates,
        "attached_pass": attached_pass,
        "overlap_pass": overlap_pass,
        "door_filtered": door_filtered,
        "crossing_pass": overlap_pass,
        "accepted": accepted,
        "accepted_groups": accepted_groups,
    }


def _extract_window_candidates(
    mask: np.ndarray,
    orientation: str,
    config: ParserConfig,
) -> list[dict[str, object]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, object]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        length = max(w, h)
        thickness = min(w, h)
        if thickness > config.window_max_thickness_px:
            continue
        if length < config.window_min_length_px or length > config.window_max_length_px:
            continue
        if orientation == "horizontal":
            start = (x, y + h // 2)
            end = (x + w - 1, y + h // 2)
        else:
            start = (x + w // 2, y)
            end = (x + w // 2, y + h - 1)
        center = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        candidates.append(
            {
                "orientation": orientation,
                "start": start,
                "end": end,
                "center": center,
                "bbox": (x, y, x + w - 1, y + h - 1),
                "length_px": float(length),
            }
        )
    return candidates


def _nearest_outer_host(
    point: tuple[int, int],
    hosts: list[WindowHost],
    orientation: str,
    max_distance: int,
) -> WindowHost | None:
    best_wall: WindowHost | None = None
    best_distance = float("inf")
    for wall in hosts:
        if wall.orientation != orientation:
            continue
        distance = _distance_point_to_wall(point, wall)
        if distance > max_distance or distance >= best_distance:
            continue
        best_distance = distance
        best_wall = wall
    return best_wall


def _distance_point_to_wall(point: tuple[int, int], wall: WindowHost | WallSegment) -> float:
    px, py = point
    if wall.orientation == "horizontal":
        if wall.span_start <= px <= wall.span_end:
            return float(abs(py - wall.center))
        nearest_x = min(max(px, wall.span_start), wall.span_end)
        return math.hypot(px - nearest_x, py - wall.center)
    if wall.span_start <= py <= wall.span_end:
        return float(abs(px - wall.center))
    nearest_y = min(max(py, wall.span_start), wall.span_end)
    return math.hypot(px - wall.center, py - nearest_y)


def _axis_overlap(candidate: dict[str, object], wall: WallSegment) -> float:
    if wall.orientation == "horizontal":
        start, end = sorted((int(candidate["start"][0]), int(candidate["end"][0])))
    else:
        start, end = sorted((int(candidate["start"][1]), int(candidate["end"][1])))
    overlap_start = max(wall.span_start, start)
    overlap_end = min(wall.span_end, end)
    return float(max(0, overlap_end - overlap_start))


def _project_offset_on_wall(point: tuple[int, int], wall: WallSegment) -> int:
    px, py = point
    if wall.orientation == "horizontal":
        projection = min(max(px, wall.span_start), wall.span_end)
        return int(abs(projection - wall.span_start))
    projection = min(max(py, wall.span_start), wall.span_end)
    return int(abs(projection - wall.span_start))


def _merge_bbox(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


def _bbox_overlap(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> bool:
    return not (left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1])


def _bbox_hits_mask(bbox: tuple[int, int, int, int], mask: np.ndarray) -> bool:
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(mask.shape[1] - 1, x2)
    y2 = min(mask.shape[0] - 1, y2)
    if x2 < x1 or y2 < y1:
        return False
    return bool(np.count_nonzero(mask[y1 : y2 + 1, x1 : x2 + 1]))


def _build_window_hosts(
    image_shape: tuple[int, int],
    walls: list[WallSegment],
    config: ParserConfig,
) -> list[WindowHost]:
    del image_shape
    margin = config.outer_wall_margin_px + 12
    min_x = min(min(wall.span_start, wall.span_end) if wall.orientation == "horizontal" else wall.center for wall in walls)
    max_x = max(max(wall.span_start, wall.span_end) if wall.orientation == "horizontal" else wall.center for wall in walls)
    min_y = min(wall.center if wall.orientation == "horizontal" else min(wall.span_start, wall.span_end) for wall in walls)
    max_y = max(wall.center if wall.orientation == "horizontal" else max(wall.span_start, wall.span_end) for wall in walls)
    perimeter: list[WallSegment] = []
    for wall in walls:
        if wall.orientation == "horizontal":
            if abs(wall.center - min_y) <= margin or abs(wall.center - max_y) <= margin:
                perimeter.append(wall)
                continue
        else:
            if abs(wall.center - min_x) <= margin or abs(wall.center - max_x) <= margin:
                perimeter.append(wall)
                continue
        if wall.kind == "outer":
            perimeter.append(wall)

    hosts: list[WindowHost] = []
    for orientation in ("horizontal", "vertical"):
        oriented = [wall for wall in perimeter if wall.orientation == orientation]
        if orientation == "horizontal":
            oriented.sort(key=lambda item: (item.center, item.span_start))
        else:
            oriented.sort(key=lambda item: (item.center, item.span_start))
        for wall in oriented:
            if not hosts or hosts[-1].orientation != orientation:
                hosts.append(
                    WindowHost(
                        orientation=wall.orientation,
                        center=wall.center,
                        span_start=wall.span_start,
                        span_end=wall.span_end,
                        wall_ids=(wall.id,),
                    )
                )
                continue
            current = hosts[-1]
            aligned = abs(current.center - wall.center) <= config.alignment_tolerance_px + 4
            close = wall.span_start <= current.span_end + config.window_max_length_px
            if aligned and close:
                hosts[-1] = WindowHost(
                    orientation=current.orientation,
                    center=int(round((current.center + wall.center) / 2.0)),
                    span_start=min(current.span_start, wall.span_start),
                    span_end=max(current.span_end, wall.span_end),
                    wall_ids=current.wall_ids + (wall.id,),
                )
            else:
                hosts.append(
                    WindowHost(
                        orientation=wall.orientation,
                        center=wall.center,
                        span_start=wall.span_start,
                        span_end=wall.span_end,
                        wall_ids=(wall.id,),
                    )
                )
    return hosts


def _host_primary_wall_id(host: WindowHost) -> str:
    return host.wall_ids[0] if host.wall_ids else ""


def _host_label(host: WindowHost | None) -> str:
    if host is None:
        return ""
    return "+".join(host.wall_ids)
