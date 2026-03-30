from __future__ import annotations

import json

import cv2
import numpy as np

from parser.config import ParserConfig
from parser.debug import DebugWriter
from parser.doors import Door, detect_doors, inspect_door_candidates
from parser.image_io import LoadedImage, load_image, load_image_bytes
from parser.scale import detect_scale
from parser.schema import build_schema
from parser.text import build_text_mask, detect_text
from parser.walls import detect_plan_and_walls, render_wall_mask, segment_walls_by_outside_adjacency
from parser.windows import Window, detect_windows, inspect_window_candidates, merge_window_host_walls


def parse_floorplan(
    config: ParserConfig | None = None,
    loaded_image: LoadedImage | None = None,
) -> dict:
    config = config or ParserConfig()
    debug = DebugWriter(config.debug_dir, enabled=config.debug_enabled)
    debug.clear()

    image = loaded_image or load_image(config)
    texts = detect_text(image.gray, config)
    text_mask = build_text_mask(image.gray.shape, texts, config)

    wall_result = detect_plan_and_walls(image.binary_inv, text_mask, config)
    door_checks = inspect_door_candidates(image.binary_inv, wall_result.walls, config)

    doors, door_mask = detect_doors(
        image.gray,
        image.binary_inv,
        wall_result.walls,
        wall_result.plan_bbox,
        config,
    )

    window_checks = inspect_window_candidates(image.binary_inv, wall_result.walls, door_mask, config)
    windows = detect_windows(image.binary_inv, wall_result.walls, door_mask, config)
    export_walls, export_doors, export_windows = merge_window_host_walls(
        wall_result.walls,
        doors,
        windows,
        config,
    )
    export_walls = segment_walls_by_outside_adjacency(image.binary_inv.shape, export_walls, config)
    export_doors = _reassign_doors_to_walls(export_doors, export_walls)
    export_windows = _reassign_windows_to_walls(export_windows, export_walls)
    export_wall_mask = render_wall_mask(image.binary_inv.shape, export_walls)
    export_structure_mask = cv2.bitwise_and(
        cv2.bitwise_and(image.binary_inv, cv2.bitwise_not(text_mask)),
        export_wall_mask,
    )
    final_wall_mask = cv2.bitwise_and(export_wall_mask, cv2.bitwise_not(door_mask))
    window_mask = _build_window_mask(image.binary_inv.shape, export_windows, export_walls)

    scale = detect_scale(image.gray, wall_result.plan_bbox, texts, config)

    payload = build_schema(
        config=config,
        scale=scale,
        plan_bbox=wall_result.plan_bbox,
        image_shape=image.gray.shape,
        walls=export_walls,
        doors=export_doors,
        windows=export_windows,
        texts=texts,
    )

    _write_debug_images(
        debug=debug,
        image=image.color,
        binary_mask=image.binary_inv,
        text_mask=text_mask,
        plan_mask=wall_result.plan_mask,
        structure_mask=export_structure_mask,
        wall_mask=export_wall_mask,
        clean_wall_mask=final_wall_mask,
        walls=export_walls,
        doors=export_doors,
        door_checks=door_checks,
        door_mask=door_mask,
        window_checks=window_checks,
        windows=export_windows,
        window_mask=window_mask,
        texts=texts,
        scale=scale,
        plan_bbox=wall_result.plan_bbox,
        slabs=payload["slabs"],
    )
    return payload


def parse_floorplan_bytes(
    data: bytes,
    config: ParserConfig | None = None,
) -> dict:
    config = config or ParserConfig()
    image = load_image_bytes(data)
    return parse_floorplan(config=config, loaded_image=image)


def main() -> None:
    config = ParserConfig()
    payload = parse_floorplan(config=config)
    config.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_debug_images(
    debug: DebugWriter,
    image: np.ndarray,
    binary_mask: np.ndarray,
    text_mask: np.ndarray,
    plan_mask: np.ndarray,
    structure_mask: np.ndarray,
    wall_mask: np.ndarray,
    clean_wall_mask: np.ndarray,
    walls: list,
    doors: list,
    door_checks: dict,
    door_mask: np.ndarray,
    window_checks: dict,
    windows: list,
    window_mask: np.ndarray,
    texts: list,
    scale,
    plan_bbox: tuple[int, int, int, int],
    slabs: list[dict],
) -> None:
    debug.write_image("debug_binary.png", binary_mask)
    debug.write_image("debug_text_mask.png", text_mask)
    debug.write_image("debug_plan_mask.png", plan_mask)
    debug.write_image("debug_structure_mask.png", structure_mask)
    debug.write_image("debug_wall_mask.png", wall_mask)
    debug.write_image("debug_clean_wall_mask.png", clean_wall_mask)
    debug.write_image("debug_doors_mask.png", door_mask)
    debug.write_image("debug_windows_mask.png", window_mask)
    debug.write_image("debug_doors_precheck.png", _draw_door_scan_bands(image, door_checks["scan_bands"]))
    debug.write_image("debug_doors_all_candidates.png", _draw_door_candidates(image, door_checks["all_candidates"], (0, 165, 255)))
    debug.write_image("debug_doors_width_pass.png", _draw_door_candidates(image, door_checks["width_pass"], (255, 0, 255)))
    debug.write_image("debug_doors_opening_pass.png", _draw_door_candidates(image, door_checks["opening_pass"], (0, 255, 255)))
    debug.write_image("debug_doors_accepted.png", _draw_door_candidates(image, door_checks["accepted"], (0, 255, 0)))
    debug.write_image("debug_windows_cleaned.png", window_checks["cleaned"])
    debug.write_image("debug_windows_thin_horizontal.png", window_checks["thin_horizontal"])
    debug.write_image("debug_windows_thin_vertical.png", window_checks["thin_vertical"])
    debug.write_image("debug_windows_all_candidates.png", _draw_window_candidates(image, window_checks["raw_candidates"], (0, 165, 255)))
    debug.write_image("debug_windows_attached_pass.png", _draw_window_candidates(image, window_checks["attached_pass"], (255, 0, 255)))
    debug.write_image("debug_windows_overlap_pass.png", _draw_window_candidates(image, window_checks["overlap_pass"], (0, 255, 255)))
    debug.write_image("debug_windows_door_filtered.png", _draw_window_candidates(image, window_checks["door_filtered"], (0, 0, 255)))
    debug.write_image("debug_windows_crossing_pass.png", _draw_window_candidates(image, window_checks["crossing_pass"], (0, 255, 0)))
    debug.write_image("debug_windows_accepted.png", _draw_window_candidates(image, window_checks["accepted"], (0, 255, 0)))

    debug.write_image("debug_walls.png", _draw_walls(image, walls))
    debug.write_image("debug_clean_walls.png", _draw_mask_on_image(image, clean_wall_mask, (180, 0, 180)))
    debug.write_image("debug_doors.png", _draw_doors(image, doors))
    debug.write_image("debug_windows.png", _draw_windows(image, windows))
    debug.write_image("debug_labels.png", _draw_labels(image, texts))
    debug.write_image("debug_scale.png", _draw_scale(image, scale))
    debug.write_image("debug_slabs.png", _draw_slabs(image, slabs, scale, plan_bbox))
    debug.write_image("debug_combined.png", _draw_combined(image, walls, doors, windows, slabs, scale, plan_bbox))


def _draw_doors(image: np.ndarray, doors: list) -> np.ndarray:
    output = image.copy()
    for door in doors:
        cx, cy = door.center
        cv2.circle(output, (cx, cy), 10, (0, 0, 255), 2)
        cv2.line(output, (cx - 12, cy), (cx + 12, cy), (0, 0, 255), 1)
        cv2.line(output, (cx, cy - 12), (cx, cy + 12), (0, 0, 255), 1)
        cv2.circle(output, (cx, cy), 3, (0, 0, 180), -1)
    return output


def _draw_door_candidates(image: np.ndarray, candidates: list, color: tuple[int, int, int]) -> np.ndarray:
    output = image.copy()
    for index, candidate in enumerate(candidates, start=1):
        cx, cy = candidate.center
        cv2.circle(output, (cx, cy), 10, color, 2)
        cv2.line(output, (cx - 12, cy), (cx + 12, cy), color, 1)
        cv2.line(output, (cx, cy - 12), (cx, cy + 12), color, 1)
        cv2.circle(output, (cx, cy), 3, color, -1)
        cv2.putText(
            output,
            f"d{index}",
            (cx + 4, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )
    return output


def _draw_door_scan_bands(image: np.ndarray, scan_bands: list) -> np.ndarray:
    output = image.copy()
    for band in scan_bands:
        x1, y1, x2, y2 = band.rect
        color = (0, 200, 0) if band.orientation == "horizontal" else (200, 0, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            output,
            band.wall_id,
            (x1, max(12, y1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            color,
            1,
        )
    return output


def _build_window_mask(
    shape: tuple[int, int], windows: list, walls: list
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    wall_lookup = {wall.id: wall for wall in walls}
    for window in windows:
        wall = wall_lookup.get(window.wall_id)
        if wall is None:
            continue
        half = max(8, int(round(window.width_px / 2.0)))
        if wall.orientation == "horizontal":
            x1 = max(0, window.center[0] - half)
            x2 = min(shape[1] - 1, window.center[0] + half)
            y1 = max(0, wall.center - 8)
            y2 = min(shape[0] - 1, wall.center + 8)
        else:
            x1 = max(0, wall.center - 8)
            x2 = min(shape[1] - 1, wall.center + 8)
            y1 = max(0, window.center[1] - half)
            y2 = min(shape[0] - 1, window.center[1] + half)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask


def _draw_windows(image: np.ndarray, windows: list) -> np.ndarray:
    output = image.copy()
    for index, window in enumerate(windows, start=1):
        cx, cy = window.center
        x1, y1, x2, y2 = window.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(output, f"win{index}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(output, window.wall_id, (x1, min(output.shape[0] - 4, y2 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    return output


def _draw_window_candidates(image: np.ndarray, candidates: list, color: tuple[int, int, int]) -> np.ndarray:
    output = image.copy()
    for index, candidate in enumerate(candidates, start=1):
        x1, y1, x2, y2 = candidate.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, f"w{index}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        if candidate.wall_id:
            cv2.putText(output, candidate.wall_id, (x1, min(output.shape[0] - 4, y2 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return output


def _draw_walls(image: np.ndarray, walls: list) -> np.ndarray:
    output = image.copy()
    for wall in walls:
        color = (0, 255, 255) if wall.kind == "outer" else (255, 0, 0)
        thickness = 2 if wall.kind == "outer" else 1
        if wall.orientation == "horizontal":
            start = (wall.span_start, wall.center)
            end = (wall.span_end, wall.center)
        else:
            start = (wall.center, wall.span_start)
            end = (wall.center, wall.span_end)
        cv2.line(output, start, end, color, thickness, lineType=cv2.LINE_AA)
        label_pos = (
            ((start[0] + end[0]) // 2, start[1] - 6)
            if wall.orientation == "horizontal"
            else (start[0] + 4, (start[1] + end[1]) // 2)
        )
        cv2.putText(
            output,
            wall.id,
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
        )
    return output


def _draw_labels(image: np.ndarray, texts: list) -> np.ndarray:
    output = image.copy()
    for text in texts:
        x1, y1, x2, y2 = text.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 255), 1)
        cv2.putText(output, text.text[:20], (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
    return output


def _draw_scale(image: np.ndarray, scale) -> np.ndarray:
    output = image.copy()
    label = f"{scale.source}: {scale.pixels_per_meter:.2f} px/m"
    cv2.putText(output, label, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return output


def _draw_combined(
    image: np.ndarray,
    walls: list,
    doors: list,
    windows: list,
    slabs: list[dict],
    scale,
    plan_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    output = _draw_slabs(image, slabs, scale, plan_bbox)
    output = _draw_walls(output, walls)
    output = _draw_doors(output, doors)
    output = _draw_windows(output, windows)
    return output


def _draw_slabs(
    image: np.ndarray,
    slabs: list[dict],
    scale,
    plan_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    output = image.copy()
    overlay = image.copy()
    palette = [
        (70, 120, 230),
        (70, 180, 120),
        (230, 180, 70),
        (200, 90, 160),
        (100, 200, 200),
        (220, 130, 100),
    ]
    for index, slab in enumerate(slabs):
        polygon = _slab_polygon_to_pixels(slab["polygon"], scale, plan_bbox)
        if len(polygon) < 3:
            continue
        color = palette[index % len(palette)]
        contour = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [contour], color)
        cv2.polylines(output, [contour], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        label_x, label_y = _slab_centroid_to_pixels(slab.get("centroid"), scale, plan_bbox)
        cv2.putText(
            output,
            f"{slab.get('id', '')} {slab.get('name', '')}".strip(),
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            lineType=cv2.LINE_AA,
        )
    return cv2.addWeighted(overlay, 0.28, output, 0.72, 0.0)


def _slab_polygon_to_pixels(
    polygon: list[list[float]],
    scale,
    plan_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    x0, y0, _, _ = plan_bbox
    pixels: list[tuple[int, int]] = []
    for x_m, z_m in polygon:
        x_px = int(round(x0 + x_m * scale.pixels_per_meter))
        y_px = int(round(y0 + z_m * scale.pixels_per_meter))
        pixels.append((x_px, y_px))
    return pixels


def _slab_centroid_to_pixels(
    centroid: list[float] | None,
    scale,
    plan_bbox: tuple[int, int, int, int],
) -> tuple[int, int]:
    if not centroid:
        return 16, 16
    x0, y0, _, _ = plan_bbox
    x_px = int(round(x0 + centroid[0] * scale.pixels_per_meter))
    y_px = int(round(y0 + centroid[2] * scale.pixels_per_meter))
    return x_px, y_px


def _draw_mask_on_image(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = image.copy()
    overlay = np.zeros_like(output)
    overlay[mask > 0] = color
    return cv2.addWeighted(output, 1.0, overlay, 0.35, 0.0)


def _reassign_doors_to_walls(doors: list[Door], walls: list) -> list[Door]:
    reassigned: list[Door] = []
    for door in doors:
        wall = _best_wall_for_opening(door.center, door.wall_id, walls)
        reassigned.append(
            Door(
                id=door.id,
                wall_id=wall.id if wall is not None else door.wall_id,
                center=door.center,
                width_px=door.width_px,
                swing=door.swing,
            )
        )
    return reassigned


def _reassign_windows_to_walls(windows: list[Window], walls: list) -> list[Window]:
    reassigned: list[Window] = []
    for window in windows:
        wall = _best_wall_for_opening(window.center, window.wall_id, walls)
        reassigned.append(
            Window(
                id=window.id,
                wall_id=wall.id if wall is not None else window.wall_id,
                center=window.center,
                width_px=window.width_px,
                bbox=window.bbox,
                wall_orientation=window.wall_orientation,
            )
        )
    return reassigned


def _best_wall_for_opening(center: tuple[int, int], wall_id: str, walls: list):
    base_id = wall_id.split("_")[0]
    candidates = [wall for wall in walls if wall.id == wall_id or wall.id == base_id or wall.id.startswith(f"{base_id}_")]
    if not candidates:
        return None

    cx, cy = center
    best = None
    best_score: tuple[float, float] | None = None
    for wall in candidates:
        if wall.orientation == "horizontal":
            cross_axis_distance = abs(cy - wall.center)
            if not (wall.span_start - 2 <= cx <= wall.span_end + 2):
                along_axis_distance = min(abs(cx - wall.span_start), abs(cx - wall.span_end))
            else:
                along_axis_distance = 0.0
        else:
            cross_axis_distance = abs(cx - wall.center)
            if not (wall.span_start - 2 <= cy <= wall.span_end + 2):
                along_axis_distance = min(abs(cy - wall.span_start), abs(cy - wall.span_end))
            else:
                along_axis_distance = 0.0
        score = (cross_axis_distance, along_axis_distance)
        if best_score is None or score < best_score:
            best = wall
            best_score = score
    return best


if __name__ == "__main__":
    main()
