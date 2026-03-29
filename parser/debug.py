import cv2
import numpy as np

from parser.config import DEBUG_OUTPUT_DIR
from parser.geometry import format_room_name


def write_debug_images(image, binary_image, wall_layer, scale_layer, labels, doors, windows, rooms, door_debug=None):
    DEBUG_OUTPUT_DIR.mkdir(exist_ok=True)

    cv2.imwrite(str(DEBUG_OUTPUT_DIR / "debug_binary.png"), binary_image)

    structural_overlay = image.copy()
    for wall in wall_layer["structural_walls"]:
        color = (0, 255, 0) if wall["type"] == "outer" else (0, 200, 255)
        cv2.line(structural_overlay, tuple(wall["start_px"]), tuple(wall["end_px"]), color, 2)
    cv2.rectangle(structural_overlay, tuple(wall_layer["building_bbox"][:2]), tuple(wall_layer["building_bbox"][2:]), (255, 0, 0), 1)
    cv2.imwrite(str(DEBUG_OUTPUT_DIR / "debug_walls.png"), structural_overlay)
    cv2.imwrite(str(DEBUG_OUTPUT_DIR / "debug_wall_mask.png"), wall_layer["structural_wall_mask"])

    export_overlay = image.copy()
    for wall in wall_layer["export_walls"]:
        color = (0, 255, 0) if wall["type"] == "outer" else (0, 200, 255)
        cv2.line(export_overlay, tuple(wall["start_px"]), tuple(wall["end_px"]), color, 2)
    cv2.imwrite(str(DEBUG_OUTPUT_DIR / "debug_export_walls.png"), export_overlay)
    cv2.imwrite(str(DEBUG_OUTPUT_DIR / "debug_export_wall_mask.png"), wall_layer["export_wall_mask"])

    symbol_overlay = image.copy()
    symbol_overlay[wall_layer["symbol_mask"] > 0] = (0, 80, 255)
    cv2.imwrite(str(DEBUG_OUTPUT_DIR / "debug_symbol_mask.png"), symbol_overlay)

    scale_overlay = image.copy()
    if scale_layer["line_bbox"]:
        x1, y1, x2, y2 = scale_layer["line_bbox"]
        cv2.rectangle(scale_overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(scale_overlay, f"scale={scale_layer['meters_per_pixel']:.4f} m/px", (max(5, x1), max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(str(DEBUG_OUTPUT_DIR / "debug_scale.png"), scale_overlay)

    label_overlay = image.copy()
    for label in labels:
        x1, y1, x2, y2 = label["bbox"]
        cv2.rectangle(label_overlay, (x1, y1), (x2, y2), (255, 0, 255), 1)
        cv2.circle(label_overlay, label["center"], 3, (255, 0, 255), -1)
        cv2.putText(label_overlay, label["text"], (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    cv2.imwrite(str(DEBUG_OUTPUT_DIR / "debug_labels.png"), label_overlay)

    door_overlay = image.copy()
    for door in doors:
        if door.get("debug_zone_bbox"):
            x1, y1, x2, y2 = door["debug_zone_bbox"]
            cv2.rectangle(door_overlay, (x1, y1), (x2, y2), (0, 165, 255), 1)
        center = tuple(door["center_px"])
        cv2.circle(door_overlay, center, door["radius_px"], (0, 0, 255), 2)
        cv2.putText(door_overlay, door["id"], (center[0] + 4, center[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imwrite(str(DEBUG_OUTPUT_DIR / "debug_doors.png"), door_overlay)

    if door_debug is not None:
        cv2.imwrite(str(DEBUG_OUTPUT_DIR / "debug_door_symbol_mask.png"), door_debug["symbol_mask"])
        if "edge_mask" in door_debug:
            cv2.imwrite(str(DEBUG_OUTPUT_DIR / "debug_door_edge_mask.png"), door_debug["edge_mask"])
        if "component_mask" in door_debug:
            cv2.imwrite(str(DEBUG_OUTPUT_DIR / "debug_door_component_mask.png"), door_debug["component_mask"])
        pre_overlay = image.copy()
        for component in door_debug.get("component_candidates", []):
            x1, y1, x2, y2 = component["bbox_px"]
            cv2.rectangle(pre_overlay, (x1, y1), (x2, y2), (255, 0, 255), 1)
        for arc in door_debug["arc_candidates"]:
            x1, y1, x2, y2 = arc["bbox_px"]
            cv2.rectangle(pre_overlay, (x1, y1), (x2, y2), (0, 255, 255), 1)
        for line in door_debug["line_candidates"]:
            cv2.line(pre_overlay, tuple(line["line_px"][:2]), tuple(line["line_px"][2:]), (255, 0, 0), 2)
            hx, hy = line["hinge_px"]
            cv2.circle(pre_overlay, (hx, hy), 3, (0, 0, 255), -1)
        cv2.imwrite(str(DEBUG_OUTPUT_DIR / "debug_door_candidates.png"), pre_overlay)

    window_overlay = image.copy()
    for window in windows:
        x1, y1, x2, y2 = window["bbox_px"]
        cv2.rectangle(window_overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(window_overlay, window["id"], (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    cv2.imwrite(str(DEBUG_OUTPUT_DIR / "debug_windows.png"), window_overlay)

    combined = image.copy()
    for wall in wall_layer["export_walls"]:
        color = (0, 255, 0) if wall["type"] == "outer" else (0, 200, 255)
        cv2.line(combined, tuple(wall["start_px"]), tuple(wall["end_px"]), color, 2)
    for room in rooms:
        pts = np.array(room["polygon"], np.int32)
        cv2.polylines(combined, [pts], True, (0, 0, 255), 2)
        if room["label"]:
            x, y = pts[0]
            cv2.putText(combined, format_room_name(room["label"]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    for door in doors:
        cv2.circle(combined, tuple(door["center_px"]), door["radius_px"], (0, 0, 255), 2)
    for window in windows:
        x1, y1, x2, y2 = window["bbox_px"]
        cv2.rectangle(combined, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.imwrite(str(DEBUG_OUTPUT_DIR / "debug_combined.png"), combined)
