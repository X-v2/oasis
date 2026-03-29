import cv2

from parser.config import DEFAULT_WALL_HEIGHT, DEFAULT_WALL_THICKNESS
from parser.debug import write_debug_images
from parser.ocr import build_text_mask, filter_room_labels, run_ocr
from parser.openings import build_symbol_mask, dedupe_windows_after_reassign, detect_doors, detect_windows, inspect_door_detector, prune_window_parallel_wall_segments, reassign_openings_to_walls
from parser.rooms import assign_room_labels, detect_rooms_from_walls, format_rooms
from parser.scale import extract_scale_layer
from parser.walls import extract_wall_layer, format_walls_for_frontend


def preprocess_to_binary(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray, binary


def parse_floorplan(image_path, debug=False):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray, binary = preprocess_to_binary(image)
    ocr_items = run_ocr(image)
    scale_layer = extract_scale_layer(binary, ocr_items)
    text_mask = build_text_mask(image.shape, ocr_items)
    structural_layer = extract_wall_layer(binary, text_mask)
    labels = filter_room_labels(ocr_items, structural_layer["building_bbox"])
    rooms_px = assign_room_labels(detect_rooms_from_walls(structural_layer["structural_wall_mask"], labels), labels)
    windows = detect_windows(structural_layer, structural_layer["building_bbox"], scale_layer["meters_per_pixel"])
    door_debug = inspect_door_detector(binary, text_mask, structural_layer, ocr_items, windows=windows)
    doors = detect_doors(binary, text_mask, structural_layer, ocr_items, scale_layer["meters_per_pixel"], windows=windows)
    symbol_mask = build_symbol_mask(image.shape, doors, windows)
    wall_layer = extract_wall_layer(binary, text_mask, symbol_mask=symbol_mask)
    wall_layer["export_walls"] = prune_window_parallel_wall_segments(wall_layer["export_walls"], windows)
    doors = reassign_openings_to_walls(doors, wall_layer["export_walls"], scale_layer["meters_per_pixel"])
    windows = reassign_openings_to_walls(windows, wall_layer["export_walls"], scale_layer["meters_per_pixel"])
    windows = dedupe_windows_after_reassign(windows)

    if debug:
        write_debug_images(image, binary, wall_layer, scale_layer, labels, doors, windows, rooms_px, door_debug=door_debug)

    frontend_walls = format_walls_for_frontend(
        wall_layer["export_walls"],
        scale_layer["meters_per_pixel"],
        DEFAULT_WALL_THICKNESS,
        DEFAULT_WALL_HEIGHT,
    )
    frontend_rooms = format_rooms(rooms_px, scale_layer["meters_per_pixel"])
    frontend_doors = [{"id": door["id"], "wallId": door["wallId"], "offset": door["offset"], "width": door["width"], "height": door["height"], "swing": door["swing"]} for door in doors]
    frontend_windows = [{"id": window["id"], "wallId": window["wallId"], "offset": window["offset"], "width": window["width"], "height": window["height"], "sillHeight": window["sillHeight"]} for window in windows]

    return {
        "meta": {
            "unit": "meter",
            "wallHeight": DEFAULT_WALL_HEIGHT,
            "defaultWallThickness": DEFAULT_WALL_THICKNESS,
        },
        "rooms": frontend_rooms,
        "walls": frontend_walls,
        "doors": frontend_doors,
        "windows": frontend_windows,
    }
