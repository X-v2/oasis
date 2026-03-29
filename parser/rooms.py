import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from parser.config import MIN_ROOM_AREA
from parser.geometry import format_room_name, scale_point


def labels_inside_component(component_mask, labels):
    inside = []
    for label in labels:
        x, y = label["center"]
        if component_mask[y, x] > 0:
            inside.append(label)
    return inside


def split_component_by_labels(component_mask, labels):
    if len(labels) <= 1:
        return [component_mask]

    points = np.argwhere(component_mask > 0)
    if len(points) == 0:
        return []

    seeds = np.array([[label["center"][1], label["center"][0]] for label in labels], dtype=np.int32)
    distances = np.sum((points[:, None, :] - seeds[None, :, :]) ** 2, axis=2)
    assignments = np.argmin(distances, axis=1)
    split_masks = []

    for label_index in range(len(labels)):
        room_pixels = points[assignments == label_index]
        if len(room_pixels) == 0:
            continue
        split_mask = np.zeros_like(component_mask)
        split_mask[room_pixels[:, 0], room_pixels[:, 1]] = 255
        split_masks.append(split_mask)

    return split_masks


def find_component_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_ROOM_AREA:
            continue
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        polygon = [(int(point[0][0]), int(point[0][1])) for point in approx]
        if len(polygon) >= 4:
            polygons.append({"polygon": polygon, "area": float(area)})

    return polygons


def detect_rooms_from_walls(wall_mask, labels):
    free_space = cv2.bitwise_not(wall_mask)
    component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(free_space, connectivity=4)
    rooms = []

    for component_id in range(1, component_count):
        x = stats[component_id, cv2.CC_STAT_LEFT]
        y = stats[component_id, cv2.CC_STAT_TOP]
        w = stats[component_id, cv2.CC_STAT_WIDTH]
        h = stats[component_id, cv2.CC_STAT_HEIGHT]
        area = stats[component_id, cv2.CC_STAT_AREA]

        if x == 0 or y == 0 or x + w >= wall_mask.shape[1] or y + h >= wall_mask.shape[0]:
            continue
        if area < MIN_ROOM_AREA:
            continue

        component_mask = np.where(component_labels == component_id, 255, 0).astype(np.uint8)
        component_labels_inside = labels_inside_component(component_mask, labels)
        split_masks = split_component_by_labels(component_mask, component_labels_inside)

        for split_index, split_mask in enumerate(split_masks):
            polygons = find_component_polygons(split_mask)
            for polygon_data in polygons:
                label = None
                if split_index < len(component_labels_inside):
                    label = component_labels_inside[split_index]["text"]
                rooms.append({"label": label, "polygon": polygon_data["polygon"], "area": polygon_data["area"]})

    return rooms


def assign_room_labels(rooms, labels):
    for room in rooms:
        polygon = Polygon(room["polygon"])
        assigned = room.get("label")
        for label in labels:
            if polygon.contains(Point(label["center"])):
                assigned = label["text"]
                break
        room["label"] = assigned

    return rooms


def format_rooms(rooms, meters_per_pixel):
    formatted = []
    for index, room in enumerate(rooms, start=1):
        formatted.append(
            {
                "id": f"r{index}",
                "name": format_room_name(room.get("label")),
                "polygon": [scale_point(point, meters_per_pixel) for point in room["polygon"]],
            }
        )
    return formatted
