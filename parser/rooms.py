from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from parser.config import ParserConfig
from parser.text import TextDetection


@dataclass(slots=True)
class Room:
    id: str
    name: str
    polygon: list[tuple[int, int]]


def detect_rooms(
    plan_mask: np.ndarray,
    wall_mask: np.ndarray,
    texts: list[TextDetection],
    plan_bbox: tuple[int, int, int, int],
    config: ParserConfig,
) -> list[Room]:
    x, y, w, h = plan_bbox
    free_space = cv2.bitwise_and(plan_mask, cv2.bitwise_not(wall_mask))
    free_roi = free_space[y : y + h, x : x + w]
    room_texts = [text for text in texts if text.kind == "room"]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(free_roi, 8)
    rooms: list[Room] = []
    room_index = 1
    for label_index in range(1, num_labels):
        left = int(stats[label_index, cv2.CC_STAT_LEFT])
        top = int(stats[label_index, cv2.CC_STAT_TOP])
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area < config.room_min_area_px:
            continue
        component_mask = np.where(labels == label_index, 255, 0).astype(np.uint8)
        component_room_texts = _texts_inside_component(component_mask, room_texts, x, y)
        if _touches_roi_border(left, top, width, height, free_roi.shape) and not component_room_texts:
            continue
        split_masks = _split_component_by_texts(component_mask, component_room_texts, x, y)
        for split_mask in split_masks:
            if int(np.count_nonzero(split_mask)) < config.room_min_area_px:
                continue
            polygon = _polygon_from_mask(split_mask, x, y)
            if polygon is None:
                continue
            name = _room_name_for_polygon(polygon, component_room_texts)
            rooms.append(Room(id=f"r{room_index}", name=name, polygon=polygon))
            room_index += 1
    return rooms


def _room_name_for_polygon(polygon: list[tuple[int, int]], texts: list[TextDetection]) -> str:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    for text in texts:
        if text.kind != "room":
            continue
        tx1, ty1, tx2, ty2 = text.bbox
        cx = (tx1 + tx2) // 2
        cy = (ty1 + ty2) // 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return text.text.title()
    return "Room"


def _touches_roi_border(
    left: int,
    top: int,
    width: int,
    height: int,
    shape: tuple[int, int],
) -> bool:
    return (
        left == 0
        or top == 0
        or left + width >= shape[1]
        or top + height >= shape[0]
    )


def _texts_inside_component(
    component_mask: np.ndarray,
    texts: list[TextDetection],
    offset_x: int,
    offset_y: int,
) -> list[TextDetection]:
    inside: list[TextDetection] = []
    height, width = component_mask.shape
    for text in texts:
        tx1, ty1, tx2, ty2 = text.bbox
        cx = ((tx1 + tx2) // 2) - offset_x
        cy = ((ty1 + ty2) // 2) - offset_y
        if 0 <= cx < width and 0 <= cy < height and component_mask[cy, cx] > 0:
            inside.append(text)
    return inside


def _split_component_by_texts(
    component_mask: np.ndarray,
    texts: list[TextDetection],
    offset_x: int,
    offset_y: int,
) -> list[np.ndarray]:
    if len(texts) <= 1:
        return [component_mask]

    points = np.argwhere(component_mask > 0)
    if len(points) == 0:
        return []

    seeds = np.array(
        [
            [
                ((text.bbox[1] + text.bbox[3]) // 2) - offset_y,
                ((text.bbox[0] + text.bbox[2]) // 2) - offset_x,
            ]
            for text in texts
        ],
        dtype=np.int32,
    )
    distances = np.sum((points[:, None, :] - seeds[None, :, :]) ** 2, axis=2)
    assignments = np.argmin(distances, axis=1)

    split_masks: list[np.ndarray] = []
    for text_index in range(len(texts)):
        room_pixels = points[assignments == text_index]
        if len(room_pixels) == 0:
            continue
        split_mask = np.zeros_like(component_mask)
        split_mask[room_pixels[:, 0], room_pixels[:, 1]] = 255
        split_masks.append(split_mask)
    return split_masks or [component_mask]


def _polygon_from_mask(
    component_mask: np.ndarray,
    offset_x: int,
    offset_y: int,
) -> list[tuple[int, int]] | None:
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return [(int(point[0][0] + offset_x), int(point[0][1] + offset_y)) for point in approx]
