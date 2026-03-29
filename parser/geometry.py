import math


def clean_text(text):
    return " ".join(text.strip().split())


def format_room_name(text):
    return clean_text(text).title() if text else None


def scale_value(value, meters_per_pixel):
    return round(float(value) * meters_per_pixel, 3)


def scale_point(point, meters_per_pixel):
    return [scale_value(point[0], meters_per_pixel), scale_value(point[1], meters_per_pixel)]


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def bbox_contains_point(bbox, point, padding=0):
    x1, y1, x2, y2 = bbox
    x, y = point
    return x1 - padding <= x <= x2 + padding and y1 - padding <= y <= y2 + padding


def segment_length(segment):
    if segment["orientation"] == "horizontal":
        return abs(segment["x2"] - segment["x1"])
    return abs(segment["y2"] - segment["y1"])


def segment_start_end(segment):
    if segment["orientation"] == "horizontal":
        return [segment["x1"], segment["y"]], [segment["x2"], segment["y"]]
    return [segment["x"], segment["y1"]], [segment["x"], segment["y2"]]


def distance_point_to_wall(point, wall):
    px, py = point
    x1, y1 = wall["start_px"]
    x2, y2 = wall["end_px"]

    if x1 == x2:
        y_min, y_max = sorted((y1, y2))
        if y_min <= py <= y_max:
            return abs(px - x1)
        return math.hypot(px - x1, py - (y_min if py < y_min else y_max))

    x_min, x_max = sorted((x1, x2))
    if x_min <= px <= x_max:
        return abs(py - y1)
    return math.hypot(px - (x_min if px < x_min else x_max), py - y1)


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def project_offset_on_wall(point, wall):
    px, py = point
    x1, y1 = wall["start_px"]
    x2, y2 = wall["end_px"]

    if x1 == x2:
        y_min, y_max = sorted((y1, y2))
        projection = min(max(py, y_min), y_max)
        return abs(projection - y1)

    x_min, x_max = sorted((x1, x2))
    projection = min(max(px, x_min), x_max)
    return abs(projection - x1)


def wall_axis_bounds(wall):
    if wall["orientation"] == "horizontal":
        return sorted((wall["start_px"][0], wall["end_px"][0]))
    return sorted((wall["start_px"][1], wall["end_px"][1]))


def wall_cross_axis_value(wall):
    if wall["orientation"] == "horizontal":
        return wall["start_px"][1]
    return wall["start_px"][0]


def interval_overlap(a1, a2, b1, b2):
    start = max(min(a1, a2), min(b1, b2))
    end = min(max(a1, a2), max(b1, b2))
    return max(0, end - start)


def line_bbox(start, end, padding=0):
    x1 = min(start[0], end[0]) - padding
    y1 = min(start[1], end[1]) - padding
    x2 = max(start[0], end[0]) + padding
    y2 = max(start[1], end[1]) + padding
    return [x1, y1, x2, y2]


def nearest_wall(point, walls, allowed_types=None, max_distance=40):
    candidates = []

    for wall in walls:
        if allowed_types and wall["type"] not in allowed_types:
            continue
        distance = distance_point_to_wall(point, wall)
        if distance <= max_distance:
            candidates.append((distance, wall))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def wall_near_perimeter(wall, building_bbox, tolerance=25):
    x1, y1, x2, y2 = building_bbox

    if wall["orientation"] == "horizontal":
        return abs(wall["start_px"][1] - y1) <= tolerance or abs(wall["start_px"][1] - y2) <= tolerance

    return abs(wall["start_px"][0] - x1) <= tolerance or abs(wall["start_px"][0] - x2) <= tolerance


def entry_zone(point, building_bbox):
    x1, y1, x2, y2 = building_bbox
    center_x = (x1 + x2) / 2.0
    return abs(point[0] - center_x) <= (x2 - x1) * 0.2 and point[1] >= y2 - 60
