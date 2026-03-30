from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ParserConfig:
    input_path: Path = Path("floorplan_test.png")
    output_path: Path = Path("output.json")
    debug_dir: Path = Path("debug_layers")
    default_wall_height_m: float = 3.0
    default_wall_thickness_m: float = 0.2
    default_door_height_m: float = 2.1
    default_window_height_m: float = 1.2
    default_window_sill_m: float = 1.0
    threshold_value: int = 220
    text_mask_padding_px: int = 10
    min_wall_length_px: int = 22
    min_wall_stub_px: int = 12
    horizontal_wall_kernel_px: int = 35
    vertical_wall_kernel_px: int = 35
    thin_horizontal_kernel_px: int = 18
    thin_vertical_kernel_px: int = 18
    short_wall_kernel_px: int = 12
    short_wall_min_length_px: int = 12
    short_wall_connection_tolerance_px: int = 12
    short_wall_bump_max_length_px: int = 24
    alignment_tolerance_px: int = 4
    wall_min_component_area_px: int = 40
    wall_min_stroke_radius_px: float = 3.2
    wall_merge_gap_px: int = 16
    wall_merge_offset_px: int = 10
    wall_bridge_gap_min_px: int = 24
    wall_bridge_gap_max_px: int = 120
    wall_bridge_alignment_px: int = 8
    wall_bridge_thickness_delta_px: float = 4.0
    door_gap_cluster_merge_px: int = 20
    door_min_merged_gap_px: int = 40
    door_symbol_line_min_px: int = 18
    door_symbol_line_max_px: int = 42
    door_symbol_min_width_px: float = 55.0
    door_symbol_wall_distance_px: int = 8
    door_symbol_end_margin_px: int = 24
    door_symbol_crossing_clearance_px: int = 18
    door_arc_min_radius_px: int = 18
    door_arc_max_radius_px: int = 110
    window_max_thickness_px: int = 8
    window_min_length_px: int = 24
    window_max_length_px: int = 220
    window_wall_distance_px: int = 42
    window_group_merge_px: int = 20
    default_scale_m: float = 4.0
    default_pixels_per_meter: float = 78.0
    plan_padding_px: int = 24
    room_min_area_px: int = 5000
    debug_enabled: bool = True
    easyocr_languages: tuple[str, ...] = ("en",)
    room_name_allowlist: tuple[str, ...] = (
        "ENTRY",
        "BATH",
        "KITCHEN",
        "LIVING ROOM",
        "BEDROOM",
        "BEDROOM 1",
        "BEDROOM 2",
        "BEDROOM 3",
        "DINING",
        "TOILET",
        "STORE",
        "HALL",
    )
    outer_wall_margin_px: int = 32
    ignored_text_keywords: tuple[str, ...] = (
        "HOUSE FLOOR PLAN",
        "PLAN A",
        "BEDROOMS",
        "BATHROOM",
    )
    ocr_scale_pattern: str = "m"
    component_precedence: tuple[str, ...] = field(
        default_factory=lambda: ("wall", "door", "window", "text")
    )
