from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from parser.config import ParserConfig


@dataclass(slots=True)
class LoadedImage:
    color: np.ndarray
    gray: np.ndarray
    binary_inv: np.ndarray


def load_image(config: ParserConfig) -> LoadedImage:
    color = cv2.imread(str(config.input_path), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(f"Could not read image: {config.input_path}")
    return _build_loaded_image(color)


def load_image_bytes(data: bytes) -> LoadedImage:
    buffer = np.frombuffer(data, dtype=np.uint8)
    color = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError("Could not decode image bytes")
    return _build_loaded_image(color)


def _build_loaded_image(color: np.ndarray) -> LoadedImage:
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    _, otsu = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    binary_inv = cv2.bitwise_or(adaptive, otsu)
    return LoadedImage(color=color, gray=gray, binary_inv=binary_inv)
