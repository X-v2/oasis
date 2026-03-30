from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class DebugWriter:
    def __init__(self, root: Path, enabled: bool = True) -> None:
        self.root = root
        self.enabled = enabled
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def clear(self) -> None:
        if not self.enabled or not self.root.exists():
            return
        for path in self.root.iterdir():
            if path.is_file():
                path.unlink()

    def write_image(self, name: str, image: np.ndarray) -> None:
        if not self.enabled:
            return
        cv2.imwrite(str(self.root / name), image)

    def overlay_mask(
        self,
        base_image: np.ndarray,
        mask: np.ndarray,
        color: tuple[int, int, int],
        alpha: float = 0.45,
    ) -> np.ndarray:
        if len(base_image.shape) == 2:
            overlay = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
        else:
            overlay = base_image.copy()
        colored = np.zeros_like(overlay)
        colored[mask > 0] = color
        return cv2.addWeighted(overlay, 1.0, colored, alpha, 0.0)
