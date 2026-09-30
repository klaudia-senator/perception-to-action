"""Image-to-example-workspace mapping using a synthetic homography."""

from __future__ import annotations

import numpy as np


class CoordinateMapper:
    def __init__(self, homography: list[list[float]] | np.ndarray) -> None:
        self.homography = np.asarray(homography, dtype=np.float64)
        if self.homography.shape != (3, 3):
            raise ValueError("Homography must be a 3x3 matrix")
        if abs(np.linalg.det(self.homography)) < 1e-12:
            raise ValueError("Homography must be invertible")

    def image_to_workspace(self, normalized_point: tuple[float, float]) -> tuple[float, float]:
        x, y = normalized_point
        homogeneous = self.homography @ np.array([x, y, 1.0])
        if abs(homogeneous[2]) < 1e-12:
            raise ValueError("Point maps to infinity")
        return (float(homogeneous[0] / homogeneous[2]), float(homogeneous[1] / homogeneous[2]))

