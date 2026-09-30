"""Target localization from segmentation probabilities and masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TargetObservation:
    pixel: tuple[float, float]
    normalized: tuple[float, float]
    confidence: float
    area: int


def localize_target(
    probability: np.ndarray, mask: np.ndarray
) -> TargetObservation | None:
    probability = np.asarray(probability, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if probability.shape != mask.shape:
        raise ValueError("Probability map and mask must have matching shapes")
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    weights = probability[rows, cols]
    total = float(weights.sum())
    if total <= 0.0:
        return None
    x = float(np.dot(cols, weights) / total)
    y = float(np.dot(rows, weights) / total)
    height, width = mask.shape
    normalized = (x / max(width - 1, 1), y / max(height - 1, 1))
    return TargetObservation(
        pixel=(x, y),
        normalized=normalized,
        confidence=float(weights.mean()),
        area=int(rows.size),
    )

