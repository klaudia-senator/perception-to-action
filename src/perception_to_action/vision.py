"""Preprocessing, probability-mask cleanup and conflict resolution."""

from __future__ import annotations

from collections import deque

import numpy as np


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Convert an HWC uint8 RGB frame to normalized CHW float32."""
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an HxWx3 frame")
    return (array.astype(np.float32) / 255.0).transpose(2, 0, 1)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(logits, dtype=np.float32), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _largest_components(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    """Remove small 8-connected components without external CV dependencies."""
    source = np.asarray(mask, dtype=bool)
    height, width = source.shape
    visited = np.zeros_like(source, dtype=bool)
    output = np.zeros_like(source, dtype=bool)
    for row in range(height):
        for col in range(width):
            if not source[row, col] or visited[row, col]:
                continue
            component: list[tuple[int, int]] = []
            queue = deque([(row, col)])
            visited[row, col] = True
            while queue:
                current_row, current_col = queue.popleft()
                component.append((current_row, current_col))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = current_row + dr, current_col + dc
                        if (
                            0 <= nr < height
                            and 0 <= nc < width
                            and source[nr, nc]
                            and not visited[nr, nc]
                        ):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
            if len(component) >= minimum_area:
                rows, cols = zip(*component)
                output[rows, cols] = True
    return output


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant")
    result = np.zeros_like(mask, dtype=bool)
    size = 2 * radius + 1
    for row in range(size):
        for col in range(size):
            result |= padded[row : row + mask.shape[0], col : col + mask.shape[1]]
    return result


def _binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    return ~_binary_dilate(~mask, radius)


def postprocess_probabilities(
    probabilities: dict[str, np.ndarray],
    threshold: float = 0.55,
    minimum_component_area: int = 24,
    morphology_radius: int = 1,
) -> dict[str, np.ndarray]:
    """Threshold, clean and make class masks mutually exclusive."""
    if not probabilities:
        raise ValueError("At least one probability map is required")
    names = tuple(probabilities)
    stack = np.stack([np.asarray(probabilities[name], dtype=np.float32) for name in names])
    if stack.ndim != 3:
        raise ValueError("Each probability map must be two-dimensional")
    winners = np.argmax(stack, axis=0)
    masks: dict[str, np.ndarray] = {}
    for index, name in enumerate(names):
        mask = (stack[index] >= threshold) & (winners == index)
        if morphology_radius:
            mask = _binary_dilate(_binary_erode(mask, morphology_radius), morphology_radius)
        masks[name] = _largest_components(mask, minimum_component_area)
    return masks


class SyntheticInferenceBackend:
    """Deterministic color-based backend used by the public demo.

    The generated frames encode scene elements with distinct RGB channels. This
    backend makes the system runnable without private data or trained weights.
    """

    head_names = ("foreground", "target", "context", "marker")

    def predict(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        image = np.asarray(frame, dtype=np.float32) / 255.0
        red, green, blue = image[..., 0], image[..., 1], image[..., 2]
        return {
            "foreground": np.clip(0.75 * red + 0.15 * green, 0.0, 1.0),
            "target": np.clip(green - 0.25 * red - 0.25 * blue, 0.0, 1.0),
            "context": np.clip(blue - 0.20 * red, 0.0, 1.0),
            "marker": np.clip(0.55 * red + 0.55 * blue - 0.20 * green, 0.0, 1.0),
        }

