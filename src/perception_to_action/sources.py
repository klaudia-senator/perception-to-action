"""Synthetic, video-file and camera frame sources."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def synthetic_frames(width: int = 320, height: int = 240, frames: int = 120) -> Iterator[np.ndarray]:
    yy, xx = np.mgrid[:height, :width]
    for index in range(frames):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[..., 2] = np.linspace(18, 48, width, dtype=np.uint8)
        center_x = int(width * (0.20 + 0.60 * index / max(frames - 1, 1)))
        center_y = int(height * (0.50 + 0.12 * np.sin(index / 9.0)))
        radius = max(7, min(width, height) // 18)
        target = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
        image[target] = (25, 245, 45)
        context = (xx - width * 0.72) ** 2 + (yy - height * 0.25) ** 2 <= (radius * 1.4) ** 2
        image[context] = (35, 55, 210)
        marker = (abs(xx - width * 0.15) < radius // 2) & (abs(yy - height * 0.18) < radius // 2)
        image[marker] = (220, 20, 220)
        yield image


def opencv_frames(source: str | int) -> Iterator[np.ndarray]:
    """Yield RGB frames from a video path or camera index using OpenCV."""
    try:
        import cv2
    except ImportError as error:  # pragma: no cover
        raise ImportError("Install requirements.txt to use video or camera input") from error
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input source: {source}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()

