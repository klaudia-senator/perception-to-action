"""Temporal event confirmation and target jump rejection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot

from .localization import TargetObservation


@dataclass(frozen=True)
class StabilizedObservation:
    observation: TargetObservation | None
    confirmed: bool
    reason: str


class TargetStabilizer:
    def __init__(
        self,
        window_size: int = 5,
        minimum_confirmations: int = 3,
        maximum_normalized_jump: float = 0.18,
    ) -> None:
        if not 1 <= minimum_confirmations <= window_size:
            raise ValueError("Confirmations must be between one and the window size")
        self.minimum_confirmations = minimum_confirmations
        self.maximum_jump = maximum_normalized_jump
        self._history: deque[TargetObservation | None] = deque(maxlen=window_size)
        self._last_accepted: TargetObservation | None = None

    def update(self, observation: TargetObservation | None) -> StabilizedObservation:
        if observation is not None and self._last_accepted is not None:
            x0, y0 = self._last_accepted.normalized
            x1, y1 = observation.normalized
            if hypot(x1 - x0, y1 - y0) > self.maximum_jump:
                self._history.append(None)
                return StabilizedObservation(None, False, "jump-rejected")
        self._history.append(observation)
        valid = [item for item in self._history if item is not None]
        if observation is None:
            return StabilizedObservation(None, False, "no-target")
        if len(valid) < self.minimum_confirmations:
            return StabilizedObservation(observation, False, "warming-up")
        x = sum(item.normalized[0] for item in valid) / len(valid)
        y = sum(item.normalized[1] for item in valid) / len(valid)
        confidence = sum(item.confidence for item in valid) / len(valid)
        area = round(sum(item.area for item in valid) / len(valid))
        pixel_x = sum(item.pixel[0] for item in valid) / len(valid)
        pixel_y = sum(item.pixel[1] for item in valid) / len(valid)
        stabilized = TargetObservation(
            pixel=(pixel_x, pixel_y),
            normalized=(x, y),
            confidence=confidence,
            area=area,
        )
        self._last_accepted = stabilized
        return StabilizedObservation(stabilized, True, "confirmed")

