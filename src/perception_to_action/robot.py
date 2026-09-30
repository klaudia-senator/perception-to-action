"""Simulation-only robot boundary.

This module intentionally contains no networking, serial communication or
hardware command serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimulatedRobotClient:
    command_log: list[tuple[float, float]] = field(default_factory=list)

    def move_to(self, target: tuple[float, float]) -> None:
        x, y = target
        if not (-1.0 <= x <= 1.0 and -1.0 <= y <= 1.0):
            raise ValueError("Synthetic target must stay inside the normalized demo workspace")
        self.command_log.append((float(x), float(y)))

