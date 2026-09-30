"""Explicit safety-oriented state machine for the simulated controller."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ControllerState(str, Enum):
    IDLE = "idle"
    ARMED = "armed"
    TRACKING = "tracking"
    HOLD = "hold"
    FAULT = "fault"
    ESTOP = "emergency-stop"


@dataclass(frozen=True)
class ControlDecision:
    state: ControllerState
    should_command: bool
    reason: str


class SafetyController:
    """Gate simulated commands by state, confidence, stability and cooldown."""

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        required_stable_frames: int = 3,
        command_cooldown_frames: int = 8,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.required_stable_frames = required_stable_frames
        self.command_cooldown_frames = command_cooldown_frames
        self.state = ControllerState.IDLE
        self._stable_frames = 0
        self._cooldown = 0

    def arm(self) -> None:
        if self.state is ControllerState.ESTOP:
            raise RuntimeError("Emergency stop must be reset before arming")
        self.state = ControllerState.ARMED
        self._stable_frames = 0

    def hold(self) -> None:
        if self.state is not ControllerState.ESTOP:
            self.state = ControllerState.HOLD

    def emergency_stop(self) -> None:
        self.state = ControllerState.ESTOP
        self._stable_frames = 0

    def reset(self) -> None:
        self.state = ControllerState.IDLE
        self._stable_frames = 0
        self._cooldown = 0

    def update(self, confirmed: bool, confidence: float) -> ControlDecision:
        if self._cooldown:
            self._cooldown -= 1
        if self.state in {ControllerState.IDLE, ControllerState.HOLD, ControllerState.FAULT, ControllerState.ESTOP}:
            return ControlDecision(self.state, False, f"controller-{self.state.value}")
        if not confirmed or confidence < self.confidence_threshold:
            self._stable_frames = 0
            self.state = ControllerState.ARMED
            return ControlDecision(self.state, False, "target-not-stable")
        self._stable_frames += 1
        self.state = ControllerState.TRACKING
        if self._stable_frames < self.required_stable_frames:
            return ControlDecision(self.state, False, "controller-confirming")
        if self._cooldown:
            return ControlDecision(self.state, False, "command-cooldown")
        self._cooldown = self.command_cooldown_frames
        return ControlDecision(self.state, True, "simulated-command-approved")

