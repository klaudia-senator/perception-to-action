"""End-to-end orchestration for perception, mapping and simulated action."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controller import ControlDecision, SafetyController
from .localization import TargetObservation, localize_target
from .mapping import CoordinateMapper
from .robot import SimulatedRobotClient
from .stabilization import StabilizedObservation, TargetStabilizer
from .vision import postprocess_probabilities


@dataclass(frozen=True)
class PipelineResult:
    masks: dict[str, np.ndarray]
    raw_target: TargetObservation | None
    stabilized_target: StabilizedObservation
    workspace_target: tuple[float, float] | None
    decision: ControlDecision


class PerceptionToActionPipeline:
    def __init__(
        self,
        inference_backend,
        mapper: CoordinateMapper,
        stabilizer: TargetStabilizer,
        controller: SafetyController,
        robot: SimulatedRobotClient,
        threshold: float = 0.55,
        minimum_component_area: int = 24,
        morphology_radius: int = 1,
    ) -> None:
        self.inference = inference_backend
        self.mapper = mapper
        self.stabilizer = stabilizer
        self.controller = controller
        self.robot = robot
        self.threshold = threshold
        self.minimum_component_area = minimum_component_area
        self.morphology_radius = morphology_radius

    def process(self, frame: np.ndarray) -> PipelineResult:
        probabilities = self.inference.predict(frame)
        masks = postprocess_probabilities(
            probabilities,
            threshold=self.threshold,
            minimum_component_area=self.minimum_component_area,
            morphology_radius=self.morphology_radius,
        )
        if "target" not in masks:
            raise KeyError("Inference backend must provide a 'target' head")
        raw_target = localize_target(probabilities["target"], masks["target"])
        stabilized = self.stabilizer.update(raw_target)
        confidence = stabilized.observation.confidence if stabilized.observation else 0.0
        decision = self.controller.update(stabilized.confirmed, confidence)
        workspace_target = None
        if stabilized.confirmed and stabilized.observation is not None:
            workspace_target = self.mapper.image_to_workspace(stabilized.observation.normalized)
        if decision.should_command and workspace_target is not None:
            self.robot.move_to(workspace_target)
        return PipelineResult(
            masks=masks,
            raw_target=raw_target,
            stabilized_target=stabilized,
            workspace_target=workspace_target,
            decision=decision,
        )

