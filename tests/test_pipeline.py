import numpy as np

from perception_to_action.controller import SafetyController
from perception_to_action.mapping import CoordinateMapper
from perception_to_action.pipeline import PerceptionToActionPipeline
from perception_to_action.robot import SimulatedRobotClient
from perception_to_action.sources import synthetic_frames
from perception_to_action.stabilization import TargetStabilizer
from perception_to_action.vision import SyntheticInferenceBackend


def test_end_to_end_pipeline_emits_only_simulated_commands():
    robot = SimulatedRobotClient()
    controller = SafetyController(0.6, required_stable_frames=1, command_cooldown_frames=1)
    controller.arm()
    pipeline = PerceptionToActionPipeline(
        SyntheticInferenceBackend(),
        CoordinateMapper([[2, 0, -1], [0, 2, -1], [0, 0, 1]]),
        TargetStabilizer(window_size=2, minimum_confirmations=1, maximum_normalized_jump=0.3),
        controller,
        robot,
        threshold=0.5,
        minimum_component_area=10,
        morphology_radius=0,
    )
    for frame in synthetic_frames(width=96, height=72, frames=5):
        result = pipeline.process(frame)
        assert isinstance(result.masks["target"], np.ndarray)
    assert robot.command_log
    assert all(-1.0 <= value <= 1.0 for command in robot.command_log for value in command)

