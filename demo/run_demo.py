"""Run the complete pipeline on safe synthetic frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from perception_to_action.controller import SafetyController
from perception_to_action.mapping import CoordinateMapper
from perception_to_action.pipeline import PerceptionToActionPipeline
from perception_to_action.robot import SimulatedRobotClient
from perception_to_action.sources import opencv_frames, synthetic_frames
from perception_to_action.stabilization import TargetStabilizer
from perception_to_action.vision import SyntheticInferenceBackend


def build_pipeline(config: dict) -> tuple[PerceptionToActionPipeline, SimulatedRobotClient]:
    perception = config["perception"]
    stabilization = config["stabilization"]
    controller_config = config["controller"]
    robot = SimulatedRobotClient()
    controller = SafetyController(**controller_config)
    controller.arm()
    pipeline = PerceptionToActionPipeline(
        inference_backend=SyntheticInferenceBackend(),
        mapper=CoordinateMapper(config["mapping"]["homography"]),
        stabilizer=TargetStabilizer(**stabilization),
        controller=controller,
        robot=robot,
        **perception,
    )
    return pipeline, robot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/demo_config.json"))
    parser.add_argument("--video", type=Path, help="Optional local video input")
    parser.add_argument("--camera", type=int, help="Optional camera index")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    pipeline, robot = build_pipeline(config)
    input_config = config["input"]
    if args.video:
        frames = opencv_frames(str(args.video))
    elif args.camera is not None:
        frames = opencv_frames(args.camera)
    else:
        frames = synthetic_frames(
            width=input_config["width"],
            height=input_config["height"],
            frames=input_config["frames"],
        )

    processed = 0
    for processed, frame in enumerate(frames, start=1):
        result = pipeline.process(frame)
        if result.decision.should_command:
            x, y = result.workspace_target or (0.0, 0.0)
            print(
                f"frame={processed:03d} state={result.decision.state.value:<8} "
                f"simulated_target=({x:+.3f}, {y:+.3f})"
            )
    print(f"Processed {processed} frames; emitted {len(robot.command_log)} simulated commands.")


if __name__ == "__main__":
    main()

