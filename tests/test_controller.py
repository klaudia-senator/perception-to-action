from perception_to_action.controller import ControllerState, SafetyController


def test_controller_requires_arm_stability_and_confidence():
    controller = SafetyController(
        confidence_threshold=0.7,
        required_stable_frames=2,
        command_cooldown_frames=3,
    )
    assert not controller.update(True, 1.0).should_command
    controller.arm()
    assert not controller.update(True, 0.5).should_command
    assert not controller.update(True, 0.9).should_command
    decision = controller.update(True, 0.9)
    assert decision.should_command
    assert decision.state is ControllerState.TRACKING
    assert not controller.update(True, 0.9).should_command


def test_emergency_stop_latches_until_reset():
    controller = SafetyController(required_stable_frames=1)
    controller.arm()
    controller.emergency_stop()
    assert not controller.update(True, 1.0).should_command
    assert controller.state is ControllerState.ESTOP
    controller.reset()
    assert controller.state is ControllerState.IDLE

