from perception_to_action.localization import TargetObservation
from perception_to_action.stabilization import TargetStabilizer


def observation(x: float, y: float, confidence: float = 0.9) -> TargetObservation:
    return TargetObservation((x * 100, y * 100), (x, y), confidence, 100)


def test_requires_repeated_observations_before_confirmation():
    stabilizer = TargetStabilizer(window_size=4, minimum_confirmations=3, maximum_normalized_jump=0.2)
    assert not stabilizer.update(observation(0.4, 0.4)).confirmed
    assert not stabilizer.update(observation(0.41, 0.4)).confirmed
    assert stabilizer.update(observation(0.42, 0.4)).confirmed


def test_rejects_sudden_target_jump_after_lock():
    stabilizer = TargetStabilizer(window_size=3, minimum_confirmations=2, maximum_normalized_jump=0.1)
    stabilizer.update(observation(0.2, 0.2))
    stabilizer.update(observation(0.21, 0.2))
    result = stabilizer.update(observation(0.8, 0.8))
    assert not result.confirmed
    assert result.reason == "jump-rejected"

