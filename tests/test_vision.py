import numpy as np

from perception_to_action.vision import postprocess_probabilities, preprocess_frame


def test_preprocess_frame_returns_normalized_chw_tensor():
    frame = np.full((12, 16, 3), 255, dtype=np.uint8)
    result = preprocess_frame(frame)
    assert result.shape == (3, 12, 16)
    assert result.dtype == np.float32
    assert result.max() == 1.0


def test_postprocessing_removes_noise_and_resolves_overlap():
    target = np.zeros((20, 20), dtype=np.float32)
    context = np.zeros_like(target)
    target[4:10, 4:10] = 0.8
    target[0, 0] = 1.0
    context[6:12, 6:12] = 0.9
    masks = postprocess_probabilities(
        {"target": target, "context": context},
        threshold=0.5,
        minimum_component_area=4,
        morphology_radius=0,
    )
    assert not masks["target"][0, 0]
    assert not np.any(masks["target"] & masks["context"])

