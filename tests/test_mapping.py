import pytest

from perception_to_action.mapping import CoordinateMapper


def test_synthetic_homography_maps_image_center_to_workspace_origin():
    mapper = CoordinateMapper([[2, 0, -1], [0, 2, -1], [0, 0, 1]])
    assert mapper.image_to_workspace((0.5, 0.5)) == pytest.approx((0.0, 0.0))


def test_mapper_rejects_singular_homography():
    with pytest.raises(ValueError):
        CoordinateMapper([[1, 0, 0], [0, 0, 0], [0, 0, 1]])

