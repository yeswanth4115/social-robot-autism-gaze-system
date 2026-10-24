import numpy as np
import pytest

from gaze_calibration import (
    fit_four_point_homography,
    gaze_coordinates,
    transform_gaze_coordinates,
)


def test_four_point_homography_maps_all_calibration_targets():
    features = np.array(
        [
            [-0.2, -0.1, -0.18, -0.12],
            [0.2, -0.1, 0.18, -0.12],
            [0.2, 0.1, 0.18, 0.12],
            [-0.2, 0.1, -0.18, 0.12],
        ]
    )
    targets = np.array(
        [
            [100, 80],
            [1436, 80],
            [1436, 784],
            [100, 784],
        ]
    )

    homography = fit_four_point_homography(features, targets)
    projected = transform_gaze_coordinates(
        gaze_coordinates(features),
        homography,
    )

    np.testing.assert_allclose(projected, targets, atol=1e-4)


def test_four_point_homography_maps_interior_point_consistently():
    features = np.array(
        [
            [-1.0, -1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0, 1.0],
        ]
    )
    targets = np.array([[0, 0], [1000, 0], [1000, 800], [0, 800]])
    homography = fit_four_point_homography(features, targets)

    projected = transform_gaze_coordinates([[0, 0]], homography)

    np.testing.assert_allclose(projected, [[500, 400]], atol=1e-4)


def test_four_point_homography_rejects_degenerate_gaze_points():
    features = np.array(
        [
            [-1.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [2.0, 0.0, 2.0, 0.0],
        ]
    )
    targets = np.array([[0, 0], [1000, 0], [1000, 800], [0, 800]])

    with pytest.raises(ValueError, match="convex quadrilateral"):
        fit_four_point_homography(features, targets)
