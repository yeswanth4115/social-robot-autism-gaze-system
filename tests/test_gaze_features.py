import numpy as np

from gaze_features import (
    estimate_gaze_confidence,
    extract_eye_metrics,
    feature_names,
    get_features,
)


class Landmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def test_extract_eye_metrics_returns_quality_metrics():
    face = [
        Landmark(0.50, 0.50),
    ] * 478

    left_eye = [33, 133]
    right_eye = [362, 263]
    left_iris = [468, 469, 470, 471, 472]
    right_iris = [473, 474, 475, 476, 477]

    for idx in left_eye:
        face[idx] = Landmark(0.40, 0.45)
    face[33] = Landmark(0.35, 0.45)
    face[133] = Landmark(0.45, 0.45)

    for idx in right_eye:
        face[idx] = Landmark(0.65, 0.45)
    face[362] = Landmark(0.60, 0.45)
    face[263] = Landmark(0.70, 0.45)

    for idx in left_iris:
        face[idx] = Landmark(0.38 + (idx - 468) * 0.01, 0.47)
    for idx in right_iris:
        face[idx] = Landmark(0.62 + (idx - 473) * 0.01, 0.47)

    metrics = extract_eye_metrics(face)

    assert metrics is not None
    assert set(metrics.keys()) >= {
        "left_iris_center",
        "right_iris_center",
        "normalized_iris_x",
        "normalized_iris_y",
        "left_eye_geometry",
        "right_eye_geometry",
        "eye_aspect_ratio",
        "head_yaw",
        "head_pitch",
        "head_roll",
    }
    assert np.isfinite(metrics["normalized_iris_x"]) and np.isfinite(metrics["normalized_iris_y"])


def test_get_features_supports_head_pose_and_compatible_output():
    face = [Landmark(0.5, 0.5) for _ in range(478)]
    face[33] = Landmark(0.40, 0.48)
    face[133] = Landmark(0.50, 0.48)
    face[362] = Landmark(0.60, 0.48)
    face[263] = Landmark(0.70, 0.48)

    for idx in range(468, 473):
        face[idx] = Landmark(0.42 + (idx - 468) * 0.02, 0.47)
    for idx in range(473, 478):
        face[idx] = Landmark(0.58 + (idx - 473) * 0.02, 0.47)

    head_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)

    vector = get_features(face, head_pose)
    assert vector is not None
    assert len(vector) == 7
    assert feature_names(use_head_pose=True) == [
        "left_h",
        "left_v",
        "right_h",
        "right_v",
        "yaw",
        "pitch",
        "roll",
    ]
    assert np.all(np.isfinite(vector))


def test_estimate_gaze_confidence_returns_bounded_score_for_valid_metrics():
    metrics = {
        "left_eye_geometry": {"width": 0.18, "height": 0.08},
        "right_eye_geometry": {"width": 0.18, "height": 0.08},
        "left_horizontal": 0.05,
        "left_vertical": 0.04,
        "right_horizontal": -0.06,
        "right_vertical": 0.03,
        "head_yaw": 0.12,
        "head_pitch": 0.10,
        "head_roll": 0.04,
        "eye_aspect_ratio": 0.30,
    }

    score = estimate_gaze_confidence(metrics)
    assert 0.0 <= score <= 1.0
    assert score > 0.5


def test_estimate_gaze_confidence_with_closed_eyes_penalizes_score():
    closed_eye_metrics = {
        "left_eye_geometry": {"width": 0.18, "height": 0.01},
        "right_eye_geometry": {"width": 0.18, "height": 0.01},
        "left_horizontal": 0.0,
        "left_vertical": 0.0,
        "right_horizontal": 0.0,
        "right_vertical": 0.0,
        "head_yaw": 0.0,
        "head_pitch": 0.0,
        "head_roll": 0.0,
        "eye_aspect_ratio": 0.02,
    }
    open_eye_metrics = dict(closed_eye_metrics, eye_aspect_ratio=0.30)
    score_closed = estimate_gaze_confidence(closed_eye_metrics)
    score_open = estimate_gaze_confidence(open_eye_metrics)
    assert score_closed < score_open
    assert score_closed <= 0.85


def test_estimate_gaze_confidence_with_extreme_head_pose_penalizes_score():
    normal_pose = {
        "left_eye_geometry": {"width": 0.18, "height": 0.08},
        "right_eye_geometry": {"width": 0.18, "height": 0.08},
        "left_horizontal": 0.0,
        "left_vertical": 0.0,
        "right_horizontal": 0.0,
        "right_vertical": 0.0,
        "head_yaw": 0.0,
        "head_pitch": 0.0,
        "head_roll": 0.0,
        "eye_aspect_ratio": 0.25,
    }
    extreme_pose = dict(normal_pose, head_yaw=1.2, head_pitch=0.8, head_roll=0.6)
    score_normal = estimate_gaze_confidence(normal_pose)
    score_extreme = estimate_gaze_confidence(extreme_pose)
    assert score_extreme < score_normal
    assert score_extreme < score_normal * 0.85


def test_estimate_gaze_confidence_with_asymmetric_eyes_penalizes_score():
    symmetric = {
        "left_eye_geometry": {"width": 0.18, "height": 0.08},
        "right_eye_geometry": {"width": 0.18, "height": 0.08},
        "left_horizontal": 0.1,
        "left_vertical": 0.05,
        "right_horizontal": 0.1,
        "right_vertical": 0.05,
        "head_yaw": 0.0,
        "head_pitch": 0.0,
        "head_roll": 0.0,
        "eye_aspect_ratio": 0.25,
    }
    asymmetric = dict(
        symmetric,
        left_horizontal=0.6,
        right_horizontal=-0.6,
        left_vertical=0.4,
        right_vertical=-0.4,
    )
    score_sym = estimate_gaze_confidence(symmetric)
    score_asym = estimate_gaze_confidence(asymmetric)
    assert score_asym < score_sym


def test_estimate_gaze_confidence_handles_invalid_or_none_metrics():
    assert estimate_gaze_confidence(None) == 0.0
    assert 0.0 <= estimate_gaze_confidence({}) <= 1.0
    assert estimate_gaze_confidence("invalid") == 0.0
    assert estimate_gaze_confidence(123) == 0.0


def test_min_gaze_confidence_threshold_gating():
    from gaze_features import MIN_GAZE_CONFIDENCE

    assert MIN_GAZE_CONFIDENCE == 0.35

    poor_metrics = {
        "left_eye_geometry": {"width": 0.02, "height": 0.01},
        "right_eye_geometry": {"width": 0.02, "height": 0.01},
        "left_horizontal": 0.5,
        "left_vertical": 0.5,
        "right_horizontal": -0.5,
        "right_vertical": -0.5,
        "head_yaw": 1.5,
        "head_pitch": 1.0,
        "head_roll": 1.0,
        "eye_aspect_ratio": 0.02,
    }
    score = estimate_gaze_confidence(poor_metrics)
    assert score < MIN_GAZE_CONFIDENCE, f"Expected poor frame to be rejected by gate: {score}"

