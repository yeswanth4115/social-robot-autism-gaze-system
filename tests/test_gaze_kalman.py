import numpy as np

from gaze_kalman import (
    DEFAULT_DEAD_ZONE,
    DEFAULT_DROPOUT_TIMEOUT_S,
    DEFAULT_MAX_MOVEMENT,
    DEFAULT_MEASUREMENT_NOISE,
    DEFAULT_PROCESS_NOISE,
    KalmanGazeFilter,
)


def test_kalman_filter_initialization():
    kf = KalmanGazeFilter()
    assert kf.state is None
    assert kf.covariance is None
    assert kf.last_valid_time is None
    assert kf.dead_zone == DEFAULT_DEAD_ZONE
    assert kf.max_movement == DEFAULT_MAX_MOVEMENT
    assert kf.dropout_timeout == DEFAULT_DROPOUT_TIMEOUT_S

    # First update directly initializes state
    x, y = kf.update(100.0, 200.0, current_time=1.0)
    assert x == 100.0
    assert y == 200.0
    assert kf.state is not None
    assert kf.last_valid_time == 1.0


def test_kalman_filter_smoothing_and_dead_zone():
    kf = KalmanGazeFilter(dead_zone=2.0)
    kf.update(100.0, 200.0)

    # Sub-pixel / sub-deadband shift should be suppressed by dead_zone
    x, y = kf.update(100.5, 200.5)
    assert x == 100.0
    assert y == 200.0

    # Significant movement should update position smoothly
    x2, y2 = kf.update(120.0, 220.0)
    assert 100.0 < x2 <= 120.0
    assert 200.0 < y2 <= 220.0


def test_kalman_filter_max_movement_clamp():
    kf = KalmanGazeFilter(max_movement=35.0, dead_zone=0.0)
    kf.update(100.0, 100.0)

    # Massive spike (e.g. 500px jump from landmark glitch)
    x, y = kf.update(600.0, 100.0)
    # The displacement must be constrained by max_movement
    displacement = np.hypot(x - 100.0, y - 100.0)
    assert displacement <= 35.05


def test_kalman_filter_reset():
    kf = KalmanGazeFilter()
    kf.update(150.0, 250.0)
    assert kf.state is not None

    kf.reset()
    assert kf.state is None
    assert kf.covariance is None
    assert kf.last_valid_time is None


def test_kalman_filter_dropout_timeout_triggers_reset():
    kf = KalmanGazeFilter(dropout_timeout=0.5)
    kf.update(100.0, 100.0, current_time=0.0)

    # Within dropout tolerance (0.2s elapsed)
    reset_occurred = kf.handle_dropout(current_time=0.2)
    assert not reset_occurred
    assert kf.state is not None

    # Exceeds dropout threshold (0.6s elapsed > 0.5s)
    reset_occurred = kf.handle_dropout(current_time=0.6)
    assert reset_occurred
    assert kf.state is None
    assert kf.last_valid_time is None


def test_kalman_filter_none_input_handles_dropout():
    kf = KalmanGazeFilter(dropout_timeout=0.5)
    kf.update(100.0, 100.0, current_time=0.0)

    # None input within tolerance returns (None, None) without resetting
    x, y = kf.update(None, None, current_time=0.2)
    assert x is None and y is None
    assert kf.state is not None

    # None input after timeout resets filter
    x, y = kf.update(None, None, current_time=0.8)
    assert x is None and y is None
    assert kf.state is None

    # Next valid point re-initializes fresh without jump artifacts
    x_new, y_new = kf.update(500.0, 500.0, current_time=1.0)
    assert x_new == 500.0
    assert y_new == 500.0
