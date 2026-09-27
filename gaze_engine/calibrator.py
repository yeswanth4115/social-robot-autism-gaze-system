"""
Gaze Calibrator: Maps normalized eye features to screen pixel coordinates
using outlier-rejected polynomial regression and an intuitive multi-point sequence.
"""

import os
import json
import time
import numpy as np
from typing import List, Tuple, Optional


class GazeCalibrator:
    """
    Polynomial & Ridge-regularized mapping from iris/face features to screen pixel coordinates.
    Includes median-filtered sample aggregation and quick-recenter (tare) offset.
    """
    def __init__(self, screen_width: int = 1280, screen_height: int = 720, config_path: Optional[str] = None):
        self.screen_width = screen_width
        self.screen_height = screen_height
        if config_path is None:
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(cur_dir, "calibration_profile.json")
        self.config_path = config_path

        # Weights: Shape (N_features + 1, 2)
        self.weights: Optional[np.ndarray] = None
        self.is_calibrated: bool = False
        self.mean_residual_px: float = 0.0

        # Live tare / offset compensation
        self.offset_x: float = 0.0
        self.offset_y: float = 0.0

        # Interactive calibration state
        # 9 standard points (normalized screen coordinates)
        self.calib_points = [
            (0.50, 0.50),  # 1. Center (crucial anchor)
            (0.18, 0.18),  # 2. Top-Left
            (0.50, 0.18),  # 3. Top-Center
            (0.82, 0.18),  # 4. Top-Right
            (0.18, 0.50),  # 5. Mid-Left
            (0.82, 0.50),  # 6. Mid-Right
            (0.18, 0.82),  # 7. Bot-Left
            (0.50, 0.82),  # 8. Bot-Center
            (0.82, 0.82),  # 9. Bot-Right
        ]
        self.current_point_idx: int = 0
        self.point_start_time: float = 0.0
        self.point_duration: float = 2.4     # 2.4s per point for comfortable, relaxed fixation
        self.point_settle_delay: float = 0.7 # Wait 700ms for saccadic landing

        # Sample storage per calibration point
        self._current_point_samples: List[np.ndarray] = []
        self.aggregated_X: List[np.ndarray] = []  # Feature rows
        self.aggregated_Y: List[Tuple[float, float]] = []  # True screen points

        # Try loading existing calibration
        self.load_calibration()

        if not self.is_calibrated:
            self._init_default_mapping()

    def _init_default_mapping(self):
        """Initializes a gentle, well-behaved baseline mapping."""
        # 10 features: [1, u, v, u^2, v^2, uv, yaw, pitch, blend_h, blend_v]
        w = np.zeros((10, 2), dtype=np.float32)
        # Bias: screen center
        w[0, 0] = self.screen_width / 2.0
        w[0, 1] = self.screen_height / 2.0
        # Gentle iris gain to prevent hyper-fast cursor flying
        w[1, 0] = self.screen_width * 0.75   # X responds moderately to u
        w[2, 1] = self.screen_height * 0.75  # Y responds moderately to v
        self.weights = w
        self.is_calibrated = False

    def _build_feature_row(self, raw_features: np.ndarray) -> np.ndarray:
        return np.concatenate(([1.0], raw_features))

    def recenter_to_point(self, current_predicted_x: float, current_predicted_y: float, target_x: float, target_y: float):
        """Tares / recenters the gaze estimate by computing an alignment offset."""
        self.offset_x = target_x - current_predicted_x
        self.offset_y = target_y - current_predicted_y
        print(f"[Calibrator] Recentered gaze offset: dx={self.offset_x:.1f}, dy={self.offset_y:.1f}")

    def reset_recenter(self):
        self.offset_x = 0.0
        self.offset_y = 0.0

    def start_calibration(self):
        """Starts a fresh 9-point calibration routine."""
        self.current_point_idx = 0
        self.point_start_time = time.perf_counter()
        self._current_point_samples = []
        self.aggregated_X = []
        self.aggregated_Y = []
        self.reset_recenter()
        print("\n[Calibrator] Starting 9-point calibration routine. Fixate steadily on each target.")

    def get_current_calibration_target(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Returns (screen_x, screen_y, progress_fraction, remaining_seconds)
        or None if calibration is complete or inactive.
        """
        if self.current_point_idx >= len(self.calib_points):
            return None

        nx, ny = self.calib_points[self.current_point_idx]
        sx = nx * self.screen_width
        sy = ny * self.screen_height

        now = time.perf_counter()
        elapsed = now - self.point_start_time
        progress = min(1.0, elapsed / self.point_duration)
        remaining = max(0.0, self.point_duration - elapsed)
        return sx, sy, progress, remaining

    def record_calibration_sample(self, raw_features: np.ndarray, is_blink: bool):
        """Records feature samples, rejects blinks, and averages steady fixation."""
        if self.current_point_idx >= len(self.calib_points):
            return

        now = time.perf_counter()
        elapsed = now - self.point_start_time

        # Accumulate only during stable fixation window and not blinking
        if elapsed > self.point_settle_delay and not is_blink:
            row = self._build_feature_row(raw_features)
            self._current_point_samples.append(row)

        # Point duration elapsed: commit median feature vector for this target point
        if elapsed >= self.point_duration:
            nx, ny = self.calib_points[self.current_point_idx]
            target_screen_x = nx * self.screen_width
            target_screen_y = ny * self.screen_height

            if len(self._current_point_samples) >= 8:
                # Compute median feature row across all valid frames at this point
                # (Resistant to sudden micro-saccades and eyelid twitches)
                point_matrix = np.array(self._current_point_samples, dtype=np.float32)
                median_features = np.median(point_matrix, axis=0)

                # Store multiple copies of the clean median representation to balance points
                for _ in range(5):
                    self.aggregated_X.append(median_features)
                    self.aggregated_Y.append((target_screen_x, target_screen_y))

            self._current_point_samples = []
            self.current_point_idx += 1
            self.point_start_time = time.perf_counter()

            if self.current_point_idx >= len(self.calib_points):
                self._solve_regression()

    def _solve_regression(self):
        """Solves L2 Ridge regression with outlier-rejected median calibration points."""
        if len(self.aggregated_X) < 15:
            print("[Calibrator] Insufficient clean samples. Maintaining existing mapping.")
            return

        X = np.array(self.aggregated_X, dtype=np.float32)
        Y = np.array(self.aggregated_Y, dtype=np.float32)

        # Regularization parameter
        ridge_lambda = 0.15
        n_features = X.shape[1]
        reg_matrix = ridge_lambda * np.identity(n_features, dtype=np.float32)
        reg_matrix[0, 0] = 0.0  # Do not regularize bias

        try:
            xtx = X.T @ X + reg_matrix
            xty = X.T @ Y
            self.weights = np.linalg.solve(xtx, xty)
            self.is_calibrated = True

            # Residual accuracy
            predictions = X @ self.weights
            errors = np.linalg.norm(predictions - Y, axis=1)
            self.mean_residual_px = float(np.mean(errors))
            print(f"[Calibrator] SUCCESS! Calibration completed with residual error: {self.mean_residual_px:.1f} px.")
            self.save_calibration()
        except Exception as e:
            print(f"[Calibrator] Error solving calibration regression: {e}")
            self._init_default_mapping()

    def predict(self, raw_features: np.ndarray) -> Tuple[float, float]:
        """Maps raw eye feature vector to screen pixel coordinates with offset compensation."""
        if self.weights is None:
            self._init_default_mapping()

        row = self._build_feature_row(raw_features)
        pred = row @ self.weights

        # Add recenter offset
        px = float(pred[0] + self.offset_x)
        py = float(pred[1] + self.offset_y)

        # Screen boundary clamping
        px = float(np.clip(px, 0, self.screen_width))
        py = float(np.clip(py, 0, self.screen_height))
        return px, py

    def save_calibration(self):
        if self.weights is None:
            return
        data = {
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "weights": self.weights.tolist(),
            "mean_residual_px": self.mean_residual_px,
            "is_calibrated": self.is_calibrated
        }
        try:
            with open(self.config_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[Calibrator] Profile saved -> {self.config_path}")
        except Exception as e:
            print(f"[Calibrator] Failed to save calibration: {e}")

    def load_calibration(self) -> bool:
        if not os.path.exists(self.config_path):
            return False
        try:
            with open(self.config_path, "r") as f:
                data = json.load(f)
            self.weights = np.array(data["weights"], dtype=np.float32)
            self.mean_residual_px = data.get("mean_residual_px", 0.0)
            self.is_calibrated = data.get("is_calibrated", True)
            print(f"[Calibrator] Loaded profile from {self.config_path} (Residual: {self.mean_residual_px:.1f}px)")
            return True
        except Exception as e:
            print(f"[Calibrator] Failed to load calibration: {e}")
            return False
