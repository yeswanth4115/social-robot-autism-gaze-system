"""
Validation Metrics & Saccade/Pursuit Classification for Eye-Object Contingency Analysis.
Computes spatial error, velocity correlation, physiological pursuit gain, and I-VT classification.
"""

import math
import collections
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np


@dataclass
class ValidationSample:
    """Instantaneous validation metrics for a single frame."""
    timestamp: float
    target_pos: Tuple[float, float]
    target_vel: Tuple[float, float]
    target_speed: float
    gaze_pos: Tuple[float, float]
    gaze_vel: Tuple[float, float]
    gaze_speed: float
    distance_error_px: float
    distance_error_deg: float
    is_on_target: bool
    eye_state: str            # "SMOOTH_PURSUIT", "SACCADE", "FIXATION", "BLINK", "DRIFT"
    pursuit_gain: float       # Gaze speed / Target speed
    directional_cosine: float # Cosine similarity between velocity vectors
    cross_correlation: float  # Lag-compensated velocity correlation
    optimal_lag_ms: float     # Estimated physiological pursuit latency in ms


class GazeObjectValidator:
    """
    Validates dynamic eye movements against the movements of an object in a game.
    Maintains a sliding temporal window for velocity estimation, lag compensation,
    cross-correlation, and saccade/pursuit segmentation.
    """
    def __init__(
        self,
        target_radius: float = 55.0,
        viewing_dist_cm: float = 60.0,
        screen_dpi: float = 96.0,
        history_window_sec: float = 3.0,
        saccade_speed_thresh_px: float = 650.0  # ~25-30 deg/sec
    ):
        self.target_radius = target_radius
        self.viewing_dist_cm = viewing_dist_cm
        self.px_per_cm = screen_dpi / 2.54
        self.history_window_sec = history_window_sec
        self.saccade_speed_thresh_px = saccade_speed_thresh_px

        # Sliding window buffer: stores (time, target_x, target_y, gaze_x, gaze_y, is_blink)
        self.history = collections.deque()

        # State tracking
        self.prev_target: Optional[Tuple[float, float, float]] = None  # (t, x, y)
        self.prev_gaze: Optional[Tuple[float, float, float]] = None    # (t, x, y)

    def _px_to_deg(self, px_dist: float) -> float:
        """Converts screen pixel distance to degrees of visual angle."""
        cm_dist = px_dist / self.px_per_cm
        rad = 2.0 * math.atan2(cm_dist / 2.0, self.viewing_dist_cm)
        return math.degrees(rad)

    def compute_frame(
        self,
        timestamp: float,
        target_pos: Tuple[float, float],
        gaze_pos: Tuple[float, float],
        is_blink: bool
    ) -> ValidationSample:
        """Computes comprehensive validation metrics for current timestamp."""
        tx, ty = target_pos
        gx, gy = gaze_pos

        # Append to sliding history buffer
        self.history.append((timestamp, tx, ty, gx, gy, is_blink))
        # Trim history older than history_window_sec
        while self.history and (timestamp - self.history[0][0]) > self.history_window_sec:
            self.history.popleft()

        # 1. Spatial Euclidean Distance
        dx = gx - tx
        dy = gy - ty
        dist_px = math.sqrt(dx * dx + dy * dy)
        dist_deg = self._px_to_deg(dist_px)
        is_on_target = dist_px <= self.target_radius and not is_blink

        # 2. Instantaneous Velocities
        # Target Velocity
        if self.prev_target is not None:
            dt_t = timestamp - self.prev_target[0]
            if dt_t > 1e-4:
                tvx = (tx - self.prev_target[1]) / dt_t
                tvy = (ty - self.prev_target[2]) / dt_t
            else:
                tvx, tvy = 0.0, 0.0
        else:
            tvx, tvy = 0.0, 0.0
        self.prev_target = (timestamp, tx, ty)
        target_speed = math.sqrt(tvx * tvx + tvy * tvy)

        # Gaze Velocity
        if self.prev_gaze is not None:
            dt_g = timestamp - self.prev_gaze[0]
            if dt_g > 1e-4:
                gvx = (gx - self.prev_gaze[1]) / dt_g
                gvy = (gy - self.prev_gaze[2]) / dt_g
            else:
                gvx, gvy = 0.0, 0.0
        else:
            gvx, gvy = 0.0, 0.0
        self.prev_gaze = (timestamp, gx, gy)
        gaze_speed = math.sqrt(gvx * gvx + gvy * gvy)

        # 3. Directional Cosine Similarity
        dot = (tvx * gvx) + (tvy * gvy)
        denom = (target_speed * gaze_speed) + 1e-6
        cos_sim = float(np.clip(dot / denom, -1.0, 1.0))

        # 4. Pursuit Gain (Gaze Speed / Target Speed)
        if target_speed > 30.0:
            gain = gaze_speed / target_speed
        else:
            gain = 1.0 if gaze_speed < 100.0 else 0.0

        # 5. Saccade vs Smooth Pursuit Classification (I-VT algorithm)
        if is_blink:
            eye_state = "BLINK"
        elif gaze_speed >= self.saccade_speed_thresh_px:
            eye_state = "SACCADE"
        elif target_speed > 40.0 and cos_sim > 0.35:
            eye_state = "SMOOTH_PURSUIT"
        elif target_speed < 40.0 and gaze_speed < 150.0:
            eye_state = "FIXATION"
        else:
            eye_state = "DRIFT"

        # 6. Sliding Window Cross-Correlation & Physiological Lag Estimation
        cross_corr, opt_lag_ms = self._compute_lag_cross_correlation()

        return ValidationSample(
            timestamp=timestamp,
            target_pos=target_pos,
            target_vel=(tvx, tvy),
            target_speed=target_speed,
            gaze_pos=gaze_pos,
            gaze_vel=(gvx, gvy),
            gaze_speed=gaze_speed,
            distance_error_px=dist_px,
            distance_error_deg=dist_deg,
            is_on_target=is_on_target,
            eye_state=eye_state,
            pursuit_gain=gain,
            directional_cosine=cos_sim,
            cross_correlation=cross_corr,
            optimal_lag_ms=opt_lag_ms
        )

    def _compute_lag_cross_correlation(self) -> Tuple[float, float]:
        """
        Calculates Pearson cross-correlation between target and gaze velocity profiles
        across physiologically realistic lags tau in [0 ms, 300 ms].
        Returns (max_correlation, optimal_lag_ms).
        """
        if len(self.history) < 30:
            return 0.0, 0.0

        data = list(self.history)
        times = np.array([d[0] for d in data])
        t_speeds = []
        g_speeds = []

        for i in range(1, len(data)):
            dt = times[i] - times[i - 1]
            if dt < 1e-4:
                continue
            # Target speed
            dtx = data[i][1] - data[i - 1][1]
            dty = data[i][2] - data[i - 1][2]
            ts = math.sqrt(dtx * dtx + dty * dty) / dt
            t_speeds.append(ts)

            # Gaze speed
            dgx = data[i][3] - data[i - 1][3]
            dgy = data[i][4] - data[i - 1][4]
            gs = math.sqrt(dgx * dgx + dgy * dgy) / dt
            g_speeds.append(gs)

        if len(t_speeds) < 25:
            return 0.0, 0.0

        ts_arr = np.array(t_speeds, dtype=np.float32)
        gs_arr = np.array(g_speeds, dtype=np.float32)

        # Standard deviations
        std_t = np.std(ts_arr)
        std_g = np.std(gs_arr)
        if std_t < 1.0 or std_g < 1.0:
            return 0.0, 0.0

        # Mean frame interval
        avg_dt = float(np.mean(np.diff(times)))
        if avg_dt < 1e-4:
            avg_dt = 1.0 / 60.0

        # Test lags from 0 to 300 ms in steps of frame dt
        max_shift = int(min(0.30 / avg_dt, len(ts_arr) // 3))
        best_corr = -1.0
        best_lag_ms = 0.0

        # Zero-centered signals
        norm_t = (ts_arr - np.mean(ts_arr)) / (std_t + 1e-6)
        norm_g = (gs_arr - np.mean(gs_arr)) / (std_g + 1e-6)

        for shift in range(0, max_shift + 1):
            if shift == 0:
                corr = np.mean(norm_g * norm_t)
            else:
                corr = np.mean(norm_g[shift:] * norm_t[:-shift])

            if corr > best_corr:
                best_corr = float(corr)
                best_lag_ms = float(shift * avg_dt * 1000.0)

        return float(np.clip(best_corr, -1.0, 1.0)), best_lag_ms

    def compute_dtw_distance(self, window_sec: float = 2.0) -> float:
        """
        Fast Dynamic Time Warping (DTW) distance between recent target and gaze trajectory curves.
        Measures structural trajectory congruence regardless of minor speed differences.
        """
        if len(self.history) < 20:
            return 0.0

        now = self.history[-1][0]
        recent = [d for d in self.history if (now - d[0]) <= window_sec]
        if len(recent) < 15:
            return 0.0

        # Subsample to ~30 points for fast real-time DTW
        step = max(1, len(recent) // 30)
        sub = recent[::step]

        target_pts = np.array([[d[1], d[2]] for d in sub], dtype=np.float32)
        gaze_pts = np.array([[d[3], d[4]] for d in sub], dtype=np.float32)

        n = len(target_pts)
        m = len(gaze_pts)

        # Distance matrix
        diff = target_pts[:, np.newaxis, :] - gaze_pts[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))

        # Accumulated cost matrix
        cost = np.full((n, m), np.inf, dtype=np.float32)
        cost[0, 0] = dist_mat[0, 0]

        for i in range(1, n):
            cost[i, 0] = cost[i - 1, 0] + dist_mat[i, 0]
        for j in range(1, m):
            cost[0, j] = cost[0, j - 1] + dist_mat[0, j]

        for i in range(1, n):
            for j in range(1, m):
                min_prev = min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
                cost[i, j] = dist_mat[i, j] + min_prev

        return float(cost[-1, -1] / (n + m))

    def reset(self):
        """Clears tracking history."""
        self.history.clear()
        self.prev_target = None
        self.prev_gaze = None
