"""
One-Euro Filter and Stabilizing Utilities for Eye Gaze Tracking.
Features:
- Adaptive One-Euro filtering (eliminates jitter without lag)
- Velocity slew-rate clamping (prevents hyper-fast cursor whipping)
- Micro-tremor deadzone (stabilizes fixations)
- Real-time sensitivity and smoothing adjustment
"""

import math
import time
import numpy as np


class LowPassFilter:
    """First-order low-pass filter with exponential smoothing."""
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.s = None

    def set_alpha(self, alpha: float):
        self.alpha = max(0.0, min(1.0, alpha))

    def filter(self, value: float) -> float:
        if self.s is None:
            self.s = value
        else:
            self.s = self.alpha * value + (1.0 - self.alpha) * self.s
        return self.s

    def reset(self):
        self.s = None


class OneEuroFilter:
    """
    1-Euro Filter: Adapts cutoff frequency dynamically based on signal speed.
    - Low speed (fixations): Lower cutoff removes high-frequency jitter.
    - High speed (saccades / pursuit): Higher cutoff eliminates tracking lag.
    """
    def __init__(self, freq: float = 60.0, mincutoff: float = 0.5, beta: float = 0.008, dcutoff: float = 1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time = None

    @staticmethod
    def _compute_alpha(rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, timestamp: float = None) -> float:
        if timestamp is None:
            timestamp = time.perf_counter()

        if self.last_time is None:
            self.last_time = timestamp
            prev_x = x
            rate = self.freq
        else:
            dt = timestamp - self.last_time
            self.last_time = timestamp
            rate = 1.0 / dt if dt > 1e-5 else self.freq
            prev_x = self.x_filter.s if self.x_filter.s is not None else x

        # Filter the derivative (speed)
        dx = (x - prev_x) * rate
        edx = self.dx_filter.filter(dx)
        alpha_d = self._compute_alpha(rate, self.dcutoff)
        self.dx_filter.set_alpha(alpha_d)

        # Dynamic cutoff frequency
        cutoff = self.mincutoff + self.beta * abs(edx)

        # Filter the signal
        alpha = self._compute_alpha(rate, cutoff)
        self.x_filter.set_alpha(alpha)
        return self.x_filter.filter(x)

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None


class GazePointFilter2D:
    """
    Enhanced 2D Gaze Stabilizer with:
    1. One-Euro Adaptive Filter
    2. Slew-rate velocity clamping (prevents cursor from shooting across screen too fast)
    3. Micro-tremor deadzone
    4. Real-time sensitivity scaling
    """
    def __init__(
        self,
        freq: float = 60.0,
        mincutoff: float = 0.5,
        beta: float = 0.008,
        deadzone_px: float = 3.5,
        max_velocity_px_s: float = 1400.0,
        sensitivity: float = 0.70
    ):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.deadzone_px = deadzone_px
        self.max_velocity_px_s = max_velocity_px_s
        self.sensitivity = sensitivity  # 0.2 to 2.0 multiplier

        self.fx = OneEuroFilter(freq=freq, mincutoff=mincutoff, beta=beta)
        self.fy = OneEuroFilter(freq=freq, mincutoff=mincutoff, beta=beta)

        self.last_out_x = None
        self.last_out_y = None
        self.last_time = None

        # Center reference for sensitivity scaling
        self.ref_center_x = 640.0
        self.ref_center_y = 360.0

    def set_screen_center(self, cx: float, cy: float):
        self.ref_center_x = cx
        self.ref_center_y = cy

    def adjust_sensitivity(self, delta: float):
        """Adjusts gaze speed / sensitivity multiplier."""
        self.sensitivity = max(0.2, min(2.0, self.sensitivity + delta))
        print(f"[Stabilizer] Gaze Sensitivity: {self.sensitivity:.2f}x")

    def adjust_smoothing(self, delta: float):
        """Adjusts jitter dampening / smoothing."""
        self.mincutoff = max(0.1, min(2.0, self.mincutoff - delta))
        self.fx.mincutoff = self.mincutoff
        self.fy.mincutoff = self.mincutoff
        print(f"[Stabilizer] Smoothing Cutoff: {self.mincutoff:.2f} Hz (Lower = Smoother)")

    def filter(self, raw_x: float, raw_y: float, timestamp: float = None) -> tuple[float, float]:
        if timestamp is None:
            timestamp = time.perf_counter()

        # 1. Apply Sensitivity scaling relative to screen center
        # Reduces erratic overshoots when eyes turn slightly
        scaled_x = self.ref_center_x + (raw_x - self.ref_center_x) * self.sensitivity
        scaled_y = self.ref_center_y + (raw_y - self.ref_center_y) * self.sensitivity

        # 2. Deadzone: reject sub-threshold micro-jitters
        if self.last_out_x is not None and self.last_out_y is not None:
            dist = math.hypot(scaled_x - self.last_out_x, scaled_y - self.last_out_y)
            if dist < self.deadzone_px:
                # Small involuntary eye wobble: maintain previous smooth position
                return self.last_out_x, self.last_out_y

        # 3. Adaptive 1-Euro Low-Pass Filter
        filtered_x = self.fx.filter(scaled_x, timestamp)
        filtered_y = self.fy.filter(scaled_y, timestamp)

        # 4. Slew-rate velocity clamp (prevents hyperspeed cursor flying)
        if self.last_out_x is not None and self.last_time is not None:
            dt = timestamp - self.last_time
            if dt > 1e-4:
                max_step = self.max_velocity_px_s * dt
                dx = filtered_x - self.last_out_x
                dy = filtered_y - self.last_out_y
                step_dist = math.hypot(dx, dy)
                if step_dist > max_step:
                    scale = max_step / step_dist
                    filtered_x = self.last_out_x + dx * scale
                    filtered_y = self.last_out_y + dy * scale

        self.last_out_x = filtered_x
        self.last_out_y = filtered_y
        self.last_time = timestamp
        return filtered_x, filtered_y

    def reset(self):
        self.fx.reset()
        self.fy.reset()
        self.last_out_x = None
        self.last_out_y = None
        self.last_time = None
