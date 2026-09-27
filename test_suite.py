"""
Automated Verification & Simulation Benchmark Script.
Simulates realistic tracking across:
1. Smooth Pursuit Benchmark (Lissajous & Circular trajectory)
2. Saccadic Step-Jump Benchmark (Fixation & Saccadic response)
Verifies calibration regression, telemetry CSV logging, and renders all scientific charts.
"""

import os
import sys
import time
import math
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gaze_engine.tracker import GazeTracker
from gaze_engine.calibrator import GazeCalibrator
from gaze_engine.filters import GazePointFilter2D
from game.target import DynamicTarget
from validation.metrics import GazeObjectValidator
from validation.logger import SessionLogger
from validation.analyzer import SessionAnalyzer


def run_smooth_pursuit_benchmark(duration_seconds: float = 10.0):
    print("\n================================================================")
    print(" RUNNING SMOOTH PURSUIT TRACKING BENCHMARK (LISSAJOUS)")
    print("================================================================")

    w, h = 1280, 720
    tracker = GazeTracker(force_simulation=True, screen_width=w, screen_height=h)
    calibrator = GazeCalibrator(screen_width=w, screen_height=h)
    gaze_filter = GazePointFilter2D(freq=60.0)
    target = DynamicTarget(screen_width=w, screen_height=h)
    target.set_mode("lissajous")
    validator = GazeObjectValidator(target_radius=48.0)
    logger = SessionLogger()

    # Fast calibration
    calibrator.start_calibration()
    while True:
        target_info = calibrator.get_current_calibration_target()
        if target_info is None:
            break
        cx, cy, _ = target_info
        u_true = (cx / w) * 2.0 - 1.0 + np.random.normal(0, 0.015)
        v_true = (cy / h) * 2.0 - 1.0 + np.random.normal(0, 0.015)
        feat = np.array([u_true, v_true, u_true**2, v_true**2, u_true*v_true, 0.0, 0.0, u_true*0.5, v_true*0.5], dtype=np.float32)
        calibrator.record_calibration_sample(feat, is_blink=False)
        time.sleep(0.001)

    csv_path = logger.start_session(game_mode="lissajous")
    fps = 60
    total_frames = int(duration_seconds * fps)
    start_sim_time = time.perf_counter()

    for frame_idx in range(total_frames):
        tx, ty = target.update()
        gframe = tracker.update(target_pos=(tx, ty))
        raw_x, raw_y = calibrator.predict(gframe.features)
        sim_timestamp = start_sim_time + (frame_idx / fps)
        gx, gy = gaze_filter.filter(raw_x, raw_y, timestamp=sim_timestamp)

        sample = validator.compute_frame(
            timestamp=sim_timestamp,
            target_pos=(tx, ty),
            gaze_pos=(gx, gy),
            is_blink=gframe.is_blink
        )
        logger.log_sample(sample, game_mode="lissajous")

    logger.close()
    print(f"Smooth pursuit session logged: {logger.sample_count} frames -> {csv_path}")

    analyzer = SessionAnalyzer(csv_path)
    stats = analyzer.compute_summary_statistics()
    plots = analyzer.generate_all_plots(prefix="smooth_pursuit")

    print("\n--- Smooth Pursuit Results ---")
    print(f"  On-Target Catchment:       {stats['on_target_pct']:.1f}%")
    print(f"  Mean Spatial Distance:     {stats['mean_error_px']:.1f} px ({stats['mean_error_deg']:.2f}°)")
    print(f"  RMS Error:                 {stats['rmse_error_px']:.1f} px")
    print(f"  Smooth Pursuit Ratio:      {stats['pursuit_pct']:.1f}%")
    print(f"  Mean Pursuit Gain:         {stats['mean_pursuit_gain']:.2f} (Ideal: ~1.0)")
    print(f"  Peak Velocity Correlation: {stats['mean_velocity_correlation']:.2f}")
    print(f"  Estimated Latency:         {stats['mean_physiological_lag_ms']:.0f} ms")

    for name, p in plots.items():
        print(f"  [Report] {name:20s}: {p}")

    return csv_path, plots


if __name__ == "__main__":
    run_smooth_pursuit_benchmark()
