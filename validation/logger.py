"""
Session Data Logger: Records high-frequency (60 Hz) time-series tracking telemetry to CSV.
"""

import os
import csv
import time
from typing import Optional
from .metrics import ValidationSample


class SessionLogger:
    """
    Asynchronous / Buffered CSV logger for eye gaze & game target telemetry.
    """
    def __init__(self, log_dir: Optional[str] = None):
        if log_dir is None:
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(os.path.dirname(cur_dir), "logs")
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.current_filepath: Optional[str] = None
        self.file_handle = None
        self.csv_writer = None
        self.session_start_time: float = 0.0
        self.is_logging: bool = False
        self.sample_count: int = 0

    def start_session(self, game_mode: str = "lissajous") -> str:
        """Initializes a new CSV log file."""
        self.close()

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"gaze_session_{timestamp_str}.csv"
        self.current_filepath = os.path.join(self.log_dir, filename)

        self.file_handle = open(self.current_filepath, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.file_handle)

        # Header
        headers = [
            "timestamp",
            "elapsed_sec",
            "target_x",
            "target_y",
            "target_vx",
            "target_vy",
            "target_speed",
            "gaze_x",
            "gaze_y",
            "gaze_vx",
            "gaze_vy",
            "gaze_speed",
            "dist_error_px",
            "dist_error_deg",
            "is_on_target",
            "eye_state",
            "pursuit_gain",
            "directional_cosine",
            "cross_correlation",
            "optimal_lag_ms",
            "game_mode"
        ]
        self.csv_writer.writerow(headers)
        self.file_handle.flush()

        self.session_start_time = time.perf_counter()
        self.is_logging = True
        self.sample_count = 0
        print(f"[SessionLogger] Started recording session -> {self.current_filepath}")
        return self.current_filepath

    def log_sample(self, sample: ValidationSample, game_mode: str):
        """Writes a single frame's telemetry to CSV."""
        if not self.is_logging or self.csv_writer is None:
            return

        elapsed = sample.timestamp - self.session_start_time
        row = [
            f"{sample.timestamp:.4f}",
            f"{elapsed:.4f}",
            f"{sample.target_pos[0]:.2f}",
            f"{sample.target_pos[1]:.2f}",
            f"{sample.target_vel[0]:.2f}",
            f"{sample.target_vel[1]:.2f}",
            f"{sample.target_speed:.2f}",
            f"{sample.gaze_pos[0]:.2f}",
            f"{sample.gaze_pos[1]:.2f}",
            f"{sample.gaze_vel[0]:.2f}",
            f"{sample.gaze_vel[1]:.2f}",
            f"{sample.gaze_speed:.2f}",
            f"{sample.distance_error_px:.2f}",
            f"{sample.distance_error_deg:.3f}",
            1 if sample.is_on_target else 0,
            sample.eye_state,
            f"{sample.pursuit_gain:.3f}",
            f"{sample.directional_cosine:.3f}",
            f"{sample.cross_correlation:.3f}",
            f"{sample.optimal_lag_ms:.1f}",
            game_mode
        ]
        self.csv_writer.writerow(row)
        self.sample_count += 1

        # Periodically flush every 60 samples (~1 sec)
        if self.sample_count % 60 == 0:
            self.file_handle.flush()

    def close(self):
        """Flushes and closes the active log file."""
        if self.file_handle:
            self.file_handle.flush()
            self.file_handle.close()
            self.file_handle = None
            self.csv_writer = None
            self.is_logging = False
            print(f"[SessionLogger] Session closed. Total samples recorded: {self.sample_count}")
