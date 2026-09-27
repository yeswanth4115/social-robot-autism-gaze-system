"""
Dynamic Eye Gaze Tracker & Game Target Validation Suite.
Validates dynamic eye gaze movements against animated game objects,
logs high-frequency telemetry to CSV, and generates scientific analysis reports.
"""

import os
import sys
import time
import argparse
import pygame
import numpy as np

from gaze_engine.tracker import GazeTracker
from gaze_engine.calibrator import GazeCalibrator
from gaze_engine.filters import GazePointFilter2D
from game.target import DynamicTarget
from game.hud import GameHUD
from validation.metrics import GazeObjectValidator, ValidationSample
from validation.logger import SessionLogger
from validation.analyzer import SessionAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Eye Gaze Tracker & Game Object Validation Suite")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--simulate", action="store_true", help="Force synthetic eye simulation mode")
    parser.add_argument("--calibrate", action="store_true", help="Launch calibration immediately on startup")
    parser.add_argument("--mode", type=str, default="lissajous", choices=DynamicTarget.MODES, help="Initial motion mode")
    parser.add_argument("--analyze", type=str, default=None, help="Run analysis & plotting on an existing CSV log and exit")
    parser.add_argument("--width", type=int, default=1280, help="Window width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Window height (default: 720)")
    args = parser.parse_args()

    # Standalone Analysis Mode
    if args.analyze:
        print(f"[Main] Running standalone analysis on: {args.analyze}")
        analyzer = SessionAnalyzer(args.analyze)
        stats = analyzer.compute_summary_statistics()
        plots = analyzer.generate_all_plots()
        print("\n=== SESSION SUMMARY METRICS ===")
        for k, v in stats.items():
            print(f"  {k:30s}: {v}")
        print("\n=== GENERATED PLOTS ===")
        for k, path in plots.items():
            print(f"  {k:20s}: {path}")
        return

    # Initialize Pygame
    pygame.init()
    pygame.display.set_caption("Dynamic Eye Gaze Tracker & Game Validation Suite")
    screen = pygame.display.set_mode((args.width, args.height))
    clock = pygame.time.Clock()

    # Instantiate Subsystems
    tracker = GazeTracker(
        camera_id=args.camera_id,
        force_simulation=args.simulate,
        screen_width=args.width,
        screen_height=args.height
    )
    calibrator = GazeCalibrator(screen_width=args.width, screen_height=args.height)
    gaze_filter = GazePointFilter2D(freq=60.0, mincutoff=1.2, beta=0.08)
    target = DynamicTarget(screen_width=args.width, screen_height=args.height, radius=38.0)
    target.set_mode(args.mode)
    validator = GazeObjectValidator(target_radius=42.0)
    hud = GameHUD(screen_width=args.width, screen_height=args.height)
    logger = SessionLogger()

    # Start CSV session
    log_file = logger.start_session(game_mode=target.mode)

    # State variables
    running = True
    is_paused = False
    is_calibrating = args.calibrate
    if is_calibrating:
        calibrator.start_calibration()

    session_start_time = time.perf_counter()
    last_sample: Optional[ValidationSample] = None

    print("\n========================================================")
    print("  DYNAMIC EYE GAZE TRACKER & GAME VALIDATION SUITE")
    print("========================================================")
    print("  Controls:")
    print("    [SPACE]   Pause / Resume target movement")
    print("    [1-4]     Switch Trajectory Modes (1: Lissajous, 2: Circular, 3: Bounce, 4: Saccadic Step)")
    print("    [C]       Start 9-Point Eye Calibration")
    print("    [S]       Toggle Physical Webcam vs Eye Simulation")
    print("    [R]       Reset Telemetry & Start New CSV Log")
    print("    [P]       Generate & Save Scientific Analysis Report")
    print("    [TAB]     Toggle Webcam PIP Display")
    print("    [ESC/Q]   Exit & Generate Comprehensive Report")
    print("========================================================\n")

    while running:
        dt = clock.tick(60) / 1000.0
        now = time.perf_counter()
        fps = clock.get_fps()

        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    running = False
                elif event.key == pygame.K_SPACE:
                    is_paused = not is_paused
                elif event.key == pygame.K_1:
                    target.set_mode("lissajous")
                elif event.key == pygame.K_2:
                    target.set_mode("circular")
                elif event.key == pygame.K_3:
                    target.set_mode("bouncing")
                elif event.key == pygame.K_4:
                    target.set_mode("saccadic_step")
                elif event.key == pygame.K_c:
                    is_calibrating = True
                    calibrator.start_calibration()
                elif event.key == pygame.K_s:
                    tracker.force_simulation = not tracker.force_simulation
                    tracker.is_simulated = tracker.force_simulation
                    print(f"[Main] Simulation toggled: {tracker.is_simulated}")
                elif event.key == pygame.K_r:
                    logger.close()
                    validator.reset()
                    log_file = logger.start_session(game_mode=target.mode)
                    session_start_time = time.perf_counter()
                elif event.key == pygame.K_p:
                    # Quick plot generation on demand
                    if logger.sample_count > 30 and logger.current_filepath:
                        logger.file_handle.flush()
                        analyzer = SessionAnalyzer(logger.current_filepath)
                        analyzer.generate_all_plots()
                elif event.key == pygame.K_TAB:
                    hud.show_pip = not hud.show_pip

        # 2. Update Game Target
        if not is_paused and not is_calibrating:
            target_x, target_y = target.update()
        else:
            target_x, target_y = target.x, target.y

        # 3. Update Eye Gaze Tracker
        gaze_frame = tracker.update(target_pos=(target_x, target_y) if tracker.is_simulated else None)

        # 4. Handle Calibration vs Normal Tracking
        if is_calibrating:
            calib_target = calibrator.get_current_calibration_target()
            if calib_target is not None:
                cx, cy, prog = calib_target
                calibrator.record_calibration_sample(gaze_frame.features, gaze_frame.is_blink)
                # Filtered gaze for display
                pred_x, pred_y = calibrator.predict(gaze_frame.features)
                filt_x, filt_y = gaze_filter.filter(pred_x, pred_y, now)
            else:
                is_calibrating = False
                print("[Main] Calibration sequence complete. Returning to game tracking.")
                filt_x, filt_y = target_x, target_y
        else:
            # Map raw eye features to screen coordinates
            raw_px, raw_py = calibrator.predict(gaze_frame.features)
            # Low-latency adaptive One-Euro filtering
            filt_x, filt_y = gaze_filter.filter(raw_px, raw_py, now)

            # 5. Scientific Validation Computation
            sample = validator.compute_frame(
                timestamp=now,
                target_pos=(target_x, target_y),
                gaze_pos=(filt_x, filt_y),
                is_blink=gaze_frame.is_blink
            )
            last_sample = sample

            # 6. Log Telemetry to CSV
            if not is_paused:
                logger.log_sample(sample, target.mode)

        # 7. Render Game Visuals
        screen.fill((12, 14, 22))  # Deep slate tech background

        # Subtle background grid
        grid_spacing = 60
        for gx in range(0, args.width, grid_spacing):
            pygame.draw.line(screen, (22, 26, 38), (gx, 0), (gx, args.height), 1)
        for gy in range(0, args.height, grid_spacing):
            pygame.draw.line(screen, (22, 26, 38), (0, gy), (args.width, gy), 1)

        # Render Dynamic Target
        is_tracked = last_sample.is_on_target if last_sample else False
        target.draw(screen, is_tracked=is_tracked)

        # Render Calibrated Gaze Cursor & Error Vector
        if last_sample and not is_calibrating:
            hud.draw_gaze_cursor(
                screen,
                gaze_x=filt_x,
                gaze_y=filt_y,
                target_x=target_x,
                target_y=target_y,
                eye_state=last_sample.eye_state,
                is_on_target=last_sample.is_on_target
            )

        # Render Telemetry HUD
        if last_sample and not is_calibrating:
            elapsed_session = now - session_start_time
            hud.draw_telemetry_panel(
                screen,
                sample=last_sample,
                mode_name=target.mode,
                source=gaze_frame.source,
                is_calibrated=calibrator.is_calibrated,
                fps=fps,
                elapsed_s=elapsed_session
            )

        # Render Webcam PIP
        hud.draw_webcam_pip(screen, gaze_frame.annotated_frame)

        # Render Bottom Help Bar
        hud.draw_bottom_controls(screen, is_paused=is_paused, is_logging=logger.is_logging)

        # Render Calibration UI if active
        if is_calibrating and calib_target is not None:
            cx, cy, prog = calib_target
            hud.draw_calibration_overlay(
                screen,
                calib_x=cx,
                calib_y=cy,
                progress=prog,
                point_idx=calibrator.current_point_idx,
                total_points=len(calibrator.calib_points)
            )

        pygame.display.flip()

    # Cleanup & Post-Session Analysis Generation
    print("\n[Main] Shutting down tracker and finalizing telemetry log...")
    saved_csv = logger.current_filepath
    logger.close()
    tracker.release()
    pygame.quit()

    if saved_csv and os.path.exists(saved_csv) and logger.sample_count > 30:
        print("\n========================================================")
        print("  GENERATING SCIENTIFIC VALIDATION REPORT & PLOTS")
        print("========================================================")
        analyzer = SessionAnalyzer(saved_csv)
        stats = analyzer.compute_summary_statistics()
        plots = analyzer.generate_all_plots()

        print("\n=== VALIDATION RESULTS ===")
        print(f"  Duration:              {stats.get('duration_sec', 0):.1f} s")
        print(f"  On-Target Accuracy:    {stats.get('on_target_pct', 0):.1f} %")
        print(f"  Mean Distance Error:   {stats.get('mean_error_px', 0):.1f} px ({stats.get('mean_error_deg', 0):.2f}°)")
        print(f"  RMSE Error:            {stats.get('rmse_error_px', 0):.1f} px")
        print(f"  Smooth Pursuit Gain:   {stats.get('mean_pursuit_gain', 0):.2f}")
        print(f"  Velocity Correlation:  {stats.get('mean_velocity_correlation', 0):.2f}")
        print(f"  Physiological Latency: {stats.get('mean_physiological_lag_ms', 0):.0f} ms")
        print(f"  Smooth Pursuit %:      {stats.get('pursuit_pct', 0):.1f} %")
        print(f"  Catch-up Saccades:     {stats.get('total_saccades', 0)}")

        print("\n=== GENERATED PLOTS & REPORT ===")
        for key, p in plots.items():
            print(f"  {key:20s}: {p}")
        print("========================================================\n")


if __name__ == "__main__":
    main()
