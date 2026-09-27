# Dynamic Eye Gaze Tracker & Game Target Validation Suite

A real-time **Eye Gaze Tracking & Dynamic Motion Validation System** built with **MediaPipe Iris Mesh**, **OpenCV**, and **Pygame**.

This solution tracks eyeball gaze in real time (using physical webcam or high-fidelity physiological simulation), pairs it against dynamic moving targets across multiple stimulus paradigms (Lissajous smooth pursuit, circular pursuit, bouncing ball physics, and saccadic step-jumps), computes scientific validation metrics (spatial error, pursuit gain, velocity cross-correlation, and saccade/pursuit I-VT classification), and logs time-series telemetry to CSV with automated publication-ready analytical reports.

---

## Architecture Overview

```
                                  +----------------------------+
                                  |   Webcam / Video Stream    |
                                  +--------------+-------------+
                                                 |
                                                 v
                                  +----------------------------+
                                  |   MediaPipe FaceLandmarker |
                                  |  (478 landmarks + Irises)  |
                                  +--------------+-------------+
                                                 |
                                                 v
       +-----------------------+  +----------------------------+
       | 9-Point Interactive   |->| Regularized Polynomial     |
       | Calibration Profile   |  | Regression & Bias Mapper   |
       +-----------------------+  +--------------+-------------+
                                                 |
                                                 v
                                  +----------------------------+
                                  |  1-Euro Adaptive Filter    |
                                  | (Jitter & Latency Removal) |
                                  +--------------+-------------+
                                                 |
                                                 v
+-------------------------------+ +----------------------------+
| Dynamic Game Stimulus Engine  | | Calibrated Eye Gaze (X, Y) |
| (Lissajous, Circle, Bounce)   | | & Real-time Gaze Reticle   |
+---------------+---------------+ +--------------+-------------+
                |                                |
                +---------------+----------------+
                                |
                                v
               +----------------------------------+
               | Scientific Validation Engine     |
               | - Euclidean Distance (px & deg)  |
               | - Smooth Pursuit Gain (Vg / Vt)  |
               | - I-VT Velocity Classification   |
               | - Lag-Compensated Cross-Corr     |
               | - 60 Hz CSV Telemetry Logger     |
               +----------------+-----------------+
                                |
                                v
               +----------------------------------+
               | Automated Analysis & Visualizer  |
               | - 2D Trajectory Path Overlay     |
               | - Synchronized X(t), Y(t) Traces |
               | - Velocity & Latency Curve       |
               | - Diagnostic Report Dashboard    |
               +----------------------------------+
```

---

## Key Features

1. **Iris & Gaze Feature Extraction**:
   - Uses MediaPipe 1.0+ Task API with 478 3D landmarks.
   - Extracts pupil/iris centers (landmarks 468 & 473) relative to the medial and lateral canthus.
   - Compensates for head pose (yaw/pitch) and measures eye blinks via Eyelid Aspect Ratio (EAR) and blendshapes.
2. **Interactive 9-Point Calibration**:
   - Collects fixational iris samples across a 9-point grid.
   - Solves Ridge-regularized 2D polynomial regression to map raw facial/iris coordinates to screen space.
   - Automatically saves and loads `calibration_profile.json`.
3. **Adaptive 1-Euro Filtering**:
   - Dynamically shifts cutoff frequency: low cutoff during steady fixations to eliminate webcam landmark jitter; high cutoff during rapid pursuit and saccades to avoid latency.
4. **Dynamic Game Motion Paradigms**:
   - **Mode 1 (Lissajous)**: Harmonic sinusoidal smooth pursuit test.
   - **Mode 2 (Circular)**: Orbiting pursuit target with adjustable orbital frequency.
   - **Mode 3 (Bouncing Ball)**: Realistic 2D kinematic bouncing physics.
   - **Mode 4 (Saccadic Step-Jump)**: Sudden position jumps with fixation periods to test saccadic latency and landing accuracy.
5. **Scientific Eye Movement Validation**:
   - **Spatial Error**: Real-time Euclidean distance in pixels and visual angle (degrees).
   - **Velocity Matching**: Instantaneous velocity vectors $\vec{V}_{target}$ and $\vec{V}_{gaze}$.
   - **Smooth Pursuit Gain**: Ratio $\|\vec{V}_{gaze}\| / \|\vec{V}_{target}\|$ (ideal human gain $\approx 0.85 - 1.10$).
   - **I-VT Classifier**: Classifies eye states in real time: `SMOOTH_PURSUIT`, `SACCADE`, `FIXATION`, `BLINK`, `DRIFT`.
   - **Physiological Lag Estimation**: Computes sliding-window cross-correlation $r(\tau)$ for lags $\tau \in [0, 300\text{ ms}]$.
6. **Telemetry & Scientific Reports**:
   - Logs continuous 60 Hz CSV data to `logs/`.
   - Generates publication-ready figures (`trajectory_2d.png`, `time_series.png`, `velocity_lag.png`, `dashboard.png`) and Markdown reports in `reports/`.

---

## Project Structure

```
EyeGazeTracker/
├── main.py                  # Primary application entry point & Pygame loop
├── test_suite.py            # Automated benchmark & validation test runner
├── models/
│   └── face_landmarker.task # MediaPipe offline landmark model bundle
├── gaze_engine/
│   ├── tracker.py           # Webcam iris detection & simulation fallback
│   ├── calibrator.py        # 9-point polynomial calibration routine
│   ├── filters.py           # 1-Euro adaptive low-pass filter
│   └── calibration_profile.json
├── game/
│   ├── target.py            # Dynamic target kinematics & trajectory modes
│   └── hud.py               # Telemetry HUD, reticles, and webcam PIP
├── validation/
│   ├── metrics.py           # Scientific metrics & I-VT state classifier
│   ├── logger.py            # 60 Hz CSV telemetry recorder
│   └── analyzer.py          # Matplotlib chart generator & markdown reporter
├── logs/                    # Session CSV logs
└── reports/                 # Generated diagnostic plots and summary reports
```

---

## Installation & Setup

All prerequisites are already installed in your Python environment:
```bash
python -m pip install opencv-python pygame numpy matplotlib scipy pandas mediapipe
```

---

## How to Run

### 1. Launch Interactive Game with Webcam
```bash
python main.py
```
*If no physical webcam is found, it automatically activates the high-fidelity eye gaze simulator.*

### 2. Force Simulation Mode (Testing without camera)
```bash
python main.py --simulate
```

### 3. Launch Calibration on Startup
```bash
python main.py --calibrate
```

### 4. Run Automated Benchmark & Generate Sample Reports
```bash
python test_suite.py
```

### 5. Re-Analyze Any Saved Session CSV
```bash
python main.py --analyze logs/gaze_session_20260927_092631.csv
```

---

## In-Game Keyboard Controls

| Key | Action |
| :--- | :--- |
| `[SPACE]` | Pause / Resume target movement and tracking |
| `[1]` | Switch to **Lissajous Smooth Pursuit** mode |
| `[2]` | Switch to **Circular Pursuit** mode |
| `[3]` | Switch to **Bouncing Ball Kinematics** mode |
| `[4]` | Switch to **Saccadic Step-Jump** mode |
| `[C]` | Launch **9-Point Interactive Calibration** |
| `[S]` | Toggle **Physical Webcam vs Synthetic Simulation** |
| `[R]` | Reset session & start fresh CSV log |
| `[P]` | Generate & save scientific report & plots immediately |
| `[TAB]` | Toggle Webcam Picture-in-Picture (PIP) preview |
| `[ESC]` / `[Q]` | Exit and automatically compile session report |
