# Eye Gaze Object Contingency Validation Report

**Session ID:** `gaze_session_20260927_094831`  
**Data Source:** `gaze_session_20260927_094831.csv`  
**Generated At:** `2026-09-27 09:49:02`  

---

## 1. Executive Summary & Validation Metrics

| Metric | Measured Value | Standard Reference Range | Status |
| :--- | :--- | :--- | :--- |
| **On-Target Tracking Time** | **0.0%** | > 75.0% | ⚠️ Review Needed |
| **Mean Spatial Error** | **343.7 px (8.66°)** | < 60 px (< 2.0°) | ⚠️ Moderate |
| **Root Mean Square Error (RMSE)** | **365.7 px (9.21°)** | < 80 px | ⚠️ Warning |
| **Smooth Pursuit Gain ($V_g / V_t$)** | **0.33** | 0.85 – 1.10 | ⚠️ Abnormal |
| **Velocity Cross-Correlation ($r_{max}$)** | **-0.00** | > 0.65 | ⚠️ Low Correlation |
| **Physiological Visual Latency** | **114 ms** | 80 – 200 ms | ✅ Within human latency window |
| **Catch-up Saccade Rate** | **1.75 saccades/s** | 0.5 – 2.5 /s | ✅ Expected frequency |

---

## 2. Ocular Movement Breakdown

- **Smooth Pursuit:** 34.6% of session
- **Saccades (Catch-up / Exploration):** 7.9% of session
- **Fixation:** 0.1% of session
- **Blinks Detected:** 0.1% of session
- **Total Duration:** 25.1 seconds (745 frames recorded at 29.7 Hz)

---

## 3. Generated Diagnostic Visualizations

1. **2D Trajectory Path Overlay:** `![2D Trajectory](gaze_session_20260927_094831_trajectory_2d.png)`
2. **Synchronized Time Series ($X(t), Y(t)$):** `![Time Series](gaze_session_20260927_094831_time_series.png)`
3. **Velocity Matching & Lag Correlation:** `![Velocity and Lag](gaze_session_20260927_094831_velocity_lag.png)`
4. **Analytical Dashboard:** `![Dashboard](gaze_session_20260927_094831_dashboard.png)`
