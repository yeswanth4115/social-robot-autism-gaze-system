# Eye Gaze Object Contingency Validation Report

**Session ID:** `smooth_pursuit_gaze_session_20260927_092442`  
**Data Source:** `gaze_session_20260927_092442.csv`  
**Generated At:** `2026-09-27 09:24:46`  

---

## 1. Executive Summary & Validation Metrics

| Metric | Measured Value | Standard Reference Range | Status |
| :--- | :--- | :--- | :--- |
| **On-Target Tracking Time** | **0.0%** | > 75.0% | ⚠️ Review Needed |
| **Mean Spatial Error** | **89.8 px (2.27°)** | < 60 px (< 2.0°) | ⚠️ Moderate |
| **Root Mean Square Error (RMSE)** | **103.2 px (2.61°)** | < 80 px | ⚠️ Warning |
| **Smooth Pursuit Gain ($V_g / V_t$)** | **1.70** | 0.85 – 1.10 | ⚠️ Abnormal |
| **Velocity Cross-Correlation ($r_{max}$)** | **0.20** | > 0.65 | ⚠️ Low Correlation |
| **Physiological Visual Latency** | **62 ms** | 80 – 200 ms | ✅ Within human latency window |
| **Catch-up Saccade Rate** | **0.00 saccades/s** | 0.5 – 2.5 /s | ✅ Expected frequency |

---

## 2. Ocular Movement Breakdown

- **Smooth Pursuit:** 31.2% of session
- **Saccades (Catch-up / Exploration):** 0.0% of session
- **Fixation:** 12.0% of session
- **Blinks Detected:** 34.8% of session
- **Total Duration:** 10.0 seconds (600 frames recorded at 60.1 Hz)

---

## 3. Generated Diagnostic Visualizations

1. **2D Trajectory Path Overlay:** `![2D Trajectory](smooth_pursuit_gaze_session_20260927_092442_trajectory_2d.png)`
2. **Synchronized Time Series ($X(t), Y(t)$):** `![Time Series](smooth_pursuit_gaze_session_20260927_092442_time_series.png)`
3. **Velocity Matching & Lag Correlation:** `![Velocity and Lag](smooth_pursuit_gaze_session_20260927_092442_velocity_lag.png)`
4. **Analytical Dashboard:** `![Dashboard](smooth_pursuit_gaze_session_20260927_092442_dashboard.png)`
