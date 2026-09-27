# Eye Gaze Object Contingency Validation Report

**Session ID:** `benchmark_gaze_session_20260927_092336`  
**Data Source:** `gaze_session_20260927_092336.csv`  
**Generated At:** `2026-09-27 09:23:40`  

---

## 1. Executive Summary & Validation Metrics

| Metric | Measured Value | Standard Reference Range | Status |
| :--- | :--- | :--- | :--- |
| **On-Target Tracking Time** | **9.3%** | > 75.0% | ⚠️ Review Needed |
| **Mean Spatial Error** | **217.1 px (5.47°)** | < 60 px (< 2.0°) | ⚠️ Moderate |
| **Root Mean Square Error (RMSE)** | **295.8 px (7.44°)** | < 80 px | ⚠️ Warning |
| **Smooth Pursuit Gain ($V_g / V_t$)** | **1.81** | 0.85 – 1.10 | ⚠️ Abnormal |
| **Velocity Cross-Correlation ($r_{max}$)** | **0.15** | > 0.65 | ⚠️ Low Correlation |
| **Physiological Visual Latency** | **87 ms** | 80 – 200 ms | ✅ Within human latency window |
| **Catch-up Saccade Rate** | **0.67 saccades/s** | 0.5 – 2.5 /s | ✅ Expected frequency |

---

## 2. Ocular Movement Breakdown

- **Smooth Pursuit:** 1.7% of session
- **Saccades (Catch-up / Exploration):** 2.5% of session
- **Fixation:** 52.5% of session
- **Blinks Detected:** 22.2% of session
- **Total Duration:** 12.0 seconds (720 frames recorded at 60.1 Hz)

---

## 3. Generated Diagnostic Visualizations

1. **2D Trajectory Path Overlay:** `![2D Trajectory](benchmark_gaze_session_20260927_092336_trajectory_2d.png)`
2. **Synchronized Time Series ($X(t), Y(t)$):** `![Time Series](benchmark_gaze_session_20260927_092336_time_series.png)`
3. **Velocity Matching & Lag Correlation:** `![Velocity and Lag](benchmark_gaze_session_20260927_092336_velocity_lag.png)`
4. **Analytical Dashboard:** `![Dashboard](benchmark_gaze_session_20260927_092336_dashboard.png)`
