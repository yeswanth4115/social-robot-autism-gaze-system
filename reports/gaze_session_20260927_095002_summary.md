# Eye Gaze Object Contingency Validation Report

**Session ID:** `gaze_session_20260927_095002`  
**Data Source:** `gaze_session_20260927_095002.csv`  
**Generated At:** `2026-09-27 09:50:39`  

---

## 1. Executive Summary & Validation Metrics

| Metric | Measured Value | Standard Reference Range | Status |
| :--- | :--- | :--- | :--- |
| **On-Target Tracking Time** | **2.5%** | > 75.0% | ⚠️ Review Needed |
| **Mean Spatial Error** | **152.9 px (3.86°)** | < 60 px (< 2.0°) | ⚠️ Moderate |
| **Root Mean Square Error (RMSE)** | **160.9 px (4.06°)** | < 80 px | ⚠️ Warning |
| **Smooth Pursuit Gain ($V_g / V_t$)** | **0.67** | 0.85 – 1.10 | ⚠️ Abnormal |
| **Velocity Cross-Correlation ($r_{max}$)** | **0.78** | > 0.65 | ✅ Strong Co-movement |
| **Physiological Visual Latency** | **160 ms** | 80 – 200 ms | ✅ Within human latency window |
| **Catch-up Saccade Rate** | **10.03 saccades/s** | 0.5 – 2.5 /s | ✅ Expected frequency |

---

## 2. Ocular Movement Breakdown

- **Smooth Pursuit:** 47.5% of session
- **Saccades (Catch-up / Exploration):** 42.6% of session
- **Fixation:** 0.0% of session
- **Blinks Detected:** 3.3% of session
- **Total Duration:** 33.4 seconds (2006 frames recorded at 60.1 Hz)

---

## 3. Generated Diagnostic Visualizations

1. **2D Trajectory Path Overlay:** `![2D Trajectory](gaze_session_20260927_095002_trajectory_2d.png)`
2. **Synchronized Time Series ($X(t), Y(t)$):** `![Time Series](gaze_session_20260927_095002_time_series.png)`
3. **Velocity Matching & Lag Correlation:** `![Velocity and Lag](gaze_session_20260927_095002_velocity_lag.png)`
4. **Analytical Dashboard:** `![Dashboard](gaze_session_20260927_095002_dashboard.png)`
