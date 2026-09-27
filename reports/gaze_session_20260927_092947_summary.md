# Eye Gaze Object Contingency Validation Report

**Session ID:** `gaze_session_20260927_092947`  
**Data Source:** `gaze_session_20260927_092947.csv`  
**Generated At:** `2026-09-27 09:30:54`  

---

## 1. Executive Summary & Validation Metrics

| Metric | Measured Value | Standard Reference Range | Status |
| :--- | :--- | :--- | :--- |
| **On-Target Tracking Time** | **3.3%** | > 75.0% | ⚠️ Review Needed |
| **Mean Spatial Error** | **132.7 px (3.35°)** | < 60 px (< 2.0°) | ⚠️ Moderate |
| **Root Mean Square Error (RMSE)** | **139.1 px (3.51°)** | < 80 px | ⚠️ Warning |
| **Smooth Pursuit Gain ($V_g / V_t$)** | **0.74** | 0.85 – 1.10 | ⚠️ Abnormal |
| **Velocity Cross-Correlation ($r_{max}$)** | **0.76** | > 0.65 | ✅ Strong Co-movement |
| **Physiological Visual Latency** | **150 ms** | 80 – 200 ms | ✅ Within human latency window |
| **Catch-up Saccade Rate** | **6.66 saccades/s** | 0.5 – 2.5 /s | ✅ Expected frequency |

---

## 2. Ocular Movement Breakdown

- **Smooth Pursuit:** 18.6% of session
- **Saccades (Catch-up / Exploration):** 72.6% of session
- **Fixation:** 0.0% of session
- **Blinks Detected:** 3.4% of session
- **Total Duration:** 65.4 seconds (3944 frames recorded at 60.3 Hz)

---

## 3. Generated Diagnostic Visualizations

1. **2D Trajectory Path Overlay:** `![2D Trajectory](gaze_session_20260927_092947_trajectory_2d.png)`
2. **Synchronized Time Series ($X(t), Y(t)$):** `![Time Series](gaze_session_20260927_092947_time_series.png)`
3. **Velocity Matching & Lag Correlation:** `![Velocity and Lag](gaze_session_20260927_092947_velocity_lag.png)`
4. **Analytical Dashboard:** `![Dashboard](gaze_session_20260927_092947_dashboard.png)`
