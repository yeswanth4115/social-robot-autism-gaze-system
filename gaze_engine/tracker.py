"""
Webcam & MediaPipe-based Eye Gaze Tracker with Iris Detection & Synthetic Fallback.
Extracts high-precision eye landmarks, iris centers, blendshapes, and head-pose proxies.
"""

import os
import sys
import time
import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import cv2

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


@dataclass
class GazeFrame:
    """Container for eye tracking results per frame."""
    timestamp: float
    raw_u: float                 # Normalized horizontal gaze feature (-1.0 to 1.0)
    raw_v: float                 # Normalized vertical gaze feature (-1.0 to 1.0)
    features: np.ndarray         # Feature vector for polynomial calibration
    is_blink: bool               # Blink flag
    face_detected: bool          # Face detection flag
    confidence: float            # Tracking confidence [0, 1]
    left_iris: Optional[Tuple[float, float]] = None
    right_iris: Optional[Tuple[float, float]] = None
    head_pose: Optional[Tuple[float, float]] = None  # Yaw, Pitch proxies
    annotated_frame: Optional[np.ndarray] = None    # BGR image for HUD PIP preview
    source: str = "webcam"       # "webcam" or "simulated"


class GazeTracker:
    """
    Real-time Eye Tracker using MediaPipe FaceLandmarker with iris detection.
    Falls back gracefully to simulated eye kinematics if camera is unavailable.
    """
    # Key Landmark Indices in 478-point Face Mesh
    LEFT_IRIS_CENTER = 468
    LEFT_IRIS_CONTOUR = [469, 470, 471, 472]
    RIGHT_IRIS_CENTER = 473
    RIGHT_IRIS_CONTOUR = [474, 475, 476, 477]

    # Eye Corner & Eyelid Landmarks
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145

    RIGHT_EYE_OUTER = 263
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374

    # Head Pose Anchors
    NOSE_TIP = 1
    CHIN = 152
    FOREHEAD = 10
    LEFT_TEMPLE = 234
    RIGHT_TEMPLE = 454

    def __init__(
        self,
        camera_id: int = 0,
        model_path: Optional[str] = None,
        force_simulation: bool = False,
        screen_width: int = 1280,
        screen_height: int = 720
    ):
        self.camera_id = camera_id
        self.force_simulation = force_simulation
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.is_simulated = force_simulation

        # Resolve model path
        if model_path is None:
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(os.path.dirname(cur_dir), "models", "face_landmarker.task")
        self.model_path = model_path

        self.cap = None
        self.landmarker = None

        # Simulation state
        self._sim_gaze_x = screen_width / 2
        self._sim_gaze_y = screen_height / 2
        self._sim_target_history = []  # For lag simulation
        self._sim_last_blink_time = time.perf_counter()
        self._sim_is_blinking = False
        self._sim_blink_duration = 0.15

        if not self.force_simulation:
            self._init_camera_and_landmarker()

        if self.landmarker is None or self.cap is None:
            print("[GazeTracker] Operating in Simulated Eye Movement mode.")
            self.is_simulated = True

    def _init_camera_and_landmarker(self):
        """Attempts to open physical camera and build MediaPipe detector."""
        if not MEDIAPIPE_AVAILABLE or not os.path.exists(self.model_path):
            print(f"[GazeTracker] MediaPipe model not found at {self.model_path}")
            return

        # Attempt opening camera with different backends
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        opened = False
        for backend in backends:
            try:
                cap = cv2.VideoCapture(self.camera_id, backend)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        self.cap = cap
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        opened = True
                        print(f"[GazeTracker] Camera {self.camera_id} opened successfully with backend {backend}.")
                        break
                    cap.release()
            except Exception as e:
                continue

        if not opened:
            print("[GazeTracker] No physical webcam accessible. Falling back to high-fidelity eye simulation.")
            return

        # Initialize MediaPipe FaceLandmarker
        try:
            base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                output_face_blendshapes=True,
                num_faces=1
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
            print("[GazeTracker] MediaPipe FaceLandmarker initialized successfully.")
        except Exception as e:
            print(f"[GazeTracker] Failed to initialize FaceLandmarker: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None

    def _compute_ear(self, landmarks, top_idx: int, bot_idx: int, inner_idx: int, outer_idx: int, w: int, h: int) -> float:
        """Eye Aspect Ratio (EAR) for blink detection."""
        top = np.array([landmarks[top_idx].x * w, landmarks[top_idx].y * h])
        bot = np.array([landmarks[bot_idx].x * w, landmarks[bot_idx].y * h])
        inner = np.array([landmarks[inner_idx].x * w, landmarks[inner_idx].y * h])
        outer = np.array([landmarks[outer_idx].x * w, landmarks[outer_idx].y * h])
        v_dist = np.linalg.norm(top - bot)
        h_dist = np.linalg.norm(inner - outer)
        return float(v_dist / (h_dist + 1e-6))

    def _extract_iris_features(self, landmarks, blendshapes, img_w: int, img_h: int) -> Tuple[float, float, np.ndarray, bool]:
        """Calculates normalized iris gaze vector and multi-feature array."""
        # 1. Left Eye Geometry (Camera's left = user's right eye, landmarks follow facial anatomy)
        # Left eye: outer 33, inner 133, top 159, bottom 145, iris 468
        lx_in = landmarks[self.LEFT_EYE_INNER].x
        lx_out = landmarks[self.LEFT_EYE_OUTER].x
        ly_top = landmarks[self.LEFT_EYE_TOP].y
        ly_bot = landmarks[self.LEFT_EYE_BOTTOM].y
        l_iris_x = landmarks[self.LEFT_IRIS_CENTER].x
        l_iris_y = landmarks[self.LEFT_IRIS_CENTER].y

        # Normalize iris position within eye bounding box [-1.0, 1.0]
        # Inner corner is towards nose, outer is towards temple
        h_left = (l_iris_x - (lx_in + lx_out) / 2.0) / (abs(lx_out - lx_in) / 2.0 + 1e-6)
        v_left = (l_iris_y - (ly_top + ly_bot) / 2.0) / (abs(ly_bot - ly_top) / 2.0 + 1e-6)

        # 2. Right Eye Geometry
        # Right eye: outer 263, inner 362, top 386, bottom 374, iris 473
        rx_in = landmarks[self.RIGHT_EYE_INNER].x
        rx_out = landmarks[self.RIGHT_EYE_OUTER].x
        ry_top = landmarks[self.RIGHT_EYE_TOP].y
        ry_bot = landmarks[self.RIGHT_EYE_BOTTOM].y
        r_iris_x = landmarks[self.RIGHT_IRIS_CENTER].x
        r_iris_y = landmarks[self.RIGHT_IRIS_CENTER].y

        h_right = (r_iris_x - (rx_in + rx_out) / 2.0) / (abs(rx_out - rx_in) / 2.0 + 1e-6)
        v_right = (r_iris_y - (ry_top + ry_bot) / 2.0) / (abs(ry_bot - ry_top) / 2.0 + 1e-6)

        # Average normalized iris coordinates
        u_raw = float((h_left + h_right) / 2.0)
        v_raw = float((v_left + v_right) / 2.0)

        # 3. Head Pose Proxy (nose position relative to eye midpoints and temples)
        nose_x = landmarks[self.NOSE_TIP].x
        nose_y = landmarks[self.NOSE_TIP].y
        mid_eyes_x = (lx_in + rx_in) / 2.0
        mid_eyes_y = (ly_top + ry_top) / 2.0
        face_width = abs(landmarks[self.RIGHT_TEMPLE].x - landmarks[self.LEFT_TEMPLE].x) + 1e-6
        face_height = abs(landmarks[self.CHIN].y - landmarks[self.FOREHEAD].y) + 1e-6
        yaw_proxy = float((nose_x - mid_eyes_x) / face_width)
        pitch_proxy = float((nose_y - mid_eyes_y) / face_height)

        # 4. Blink Detection using EAR and Blendshapes
        ear_left = self._compute_ear(landmarks, self.LEFT_EYE_TOP, self.LEFT_EYE_BOTTOM, self.LEFT_EYE_INNER, self.LEFT_EYE_OUTER, img_w, img_h)
        ear_right = self._compute_ear(landmarks, self.RIGHT_EYE_TOP, self.RIGHT_EYE_BOTTOM, self.RIGHT_EYE_INNER, self.RIGHT_EYE_OUTER, img_w, img_h)
        avg_ear = (ear_left + ear_right) / 2.0

        is_blink = avg_ear < 0.16

        # Blendshapes contribution if available
        blend_h = 0.0
        blend_v = 0.0
        if blendshapes:
            shape_dict = {b.category_name: b.score for b in blendshapes}
            blink_l = shape_dict.get("eyeBlinkLeft", 0.0)
            blink_r = shape_dict.get("eyeBlinkRight", 0.0)
            if (blink_l + blink_r) / 2.0 > 0.45:
                is_blink = True

            # Blendshape eye gaze vectors
            # eyeLookOut - eyeLookIn
            look_out_l = shape_dict.get("eyeLookOutLeft", 0.0)
            look_in_l = shape_dict.get("eyeLookInLeft", 0.0)
            look_out_r = shape_dict.get("eyeLookOutRight", 0.0)
            look_in_r = shape_dict.get("eyeLookInRight", 0.0)
            look_up_l = shape_dict.get("eyeLookUpLeft", 0.0)
            look_down_l = shape_dict.get("eyeLookDownLeft", 0.0)
            look_up_r = shape_dict.get("eyeLookUpRight", 0.0)
            look_down_r = shape_dict.get("eyeLookDownRight", 0.0)

            # Invert horizontal so left is negative, right is positive
            blend_h = float((look_out_l - look_in_l + look_in_r - look_out_r) / 2.0)
            blend_v = float(((look_up_l + look_up_r) / 2.0) - ((look_down_l + look_down_r) / 2.0))

        # Composite feature vector for polynomial calibration regression:
        # [u, v, u^2, v^2, u*v, yaw, pitch, blend_h, blend_v]
        features = np.array([
            u_raw,
            v_raw,
            u_raw ** 2,
            v_raw ** 2,
            u_raw * v_raw,
            yaw_proxy,
            pitch_proxy,
            blend_h,
            blend_v
        ], dtype=np.float32)

        return u_raw, v_raw, features, is_blink

    def _draw_hud_overlay(self, frame: np.ndarray, landmarks, is_blink: bool) -> np.ndarray:
        """Renders eye contour, iris circles, and bounding mesh on preview frame."""
        h, w, _ = frame.shape
        annotated = frame.copy()

        # Draw left iris
        l_pt = (int(landmarks[self.LEFT_IRIS_CENTER].x * w), int(landmarks[self.LEFT_IRIS_CENTER].y * h))
        cv2.circle(annotated, l_pt, 4, (0, 255, 0), -1)
        for idx in self.LEFT_IRIS_CONTOUR:
            pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
            cv2.circle(annotated, pt, 2, (0, 200, 255), -1)

        # Draw right iris
        r_pt = (int(landmarks[self.RIGHT_IRIS_CENTER].x * w), int(landmarks[self.RIGHT_IRIS_CENTER].y * h))
        cv2.circle(annotated, r_pt, 4, (0, 255, 0), -1)
        for idx in self.RIGHT_IRIS_CONTOUR:
            pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
            cv2.circle(annotated, pt, 2, (0, 200, 255), -1)

        # Draw eye contours
        for idx in [self.LEFT_EYE_OUTER, self.LEFT_EYE_INNER, self.LEFT_EYE_TOP, self.LEFT_EYE_BOTTOM]:
            pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
            cv2.circle(annotated, pt, 2, (255, 100, 0), -1)
        for idx in [self.RIGHT_EYE_OUTER, self.RIGHT_EYE_INNER, self.RIGHT_EYE_TOP, self.RIGHT_EYE_BOTTOM]:
            pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
            cv2.circle(annotated, pt, 2, (255, 100, 0), -1)

        # Blink banner
        if is_blink:
            cv2.putText(annotated, "BLINK DETECTED", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(annotated, "TRACKING OK", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return annotated

    def update(self, target_pos: Optional[Tuple[float, float]] = None) -> GazeFrame:
        """
        Processes one tracking step.
        If a target_pos is provided (during simulation), simulates human smooth pursuit tracking
        with physiological lag, microsaccades, tremor noise, and periodic blinks.
        """
        now = time.perf_counter()

        if self.is_simulated or self.cap is None or self.landmarker is None:
            return self._simulate_gaze_step(now, target_pos)

        # Read from physical webcam
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return self._simulate_gaze_step(now, target_pos)

        frame = cv2.flip(frame, 1)  # Mirror view
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            # Face lost: return blank frame
            return GazeFrame(
                timestamp=now,
                raw_u=0.0,
                raw_v=0.0,
                features=np.zeros(9, dtype=np.float32),
                is_blink=False,
                face_detected=False,
                confidence=0.0,
                annotated_frame=frame,
                source="webcam_no_face"
            )

        landmarks = result.face_landmarks[0]
        blendshapes = result.face_blendshapes[0] if result.face_blendshapes else None

        u_raw, v_raw, features, is_blink = self._extract_iris_features(landmarks, blendshapes, w, h)
        annotated = self._draw_hud_overlay(frame, landmarks, is_blink)

        l_iris = (landmarks[self.LEFT_IRIS_CENTER].x, landmarks[self.LEFT_IRIS_CENTER].y)
        r_iris = (landmarks[self.RIGHT_IRIS_CENTER].x, landmarks[self.RIGHT_IRIS_CENTER].y)

        return GazeFrame(
            timestamp=now,
            raw_u=u_raw,
            raw_v=v_raw,
            features=features,
            is_blink=is_blink,
            face_detected=True,
            confidence=0.95 if not is_blink else 0.2,
            left_iris=l_iris,
            right_iris=r_iris,
            annotated_frame=annotated,
            source="webcam"
        )

    def _simulate_gaze_step(self, now: float, target_pos: Optional[Tuple[float, float]]) -> GazeFrame:
        """
        High-fidelity physiological simulation of human eye gaze.
        Models:
        - Latency buffer (100 - 140ms pursuit lag)
        - Catch-up saccades when tracking error exceeds threshold (>60px)
        - Microsaccadic ocular tremor (Gaussian noise)
        - Periodic blinks (every 3.5 - 6s, lasting 120 - 180ms)
        """
        # Periodic blink simulation
        if not self._sim_is_blinking:
            if now - self._sim_last_blink_time > random.uniform(4.0, 7.0):
                self._sim_is_blinking = True
                self._sim_last_blink_time = now
                self._sim_blink_duration = random.uniform(0.12, 0.18)
        else:
            if now - self._sim_last_blink_time > self._sim_blink_duration:
                self._sim_is_blinking = False
                self._sim_last_blink_time = now

        # Target history queue for realistic pursuit lag (~120ms)
        if target_pos is not None:
            self._sim_target_history.append((now, target_pos[0], target_pos[1]))
            # Keep history within 500ms
            self._sim_target_history = [item for item in self._sim_target_history if now - item[0] < 0.5]

        # Determine lagged target position
        lag_time = 0.12  # 120 ms human pursuit latency
        effective_target = target_pos if target_pos else (self.screen_width / 2, self.screen_height / 2)
        for t, x, y in reversed(self._sim_target_history):
            if now - t >= lag_time:
                effective_target = (x, y)
                break

        # Move simulated gaze towards lagged target
        dx = effective_target[0] - self._sim_gaze_x
        dy = effective_target[1] - self._sim_gaze_y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist > 70.0:
            # Catch-up saccade: rapid ballistic jump toward target
            saccade_gain = 0.65
            self._sim_gaze_x += dx * saccade_gain
            self._sim_gaze_y += dy * saccade_gain
        else:
            # Smooth pursuit tracking: agile foveal pursuit with velocity tracking
            pursuit_gain = 0.42
            self._sim_gaze_x += dx * pursuit_gain
            self._sim_gaze_y += dy * pursuit_gain

        # Add physiological ocular jitter / microsaccades (standard deviation ~3.5px)
        noise_x = random.gauss(0, 3.0)
        noise_y = random.gauss(0, 3.0)
        noisy_x = self._sim_gaze_x + noise_x
        noisy_y = self._sim_gaze_y + noise_y

        # Normalize to [-1.0, 1.0] for calibration feature vector
        u_norm = (noisy_x / self.screen_width) * 2.0 - 1.0
        v_norm = (noisy_y / self.screen_height) * 2.0 - 1.0

        features = np.array([
            u_norm,
            v_norm,
            u_norm ** 2,
            v_norm ** 2,
            u_norm * v_norm,
            0.0,
            0.0,
            u_norm * 0.5,
            v_norm * 0.5
        ], dtype=np.float32)

        # Generate a synthetic preview thumbnail for HUD PIP
        pip_frame = np.zeros((160, 240, 3), dtype=np.uint8)
        # Draw stylized synthetic eyes
        cv2.rectangle(pip_frame, (10, 10), (230, 150), (40, 40, 45), -1)
        cv2.putText(pip_frame, "SIMULATED EYE GAZE", (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

        # Left & Right Eye Sockets
        cv2.ellipse(pip_frame, (75, 85), (35, 20), 0, 0, 360, (220, 220, 220), -1)
        cv2.ellipse(pip_frame, (165, 85), (35, 20), 0, 0, 360, (220, 220, 220), -1)

        if not self._sim_is_blinking:
            # Iris displacement based on u_norm, v_norm
            iris_dx = int(u_norm * 14)
            iris_dy = int(v_norm * 8)
            cv2.circle(pip_frame, (75 + iris_dx, 85 + iris_dy), 10, (180, 100, 30), -1)
            cv2.circle(pip_frame, (75 + iris_dx, 85 + iris_dy), 4, (10, 10, 10), -1)
            cv2.circle(pip_frame, (165 + iris_dx, 85 + iris_dy), 10, (180, 100, 30), -1)
            cv2.circle(pip_frame, (165 + iris_dx, 85 + iris_dy), 4, (10, 10, 10), -1)
        else:
            cv2.line(pip_frame, (40, 85), (110, 85), (80, 80, 80), 3)
            cv2.line(pip_frame, (130, 85), (200, 85), (80, 80, 80), 3)
            cv2.putText(pip_frame, "BLINK", (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        return GazeFrame(
            timestamp=now,
            raw_u=u_norm,
            raw_v=v_norm,
            features=features,
            is_blink=self._sim_is_blinking,
            face_detected=True,
            confidence=0.98 if not self._sim_is_blinking else 0.1,
            left_iris=(75 / 240, 85 / 160),
            right_iris=(165 / 240, 85 / 160),
            head_pose=(0.0, 0.0),
            annotated_frame=pip_frame,
            source="simulated"
        )

    def release(self):
        """Releases camera and model resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None
