import csv
import json
import math
import time
from collections import deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
import tkinter as tk

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gaze_features import (
    FEATURE_VERSION,
    MIN_GAZE_CONFIDENCE,
    estimate_gaze_confidence,
    extract_eye_metrics,
    get_features,
)
from gaze_kalman import KalmanGazeFilter

root = tk.Tk()
root.withdraw()
SCREEN_WIDTH = root.winfo_screenwidth()
SCREEN_HEIGHT = root.winfo_screenheight()
root.destroy()

print("Screen:", SCREEN_WIDTH, "x", SCREEN_HEIGHT)

with open("gaze_model_metadata.json", "r") as f:
    metadata = json.load(f)

if metadata.get("feature_version") != FEATURE_VERSION:
    print(
        "ERROR: Model feature version mismatch. Model expects"
        f" '{metadata.get('feature_version')}', but script has"
        f" '{FEATURE_VERSION}'."
    )
    exit()

model = joblib.load("gaze_model.pkl")
USE_HEAD_POSE = metadata.get("use_head_pose", False)
affine_correction = np.asarray(metadata.get("affine_correction", []), dtype=float)
if affine_correction.shape != (3, 2):
    affine_correction = None

print()
print("==============================================")
print("GAZE MODEL LOADED")
print("Model:", metadata.get("model_type"))
print("Feature version:", metadata.get("feature_version"))
print("Calibration error:", f"{metadata.get('cv_mean_error_px', 0):.1f}px")
print("==============================================")

MODEL_PATH = "models/face_landmarker.task"
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_facial_transformation_matrixes=USE_HEAD_POSE,
)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    detector.close()
    raise RuntimeError("ERROR: Could not open webcam.")

WINDOW_NAME = "Autism Visual Attention Assessment"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

target_x = SCREEN_WIDTH // 2
target_y = SCREEN_HEIGHT // 2
target_radius = 60
speed_x = 4
speed_y = 3

TRACKER_DEADBAND = 2.0
MAX_FRAME_MOVEMENT = 35.0
DROPOUT_TIMEOUT_S = 0.5

gaze_filter = KalmanGazeFilter(
    process_noise=800.0,
    measurement_noise=225.0,
    dead_zone=TRACKER_DEADBAND,
    max_movement=MAX_FRAME_MOVEMENT,
    dropout_timeout=DROPOUT_TIMEOUT_S,
)

CONCENTRATION_RADIUS = 300
total_frames = 0
valid_frames = 0
dropped_frames = 0
concentration_sum = 0.0
distance_sum = 0.0

csv_file = open("concentration_session.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "Frame", "Target_X", "Target_Y", "Gaze_X", "Gaze_Y", "Distance",
    "Concentration_Percentage", "Confidence", "Status"
])

start_time = time.perf_counter()
print()
print("Concentration assessment started.")
print("Follow the moving target using your eyes.")
print("Press Q to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.perf_counter() - start_time) * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        target_x += speed_x
        target_y += speed_y
        if target_x - target_radius <= 0 or target_x + target_radius >= SCREEN_WIDTH:
            speed_x *= -1
        if target_y - target_radius <= 0 or target_y + target_radius >= SCREEN_HEIGHT:
            speed_y *= -1

        canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        gaze_x = None
        gaze_y = None
        distance = None
        concentration = 0.0
        valid_prediction = False
        gaze_confidence = 0.0

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            matrix = None
            if (
                USE_HEAD_POSE
                and result.facial_transformation_matrixes
                and len(result.facial_transformation_matrixes) > 0
            ):
                matrix = result.facial_transformation_matrixes[0]

            features = get_features(landmarks, matrix)
            metrics = extract_eye_metrics(landmarks, matrix)
            if features is not None and metrics is not None:
                gaze_confidence = estimate_gaze_confidence(metrics)
                if gaze_confidence >= MIN_GAZE_CONFIDENCE:
                    feature_array = np.asarray(features, dtype=float).reshape(1, -1)
                    prediction = model.predict(feature_array)[0]
                    raw_x = float(prediction[0])
                    raw_y = float(prediction[1])
                    if affine_correction is not None:
                        corrected = np.array([raw_x, raw_y, 1.0]) @ affine_correction
                        raw_x = float(corrected[0])
                        raw_y = float(corrected[1])

                    filtered_x, filtered_y = gaze_filter.update(raw_x, raw_y)
                    gaze_x = int(np.clip(filtered_x, 0, SCREEN_WIDTH - 1))
                    gaze_y = int(np.clip(filtered_y, 0, SCREEN_HEIGHT - 1))
                    distance = math.hypot(gaze_x - target_x, gaze_y - target_y)
                    if distance <= CONCENTRATION_RADIUS:
                        proximity = 1.0 - (distance / CONCENTRATION_RADIUS)
                        concentration = (proximity ** 0.65) * 100.0
                    valid_prediction = True

        total_frames += 1

        if not valid_prediction:
            gaze_filter.handle_dropout()
            dropped_frames += 1
            status_text = "DROPPED"
        else:
            valid_frames += 1
            concentration_sum += concentration
            distance_sum += distance
            status_text = "VALID"

        csv_writer.writerow([
            total_frames,
            target_x,
            target_y,
            gaze_x if gaze_x is not None else "",
            gaze_y if gaze_y is not None else "",
            f"{distance:.1f}" if distance is not None else "",
            f"{concentration:.1f}",
            f"{gaze_confidence:.2f}",
            status_text,
        ])

        cv2.circle(canvas, (target_x, target_y), target_radius, (0, 255, 0), -1)
        if gaze_x is not None and gaze_y is not None:
            cv2.circle(canvas, (gaze_x, gaze_y), 20, (255, 0, 0), -1)
            cv2.line(canvas, (target_x, target_y), (gaze_x, gaze_y), (255, 255, 255), 2)

        cv2.putText(canvas, f"Concentration: {concentration:.1f}%", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        if valid_prediction:
            cv2.putText(canvas, f"Confidence: {gaze_confidence * 100:.0f}%", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(canvas, "TRACKING LOST / LOW CONFIDENCE", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.putText(canvas, "Follow the moving target using your eyes", (50, SCREEN_HEIGHT - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    detector.close()
    csv_file.close()
    cv2.destroyAllWindows()
    if total_frames > 0:
        average_concentration = (concentration_sum / valid_frames) if valid_frames > 0 else 0.0
        average_distance = (distance_sum / valid_frames) if valid_frames > 0 else 0.0
        dropout_rate = (dropped_frames / total_frames) * 100.0
        valid_rate = (valid_frames / total_frames) * 100.0
        print()
        print("==============================================")
        print("SESSION COMPLETE")
        print("==============================================")
        print(f"Total frames processed: {total_frames}")
        print(f"Valid gaze frames:      {valid_frames} ({valid_rate:.1f}%)")
        print(f"Dropped/uncertain:      {dropped_frames} ({dropout_rate:.1f}%)")
        print(f"Average concentration:  {average_concentration:.2f}%")
        print(f"Average gaze distance:  {average_distance:.2f}px")
        print("CSV saved as: concentration_session.csv")
        print("==============================================")
