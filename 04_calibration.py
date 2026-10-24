import csv
import json
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import tkinter as tk

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gaze_features import (
    FEATURE_VERSION,
    feature_names,
    get_features,
)
from gaze_calibration import fit_four_point_homography


MARGIN_X = 100
MARGIN_Y = 80

SETTLE_TIME = 1.5
COLLECT_TIME = 3.0

MIN_SAMPLES_PER_POINT = 20
MAD_THRESHOLD = 3.5
MAX_NORMALIZED_OFFSET = 2.0
MAX_HEAD_POSE_RAD = 1.5

USE_HEAD_POSE = False
MODEL_PATH = "models/face_landmarker.task"
WINDOW_NAME = "Gaze Calibration"

CALIBRATION_PARTICIPANT_ID = "child_session_default"
CALIBRATION_SESSION_ID = f"calibration_{int(time.time())}"


root = tk.Tk()
root.withdraw()

SCREEN_WIDTH = root.winfo_screenwidth()
SCREEN_HEIGHT = root.winfo_screenheight()

root.destroy()


def generate_targets():
    return [
        (MARGIN_X, MARGIN_Y),
        (SCREEN_WIDTH - MARGIN_X, MARGIN_Y),
        (SCREEN_WIDTH - MARGIN_X, SCREEN_HEIGHT - MARGIN_Y),
        (MARGIN_X, SCREEN_HEIGHT - MARGIN_Y),
    ]


def draw_target(x, y, point_number, total_points):
    canvas = np.zeros(
        (SCREEN_HEIGHT, SCREEN_WIDTH, 3),
        dtype=np.uint8,
    )

    cv2.circle(
        canvas,
        (x, y),
        28,
        (0, 0, 255),
        -1,
    )

    cv2.circle(
        canvas,
        (x, y),
        8,
        (255, 255, 255),
        -1,
    )

    cv2.putText(
        canvas,
        "Look at the red dot",
        (40, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"Point {point_number}/{total_points}",
        (40, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow(WINDOW_NAME, canvas)


def reject_outliers(samples):
    feature_count = len(
        feature_names(USE_HEAD_POSE)
    )

    values = np.asarray(
        samples,
        dtype=float,
    )

    if len(values) < MIN_SAMPLES_PER_POINT:
        return np.empty(
            (0, feature_count),
            dtype=float,
        )

    median = np.median(
        values,
        axis=0,
    )

    mad = np.median(
        np.abs(values - median),
        axis=0,
    )

    mad[mad < 0.000001] = 0.000001

    deviation = np.abs(
        values - median
    ) / mad

    keep_mask = np.all(
        deviation <= MAD_THRESHOLD,
        axis=1,
    )

    return values[keep_mask]


def is_valid_feature_vector(values):
    feature_vector = np.asarray(values, dtype=float)
    if feature_vector.ndim != 1 or feature_vector.size not in (4, 7):
        return False
    if not np.all(np.isfinite(feature_vector)):
        return False
    if np.max(np.abs(feature_vector[:4])) > MAX_NORMALIZED_OFFSET:
        return False
    if feature_vector.size > 4 and np.max(np.abs(feature_vector[4:])) > MAX_HEAD_POSE_RAD:
        return False
    return True


targets = generate_targets()

print()
print("==============================================")
print("GAZE CALIBRATION")
print("==============================================")
print(f"Screen: {SCREEN_WIDTH} x {SCREEN_HEIGHT}")
print(f"Calibration points: {len(targets)}")
print("Look at each corner target, keeping your head still.")
print("Press Q to stop.")
print()

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH,
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.6,
    min_face_presence_confidence=0.6,
    min_tracking_confidence=0.6,
    output_facial_transformation_matrixes=USE_HEAD_POSE,
)

detector = vision.FaceLandmarker.create_from_options(
    options
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    detector.close()
    raise RuntimeError(
        "Could not open webcam."
    )

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL,
)

cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN,
)


features_all = []
targets_all = []
groups_all = []

session_start_time = time.perf_counter()
last_timestamp_ms = -1


def get_next_timestamp():
    global last_timestamp_ms

    timestamp_ms = int(
        (time.perf_counter() - session_start_time)
        * 1000
    )

    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1

    last_timestamp_ms = timestamp_ms

    return timestamp_ms


try:
    time.sleep(2)

    for point_index, (target_x, target_y) in enumerate(
        targets
    ):
        point_number = point_index + 1

        print(
            f"Collecting point "
            f"{point_number}/{len(targets)} "
            f"at ({target_x}, {target_y})"
        )

        settle_start = time.perf_counter()

        while (
            time.perf_counter() - settle_start
            < SETTLE_TIME
        ):
            draw_target(
                target_x,
                target_y,
                point_number,
                len(targets),
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

        point_samples = []
        collect_start = time.perf_counter()

        while (
            time.perf_counter() - collect_start
            < COLLECT_TIME
        ):
            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            result = detector.detect_for_video(
                image,
                get_next_timestamp(),
            )

            if result.face_landmarks:
                matrix = None

                if (
                    USE_HEAD_POSE
                    and result.facial_transformation_matrixes
                ):
                    matrix = (
                        result.facial_transformation_matrixes[0]
                    )

                values = get_features(
                    result.face_landmarks[0],
                    matrix,
                )

                if values is not None:
                    values = np.asarray(
                        values,
                        dtype=float,
                    )

                    if is_valid_feature_vector(values):
                        point_samples.append(values)

            draw_target(
                target_x,
                target_y,
                point_number,
                len(targets),
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

        accepted = reject_outliers(point_samples)

        if len(accepted) < MIN_SAMPLES_PER_POINT:
            print(
                f"Rejected point: only "
                f"{len(accepted)} stable samples"
            )
            continue

        # Use one robust feature vector per target. This prevents many
        # adjacent frames from letting the model memorize camera noise.
        point_feature = np.median(accepted, axis=0)
        features_all.append(point_feature)
        targets_all.append([target_x, target_y])
        groups_all.append(point_index)

        print(
            f"Accepted {len(accepted)}/"
            f"{len(point_samples)} samples"
        )

finally:
    cap.release()
    detector.close()
    cv2.destroyAllWindows()


features_all = np.asarray(
    features_all,
    dtype=float,
)

targets_all = np.asarray(
    targets_all,
    dtype=float,
)

groups_all = np.asarray(
    groups_all,
    dtype=int,
)

if len(features_all) == 0:
    raise RuntimeError(
        "No calibration samples were collected."
    )

valid_points = len(
    np.unique(groups_all)
)

if valid_points != 4:
    raise RuntimeError(
        f"Four valid calibration points are required; collected {valid_points}. "
        "Improve lighting and camera position, then retry."
    )

expected_features = len(
    feature_names(USE_HEAD_POSE)
)

if features_all.shape[1] != expected_features:
    raise RuntimeError(
        f"Feature mismatch: received "
        f"{features_all.shape[1]}, expected "
        f"{expected_features}."
    )


homography = fit_four_point_homography(
    features_all,
    targets_all,
)

joblib.dump(
    {
        "calibration_method": "four_point_homography",
        "homography": homography,
    },
    "gaze_model.pkl",
)


with open(
    "calibration_raw.csv",
    "w",
    newline="",
) as file:
    writer = csv.writer(file)

    writer.writerow(
        feature_names(USE_HEAD_POSE)
        + [
            "target_x",
            "target_y",
            "point_id",
        ]
    )

    for values, target, group in zip(
        features_all,
        targets_all,
        groups_all,
    ):
        writer.writerow(
            list(values)
            + [
                int(target[0]),
                int(target[1]),
                int(group),
            ]
        )


metadata = {
    "feature_version": FEATURE_VERSION,
    "feature_names": feature_names(USE_HEAD_POSE),
    "feature_count": int(
        features_all.shape[1]
    ),
    "use_head_pose": USE_HEAD_POSE,
    "participant_id": CALIBRATION_PARTICIPANT_ID,
    "session_id": CALIBRATION_SESSION_ID,
    "model_type": "four_point_homography",
    "screen_width": SCREEN_WIDTH,
    "screen_height": SCREEN_HEIGHT,
    "margin_x": MARGIN_X,
    "margin_y": MARGIN_Y,
    "n_calibration_points": valid_points,
    "n_samples": int(len(features_all)),
    "calibration_method": "four_point_homography",
    "quality_notes": "Four corner targets are fitted with a projective mapping from the averaged binocular gaze features.",
}

with open(
    "gaze_model_metadata.json",
    "w",
) as file:
    json.dump(
        metadata,
        file,
        indent=4,
    )


print()
print("==============================================")
print("CALIBRATION COMPLETE")
print("==============================================")
print("Calibration method: four-point homography")
print("Four screen-corner anchors fitted.")
print("Saved gaze_model.pkl")
print("Saved gaze_model_metadata.json")