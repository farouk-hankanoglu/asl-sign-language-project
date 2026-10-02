
import os
import csv
import argparse
import urllib.request
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    raise SystemExit("Run: pip install mediapipe")

import cv2

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = "hand_landmarker.task"


def ensure_model_downloaded():
    """Downloads the hand-landmark model once, if not already present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand-landmark model (one-time, ~8MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded:", MODEL_PATH)


def normalize_landmarks(landmarks):
    """
    landmarks: list of 21 (x, y, z) tuples from MediaPipe.
    Returns a flat list of 63 normalized numbers.
    """
    pts = np.array(landmarks, dtype=np.float32)  # shape (21, 3)

    # Step 1: translate so wrist (point 0) is the origin
    wrist = pts[0].copy()
    pts = pts - wrist

    # Step 2: scale so wrist-to-middle-fingertip (point 12) distance = 1.0
    scale = np.linalg.norm(pts[12])
    if scale < 1e-6:
        scale = 1e-6  # avoid divide-by-zero on degenerate detections
    pts = pts / scale

    return pts.flatten().tolist()  # 63 numbers


def extract_from_dataset(input_dir, output_csv, max_per_class=None):
    ensure_model_downloaded()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,  # each image is independent
        num_hands=1,                                # ASL fingerspelling = one hand
        min_hand_detection_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    rows_written = 0
    skipped_no_hand = 0

    classes = sorted(
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    )

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"]
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(header)

        for label in classes:
            class_dir = os.path.join(input_dir, label)
            image_files = [
                fn for fn in os.listdir(class_dir)
                if fn.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            if max_per_class:
                image_files = image_files[:max_per_class]

            for fn in image_files:
                img_path = os.path.join(class_dir, fn)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                result = landmarker.detect(mp_image)

                if not result.hand_landmarks:
                    skipped_no_hand += 1
                    continue

                hand = result.hand_landmarks[0]  # first detected hand
                raw_points = [(lm.x, lm.y, lm.z) for lm in hand]
                normalized = normalize_landmarks(raw_points)

                writer.writerow([label] + normalized)
                rows_written += 1

    landmarker.close()
    print(f"Done. Wrote {rows_written} rows to {output_csv}")
    print(f"Skipped {skipped_no_hand} images where no hand was detected "
          f"(this is normal - some dataset images are low quality)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to dataset root folder")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument("--max_per_class", type=int, default=None,
                         help="Optional cap on images per class, for quick testing")
    args = parser.parse_args()

    extract_from_dataset(args.input, args.output, args.max_per_class)
