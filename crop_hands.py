

import os
import argparse
import urllib.request
import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = "hand_landmarker.task"

IMG_SIZE = 224          # standard input size for MobileNetV3
PADDING_FRACTION = 0.35  # extra margin around the detected hand box

LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def ensure_model_downloaded():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand-landmark model (one-time, ~8MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded:", MODEL_PATH)


def crop_with_padding(image, landmarks_px, pad_frac):
    xs = [p[0] for p in landmarks_px]
    ys = [p[1] for p in landmarks_px]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    box_w = x_max - x_min
    box_h = y_max - y_min
    pad_x = box_w * pad_frac
    pad_y = box_h * pad_frac

    h, w = image.shape[:2]
    x_min = max(0, int(x_min - pad_x))
    x_max = min(w, int(x_max + pad_x))
    y_min = max(0, int(y_min - pad_y))
    y_max = min(h, int(y_max + pad_y))

    if x_max <= x_min or y_max <= y_min:
        return None
    return image[y_min:y_max, x_min:x_max]


def process_dataset(input_dir, output_dir, prefix, max_per_class, letters_only):
    ensure_model_downloaded()
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    classes = sorted(
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    )
    if letters_only:
        classes = [c for c in classes if c in LETTERS]

    total_saved = 0
    total_skipped = 0

    for label in classes:
        class_in = os.path.join(input_dir, label)
        class_out = os.path.join(output_dir, label)
        os.makedirs(class_out, exist_ok=True)

        image_files = [
            fn for fn in os.listdir(class_in)
            if fn.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if max_per_class:
            image_files = image_files[:max_per_class]

        saved_this_class = 0
        for fn in image_files:
            img_path = os.path.join(class_in, fn)
            image = cv2.imread(img_path)
            if image is None:
                total_skipped += 1
                continue

            h, w = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = landmarker.detect(mp_image)

            if not result.hand_landmarks:
                total_skipped += 1
                continue

            hand = result.hand_landmarks[0]
            landmarks_px = [(lm.x * w, lm.y * h) for lm in hand]

            crop = crop_with_padding(image, landmarks_px, PADDING_FRACTION)
            if crop is None:
                total_skipped += 1
                continue

            crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            out_path = os.path.join(class_out, f"{prefix}_{saved_this_class}.jpg")
            cv2.imwrite(out_path, crop_resized)
            saved_this_class += 1
            total_saved += 1

        print(f"  {label}: saved {saved_this_class} cropped images")

    landmarker.close()
    print(f"\nDone. Total saved: {total_saved}, total skipped: {total_skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix", required=True,
                         help="Short tag for this dataset, used in output filenames "
                              "so multiple datasets can share the same output folder.")
    parser.add_argument("--max_per_class", type=int, default=300)
    parser.add_argument("--letters_only", action="store_true", default=True,
                         help="Only process A-Z folders, skip digits/space/del/nothing.")
    args = parser.parse_args()

    process_dataset(args.input, args.output, args.prefix, args.max_per_class, args.letters_only)
