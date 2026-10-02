

import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
import time
import json


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values
    return X, y


def build_nn(input_dim, num_classes):
    """
    Small feed-forward network:
      63 inputs -> 64 -> 32 -> num_classes (softmax)
    Small enough to run in real time on a phone via TensorFlow Lite.
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main(primary_path, secondary_path):
    print("=" * 60)
    print("Loading primary dataset:", primary_path)
    X, y_raw = load_data(primary_path)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    print(f"Loaded {len(X)} samples, {num_classes} classes: {list(le.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = {}

    # ---------- Random Forest ----------
    print("\n--- Training Random Forest ---")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - t0

    t0 = time.time()
    rf_preds = rf.predict(X_test)
    rf_infer_time = (time.time() - t0) / len(X_test) * 1000  # ms per sample

    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Random Forest accuracy: {rf_acc:.4f}")
    print(f"Train time: {rf_train_time:.2f}s | Inference: {rf_infer_time:.4f} ms/sample")
    results["random_forest"] = {
        "accuracy": rf_acc,
        "train_time_sec": rf_train_time,
        "inference_ms_per_sample": rf_infer_time,
    }

    # ---------- Neural Network ----------
    print("\n--- Training Neural Network ---")
    nn = build_nn(X.shape[1], num_classes)
    t0 = time.time()
    nn.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0,
           validation_split=0.1)
    nn_train_time = time.time() - t0

    t0 = time.time()
    nn_preds = np.argmax(nn.predict(X_test, verbose=0), axis=1)
    nn_infer_time = (time.time() - t0) / len(X_test) * 1000

    nn_acc = accuracy_score(y_test, nn_preds)
    print(f"Neural Network accuracy: {nn_acc:.4f}")
    print(f"Train time: {nn_train_time:.2f}s | Inference: {nn_infer_time:.4f} ms/sample")
    results["neural_network"] = {
        "accuracy": nn_acc,
        "train_time_sec": nn_train_time,
        "inference_ms_per_sample": nn_infer_time,
    }

    # ---------- Confusion matrix (on the better model) ----------
    best_model_name = "random_forest" if rf_acc >= nn_acc else "neural_network"
    best_preds = rf_preds if best_model_name == "random_forest" else nn_preds
    cm = confusion_matrix(y_test, best_preds)
    print(f"\n--- Confusion matrix ({best_model_name}, rows=true, cols=predicted) ---")
    print("Classes:", list(le.classes_))
    print(cm)

    report = classification_report(y_test, best_preds, target_names=le.classes_,
                                    zero_division=0)
    print(f"\n--- Per-letter report ({best_model_name}) ---")
    print(report)

    # ---------- Cross-dataset evaluation ----------
    if secondary_path:
        print("\n" + "=" * 60)
        print("Cross-dataset test: train on primary, evaluate on:", secondary_path)
        X2, y2_raw = load_data(secondary_path)
        # only keep classes seen during training
        mask = np.isin(y2_raw, le.classes_)
        X2, y2_raw = X2[mask], y2_raw[mask]
        y2 = le.transform(y2_raw)

        rf_cross_preds = rf.predict(X2)
        rf_cross_acc = accuracy_score(y2, rf_cross_preds)

        nn_cross_preds = np.argmax(nn.predict(X2, verbose=0), axis=1)
        nn_cross_acc = accuracy_score(y2, nn_cross_preds)

        print(f"Random Forest cross-dataset accuracy: {rf_cross_acc:.4f} "
              f"(same-dataset was {rf_acc:.4f})")
        print(f"Neural Network cross-dataset accuracy: {nn_cross_acc:.4f} "
              f"(same-dataset was {nn_acc:.4f})")

        results["random_forest"]["cross_dataset_accuracy"] = rf_cross_acc
        results["neural_network"]["cross_dataset_accuracy"] = nn_cross_acc

    with open("/home/claude/asl_project/outputs/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved results to outputs/results.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--secondary", default=None)
    args = parser.parse_args()
    main(args.primary, args.secondary)
