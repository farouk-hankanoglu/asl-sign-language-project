"""
train_with_augmentation.py

Real fix attempt for the generalization gap found in test_third_dataset.py.

Hypothesis: the systematic letter-pair swaps (G<->Z, O<->E, J<->I) are
caused by left/right hand mirroring inconsistencies between datasets
(e.g. some cameras/datasets mirror the image, others don't). A model
that only ever saw one orientation during training has no way to
recognize the other.

Fix: augment the training data with a horizontally-mirrored copy of
every sample. Mirroring a hand landmark set = flipping the x-coordinate
(x -> -x), keeping y and z the same. The mirrored version still
represents the SAME letter (just viewed as if from the other hand /
a mirrored camera), so it keeps the same label.

This roughly doubles the effective training set size and should make
the model robust to whichever mirroring convention a given dataset used.

Usage:
    python train_with_augmentation.py --kaggle landmarks_train.csv --aslhg landmarks_aslhg.csv --third landmarks_signalphaset.csv
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
import time

LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def load_and_filter(path):
    df = pd.read_csv(path)
    return df[df["label"].isin(LETTERS)].copy()


def mirror_landmarks(X):
    """
    X: array of shape (n_samples, 63), where each row is
    [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20].
    Mirrors horizontally by flipping every x-coordinate (every 3rd
    value starting at index 0).
    """
    X_mirrored = X.copy()
    X_mirrored[:, 0::3] *= -1  # flip all x values
    return X_mirrored


def build_nn(input_dim, num_classes):
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


def main(kaggle_path, aslhg_path, third_path):
    print("=" * 60)
    df_kaggle = load_and_filter(kaggle_path)
    df_aslhg = load_and_filter(aslhg_path)
    df_third = load_and_filter(third_path)

    train_df = pd.concat([df_kaggle, df_aslhg], ignore_index=True)
    feature_cols = [c for c in train_df.columns if c != "label"]

    all_labels = pd.concat([train_df["label"], df_third["label"]])
    le = LabelEncoder()
    le.fit(all_labels)

    X_train_orig = train_df[feature_cols].values.astype(np.float32)
    y_train_orig = le.transform(train_df["label"].values)

    # --- Augmentation: add mirrored copies ---
    X_train_mirrored = mirror_landmarks(X_train_orig)
    X_train = np.concatenate([X_train_orig, X_train_mirrored], axis=0)
    y_train = np.concatenate([y_train_orig, y_train_orig], axis=0)

    print(f"Original training samples: {len(X_train_orig)}")
    print(f"After adding mirrored copies: {len(X_train)}")

    X_test = df_third[feature_cols].values.astype(np.float32)
    y_test = le.transform(df_third["label"].values)
    print(f"Testing on held-out SignAlphaSet: {len(X_test)} samples")

    num_classes = len(le.classes_)

    # ---------- Random Forest ----------
    print("\n--- Training Random Forest (with mirror augmentation) ---")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    print(f"Trained in {time.time()-t0:.1f}s")
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Random Forest accuracy on SignAlphaSet: {rf_acc:.4f}")

    # ---------- Neural Network ----------
    print("\n--- Training Neural Network (with mirror augmentation) ---")
    nn = build_nn(X_train.shape[1], num_classes)
    t0 = time.time()
    nn.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0, validation_split=0.1)
    print(f"Trained in {time.time()-t0:.1f}s")
    nn_preds = np.argmax(nn.predict(X_test, verbose=0), axis=1)
    nn_acc = accuracy_score(y_test, nn_preds)
    print(f"Neural Network accuracy on SignAlphaSet: {nn_acc:.4f}")

    best_name = "random_forest" if rf_acc >= nn_acc else "neural_network"
    best_preds = rf_preds if best_name == "random_forest" else nn_preds

    print(f"\n--- Per-letter report on SignAlphaSet ({best_name}, WITH augmentation) ---")
    print(classification_report(y_test, best_preds, target_names=le.classes_, zero_division=0))

    cm = confusion_matrix(y_test, best_preds)
    print(f"--- Confusion matrix on SignAlphaSet ({best_name}, WITH augmentation) ---")
    print("Classes:", list(le.classes_))
    print(cm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", required=True)
    parser.add_argument("--aslhg", required=True)
    parser.add_argument("--third", required=True)
    args = parser.parse_args()
    main(args.kaggle, args.aslhg, args.third)
