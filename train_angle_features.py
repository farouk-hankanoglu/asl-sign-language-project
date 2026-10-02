

import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
import time

from angle_features import landmarks_to_angles, angle_feature_names

LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def load_and_filter(path):
    df = pd.read_csv(path)
    return df[df["label"].isin(LETTERS)].copy()


def to_angle_matrix(df, feature_cols):
    """Converts a dataframe of raw landmark rows into a matrix of angle features."""
    raw = df[feature_cols].values.astype(np.float32)
    angle_rows = [landmarks_to_angles(row) for row in raw]
    return np.array(angle_rows, dtype=np.float32)


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

    print("Converting raw landmarks to rotation-invariant angle features...")
    X_train = to_angle_matrix(train_df, feature_cols)
    X_test = to_angle_matrix(df_third, feature_cols)
    print(f"Feature vector size: {X_train.shape[1]} (was 63 raw coords, now angle features)")

    all_labels = pd.concat([train_df["label"], df_third["label"]])
    le = LabelEncoder()
    le.fit(all_labels)
    y_train = le.transform(train_df["label"].values)
    y_test = le.transform(df_third["label"].values)
    num_classes = len(le.classes_)

    print(f"Training samples: {len(X_train)} | Testing on SignAlphaSet: {len(X_test)}")

    # ---------- Random Forest ----------
    print("\n--- Training Random Forest (angle features) ---")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    print(f"Trained in {time.time()-t0:.1f}s")
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Random Forest accuracy on SignAlphaSet: {rf_acc:.4f}")

    # ---------- Neural Network ----------
    print("\n--- Training Neural Network (angle features) ---")
    nn = build_nn(X_train.shape[1], num_classes)
    t0 = time.time()
    nn.fit(X_train, y_train, epochs=40, batch_size=16, verbose=0, validation_split=0.1)
    print(f"Trained in {time.time()-t0:.1f}s")
    nn_preds = np.argmax(nn.predict(X_test, verbose=0), axis=1)
    nn_acc = accuracy_score(y_test, nn_preds)
    print(f"Neural Network accuracy on SignAlphaSet: {nn_acc:.4f}")

    best_name = "random_forest" if rf_acc >= nn_acc else "neural_network"
    best_preds = rf_preds if best_name == "random_forest" else nn_preds

    print(f"\n--- Per-letter report on SignAlphaSet ({best_name}, ANGLE FEATURES) ---")
    print(classification_report(y_test, best_preds, target_names=le.classes_, zero_division=0))

    cm = confusion_matrix(y_test, best_preds)
    print(f"--- Confusion matrix on SignAlphaSet ({best_name}, ANGLE FEATURES) ---")
    print("Classes:", list(le.classes_))
    print(cm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", required=True)
    parser.add_argument("--aslhg", required=True)
    parser.add_argument("--third", required=True)
    args = parser.parse_args()
    main(args.kaggle, args.aslhg, args.third)
