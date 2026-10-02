

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


LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def load_and_filter(path, source_name):
    df = pd.read_csv(path)
    df = df[df["label"].isin(LETTERS)].copy()
    df["source"] = source_name
    return df


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


def main(kaggle_path, aslhg_path):
    print("=" * 60)
    df_kaggle = load_and_filter(kaggle_path, "kaggle")
    df_aslhg = load_and_filter(aslhg_path, "aslhg")
    print(f"Kaggle (letters only): {len(df_kaggle)} samples")
    print(f"ASL-HG (letters only): {len(df_aslhg)} samples")

    combined = pd.concat([df_kaggle, df_aslhg], ignore_index=True)
    print(f"Combined total: {len(combined)} samples")

    feature_cols = [c for c in combined.columns if c not in ("label", "source")]
    X = combined[feature_cols].values.astype(np.float32)
    y_raw = combined["label"].values
    source = combined["source"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)

    # Stratify by label so each letter is proportionally represented
    # in train/test; we keep the source labels aligned via indices.
    indices = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        indices, test_size=0.2, stratify=y, random_state=42
    )

    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    source_test = source[idx_test]

    # ---------- Random Forest ----------
    print("\n--- Training Random Forest on COMBINED data ---")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - t0
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Random Forest overall accuracy: {rf_acc:.4f} (train time {rf_train_time:.1f}s)")

    for src in ["kaggle", "aslhg"]:
        mask = source_test == src
        acc = accuracy_score(y_test[mask], rf_preds[mask])
        print(f"  -> accuracy on {src} test samples: {acc:.4f} (n={mask.sum()})")

    # ---------- Neural Network ----------
    print("\n--- Training Neural Network on COMBINED data ---")
    nn = build_nn(X.shape[1], num_classes)
    t0 = time.time()
    nn.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0, validation_split=0.1)
    nn_train_time = time.time() - t0
    nn_preds = np.argmax(nn.predict(X_test, verbose=0), axis=1)
    nn_acc = accuracy_score(y_test, nn_preds)
    print(f"Neural Network overall accuracy: {nn_acc:.4f} (train time {nn_train_time:.1f}s)")

    for src in ["kaggle", "aslhg"]:
        mask = source_test == src
        acc = accuracy_score(y_test[mask], nn_preds[mask])
        print(f"  -> accuracy on {src} test samples: {acc:.4f} (n={mask.sum()})")

    # ---------- Confusion matrix + report on the better model ----------
    best_name = "random_forest" if rf_acc >= nn_acc else "neural_network"
    best_preds = rf_preds if best_name == "random_forest" else nn_preds
    print(f"\n--- Per-letter report ({best_name}, combined test set) ---")
    print(classification_report(y_test, best_preds, target_names=le.classes_, zero_division=0))

    cm = confusion_matrix(y_test, best_preds)
    print(f"--- Confusion matrix ({best_name}) ---")
    print("Classes:", list(le.classes_))
    print(cm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", required=True)
    parser.add_argument("--aslhg", required=True)
    args = parser.parse_args()
    main(args.kaggle, args.aslhg)
