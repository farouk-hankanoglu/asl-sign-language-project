"""
test_third_dataset.py

The real generalization test: trains on ALL of the combined Kaggle +
ASL-HG data (no held-out split this time, since we now have a true
third, independent dataset to test on), then evaluates purely on
SignAlphaSet - a dataset from different people, different country,
different camera setup, that the model has NEVER seen in any form.

This is the strongest possible test of whether the approach actually
generalizes to new signers, or is still tied to the specific datasets
it was trained on.

Usage:
    python test_third_dataset.py --kaggle landmarks_train.csv --aslhg landmarks_aslhg.csv --third landmarks_signalphaset.csv
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


def main(kaggle_path, aslhg_path, third_path):
    print("=" * 60)
    df_kaggle = load_and_filter(kaggle_path, "kaggle")
    df_aslhg = load_and_filter(aslhg_path, "aslhg")
    df_third = load_and_filter(third_path, "signalphaset")

    train_df = pd.concat([df_kaggle, df_aslhg], ignore_index=True)
    print(f"Training on combined Kaggle + ASL-HG: {len(train_df)} samples")
    print(f"Testing on held-out SignAlphaSet: {len(df_third)} samples (never seen)")

    feature_cols = [c for c in train_df.columns if c not in ("label", "source")]

    # Fit the label encoder on the UNION of classes from all three datasets,
    # so any letter present in the third dataset is handled correctly.
    all_labels = pd.concat([train_df["label"], df_third["label"]])
    le = LabelEncoder()
    le.fit(all_labels)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = le.transform(train_df["label"].values)

    X_test = df_third[feature_cols].values.astype(np.float32)
    y_test = le.transform(df_third["label"].values)

    num_classes = len(le.classes_)

    # ---------- Random Forest ----------
    print("\n--- Training Random Forest on Kaggle+ASL-HG ---")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    print(f"Trained in {time.time()-t0:.1f}s")
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Random Forest accuracy on NEW dataset (SignAlphaSet): {rf_acc:.4f}")

    # ---------- Neural Network ----------
    print("\n--- Training Neural Network on Kaggle+ASL-HG ---")
    nn = build_nn(X_train.shape[1], num_classes)
    t0 = time.time()
    nn.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0, validation_split=0.1)
    print(f"Trained in {time.time()-t0:.1f}s")
    nn_preds = np.argmax(nn.predict(X_test, verbose=0), axis=1)
    nn_acc = accuracy_score(y_test, nn_preds)
    print(f"Neural Network accuracy on NEW dataset (SignAlphaSet): {nn_acc:.4f}")

    # ---------- Detailed report on the better model ----------
    best_name = "random_forest" if rf_acc >= nn_acc else "neural_network"
    best_preds = rf_preds if best_name == "random_forest" else nn_preds
    print(f"\n--- Per-letter report on SignAlphaSet ({best_name}) ---")
    print(classification_report(y_test, best_preds, target_names=le.classes_, zero_division=0))

    cm = confusion_matrix(y_test, best_preds)
    print(f"--- Confusion matrix on SignAlphaSet ({best_name}) ---")
    print("Classes:", list(le.classes_))
    print(cm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", required=True)
    parser.add_argument("--aslhg", required=True)
    parser.add_argument("--third", required=True)
    args = parser.parse_args()
    main(args.kaggle, args.aslhg, args.third)
