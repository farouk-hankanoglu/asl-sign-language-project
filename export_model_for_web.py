"""
export_model_for_web.py

Trains the small feedforward neural network (our landmark classifier)
on the combined, mirror-augmented Kaggle+ASL-HG data, then exports its
weights as a single JSON file that a webpage can load directly - no
Python/TensorFlow needed in the browser, just plain JavaScript math.

Why the neural network (not Random Forest) for the web demo:
Random Forest is a collection of decision trees, which doesn't export
to a simple numeric format - it's awkward to run in JavaScript. Our
small neural network is just a few matrix multiplications, which is
easy and fast to reimplement in plain JS.

Usage:
    python export_model_for_web.py --kaggle landmarks_train.csv --aslhg landmarks_aslhg.csv --output model_weights.json
"""

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models

LETTERS = sorted("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def load_and_filter(path, mirror=True):
    df = pd.read_csv(path)
    df = df[df["label"].isin(LETTERS)].copy()
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values
    if mirror:
        X_mirrored = X.copy()
        X_mirrored[:, 0::3] *= -1
        X = np.concatenate([X, X_mirrored], axis=0)
        y = np.concatenate([y, y], axis=0)
    return X, y


def build_nn(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def export_weights_to_json(model, class_names, output_path):
    """
    Exports each Dense layer's weights and biases as plain nested
    lists, plus the class label order, so JavaScript can reimplement
    the forward pass with simple matrix math.
    """
    export = {"layers": [], "classes": class_names}
    for layer in model.layers:
        if isinstance(layer, layers.Dense):
            W, b = layer.get_weights()
            export["layers"].append({
                "weights": W.tolist(),   # shape: [input_dim, output_dim]
                "bias": b.tolist(),      # shape: [output_dim]
                "activation": layer.activation.__name__,  # "relu" or "softmax"
            })
    with open(output_path, "w") as f:
        json.dump(export, f)
    print(f"Exported model to {output_path}")


def main(kaggle_path, aslhg_path, output_path):
    print("Loading and combining training data (mirror-augmented)...")
    X_kaggle, y_kaggle = load_and_filter(kaggle_path)
    X_aslhg, y_aslhg = load_and_filter(aslhg_path)
    X_train = np.concatenate([X_kaggle, X_aslhg], axis=0)
    y_train_raw = np.concatenate([y_kaggle, y_aslhg], axis=0)

    le = LabelEncoder()
    le.fit(LETTERS)
    y_train = le.transform(y_train_raw)

    print(f"Training on {len(X_train)} samples...")
    model = build_nn(X_train.shape[1], len(LETTERS))
    model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=1, validation_split=0.1)

    export_weights_to_json(model, list(le.classes_), output_path)
    print("\nDone. Upload this JSON file into the web demo to use your trained model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", required=True)
    parser.add_argument("--aslhg", required=True)
    parser.add_argument("--output", default="model_weights.json")
    args = parser.parse_args()
    main(args.kaggle, args.aslhg, args.output)
