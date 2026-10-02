

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

IMG_SIZE = 224
BATCH_SIZE = 32


def load_datasets(train_dir, test_dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        label_mode="int", shuffle=True, seed=42, validation_split=0.1,
        subset="training",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        label_mode="int", shuffle=True, seed=42, validation_split=0.1,
        subset="validation",
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        label_mode="int", shuffle=False,
    )
    class_names = train_ds.class_names
    return train_ds, val_ds, test_ds, class_names


def build_model(num_classes):
    base = MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False  # Phase 1: frozen

   
    augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.15),        # +/- ~27 degrees
        layers.RandomZoom(0.2),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomBrightness(0.3),
        layers.RandomContrast(0.3),
        layers.RandomFlip("horizontal"),
    ])

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augmentation(inputs)
    x = preprocess_input(x)
    x = base(x, training=False)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    return model, base


def main(train_dir, test_dir, phase1_epochs, phase2_epochs):
    print("=" * 60)
    print("Loading datasets...")
    train_ds, val_ds, test_ds, class_names = load_datasets(train_dir, test_dir)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # Prefetch for speed
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds_eval = test_ds.prefetch(AUTOTUNE)

    model, base = build_model(num_classes)

    # ---------- Phase 1: train head only, backbone frozen ----------
    print("\n--- Phase 1: training classifier head (backbone frozen) ---")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    t0 = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=phase1_epochs, verbose=1)
    print(f"Phase 1 done in {time.time()-t0:.1f}s")

    # ---------- Phase 2: fine-tune whole network, low LR ----------
    print("\n--- Phase 2: fine-tuning full network (backbone unfrozen) ---")
    base.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    t0 = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=phase2_epochs, verbose=1)
    print(f"Phase 2 done in {time.time()-t0:.1f}s")

    # ---------- Evaluate on held-out third dataset ----------
    print("\n--- Evaluating on held-out test set (never seen during training) ---")
    y_true = []
    y_pred = []
    for images, labels in test_ds_eval:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    acc = accuracy_score(y_true, y_pred)
    print(f"\nFINAL accuracy on held-out dataset: {acc:.4f}")

    print("\n--- Per-letter report ---")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print("--- Confusion matrix ---")
    print("Classes:", class_names)
    print(cm)

    model.save("asl_cnn_model.keras")
    print("\nModel saved to asl_cnn_model.keras")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Folder with combined training images (class subfolders)")
    parser.add_argument("--test", required=True, help="Folder with held-out test images (class subfolders)")
    parser.add_argument("--phase1_epochs", type=int, default=8)
    parser.add_argument("--phase2_epochs", type=int, default=8)
    args = parser.parse_args()
    main(args.train, args.test, args.phase1_epochs, args.phase2_epochs)
