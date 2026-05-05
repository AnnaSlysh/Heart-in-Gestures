#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Retrain the keypoint classifier with:
  - Horizontal-mirror augmentation  → makes the model work for BOTH hands
  - Random landmark jitter           → improves robustness / accuracy
  - Class-weight balancing           → compensates for Y (308) vs R (2429)

Run from the project root:
    python model/keypoint_classifier/retrain_with_mirror.py
"""

import csv
import numpy as np
import tensorflow as tf
from collections import Counter

KEYPOINT_CSV = 'model/keypoint.csv'
LABEL_CSV    = 'model/keypoint_classifier/keypoint_classifier_label.csv'
MODEL_KERAS  = 'model/keypoint_classifier/keypoint_classifier.keras'
MODEL_TFLITE = 'model/keypoint_classifier/keypoint_classifier.tflite'

NUM_FEATURES = 42   # 21 landmarks × 2 (x, y)
JITTER_STD   = 0.02 # std of gaussian noise added to landmarks during augmentation
JITTER_COPIES = 3   # extra jittered copies per original sample


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data():
    X, y = [], []
    with open(KEYPOINT_CSV, encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if not row:
                continue
            y.append(int(row[0]))
            X.append([float(v) for v in row[1:]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def mirror_x(X):
    """Negate x-coordinates (even indices 0, 2, 4, …) — turns left hand into right hand."""
    m = X.copy()
    m[:, 0::2] = -m[:, 0::2]
    return m


def add_jitter(X, y, std=JITTER_STD, copies=JITTER_COPIES, rng=None):
    """Add Gaussian noise to landmarks to simulate slight hand-position variation."""
    if rng is None:
        rng = np.random.default_rng(0)
    parts = [X]
    parts_y = [y]
    for _ in range(copies):
        noise = rng.normal(0, std, size=X.shape).astype(np.float32)
        parts.append(np.clip(X + noise, -1.0, 1.0))
        parts_y.append(y)
    return np.vstack(parts), np.concatenate(parts_y)


def build_augmented_dataset(rng):
    print("Loading training data …")
    X, y = load_data()
    print(f"  Original samples  : {len(X)}")

    X_mirror = mirror_x(X)
    X_all = np.vstack([X, X_mirror])
    y_all = np.concatenate([y, y])
    print(f"  After mirroring   : {len(X_all)}")

    X_all, y_all = add_jitter(X_all, y_all, rng=rng)
    print(f"  After jitter aug  : {len(X_all)}")
    return X_all, y_all


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes):
    try:
        model = tf.keras.models.load_model(MODEL_KERAS)
        print(f"Loaded existing model from {MODEL_KERAS}")
        # Re-compile with a lower LR for fine-tuning
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        return model
    except Exception as e:
        print(f"Could not load existing model ({e}) — building a fresh one.")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(NUM_FEATURES,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(LABEL_CSV, encoding='utf-8-sig') as f:
        labels = [row[0] for row in csv.reader(f)]
    num_classes = len(labels)
    print(f"Classes ({num_classes}): {labels}\n")

    rng = np.random.default_rng(42)
    X_all, y_all = build_augmented_dataset(rng)

    # Shuffle
    idx = rng.permutation(len(X_all))
    X_all, y_all = X_all[idx], y_all[idx]

    split = int(0.85 * len(X_all))
    X_train, X_val = X_all[:split], X_all[split:]
    y_train, y_val = y_all[:split], y_all[split:]
    print(f"\nTrain : {len(X_train)}  |  Val : {len(X_val)}\n")

    # Class weights to compensate for imbalance (Y=308 vs R=2429)
    counts = Counter(y_train.tolist())
    max_count = max(counts.values())
    class_weight = {c: max_count / counts[c] for c in counts}
    print("Class weights (top imbalanced):")
    for c in sorted(class_weight, key=lambda c: -class_weight[c])[:5]:
        print(f"  {labels[c]:4s}: {class_weight[c]:.2f}x")
    print()

    model = build_model(num_classes)
    model.summary()
    print()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=20,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=8, verbose=1,
        ),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=64,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Save
    model.save(MODEL_KERAS)
    print(f"\nSaved Keras model  → {MODEL_KERAS}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(MODEL_TFLITE, 'wb') as f_out:
        f_out.write(tflite_model)
    print(f"Saved TFLite model → {MODEL_TFLITE}")
    print("\nDone! Restart Streamlit to use the updated model.")


if __name__ == '__main__':
    main()
