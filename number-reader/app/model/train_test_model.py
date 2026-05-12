"""
Train/Test pipeline for Thai digit classifier.

- Splits dataset into train (80%) / test (20%) using a fixed seed
- Trains the CNN on train set with augmentation
- Evaluates on the held-out test set
- Saves: model file, metrics.json, confusion_matrix.png, classification_report.txt

Run from number-reader/:
    python app/model/train_test_model.py
"""
import os
import json
import shutil
import random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_recall_fscore_support
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 60
SEED = 42
DATASET_PATH = "dataset"
SPLIT_PATH = "dataset_split"   # temp folder created for train/test split
OUTPUT_DIR = "app/model"
USED_CLASSES = ["thai_0", "thai_1", "thai_2", "thai_3", "thai_4", "thai_5"]
THAI_DIGITS = ["๐", "๑", "๒", "๓", "๔", "๕"]

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── 1. Build train/test split folders ───────────────────────────────────
def build_split():
    if os.path.exists(SPLIT_PATH):
        shutil.rmtree(SPLIT_PATH)

    counts = {"train": {}, "test": {}}
    for cls in USED_CLASSES:
        src = os.path.join(DATASET_PATH, cls)
        files = [f for f in os.listdir(src) if f.endswith(".png")]
        random.shuffle(files)
        n_test = max(1, int(len(files) * 0.2))
        test_files = files[:n_test]
        train_files = files[n_test:]

        for split, flist in [("train", train_files), ("test", test_files)]:
            dst = os.path.join(SPLIT_PATH, split, cls)
            os.makedirs(dst, exist_ok=True)
            for f in flist:
                shutil.copy(os.path.join(src, f), os.path.join(dst, f))
            counts[split][cls] = len(flist)

    print("Split sizes:")
    for split in ("train", "test"):
        total = sum(counts[split].values())
        print(f"  {split}: {total} images  {counts[split]}")
    return counts


# ── 2. Augmentation (matches inference preprocessing) ───────────────────
def augment_stroke_width(image):
    img = image[:, :, 0].astype(np.uint8)
    inv = cv2.bitwise_not(img)
    # Mix of dilation, erosion, and identity so the model sees a wide range
    # of stroke thicknesses (matters for ๕ vs ๐ which are confused when
    # strokes are thick enough to close ๕'s open loop).
    op = np.random.choice(["dilate", "erode", "none"], p=[0.5, 0.25, 0.25])
    if op == "dilate":
        k = np.random.randint(1, 4)
        kernel = np.ones((k, k), np.uint8)
        inv = cv2.dilate(inv, kernel, iterations=1)
    elif op == "erode":
        kernel = np.ones((2, 2), np.uint8)
        inv = cv2.erode(inv, kernel, iterations=1)
    result = cv2.bitwise_not(inv).astype(np.float32)
    return result[:, :, np.newaxis]


# ── 3. Train ────────────────────────────────────────────────────────────
def train_model():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=augment_stroke_width,
        rotation_range=20,
        width_shift_range=0.12,
        height_shift_range=0.12,
        zoom_range=0.15,
        shear_range=10,
        validation_split=0.15,
    )
    train_data = train_datagen.flow_from_directory(
        os.path.join(SPLIT_PATH, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=USED_CLASSES,
        shuffle=True,
        seed=SEED,
        subset="training",
    )
    val_data = train_datagen.flow_from_directory(
        os.path.join(SPLIT_PATH, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=USED_CLASSES,
        shuffle=False,
        seed=SEED,
        subset="validation",
    )

    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(len(USED_CLASSES), activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1
        ),
    ]

    print("\nTraining …")
    model.fit(train_data, validation_data=val_data,
              epochs=EPOCHS, verbose=2, callbacks=callbacks)
    return model


# ── 4. Evaluate on held-out test set ────────────────────────────────────
def evaluate_model(model):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_data = test_datagen.flow_from_directory(
        os.path.join(SPLIT_PATH, "test"),
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=USED_CLASSES,
        shuffle=False,
    )

    print("\nEvaluating on test set …")
    probs = model.predict(test_data, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = test_data.classes

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(USED_CLASSES))))
    report = classification_report(
        y_true, y_pred,
        target_names=[f"{c} ({d})" for c, d in zip(USED_CLASSES, THAI_DIGITS)],
        digits=4, zero_division=0,
    )

    print(f"\nAccuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification report:")
    print(report)
    print("Confusion matrix:")
    print(cm)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "report": report,
    }


# ── 5. Save outputs ─────────────────────────────────────────────────────
def save_outputs(model, metrics):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_path = os.path.join(OUTPUT_DIR, "thai_digit_model.h5")
    model.save(model_path)
    print(f"\nSaved model: {model_path}")

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in metrics.items() if k != "report"},
                  f, indent=2, ensure_ascii=False)
    print(f"Saved metrics: {metrics_path}")

    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy : {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall   : {metrics['recall']:.4f}\n")
        f.write(f"F1-score : {metrics['f1']:.4f}\n\n")
        f.write(metrics["report"])
    print(f"Saved report: {report_path}")

    # Confusion matrix plot
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(USED_CLASSES)))
    ax.set_yticks(range(len(USED_CLASSES)))
    ax.set_xticklabels(THAI_DIGITS)
    ax.set_yticklabels(THAI_DIGITS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=120)
    print(f"Saved confusion matrix: {cm_path}")


# ── Main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_split()
    model = train_model()
    metrics = evaluate_model(model)
    save_outputs(model, metrics)
    print("\nDone.")
