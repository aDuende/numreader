import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import numpy as np
import cv2

IMG_SIZE = 64
BATCH_SIZE = 32
DATASET_PATH = "dataset"

# Numbers 10-15 only use digits 0-5 as components.
# Exclude thai_6 through thai_9 so the model only learns what it needs.
USED_CLASSES = ["thai_0", "thai_1", "thai_2", "thai_3", "thai_4", "thai_5"]


def augment_stroke_width(image):
    """
    Randomly dilate (thicken) the digit strokes to simulate canvas drawings.
    NOTE: preprocessing_function is called BEFORE rescale, so image is in [0,255] float32.
    After training, inference normalizes stroke coverage to 15%+ before prediction,
    so we train with similarly thick strokes (kernel 2-8px, larger than before).
    """
    img = image[:, :, 0].astype(np.uint8)
    # Invert: white on black so dilation expands the stroke
    inv = cv2.bitwise_not(img)
    # Random kernel 1-3: widens strokes without closing open shapes (e.g. ๕, ๒)
    k = np.random.randint(1, 4)
    kernel = np.ones((k, k), np.uint8)
    dilated = cv2.dilate(inv, kernel, iterations=1)
    result = cv2.bitwise_not(dilated).astype(np.float32)
    return result[:, :, np.newaxis]


train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    preprocessing_function=augment_stroke_width,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=USED_CLASSES,
    subset="training"
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=USED_CLASSES,
    subset="validation"
)

model = models.Sequential([
    layers.Input(shape=(64, 64, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=35
)

model.save("app/model/thai_digit_model.h5")

print("Thai digit model saved!")
print("Classes:", train_data.class_indices)