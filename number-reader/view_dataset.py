"""
view_dataset.py  –  Show a grid of sample images from every class in the dataset.
Run from the number-reader folder:
    py view_dataset.py
"""

import os
import random
import cv2
import numpy as np

DATASET_PATH = "dataset"
SAMPLES_PER_CLASS = 10   # how many images to show per class
IMG_SIZE = 64            # display size of each thumbnail
PADDING = 4              # pixels between thumbnails

THAI_LABELS = {
    "thai_0": "๐", "thai_1": "๑", "thai_2": "๒",
    "thai_3": "๓", "thai_4": "๔", "thai_5": "๕",
    "thai_6": "๖", "thai_7": "๗", "thai_8": "๘", "thai_9": "๙",
}

classes = sorted(os.listdir(DATASET_PATH))
rows = []

for cls in classes:
    cls_path = os.path.join(DATASET_PATH, cls)
    files = [f for f in os.listdir(cls_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    sample = random.sample(files, min(SAMPLES_PER_CLASS, len(files)))

    thumbs = []
    for fname in sample:
        img = cv2.imread(os.path.join(cls_path, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # Convert to BGR so we can draw coloured text
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        thumbs.append(img_bgr)

    # Pad to exactly SAMPLES_PER_CLASS columns
    while len(thumbs) < SAMPLES_PER_CLASS:
        thumbs.append(np.full((IMG_SIZE, IMG_SIZE, 3), 240, dtype=np.uint8))

    # Label column (class name)
    label_col = np.full((IMG_SIZE, IMG_SIZE, 3), 30, dtype=np.uint8)
    label = THAI_LABELS.get(cls, cls)
    count = len(files)
    cv2.putText(label_col, label, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 220, 255), 2)
    cv2.putText(label_col, cls, (4, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    cv2.putText(label_col, f"n={count}", (4, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 200, 140), 1)

    row_img = np.hstack([label_col] + [np.hstack([t, np.full((IMG_SIZE, PADDING, 3), 245, dtype=np.uint8)]) for t in thumbs])
    separator = np.full((PADDING, row_img.shape[1], 3), 245, dtype=np.uint8)
    rows.append(row_img)
    rows.append(separator)

grid = np.vstack(rows)

# Header bar
header_h = 36
header = np.full((header_h, grid.shape[1], 3), 20, dtype=np.uint8)
cv2.putText(header, "Dataset viewer  –  Thai digits ๐-๙", (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
grid = np.vstack([header, grid])

print(f"Showing {SAMPLES_PER_CLASS} random samples per class  (press any key to close)")
cv2.imshow("Dataset – Thai digit classes", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()
