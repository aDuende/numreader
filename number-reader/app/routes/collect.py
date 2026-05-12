from fastapi import APIRouter, UploadFile, File, Form
import os
import time
import cv2
import numpy as np

from typing import Optional

from app.utils.split_digits import split_digits_from_bytes

router = APIRouter(prefix="/api", tags=["Dataset"])

# Map full-number label (thai_0..thai_5) to the units digit folder
ALLOWED_LABELS = ["thai_0", "thai_1", "thai_2", "thai_3", "thai_4", "thai_5"]

# Absolute path to number-reader/dataset/ regardless of CWD
DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dataset")


def _crop_units_digit(image_bytes: bytes) -> Optional[np.ndarray]:
    """Use the same splitter as predict so the saved sample matches what the
    model will see at inference. Returns the rightmost (units) tile."""
    tiles = split_digits_from_bytes(image_bytes)
    if not tiles:
        return None
    # Rightmost tile is the units digit
    return tiles[-1]


@router.post("/collect")
async def collect_sample(
    label: str = Form(...),
    file: UploadFile = File(...)
):
    if label not in ALLOWED_LABELS:
        return {"error": "Invalid label"}

    image_bytes = await file.read()
    tile = _crop_units_digit(image_bytes)

    if tile is None:
        return {"error": "Could not detect digit in image"}

    folder = os.path.join(DATASET_ROOT, label)
    os.makedirs(folder, exist_ok=True)

    filename = f"handdrawn_{int(time.time() * 1000)}.png"
    path = os.path.join(folder, filename)
    # cv2.imwrite silently fails on non-ASCII paths on Windows.
    # Encode to PNG in-memory then write bytes via Python so the Thai
    # characters in the workspace path (เดสก์ท็อป/...) are handled correctly.
    ok, buf = cv2.imencode(".png", tile)
    if not ok:
        return {"error": "Encode failed"}
    with open(path, "wb") as f:
        f.write(buf.tobytes())

    # Count total samples in this folder
    count = len([f for f in os.listdir(folder) if f.endswith(".png")])

    return {
        "message": "saved",
        "label": label,
        "path": path,
        "total_in_class": count,
    }


@router.get("/dataset/stats")
def dataset_stats():
    """Return number of images per class in the dataset folder."""
    stats = {}
    for label in ALLOWED_LABELS:
        folder = os.path.join(DATASET_ROOT, label)
        if os.path.isdir(folder):
            stats[label] = len([f for f in os.listdir(folder) if f.endswith(".png")])
        else:
            stats[label] = 0
    return {"stats": stats}
