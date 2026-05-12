"""Model management endpoints: list, upload, activate, metrics."""
import os
import io
import base64
import json
import shutil
import datetime
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services import model_service

router = APIRouter(prefix="/api/models", tags=["Models"])

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "app", "model"
)
ACTIVE_FILE = os.path.join(MODEL_DIR, "thai_digit_model.h5")
ACTIVE_POINTER = os.path.join(MODEL_DIR, "active_model.txt")
ALLOWED_EXT = (".h5",)  # only Keras .h5 supported by current loader


def _get_active_name() -> Optional[str]:
    if os.path.exists(ACTIVE_POINTER):
        try:
            with open(ACTIVE_POINTER, "r", encoding="utf-8") as f:
                name = f.read().strip()
            if name and os.path.exists(os.path.join(MODEL_DIR, name)):
                return name
        except Exception:
            pass
    return None


@router.get("")
def list_models():
    items = []
    active = _get_active_name()
    for fn in sorted(os.listdir(MODEL_DIR)):
        if not fn.lower().endswith(ALLOWED_EXT):
            continue
        if fn == "thai_digit_model.h5":
            # this is the active file mirror, skip from the catalog
            continue
        full = os.path.join(MODEL_DIR, fn)
        items.append({
            "name": fn,
            "size": os.path.getsize(full),
            "uploaded": datetime.datetime.fromtimestamp(
                os.path.getmtime(full)
            ).strftime("%Y-%m-%d %H:%M"),
            "active": (fn == active),
        })
    return {"models": items, "active": active}


@router.post("/upload")
async def upload_model(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(ALLOWED_EXT):
        raise HTTPException(400, "Only .h5 files are supported")
    safe_name = os.path.basename(file.filename)
    dest = os.path.join(MODEL_DIR, safe_name)
    if os.path.exists(dest):
        # add timestamp suffix to avoid overwrite
        stem, ext = os.path.splitext(safe_name)
        safe_name = f"{stem}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        dest = os.path.join(MODEL_DIR, safe_name)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"success": True, "name": safe_name}


@router.post("/activate")
def activate_model(payload: dict):
    name = payload.get("name", "")
    if not name or os.path.basename(name) != name:
        raise HTTPException(400, "Invalid name")
    src = os.path.join(MODEL_DIR, name)
    if not os.path.exists(src):
        raise HTTPException(404, "Model not found")
    # validate it loads before swapping
    import tensorflow as tf
    try:
        tf.keras.models.load_model(src)
    except Exception as e:
        raise HTTPException(400, f"Cannot load model: {e}")
    shutil.copyfile(src, ACTIVE_FILE)
    with open(ACTIVE_POINTER, "w", encoding="utf-8") as f:
        f.write(name)
    model_service.reload_model()
    return {"success": True, "active": name}


@router.delete("/{name}")
def delete_model(name: str):
    if os.path.basename(name) != name or not name.lower().endswith(ALLOWED_EXT):
        raise HTTPException(400, "Invalid name")
    if name == "thai_digit_model.h5":
        raise HTTPException(400, "Cannot delete the active model file")
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(404, "Not found")
    os.remove(path)
    if _get_active_name() == name:
        os.remove(ACTIVE_POINTER)
    return {"success": True}


@router.get("/metrics")
def get_metrics():
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        return {"metrics": None}
    with open(metrics_path, "r", encoding="utf-8") as f:
        return {"metrics": json.load(f)}


# Order must match training: thai_0 .. thai_5  ->  ๐ ๑ ๒ ๓ ๔ ๕
USED_CLASSES = ["thai_0", "thai_1", "thai_2", "thai_3", "thai_4", "thai_5"]
THAI_DIGITS = ["๐", "๑", "๒", "๓", "๔", "๕"]
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TEST_DIR = os.path.join(REPO_ROOT, "dataset_split", "test")


@router.get("/misclassified")
def get_misclassified(limit: int = 12):
    """Run the active model over the held-out test set and return
    misclassified samples (image + true/predicted labels + confidence)."""
    import numpy as np
    import cv2
    from PIL import Image

    if not os.path.isdir(TEST_DIR):
        return {"items": [], "total": 0, "message": "test split not found"}

    items = []
    for cls_idx, cls in enumerate(USED_CLASSES):
        cls_dir = os.path.join(TEST_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fn in sorted(os.listdir(cls_dir)):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(cls_dir, fn)
            # PIL handles Unicode paths on Windows; cv2.imread does not.
            try:
                img = np.array(Image.open(path).convert("L"))
            except Exception:
                continue
            if img.shape != (64, 64):
                img = cv2.resize(img, (64, 64))

            pred_digit, conf = model_service.predict_single_digit_array(
                img.astype(np.float32)
            )
            if pred_digit == THAI_DIGITS[cls_idx]:
                continue  # correctly classified

            # Encode as base64 PNG for the frontend
            ok, buf = cv2.imencode(".png", img)
            if not ok:
                continue
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")

            items.append({
                "file": fn,
                "actual": THAI_DIGITS[cls_idx],
                "predicted": pred_digit,
                "confidence": round(conf * 100, 1),
                "image": f"data:image/png;base64,{b64}",
            })

    # Sort worst (highest-confidence mistakes) first, then truncate
    items.sort(key=lambda x: -x["confidence"])
    return {"items": items[:max(0, int(limit))], "total": len(items)}
