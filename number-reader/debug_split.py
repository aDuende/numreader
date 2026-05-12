"""
debug_split.py  –  Diagnose digit splitting pipeline.
Draw a number in the web UI, click Predict, then immediately run this script.
It saves debug images so you can see exactly what the model receives.

Run from number-reader folder:
    py debug_split.py <path_to_image.png>
Or drop any PNG as argument.
"""

import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import io

# ── helpers ──────────────────────────────────────────────────────────────────

def merge_nearby_boxes(boxes, gap_threshold=15):
    if not boxes:
        return boxes
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = [list(boxes[0])]
    for x, y, w, h in boxes[1:]:
        prev = merged[-1]
        px, py, pw, ph = prev
        if x - (px + pw) <= gap_threshold:
            merged[-1] = [min(px, x), min(py, y),
                          max(px+pw, x+w) - min(px, x),
                          max(py+ph, y+h) - min(py, y)]
        else:
            merged.append([x, y, w, h])
    return [tuple(b) for b in merged]


def run_split_debug(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Test multiple kernel sizes to find one that merges within-digit gaps
    # but does NOT bridge the gap between two adjacent digits
    for ksize, iters in [(6,2), (8,2), (10,3), (12,3)]:
        kernel = np.ones((ksize, ksize), np.uint8)
        d = cv2.dilate(thresh, kernel, iterations=iters)
        cnts, _ = cv2.findContours(d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bxs = [(x,y,w,h) for c in cnts for x,y,w,h in [cv2.boundingRect(c)] if w>=20 and h>=20]
        print(f"  kernel={ksize}x{ksize} iters={iters}  ->  {len(bxs)} box(es)")

    # Use production kernel (10x10, 3 iter)
    kernel = np.ones((10, 10), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= 20 and h >= 20:
            boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: b[0])

    print(f"\n  FINAL boxes : {len(boxes)} digit(s)")
    for i, (x, y, w, h) in enumerate(boxes):
        print(f"    box[{i}]  x={x} y={y} w={w} h={h}")

    # ── annotated overview ──────────────────────────────────────────────────
    overview = img_bgr.copy()
    colors = [(0,200,0),(0,100,255),(255,80,0),(180,0,180)]
    for i, (x, y, w, h) in enumerate(merged):
        pad = 8
        x1, y1 = max(x-pad,0), max(y-pad,0)
        x2, y2 = min(x+w+pad, img_bgr.shape[1]), min(y+h+pad, img_bgr.shape[0])
        col = colors[i % len(colors)]
        cv2.rectangle(overview, (x1, y1), (x2, y2), col, 2)
        cv2.putText(overview, f"#{i}", (x1+2, y1+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    cv2.imwrite("debug_overview.png", overview)
    print("\n  Saved: debug_overview.png  (bounding boxes on original)")

    # ── individual crops (what model sees) ─────────────────────────────────
    crops = []
    for i, (x, y, w, h) in enumerate(merged):
        pad = 8
        x1, y1 = max(x-pad,0), max(y-pad,0)
        x2, y2 = min(x+w+pad, thresh.shape[1]), min(y+h+pad, thresh.shape[0])
        crop = thresh[y1:y2, x1:x2]
        size = max(crop.shape[0], crop.shape[1]) + 20
        canvas = np.zeros((size, size), dtype=np.uint8)
        yo = (size - crop.shape[0]) // 2
        xo = (size - crop.shape[1]) // 2
        canvas[yo:yo+crop.shape[0], xo:xo+crop.shape[1]] = crop
        canvas64 = cv2.resize(canvas, (64, 64))
        crops.append(canvas64)
        out = cv2.resize(canvas64, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f"debug_crop_{i}.png", out)
        print(f"  Saved: debug_crop_{i}.png  (what model sees for digit #{i})")

    # ── run model predictions ───────────────────────────────────────────────
    try:
        import tensorflow as tf
        import numpy as npp
        print("\n  Loading model …")
        model = tf.keras.models.load_model("app/model/thai_digit_model.h5")
        CLASS_MAP = {0:"๐",1:"๑",2:"๒",3:"๓",4:"๔",5:"๕",6:"๖",7:"๗",8:"๘",9:"๙"}
        print("\n  Model predictions:")
        for i, c in enumerate(crops):
            arr = c.astype("float32") / 255.0
            arr = arr.reshape(1, 64, 64, 1)
            probs = model.predict(arr, verbose=0)[0]
            top3 = npp.argsort(probs)[::-1][:3]
            print(f"    crop[{i}]:  " +
                  "  |  ".join(f"{CLASS_MAP[k]} {probs[k]*100:.1f}%" for k in top3))
    except Exception as e:
        print(f"\n  (Could not run model: {e})")

    print("\n  Done. Open debug_overview.png and debug_crop_*.png to inspect.\n")


# ── main ──────────────────────────────────────────────────────────────────────

if len(sys.argv) > 1:
    path = sys.argv[1]
    img = cv2.imread(path)
    if img is None:
        print(f"Cannot read image: {path}")
        sys.exit(1)
    print(f"\nAnalysing: {path}")
else:
    # Create a synthetic test image: draw ๑๐ side by side
    print("\nNo image supplied – generating a synthetic ๑๐ test image …")
    pil = Image.new("RGB", (360, 200), "white")
    from PIL import ImageFont
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/Tahoma.ttf", 140)
    except:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(pil)
    d.text((10, 10), "๑๐", font=font, fill="black")
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite("debug_synthetic.png", img)
    print("  Saved: debug_synthetic.png")

run_split_debug(img)
