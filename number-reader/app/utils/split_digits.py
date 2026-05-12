import cv2
import numpy as np
from PIL import Image
import io


def _crop_to_ink(thresh):
    """Return bounding rect of all ink pixels, or None if blank."""
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h


def _make_tile(thresh_crop):
    """Pad crop to square, resize to 64x64, normalize stroke width, invert.

    thresh_crop: white (255) strokes on black (0) bg (from THRESH_BINARY_INV).
    Training images: dark stroke on white bg with ~20-40% stroke coverage.
    Problem: hand-drawn strokes on a large canvas scale down to nearly invisible
    pixels (mean>230/255) which the model cannot classify. Fix: adaptively dilate
    strokes after resize so coverage is at least 10% of the tile area.
    """
    size = max(thresh_crop.shape[0], thresh_crop.shape[1]) + 20
    canvas = np.zeros((size, size), dtype=np.uint8)
    yo = (size - thresh_crop.shape[0]) // 2
    xo = (size - thresh_crop.shape[1]) // 2
    canvas[yo:yo + thresh_crop.shape[0], xo:xo + thresh_crop.shape[1]] = thresh_crop
    resized = cv2.resize(canvas, (64, 64))

    # Normalize stroke thickness to match training data.
    # resized: white strokes (255) on black (0). coverage = fraction of white pixels.
    # Use a small 2x2 kernel and low target (8%) so open-shaped digits like ๕
    # are NOT over-dilated into a closed circle that looks like ๐.
    coverage = float(np.count_nonzero(resized)) / (64 * 64)
    print(f"[split_digits] tile coverage before norm: {coverage:.3f}")
    if coverage < 0.08:
        kernel = np.ones((2, 2), np.uint8)
        for _ in range(4):
            resized = cv2.dilate(resized, kernel, iterations=1)
            coverage = float(np.count_nonzero(resized)) / (64 * 64)
            if coverage >= 0.08:
                break
    print(f"[split_digits] tile coverage after norm:  {coverage:.3f}")

    # Training data is WHITE digit on BLACK background (mean ~65-130).
    # `resized` is already white-on-black from THRESH_BINARY_INV, so return as-is.
    return resized


def _valley_split(thresh, ink_rect):
    """
    Find the x-column with minimum ink in the middle third of the ink bounding box.
    Works even when digits are drawn touching each other.
    """
    ix, iy, iw, ih = ink_rect
    search_start = ix + iw // 3
    search_end   = ix + 2 * iw // 3

    col_hist = thresh[:, search_start:search_end].sum(axis=0).astype(float)
    # Smooth to avoid noisy local minima
    k = np.ones(7) / 7
    smoothed = np.convolve(col_hist, k, mode='same')

    valley_rel = int(np.argmin(smoothed))
    split_x = search_start + valley_rel
    print(f"[split_digits] valley split at x={split_x}  (searched {search_start}-{search_end})")
    return split_x


def split_digits_from_bytes(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = np.array(image)

    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ink_rect = _crop_to_ink(thresh)
    if ink_rect is None:
        print("[split_digits] blank canvas")
        return []

    ix, iy, iw, ih = ink_rect
    print(f"[split_digits] ink rect x={ix} y={iy} w={iw} h={ih}  ratio={iw/ih:.2f}")

    # ── 1. Try contour-based split ────────────────────────────────────
    # Small kernel: merge within-digit gaps without bridging inter-digit gap
    kernel = np.ones((6, 6), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = sorted(
        [(x, y, w, h) for c in contours
         for x, y, w, h in [cv2.boundingRect(c)]
         if w >= 20 and h >= 20],
        key=lambda b: b[0]
    )
    print(f"[split_digits] contour boxes found: {len(boxes)}")

    # ── 2. Decide whether to valley-split or treat as single digit ────────
    # A single Thai digit is roughly square (width ≈ height).
    # Two digits side-by-side have width > 1.3 × height.
    is_wide_enough_for_two = iw > ih * 1.3

    if is_wide_enough_for_two:
        # Always use valley split when drawing is wide — it's more reliable
        # than contour detection which can merge touching digits.
        print("[split_digits] wide enough → valley split")
        split_x = _valley_split(thresh, ink_rect)
        pad = 10
        y1 = max(iy - pad, 0)
        y2 = min(iy + ih + pad, thresh.shape[0])
        boxes = [
            (max(ix - pad, 0), y1,
             split_x + pad - max(ix - pad, 0), y2 - y1),
            (max(split_x - pad, 0), y1,
             min(ix + iw + pad, thresh.shape[1]) - max(split_x - pad, 0), y2 - y1),
        ]
    elif len(boxes) == 2:
        # Contour split found 2 boxes and drawing isn't wide — use them
        pass
    elif len(boxes) == 1:
        # Single digit drawn — do NOT split
        print("[split_digits] single digit detected")
        boxes = [(max(ix - 10, 0), max(iy - 10, 0),
                  min(iw + 20, thresh.shape[1]), min(ih + 20, thresh.shape[0]))]
    else:
        print("[split_digits] no usable boxes found")
        return []

    # ── 2b. Validate: reject if one box has tiny ink area (e.g. a stray dot) ──
    if len(boxes) == 2:
        ink_areas = []
        for x, y, w, h in boxes:
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + w, thresh.shape[1]), min(y + h, thresh.shape[0])
            ink_areas.append(int(np.count_nonzero(thresh[y1:y2, x1:x2])))
        bigger = max(ink_areas)
        smaller = min(ink_areas)
        ratio = smaller / bigger if bigger > 0 else 0
        print(f"[split_digits] ink areas: {ink_areas}  ratio={ratio:.2f}")
        if ratio < 0.25:
            # One side is just a dot/noise, not a real digit
            print("[split_digits] one side too small → treat as single digit, reject")
            return []

    # ── 3. Build 64x64 tiles ──────────────────────────────────────────────
    digit_images = []
    for x, y, w, h in boxes:
        x1 = max(x, 0);  y1 = max(y, 0)
        x2 = min(x + w, thresh.shape[1]);  y2 = min(y + h, thresh.shape[0])
        crop = thresh[y1:y2, x1:x2]
        digit_images.append(_make_tile(crop))

    print(f"[split_digits] returning {len(digit_images)} tile(s)")
    return digit_images