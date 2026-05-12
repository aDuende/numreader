import cv2
import os

# Load image
image = cv2.imread("dataset_source/thai_digits_sheet.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold
_, thresh = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)

# Find contours
contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

output_dir = "cropped_digits"
os.makedirs(output_dir, exist_ok=True)

count = 0

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    # Skip tiny noise
    if w < 5 or h < 5:
        continue

    crop = thresh[y:y+h, x:x+w]

    filename = os.path.join(
        output_dir,
        f"digit_{count}.png"
    )

    cv2.imwrite(filename, crop)

    count += 1

print(f"Saved {count} cropped images!")