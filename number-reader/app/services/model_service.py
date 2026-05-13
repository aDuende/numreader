import os
import numpy as np
import tensorflow as tf
from app.utils.split_digits import split_digits_from_bytes

ACTIVE_MODEL_PATH = "app/model/thai_digit_model.h5"


def _download_model_if_missing():
    if not os.path.exists(ACTIVE_MODEL_PATH):
        model_url = os.environ.get("MODEL_URL")
        if not model_url:
            raise RuntimeError(
                "Model file not found and MODEL_URL env var is not set. "
                "Set MODEL_URL to the Google Drive shareable link."
            )
        print(f"[model] Downloading model from Google Drive...")
        import gdown
        os.makedirs(os.path.dirname(ACTIVE_MODEL_PATH), exist_ok=True)
        gdown.download(model_url, ACTIVE_MODEL_PATH, fuzzy=True)
        print(f"[model] Model downloaded to {ACTIVE_MODEL_PATH}")


_download_model_if_missing()
model = tf.keras.models.load_model(ACTIVE_MODEL_PATH)


def reload_model():
    """Reload the active .h5 file from disk into memory."""
    global model
    model = tf.keras.models.load_model(ACTIVE_MODEL_PATH)
    return os.path.getmtime(ACTIVE_MODEL_PATH)

# Only classes 0-5 are used (numbers 10-15 need only ๐๑๒๓๔๕ as digit components)
CLASS_MAP = {
    0: "๐",
    1: "๑",
    2: "๒",
    3: "๓",
    4: "๔",
    5: "๕",
}

def predict_single_digit_array(img_array):
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 64, 64, 1)

    prediction = model.predict(img_array, verbose=0)

    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return CLASS_MAP[predicted_class], confidence


def predict_number(image_bytes: bytes):
    digit_images = split_digits_from_bytes(image_bytes)

    if len(digit_images) < 2:
        # Not enough digits split out — return empty so frontend shows warning
        return {"prediction": "", "digits": []}

    # Numbers 10-15 always start with ๑ — hardcode first digit.
    # Only predict the second (units) digit to avoid compounding errors.
    units_img = digit_images[1]
    units_digit, confidence = predict_single_digit_array(units_img)
    print(f"[model] first digit: ๑ (hardcoded)  units: {units_digit}  confidence: {confidence:.2f}")

    final_number = "๑" + units_digit

    return {
        "prediction": final_number,
        "digits": [
            {"digit": "๑", "confidence": 1.0},
            {"digit": units_digit, "confidence": confidence},
        ]
    }