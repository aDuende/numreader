from fastapi import APIRouter, UploadFile, File
from app.services.model_service import predict_number
import os

router = APIRouter(prefix="/api", tags=["Prediction"])

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # Save the last received canvas image for debugging
    os.makedirs("debug_images", exist_ok=True)
    with open("debug_images/last_canvas.png", "wb") as f:
        f.write(image_bytes)

    result = predict_number(image_bytes)

    return {
        "filename": file.filename,
        "prediction": result["prediction"],
        "digits": result["digits"]
    }