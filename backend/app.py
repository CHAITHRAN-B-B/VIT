from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from inference import predict_image

app = FastAPI(title="AI vs Real Image Detector", version="10.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # restrict in production
    allow_credentials=False,   # must be False when allow_origins="*"
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image upload and returns whether it is AI-generated or Real.

    Response:
        prediction  (str):   "ai" or "real"
        confidence  (float): percentage confidence in the prediction (0–100)
    """
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        label, confidence = predict_image(image)

        return {
            "prediction": label,
            "confidence": confidence,
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
async def health():
    """Simple liveness check."""
    return {"status": "ok", "model_version": "vitv10"}