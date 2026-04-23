from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
import logging

# Load .env file if present (override=False so real env vars take priority)
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on real environment variables

from utils import normalize_landmarks

# ── Structured logging ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")
logger = logging.getLogger("signbridge_ml")

app = FastAPI(title="SignBridge ASL Classifier API", version="1.0.0")

# ── C.2.1: Restrict CORS to known frontend origins only ─────────────────────
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://signbridgev2.vercel.app,http://localhost:5173,http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# ── C.1.1: Lazy model loading — do NOT load at import time (cold-start safe) ─
MODEL_PATH = "sign_classifier.pkl"
_model = None


def get_model():
    """Lazy-load the model on first inference request, then cache it globally."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            logger.error("Model file not found: %s", MODEL_PATH)
            raise RuntimeError("Model file not found. Run train.py first.")
        logger.info("Loading model from %s …", MODEL_PATH)
        _model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully.")
    return _model


# ── Pydantic schemas ─────────────────────────────────────────────────────────
class LandmarkItem(BaseModel):
    x: float
    y: float
    z: float


class PredictionRequest(BaseModel):
    landmarks: list[LandmarkItem]


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    """Health check — confirms the API is running and whether the model is loaded."""
    model_ready = os.path.exists(MODEL_PATH)
    return {"status": "online", "model_ready": model_ready, "version": "1.0.0"}


@app.post("/classify")
def classify(request: PredictionRequest):
    # C.1.3: Validate landmark count before touching the model
    if len(request.landmarks) != 21:
        raise HTTPException(
            status_code=422,
            detail=f"Expected exactly 21 landmarks, received {len(request.landmarks)}."
        )

    # C.1.1: Lazy-load model (safe for serverless cold starts)
    try:
        model = get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        raw_landmarks = [{"x": l.x, "y": l.y, "z": l.z} for l in request.landmarks]
        features = normalize_landmarks(raw_landmarks)

        prediction = model.predict([features])[0]
        probs = model.predict_proba([features])[0]
        confidence = float(max(probs))

        logger.info("Prediction: %s (confidence=%.4f)", prediction, confidence)

        return {"sign": str(prediction), "confidence": round(confidence, 4)}

    except Exception as exc:
        # C.2.3: Do NOT expose raw exception strings to the client
        logger.error("Prediction error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=400,
            detail="Prediction failed due to invalid input data."
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
