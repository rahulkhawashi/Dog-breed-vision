"""
main.py – FastAPI application for Dog Breed Classification.

Endpoints:
  GET  /           → serves static/index.html
  GET  /health     → model health check
  POST /predict    → multipart image upload → top-3 breed predictions
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from breeds import BREED_CLASSES, decode_label
from model_utils import load_keras_model, predict_breed, preprocess_image
from schemas import (
    BreedPrediction,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()  # Load .env if present

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH: str = os.getenv("MODEL_PATH", "./model/dog_breed_model.keras")

# Allowed MIME types for uploaded images
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/tiff",
}

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class AppState:
    model: Optional[object] = None
    model_loaded: bool = False
    breed_info: dict = {}


app_state = AppState()


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model at startup; release resources at shutdown."""
    logger.info("=== Dog Breed Classifier starting up ===")
    logger.info("Model path: %s", MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        logger.warning(
            "Model file not found at '%s'. "
            "Place your .h5 / .keras / .joblib model there before sending requests.",
            MODEL_PATH,
        )
    else:
        try:
            app_state.model = load_keras_model(MODEL_PATH)
            app_state.model_loaded = True
            logger.info("Model loaded and ready.")
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)

    # Try loading breed_info.json
    try:
        with open("breed_info.json", "r", encoding="utf-8") as f:
            app_state.breed_info = json.load(f)
        logger.info("Loaded breed_info.json")
    except Exception as exc:
        logger.warning("Could not load breed_info.json: %s", exc)

    yield  # Application runs here

    logger.info("=== Dog Breed Classifier shutting down ===")
    app_state.model = None
    app_state.model_loaded = False


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Dog Breed Classifier API",
    description=(
        "Upload a dog photo and get the top-3 predicted breeds "
        "from the Stanford Dogs Dataset (120 classes)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS – allow all origins in development; tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static folder (HTML / CSS / JS) – mount AFTER route definitions
# so /predict and /health take priority.
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Return the current health status and whether the model is loaded."""
    return HealthResponse(
        status="ok" if app_state.model_loaded else "degraded",
        model_loaded=app_state.model_loaded,
        model_path=MODEL_PATH,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid image or payload"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    tags=["Inference"],
)
async def predict(
    file: UploadFile = File(..., description="Dog image (JPEG, PNG, WebP …)"),
) -> PredictionResponse:
    """
    Accept a dog image and return the top-3 breed predictions.

    - Validates Content-Type against allowed image MIME types.
    - Preprocesses: open → RGB → 224×224 → normalise → batch tensor.
    - Runs model inference and returns ranked predictions.
    """
    # --- Guard: model must be loaded ---
    if not app_state.model_loaded or app_state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Model is not loaded. "
                f"Ensure the model file exists at '{MODEL_PATH}' and restart the server."
            ),
        )

    # --- Validate MIME type ---
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                detail=f"Unsupported file type '{content_type}'. "
                       f"Please upload a valid image (JPEG, PNG, WebP, etc.).",
                errors=[
                    ErrorDetail(
                        loc=["body", "file"],
                        msg=f"Invalid content type: {content_type}",
                        type="value_error.image_type",
                    )
                ],
            ).model_dump(),
        )

    # --- Read bytes ---
    try:
        image_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to read uploaded file: {exc}",
        )

    # --- Preprocess ---
    try:
        tensor = preprocess_image(image_bytes)
    except ValueError as exc:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                detail=str(exc),
                errors=[
                    ErrorDetail(
                        loc=["body", "file"],
                        msg=str(exc),
                        type="value_error.image_decode",
                    )
                ],
            ).model_dump(),
        )

    # --- Inference ---
    try:
        raw_preds = predict_breed(app_state.model, tensor, top_k=3)
    except RuntimeError as exc:
        err_str = str(exc)
        # Colab-model platform mismatch → return 503 with a clear message
        if "Re-export" in err_str or "dill pickle" in err_str or "EXPORT_MODEL" in err_str:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "The model file cannot be loaded on this platform. "
                    "Please re-export your model from Google Colab using "
                    "model.save('dog_breed_model.keras') and restart the server. "
                    "See EXPORT_MODEL_FROM_COLAB.md for full instructions."
                ),
            )
        logger.exception("Inference error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        )
    except Exception as exc:
        logger.exception("Inference error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        )

    # --- Build response ---
    predictions = [
        BreedPrediction(
            rank=rank,
            breed=decode_label(p["index"]),
            confidence=round(p["confidence"], 6),
            confidence_pct=round(p["confidence"] * 100, 2),
            info=app_state.breed_info.get(decode_label(p["index"]))
        )
        for rank, p in enumerate(raw_preds, start=1)
    ]

    return PredictionResponse(
        predictions=predictions,
        top_breed=predictions[0].breed,
        top_confidence=predictions[0].confidence,
    )


# ---------------------------------------------------------------------------
# Static files – served last so API routes take priority
# ---------------------------------------------------------------------------

if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
else:
    logger.warning("Static directory '%s' not found; frontend will not be served.", STATIC_DIR)


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
