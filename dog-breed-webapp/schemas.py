"""
Pydantic v2 request / response schemas for the Dog Breed Classification API.
"""

from typing import List
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class BreedPrediction(BaseModel):
    """A single breed prediction entry."""

    rank: int = Field(..., ge=1, description="Rank of the prediction (1 = top match).")
    breed: str = Field(..., description="Human-readable breed name.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Softmax confidence score in [0, 1]."
    )
    confidence_pct: float = Field(
        ..., ge=0.0, le=100.0, description="Confidence as a percentage (0–100)."
    )
    info: dict | None = Field(
        None, description="Detailed information or specification about the breed."
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    """Successful /predict response."""

    success: bool = True
    predictions: List[BreedPrediction]
    top_breed: str = Field(..., description="The highest-confidence breed name.")
    top_confidence: float = Field(..., description="Confidence of the top prediction.")


class HealthResponse(BaseModel):
    """Response for the /health endpoint."""

    status: str
    model_loaded: bool
    model_path: str


class ErrorDetail(BaseModel):
    """Single error entry, mirrors FastAPI's default 422 structure."""

    loc: List[str] = Field(default_factory=list)
    msg: str
    type: str


class ErrorResponse(BaseModel):
    """Structured error response body."""

    success: bool = False
    detail: str
    errors: List[ErrorDetail] = Field(default_factory=list)
