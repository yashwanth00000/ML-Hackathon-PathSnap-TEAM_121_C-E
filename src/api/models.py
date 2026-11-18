"""
models.py
=========
Pydantic request / response schemas for the PathSnap FastAPI backend.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field



class PredictRequest(BaseModel):
    """Body for POST /predict."""

    trip_id: str = Field(..., description="Evaluator-test trip identifier.")


# ---------------------------------------------------------------------------
# Response sub-schemas
# ---------------------------------------------------------------------------

class ArcGeometry(BaseModel):
    """Geographic endpoints for a single arc (for map rendering)."""

    arc_id: int
    from_lon: float
    from_lat: float
    to_lon: float
    to_lat: float


class RouteMetrics(BaseModel):
    """Evaluation metrics for a predicted route against ground truth."""

    pointwise_accuracy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of GPS observations assigned to the correct arc.",
    )
    route_continuity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of consecutive arc-pairs that are road-adjacent.",
    )
    macro_f1: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Macro-averaged F1 over arc-ID classes.",
    )


# ---------------------------------------------------------------------------
# Primary response schema
# ---------------------------------------------------------------------------

class PredictResponse(BaseModel):
    """Full response returned by POST /predict and GET /predict/{trip_id}."""

    trip_id: str
    predicted_route: List[int] = Field(
        ..., description="Ordered list of arc IDs on the predicted route."
    )
    gps_points: List[List[float]] = Field(
        ..., description="Raw GPS observations as [[lon, lat], ...]."
    )
    arc_geometries: List[ArcGeometry] = Field(
        ..., description="Arc endpoint coordinates for map rendering."
    )
    metrics: RouteMetrics
    inference_time_ms: float = Field(
        ..., ge=0.0, description="End-to-end inference time in milliseconds."
    )


# ---------------------------------------------------------------------------
# Health / info schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    model_name: Optional[str] = None
    num_evaluator_trips: int = 0


class TripsResponse(BaseModel):
    trip_ids: List[str]
