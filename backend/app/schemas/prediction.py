"""Schemas for health checks and transparent classification responses."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.app.schemas.inventory import PartRecord


class HealthResponse(BaseModel):
    """Report inference readiness and the active knowledge-data source."""

    status: str
    model_loaded: bool
    catalog_loaded: bool
    neo4j_configured: bool
    neo4j_connected: bool
    knowledge_source: Literal["neo4j", "static_catalog_fallback"]
    neo4j_message: str


class PredictionResult(BaseModel):
    """Represent the backend-facing output of the inference service."""

    model_config = ConfigDict(extra="forbid")

    predicted_class: str
    most_likely_class: str
    class_index: int | None
    confidence: float = Field(ge=0.0, le=1.0)
    requires_manual_review: bool
    message: str


class PredictionResponse(BaseModel):
    """Represent the complete transparent response returned to the UI."""

    prediction_id: str
    filename: str
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predicted_class: str
    most_likely_class: str
    display_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    requires_manual_review: bool
    feedback_applied: bool
    message: str
    repair_steps: list[str]
    parts: list[PartRecord]
    inventory_last_updated: str
    inventory_is_simulated: bool
    knowledge_source: Literal["neo4j", "static_catalog_fallback"]
    poc_disclaimer: str
