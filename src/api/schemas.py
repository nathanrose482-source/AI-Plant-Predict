"""
Pydantic v2 schemas for all API request/response contracts.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ── Health ────────────────────────────────────────────────────────────────────

class ComponentStatus(BaseModel):
    ready: bool
    detail: str


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str = "1.0.0"
    components: dict[str, ComponentStatus]
    uptime_seconds: float


# ── Classification ────────────────────────────────────────────────────────────

class KnowledgeCard(BaseModel):
    common_name: str
    scientific_name: Optional[str] = None
    family: Optional[str] = None
    description: Optional[str] = None
    edibility: Optional[str] = None
    cultivation: Optional[str] = None
    nutritional_highlights: Optional[str] = None
    traditional_uses: Optional[list[str]] = None
    caution: Optional[str] = None
    fgvc_notes: Optional[str] = None
    extra: Optional[dict[str, Any]] = None


class ClassifyResponse(BaseModel):
    prediction_tier: Literal["tier1_local", "tier2_vlm"]
    predicted_class: str
    raw_confidence: float = Field(..., ge=0.0, le=1.0)
    elapsed_latency_ms: float
    threshold_used: float
    is_ood: bool
    knowledge_card: KnowledgeCard
    all_probabilities: Optional[dict[str, float]] = None
    vlm_scientific_name: Optional[str] = None
    vlm_description: Optional[str] = None


# ── Explain ───────────────────────────────────────────────────────────────────

class ExplainResponse(BaseModel):
    predicted_class: str
    raw_confidence: float
    heatmap_base64: str  # Base64-encoded PNG/JPEG of the blended Grad-CAM overlay
    mime_type: str = "image/jpeg"
    message: str = "Grad-CAM attribution heatmap generated successfully."


# ── Error ─────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str
    error_type: str
