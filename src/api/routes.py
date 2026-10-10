"""
FastAPI route handlers: /api/v1/health, /api/v1/classify, /api/v1/explain
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Annotated

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from src.api.schemas import (
    ClassifyResponse,
    ComponentStatus,
    ErrorResponse,
    ExplainResponse,
    HealthResponse,
    KnowledgeCard,
)
from src.config import DEFAULT_CONFIDENCE_THRESHOLD
from src.core.gradcam import gradcam_engine
from src.core.knowledge_base import knowledge_base
from src.core.onnx_engine import onnx_classifier
from src.core.vlm_fallback import vlm_fallback

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Plant Vision API"])

_start_time = time.time()

# ── Allowed MIME types ────────────────────────────────────────────────────────
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


def _validate_image(upload: UploadFile) -> None:
    """Raise 422 if the uploaded file is not a supported image type."""
    ct = upload.content_type or ""
    if ct not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unsupported content type '{ct}'. "
                f"Accepted: {sorted(_ALLOWED_CONTENT_TYPES)}"
            ),
        )


# ── GET /api/v1/health ────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health and readiness telemetry",
)
async def health() -> HealthResponse:
    onnx_ready = onnx_classifier.is_ready
    cam_ready = gradcam_engine.is_ready
    vlm_ready = vlm_fallback.is_ready

    components = {
        "onnx_engine": ComponentStatus(
            ready=onnx_ready,
            detail="ONNX session active" if onnx_ready else "Model file not found — run scripts/export_onnx.py",
        ),
        "pytorch_gradcam": ComponentStatus(
            ready=cam_ready,
            detail="PyTorch checkpoint loaded" if cam_ready else "Checkpoint missing — run scripts/train.py",
        ),
        "vlm_fallback": ComponentStatus(
            ready=vlm_ready,
            detail="Gemini API reachable" if vlm_ready else "GEMINI_API_KEY not set or SDK missing",
        ),
    }

    ready_count = sum(1 for c in components.values() if c.ready)
    if ready_count == 3:
        overall = "healthy"
    elif ready_count >= 1:
        overall = "degraded"
    else:
        overall = "unhealthy"

    return HealthResponse(
        status=overall,  # type: ignore[arg-type]
        components=components,
        uptime_seconds=time.time() - _start_time,
    )


# ── POST /api/v1/classify ─────────────────────────────────────────────────────

@router.post(
    "/classify",
    response_model=ClassifyResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="Two-tier plant classification with confidence gate",
)
async def classify(
    image: Annotated[UploadFile, File(description="Plant image (JPEG/PNG/WebP)")],
    threshold: Annotated[
        float,
        Form(description="Confidence threshold for Tier-1 acceptance (0–1)", ge=0.0, le=1.0),
    ] = DEFAULT_CONFIDENCE_THRESHOLD,
    force_fallback: Annotated[
        bool,
        Form(description="Skip Tier-1 and route directly to Gemini VLM"),
    ] = False,
) -> ClassifyResponse:
    _validate_image(image)

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Empty image payload.",
        )

    mime = image.content_type or "image/jpeg"
    prediction_tier: str
    predicted_class: str
    raw_confidence: float
    elapsed_ms: float
    is_ood: bool = False

    # ── Path A: Tier-1 local ONNX ────────────────────────────────────────────
    if not force_fallback and onnx_classifier.is_ready:
        try:
            predicted_class, raw_confidence, elapsed_ms = onnx_classifier.predict(image_bytes)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

        logger.info(
            "Tier-1: class=%s conf=%.3f latency=%.1fms threshold=%.2f",
            predicted_class,
            raw_confidence,
            elapsed_ms,
            threshold,
        )

        if raw_confidence >= threshold:
            # ── Accept Tier-1 prediction ──────────────────────────────────────
            prediction_tier = "tier1_local"
            card_data = knowledge_base.get(predicted_class)
            knowledge_card = _build_knowledge_card(card_data)

            return ClassifyResponse(
                prediction_tier="tier1_local",  # type: ignore[arg-type]
                predicted_class=predicted_class,
                raw_confidence=raw_confidence,
                elapsed_latency_ms=elapsed_ms,
                threshold_used=threshold,
                is_ood=False,
                knowledge_card=knowledge_card,
            )

        # Low confidence → fall through to Tier-2
        logger.info(
            "Tier-1 confidence %.3f < threshold %.2f — routing to Tier-2 VLM.",
            raw_confidence,
            threshold,
        )
        is_ood = True

    # ── Path B: Tier-2 Gemini VLM ────────────────────────────────────────────
    if not vlm_fallback.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Tier-1 confidence is below threshold and Gemini VLM fallback "
                "is unavailable. Set GEMINI_API_KEY in your .env file."
            ),
        )

    t0 = time.perf_counter()
    try:
        vlm_result = vlm_fallback.identify(image_bytes, mime_type=mime)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"VLM fallback failed: {exc}",
        ) from exc

    vlm_elapsed_ms = (time.perf_counter() - t0) * 1000.0

    plant_name: str = vlm_result.get("plant_name", "Unknown Plant")
    scientific_name: str | None = vlm_result.get("scientific_name")
    vlm_confidence: float = float(vlm_result.get("confidence", 0.5))
    description: str | None = vlm_result.get("description")
    is_plant: bool = bool(vlm_result.get("is_plant", True))
    ood_reason: str = vlm_result.get("ood_reason", "")

    # Build knowledge card from VLM output + any local KB data
    local_card = knowledge_base.get(plant_name)
    vlm_card = KnowledgeCard(
        common_name=plant_name,
        scientific_name=scientific_name or local_card.get("scientific_name"),
        description=description or local_card.get("description"),
        family=local_card.get("family"),
        edibility=local_card.get("edibility"),
        cultivation=local_card.get("cultivation"),
        nutritional_highlights=local_card.get("nutritional_highlights"),
        traditional_uses=local_card.get("traditional_uses"),
        caution=local_card.get("caution"),
        fgvc_notes=local_card.get("fgvc_notes"),
        extra={"ood_reason": ood_reason} if ood_reason else None,
    )

    return ClassifyResponse(
        prediction_tier="tier2_vlm",  # type: ignore[arg-type]
        predicted_class=plant_name,
        raw_confidence=vlm_confidence,
        elapsed_latency_ms=vlm_elapsed_ms,
        threshold_used=threshold,
        is_ood=not is_plant,
        knowledge_card=vlm_card,
        vlm_scientific_name=scientific_name,
        vlm_description=description,
    )


# ── POST /api/v1/explain ──────────────────────────────────────────────────────

@router.post(
    "/explain",
    response_model=ExplainResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="Grad-CAM saliency heatmap for model interpretability",
)
async def explain(
    image: Annotated[UploadFile, File(description="Plant image (JPEG/PNG/WebP)")],
) -> ExplainResponse:
    _validate_image(image)

    if not gradcam_engine.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Grad-CAM engine not loaded. "
                "Ensure scripts/train.py has been run and models/plant_classifier.pt exists."
            ),
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Empty image payload.",
        )

    try:
        heatmap_bgr = gradcam_engine.generate_heatmap(image_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    # Also run ONNX for metadata (best-effort)
    predicted_class = "unknown"
    raw_confidence = 0.0
    if onnx_classifier.is_ready:
        try:
            predicted_class, raw_confidence, _ = onnx_classifier.predict(image_bytes)
        except Exception:
            pass

    # Encode heatmap as JPEG
    success, encoded = cv2.imencode(".jpg", heatmap_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to encode heatmap image.",
        )

    b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

    return ExplainResponse(
        predicted_class=predicted_class,
        raw_confidence=raw_confidence,
        heatmap_base64=b64,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_knowledge_card(data: dict) -> KnowledgeCard:
    """Convert a raw knowledge-base dict to a KnowledgeCard schema."""
    return KnowledgeCard(
        common_name=data.get("common_name", "Unknown"),
        scientific_name=data.get("scientific_name"),
        family=data.get("family"),
        description=data.get("description"),
        edibility=data.get("edibility"),
        cultivation=data.get("cultivation"),
        nutritional_highlights=data.get("nutritional_highlights"),
        traditional_uses=data.get("traditional_uses"),
        caution=data.get("caution"),
        fgvc_notes=data.get("fgvc_notes"),
    )
