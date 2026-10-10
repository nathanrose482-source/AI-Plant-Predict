"""
FastAPI application entry point.

Start with:  uvicorn src.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.config import APP_HOST, APP_PORT, LOG_LEVEL, STATIC_DIR
from src.core.gradcam import gradcam_engine
from src.core.knowledge_base import knowledge_base
from src.core.onnx_engine import onnx_classifier
from src.core.vlm_fallback import vlm_fallback

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=LOG_LEVEL.upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Load all heavy resources once at startup."""
    logger.info("🌿 Plant Vision — startup sequence initiated")
    onnx_classifier.load()
    gradcam_engine.load()
    knowledge_base.load()
    logger.info(
        "VLM fallback: %s",
        "ready" if vlm_fallback.is_ready else "disabled (GEMINI_API_KEY not set)",
    )
    logger.info("✅ Plant Vision — all engines loaded. Serving on %s:%s", APP_HOST, APP_PORT)
    yield
    logger.info("🌿 Plant Vision — shutting down gracefully.")


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Plant Vision API",
    description=(
        "Edge-optimised plant classification & knowledge retrieval system.\n\n"
        "**Two-tier pipeline:** Local MobileNetV3 ONNX (Tier-1) → Gemini VLM fallback (Tier-2).\n\n"
        "Target plants: Colocasia/Taro · Sorrel · Dill · Fenugreek · Spinach"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── API routes ────────────────────────────────────────────────────────────────
app.include_router(router)

# ── Static frontend ───────────────────────────────────────────────────────────
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
else:
    logger.warning("Static directory %s not found — no frontend served.", STATIC_DIR)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=APP_HOST,
        port=APP_PORT,
        log_level=LOG_LEVEL,
        workers=1,        # Single worker for memory-constrained environments
        reload=False,
    )
