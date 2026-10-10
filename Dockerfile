# syntax=docker/dockerfile:1
# ────────────────────────────────────────────────────────────────────────────
#  Plant Vision — CPU-optimised Docker image
#  Target footprint: < 1 GB RAM  |  Single uvicorn worker
#  Build: docker build -t plant-vision .
#  Run:   docker run -p 8000:8000 --env-file .env plant-vision
# ────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 \
      libgl1 \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python environment ────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── Install Python deps ───────────────────────────────────────────────────────
# CPU-only wheels keep the image lean
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt

# ── Copy source ───────────────────────────────────────────────────────────────
COPY src/       ./src/
COPY static/    ./static/
COPY data/      ./data/
# models/ is volume-mounted at runtime (not baked in to keep image small)
# COPY models/  ./models/

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd -m appuser
RUN mkdir -p /app/models && chown -R appuser:appuser /app
USER appuser

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

EXPOSE 8000

# ── Single-worker uvicorn (memory-constrained) ────────────────────────────────
CMD ["uvicorn", "src.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
