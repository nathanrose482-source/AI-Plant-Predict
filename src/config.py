"""
Central configuration loaded from environment variables.
All tunable knobs live here — import this module, never os.getenv directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv




# Load .env from repo root (safe no-op if missing)
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parents[1]
MODEL_DIR: Path = ROOT_DIR / "models"
DATA_DIR: Path = ROOT_DIR / "data"
KNOWLEDGE_DIR: Path = DATA_DIR / "knowledge"
STATIC_DIR: Path = ROOT_DIR / "static"

ONNX_MODEL_PATH: Path = MODEL_DIR / "plant_classifier.onnx"
PYTORCH_MODEL_PATH: Path = MODEL_DIR / "plant_classifier.pt"
CLASS_NAMES_PATH: Path = MODEL_DIR / "class_names.json"
KNOWLEDGE_DB_PATH: Path = KNOWLEDGE_DIR / "plant_knowledge.json"
DATASET_CSV_PATH: Path = DATA_DIR / "dataset_master.csv"

def _get_num_classes() -> int:
    if CLASS_NAMES_PATH.exists():
        try:
            import json
            with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
                return len(json.load(f))
        except Exception:
            pass
    return int(os.getenv("NUM_CLASSES", "4"))

NUM_CLASSES: int = _get_num_classes()
IMAGE_SIZE: int = 256          # resize target (shorter edge)
CROP_SIZE: int = 224           # center-crop target
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

# ── Confidence gate ───────────────────────────────────────────────────────────
DEFAULT_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("CONFIDENCE_THRESHOLD", "0.75")
)

# ── Gemini VLM ────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# ── ONNX Runtime ──────────────────────────────────────────────────────────────
ONNX_PROVIDERS: list[str] = ["CPUExecutionProvider"]

# ── Server ────────────────────────────────────────────────────────────────────
APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
