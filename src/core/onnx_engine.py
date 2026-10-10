"""
ONNX Runtime inference wrapper for plant classification.

Preprocessing pipeline (matches training transforms exactly):
  PIL image → Resize(256, bicubic) → CenterCrop(224) → ToTensor → Normalize(ImageNet)

Target latency: < 15 ms on modern CPU (single-thread).
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from PIL import Image

from src.config import (
    CLASS_NAMES_PATH,
    CROP_SIZE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ONNX_MODEL_PATH,
    ONNX_PROVIDERS,
)

logger = logging.getLogger(__name__)


# ── Image preprocessing ───────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes to a normalised float32 tensor ready for ONNX.

    Returns:
        np.ndarray of shape (1, 3, 224, 224) with dtype float32.

    Raises:
        ValueError: If the bytes cannot be decoded as an image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    # Resize shorter edge to IMAGE_SIZE keeping aspect ratio
    w, h = img.size
    scale = IMAGE_SIZE / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)

    # Center crop to CROP_SIZE
    left = (new_w - CROP_SIZE) // 2
    top = (new_h - CROP_SIZE) // 2
    img = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))

    # HWC → CHW float32 in [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, C)
    arr = arr.transpose(2, 0, 1)                    # (C, H, W)

    # ImageNet normalisation
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    arr = (arr - mean) / std

    return arr[np.newaxis]  # (1, C, H, W)


# ── ONNX session singleton ────────────────────────────────────────────────────

class ONNXClassifier:
    """Thread-safe, lazy-loading ONNX inference engine."""

    _session: Optional[ort.InferenceSession] = None
    _class_names: Optional[list[str]] = None
    _ready: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(
        self,
        model_path: Path = ONNX_MODEL_PATH,
        class_names_path: Path = CLASS_NAMES_PATH,
    ) -> None:
        """Initialise the ONNX session and load class names."""
        if self._ready:
            return

        if not model_path.exists():
            logger.warning("ONNX model not found at %s — Tier-1 disabled.", model_path)
            return

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1   # single-threaded for predictable latency
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=ONNX_PROVIDERS,
        )

        if class_names_path.exists():
            with open(class_names_path, encoding="utf-8") as f:
                self._class_names = json.load(f)
        else:
            logger.warning("class_names.json missing — using integer labels.")
            self._class_names = None

        self._ready = True
        logger.info(
            "ONNX session loaded: %s | classes: %s",
            model_path.name,
            len(self._class_names) if self._class_names else "unknown",
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready and self._session is not None

    @property
    def class_names(self) -> list[str] | None:
        return self._class_names

    def predict(self, image_bytes: bytes) -> tuple[str, float, float]:
        """
        Run a single-image forward pass through the ONNX graph.

        Returns:
            (class_name, confidence, elapsed_ms)

        Raises:
            RuntimeError: If the session is not loaded.
            ValueError: If image decoding fails.
        """
        if not self.is_ready:
            raise RuntimeError("ONNX classifier is not loaded.")

        tensor = preprocess_image(image_bytes)

        input_name = self._session.get_inputs()[0].name  # type: ignore[union-attr]

        t0 = time.perf_counter()
        outputs = self._session.run(None, {input_name: tensor})  # type: ignore[union-attr]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        logits: np.ndarray = outputs[0][0]  # shape (num_classes,)
        # Softmax
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()

        class_idx = int(probs.argmax())
        confidence = float(probs[class_idx])

        if self._class_names and class_idx < len(self._class_names):
            class_name = self._class_names[class_idx]
        else:
            class_name = str(class_idx)

        return class_name, confidence, elapsed_ms

    def predict_all_probs(self, image_bytes: bytes) -> tuple[dict[str, float], float]:
        """
        Return probabilities for all classes along with latency.

        Returns:
            ({class_name: prob, ...}, elapsed_ms)
        """
        if not self.is_ready:
            raise RuntimeError("ONNX classifier is not loaded.")

        tensor = preprocess_image(image_bytes)
        input_name = self._session.get_inputs()[0].name  # type: ignore[union-attr]

        t0 = time.perf_counter()
        outputs = self._session.run(None, {input_name: tensor})  # type: ignore[union-attr]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        logits: np.ndarray = outputs[0][0]
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()

        names = (
            self._class_names
            if self._class_names
            else [str(i) for i in range(len(probs))]
        )
        prob_map = {names[i]: float(probs[i]) for i in range(len(probs))}
        return prob_map, elapsed_ms


# Module-level singleton
onnx_classifier = ONNXClassifier()
