"""
Grad-CAM explainability module backed by native PyTorch.

Hooks onto the last convolutional feature layer of MobileNetV3-Small
to produce a normalised heatmap that is blended over the original image.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.config import (
    CLASS_NAMES_PATH,
    CROP_SIZE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PYTORCH_MODEL_PATH,
)

logger = logging.getLogger(__name__)

# ── Preprocessing (must mirror ONNX preprocessing) ───────────────────────────
_transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


# ── Hook storage ──────────────────────────────────────────────────────────────

class _GradCamHooks:
    """Attach forward/backward hooks to a target layer."""

    def __init__(self) -> None:
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles: list = []

    def register(self, layer: nn.Module) -> None:
        self._handles.append(
            layer.register_forward_hook(self._save_activation)
        )
        self._handles.append(
            layer.register_full_backward_hook(self._save_gradient)
        )

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _save_activation(
        self, _module: nn.Module, _input: tuple, output: torch.Tensor
    ) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self, _module: nn.Module, _grad_input: tuple, grad_output: tuple
    ) -> None:
        self.gradients = grad_output[0].detach()


# ── Grad-CAM engine ───────────────────────────────────────────────────────────

class GradCAMEngine:
    """
    Loads a saved PyTorch MobileNetV3-Small checkpoint and generates
    Grad-CAM heatmaps blended over the input image.
    """

    _model: Optional[nn.Module] = None
    _target_layer: Optional[nn.Module] = None
    _ready: bool = False

    def load(
        self,
        model_path: Path = PYTORCH_MODEL_PATH,
        class_names_path: Path = CLASS_NAMES_PATH,
        num_classes: Optional[int] = None,
    ) -> None:
        """Load a saved checkpoint; graceful degradation if missing."""
        if self._ready:
            return

        if not model_path.exists():
            logger.warning(
                "PyTorch model not found at %s — Grad-CAM disabled.", model_path
            )
            return

        import json

        if class_names_path.exists():
            with open(class_names_path, encoding="utf-8") as f:
                class_names: list[str] = json.load(f)
            n_classes = len(class_names)
        else:
            n_classes = num_classes or 5

        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        model = mobilenet_v3_small(weights=None)
        # Replace classifier head to match training
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, n_classes)

        state = torch.load(model_path, map_location="cpu", weights_only=True)
        # Support bare state_dict or {'model_state_dict': ...} checkpoint dicts
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval()

        self._model = model
        # MobileNetV3-Small: last conv is features[-1][0]
        self._target_layer = model.features[-1][0]
        self._ready = True
        logger.info("PyTorch Grad-CAM engine loaded from %s", model_path.name)

    @property
    def is_ready(self) -> bool:
        return self._ready and self._model is not None

    # ── Core Grad-CAM logic ───────────────────────────────────────────────────

    def generate_heatmap(self, image_bytes: bytes) -> np.ndarray:
        """
        Produce a Grad-CAM heatmap blended over the input image.

        Returns:
            np.ndarray — BGR uint8 image (H, W, 3) with jet colourmap overlay.

        Raises:
            RuntimeError: If PyTorch engine is not loaded.
            ValueError: If image bytes are invalid.
        """
        if not self.is_ready:
            raise RuntimeError("Grad-CAM engine is not loaded.")

        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot decode image: {exc}") from exc

        tensor: torch.Tensor = _transform(pil_img).unsqueeze(0)  # (1, C, H, W)

        hooks = _GradCamHooks()
        hooks.register(self._target_layer)  # type: ignore[arg-type]

        # Forward pass
        self._model.zero_grad()  # type: ignore[union-attr]
        logits: torch.Tensor = self._model(tensor)  # type: ignore[union-attr]
        pred_class = logits.argmax(dim=1).item()

        # Backward pass w.r.t. predicted class score
        score = logits[0, pred_class]
        score.backward()

        hooks.remove()

        if hooks.activations is None or hooks.gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients.")

        # GAP over spatial dims → channel weights
        weights = hooks.gradients.squeeze(0).mean(dim=(1, 2))  # (C,)
        cams: torch.Tensor = hooks.activations.squeeze(0)      # (C, h, w)

        cam = torch.relu((weights[:, None, None] * cams).sum(dim=0))  # (h, w)

        # Normalise
        cam_np = cam.numpy()
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()

        # Up-sample to CROP_SIZE × CROP_SIZE
        cam_resized = cv2.resize(cam_np, (CROP_SIZE, CROP_SIZE))
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )  # BGR

        # Blend with original image (centre-cropped)
        orig = np.array(
            pil_img.resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC), dtype=np.uint8
        )
        orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
        blended = cv2.addWeighted(orig_bgr, 0.55, heatmap, 0.45, 0)

        return blended


# Module-level singleton
gradcam_engine = GradCAMEngine()
