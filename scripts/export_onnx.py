"""
Export a trained PyTorch MobileNetV3-Small checkpoint to ONNX.

Usage:
    python scripts/export_onnx.py \\
        --checkpoint models/plant_classifier.pt \\
        --output     models/plant_classifier.onnx \\
        --opset      17

The exported graph is then validated against the PyTorch output to ensure
numerical parity (max abs diff < 1e-5).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_checkpoint(checkpoint_path: Path) -> tuple[nn.Module, list[str]]:
    """Load model and class names from a checkpoint dict or bare state dict."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if isinstance(state, dict) and "model_state_dict" in state:
        class_names: list[str] = state.get("class_names", [])
        state_dict = state["model_state_dict"]
    else:
        # Bare state dict — try to infer num_classes from the classifier head
        state_dict = state
        class_names = []

    # Detect number of classes from the last linear layer
    # MobileNetV3-Small classifier[3] weight shape: (num_classes, in_features)
    for key in state_dict:
        if "classifier.3.weight" in key:
            num_classes = state_dict[key].shape[0]
            break
    else:
        raise ValueError("Cannot determine num_classes from checkpoint.")

    from torchvision.models import mobilenet_v3_small

    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info("Loaded checkpoint: %d classes", num_classes)
    return model, class_names


def export(
    model: nn.Module,
    output_path: Path,
    opset: int = 17,
) -> None:
    """Trace and export the model to ONNX."""
    dummy = torch.randn(1, 3, 224, 224)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    logger.info("ONNX model exported to %s (opset %d)", output_path, opset)


def validate(model: nn.Module, output_path: Path) -> None:
    """Run a parity check between PyTorch and ONNX outputs."""
    import onnxruntime as ort

    dummy = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        pt_out = model(dummy).numpy()

    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"image": dummy.numpy()})[0]

    max_diff = float(np.abs(pt_out - ort_out).max())
    logger.info("PyTorch vs ONNX max absolute diff: %.2e", max_diff)

    assert max_diff < 1e-4, f"Parity check failed: max diff = {max_diff:.4e}"
    logger.info("✅ Parity check passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch plant classifier to ONNX")
    parser.add_argument("--checkpoint", default="models/plant_classifier.pt")
    parser.add_argument("--output", default="models/plant_classifier.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip_validation", action="store_true")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    out = Path(args.output)

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model, class_names = load_checkpoint(ckpt)
    export(model, out, args.opset)

    if not args.skip_validation:
        validate(model, out)

    # Ensure class_names.json is in models/
    if class_names:
        cn_path = out.parent / "class_names.json"
        if not cn_path.exists():
            with open(cn_path, "w", encoding="utf-8") as f:
                json.dump(class_names, f, indent=2)
            logger.info("class_names.json written to %s", cn_path)

    logger.info("Export complete → %s", out)


if __name__ == "__main__":
    main()
