import sys
sys.path.insert(0, ".")
from pathlib import Path
from src.core.onnx_engine import onnx_classifier
from src.core.gradcam import gradcam_engine
from src.core.knowledge_base import knowledge_base

# Load engines
onnx_classifier.load()
gradcam_engine.load()
knowledge_base.load()

print(f"ONNX ready: {onnx_classifier.is_ready}")
print(f"Grad-CAM ready: {gradcam_engine.is_ready}")

test_files = [
    ("amaranth", Path("data/raw/amaranth/amaranth_0000.jpg")),
    ("colocasia_leaves", Path("data/raw/colocasia_leaves/colocasia_leaves_0000.jpg")),
    ("coriander", Path("data/raw/coriander/coriander_0000.jpeg")),
    ("red_amaranth", Path("data/raw/red_amaranth/red_amaranth_0000.jpeg")),
]

for true_label, p in test_files:
    if p.exists():
        img_bytes = p.read_bytes()
        pred_class, conf, latency = onnx_classifier.predict(img_bytes)
        card = knowledge_base.get(pred_class)
        match = "MATCH" if pred_class == true_label else "MISMATCH"
        print(f"True: {true_label:<18} -> Predicted: {pred_class:<18} (conf={conf*100:.1f}%, latency={latency:.1f}ms) [{match}]")
        print(f"   Common name: {card.get('common_name')} | Scientific: {card.get('scientific_name')}")

# Test Grad-CAM
coriander_bytes = test_files[2][1].read_bytes()
heatmap = gradcam_engine.generate_heatmap(coriander_bytes)
print(f"Grad-CAM heatmap generated successfully with shape: {heatmap.shape}")
