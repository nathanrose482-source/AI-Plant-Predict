"""
Pytest test suite for the Plant Vision API.

Tests cover:
  - Health endpoint contract
  - Classify endpoint: valid image, low-confidence gate, force_fallback, bad input
  - Explain endpoint: valid image, missing model, bad input
  - ONNX preprocessing correctness
  - Knowledge base retrieval
"""

from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_image_bytes(width: int = 256, height: int = 256, colour: str = "green") -> bytes:
    """Generate a synthetic PIL image as JPEG bytes."""
    img = Image.new("RGB", (width, height), colour)
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def green_image() -> bytes:
    return _make_image_bytes(colour="green")


@pytest.fixture(scope="session")
def small_image() -> bytes:
    """Very small image — should still work after resize."""
    return _make_image_bytes(width=64, height=64, colour="darkgreen")


@pytest.fixture(scope="session")
def corrupt_bytes() -> bytes:
    return b"not-an-image-at-all-xxxx"


# ── App client with mocked engines ────────────────────────────────────────────

@pytest.fixture()
def client_no_models():
    """TestClient with all engines reporting not-ready."""
    from src.main import app

    with patch("src.core.onnx_engine.onnx_classifier") as mock_onnx, \
         patch("src.core.gradcam.gradcam_engine") as mock_cam, \
         patch("src.core.vlm_fallback.vlm_fallback") as mock_vlm:

        mock_onnx.is_ready = False
        mock_cam.is_ready = False
        mock_vlm.is_ready = False

        with TestClient(app, raise_server_exceptions=False) as tc:
            yield tc


@pytest.fixture()
def client_tier1_ready(green_image):
    """TestClient with ONNX returning high-confidence spinach."""
    from src.main import app

    with patch("src.api.routes.onnx_classifier") as mock_onnx, \
         patch("src.api.routes.gradcam_engine") as mock_cam, \
         patch("src.api.routes.vlm_fallback") as mock_vlm:

        mock_onnx.is_ready = True
        mock_onnx.predict.return_value = ("spinach", 0.92, 8.3)
        mock_cam.is_ready = False
        mock_vlm.is_ready = False

        with TestClient(app) as tc:
            yield tc


@pytest.fixture()
def client_low_confidence(green_image):
    """TestClient: ONNX returns low confidence → falls to VLM."""
    from src.main import app

    vlm_result = {
        "plant_name": "Spinach",
        "scientific_name": "Spinacia oleracea",
        "confidence": 0.85,
        "description": "Smooth dark-green leafy vegetable.",
        "is_plant": True,
        "ood_reason": "",
    }

    with patch("src.api.routes.onnx_classifier") as mock_onnx, \
         patch("src.api.routes.gradcam_engine") as mock_cam, \
         patch("src.api.routes.vlm_fallback") as mock_vlm:

        mock_onnx.is_ready = True
        mock_onnx.predict.return_value = ("spinach", 0.40, 10.1)
        mock_cam.is_ready = False
        mock_vlm.is_ready = True
        mock_vlm.identify.return_value = vlm_result

        with TestClient(app) as tc:
            yield tc


@pytest.fixture()
def client_gradcam_ready(green_image):
    """TestClient with Grad-CAM returning a fake heatmap."""
    from src.main import app

    # Fake BGR heatmap (224×224×3)
    fake_heatmap = np.zeros((224, 224, 3), dtype=np.uint8)
    fake_heatmap[:, :, 1] = 255  # green

    with patch("src.api.routes.onnx_classifier") as mock_onnx, \
         patch("src.api.routes.gradcam_engine") as mock_cam, \
         patch("src.api.routes.vlm_fallback") as mock_vlm:

        mock_onnx.is_ready = True
        mock_onnx.predict.return_value = ("fenugreek_leaves", 0.88, 9.0)
        mock_cam.is_ready = True
        mock_cam.generate_heatmap.return_value = fake_heatmap
        mock_vlm.is_ready = False

        with TestClient(app) as tc:
            yield tc


# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client_no_models):
        r = client_no_models.get("/api/v1/health")
        assert r.status_code == 200

    def test_health_schema_keys(self, client_no_models):
        data = client_no_models.get("/api/v1/health").json()
        assert "status" in data
        assert "components" in data
        assert "uptime_seconds" in data
        assert "version" in data

    def test_health_components_present(self, client_no_models):
        data = client_no_models.get("/api/v1/health").json()
        comps = data["components"]
        assert "onnx_engine" in comps
        assert "pytorch_gradcam" in comps
        assert "vlm_fallback" in comps

    def test_health_unhealthy_when_no_models(self, client_no_models):
        data = client_no_models.get("/api/v1/health").json()
        assert data["status"] in {"unhealthy", "degraded"}

    def test_health_status_is_valid_literal(self, client_no_models):
        data = client_no_models.get("/api/v1/health").json()
        assert data["status"] in {"healthy", "degraded", "unhealthy"}

    def test_uptime_increases(self, client_no_models):
        r1 = client_no_models.get("/api/v1/health").json()
        time.sleep(0.05)
        r2 = client_no_models.get("/api/v1/health").json()
        assert r2["uptime_seconds"] >= r1["uptime_seconds"]


# ── Classify endpoint ─────────────────────────────────────────────────────────

class TestClassify:
    def test_classify_tier1_high_confidence(self, client_tier1_ready, green_image):
        r = client_tier1_ready.post(
            "/api/v1/classify",
            files={"image": ("leaf.jpg", green_image, "image/jpeg")},
            data={"threshold": "0.75"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["prediction_tier"] == "tier1_local"
        assert body["predicted_class"] == "spinach"
        assert body["raw_confidence"] == pytest.approx(0.92, abs=1e-4)
        assert body["elapsed_latency_ms"] > 0
        assert "knowledge_card" in body
        assert body["knowledge_card"]["common_name"] is not None

    def test_classify_low_confidence_routes_to_vlm(self, client_low_confidence, green_image):
        r = client_low_confidence.post(
            "/api/v1/classify",
            files={"image": ("leaf.jpg", green_image, "image/jpeg")},
            data={"threshold": "0.75"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["prediction_tier"] == "tier2_vlm"
        assert body["raw_confidence"] == pytest.approx(0.85, abs=1e-4)

    def test_classify_force_fallback(self, client_low_confidence, green_image):
        r = client_low_confidence.post(
            "/api/v1/classify",
            files={"image": ("leaf.jpg", green_image, "image/jpeg")},
            data={"threshold": "0.75", "force_fallback": "true"},
        )
        assert r.status_code == 200
        assert r.json()["prediction_tier"] == "tier2_vlm"

    def test_classify_returns_knowledge_card_fields(self, client_tier1_ready, green_image):
        body = client_tier1_ready.post(
            "/api/v1/classify",
            files={"image": ("leaf.jpg", green_image, "image/jpeg")},
        ).json()
        card = body["knowledge_card"]
        # knowledge_card must have at least common_name
        assert "common_name" in card

    def test_classify_rejects_non_image(self, client_tier1_ready):
        r = client_tier1_ready.post(
            "/api/v1/classify",
            files={"image": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert r.status_code == 422

    def test_classify_rejects_corrupt_bytes(self, client_tier1_ready, corrupt_bytes):
        with patch("src.api.routes.onnx_classifier") as mock_onnx:
            mock_onnx.is_ready = True
            mock_onnx.predict.side_effect = ValueError("Cannot decode image")
            r = client_tier1_ready.post(
                "/api/v1/classify",
                files={"image": ("bad.jpg", corrupt_bytes, "image/jpeg")},
            )
        assert r.status_code == 422

    def test_classify_503_when_vlm_unavailable_and_low_conf(self, green_image):
        """503 returned when Tier-1 is low-conf AND VLM is not ready."""
        from src.main import app

        with patch("src.api.routes.onnx_classifier") as mock_onnx, \
             patch("src.api.routes.vlm_fallback") as mock_vlm:
            mock_onnx.is_ready = True
            mock_onnx.predict.return_value = ("spinach", 0.30, 7.0)
            mock_vlm.is_ready = False

            with TestClient(app) as tc:
                r = tc.post(
                    "/api/v1/classify",
                    files={"image": ("leaf.jpg", green_image, "image/jpeg")},
                    data={"threshold": "0.75"},
                )
        assert r.status_code == 503

    def test_classify_threshold_override(self, client_tier1_ready, green_image):
        """With threshold=0.95, confidence=0.92 should fail → 503 (no VLM)."""
        from src.main import app

        with patch("src.api.routes.onnx_classifier") as mock_onnx, \
             patch("src.api.routes.vlm_fallback") as mock_vlm:
            mock_onnx.is_ready = True
            mock_onnx.predict.return_value = ("spinach", 0.92, 8.0)
            mock_vlm.is_ready = False

            with TestClient(app) as tc:
                r = tc.post(
                    "/api/v1/classify",
                    files={"image": ("leaf.jpg", green_image, "image/jpeg")},
                    data={"threshold": "0.95"},
                )
        assert r.status_code == 503


# ── Explain endpoint ──────────────────────────────────────────────────────────

class TestExplain:
    def test_explain_returns_base64_heatmap(self, client_gradcam_ready, green_image):
        r = client_gradcam_ready.post(
            "/api/v1/explain",
            files={"image": ("leaf.jpg", green_image, "image/jpeg")},
        )
        assert r.status_code == 200
        body = r.json()
        assert "heatmap_base64" in body
        # Verify it is valid base64
        decoded = base64.b64decode(body["heatmap_base64"])
        assert len(decoded) > 0

    def test_explain_503_when_cam_not_ready(self, client_no_models, green_image):
        r = client_no_models.post(
            "/api/v1/explain",
            files={"image": ("leaf.jpg", green_image, "image/jpeg")},
        )
        assert r.status_code == 503

    def test_explain_rejects_non_image(self, client_gradcam_ready):
        r = client_gradcam_ready.post(
            "/api/v1/explain",
            files={"image": ("report.pdf", b"%PDF", "application/pdf")},
        )
        assert r.status_code == 422

    def test_explain_response_schema(self, client_gradcam_ready, green_image):
        body = client_gradcam_ready.post(
            "/api/v1/explain",
            files={"image": ("leaf.jpg", green_image, "image/jpeg")},
        ).json()
        assert "predicted_class" in body
        assert "raw_confidence" in body
        assert "heatmap_base64" in body
        assert "mime_type" in body
        assert "message" in body


# ── ONNX preprocessing unit tests ────────────────────────────────────────────

class TestPreprocessing:
    def test_output_shape(self, green_image):
        from src.core.onnx_engine import preprocess_image

        arr = preprocess_image(green_image)
        assert arr.shape == (1, 3, 224, 224)
        assert arr.dtype == np.float32

    def test_output_normalised(self, green_image):
        """Check that normalisation is applied (values should not be in [0,1])."""
        from src.core.onnx_engine import preprocess_image

        arr = preprocess_image(green_image)
        # Normalised values can be negative
        assert arr.min() < 0 or arr.max() > 1, "Expected normalised values"

    def test_small_image_processed(self, small_image):
        from src.core.onnx_engine import preprocess_image

        arr = preprocess_image(small_image)
        assert arr.shape == (1, 3, 224, 224)

    def test_corrupt_bytes_raises_value_error(self, corrupt_bytes):
        from src.core.onnx_engine import preprocess_image

        with pytest.raises(ValueError, match="Cannot decode image"):
            preprocess_image(corrupt_bytes)


# ── Knowledge base unit tests ─────────────────────────────────────────────────

class TestKnowledgeBase:
    def test_get_known_plant(self):
        from src.core.knowledge_base import knowledge_base

        knowledge_base.load()
        card = knowledge_base.get("spinach")
        assert card["scientific_name"] == "Spinacia oleracea"

    def test_get_normalised_key(self):
        from src.core.knowledge_base import knowledge_base

        knowledge_base.load()
        card = knowledge_base.get("Colocasia Leaves")  # mixed case + spaces
        assert "Colocasia" in card["scientific_name"]

    def test_get_unknown_plant_returns_stub(self):
        from src.core.knowledge_base import knowledge_base

        knowledge_base.load()
        card = knowledge_base.get("totally_unknown_xxxx")
        assert "common_name" in card

    def test_all_five_plants_present(self):
        from src.core.knowledge_base import knowledge_base

        knowledge_base.load()
        expected = {
            "colocasia_leaves",
            "sorrel_leaves",
            "dill_leaves",
            "fenugreek_leaves",
            "spinach",
        }
        keys = set(knowledge_base.all_keys())
        assert expected.issubset(keys)
