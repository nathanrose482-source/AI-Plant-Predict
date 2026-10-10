# 🌿 Plant Vision — Edge-Optimized Plant Classification & Knowledge Retrieval

A production-grade two-tier computer vision pipeline for identifying leafy plants, built with FastAPI, MobileNetV3 ONNX, Grad-CAM, and Google Gemini VLM fallback.

**Target plants:** Colocasia/Taro leaves · Green Sorrel · Dill leaves · Fenugreek (Methi) · Spinach

---

## Architecture Overview

```
Image Upload
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Tier-1: Local ONNX Inference (MobileNetV3-Small, CPU)      │
│  Resize(256) → CenterCrop(224) → Normalize → ONNX Runtime   │
│  Target latency: < 15 ms                                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
           ┌───────────────┴──────────────┐
           │  Confidence Gate (≥ 0.75?)   │
           └───────────────┬──────────────┘
                  ┌────────┴────────┐
                  │ YES             │ NO / OOD
                  ▼                 ▼
        ┌──────────────┐   ┌───────────────────────────┐
        │  Path A      │   │  Tier-2: Gemini VLM       │
        │  Local KB    │   │  Zero-shot multimodal ID  │
        │  Knowledge   │   │  (gemini-3.8-flash)       │
        │  Card        │   └───────────────────────────┘
        └──────────────┘             │
                  │                  │
                  └────────┬─────────┘
                           ▼
                    JSON Response
              (tier, confidence, latency,
               knowledge_card)
```

---

## Project Structure

```
plant-vision/
├── src/
│   ├── main.py                   # FastAPI app entry point
│   ├── config.py                 # Centralised env-var config
│   ├── api/
│   │   ├── routes.py             # /health, /classify, /explain endpoints
│   │   └── schemas.py            # Pydantic v2 request/response models
│   └── core/
│       ├── onnx_engine.py        # ONNX Runtime inference wrapper
│       ├── gradcam.py            # Native PyTorch Grad-CAM engine
│       ├── vlm_fallback.py       # Gemini VLM Tier-2 fallback
│       └── knowledge_base.py     # Botanical knowledge database
├── scripts/
│   ├── train.py                  # MobileNetV3-Small fine-tuning
│   ├── export_onnx.py            # PyTorch → ONNX export + validation
│   └── preprocess_dataset.py     # Dataset indexing & CSV generation
├── tests/
│   ├── conftest.py               # Pytest configuration
│   └── test_api.py               # Full API + unit test suite
├── static/
│   ├── index.html                # Vanilla JS frontend
│   ├── style.css                 # CSS styles
│   └── app.js                    # Client-side logic
├── data/
│   └── knowledge/
│       └── plant_knowledge.json  # Botanical knowledge base
├── models/                       # (gitignored) ONNX + PT checkpoints
├── Dockerfile
├── requirements.txt
├── .env.example
└── pytest.ini
```

---

## Quick Start (Local)

### 1. Clone & install

```bash
git clone <your-repo-url>
cd plant-vision

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

### 3. Prepare your dataset

Organise plant images in this folder layout:

```
data/raw/
  colocasia_leaves/   *.jpg
  sorrel_leaves/      *.jpg
  dill_leaves/        *.jpg
  fenugreek_leaves/   *.jpg
  spinach/            *.jpg
```

Then run preprocessing:

```bash
python scripts/preprocess_dataset.py \
    --data_dir data/raw \
    --output   data/dataset_master.csv \
    --augment_minority      # optional: offline augment minority classes
```

### 4. Train

```bash
python scripts/train.py \
    --data_dir   data/raw \
    --csv        data/dataset_master.csv \
    --epochs     30 \
    --batch_size 32 \
    --lr         1e-3 \
    --output     models/plant_classifier.pt
```

Training features:
- MobileNetV3-Small pre-trained on ImageNet; last 2 blocks fine-tuned
- **Class-weighted cross-entropy loss** for imbalanced datasets
- **WeightedRandomSampler** for balanced mini-batches
- FGVC augmentations: ColorJitter, RandomAffine, RandomErasing
- CosineAnnealingLR scheduler
- Best checkpoint saved by val accuracy

### 5. Export to ONNX

```bash
python scripts/export_onnx.py \
    --checkpoint models/plant_classifier.pt \
    --output     models/plant_classifier.onnx
```

A parity check (PyTorch vs. ONNX max absolute diff < 1e-4) runs automatically.

### 6. Run the server

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Open the UI at **http://localhost:8000** and the API docs at **http://localhost:8000/docs**.

---

## API Reference

### `GET /api/v1/health`

Returns readiness state of all three engines.

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 42.5,
  "components": {
    "onnx_engine":     { "ready": true,  "detail": "ONNX session active" },
    "pytorch_gradcam": { "ready": true,  "detail": "PyTorch checkpoint loaded" },
    "vlm_fallback":    { "ready": true,  "detail": "Gemini API reachable" }
  }
}
```

### `POST /api/v1/classify`

**Form fields:**
| Field | Type | Default | Description |
|---|---|---|---|
| `image` | `UploadFile` | — | Plant image (JPEG/PNG/WebP) |
| `threshold` | `float` | `0.75` | Confidence gate (0–1) |
| `force_fallback` | `bool` | `false` | Skip Tier-1, route direct to Gemini |

**Response:**
```json
{
  "prediction_tier": "tier1_local",
  "predicted_class": "spinach",
  "raw_confidence": 0.923,
  "elapsed_latency_ms": 8.4,
  "threshold_used": 0.75,
  "is_ood": false,
  "knowledge_card": {
    "common_name": "Spinach",
    "scientific_name": "Spinacia oleracea",
    "family": "Amaranthaceae",
    "description": "...",
    "edibility": "...",
    "nutritional_highlights": "...",
    "traditional_uses": ["Palak paneer", "Saag"],
    "caution": "...",
    "fgvc_notes": "..."
  }
}
```

### `POST /api/v1/explain`

**Form fields:**
| Field | Type | Description |
|---|---|---|
| `image` | `UploadFile` | Plant image |

**Response:**
```json
{
  "predicted_class": "fenugreek_leaves",
  "raw_confidence": 0.881,
  "heatmap_base64": "<base64-encoded JPEG>",
  "mime_type": "image/jpeg",
  "message": "Grad-CAM attribution heatmap generated successfully."
}
```

---

## Docker Deployment

### Build

```bash
docker build -t plant-vision .
```

### Run

```bash
# Mount your models directory; pass env file
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/models:/app/models:ro \
  --memory=900m \
  --cpus=2 \
  --name plant-vision \
  plant-vision
```

### Verify

```bash
curl http://localhost:8000/api/v1/health
```

### Memory profile

| Component | Approx. RAM |
|---|---|
| Python / FastAPI baseline | ~80 MB |
| ONNX Runtime + model | ~80 MB |
| PyTorch (Grad-CAM only, eval) | ~250 MB |
| google-genai SDK | ~30 MB |
| **Total (all engines)** | **~450 MB** |

> To stay under 1 GB even with traffic bursts, the server runs a single Uvicorn worker.

---

## Running Tests

```bash
pytest tests/ -v
```

The suite covers:
- Health endpoint contract (status, component keys, literal values)
- Classify: Tier-1 high confidence, Tier-2 fallback, force_fallback flag, threshold override
- Classify: rejection of non-image files and corrupt bytes
- Classify: 503 when VLM unavailable and Tier-1 under-threshold
- Explain: heatmap shape/base64 validity, 503 without model, schema validation
- ONNX preprocessing: output shape `(1, 3, 224, 224)`, normalisation, corrupt input
- Knowledge base: exact/fuzzy lookup, stub fallback, all 5 plants present

---

## Fine-Grained Visual Categorisation (FGVC) Notes

| Plant pair | Discriminating feature |
|---|---|
| Colocasia vs. Alocasia | Peltate leaf attachment; matte vs. glossy underside |
| Fenugreek vs. Clover | Obovate trifoliate with finely toothed margins |
| Dill vs. Fennel fronds | Softer anise scent; slightly greyer-blue colour |
| Sorrel vs. Spinach | Arrow-shaped, toothed base lobes vs. flat oval |
| Spinach vs. Malabar spinach | Thin flat leaf vs. thick succulent with woody stem |

These notes are embedded in each plant's `knowledge_card.fgvc_notes` field and displayed in the UI.

---

## Dataset Imbalance Handling

The training script uses two complementary strategies:

1. **Class-weighted cross-entropy loss** — rarer classes contribute proportionally more to the loss.
2. **WeightedRandomSampler** — oversamples minority classes in each mini-batch so every class is seen roughly equally per epoch.
3. **Offline augmentation** (`--augment_minority` flag) — generates flipped/rotated copies for classes below 50% of the majority class size.

---

## Extending to New Plant Classes

1. Add images to `data/raw/<new_class_name>/`
2. Add the class to `CLASS_NAMES` in `scripts/train.py`
3. Add a knowledge entry to `data/knowledge/plant_knowledge.json`
4. Re-run preprocessing → train → export ONNX
5. Update `NUM_CLASSES` in `.env`

---

## Licence

MIT
