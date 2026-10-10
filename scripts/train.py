"""
Training script for the plant classification model.

Usage:
    python scripts/train.py \\
        --data_dir  data/raw \\
        --csv       data/dataset_master.csv \\
        --epochs    30 \\
        --batch_size 32 \\
        --lr        1e-3 \\
        --output    models/plant_classifier.pt

The script:
  1. Indexes the dataset from dataset_master.csv (or scans data_dir if CSV absent).
  2. Computes per-class weights to handle imbalanced distributions.
  3. Fine-tunes MobileNetV3-Small (ImageNet pre-trained) with class-weighted
     cross-entropy loss.
  4. Saves the full model state dict + class names JSON.
  5. Prints per-epoch train/val metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

# ── Plant class map (change order if your folders differ) ─────────────────────
DEFAULT_CLASS_NAMES: list[str] = [
    "colocasia_leaves",
    "sorrel_leaves",
    "dill_leaves",
    "fenugreek_leaves",
    "spinach",
]

# ── Transforms ────────────────────────────────────────────────────────────────
TRAIN_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        # FGVC augmentations
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1),
    ]
)

VAL_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PlantDataset(Dataset):
    """Reads from dataset_master.csv or scans a directory tree."""

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        class_names: list[str],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.samples = samples
        self.class_names = class_names
        self.transform = transform

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        class_names: list[str],
        transform: transforms.Compose | None = None,
    ) -> "PlantDataset":
        import csv

        samples: list[tuple[Path, int]] = []
        name_to_idx = {n: i for i, n in enumerate(class_names)}

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = Path(row["image_path"])
                label_str = row.get("label", row.get("class", ""))
                label_idx = name_to_idx.get(label_str)
                if label_idx is None:
                    logger.warning("Unknown label '%s' in CSV — skipping.", label_str)
                    continue
                if not img_path.exists():
                    logger.warning("Image not found: %s — skipping.", img_path)
                    continue
                samples.append((img_path, label_idx))

        logger.info("CSV dataset: %d samples across %d classes.", len(samples), len(class_names))
        return cls(samples, class_names, transform)

    @classmethod
    def from_directory(
        cls,
        root: Path,
        class_names: list[str],
        transform: transforms.Compose | None = None,
    ) -> "PlantDataset":
        """Scan root/<class_name>/*.jpg structure."""
        samples: list[tuple[Path, int]] = []
        name_to_idx = {n: i for i, n in enumerate(class_names)}

        for cls_dir in sorted(root.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name.lower().replace("-", "_").replace(" ", "_")
            if cls_name not in name_to_idx:
                logger.debug("Skipping unrecognised class dir: %s", cls_dir.name)
                continue
            idx = name_to_idx[cls_name]
            for img_file in cls_dir.rglob("*"):
                if img_file.suffix.lower() in cls.SUPPORTED_EXTS:
                    samples.append((img_file, idx))

        logger.info("Directory dataset: %d samples.", len(samples))
        return cls(samples, class_names, transform)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        from PIL import Image

        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)  # type: ignore[assignment]
        return img, label  # type: ignore[return-value]


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(
    samples: list[tuple[Path, int]], num_classes: int
) -> torch.Tensor:
    """Inverse-frequency class weights for cross-entropy loss."""
    counts = Counter(label for _, label in samples)
    total = len(samples)
    weights = []
    for i in range(num_classes):
        n = counts.get(i, 1)
        weights.append(total / (num_classes * n))
    logger.info("Class weights: %s", {i: f"{w:.3f}" for i, w in enumerate(weights)})
    return torch.tensor(weights, dtype=torch.float32)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # Freeze all backbone layers except the last conv block
    for name, param in model.features.named_parameters():
        layer_idx = int(name.split(".")[0]) if name[0].isdigit() else -1
        param.requires_grad = layer_idx >= 10  # fine-tune last 2 blocks

    # Replace classifier head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train plant classifier (MobileNetV3-Small)")
    parser.add_argument("--data_dir", default="data/raw", help="Root directory with class sub-folders")
    parser.add_argument("--csv", default="data/dataset_master.csv", help="Optional dataset_master.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="models/plant_classifier.pt")
    parser.add_argument("--class_names", nargs="+", default=DEFAULT_CLASS_NAMES)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    # ── Load dataset ──────────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    data_dir = Path(args.data_dir)

    if csv_path.exists():
        full_ds = PlantDataset.from_csv(csv_path, args.class_names)
    elif data_dir.exists():
        full_ds = PlantDataset.from_directory(data_dir, args.class_names)
    else:
        raise FileNotFoundError(
            f"Neither CSV ({csv_path}) nor data directory ({data_dir}) found. "
            "Please prepare your dataset first."
        )

    # Train / val split
    indices = list(range(len(full_ds)))
    random.shuffle(indices)
    val_size = int(len(indices) * args.val_split)
    train_idx, val_idx = indices[val_size:], indices[:val_size]

    from torch.utils.data import Subset

    train_samples = [full_ds.samples[i] for i in train_idx]
    class_weights = compute_class_weights(train_samples, len(args.class_names))

    train_subset = Subset(PlantDataset(train_samples, args.class_names, TRAIN_TRANSFORMS), list(range(len(train_samples))))
    val_subset = Subset(
        PlantDataset([full_ds.samples[i] for i in val_idx], args.class_names, VAL_TRANSFORMS),
        list(range(len(val_idx))),
    )

    # Weighted sampler for balanced mini-batches
    sample_weights = [class_weights[label].item() for _, label in train_samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ── Model + loss + optim ──────────────────────────────────────────────────
    model = build_model(len(args.class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            "Epoch %2d/%d | train loss=%.4f acc=%.3f | val loss=%.4f acc=%.3f | lr=%.2e",
            epoch, args.epochs,
            train_loss, train_acc,
            val_loss, val_acc,
            scheduler.get_last_lr()[0],
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "class_names": args.class_names,
                },
                output_path,
            )
            logger.info("  ✓ New best val_acc=%.4f — checkpoint saved to %s", val_acc, output_path)

    # Save class names JSON for inference
    cn_path = output_path.parent / "class_names.json"
    with open(cn_path, "w", encoding="utf-8") as f:
        json.dump(args.class_names, f, indent=2)
    logger.info("Class names saved to %s", cn_path)
    logger.info("Training complete. Best val accuracy: %.4f", best_val_acc)


if __name__ == "__main__":
    main()
