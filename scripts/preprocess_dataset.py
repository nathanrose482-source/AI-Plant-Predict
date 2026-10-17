"""
Dataset preprocessing and indexing script.

Generates `data/dataset_master.csv` from a raw directory tree:

    data/raw/
      colocasia_leaves/  *.jpg
      sorrel_leaves/     *.jpg
      dill_leaves/       *.jpg
      fenugreek_leaves/  *.jpg
      spinach/           *.jpg

Usage:
    python scripts/preprocess_dataset.py \\
        --data_dir data/raw \\
        --output   data/dataset_master.csv \\
        --augment_minority            # optional: save augmented copies for minority classes
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import shutil
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CLASS_NAMES = [
    "colocasia_leaves",
    "sorrel_leaves",
    "dill_leaves",
    "fenugreek_leaves",
    "spinach",
]


# ── Scan ──────────────────────────────────────────────────────────────────────

def scan_directory(data_dir: Path) -> list[tuple[Path, str]]:
    """Return (image_path, class_name) pairs from a class-folder layout."""
    samples: list[tuple[Path, str]] = []
    for cls_dir in sorted(data_dir.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name.startswith("."):
            continue
        cls_name = cls_dir.name.lower().replace("-", "_").replace(" ", "_")
        found = [
            f for f in cls_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS and "augmented" not in f.parts
        ]
        if found:
            logger.info("  %s: %d images", cls_name, len(found))
            for f in found:
                samples.append((f, cls_name))
    return samples


# ── Imbalance report ──────────────────────────────────────────────────────────

def report_balance(samples: list[tuple[Path, str]]) -> dict[str, int]:
    counts: dict[str, int] = Counter(c for _, c in samples)  # type: ignore[assignment]
    logger.info("Class distribution:")
    for cls, n in sorted(counts.items()):
        logger.info("  %-25s %4d images", cls, n)
    max_count = max(counts.values())
    min_count = min(counts.values())
    logger.info("Imbalance ratio: %.2f×", max_count / max(min_count, 1))
    return counts


# ── Offline augmentation for severe imbalance ─────────────────────────────────

def augment_minority_classes(
    samples: list[tuple[Path, str]],
    counts: dict[str, int],
    data_dir: Path,
    target_ratio: float = 0.5,
) -> list[tuple[Path, str]]:
    """
    For classes with count < target_ratio × max_count, generate simple
    augmented copies (horizontal flip, rotation) and save to disk.
    """
    try:
        from PIL import Image, ImageOps
    except ImportError:
        logger.warning("Pillow not available — skipping offline augmentation.")
        return samples

    max_count = max(counts.values())
    target_count = int(max_count * target_ratio)
    new_samples = list(samples)

    for cls_name, n in counts.items():
        if n >= target_count:
            continue

        needed = target_count - n
        logger.info("Augmenting '%s': generating %d synthetic images.", cls_name, needed)
        cls_samples = [p for p, c in samples if c == cls_name]
        aug_dir = data_dir / cls_name / "augmented"
        aug_dir.mkdir(parents=True, exist_ok=True)

        for i in range(needed):
            src = cls_samples[i % len(cls_samples)]
            try:
                img = Image.open(src).convert("RGB")
            except Exception:
                continue

            # Simple transforms for variety
            ops = [
                lambda im: ImageOps.mirror(im),
                lambda im: im.rotate(random.choice([90, 180, 270])),
                lambda im: ImageOps.flip(im),
            ]
            img = random.choice(ops)(img)

            dst = aug_dir / f"aug_{i:05d}_{src.stem}.jpg"
            img.save(dst, "JPEG", quality=90)
            new_samples.append((dst, cls_name))

    return new_samples


# ── CSV writer ────────────────────────────────────────────────────────────────

def write_csv(samples: list[tuple[Path, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        for img_path, cls_name in samples:
            writer.writerow({"image_path": str(img_path), "label": cls_name})
    logger.info("Dataset CSV written: %s (%d rows)", output_path, len(samples))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess and index plant dataset")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--output", default="data/dataset_master.csv")
    parser.add_argument(
        "--augment_minority",
        action="store_true",
        help="Offline augment minority classes to reduce imbalance",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory '{data_dir}' not found.\n"
            "Create it with class sub-folders:\n"
            "  data/raw/colocasia_leaves/\n"
            "  data/raw/sorrel_leaves/\n"
            "  data/raw/dill_leaves/\n"
            "  data/raw/fenugreek_leaves/\n"
            "  data/raw/spinach/"
        )

    logger.info("Scanning %s ...", data_dir)
    samples = scan_directory(data_dir)

    if not samples:
        logger.error("No images found — check that class directories exist and contain images.")
        return

    counts = report_balance(samples)

    if args.augment_minority:
        samples = augment_minority_classes(samples, counts, data_dir)
        logger.info("After augmentation:")
        report_balance(samples)

    random.shuffle(samples)
    write_csv(samples, Path(args.output))


if __name__ == "__main__":
    main()
