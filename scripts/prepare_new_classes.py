"""
Copy new class images into data/raw/ and add mint knowledge card.
"""
import shutil, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Source paths ──────────────────────────────────────────────────────────────
MINT_SRC    = Path(r"C:\Users\natha\OneDrive\Desktop\mint")
FEN_SRC     = Path(r"C:\Users\natha\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\sessions\53459CEFFD6DA348BBD1C8E8A4DC68A1596CE6ED\transfers\2026-42\fenugreek\fenugreek")
SP_SRC      = Path(r"C:\Users\natha\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\sessions\53459CEFFD6DA348BBD1C8E8A4DC68A1596CE6ED\transfers\2026-42\spinach\spinach")

EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.JPG', '.JPEG', '.PNG'}

def copy_class(src: Path, dst_name: str) -> int:
    dst = ROOT / "data" / "raw" / dst_name
    dst.mkdir(parents=True, exist_ok=True)
    existing = len(list(dst.iterdir()))
    if existing > 0:
        print(f"  [{dst_name}] already has {existing} files — skipping copy")
        return existing
    imgs = [f for f in src.iterdir() if f.suffix in EXTS]
    for i, f in enumerate(imgs):
        shutil.copy(f, dst / f"{dst_name}_{i:04d}{f.suffix.lower()}")
    print(f"  [{dst_name}] copied {len(imgs)} images")
    return len(imgs)

print("=== Copying new class images ===")
n_mint = copy_class(MINT_SRC,  "mint")
n_fen  = copy_class(FEN_SRC,   "fenugreek_leaves")
n_sp   = copy_class(SP_SRC,    "spinach")
print(f"Total new images: {n_mint + n_fen + n_sp}")

# ── Update plant_knowledge.json ───────────────────────────────────────────────
kb_path = ROOT / "data" / "knowledge" / "plant_knowledge.json"
with open(kb_path, encoding="utf-8") as f:
    kb = json.load(f)

mint_card = {
    "common_name": "Mint",
    "local_name": "Pudina",
    "scientific_name": "Mentha spicata / Mentha piperita",
    "family": "Lamiaceae",
    "description": "Mint is a fast-growing aromatic herb with bright-green serrated leaves and a distinctive cool menthol fragrance. Widely used in chutneys, drinks, and as a garnish.",
    "nutritional_info": "Rich in vitamins A and C, iron, manganese, and antioxidants.",
    "culinary_uses": ["chutneys", "raita", "mojitos", "mint tea", "salads", "garnish"],
    "medicinal_uses": ["aids digestion", "relieves nausea", "freshens breath", "cooling effect"],
    "growing_tips": "Grows best in moist, well-drained soil with partial shade. Spreads quickly; best grown in containers.",
    "harvest_season": "Year-round in warm climates",
    "storage_tips": "Store wrapped in a damp paper towel in the refrigerator for up to 2 weeks, or dry/freeze for longer storage.",
    "class_label": "mint"
}

# Check if mint already exists
existing_labels = [card.get("class_label") for card in kb]
if "mint" not in existing_labels:
    kb.append(mint_card)
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)
    print(f"\n[knowledge] Added mint card. Total cards: {len(kb)}")
else:
    print(f"\n[knowledge] Mint card already exists. Skipping.")

print("\nDone! Now run preprocess_dataset.py")
