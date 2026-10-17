import os
import zipfile
import shutil
from pathlib import Path

DATA_RAW = Path("data/raw")
TRANSFERS = Path(r"C:\Users\natha\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\sessions\53459CEFFD6DA348BBD1C8E8A4DC68A1596CE6ED\transfers\2026-42")

ZIP_MAPPINGS = {
    "amaranth.zip": "amaranth",
    "coriander.zip": "coriander",
    "red_amaranth.zip": "red_amaranth",
    "Taro_dataset.zip.zip": "colocasia_leaves",
}

def extract_all():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    
    for zip_name, class_name in ZIP_MAPPINGS.items():
        zip_path = TRANSFERS / zip_name
        dest_dir = DATA_RAW / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing {zip_name} -> {dest_dir}...")
        count = 0
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                filename = os.path.basename(member.filename)
                if not filename:
                    continue
                ext = os.path.splitext(filename)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.webp']:
                    # Extract to class directory with a clean filename
                    target_name = f"{class_name}_{count:04d}{ext}"
                    target_path = dest_dir / target_name
                    with zf.open(member) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                    count += 1
        print(f"Extracted {count} images for class '{class_name}'.")

    print("\nSummary of extracted dataset in data/raw:")
    total = 0
    for cdir in sorted(DATA_RAW.iterdir()):
        if cdir.is_dir():
            n = len([f for f in cdir.iterdir() if f.is_file()])
            total += n
            print(f"  {cdir.name}: {n} images")
    print(f"Total images: {total}")

if __name__ == "__main__":
    extract_all()
