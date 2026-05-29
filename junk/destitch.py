"""
destitch.py — split 2-panel stitched PNGs into individual page images.

Usage:
    python destitch.py <input_dir> <output_dir>

Example:
    python destitch.py screenshots/ split/
        01_CAREER_PACK_1_2.png  ->  01_CAREER_PACK_1.png + 01_CAREER_PACK_2.png
        01_CAREER_PACK_3_4.png  ->  01_CAREER_PACK_3.png + 01_CAREER_PACK_4.png
"""

import sys
from pathlib import Path
from PIL import Image, ImageOps


def split_image(img_path: Path, out_dir: Path, left_num: int, right_num: int, prefix: str):
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)  # fix phone camera rotation

    w, h = img.size
    mid = w // 2

    left  = img.crop((0,   0, mid, h))
    right = img.crop((mid, 0, w,   h))

    out1 = out_dir / f"{prefix}_{left_num:02d}.png"
    out2 = out_dir / f"{prefix}_{right_num:02d}.png"

    left.save(out1,  optimize=True)
    right.save(out2, optimize=True)

    print(f"  {img_path.name}  ->  {out1.name}  +  {out2.name}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    in_dir  = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])

    if not in_dir.exists():
        print(f"ERROR: {in_dir} not found")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted([f for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG") for f in in_dir.glob(ext)])
    if not pngs:
        print(f"No image files found in {in_dir}")
        sys.exit(1)

    # use input folder name as prefix (e.g. "01_CAREER_PACK")
    prefix = in_dir.name

    print(f"Found {len(pngs)} image(s) in {in_dir}\n")
    for i, p in enumerate(pngs):
        left_num  = i * 2 + 1
        right_num = i * 2 + 2
        split_image(p, out_dir, left_num, right_num, prefix)

    print(f"\nDone. {out_dir}")


if __name__ == "__main__":
    main()
