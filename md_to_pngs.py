"""
md_to_pngs.py – render a .md file to stitched paginated PNGs via PDF.

Usage:
    python md_to_pngs.py <input.md> <output_dir> [stitch_count]

Examples:
    python md_to_pngs.py 00_HUB.md 00_HUB
        -> default stitch=3: 00_HUB_1_2_3.png, 00_HUB_4_5_6.png, ...

    python md_to_pngs.py 00_HUB.md 00_HUB 1
        -> stitch=1 (no stitching): 00_HUB_1.png, 00_HUB_2.png, ...

    python md_to_pngs.py 03_LEARNING_PACK.md out 4
        -> stitch=4: out/03_LEARNING_PACK_1_2_3_4.png, _5_6_7_8.png, ...

Pipeline:
    .md   -- markdown lib  -->  HTML
    HTML  -- Edge headless  -->  PDF (natural pagination, no height limit)
    PDF   -- PyMuPDF        -->  rendered page images
    pages --                -->  stitched side-by-side PNG groups
"""

import sys
import subprocess
from pathlib import Path
import markdown
import fitz  # PyMuPDF
from PIL import Image

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
DPI = 150
DEFAULT_STITCH = 2

CSS = """
@page { size: A4; margin: 1.2cm 1.4cm; }
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.5;
       color: #1a1a1a; font-size: 11pt; }
h1, h2, h3, h4 { font-weight: 600; margin-top: 1em; page-break-after: avoid; }
h1 { font-size: 1.7em; border-bottom: 2px solid #ddd; padding-bottom: 6px; }
h2 { font-size: 1.35em; border-bottom: 1px solid #eee; padding-bottom: 3px; }
h3 { font-size: 1.15em; }
p { margin: 8px 0; }
code { background: #f3f3f3; padding: 1px 4px; border-radius: 3px;
       font-family: 'Cascadia Mono', Consolas, monospace; font-size: 0.92em; }
pre { background: #1e1e1e; color: #d4d4d4; padding: 10px; border-radius: 5px;
      overflow: hidden; font-size: 0.85em; page-break-inside: avoid;
      white-space: pre-wrap; word-wrap: break-word; }
pre code { background: transparent; color: inherit; padding: 0; }
table { border-collapse: collapse; margin: 10px 0; font-size: 0.95em;
        page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 6px 9px; }
th { background: #f5f5f5; font-weight: 600; }
blockquote { border-left: 4px solid #ddd; padding-left: 12px; color: #555;
             margin: 10px 0; page-break-inside: avoid; }
ul, ol { margin: 10px 0; padding-left: 26px; }
li { margin: 2px 0; }
img { max-width: 100%; }
hr { border: 0; border-top: 1px solid #ddd; margin: 18px 0; }
"""


def render_md_to_html(md_text: str) -> str:
    body = markdown.markdown(
        md_text,
        extensions=['tables', 'fenced_code', 'sane_lists', 'toc'],
    )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )


def file_url(path: Path) -> str:
    return "file:///" + str(path.resolve()).replace("\\", "/")


def html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    cmd = [
        EDGE,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        file_url(html_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if not pdf_path.exists():
        raise RuntimeError(
            f"Edge print-to-pdf failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


def pdf_to_page_images(pdf_path: Path, dpi: int = DPI) -> list:
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pages = []
    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pages.append(img)
    doc.close()
    return pages


def stitch_horizontally(imgs: list) -> Image.Image:
    widths = [im.size[0] for im in imgs]
    heights = [im.size[1] for im in imgs]
    total_w = sum(widths)
    max_h = max(heights)
    canvas = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.size[0]
    return canvas


def write_stitched_pngs(pages: list, out_dir: Path, stem: str, stitch: int) -> list:
    paths = []
    n = len(pages)
    for start in range(0, n, stitch):
        group = pages[start:start + stitch]
        indices = list(range(start + 1, start + 1 + len(group)))
        if len(group) == 1:
            composite = group[0]
        else:
            composite = stitch_horizontally(group)
        suffix = "_".join(str(i) for i in indices)
        out_path = out_dir / f"{stem}_{suffix}.png"
        composite.save(out_path, optimize=True)
        paths.append(out_path)
    return paths


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    md_path = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    stitch = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_STITCH
    if stitch < 1:
        print(f"ERROR: stitch_count must be >= 1 (got {stitch})")
        sys.exit(1)
    if not md_path.exists():
        print(f"ERROR: {md_path} not found")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = md_path.stem

    md_text = md_path.read_text(encoding='utf-8')
    line_count = md_text.count('\n') + 1
    print(f"[{md_path.name}] {line_count} lines  (stitch={stitch})")

    html = render_md_to_html(md_text)
    tmp_html = out_dir / f"{stem}.html"
    tmp_html.write_text(html, encoding='utf-8')

    tmp_pdf = out_dir / f"{stem}.pdf"
    if tmp_pdf.exists():
        tmp_pdf.unlink()

    print(f"  rendering HTML -> PDF ...")
    html_to_pdf(tmp_html, tmp_pdf)

    print(f"  rasterising PDF @ {DPI}dpi ...")
    pages = pdf_to_page_images(tmp_pdf)
    print(f"  {len(pages)} PDF page(s) -> stitching {stitch} per image ...")

    paths = write_stitched_pngs(pages, out_dir, stem, stitch)
    print(f"  wrote {len(paths)} file(s) to {out_dir}:")
    for p in paths:
        print(f"    {p.name}")

    tmp_html.unlink(missing_ok=True)
    tmp_pdf.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
