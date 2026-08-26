#!/usr/bin/env python3
"""Render resume HTML -> PDF at full fidelity (backgrounds, A4, no browser chrome).

Usage:  ./pdfenv/bin/python make_pdf.py resume_sameer_khan_v2.html
"""
import sys, pathlib
from playwright.sync_api import sync_playwright

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "resume_sameer_khan_v2.html").resolve()
out = src.with_suffix(".pdf")

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.goto(src.as_uri(), wait_until="networkidle")
    pg.pdf(path=str(out), format="A4",
           print_background=True,                      # keeps navy band + tinted sidebar
           margin={k: "0mm" for k in ("top","bottom","left","right")},
           prefer_css_page_size=True)
    b.close()
print(f"{out}  ({out.stat().st_size/1024:.0f} KB)")
