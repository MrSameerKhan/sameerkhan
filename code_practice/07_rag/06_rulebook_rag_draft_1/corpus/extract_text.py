"""data/pdfs/ -> data/real/documents.jsonl. §22.6.

One record per PDF (all pages concatenated) — these are single-instance official forms,
not multi-page bundles of unrelated documents, so the §5.4 "all pages of one form
concatenated" case applies, not one-record-per-page.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import fitz  # pymupdf

import config

VALID_SOURCE = {"real", "synthetic"}
VALID_GENERATED_FROM = {"form_text_only", "none"}


def extract_pages(pdf_path: Path) -> list[str]:
    doc = fitz.open(pdf_path)
    try:
        return [page.get_text() for page in doc]
    finally:
        doc.close()


def clean_text(s: str) -> str:
    s = "".join(ch for ch in s if ch == "\n" or ch == "\t" or ch >= " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def to_records(pages: list[str], class_id: str, source_url: str, document_type_id: str) -> list[dict]:
    text = clean_text("\n".join(pages))
    record = {
        "doc_id": f"real_{class_id}_001",
        "text": text,
        "true_class_id": class_id,
        "true_document_type_id": document_type_id,
        "source": "real",
        "generated_from": "none",
        "generator_model": None,
        "page_count": len(pages),
        "source_url": source_url,
    }
    _validate_record(record)
    return [record]


def _validate_record(record: dict) -> None:
    required = ["doc_id", "text", "true_class_id", "true_document_type_id",
                "source", "generated_from", "page_count"]
    missing = [k for k in required if k not in record]
    if missing:
        raise ValueError(f"record missing fields {missing}: {record.get('doc_id')}")
    if record["source"] not in VALID_SOURCE:
        raise ValueError(f"invalid source: {record['source']}")
    if record["generated_from"] not in VALID_GENERATED_FROM:
        raise ValueError(f"invalid generated_from: {record['generated_from']}")


def _class_to_doctype() -> dict:
    tax = json.loads(config.PATHS.taxonomy_json.read_text())
    out = {}
    for cat in tax["categories"]:
        for dt in cat["document_types"]:
            for cls in dt["classes"]:
                out[cls["class_id"]] = dt["document_type_id"]
    return out


def cmd_extract_text(args: argparse.Namespace) -> int:
    index = json.loads((config.PATHS.pdfs_dir / "index.json").read_text())
    class_to_doctype = _class_to_doctype()
    items = list(index.items())
    if args.limit:
        items = items[: args.limit]

    if args.dry_run:
        for class_id, meta in items:
            pdf_path = Path(meta["path"])
            pages = extract_pages(pdf_path)
            page1 = pages[0] if pages else ""
            flag = " [LIKELY IMAGE-ONLY]" if len("".join(pages)) < 200 else ""
            print(f"{pdf_path.name}  class={class_id}  pages={len(pages)}  "
                  f"page1_chars={len(page1)}{flag}")
            print(f"  first 200 chars: {page1[:200]!r}")
        return 0

    all_records = []
    warnings = []
    for class_id, meta in items:
        pdf_path = Path(meta["path"])
        try:
            pages = extract_pages(pdf_path)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"{class_id}: failed to open {pdf_path.name} ({e})")
            continue
        total_chars = len("".join(pages))
        if total_chars < 200:
            warnings.append(f"{class_id}: only {total_chars} chars extracted, likely image-only")
        document_type_id = class_to_doctype.get(class_id, "unknown")
        all_records.extend(to_records(pages, class_id, meta["url"], document_type_id))

    config.PATHS.real_docs.parent.mkdir(parents=True, exist_ok=True)
    with open(config.PATHS.real_docs, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    report = [
        "# Extract report", "",
        f"- PDFs processed: {len(items)}",
        f"- records written: {len(all_records)}",
        f"- warnings: {len(warnings)}", "",
    ]
    report += [f"- {w}" for w in warnings]
    config.PATHS.reports_dir.mkdir(parents=True, exist_ok=True)
    (config.PATHS.reports_dir / "extract_report.md").write_text("\n".join(report))

    print(f"{len(items)} PDFs -> {len(all_records)} real documents ({len(warnings)} warnings)")
    print(f"-> wrote {config.PATHS.real_docs}")
    return 0
