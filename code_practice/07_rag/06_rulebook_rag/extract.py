"""The data layer: sources.yaml -> taxonomy.json, and PDFs -> data/pages.jsonl (section 5.3/5.4)."""

import json
import random
import re

import fitz  # PyMuPDF
import yaml

import config

# Collapse horizontal whitespace but KEEP newlines. The line structure of an
# instructions booklet is the only signal that separates a printed field caption
# from the prose explaining it - flattening it made caption mining impossible.
HSPACE = re.compile(r"[ \t ]+")
BLANKS = re.compile(r"\n{3,}")


# ---------------------------------------------------------------- taxonomy

def cmd_build_taxonomy(args):
    doc = yaml.safe_load(config.PATHS.sources_yaml.read_text())
    entries = sorted(doc["classes"], key=lambda e: e["slug"])
    assert len(entries) == config.N_CLASSES, \
        f"sources.yaml has {len(entries)} classes, config.N_CLASSES is {config.N_CLASSES}"

    # Shuffle before numbering so CLASS_017 is not guessable from alphabetical
    # order - the anon id must carry no information about the form (section 1.2).
    rng = random.Random(config.SEED)
    order = list(range(len(entries)))
    rng.shuffle(order)

    classes = []
    for anon_n, idx in enumerate(order):
        e = entries[idx]
        classes.append({
            "class_id": f"CLASS_{anon_n:03d}",
            "real_name": e["real_name"],
            "slug": e["slug"],
            "held_back": False,
        })

    held = random.Random(config.SEED).sample(
        sorted(c["class_id"] for c in classes), config.N_HELD_BACK)
    for c in classes:
        c["held_back"] = c["class_id"] in held

    config.PATHS.taxonomy_json.write_text(json.dumps(
        {"seed": config.SEED, "n_classes": len(classes), "classes": classes}, indent=2))

    print(f"  wrote taxonomy.json  {len(classes)} classes, seed {config.SEED}")
    print(f"  held back ({config.N_HELD_BACK}): " + ", ".join(sorted(held)))
    if args.show_mapping:
        print("\n  class_id -> slug (NEVER goes in a prompt):")
        for c in classes:
            flag = "  [held-back]" if c["held_back"] else ""
            print(f"    {c['class_id']}  {c['slug']:10}{flag}")
    return 0


def load_taxonomy():
    return json.loads(config.PATHS.taxonomy_json.read_text())["classes"]


# ---------------------------------------------------------------- pages

def cmd_extract_text(args):
    by_slug = {c["slug"]: c for c in load_taxonomy()}
    index = json.loads((config.PATHS.pdfs_dir / "index.json").read_text())

    records, excluded = [], 0
    for url, meta in sorted(index.items(), key=lambda kv: (kv[1]["slug"], kv[1]["kind"])):
        cls = by_slug.get(meta["slug"])
        if cls is None:
            continue
        source_doc = url.rsplit("/", 1)[1]
        with fitz.open(config.PATHS.pdfs_dir / meta["path"]) as pdf:
            for i, page in enumerate(pdf):
                text = HSPACE.sub(" ", page.get_text())
                text = re.sub(r" *\n *", "\n", text)
                text = BLANKS.sub("\n\n", text).strip()
                image_only = len(text) < config.MIN_PAGE_CHARS
                excluded += image_only
                records.append({
                    "page_id": f"{cls['class_id']}_{meta['kind'][:4]}_p{i:03d}",
                    "class_id": cls["class_id"],
                    "kind": meta["kind"],
                    "source_doc": source_doc,   # provenance only - never prompted
                    "n_chars": len(text),
                    "image_only": image_only,
                    "eval_case": False,         # set below
                    "text": text,
                })

    # Balanced eval set: up to EVAL_PAGES_PER_CLASS form pages per class, sampled
    # with the committed seed. Taking every page would let long forms dominate
    # leaf accuracy; taking the first N would bias toward page 1 (section 5.6).
    rng = random.Random(config.SEED)
    by_class = {}
    for r in records:
        if r["kind"] == "form" and not r["image_only"]:
            by_class.setdefault(r["class_id"], []).append(r)
    for class_id in sorted(by_class):
        pool = sorted(by_class[class_id], key=lambda r: r["page_id"])
        for r in rng.sample(pool, min(config.EVAL_PAGES_PER_CLASS, len(pool))):
            r["eval_case"] = True

    config.PATHS.pages_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with config.PATHS.pages_jsonl.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    kept = [r for r in records if not r["image_only"]]
    instr = [r for r in kept if r["kind"] == "instructions"]
    forms = [r for r in kept if r["kind"] == "form"]
    evals = [r for r in kept if r["eval_case"]]
    counts = {}
    for r in evals:
        counts[r["class_id"]] = counts.get(r["class_id"], 0) + 1

    print(f"  pages total ........ {len(records)}")
    print(f"  image-only excluded  {excluded}  (<{config.MIN_PAGE_CHARS} chars)")
    print(f"  instruction pages .. {len(instr)}   -> rules.py reads these")
    print(f"  form pages ......... {len(forms)}   (all available)")
    print(f"  EVAL SET ........... {len(evals)}   (<={config.EVAL_PAGES_PER_CLASS}/class, seed {config.SEED})")
    hist = {}
    for n in counts.values():
        hist[n] = hist.get(n, 0) + 1
    print("  eval pages/class ... " + ", ".join(f"{n} page(s): {k} classes"
                                                for n, k in sorted(hist.items())))
    zero = [c["class_id"] for c in load_taxonomy() if c["class_id"] not in counts]
    if zero:
        print(f"  CLASSES WITH NO EVAL PAGES ({len(zero)}): {', '.join(zero)}")
    return 0



def build_subparsers(sub):
    p = sub.add_parser("build-taxonomy", help="sources.yaml -> taxonomy.json")
    p.add_argument("--show-mapping", action="store_true")
    p.set_defaults(func=cmd_build_taxonomy)

    p = sub.add_parser("extract-text", help="PDFs -> data/pages.jsonl")
    p.set_defaults(func=cmd_extract_text)
