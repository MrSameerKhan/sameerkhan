"""seed_taxonomy.yaml -> taxonomy.json + validation. §22.4.

Held-back selection: seed=42 random sample of 20 leaf classes, excluding every class that
is a member of a context_dependent pair or a name-near-duplicate pair (§5.3) — splitting
either kind of pair across index/held-back would break the near_duplicate and
context_dependent eval case types, which assume both members are in the index together.
"""

import argparse
import random
from pathlib import Path

import yaml

import config

DEFAULT_SEED_PATH = config.PATHS.taxonomy_seed
DEFAULT_OUT_PATH = config.PATHS.taxonomy_json
DEFAULT_STATS_PATH = config.PATHS.reports_dir / "taxonomy_stats.md"


def load_seed(path: Path = DEFAULT_SEED_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def _leaf_records(tax: dict) -> list[dict]:
    """Flatten to one record per class, each carrying its documentType and category context."""
    out = []
    for cat in tax["categories"]:
        for dt in cat["document_types"]:
            for cls in dt["classes"]:
                out.append({**cls, "_document_type": dt, "_category": cat})
    return out


def _normalized_edit_distance(a: str, b: str) -> float:
    a, b = a.lower(), b.lower()
    if not a and not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def near_duplicate_pairs(records: list[dict], threshold: float = config.DEFECT_NAME_DISTANCE_THRESHOLD):
    pairs = []
    for i, a in enumerate(records):
        for b in records[i + 1:]:
            d = _normalized_edit_distance(a["class_name"], b["class_name"])
            if d < threshold:
                pairs.append((a["class_id"], b["class_id"], round(d, 3)))
    return pairs


def validate(tax: dict) -> list[str]:
    errors = []
    records = _leaf_records(tax)

    seen_class_ids = set()
    for r in records:
        if r["class_id"] in seen_class_ids:
            errors.append(f"duplicate class_id: {r['class_id']}")
        seen_class_ids.add(r["class_id"])

    seen_dt_ids = set()
    for cat in tax["categories"]:
        for dt in cat["document_types"]:
            if dt["document_type_id"] in seen_dt_ids:
                errors.append(f"duplicate document_type_id: {dt['document_type_id']}")
            seen_dt_ids.add(dt["document_type_id"])
            if "page_count" not in dt or "min" not in dt["page_count"] or "max" not in dt["page_count"]:
                errors.append(f"documentType with no page_count: {dt['document_type_id']}")

    seen_cat_ids = set()
    for cat in tax["categories"]:
        if cat["category_id"] in seen_cat_ids:
            errors.append(f"duplicate category_id: {cat['category_id']}")
        seen_cat_ids.add(cat["category_id"])

    if len(records) < config.MIN_LEAF_CLASSES:
        errors.append(f"fewer than {config.MIN_LEAF_CLASSES} leaf classes: {len(records)}")

    for r in records:
        pw = r.get("pair_with")
        if pw and pw not in seen_class_ids:
            errors.append(f"context_dependent pair_with references unknown class: {r['class_id']} -> {pw}")

    return errors


def _assign_held_back(records: list[dict], pairs: list[tuple], seed: int = config.TAXONOMY_SEED) -> set[str]:
    excluded = set()
    for r in records:
        if r.get("context_dependent"):
            excluded.add(r["class_id"])
    for a, b, _ in pairs:
        excluded.add(a)
        excluded.add(b)

    eligible = [r["class_id"] for r in records if r["class_id"] not in excluded]
    rng = random.Random(seed)
    return set(rng.sample(eligible, config.N_HELD_BACK))


def build(seed_path: Path = DEFAULT_SEED_PATH) -> dict:
    tax = load_seed(seed_path)
    records = _leaf_records(tax)
    dup_pairs = near_duplicate_pairs(records)
    held_back_ids = _assign_held_back(records, dup_pairs)

    out = {"taxonomy_version": tax["taxonomy_version"], "categories": []}
    for cat in tax["categories"]:
        out_cat = {"category_id": cat["category_id"], "category_name": cat["category_name"], "document_types": []}
        for dt in cat["document_types"]:
            out_dt = {
                "document_type_id": dt["document_type_id"],
                "document_type_name": dt["document_type_name"],
                "page_count": dt["page_count"],
                "classes": [],
            }
            for cls in dt["classes"]:
                out_dt["classes"].append({
                    "class_id": cls["class_id"],
                    "class_name": cls["class_name"],
                    "form_number": cls.get("form_number"),
                    "structured": cls["structured"],
                    "held_back": cls["class_id"] in held_back_ids,
                    "real_text_available": cls["real_text_available"],
                    "context_dependent": cls.get("context_dependent", False),
                    "pair_with": cls.get("pair_with"),
                })
            out_cat["document_types"].append(out_dt)
        out["categories"].append(out_cat)
    return out


def class_index(tax: dict) -> dict:
    """class_id -> {class, document_type, category}"""
    idx = {}
    for cat in tax["categories"]:
        cat_meta = {"category_id": cat["category_id"], "category_name": cat["category_name"]}
        for dt in cat["document_types"]:
            dt_meta = {"document_type_id": dt["document_type_id"], "document_type_name": dt["document_type_name"]}
            for cls in dt["classes"]:
                idx[cls["class_id"]] = {"class": cls, "document_type": dt_meta, "category": cat_meta}
    return idx


def _summary(tax_seed: dict, dup_pairs: list[tuple], held_back_ids: set[str] = None) -> dict:
    records = _leaf_records(tax_seed)
    by_dt: dict = {}
    for r in records:
        by_dt.setdefault(r["_document_type"]["document_type_id"], []).append(r["class_id"])
    deepest = max(by_dt.items(), key=lambda kv: len(kv[1]))
    context_pairs = {(r["class_id"], r["pair_with"]) for r in records if r.get("context_dependent")}
    context_pairs = {tuple(sorted(p)) for p in context_pairs}
    return {
        "n_categories": len(tax_seed["categories"]),
        "n_document_types": sum(len(c["document_types"]) for c in tax_seed["categories"]),
        "n_classes": len(records),
        "deepest_fanout": (deepest[0], len(deepest[1])),
        "n_near_duplicate_pairs": len(dup_pairs),
        "n_context_dependent_pairs": len(context_pairs),
        "n_held_back": len(held_back_ids) if held_back_ids else 0,
    }


def cmd_build_taxonomy(args: argparse.Namespace) -> int:
    tax_seed = load_seed(DEFAULT_SEED_PATH)
    errors = validate(tax_seed)
    dup_pairs = near_duplicate_pairs(_leaf_records(tax_seed))

    if args.dry_run:
        s = _summary(tax_seed, dup_pairs)
        print(f"categories: {s['n_categories']}  documentTypes: {s['n_document_types']}  "
              f"leaf classes: {s['n_classes']}")
        print(f"deepest fan-out: {s['deepest_fanout'][0]} ({s['deepest_fanout'][1]} classes)")
        print(f"near-duplicate pairs (name-distance < {config.DEFECT_NAME_DISTANCE_THRESHOLD}): "
              f"{s['n_near_duplicate_pairs']}")
        for a, b, d in dup_pairs:
            print(f"  {a} <-> {b}  (distance={d})")
        print(f"context-dependent pairs: {s['n_context_dependent_pairs']}")
        print(f"held-back: not yet assigned (dry run) — target {config.N_HELD_BACK}")
        if errors:
            print(f"validation errors ({len(errors)}):")
            for e in errors:
                print(f"  - {e}")
        else:
            print("validation: 0 errors")
        return 0 if not errors else 1

    if errors:
        print(f"validation FAILED ({len(errors)} errors):")
        for e in errors:
            print(f"  - {e}")
        return 1

    out = build(DEFAULT_SEED_PATH)
    held_back_ids = {c["class_id"] for cat in out["categories"] for dt in cat["document_types"]
                      for c in dt["classes"] if c["held_back"]}
    if len(held_back_ids) != config.N_HELD_BACK:
        print(f"validation FAILED: held-back count = {len(held_back_ids)}, expected {config.N_HELD_BACK}")
        return 1

    DEFAULT_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    import json
    DEFAULT_OUT_PATH.write_text(json.dumps(out, indent=2))

    s = _summary(tax_seed, dup_pairs, held_back_ids)
    DEFAULT_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_STATS_PATH.write_text(
        f"# Taxonomy stats — v{out['taxonomy_version']}\n\n"
        f"- categories: {s['n_categories']}\n"
        f"- documentTypes: {s['n_document_types']}\n"
        f"- leaf classes: {s['n_classes']}\n"
        f"- deepest fan-out: `{s['deepest_fanout'][0]}` ({s['deepest_fanout'][1]} classes)\n"
        f"- near-duplicate pairs: {s['n_near_duplicate_pairs']}\n"
        f"- context-dependent pairs: {s['n_context_dependent_pairs']}\n"
        f"- held-back classes: {s['n_held_back']} (seed={config.TAXONOMY_SEED})\n"
        f"- held-back class_ids: {sorted(held_back_ids)}\n"
    )

    print(f"categories: {s['n_categories']}  documentTypes: {s['n_document_types']}  "
          f"leaf classes: {s['n_classes']}")
    print(f"deepest fan-out: {s['deepest_fanout'][0]} ({s['deepest_fanout'][1]} classes)")
    print(f"near-duplicate pairs: {s['n_near_duplicate_pairs']}  "
          f"context-dependent pairs: {s['n_context_dependent_pairs']}")
    print(f"held-back: {s['n_held_back']} (seed={config.TAXONOMY_SEED})")
    print(f"-> wrote {DEFAULT_OUT_PATH}")
    print(f"-> wrote {DEFAULT_STATS_PATH}")
    return 0
