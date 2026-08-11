"""Rulebook -> searchable index. One vector per RULE CLAUSE, not per class (section 6.2).

Why per clause: a class has 8-11 clauses covering different observable features. Averaging
them into one class vector blurs exactly the detail that separates confusable siblings.
Retrieving clauses and then rolling them up to a class score (max, in retrieve.py) keeps
the discriminating clause visible.

Why the context prefix: a bare clause is ambiguous once detached from its class. "Reports
income allocated among several partners" could belong to three classes. Prefixing with
class id and document type restores the context the embedding needs.

    {class_id} | {document_type} | {field}: {clause text} | quote: {quote}

The quote is included deliberately. It is literal text printed on the page, so for the
lexical retriever it is the single most matchable signal available - leaving it out would
cripple the bm25_only ablation and understate what lexical retrieval can do.
"""

import hashlib
import json
import pickle
import time

import numpy as np

import config
from bm25 import tokenize
from extract import load_taxonomy

FIELDS = (("includes", "includes"),
          ("excludes", "excludes"),
          ("discriminators", "discriminator"))


def load_rulebook() -> list:
    entries = []
    for p in sorted(config.PATHS.rulebook_dir.glob("CLASS_*.json")):
        entries.append(json.loads(p.read_text()))
    return entries


def rulebook_hash(entries: list) -> str:
    h = hashlib.sha256()
    for e in sorted(entries, key=lambda x: x["class_id"]):
        h.update(json.dumps(e, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:16]


def clause_records(entries: list, skip: set) -> list:
    """Flatten the rulebook into one record per clause, with its context prefix."""
    out = []
    for e in entries:
        cid = e["class_id"]
        if cid in skip:
            continue
        doctype = e.get("document_type", "unknown")
        for field, label in FIELDS:
            for c in e.get(field, []):
                text = c.get("text", "").strip()
                quote = c.get("quote", "").strip()
                prefixed = f"{cid} | {doctype} | {label}: {text}"
                if quote:
                    prefixed += f" | quote: {quote}"
                out.append({
                    "rule_id": c["rule_id"],
                    "class_id": cid,
                    "field": label,
                    "text": text,
                    "quote": quote,
                    "vs_class_id": c.get("vs_class_id", ""),
                    "prefixed": prefixed,
                })
    return out


def cmd_build_index(args):
    from sentence_transformers import SentenceTransformer

    tax = load_taxonomy()
    held = {c["class_id"] for c in tax if c["held_back"]}
    skip = held if args.exclude_held_back else set()

    entries = load_rulebook()
    if not entries:
        print("  no rulebook entries - run `cli build-rules` first")
        return 2

    records = clause_records(entries, skip)
    if not records:
        print("  no clauses to index")
        return 2

    indexed_classes = sorted({r["class_id"] for r in records})
    print(f"  classes indexed .... {len(indexed_classes)}"
          f"{'  (held-back EXCLUDED)' if skip else '  (all, held-back included)'}")
    print(f"  clauses indexed .... {len(records)}")

    print(f"  embedding with {config.EMBED_MODEL} ...")
    model = SentenceTransformer(config.EMBED_MODEL)
    vecs = model.encode([r["prefixed"] for r in records],
                        normalize_embeddings=True,
                        batch_size=64,
                        show_progress_bar=False)
    vecs = np.asarray(vecs, dtype=np.float32)

    config.PATHS.index_dir.mkdir(parents=True, exist_ok=True)
    np.save(config.PATHS.index_vectors, vecs)
    config.PATHS.index_ids.write_text(json.dumps(records, indent=2))

    # Store the tokenised docs rather than a pickled BM25 instance, so the index does
    # not break when the class definition changes.
    with config.PATHS.index_bm25.open("wb") as f:
        pickle.dump({"docs": [tokenize(r["prefixed"]) for r in records],
                     "k1": config.BM25_K1, "b": config.BM25_B}, f)

    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rulebook_hash": rulebook_hash(entries),
        "embed_model": config.EMBED_MODEL,
        "rerank_model": config.RERANK_MODEL,
        "bm25": {"k1": config.BM25_K1, "b": config.BM25_B},
        "held_back": sorted(held),
        "held_back_excluded": bool(skip),
        "classes_indexed": indexed_classes,
        "n_classes": len(indexed_classes),
        "n_clauses": len(records),
        "vector_dim": int(vecs.shape[1]),
    }
    config.PATHS.index_manifest.write_text(json.dumps(manifest, indent=2))

    print(f"  vectors ............ {vecs.shape[0]} x {vecs.shape[1]}")
    print(f"  rulebook hash ...... {manifest['rulebook_hash']}")
    if skip:
        print(f"  HELD OUT ........... {', '.join(sorted(held))}")
    print(f"  wrote {config.PATHS.index_dir}/ (vectors.npy, ids.json, bm25.pkl, "
          f"manifest.json)")
    return 0


def cmd_add_held_back(args):
    """Rebuild with every class present. No retraining, no fine-tuning - the held-back
    classes join the index purely by having their rulebook entries embedded (section 7)."""
    print("  adding held-back classes to the index (no retraining)")
    args.exclude_held_back = False
    return cmd_build_index(args)


def load_index():
    """Returns (records, vectors, bm25, manifest) for retrieve.py."""
    from bm25 import BM25
    records = json.loads(config.PATHS.index_ids.read_text())
    vecs = np.load(config.PATHS.index_vectors)
    with config.PATHS.index_bm25.open("rb") as f:
        blob = pickle.load(f)
    bm = BM25(blob["docs"], k1=blob["k1"], b=blob["b"])
    manifest = json.loads(config.PATHS.index_manifest.read_text())
    return records, vecs, bm, manifest


def build_subparsers(sub):
    p = sub.add_parser("build-index", help="rulebook -> vectors + bm25 + manifest")
    p.add_argument("--exclude-held-back", action="store_true",
                   help="Leave the held-back classes out (the zero-shot setup).")
    p.set_defaults(func=cmd_build_index)

    p = sub.add_parser("add-held-back",
                       help="Rebuild the index with held-back classes included.")
    p.set_defaults(func=cmd_add_held_back)
