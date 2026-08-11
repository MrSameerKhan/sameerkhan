"""data/real/documents.jsonl + taxonomy -> data/synthetic/documents.jsonl. §22.7.

🔴 HARD CONSTRAINT: this module must not import anything from rulebook/, and must not receive
definitions, includes, excludes or discriminators. Its only class-level input is the class
*name* and the *real form text* (when available — see decide/prompts.py's no-real-text
fallback for the ~78 classes with no downloadable form). Enforced by tests/test_no_rulebook_import.py.

One LLM call per variant, not N-variants-in-one-response: some fetched "form" PDFs turned out to
be 40-60K characters (bundled worksheets/multiple copies), and asking for 8 full variants of a
60K-char document in a single JSON response has no realistic token budget. Per-variant calls keep
each call's size bounded to one document regardless of N, and a failure on variant 5 doesn't lose
variants 1-4. The source text fed into the prompt is also truncated (head, MAX_SEED_CHARS) — for a
blank template the headings/field labels at the top are what matters; the model doesn't need the
entire bundled PDF to produce one plausible filled variant.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import config
from decide.prompts import (
    SYNTHETIC_GEN_SYSTEM,
    SYNTHETIC_GEN_SYSTEM_NO_REAL_TEXT,
    build_synthetic_user_no_real_text,
)
from llm import get_llm

MAX_SEED_CHARS = 6000

SINGLE_VARIANT_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


def _truncate_seed(real_form_text: str) -> str:
    if len(real_form_text) <= MAX_SEED_CHARS:
        return real_form_text
    return real_form_text[:MAX_SEED_CHARS]


def build_generation_prompt(real_form_text: str, class_name: str, n_variants: int) -> str:
    """n_variants kept in the signature per §22.7, but each call now produces exactly one
    variant — see generate_for_class's loop. n_variants informs phrasing only ("one of several
    variants you'll be asked for") so the model still varies values across calls."""
    if real_form_text:
        seed = _truncate_seed(real_form_text)
        return (
            f"Blank form text (class: {class_name}):\n\n{seed}\n\n"
            f"Produce ONE filled-in variant of this form as plain text. "
            f"(This is one of {n_variants} variants being generated separately — vary the "
            f"fictitious values you choose from what a typical example would use.)"
        )
    return build_synthetic_user_no_real_text(class_name, 1)


def _load_real_text_by_class() -> dict:
    if not config.PATHS.real_docs.exists():
        return {}
    out = {}
    with open(config.PATHS.real_docs) as f:
        for line in f:
            r = json.loads(line)
            out[r["true_class_id"]] = r["text"]
    return out


def _load_taxonomy_classes() -> list[dict]:
    tax = json.loads(config.PATHS.taxonomy_json.read_text())
    out = []
    for cat in tax["categories"]:
        for dt in cat["document_types"]:
            for cls in dt["classes"]:
                out.append({**cls, "document_type_id": dt["document_type_id"]})
    return out


def _max_tokens_for_one(real_text: Optional[str]) -> int:
    seed_chars = len(_truncate_seed(real_text)) if real_text else 800
    # /3 chars-per-token (not the usual /4) covers JSON string-escaping inflation (quotes, \n).
    return min(8000, max(2000, int(seed_chars / 3) + 1500))


def generate_for_class(class_id: str, class_name: str, document_type_id: str,
                        real_text: Optional[str], n: int, llm, model: Optional[str] = None) -> list[dict]:
    had_real_seed = bool(real_text)
    system = SYNTHETIC_GEN_SYSTEM if had_real_seed else SYNTHETIC_GEN_SYSTEM_NO_REAL_TEXT
    max_tokens = _max_tokens_for_one(real_text)

    records = []
    for i in range(1, n + 1):
        user = build_generation_prompt(real_text or "", class_name, n)
        try:
            result = llm.call(system=system, user=user, schema=SINGLE_VARIANT_SCHEMA, model=model,
                               effort=config.EFFORT_GENERATE, max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001
            print(f"    {class_id} variant {i}/{n} failed: {type(e).__name__}: {e}")
            continue
        text = result["text"] if result else ""
        if not text:
            continue
        records.append({
            "doc_id": f"syn_{class_id}_{i:03d}",
            "text": text,
            "true_class_id": class_id,
            "true_document_type_id": document_type_id,
            "source": "synthetic",
            "generated_from": "form_text_only",  # §6.3 enum; had_real_seed tracks the nuance below
            "generator_model": model or config.LLM_MODEL_BULK,
            "page_count": 1,
            "had_real_seed": had_real_seed,
        })
    return records


def cmd_generate_synthetic(args: argparse.Namespace) -> int:
    real_text_by_class = _load_real_text_by_class()
    classes = _load_taxonomy_classes()
    if args.limit:
        classes = classes[: args.limit]

    llm = get_llm(use_stub=args.stub_llm)
    per_class = args.per_class

    if args.dry_run:
        sample = classes[:2]
        for c in sample:
            real_text = real_text_by_class.get(c["class_id"], "")
            prompt = build_generation_prompt(real_text, c["class_name"], per_class)
            print(f"=== class: {c['class_id']} (real_seed={'yes' if real_text else 'no'}, "
                  f"seed_chars={len(_truncate_seed(real_text)) if real_text else 0}, "
                  f"max_tokens={_max_tokens_for_one(real_text)}) ===")
            print(prompt[:500] + ("..." if len(prompt) > 500 else ""))
            print()
        total_calls = len(classes) * per_class
        avg_in = sum(len(_truncate_seed(real_text_by_class.get(c["class_id"], ""))) for c in classes) \
            // max(1, len(classes)) // 4
        avg_out = 1200  # rough chars/4 for one filled document
        from llm import estimate_cost
        cost = estimate_cost(total_calls, avg_in, avg_out, config.LLM_MODEL_BULK)
        print(f"{len(classes)} classes x {per_class} = {total_calls} calls (one variant per call)")
        print(f"projected cost (model={config.LLM_MODEL_BULK}): ${cost:.2f}")
        return 0

    all_records = []
    skipped = 0
    for c in classes:
        real_text = real_text_by_class.get(c["class_id"])
        recs = generate_for_class(c["class_id"], c["class_name"], c["document_type_id"],
                                   real_text, per_class, llm, model=config.LLM_MODEL_BULK)
        if not recs:
            skipped += 1
            continue
        for r in recs:
            if r["generated_from"] != "form_text_only":
                raise ValueError(f"invalid generated_from on {r['doc_id']}")
        all_records.extend(recs)
        print(f"  {c['class_id']}: {len(recs)}/{per_class}")

    config.PATHS.synthetic_docs.parent.mkdir(parents=True, exist_ok=True)
    with open(config.PATHS.synthetic_docs, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    print(f"{len(all_records)} generated · {skipped} classes skipped (0 variants succeeded) "
          f"-> {config.PATHS.synthetic_docs}")
    return 0
