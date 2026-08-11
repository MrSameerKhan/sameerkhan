"""Metrics, the six ablations, and the RESULTS tables (section 7).

Every metric here is reported in the pair section 7 demands: accuracy on ANSWERED cases
and accuracy on ALL cases with abstentions counted wrong. Reporting only the first is
how RAG numbers get inflated - a system that abstains on everything hard scores 100%
answered and is useless.

recall@8 is the ceiling on accuracy. If accuracy is 0.72 and recall@8 is 0.74, no amount
of prompt work will help; the true class simply is not in the candidate set. That single
comparison decides where the next day of work should go.
"""

import json
import statistics
import time

import config
from decide import classify_page
from extract import load_taxonomy
from index import load_index, load_rulebook
from llm import USAGE, get_llm
from retrieve import Retriever, load_pages

RUNS = ("bm25_only", "dense_only", "hybrid", "hybrid_rerank",
        "names_only", "prompt_stuffing")


def eval_cases(only_heldback=False, exclude_heldback=False) -> list:
    tax = {c["class_id"]: c for c in load_taxonomy()}
    out = []
    for r in load_pages().values():
        if not r.get("eval_case"):
            continue
        held = tax[r["class_id"]]["held_back"]
        if only_heldback and not held:
            continue
        if exclude_heldback and held:
            continue
        out.append(r)
    return sorted(out, key=lambda r: r["page_id"])


def build_redactor():
    """Strip a page's self-identification: its form number, "Form X"/"Schedule X", any
    class's proper title, and the irs.gov URL that names the form.

    Why this ablation exists. Ablation 5 showed that class names alone beat the full
    system, because every eval page prints its own name and the model has memorised
    these forms. That measures contamination, not the method. Redacting the page's
    self-identification removes the memory channel and asks the question the project is
    actually about: when the model CANNOT recognise the document, does the rulebook
    carry the decision?

    This is not a claim about production documents - real forms do print their titles.
    It isolates content-based classification from recognition.
    """
    import re as _re
    from rules import SCRUBS, title_of

    titles = sorted({title_of(c["real_name"]) for c in load_taxonomy()},
                    key=len, reverse=True)
    pats = [_re.compile(_re.escape(t), _re.I) for t in titles if len(t) > 10]
    pats += [p for p, _ in SCRUBS]
    pats.append(_re.compile(r"irs\.gov/\S+", _re.I))

    def redact(text: str) -> str:
        for p in pats:
            text = p.sub("[REDACTED]", text)
        return text

    return redact


def rank_of(true_class: str, candidates: list) -> int:
    """1-based rank of the true class in the retrieved list, 0 if absent."""
    for i, c in enumerate(candidates, start=1):
        if c == true_class:
            return i
    return 0


def compute_metrics(rows: list, index_rule_ids: set, ranked: bool = True) -> dict:
    """ranked=False for the no-retrieval ablations. They have no candidate ordering, so
    recall@k and MRR are undefined - reporting the position of the true class in a
    sorted class list would be a fabricated number, not a weak one."""
    n = len(rows)
    if not n:
        return {}

    answered = [r for r in rows if not r["abstain"]]
    correct_answered = [r for r in answered if r["class_id"] == r["true_class"]]

    ranks = [r["rank"] for r in rows]
    def recall_at(k):
        return sum(1 for x in ranks if 0 < x <= k) / n

    cites = [c for r in rows for c in r["citations"]]
    bad_cites = [c for c in cites if c not in index_rule_ids]
    fabricated_rows = [r for r in rows
                       if any(c not in index_rule_ids for c in r["citations"])]

    lat = [r["latency_s"] for r in rows if r.get("latency_s")]

    return {
        "n": n,
        "ranked": ranked,
        "answered": len(answered),
        "coverage": len(answered) / n,
        "acc_answered": (len(correct_answered) / len(answered)) if answered else 0.0,
        "acc_all": len(correct_answered) / n,
        "recall@1": recall_at(1) if ranked else None,
        "recall@3": recall_at(3) if ranked else None,
        "recall@5": recall_at(5) if ranked else None,
        "recall@8": recall_at(8) if ranked else None,
        "mrr": (sum(1.0 / x for x in ranks if x > 0) / n) if ranked else None,
        "citations_total": len(cites),
        "citations_invalid": len(bad_cites),
        "citation_validity": (1 - len(bad_cites) / len(cites)) if cites else 1.0,
        "fabrication_rate": len(fabricated_rows) / n,
        "forced_abstain_rate": sum(1 for r in rows if r["forced_abstain"]) / n,
        "abstain_rate": 1 - len(answered) / n,
        "p50_latency": statistics.median(lat) if lat else 0.0,
        "p95_latency": (sorted(lat)[min(len(lat) - 1, int(0.95 * len(lat)))]
                        if lat else 0.0),
    }


def run_one(run: str, cases: list, client, real_names: bool, limit=None,
            redact=False) -> list:
    """Execute one ablation over the eval cases."""
    tax = {c["class_id"]: c for c in load_taxonomy()}
    records, _, _, manifest = load_index()
    index_rule_ids = {r["rule_id"] for r in records}

    # The no-retrieval ablations must see exactly the classes that are IN the index.
    # Reading them off disk instead would put held-back classes back into the prompt
    # and silently void the zero-shot result (section 7).
    indexed = set(manifest["classes_indexed"])

    style = "rules"
    retriever = None
    name_map = entries = None
    if run == "names_only":
        style = "names_only"
        name_map = {c["class_id"]: c["real_name"]
                    for c in tax.values() if c["class_id"] in indexed}
    elif run == "prompt_stuffing":
        style = "stuffing"
        entries = [e for e in load_rulebook() if e["class_id"] in indexed]
    else:
        retriever = Retriever(mode=run)

    names = {c["class_id"]: c["real_name"] for c in tax.values()} if real_names else None

    redactor = build_redactor() if redact else None
    if redactor:
        sample = cases[0]["text"]
        print(f"      redaction on: sample page {len(sample)} chars -> "
              f"{len(redactor(sample))} chars, "
              f"{redactor(sample).count('[REDACTED]')} spans removed")

    rows = []
    for i, page in enumerate(cases[:limit], 1):
        t0 = time.monotonic()
        text = redactor(page["text"]) if redactor else page["text"]
        out = classify_page(text, retriever, client, index_rule_ids,
                            real_names=names, style=style,
                            name_map=name_map, entries=entries)
        rows.append({
            "page_id": page["page_id"],
            "true_class": page["class_id"],
            "class_id": out["class_id"],
            "proposed_class_id": out["proposed_class_id"],
            "abstain": out["abstain"],
            "forced_abstain": out["forced_abstain"],
            "reason": out["reason"],
            "confidence": out["confidence"],
            "citations": out["citations"],
            "reasoning": out["reasoning"],
            "rank": rank_of(page["class_id"], out["candidates"]),
            "latency_s": round(time.monotonic() - t0, 3),
        })
        if i % 10 == 0:
            print(f"      {i}/{len(cases[:limit])} ...", flush=True)
    return rows


def print_metrics(label: str, m: dict):
    print(f"\n  === {label}  (n={m['n']}) ===")
    print(f"    accuracy answered ... {m['acc_answered']:.3f}"
          f"    accuracy ALL ....... {m['acc_all']:.3f}")
    print(f"    coverage ............ {m['coverage']:.3f}"
          f"    abstain rate ....... {m['abstain_rate']:.3f}")
    if m.get("ranked", True):
        print(f"    recall@1/3/5/8 ...... {m['recall@1']:.3f} / {m['recall@3']:.3f} / "
              f"{m['recall@5']:.3f} / {m['recall@8']:.3f}      MRR {m['mrr']:.3f}")
        ceiling = m["recall@8"] - m["acc_all"]
        print(f"    ceiling gap ......... recall@8 - accuracy_all = {ceiling:+.3f}"
              f"   {'<- retrieval is the bottleneck' if ceiling < 0.05 else '<- headroom above retrieval'}")
    else:
        print("    recall@k / MRR ...... n/a (no retrieval in this ablation)")
    print(f"    citation validity ... {m['citation_validity']:.3f}"
          f"   ({m['citations_invalid']}/{m['citations_total']} invalid)")
    print(f"    fabrication rate .... {m['fabrication_rate']:.3f}"
          f"    forced abstain ..... {m['forced_abstain_rate']:.3f}")
    print(f"    latency p50/p95 ..... {m['p50_latency']:.2f}s / {m['p95_latency']:.2f}s")


def cmd_eval(args):
    if args.limit:
        print("  WARNING: --limit takes the FIRST N cases, which is not a random "
              "sample. Use it for smoke tests only, never for a reported number.")

    cases = eval_cases(only_heldback=args.only_heldback,
                       exclude_heldback=args.exclude_heldback)
    if not cases:
        print("  no eval cases match those filters")
        return 2

    records, _, _, manifest = load_index()
    index_rule_ids = {r["rule_id"] for r in records}
    client = get_llm(use_stub=args.stub_llm)

    # Section 7 asserts this in code: a held-back class must be absent from the index.
    if args.only_heldback:
        held = set(manifest["held_back"])
        indexed = set(manifest["classes_indexed"])
        missing = held - indexed
        if missing:
            print(f"  ERROR: held-back classes are NOT in the index: "
                  f"{sorted(missing)}\n  run `cli add-held-back` first")
            return 2
    if args.exclude_heldback and not manifest["held_back_excluded"]:
        print("  WARNING: index still contains the held-back classes. For the honest "
              "main number run `cli build-index --exclude-held-back` first.")

    runs = [args.mode] if args.mode else list(args.runs or [config.DEFAULT_MODE])
    label_bits = []
    if args.only_heldback:
        label_bits.append("held-back only")
    if args.real_names:
        label_bits.append("real_names")
    if args.redact_pages:
        label_bits.append("redacted")
    suffix = f" [{', '.join(label_bits)}]" if label_bits else ""

    config.PATHS.results_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    for run in runs:
        print(f"\n  running {run}{suffix} over {len(cases[:args.limit] if args.limit else cases)} cases ...")
        rows = run_one(run, cases, client, args.real_names, args.limit,
                       redact=args.redact_pages)
        m = compute_metrics(rows, index_rule_ids,
                            ranked=run not in ('names_only', 'prompt_stuffing'))
        all_metrics[run] = m
        print_metrics(run + suffix, m)

        tag = run + ("_heldback" if args.only_heldback else "") + \
            ("_realnames" if args.real_names else "") + \
            ("_redacted" if args.redact_pages else "")
        (config.PATHS.results_dir / f"eval_{tag}.json").write_text(
            json.dumps({"run": run, "suffix": suffix, "metrics": m, "rows": rows},
                       indent=2))

    if len(all_metrics) > 1:
        print("\n  === ABLATION TABLE ===")
        print(f"    {'run':18} {'acc_ans':>8} {'acc_all':>8} {'cover':>7} "
              f"{'r@1':>6} {'r@8':>6} {'fabric':>7}")
        for run, m in all_metrics.items():
            r1 = f"{m['recall@1']:6.3f}" if m.get("ranked", True) else "   n/a"
            r8 = f"{m['recall@8']:6.3f}" if m.get("ranked", True) else "   n/a"
            print(f"    {run:18} {m['acc_answered']:8.3f} {m['acc_all']:8.3f} "
                  f"{m['coverage']:7.3f} {r1} {r8} "
                  f"{m['fabrication_rate']:7.3f}")

    print(f"\n  {USAGE.summary()}")
    n_cases = len(cases[:args.limit] if args.limit else cases) * len(runs)
    if n_cases:
        print(f"  cost per 100 pages: ${100 * USAGE.cost_usd / n_cases:.3f}")
    return 0


def build_subparser(sub):
    p = sub.add_parser("eval", help="Run the eval set and report section 7 metrics.")
    p.add_argument("--mode", default=None, choices=list(RUNS),
                   help="Run a single ablation.")
    p.add_argument("--runs", nargs="*", default=None, choices=list(RUNS),
                   help="Run several ablations and print the comparison table.")
    p.add_argument("--only-heldback", action="store_true",
                   help="The zero-shot headline: only the held-back classes.")
    p.add_argument("--exclude-heldback", action="store_true",
                   help="The main number: skip held-back cases.")
    p.add_argument("--real-names", action="store_true",
                   help="Show real class names in the prompt (leakage measurement).")
    p.add_argument("--limit", type=int, default=None,
                   help="Smoke test only - takes the FIRST N, not a sample.")
    p.add_argument("--redact-pages", action="store_true",
                   help="Strip the page's own form number/title before classifying - "
                        "removes the memorisation channel (see build_redactor).")
    p.add_argument("--stub-llm", action="store_true")
    p.set_defaults(func=cmd_eval)
