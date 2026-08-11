"""Page text -> top-K candidate classes with their supporting clauses (section 6.3).

Flat retrieval, no hierarchy. Section 6 cuts the two-stage design deliberately: at this
taxonomy size a wrong stage-1 pick is unrecoverable and buys nothing.

Pipeline, with `mode` as the single ablation switch:

    bm25_only      lexical ranking over clauses
    dense_only     cosine over clause embeddings
    hybrid         reciprocal rank fusion of both
    hybrid_rerank  hybrid, then a cross-encoder re-scores the fused top-N

RRF fuses on RANKS, never scores:  score(d) = SUM over lists of 1 / (RRF_K + rank)
A bm25 score and a cosine live on incompatible scales; ranks are all they share, so
fusing this way avoids inventing a normalisation nobody can justify.

Class score = MAX over that class's clauses, not mean. One clause matching strongly is
the signal; averaging it against the class's other clauses dilutes exactly the
discriminating evidence we indexed per clause to preserve.
"""

import json

import numpy as np

import config
from bm25 import tokenize
from index import load_index


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Retriever:
    def __init__(self, mode: str = None, verbose: bool = False):
        self.mode = mode or config.DEFAULT_MODE
        if self.mode not in config.MODES:
            raise ValueError(f"mode must be one of {config.MODES}, got {self.mode!r}")
        self.verbose = verbose
        self.records, self.vecs, self.bm, self.manifest = load_index()
        self._embed = None
        self._rerank = None

    # models are loaded lazily so bm25_only never pays for them
    @property
    def embed(self):
        if self._embed is None:
            from sentence_transformers import SentenceTransformer
            self._embed = SentenceTransformer(config.EMBED_MODEL)
        return self._embed

    @property
    def reranker(self):
        if self._rerank is None:
            from sentence_transformers import CrossEncoder
            self._rerank = CrossEncoder(config.RERANK_MODEL)
        return self._rerank

    def _bm25_ranked(self, page_text: str, n: int):
        return self.bm.top(tokenize(page_text), n)

    def _dense_ranked(self, page_text: str, n: int):
        # bge-small truncates at 512 tokens; feed it the head of the page, which is
        # where forms put their identifying field labels.
        q = self.embed.encode([page_text[:config.MAX_QUERY_CHARS]],
                              normalize_embeddings=True)
        sims = (self.vecs @ np.asarray(q, dtype=np.float32).T).ravel()
        order = np.argsort(-sims)[:n]
        return [(int(i), float(sims[i])) for i in order]

    @staticmethod
    def _rrf(*ranked_lists):
        fused = {}
        for lst in ranked_lists:
            for rank, (idx, _) in enumerate(lst, start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (config.RRF_K + rank)
        return sorted(fused.items(), key=lambda x: (-x[1], x[0]))

    def _rerank_pairs(self, page_text: str, candidates):
        q = page_text[:config.MAX_QUERY_CHARS]
        pairs = [(q, self.records[i]["prefixed"]) for i, _ in candidates]
        raw = np.asarray(self.reranker.predict(pairs), dtype=np.float64)
        # Cross-encoder returns raw logits, often all negative. Squash to (0,1) so the
        # confidence margin (top1 - top2)/top1 in decide.py stays well defined.
        scores = _sigmoid(raw)
        out = [(idx, float(s)) for (idx, _), s in zip(candidates, scores)]
        out.sort(key=lambda x: (-x[1], x[0]))
        return out

    def clause_ranking(self, page_text: str):
        n = config.RETRIEVE_TOP_N
        if self.mode == "bm25_only":
            return self._bm25_ranked(page_text, n)
        if self.mode == "dense_only":
            return self._dense_ranked(page_text, n)

        fused = self._rrf(self._bm25_ranked(page_text, n),
                          self._dense_ranked(page_text, n))[:n]
        if self.mode == "hybrid":
            return fused
        return self._rerank_pairs(page_text, fused)

    def retrieve(self, page_text: str, k: int = None):
        """Returns (candidates, debug). candidates: list of
        {class_id, score, rules:[record...]} sorted best first, length <= k."""
        k = k or config.K
        ranked = self.clause_ranking(page_text)

        by_class = {}
        for idx, score in ranked:
            rec = self.records[idx]
            cid = rec["class_id"]
            slot = by_class.setdefault(cid, {"class_id": cid, "score": 0.0, "rules": []})
            slot["score"] = max(slot["score"], max(score, 0.0))   # MAX, not mean
            slot["rules"].append(rec)

        cands = sorted(by_class.values(), key=lambda c: (-c["score"], c["class_id"]))[:k]
        debug = {
            "mode": self.mode,
            "clauses_ranked": len(ranked),
            "classes_found": len(by_class),
            "top_clauses": [
                {"rule_id": self.records[i]["rule_id"], "score": round(s, 4)}
                for i, s in ranked[:8]
            ],
        }
        return cands, debug


def retrieval_margin(cands) -> float:
    """(top1 - top2) / top1, the separation between the best and second-best class.
    Feeds the confidence blend in section 6.6."""
    if not cands:
        return 0.0
    top1 = cands[0]["score"]
    if top1 <= 0:
        return 0.0
    top2 = cands[1]["score"] if len(cands) > 1 else 0.0
    return max(0.0, (top1 - top2) / top1)


def load_pages() -> dict:
    return {r["page_id"]: r
            for r in (json.loads(l) for l in config.PATHS.pages_jsonl.open())}


def cmd_retrieve(args):
    """Retrieval only, no LLM, no cost - for inspecting the candidate set."""
    pages = load_pages()
    if args.page_id not in pages:
        evals = [p for p, r in pages.items() if r.get("eval_case")]
        print(f"  unknown page_id. {len(evals)} eval pages, e.g. {evals[:3]}")
        return 2

    page = pages[args.page_id]
    r = Retriever(mode=args.mode)
    cands, debug = r.retrieve(page["text"])

    print(f"  page {args.page_id}  (true class {page['class_id']}, "
          f"{page['n_chars']} chars)")
    print(f"  mode {debug['mode']}  |  index {r.manifest['n_classes']} classes, "
          f"{r.manifest['n_clauses']} clauses")
    print(f"  clauses ranked {debug['clauses_ranked']}, "
          f"classes touched {debug['classes_found']}\n")
    for i, c in enumerate(cands, 1):
        hit = "  <== TRUE" if c["class_id"] == page["class_id"] else ""
        print(f"  {i}. {c['class_id']}  score {c['score']:.4f}  "
              f"({len(c['rules'])} clauses){hit}")
        if args.verbose:
            for rec in c["rules"][:3]:
                print(f"        {rec['rule_id']:24} {rec['quote'][:60]!r}")
    rank = next((i for i, c in enumerate(cands, 1)
                 if c["class_id"] == page["class_id"]), None)
    print(f"\n  true class rank: {rank if rank else f'NOT IN TOP {config.K}'}")
    print(f"  retrieval margin: {retrieval_margin(cands):.3f}")
    return 0


def build_subparser(sub):
    p = sub.add_parser("retrieve", help="Retrieval only for one page (no LLM, free).")
    p.add_argument("--page-id", required=True)
    p.add_argument("--mode", default=None, choices=list(config.MODES))
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_retrieve)
