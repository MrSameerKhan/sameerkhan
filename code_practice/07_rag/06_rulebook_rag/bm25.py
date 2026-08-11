"""Okapi BM25, written out by hand. No library (section 6.1).

    score(q, d) = SUM over query terms t of   idf(t) * tf_saturated(t, d)

                        N - df(t) + 0.5
    idf(t) = ln( ------------------------- + 1 )
                        df(t) + 0.5

                            f(t,d) * (k1 + 1)
    tf_saturated(t,d) = ---------------------------------------
                        f(t,d) + k1 * (1 - b + b * |d| / avgdl)

Two ideas carry the whole formula:

  saturation (k1)         the 2nd occurrence of a term says a lot more than the 10th.
                          tf_saturated rises towards an asymptote of (k1 + 1) instead
                          of growing linearly with raw count.

  length normalisation (b) a long document contains more of everything, so raw counts
                          favour it unfairly. Dividing |d| by avgdl discounts long
                          documents; b = 0 disables it, b = 1 applies it fully.

The "+ 1" inside idf is the Lucene variant. Classic BM25 idf goes negative for terms
appearing in more than half the corpus, which lets a common term subtract from a
document's score. With a 36-class rulebook and clauses sharing tax vocabulary, that
happens often, so the non-negative form is the right choice here.
"""

import math
import re

import config

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs, k1: float = None, b: float = None):
        """docs: list of token lists, one per indexed clause."""
        self.k1 = config.BM25_K1 if k1 is None else k1
        self.b = config.BM25_B if b is None else b
        self.docs = docs
        self.n_docs = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0

        self.tf = []                      # per doc: {term: count}
        df = {}                           # term -> number of docs containing it
        for d in docs:
            counts = {}
            for t in d:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in counts:
                df[t] = df.get(t, 0) + 1

        self.idf = {
            t: math.log((self.n_docs - n + 0.5) / (n + 0.5) + 1.0)
            for t, n in df.items()
        }

    def score(self, query_tokens) -> list:
        scores = [0.0] * self.n_docs
        for t in query_tokens:
            idf = self.idf.get(t)
            if idf is None:               # term absent from the corpus
                continue
            for i in range(self.n_docs):
                f = self.tf[i].get(t, 0)
                if not f:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / (f + norm)
        return scores

    def top(self, query_tokens, n: int) -> list:
        scored = list(enumerate(self.score(query_tokens)))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [(i, s) for i, s in scored[:n] if s > 0.0]


def _selftest() -> int:
    """Known-answer fixture. The expected values below were derived by hand from the
    formula in the module docstring, not captured from this implementation - so a bug
    here fails the test rather than being baked into it."""
    docs = [tokenize(t) for t in [
        "the quick brown fox",          # d0: len 4, one 'quick', one 'fox'
        "the lazy brown dog",           # d1: len 4, no query terms
        "the quick quick fox fox fox",  # d2: len 6, two 'quick', three 'fox'
    ]]
    bm = BM25(docs, k1=1.5, b=0.75)

    # N = 3; 'quick' and 'fox' each appear in 2 of 3 docs
    #   idf = ln((3 - 2 + 0.5) / (2 + 0.5) + 1) = ln(1.6) = 0.4700036
    expect_idf = math.log(1.6)
    assert abs(bm.idf["quick"] - expect_idf) < 1e-9, bm.idf["quick"]
    assert abs(bm.idf["fox"] - expect_idf) < 1e-9, bm.idf["fox"]

    # avgdl = (4 + 4 + 6) / 3 = 4.666667
    assert abs(bm.avgdl - 14 / 3) < 1e-9, bm.avgdl

    scores = bm.score(tokenize("quick fox"))

    # d0: norm = 1.5 * (0.25 + 0.75 * 4/4.666667) = 1.3392857
    #     each term: 1 * 2.5 / (1 + 1.3392857) = 1.0687023
    #     score = 0.4700036 * (1.0687023 * 2) = 1.0045923
    d0_norm = 1.5 * (1 - 0.75 + 0.75 * 4 / (14 / 3))
    d0_expect = expect_idf * 2 * (1 * 2.5 / (1 + d0_norm))

    # d2: norm = 1.5 * (0.25 + 0.75 * 6/4.666667) = 1.8214286
    #     quick: 2 * 2.5 / (2 + 1.8214286) = 1.3084046
    #     fox:   3 * 2.5 / (3 + 1.8214286) = 1.5555063
    d2_norm = 1.5 * (1 - 0.75 + 0.75 * 6 / (14 / 3))
    d2_expect = expect_idf * ((2 * 2.5 / (2 + d2_norm)) + (3 * 2.5 / (3 + d2_norm)))

    checks = [
        ("d0 score", scores[0], d0_expect),
        ("d1 score", scores[1], 0.0),
        ("d2 score", scores[2], d2_expect),
    ]
    ok = True
    for label, got, want in checks:
        good = abs(got - want) < 1e-9
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'}  {label}: {got:.7f}  (expected {want:.7f})")

    # Saturation sanity: d2 has 5 query-term occurrences to d0's 2, but scores far
    # less than 2.5x - that is the point of the k1 term.
    ratio = scores[2] / scores[0]
    sat_ok = 1.0 < ratio < 1.6
    ok &= sat_ok
    print(f"  {'ok  ' if sat_ok else 'FAIL'}  saturation: d2/d0 = {ratio:.3f} "
          f"(5 occurrences vs 2, but ratio well under 2.5)")

    # Ranking must be d2 > d0 > d1
    rank_ok = scores[2] > scores[0] > scores[1]
    ok &= rank_ok
    print(f"  {'ok  ' if rank_ok else 'FAIL'}  ranking: d2 > d0 > d1")

    top = bm.top(tokenize("quick fox"), 5)
    top_ok = [i for i, _ in top] == [2, 0]
    ok &= top_ok
    print(f"  {'ok  ' if top_ok else 'FAIL'}  top() drops zero-score docs: "
          f"{[i for i, _ in top]}")

    print(f"\n  -> {'BM25 SELFTEST PASSED' if ok else 'BM25 SELFTEST FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
