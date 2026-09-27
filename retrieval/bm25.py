"""Minimal Okapi BM25 over chunk texts — the "words" half of hybrid search.

Hand-rolled per the course rule (first by hand, then frameworks): tokenization
is lowercase + alphanumeric runs, scoring is the standard Okapi formula.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(
        self,
        ids: list[str],
        texts: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length")
        self.ids = list(ids)
        self.k1 = k1
        self.b = b
        docs = [tokenize(t) for t in texts]
        self.doc_len = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_len) / len(docs)) if docs else 0.0
        self.n = len(docs)
        self.df: Counter[str] = Counter()
        for d in docs:
            self.df.update(set(d))
        self.tfs = [Counter(d) for d in docs]

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> dict[str, float]:
        """BM25 score per document id (only documents with a nonzero score)."""
        q = tokenize(query)
        out: dict[str, float] = {}
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for term in q:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1.0 - self.b + self.b * self.doc_len[i] / self.avgdl)
                s += self._idf(term) * f * (self.k1 + 1.0) / denom
            if s > 0.0:
                out[self.ids[i]] = s
        return out

    def search(self, query: str, top_k: int = 10) -> list[str]:
        ranked = sorted(self.scores(query).items(), key=lambda kv: (-kv[1], kv[0]))
        return [cid for cid, _ in ranked[:top_k]]
