"""Retrieval scoring: evidence matching, recall@k, MRR and RRF fusion.

Everything here is pure-python and model-free, so it is unit-testable without
Milvus, embeddings or an API key.
"""

from __future__ import annotations

import re
from collections import Counter

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Collapse all whitespace runs to single spaces — PDF line breaks must not
    break evidence matching."""
    return _WS.sub(" ", text).strip()


def correct_chunk_ids(records: list[dict], evidence: str) -> set[str]:
    """Ids of chunks whose text contains the evidence quote (normalized)."""
    ev = normalize(evidence)
    return {r["chunk_id"] for r in records if ev in normalize(r["text"])}


def recall_at_k(ranked_ids: list[str], correct: set[str], k: int) -> bool:
    """Did any correct chunk make it into the first k positions?"""
    return bool(set(ranked_ids[:k]) & correct)


def reciprocal_rank(ranked_ids: list[str], correct: set[str]) -> float:
    """1/rank of the first correct chunk, 0 if none."""
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in correct:
            return 1.0 / rank
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion.

    score(d) = sum over rankings of 1 / (k + rank(d)), rank starting at 1.
    Ties are broken alphabetically for determinism.
    """
    scores: Counter[str] = Counter()
    for ranking in rankings:
        for pos, cid in enumerate(ranking):
            scores[cid] += 1.0 / (k + pos + 1)
    return sorted(scores, key=lambda c: (-scores[c], c))
