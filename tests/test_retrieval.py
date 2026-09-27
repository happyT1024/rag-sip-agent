"""Tests for retrieval scoring, BM25 and RRF — pure python, no Milvus/model/key."""

from __future__ import annotations

from retrieval.bm25 import BM25Index, tokenize
from retrieval.scoring import (
    correct_chunk_ids,
    normalize,
    recall_at_k,
    reciprocal_rank,
    rrf_fuse,
)

RECORDS = [
    {"chunk_id": "a", "text": "The INVITE transaction consists of a\nthree-way handshake."},
    {"chunk_id": "b", "text": "PRACK  plays  the same role as ACK, but for provisional responses."},
    {"chunk_id": "c", "text": "MESSAGE requests do not establish dialogs."},
]


def test_normalize_collapses_whitespace() -> None:
    assert normalize("a\n  b\t c") == "a b c"


def test_evidence_match_across_line_breaks() -> None:
    ev = "The INVITE transaction consists of a three-way handshake."
    assert correct_chunk_ids(RECORDS, ev) == {"a"}


def test_recall_and_rr() -> None:
    correct = {"b"}
    assert recall_at_k(["c", "b", "a"], correct, 1) is False
    assert recall_at_k(["c", "b", "a"], correct, 2) is True
    assert reciprocal_rank(["c", "b", "a"], correct) == 0.5
    assert reciprocal_rank(["b", "a"], correct) == 1.0
    assert reciprocal_rank(["c", "a"], correct) == 0.0


def test_tokenize_lowercases_and_drops_punct() -> None:
    assert tokenize("PRACK = ACK, but for 1xx!") == ["prack", "ack", "but", "for", "1xx"]


def test_bm25_ranks_relevant_doc_first() -> None:
    index = BM25Index([r["chunk_id"] for r in RECORDS], [r["text"] for r in RECORDS])
    assert index.search("PRACK provisional responses", top_k=3)[0] == "b"
    assert index.search("MESSAGE dialogs", top_k=3)[0] == "c"
    # A term absent from the corpus yields no hits.
    assert index.search("stunwebrtc", top_k=3) == []


def test_rrf_prefers_doc_high_in_both_lists() -> None:
    vec = ["a", "b", "c"]
    words = ["b", "d", "a"]
    fused = rrf_fuse([vec, words], k=60)
    assert fused[0] == "b"  # 2nd in both beats 1st in one
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_deterministic_tie_break() -> None:
    assert rrf_fuse([["x", "y"]], k=10) == ["x", "y"]
    assert rrf_fuse([["y", "x"]], k=10) == ["y", "x"]
