"""Vectorstore tests: Milvus Lite (local file) + fake embedder.

No docker, no embedding model download, no API key, no LLM — per the course
requirement. The real Embedder cache logic is tested separately with a
counting stub instead of the model.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from pymilvus import MilvusClient

from vectorstore.embed import embed_texts
from vectorstore.ingest import ensure_collection, ingest_records, search

DIM = 8


def _vec(text: str) -> np.ndarray:
    """Deterministic per-text vector (md5 seed, normalized)."""
    seed = int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    v = rng.random(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeEmbedder:
    model_name = "fake-model"
    dim = DIM

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([_vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return _vec(text)


RECORDS = [
    {
        "chunk_id": "structural-3261-0007",
        "method": "structural",
        "rfc": 3261,
        "title": "SIP: Session Initiation Protocol",
        "text": "INVITE transaction rules",
        "section_id": "9.1",
        "section_path": "9.1 Client Behavior",
        "page_start": {"pdf": 100, "printed": 64},
        "page_end": {"pdf": 101, "printed": 65},
    },
    {
        "chunk_id": "structural-3261-0008",
        "method": "structural",
        "rfc": 3261,
        "title": "SIP: Session Initiation Protocol",
        "text": "REGISTER binding refresh",
        "section_id": "10.2",
        "section_path": "10.2 Registering",
        "page_start": {"pdf": 110, "printed": 74, },
        "page_end": {"pdf": 111, "printed": 75},
    },
    {
        "chunk_id": "naive-3515-0002",
        "method": "naive",
        "rfc": 3515,
        "title": "The SIP Refer Method",
        "text": "REFER method and Refer-To header",
        "section_id": None,
        "section_path": None,
        "page_start": {"pdf": 3, "printed": None},
        "page_end": {"pdf": 4, "printed": 3},
    },
]


@pytest.fixture
def lite_client(tmp_path: Path) -> MilvusClient:
    return MilvusClient(uri=str(tmp_path / "lite.db"))


@pytest.fixture
def filled(lite_client: MilvusClient) -> MilvusClient:
    ensure_collection(lite_client, "rag_structural", DIM)
    ingest_records(lite_client, FakeEmbedder(), RECORDS, "rag_structural")
    return lite_client


def test_ingest_row_count(filled: MilvusClient) -> None:
    assert filled.get_collection_stats("rag_structural")["row_count"] == 3


def test_recreate_drops_old_rows(lite_client: MilvusClient) -> None:
    assert ensure_collection(lite_client, "rag_structural", DIM)
    ingest_records(lite_client, FakeEmbedder(), RECORDS, "rag_structural")
    # Second ensure with recreate=True must start from zero.
    assert ensure_collection(lite_client, "rag_structural", DIM)
    assert lite_client.get_collection_stats("rag_structural")["row_count"] == 0
    # With recreate=False the existing collection is kept as is.
    assert not ensure_collection(lite_client, "rag_structural", DIM, recreate=False)


def test_search_finds_exact_text_top1(filled: MilvusClient) -> None:
    hits = search(filled, FakeEmbedder(), "rag_structural", "REGISTER binding refresh", limit=3)
    assert hits[0]["chunk_id"] == "structural-3261-0008"
    assert hits[0]["title"] == "SIP: Session Initiation Protocol"
    assert hits[0]["section_path"] == "10.2 Registering"
    assert hits[0]["page"] == 74


def test_search_with_filter_by_rfc(filled: MilvusClient) -> None:
    hits = search(
        filled,
        FakeEmbedder(),
        "rag_structural",
        "REGISTER binding refresh",
        limit=3,
        filter_expr="rfc == 3261",
    )
    assert {h["rfc"] for h in hits} == {3261}
    hits = search(
        filled,
        FakeEmbedder(),
        "rag_structural",
        "REGISTER binding refresh",
        limit=3,
        filter_expr="rfc == 9999",
    )
    assert hits == []


def test_search_with_filter_by_section(filled: MilvusClient) -> None:
    hits = search(
        filled,
        FakeEmbedder(),
        "rag_structural",
        "INVITE transaction rules",
        limit=3,
        filter_expr='section_path like "9.%"',
    )
    assert [h["chunk_id"] for h in hits] == ["structural-3261-0007"]


def test_nullable_metadata_survive(filled: MilvusClient) -> None:
    hits = search(
        filled,
        FakeEmbedder(),
        "rag_structural",
        "REFER method and Refer-To header",
        limit=1,
        filter_expr="rfc == 3515",
    )
    assert hits[0]["chunk_id"] == "naive-3515-0002"
    assert hits[0]["section_path"] is None
    assert hits[0]["page"] is None


class CountingEncoder:
    """Stands in for the real model: counts calls, returns deterministic vectors."""

    calls = 0

    def __call__(self, texts: list[str]) -> np.ndarray:
        CountingEncoder.calls += 1
        return np.stack([_vec(t) for t in texts])


def test_embed_cache_avoids_second_encode(tmp_path: Path) -> None:
    cache = tmp_path / "cache.sqlite3"
    encode = CountingEncoder()
    keys = lambda ts: [hashlib.sha256(t.encode()).hexdigest() for t in ts]

    first = embed_texts("m", ts := ["alpha", "beta"], keys(ts), encode, cache)
    assert CountingEncoder.calls == 1

    # Same texts: served from the cache, the encoder is never called.
    second = embed_texts("m", ts, keys(ts), encode, cache)
    assert CountingEncoder.calls == 1
    np.testing.assert_array_equal(first, second)

    # Different model namespace: same texts, cache miss.
    embed_texts("other", ts, keys(ts), encode, cache)
    assert CountingEncoder.calls == 2
