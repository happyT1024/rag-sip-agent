"""Real-corpus smoke test (CAP-3/CAP-4 invariants).

Still keyless: it reads the local PDFs and runs pure-text code — no network,
no LLM, no API key. Skips only when the corpus is absent (e.g. CI checkout
without data/).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from chunker.cli import process_pdf

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

corpus_missing = not sorted(DATA_DIR.glob("*.pdf"))

pytestmark = pytest.mark.skipif(
    corpus_missing, reason="corpus data/*.pdf not present"
)

NAIVE_BAND = (800, 1200)
STRUCTURAL_BAND = (400, 600)
STRUCTURAL_FLOOR = 300
# Bands from the assignment spec (problem/homework02.pdf): naive ~1000 (accept 800-1200), structural 400-600, floor 300.
PREFIX_RE = re.compile(r"^RFC \d+ — .+ · §")
REFS_LINE_RE = re.compile(r"^\s*\d+\.?\s*(?:Normative\s+|Informative\s+)?References\b")


@pytest.fixture(scope="module")
def results():
    naive, structural = [], []
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        n_chunks, s_chunks, _ = process_pdf(pdf_path)
        naive.extend(n_chunks)
        structural.extend(s_chunks)
    return naive, structural


def test_all_rfcs_present(results):
    naive, structural = results
    assert {c.rfc for c in naive} == {2976, 3261, 3262, 3311, 3428, 3515}
    assert {c.rfc for c in structural} == {2976, 3261, 3262, 3311, 3428, 3515}


def test_naive_count_in_band(results):
    naive, _ = results
    assert NAIVE_BAND[0] <= len(naive) <= NAIVE_BAND[1]


def test_structural_count_in_band(results):
    _, structural = results
    assert len(structural) >= STRUCTURAL_FLOOR
    assert STRUCTURAL_BAND[0] <= len(structural) <= STRUCTURAL_BAND[1]


def test_no_residue_anywhere(results):
    naive, structural = results
    for chunk in naive + structural:
        text = chunk.text
        assert "Standards Track" not in text, chunk.chunk_id
        assert "Author's Address" not in text, chunk.chunk_id
        assert "Full Copyright Statement" not in text, chunk.chunk_id
        assert not re.search(r"^Table of Contents\s*$", text, re.M), chunk.chunk_id
        assert not re.search(r"\.{4,}", text), chunk.chunk_id
        # No running-header-shaped line; the structural prefix "RFC 3261 — "
        # uses one space after the number and must not match.
        for line in text.splitlines():
            assert not re.match(r"^RFC \d+\s{2,}\S", line), (chunk.chunk_id, line[:60])


def test_structural_prefix_format(results):
    _, structural = results
    for chunk in structural:
        assert PREFIX_RE.match(chunk.text), chunk.text[:70]
        assert chunk.section_path.startswith(f"{chunk.section_id} "), chunk.section_path
        assert chunk.text.split("\n", 1)[0].endswith(chunk.section_path), chunk.text[:90]


def test_naive_window_bound(results):
    naive, _ = results
    for chunk in naive:
        assert chunk.n_chars <= 1010, (chunk.chunk_id, chunk.n_chars)


def test_3261_section_9_1_present(results):
    _, structural = results
    matches = [
        c for c in structural
        if c.rfc == 3261 and c.section_path == "9.1 Client Behavior"
    ]
    assert matches, "RFC 3261 §9.1 Client Behavior must be retrievable"
    chunk = matches[0]
    assert chunk.text.startswith("RFC 3261 — ")
    assert "SIP: Session Initiation Protocol" in chunk.text.split("\n")[0]
    assert isinstance(chunk.page_start["pdf"], int)
    assert chunk.page_end["pdf"] >= chunk.page_start["pdf"]
    assert chunk.page_start["printed"] is not None
    assert chunk.page_end["printed"] is not None


def test_references_sections_survive_everywhere(results):
    _, structural = results
    for rfc in (2976, 3261, 3262, 3311, 3428, 3515):
        by_path = [
            c for c in structural
            if c.rfc == rfc and "References" in (c.section_path or "")
        ]
        # MIN-merged References (3262 §13, 3515 §9.2, 2976) keep their
        # heading line inside the neighbor chunk's text.
        by_text = [
            c for c in structural
            if c.rfc == rfc
            and any(REFS_LINE_RE.match(ln) for ln in c.text.splitlines())
        ]
        assert by_path or by_text, f"RFC {rfc} lost its References section"


def test_metadata_schema_complete(results):
    naive, structural = results
    for chunk in naive + structural:
        record = chunk.to_record()
        assert set(record) == {
            "chunk_id", "method", "rfc", "title", "text", "n_chars",
            "section_id", "section_path", "page_start", "page_end",
        }
        assert record["title"], f"{chunk.chunk_id} has no title"
        assert set(record["page_start"]) == {"pdf", "printed"}
