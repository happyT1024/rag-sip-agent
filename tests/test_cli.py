"""CLI-level test: ``chunker.cli.run()`` on a tiny generated PDF.

Offline and keyless — the PDF is built in-process with pymupdf (same
technique as test_extract.py::test_parse_title_reads_raw_headers).
"""

import json
import re

import pytest

from chunker.cli import run

pytest.importorskip("pymupdf")

CHUNK_ID_RE = re.compile(r"^(naive|structural)-(\d+)-(\d{4})$")
RECORD_KEYS = {
    "chunk_id", "method", "rfc", "title", "text", "n_chars",
    "section_id", "section_path", "page_start", "page_end",
}


def _write_mini_pdf(path) -> None:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    lines = [
        "RFC 4001                The Tiny Protocol              June 2002",
        "1 Introduction",
        "   The Tiny Protocol carries fictional signaling for the chunker",
        "   command-line test.  It has one section with a couple of short",
        "   paragraphs so that both chunkers produce at least one chunk.",
        "Full Copyright Statement",
        "   Copyright (C) The Internet Society 2002.",
        "Bernstein            Standards Track            [Page 1]",
    ]
    y = 72.0
    for line in lines:
        page.insert_text((72.0, y), line)
        y += 14.0
    doc.save(str(path))
    doc.close()


def test_run_writes_chunk_jsonl(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_mini_pdf(data_dir / "RFC 4001.pdf")
    out_dir = tmp_path / "processed"

    stats = run(data_dir, out_dir)

    assert stats["naive"] >= 1
    assert stats["structural"] >= 1
    for name in ("naive.jsonl", "structural.jsonl"):
        records = [
            json.loads(line)
            for line in (out_dir / name).read_text(encoding="utf-8").splitlines()
        ]
        assert records, f"{name} is empty"
        seen = set()
        for record in records:
            assert set(record) == RECORD_KEYS
            assert record["chunk_id"], "chunk_id must be populated"
            assert CHUNK_ID_RE.match(record["chunk_id"]), record["chunk_id"]
            assert record["chunk_id"] not in seen, "chunk_id must be unique"
            seen.add(record["chunk_id"])
