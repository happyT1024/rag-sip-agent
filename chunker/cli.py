"""CLI: ``python -m chunker run`` → both JSONL chunk sets + corpus stats."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .clean import clean_pages
from .extract import extract_pages, parse_title, rfc_number_from_path
from .naive import chunk_naive
from .structural import chunk_structural

HISTOGRAM_BIN = 250


def _assign_ids(chunks, rfc: int) -> None:
    for i, chunk in enumerate(chunks):
        chunk.chunk_id = f"{chunk.method}-{rfc}-{i:04d}"


def process_pdf(pdf_path: Path) -> tuple[list, list, int]:
    """Extract → clean → chunk one PDF. Returns (naive, structural, n_pages)."""
    rfc = rfc_number_from_path(pdf_path)
    pages = extract_pages(pdf_path)
    title = parse_title(pdf_path)
    cleaned = clean_pages(pages)
    naive_chunks = chunk_naive(cleaned, rfc, title)
    structural_chunks = chunk_structural(cleaned, rfc, title)
    _assign_ids(naive_chunks, rfc)
    _assign_ids(structural_chunks, rfc)
    return naive_chunks, structural_chunks, len(pages)


def _histogram(chunks: list) -> list[str]:
    counter: Counter[int] = Counter()
    for chunk in chunks:
        counter[min(chunk.n_chars // HISTOGRAM_BIN, 12)] += 1
    lines = []
    for bucket in sorted(counter):
        label = f"{bucket * HISTOGRAM_BIN}-{(bucket + 1) * HISTOGRAM_BIN - 1}"
        if bucket == 12:
            label = f">= {bucket * HISTOGRAM_BIN}"
        lines.append(f"    {label:>12}: {counter[bucket]}")
    return lines


def run(data_dir: str | Path = "data", out_dir: str | Path = "data/processed") -> dict:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        print(f"no PDFs found in {data_dir}", file=sys.stderr)
        raise SystemExit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    all_naive: list = []
    all_structural: list = []

    for pdf_path in pdfs:
        naive_chunks, structural_chunks, n_pages = process_pdf(pdf_path)
        all_naive.extend(naive_chunks)
        all_structural.extend(structural_chunks)
        title = naive_chunks[0].title if naive_chunks else (
            structural_chunks[0].title if structural_chunks else "?"
        )
        print(f"{pdf_path.name}: {n_pages} pages, title={title!r}")
        print(f"  naive:      {len(naive_chunks):5d} chunks")
        print(f"  structural: {len(structural_chunks):5d} chunks")

    naive_path = out_dir / "naive.jsonl"
    structural_path = out_dir / "structural.jsonl"
    with naive_path.open("w", encoding="utf-8") as f:
        for chunk in all_naive:
            f.write(json.dumps(chunk.to_record(), ensure_ascii=False) + "\n")
    with structural_path.open("w", encoding="utf-8") as f:
        for chunk in all_structural:
            f.write(json.dumps(chunk.to_record(), ensure_ascii=False) + "\n")

    print(f"\nnaive.jsonl:      {len(all_naive)} chunks -> {naive_path}")
    print(f"structural.jsonl: {len(all_structural)} chunks -> {structural_path}")
    print("\nnaive length histogram (chars):")
    print("\n".join(_histogram(all_naive)))
    print("\nstructural length histogram (chars):")
    print("\n".join(_histogram(all_structural)))
    return {
        "naive": len(all_naive),
        "structural": len(all_structural),
        "naive_path": str(naive_path),
        "structural_path": str(structural_path),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="chunker",
        description="Chunk the SIP RFC PDF corpus two ways (naive window + structural).",
    )
    sub = parser.add_subparsers(dest="command")
    run_parser = sub.add_parser("run", help="produce data/processed/{naive,structural}.jsonl")
    run_parser.add_argument("--data-dir", default="data")
    run_parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.data_dir, args.out_dir)
    else:
        # Default (no subcommand): run with defaults.
        run()


if __name__ == "__main__":
    main()
