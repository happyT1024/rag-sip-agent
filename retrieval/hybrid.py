"""Hybrid search: vector + BM25 fused with RRF, and the vector/words/hybrid table.

    .venv/bin/python -m retrieval.hybrid                      # structural
    .venv/bin/python -m retrieval.hybrid --method naive
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vectorstore.embed import Embedder
from vectorstore.ingest import load_jsonl, search

from .bm25 import BM25Index
from .evaluate import K_VALUES, load_questions
from .scoring import correct_chunk_ids, mean, rrf_fuse

RECORDS_DIR = Path("data/processed")


def evaluate_hybrid(
    client,
    embedder: Embedder,
    records: list[dict],
    questions: list[dict],
    collection: str,
    max_k: int,
    rrf_k: int = 60,
) -> dict:
    index = BM25Index([r["chunk_id"] for r in records], [r["text"] for r in records])
    rankings: dict[str, dict[str, list[str]]] = {
        q["id"]: {} for q in questions
    }
    for q in questions:
        vec_hits = search(client, embedder, collection, q["question"], limit=max_k)
        vec = [h["chunk_id"] for h in vec_hits]
        words = index.search(q["question"], top_k=max_k)
        rankings[q["id"]] = {
            "vector": vec,
            "words": words,
            "hybrid": rrf_fuse([vec, words], k=rrf_k),
        }

    corrects = {q["id"]: correct_chunk_ids(records, q["evidence"]) for q in questions}
    table: dict = {"collection": collection, "n_questions": len(questions), "rrf_k": rrf_k}
    for source in ("vector", "words", "hybrid"):
        for k in K_VALUES:
            flags = [
                bool(set(rankings[q["id"]][source][:k]) & corrects[q["id"]])
                for q in questions
            ]
            table[f"{source}_recall@{k}"] = round(mean([1.0 if f else 0.0 for f in flags]), 4)
    table["rankings"] = rankings
    table["correct"] = {qid: sorted(ids) for qid, ids in corrects.items()}
    return table


def print_table(t: dict) -> None:
    header = f"{'source':<10} " + " ".join(f"{'@' + str(k):>8}" for k in K_VALUES)
    print(header)
    for source in ("vector", "words", "hybrid"):
        cells = " ".join(f"{t[f'{source}_recall@{k}']:>8.3f}" for k in K_VALUES)
        print(f"{source:<10} {cells}")


def run(
    method: str = "structural",
    uri: str = "http://localhost:19530",
    questions_path: Path = Path("data/questions.jsonl"),
    out_path: Path = Path("data/processed/hybrid_report.json"),
    max_k: int = 10,
) -> dict:
    from pymilvus import MilvusClient

    collection = f"rag_{method}"
    records = load_jsonl(RECORDS_DIR / f"{method}.jsonl")
    questions = [q for q in load_questions(questions_path) if q["answerable"]]

    client = MilvusClient(uri)
    embedder = Embedder()
    print(f"hybrid eval: {collection}, {len(questions)} questions (vector + BM25 + RRF)")
    table = evaluate_hybrid(client, embedder, records, questions, collection, max_k=max_k)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print_table(table)
    print(f"report -> {out_path}")
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="retrieval.hybrid")
    parser.add_argument("--method", default="structural", choices=("naive", "structural"))
    parser.add_argument("--uri", default="http://localhost:19530")
    parser.add_argument("--questions", default="data/questions.jsonl")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    out = args.out or f"data/processed/hybrid_report_{args.method}.json"
    run(args.method, args.uri, Path(args.questions), Path(out))


if __name__ == "__main__":
    main()
