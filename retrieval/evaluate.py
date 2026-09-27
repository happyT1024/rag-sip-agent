"""Measure recall@k and MRR for both chunkings over the question set.

Run against the standalone Milvus from docker-compose (default) or a Milvus
Lite file (``--uri ./something.db``). The per-question verdict uses the
evidence quote: a chunk counts as correct iff its text contains the quote.

    .venv/bin/python -m retrieval.evaluate
    .venv/bin/python -m retrieval.evaluate --plot reports/recall_curve.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vectorstore.embed import Embedder
from vectorstore.ingest import COLLECTIONS, load_jsonl, search

from .scoring import correct_chunk_ids, mean, recall_at_k, reciprocal_rank

QUESTIONS_PATH = Path("data/questions.jsonl")
K_VALUES = (1, 3, 5, 10)


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate_method(
    client,
    embedder: Embedder,
    records: list[dict],
    questions: list[dict],
    collection: str,
    max_k: int,
) -> dict:
    """Search every question in one collection, score recall@k / MRR."""
    by_id = {r["chunk_id"]: r for r in records}
    per_question: list[dict] = []
    for q in questions:
        hits = search(client, embedder, collection, q["question"], limit=max_k)
        ranked = [h["chunk_id"] for h in hits]
        correct = correct_chunk_ids(records, q["evidence"])
        entry = {
            "id": q["id"],
            "ranked": ranked,
            "correct": sorted(correct),
            "correct_hits": sorted(set(ranked) & correct),
        }
        if not correct:
            print(f"  ! no chunk contains evidence of {q['id']}")
        per_question.append(entry)

    report: dict = {"collection": collection, "n_questions": len(per_question)}
    for k in K_VALUES:
        if k > max_k:
            continue
        hits_flags = [recall_at_k(e["ranked"], set(e["correct"]), k) for e in per_question]
        report[f"recall@{k}"] = round(mean([1.0 if h else 0.0 for h in hits_flags]), 4)
    report["mrr"] = round(
        mean([reciprocal_rank(e["ranked"], set(e["correct"])) for e in per_question]), 4
    )
    report["per_question"] = per_question
    report["records_by_id"] = {cid: {"page": by_id[cid]["page_start"],
                                     "section_path": by_id[cid]["section_path"]}
                               for cid in by_id}
    return report


def plot_curve(reports: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for rep in reports:
        ks = [k for k in K_VALUES if f"recall@{k}" in rep]
        vals = [rep[f"recall@{k}"] for k in ks]
        ax.plot(ks, vals, marker="o", label=f"{rep['collection']} ({rep['n_questions']} вопр.)")
    ax.set_xlabel("k")
    ax.set_ylabel("recall@k")
    ax.set_xticks(list(K_VALUES))
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"plot -> {out_path}")


def run(
    uri: str = "http://localhost:19530",
    questions_path: Path = QUESTIONS_PATH,
    plot_path: Path | None = None,
    out_path: Path = Path("data/processed/recall_report.json"),
) -> list[dict]:
    from pymilvus import MilvusClient

    questions = [q for q in load_questions(questions_path) if q["answerable"]]
    client = MilvusClient(uri)
    embedder = Embedder()

    reports = []
    for method, collection in COLLECTIONS.items():
        records = load_jsonl(Path("data/processed") / f"{method}.jsonl")
        print(f"evaluating {collection} on {len(questions)} questions...")
        reports.append(
            evaluate_method(client, embedder, records, questions, collection, max_k=max(K_VALUES))
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    slim = [{k: v for k, v in r.items() if k != "records_by_id"} for r in reports]
    out_path.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'collection':<16} " + " ".join(f"{'@' + str(k):>8}" for k in K_VALUES) + f" {'MRR':>8}")
    for r in reports:
        cells = " ".join(f"{r[f'recall@{k}']:>8.3f}" for k in K_VALUES)
        print(f"{r['collection']:<16} {cells} {r['mrr']:>8.3f}")
    print(f"report -> {out_path}")

    if plot_path:
        plot_curve(reports, plot_path)
    return slim


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="retrieval.evaluate")
    parser.add_argument("--uri", default="http://localhost:19530")
    parser.add_argument("--questions", default=str(QUESTIONS_PATH))
    parser.add_argument("--plot", default=None, help="write recall@k curve PNG to this path")
    parser.add_argument("--out", default="data/processed/recall_report.json")
    args = parser.parse_args(argv)
    run(args.uri, Path(args.questions), Path(args.plot) if args.plot else None, Path(args.out))


if __name__ == "__main__":
    main()
