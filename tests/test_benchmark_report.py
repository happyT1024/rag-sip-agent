"""Offline tests for the report renderer: synthetic judged records, no LLM."""

from __future__ import annotations

from benchmark.report import aggregate, load_records, render_refusal_table, render_report_table

RECORDS = [
    # cheap_rag: 2 answerable (1 correct, 1 refused) + 1 unanswerable (refused)
    {"config": "cheap_rag", "qid": "q01", "answerable": True, "verdict": "correct", "cost_usd": 0.001},
    {"config": "cheap_rag", "qid": "q02", "answerable": True, "verdict": "refused", "cost_usd": 0.001},
    {"config": "cheap_rag", "qid": "u01", "answerable": False, "verdict": "refused", "cost_usd": 0.001},
    # cheap_notools: 1 answerable correct + 1 unanswerable hallucinated
    {"config": "cheap_notools", "qid": "q01", "answerable": True, "verdict": "correct", "cost_usd": 0.0002},
    {"config": "cheap_notools", "qid": "u01", "answerable": False, "verdict": "wrong", "cost_usd": 0.0002},
]


def test_load_records_dedup_keeps_last(tmp_path):
    p = tmp_path / "results.jsonl"
    first = {"config": "c", "qid": "q1", "answerable": True, "cost_usd": 0.1}
    second = dict(first, verdict="wrong", cost_usd=0.2)
    p.write_text(
        "\n".join([__import__("json").dumps(first), __import__("json").dumps(second)]) + "\n"
    )
    recs = load_records(p)
    assert len(recs) == 1 and recs[0]["cost_usd"] == 0.2  # judged copy wins


def test_aggregate_metrics():
    stats = aggregate(RECORDS)
    rag = stats["cheap_rag"]
    assert rag["accuracy"] == 0.5  # 1 of 2 answerable
    assert rag["honest_refusals"] == 1.0
    assert rag["false_refusals"] == 0.5
    assert rag["cost_per_question"] == 0.001
    assert rag["cost_per_correct"] == 0.003 / 1
    nt = stats["cheap_notools"]
    assert nt["accuracy"] == 1.0
    assert nt["honest_refusals"] == 0.0


def test_render_tables_contain_rows():
    stats = aggregate(RECORDS)
    table = render_report_table(stats)
    assert "дешёвая модель, RAG всегда" in table
    assert "50.00%" in table
    refusals = render_refusal_table(stats)
    assert "Ложный отказ" in refusals
    # strong_notools absent from records -> absent from tables
    assert "сильная модель" not in table
