"""Render the spec deliverables from the benchmark checkpoint:

- report table: 5 configurations x (доля верных, цена вопроса, цена верного ответа)
- refusal table: honest refusals on unanswerable / false refusals on answerable
- money chart: accuracy vs cost per question, one point per configuration

Works offline on reports/benchmark_results.jsonl — no API access needed.
    .venv/bin/python -m benchmark.report
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from benchmark.run import CHECKPOINT, CONFIGS, REPORTS

CONFIG_TITLES = {
    "strong_notools": "сильная модель, без инструментов",
    "cheap_notools": "дешёвая модель, без инструментов",
    "cheap_rag": "дешёвая модель, RAG всегда",
    "cheap_agent": "дешёвая модель, агент с knowledge_base",
    "mid_rag": "средняя модель, RAG всегда",
}
CONFIG_ORDER = ["strong_notools", "cheap_notools", "cheap_rag", "cheap_agent", "mid_rag"]


def load_records(path: Path = CHECKPOINT) -> list[dict]:
    """Latest record per (config, qid); only fully judged ones."""
    latest: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return []
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            latest[(rec["config"], rec["qid"])] = rec
    return [r for r in latest.values() if "verdict" in r]


def aggregate(records: list[dict]) -> dict[str, dict]:
    by_config: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_config[rec["config"]].append(rec)

    stats: dict[str, dict] = {}
    for config, recs in by_config.items():
        answerable = [r for r in recs if r["answerable"]]
        unanswerable = [r for r in recs if not r["answerable"]]
        n_correct = sum(r["verdict"] == "correct" for r in answerable)
        n_refused_ans = sum(r["verdict"] == "refused" for r in answerable)
        n_refused_unans = sum(r["verdict"] == "refused" for r in unanswerable)
        total_cost = sum(r.get("cost_usd", 0.0) for r in recs)
        n = len(recs) or 1
        stats[config] = {
            "n": len(recs),
            "accuracy": n_correct / len(answerable) if answerable else 0.0,
            "honest_refusals": n_refused_unans / len(unanswerable) if unanswerable else 0.0,
            "false_refusals": n_refused_ans / len(answerable) if answerable else 0.0,
            "cost_per_question": total_cost / n,
            "cost_per_correct": (total_cost / n_correct) if n_correct else None,
        }
    return stats


def render_report_table(stats: dict[str, dict]) -> str:
    header = (
        "| Конфигурация | Доля верных | Цена вопроса | Цена верного ответа |\n"
        "|---|---|---|---|\n"
    )
    rows = []
    for config in CONFIG_ORDER:
        if config not in stats:
            continue
        s = stats[config]
        cpc = f"${s['cost_per_correct']:.4f}" if s["cost_per_correct"] is not None else "—"
        rows.append(
            f"| {CONFIG_TITLES[config]} ({config}) "
            f"| {s['accuracy']:.2%} | ${s['cost_per_question']:.4f} | {cpc} |"
        )
    return header + "\n".join(rows) + "\n"


def render_refusal_table(stats: dict[str, dict]) -> str:
    header = (
        "| Конфигурация | Отказ на неотвечаемых (из 10) | Ложный отказ на отвечаемых (из 40) |\n"
        "|---|---|---|\n"
    )
    rows = []
    for config in CONFIG_ORDER:
        if config not in stats:
            continue
        s = stats[config]
        # Percentages are over the judged subsets actually present in the run.
        n_unans = round(s["honest_refusals"] * 10)
        n_ans = round(s["false_refusals"] * 40)
        rows.append(
            f"| {CONFIG_TITLES[config]} ({config}) | {n_unans} | {n_ans} |"
        )
    return header + "\n".join(rows) + "\n"


def money_chart(stats: dict[str, dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for config in CONFIG_ORDER:
        if config not in stats:
            continue
        s = stats[config]
        ax.scatter(s["cost_per_question"], s["accuracy"], s=90)
        ax.annotate(
            config,
            (s["cost_per_question"], s["accuracy"]),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=9,
        )
    ax.set_xlabel("цена вопроса, $")
    ax.set_ylabel("доля верных ответов")
    ax.set_title("money_chart: качество за деньги")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"chart -> {out_path}")


def run(
    checkpoint: Path = CHECKPOINT,
    out_dir: Path = REPORTS,
) -> dict:
    records = load_records(checkpoint)
    if not records:
        print("no judged records yet — run benchmark.run first")
        return {}
    stats = aggregate(records)
    out_dir.mkdir(exist_ok=True)
    (out_dir / "report_table.md").write_text(render_report_table(stats), encoding="utf-8")
    (out_dir / "refusal_table.md").write_text(render_refusal_table(stats), encoding="utf-8")
    money_chart(stats, out_dir / "money_chart.png")
    print(render_report_table(stats))
    print(render_refusal_table(stats))
    return stats


def main(argv: list[str] | None = None) -> None:
    run()


if __name__ == "__main__":
    main()
