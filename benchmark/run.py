"""Benchmark: 50 questions x 5 configurations, cost accounting, report tables.

Configurations (per homework02 spec, «живой поиск» из первой домашки у нас
нет, поэтому строка 2 заменена на «дешёвая модель, без инструментов»):

    strong_notools  — сильная модель, без инструментов (обгоняем её)
    cheap_notools   — дешёвая модель, без инструментов (прошлая неделя)
    cheap_rag       — дешёвая модель, RAG всегда (k=5, structural)
    cheap_agent     — дешёвая модель, агент с knowledge_base
    mid_rag         — средняя модель, RAG всегда

Every config/question result is checkpointed to disk as soon as it exists,
and finished pairs are skipped on resume — an interrupted run never pays
twice. A hard dollar budget aborts the whole run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from agent.loop import run_agent
from benchmark.judge import judge_answer
from llm.openrouter import BudgetExceeded, OpenRouterClient
from pymilvus import MilvusClient as make_client
from rag.answer import NO_TOOLS_SYSTEM_PROMPT, rag_answer
from rag.tool import make_knowledge_base
from retrieval.evaluate import load_questions
from vectorstore.embed import Embedder

MODELS = {
    "cheap": "openai/gpt-4o-mini",
    "mid": "google/gemini-2.5-flash",
    "strong": "openai/gpt-4.1",
}

CONFIGS = {
    "strong_notools": {"model": MODELS["strong"], "mode": "notools"},
    "cheap_notools": {"model": MODELS["cheap"], "mode": "notools"},
    "cheap_rag": {"model": MODELS["cheap"], "mode": "rag"},
    "cheap_agent": {"model": MODELS["cheap"], "mode": "agent"},
    "mid_rag": {"model": MODELS["mid"], "mode": "rag"},
}

REPORTS = Path("reports")
CHECKPOINT = REPORTS / "benchmark_results.jsonl"
JUDGE_CACHE = REPORTS / "judge_cache.json"
URI = "http://localhost:19530"


def load_env(path: Path = Path(".env")) -> None:
    """Minimal .env loader: KEY=VALUE lines into os.environ (no echo)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def answer_once(llm, kb, question: str, config: dict) -> dict:
    mode = config["mode"]
    model = config["model"]
    if mode == "notools":
        messages = [
            {"role": "system", "content": NO_TOOLS_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        reply = llm.chat(model=model, messages=messages, max_tokens=400)
        return {
            "answer": (reply.content or "").strip(),
            "cost_usd": reply.usage.cost_usd,
            "tool_calls": [],
            "steps": 1,
        }
    if mode == "rag":
        client = config["_client"]
        embedder = config["_embedder"]
        record = rag_answer(llm, client, embedder, question, model=model)
        return {
            "answer": record["answer"],
            "cost_usd": record["cost_usd"],
            "tool_calls": [],
            "steps": 1,
            "chunk_ids": record["chunk_ids"],
            "prompt_tokens": record["prompt_tokens"],
            "completion_tokens": record["completion_tokens"],
        }
    return run_agent(llm, model, kb, question)


def run(
    limit: int | None = None,
    only_configs: list[str] | None = None,
    budget_usd: float = 1.50,
    uri: str = URI,
    judge_model: str | None = None,
) -> Path:
    load_env()
    questions = load_questions()
    if limit:
        # Keep the answerable/unanswerable mix: take a slice of each.
        ans = [q for q in questions if q["answerable"]][:limit]
        una = [q for q in questions if not q["answerable"]][: max(1, limit // 4)]
        questions = ans + una

    client = make_client(uri)
    embedder = Embedder()
    kb = make_knowledge_base(client, embedder)

    llm = OpenRouterClient(budget_usd=budget_usd)
    REPORTS.mkdir(exist_ok=True)
    done: set[tuple[str, str]] = set()
    if CHECKPOINT.exists():
        for line in CHECKPOINT.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done.add((rec["config"], rec["qid"]))

    active = {name: cfg for name, cfg in CONFIGS.items() if not only_configs or name in only_configs}
    for cname, cfg in active.items():
        cfg["_client"] = client
        cfg["_embedder"] = embedder
        started = time.time()
        for q in questions:
            if (cname, q["id"]) in done:
                continue
            if llm.spend_usd >= budget_usd:
                raise BudgetExceeded(f"budget reached at ${llm.spend_usd:.4f}")
            result = answer_once(llm, kb, q["question"], cfg)
            record = {
                "config": cname,
                "qid": q["id"],
                "answerable": q["answerable"],
                "model": cfg["model"],
                "question": q["question"],
                "gold": q["answer"],
                **result,
            }
            with CHECKPOINT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            done.add((cname, q["id"]))
            print(f"[{cname}] {q['id']}: ${result['cost_usd']:.5f} (run total ${llm.spend_usd:.4f})")
        print(f"--- {cname} done in {time.time() - started:.0f}s, spend so far ${llm.spend_usd:.4f}")

    # Judge (skips pairs already judged).
    judge_model = judge_model or MODELS["cheap"]
    cache: dict[str, str] = {}
    if JUDGE_CACHE.exists():
        cache = json.loads(JUDGE_CACHE.read_text())
    # Dedup: a judged copy of a record is appended after the raw one — keep last.
    latest: dict[tuple[str, str], dict] = {}
    for line in CHECKPOINT.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            latest[(rec["config"], rec["qid"])] = rec
    records = list(latest.values())
    for rec in records:
        key = f"{rec['config']}/{rec['qid']}"
        if "verdict" in rec:
            cache[key] = rec["verdict"]
            continue
        if key in cache:
            rec["verdict"] = cache[key]
            with CHECKPOINT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            continue
        if llm.spend_usd >= budget_usd:
            raise BudgetExceeded(f"budget reached during judging at ${llm.spend_usd:.4f}")
        verdict, cost = judge_answer(llm, judge_model, rec["question"], rec["gold"], rec["answer"])
        cache[key] = verdict
        rec["verdict"] = verdict
        JUDGE_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
        with CHECKPOINT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out = REPORTS / "benchmark_results.jsonl"
    print(f"\nresults: {out}\nspend: ${llm.spend_usd:.4f}")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="benchmark.run")
    parser.add_argument("--limit", type=int, default=None, help="smoke: first N answerable (+ unanswerable)")
    parser.add_argument("--configs", default=None, help="comma list, e.g. cheap_rag,strong_notools")
    parser.add_argument("--budget", type=float, default=1.50)
    parser.add_argument("--uri", default=URI)
    args = parser.parse_args(argv)
    run(
        limit=args.limit,
        only_configs=args.configs.split(",") if args.configs else None,
        budget_usd=args.budget,
        uri=args.uri,
    )


if __name__ == "__main__":
    main()
