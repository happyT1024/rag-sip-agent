"""Tests for RAG answering, the knowledge_base tool, the agent loop and the judge.

No network, no API key, no real LLM, no docker: Milvus Lite + FakeEmbedder +
scripted fake LLM. This is the spec's «тест на три случая» for knowledge_base
plus wiring tests for everything around it.
"""

from __future__ import annotations

import pytest
from pymilvus import MilvusClient

from agent.loop import run_agent
from benchmark.judge import build_judge_messages, parse_verdict
from llm.openrouter import BudgetExceeded, OpenRouterClient, Reply, Usage
from rag.answer import REFUSAL, build_messages, format_context, rag_answer
from rag.tool import TOOL_SCHEMA, make_knowledge_base, parse_tool_args
from tests.test_vectorstore import RECORDS, FakeEmbedder

COLL = "rag_structural"


@pytest.fixture
def kb(tmp_path):
    client = MilvusClient(uri=str(tmp_path / "lite.db"))
    from vectorstore.ingest import ensure_collection, ingest_records

    ensure_collection(client, COLL, FakeEmbedder.dim)
    ingest_records(client, FakeEmbedder(), RECORDS, COLL)
    return make_knowledge_base(client, FakeEmbedder(), collection=COLL, k=5), client, FakeEmbedder()


# --- knowledge_base: the three required cases -------------------------------


def test_kb_case1_relevant_chunks(kb):
    tool, *_ = kb
    out = tool("REGISTER binding refresh")
    assert "REGISTER binding refresh" in out
    assert "[1]" in out and "RFC 3261" in out  # citation with source


def test_kb_case2_rfc_filter(kb):
    tool, *_ = kb
    # Milvus always returns top-k (no similarity cutoff), so a filter to a
    # foreign RFC still returns that RFC's chunks — but never foreign ones.
    out = tool("REGISTER binding refresh", rfc=3515)
    assert "RFC 3515" in out and "RFC 3261" not in out
    out = tool("REGISTER binding refresh", rfc=3261)
    assert "RFC 3261" in out and "RFC 3515" not in out


def test_kb_case3_empty_collection_returns_marker(tmp_path):
    from vectorstore.ingest import ensure_collection

    client = MilvusClient(uri=str(tmp_path / "empty.db"))
    ensure_collection(client, COLL, FakeEmbedder.dim)
    tool = make_knowledge_base(client, FakeEmbedder(), collection=COLL)
    assert tool("anything at all") == "Ничего не найдено по этому запросу."


def test_kb_schema_shape():
    fn = TOOL_SCHEMA["function"]
    assert fn["name"] == "knowledge_base"
    assert fn["parameters"]["required"] == ["query"]
    assert set(fn["parameters"]["properties"]) == {"query", "rfc"}


def test_parse_tool_args_tolerates_garbage():
    assert parse_tool_args({"function": {"arguments": '{"query": "x"}'}}) == {"query": "x"}
    assert parse_tool_args({"function": {"arguments": "not json"}}) == {}
    assert parse_tool_args({}) == {}


# --- format_context / rag_answer --------------------------------------------


def test_format_context_numbering_and_source():
    hits = [
        {"chunk_id": "c1", "rfc": 3261, "section_path": "9.2", "page": 55, "text": "abc"},
        {"chunk_id": "c2", "rfc": 3515, "section_path": None, "page": 7, "text": "def"},
    ]
    ctx = format_context(hits)
    assert "[1] RFC 3261 · §9.2" in ctx
    assert "[2] RFC 3515 · стр. 7" in ctx
    assert "abc" in ctx and "def" in ctx


class ScriptedLLM:
    """Returns canned replies; records every call for assertions."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, model, messages, tools=None, max_tokens=500, temperature=0.0):
        self.calls.append({"model": model, "messages": messages, "tools": tools})
        reply = self.replies.pop(0)
        reply.usage.cost_usd = reply.usage.cost_usd or 0.0
        return reply


def test_rag_answer_passes_context_and_records_provenance(kb):
    _, client, embedder = kb
    llm = ScriptedLLM([Reply(content="Не знаю.")])
    rec = rag_answer(llm, client, embedder, "REGISTER binding refresh", model="fake-model")
    assert rec["chunk_ids"][0] == "structural-3261-0008"  # exact text = top-1
    assert rec["cost_usd"] == 0.0
    user_msg = llm.calls[0]["messages"][1]["content"]
    assert "Вопрос: REGISTER binding refresh" in user_msg
    assert "[1]" in user_msg
    assert REFUSAL in llm.calls[0]["messages"][0]["content"]


def test_build_messages_empty_hits():
    msgs = build_messages("q?", [])
    assert "(пусто)" in msgs[1]["content"]


# --- agent loop --------------------------------------------------------------


def _tc(name="knowledge_base", args='{"query": "REGISTER refresh"}', tc_id="call_1"):
    return {"id": tc_id, "type": "function", "function": {"name": name, "arguments": args}}


def test_agent_calls_tool_then_answers(kb):
    tool, *_ = kb
    llm = ScriptedLLM(
        [
            Reply(content="", tool_calls=[_tc()]),
            Reply(content="Ответ со ссылкой [1]."),
        ]
    )
    out = run_agent(llm, "fake-model", tool, "Как обновить регистрацию?")
    assert out["answer"] == "Ответ со ссылкой [1]."
    assert out["tool_calls"] == [{"query": "REGISTER refresh"}]
    assert out["steps"] == 2
    # The tool result was fed back as a tool message.
    roles = [m["role"] for m in llm.calls[1]["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]


def test_agent_forces_final_answer_at_step_cap(kb):
    tool, *_ = kb
    llm = ScriptedLLM(
        [Reply(content="", tool_calls=[_tc(tc_id="c0")]), Reply(content="", tool_calls=[_tc(tc_id="c1")])]
        + [Reply(content="финал")]
    )
    out = run_agent(llm, "fake-model", tool, "вопрос", max_steps=2)
    assert out["answer"] == "финал"
    assert len(out["tool_calls"]) == 2
    last = llm.calls[-1]["messages"][-1]
    assert "финальный ответ" in last["content"]
    assert llm.calls[-1]["tools"] is None  # the forced final call goes without tools


def test_agent_malformed_tool_args_degrade_to_empty_query():
    queries = []

    def tool(query, rfc=None):
        queries.append(query)
        return "найдено"

    llm = ScriptedLLM(
        [Reply(content="", tool_calls=[_tc(args="not json")]), Reply(content="ok")]
    )
    out = run_agent(llm, "fake-model", tool, "q")
    assert out["answer"] == "ok"
    assert queries == [""], "unparsable arguments must fall back to the empty query"
    messages = llm.calls[1]["messages"]
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["content"] == "найдено"


def test_agent_tool_error_does_not_crash(kb):
    tool, *_ = kb

    def broken(**kwargs):
        raise RuntimeError("boom")

    llm = ScriptedLLM([Reply(content="", tool_calls=[_tc()]), Reply(content="ok")])
    out = run_agent(llm, "fake-model", broken, "q")
    assert out["answer"] == "ok"


# --- judge --------------------------------------------------------------------


def test_parse_verdict_from_fenced_json():
    assert parse_verdict(' Вот ответ:\n```json\n{"verdict": "correct", "reason": "ок"}\n```') == "correct"


def test_parse_verdict_fallback():
    assert parse_verdict("system refused to answer") == "refused"
    assert parse_verdict("не понял ничего") == "wrong"


def test_judge_messages_contain_gold_and_answer():
    msgs = build_judge_messages("q", "gold answer", "system answer")
    assert "gold answer" in msgs[1]["content"]
    assert "system answer" in msgs[1]["content"]
    msgs = build_judge_messages("q", None, "system answer")
    assert "эталонного ответа нет" in msgs[1]["content"]


# --- llm client: budget & cost math (no network) -------------------------------


def test_budget_is_checked_before_any_network():
    client = OpenRouterClient(api_key="test-key", budget_usd=0.0)
    with pytest.raises(BudgetExceeded):
        client.chat("any/model", [{"role": "user", "content": "hi"}])
    assert client.spend_usd == 0.0


def test_estimate_cost_from_pricing_table():
    from llm.openrouter import estimate_cost

    # OpenRouter prices are dollars per token.
    pricing = {"m/x": {"prompt": 1.5e-07, "completion": 6e-07}}
    cost = estimate_cost("m/x", Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000), pricing)
    assert cost == pytest.approx(0.75)
    # Unknown model prices at zero rather than crashing.
    assert estimate_cost("unknown", Usage(prompt_tokens=100), pricing) == 0.0
