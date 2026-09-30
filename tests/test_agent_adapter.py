"""Offline tests for the OpenRouterAgent adapter (bot.port implemented).

Milvus Lite + FakeEmbedder + ScriptedLLM: no network, no API key, no LLM,
no docker. Covers the I/O matrix of the plan: citation mapping (last kb call
wins, dedupe, out-of-range fallback), refusal, history pass-through, cost
passthrough and kb-hits recording; ``ask`` runs through asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent.adapter import OpenRouterAgent
from agent.loop import AGENT_SYSTEM_PROMPT
from bot.port import HistoryItem, Source
from conftest import feed, make_bot, make_config, run, sent_texts
from llm.openrouter import Reply, Usage
from rag.answer import COLLECTION, REFUSAL
from tests.test_rag import ScriptedLLM
from tests.test_vectorstore import RECORDS, FakeEmbedder
from vectorstore.ingest import ensure_collection, ingest_records, search

COLL = COLLECTION
TOP_HIT = Source(rfc=3261, section="10.2 Registering", page=74)


def _tc(args='{"query": "REGISTER binding refresh"}', tc_id="call_1"):
    return {"id": tc_id, "type": "function", "function": {"name": "knowledge_base", "arguments": args}}


def _make_agent(milvus_uri: str) -> OpenRouterAgent:
    agent = OpenRouterAgent(
        model="fake-model",
        milvus_uri=milvus_uri,
        budget_usd=1.0,
        api_key="test-key",
    )
    ensure_collection(agent._client, COLL, FakeEmbedder.dim)
    ingest_records(agent._client, FakeEmbedder(), RECORDS, COLL)
    agent._embedder = FakeEmbedder()
    return agent


@pytest.fixture
def agent(tmp_path):
    return _make_agent(str(tmp_path / "lite.db"))


def _script(agent: OpenRouterAgent, replies: list[Reply]) -> ScriptedLLM:
    llm = ScriptedLLM(replies)
    agent._llm = llm
    return llm


def test_probe_true_after_ingest(agent) -> None:
    assert agent.probe() is True


def test_probe_false_without_collection(tmp_path) -> None:
    agent = OpenRouterAgent(
        model="fake-model",
        milvus_uri=str(tmp_path / "fresh.db"),
        budget_usd=1.0,
        api_key="test-key",
    )
    agent._embedder = FakeEmbedder()
    assert agent.probe() is False


def test_probe_false_for_existing_but_empty_collection(tmp_path) -> None:
    agent = OpenRouterAgent(
        model="fake-model",
        milvus_uri=str(tmp_path / "empty.db"),
        budget_usd=1.0,
        api_key="test-key",
    )
    ensure_collection(agent._client, COLL, FakeEmbedder.dim)
    agent._embedder = FakeEmbedder()
    assert agent.probe() is False


def test_ask_cited_answer_maps_top_hit_and_sums_cost(agent) -> None:
    _script(
        agent,
        [
            Reply(content="", tool_calls=[_tc()], usage=Usage(cost_usd=0.01)),
            Reply(content="Ответ со ссылкой [1].", usage=Usage(cost_usd=0.02)),
        ],
    )
    out = asyncio.run(agent.ask("Как обновить регистрацию?", []))
    assert out.text == "Ответ со ссылкой [1]."
    assert out.sources == [TOP_HIT]
    assert out.cost_usd == pytest.approx(0.03)


def test_ask_with_two_recorded_hits_cites_hit1(agent) -> None:
    _script(
        agent,
        [
            Reply(
                content="",
                tool_calls=[_tc(args='{"query": "REGISTER binding refresh", "rfc": 3261}')],
                usage=Usage(cost_usd=0.005),
            ),
            Reply(content="Ответ «так» [1].", usage=Usage(cost_usd=0.015)),
        ],
    )
    out = asyncio.run(agent.ask("Как обновить регистрацию?", []))
    hits = search(agent._client, FakeEmbedder(), COLL, "REGISTER binding refresh", limit=5, filter_expr="rfc == 3261")
    assert len(hits) == 2, "rfc filter leaves exactly 2 hits"
    hit1 = hits[0]
    assert out.sources == [
        Source(rfc=hit1["rfc"], section=hit1["section_path"], page=hit1["page"])
    ]
    assert out.sources[0] == TOP_HIT
    assert out.cost_usd == pytest.approx(0.02)


def test_citations_map_to_last_kb_call(agent) -> None:
    llm = _script(
        agent,
        [
            Reply(
                content="",
                tool_calls=[_tc(tc_id="c1", args='{"query": "INVITE transaction rules"}')],
            ),
            Reply(
                content="",
                tool_calls=[_tc(tc_id="c2", args='{"query": "REGISTER binding refresh"}')],
            ),
            Reply(content="Итог [1]."),
        ],
    )
    out = asyncio.run(agent.ask("вопрос", []))
    assert len(llm.calls) == 3 and all(
        call["tools"] for call in llm.calls[:2]
    ), "two kb calls happened (the third round answers without tools)"
    # [1] of the LAST call is the REGISTER chunk, not the INVITE one.
    assert out.sources == [TOP_HIT]


def test_citations_fall_back_to_last_non_empty_call(agent) -> None:
    _script(
        agent,
        [
            Reply(
                content="",
                tool_calls=[_tc(tc_id="c1", args='{"query": "REGISTER binding refresh"}')],
            ),
            Reply(
                content="",
                tool_calls=[
                    _tc(tc_id="c2", args='{"query": "REGISTER binding refresh", "rfc": 9999}')
                ],
            ),
            Reply(content="Итог [1]."),
        ],
    )
    empty = search(
        agent._client, FakeEmbedder(), COLL, "REGISTER binding refresh",
        limit=5, filter_expr="rfc == 9999",
    )
    assert empty == [], "the rfc=9999 filter finds nothing"
    # The last call is empty; [1] must resolve against the earlier call.
    out = asyncio.run(agent.ask("вопрос", []))
    assert out.sources == [TOP_HIT]


def test_citations_dedupe_by_chunk_preserving_order(agent) -> None:
    _script(
        agent,
        [
            Reply(content="", tool_calls=[_tc()]),
            Reply(content="Итог [2] потом [1] и снова [2]."),
        ],
    )
    hits = search(agent._client, FakeEmbedder(), COLL, "REGISTER binding refresh", limit=5)

    out = asyncio.run(agent.ask("вопрос", []))

    expected: list[Source] = []
    seen: set[str] = set()
    for n in (2, 1, 2):
        hit = hits[n - 1]
        if hit["chunk_id"] in seen:
            continue
        seen.add(hit["chunk_id"])
        expected.append(
            Source(rfc=hit["rfc"], section=hit["section_path"], page=hit["page"])
        )
    assert out.sources == expected


def test_out_of_range_citation_falls_back_to_top1(agent) -> None:
    _script(
        agent,
        [
            Reply(content="", tool_calls=[_tc()]),
            Reply(content="Итог [9]."),
        ],
    )
    out = asyncio.run(agent.ask("вопрос", []))
    assert out.sources == [TOP_HIT]


def test_ungrounded_answer_gets_top1(agent) -> None:
    _script(
        agent,
        [
            Reply(content="", tool_calls=[_tc()]),
            Reply(content="Ответ без ссылок."),
        ],
    )
    out = asyncio.run(agent.ask("вопрос", []))
    assert out.sources == [TOP_HIT]


def test_exact_refusal_yields_no_sources(agent) -> None:
    _script(
        agent,
        [
            Reply(content="", tool_calls=[_tc()]),
            Reply(content=REFUSAL),
        ],
    )
    out = asyncio.run(agent.ask("вопрос", []))
    assert out.text == REFUSAL
    assert out.sources == []


def test_wrapped_refusal_is_normalized_without_sources(agent) -> None:
    _script(
        agent,
        [
            Reply(content="", tool_calls=[_tc()]),
            Reply(content=f"Определение не найдено. {REFUSAL} Уточните запрос."),
        ],
    )
    out = asyncio.run(agent.ask("вопрос", []))
    assert out.text == REFUSAL, "wrapped refusal normalizes to the exact phrase"
    assert out.sources == []


def test_kb_found_nothing_answer_sent_as_is(tmp_path) -> None:
    agent = OpenRouterAgent(
        model="fake-model",
        milvus_uri=str(tmp_path / "empty.db"),
        budget_usd=1.0,
        api_key="test-key",
    )
    ensure_collection(agent._client, COLL, FakeEmbedder.dim)
    agent._embedder = FakeEmbedder()
    _script(
        agent,
        [
            Reply(content="", tool_calls=[_tc()]),
            Reply(content="В корпусе этого нет."),
        ],
    )
    out = asyncio.run(agent.ask("вопрос", []))
    assert out.text == "В корпусе этого нет."
    assert out.sources == []


def test_answer_without_kb_call_has_no_sources(agent) -> None:
    _script(agent, [Reply(content="Ответ [1] без поиска.")])
    out = asyncio.run(agent.ask("вопрос", []))
    assert out.text == "Ответ [1] без поиска."
    assert out.sources == []


def test_history_passed_between_system_and_question(agent) -> None:
    llm = _script(agent, [Reply(content="ok")])
    history = [
        HistoryItem(role="user", text="привет"),
        HistoryItem(role="assistant", text="здравствуйте"),
    ]
    asyncio.run(agent.ask("Как дела?", history))
    messages = llm.calls[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[0]["content"] == AGENT_SYSTEM_PROMPT
    assert messages[1] == {"role": "user", "content": "привет"}
    assert messages[2] == {"role": "assistant", "content": "здравствуйте"}
    assert messages[3] == {"role": "user", "content": "Как дела?"}


def test_composed_pipeline_answers_and_journals_once(tmp_path) -> None:
    from bot.__main__ import build_dispatcher
    from bot.sessions import MemorySessionStore
    from bot.spend import SpendJournal

    agent = _make_agent(str(tmp_path / "lite.db"))
    _script(
        agent,
        [
            Reply(content="", tool_calls=[_tc()], usage=Usage(cost_usd=0.01)),
            Reply(
                content="INVITE — метод установления сессии [1].",
                usage=Usage(cost_usd=0.02),
            ),
        ],
    )
    bot, session = make_bot()
    dp = build_dispatcher(
        make_config(),
        agent,
        sessions=MemorySessionStore(),
        spend=SpendJournal(tmp_path / "spend.jsonl"),
    )

    run(feed(dp, bot, "Что такое INVITE?"))

    texts = sent_texts(session)
    assert any("RFC" in t for t in texts), "citation block must name the RFC"
    lines = (tmp_path / "spend.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1, "exactly one journal entry per answered question"
    assert json.loads(lines[0])["cost_usd"] == pytest.approx(0.03)
