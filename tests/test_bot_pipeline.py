"""Offline pipeline tests: guards, agent-call stage, spend journal (AD-3/AD-7).

Hand-built Updates are fed through the same Dispatcher wiring as production
(``bot.__main__.build_dispatcher``), with FakeAgent, an in-memory
SessionStore and a tmp_path JSONL journal. No network, no keys, no LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import bot.handlers as handlers_mod
from bot.config import BotConfig
from bot.fakes import FakeAgent
from bot.middleware import (
    MSG_AGENT_ERROR,
    MSG_CAP_REACHED,
    MSG_RATE_LIMITED,
    MSG_TIMEOUT,
    MSG_TOO_LONG,
    MSG_UNEXPECTED,
    RateLimitMiddleware,
)
from bot.port import HistoryItem, Source
from bot.sessions import MemorySessionStore
from bot.spend import SpendJournal
from conftest import (
    CHAT_ID,
    OTHER_CHAT_ID,
    build_test_dispatcher,
    feed,
    make_bot,
    run,
    sent_texts,
)


def make_config(**overrides) -> BotConfig:
    defaults = dict(
        token="42:TEST",
        backend="fake",
        spend_cap_usd=10.0,
        rate_limit_per_minute=5,
        max_question_chars=1000,
        agent_timeout_seconds=120.0,
        agent_error_fallback_cost_usd=0.01,
    )
    defaults.update(overrides)
    return BotConfig(**defaults)


def build_setup(tmp_path: Path, **config_overrides):
    config = make_config(**config_overrides)
    agent = FakeAgent(sources=[Source(rfc=3261, section="4.1", page=12)])
    sessions = MemorySessionStore()
    spend = SpendJournal(tmp_path / "spend.jsonl")
    return config, agent, sessions, spend


def test_defaults_follow_bot_limits(monkeypatch) -> None:
    for name in (
        "TELEGRAM_BOT_TOKEN",
        "AGENT_BACKEND",
        "SPEND_CAP_USD",
        "RATE_LIMIT_PER_MINUTE",
        "MAX_QUESTION_CHARS",
        "AGENT_TIMEOUT_SECONDS",
        "AGENT_ERROR_FALLBACK_COST_USD",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "42:TEST")
    config = BotConfig.from_env()
    assert config.spend_cap_usd == 0.50
    assert config.rate_limit_per_minute == 5
    assert config.max_question_chars == 1000
    assert config.agent_timeout_seconds == 120
    assert config.agent_error_fallback_cost_usd == 0.01
    assert config.backend == "fake"


def test_missing_token_names_variable(monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    try:
        BotConfig.from_env()
    except Exception as exc:
        assert "TELEGRAM_BOT_TOKEN" in str(exc)
    else:
        raise AssertionError("missing token must fail fast")


def test_happy_path_citation_history_and_one_journal_entry(tmp_path) -> None:
    config, agent, sessions, spend = build_setup(tmp_path)
    bot, session = make_bot()
    dp = build_test_dispatcher(config, agent, sessions, spend)

    run(feed(dp, bot, "Что такое INVITE?"))

    texts = sent_texts(session)
    assert texts, "user must receive a reply"
    full = "\n".join(texts)
    assert "RFC 3261 §4.1 p.12" in full
    assert all(len(t) <= 4096 for t in texts)
    assert len(agent.calls) == 1
    assert agent.calls[0][0] == "Что такое INVITE?"
    lines = spend.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1, "exactly one spend entry per answered question"
    entry = json.loads(lines[0])
    assert entry["chat_id"] == CHAT_ID
    assert entry["cost_usd"] == 0.0
    assert "ts" in entry
    assert sessions.get(CHAT_ID) == [
        HistoryItem(role="user", text="Что такое INVITE?"),
        HistoryItem(role="assistant", text=agent.answer),
    ]


def test_rate_limit_refuses_question_over_quota(tmp_path) -> None:
    config, agent, sessions, spend = build_setup(tmp_path, rate_limit_per_minute=2)
    bot, session = make_bot()
    dp = build_test_dispatcher(config, agent, sessions, spend)

    async def scenario() -> None:
        await feed(dp, bot, "Первый?")
        await feed(dp, bot, "Второй?")
        await feed(dp, bot, "Третий?")

    run(scenario())

    assert len(agent.calls) == 2, "the refused question must not reach the agent"
    assert MSG_RATE_LIMITED in sent_texts(session)
    assert spend.total() == 0.0


def test_long_question_refused_without_agent_call(tmp_path) -> None:
    config, agent, sessions, spend = build_setup(tmp_path, max_question_chars=10)
    bot, session = make_bot()
    dp = build_test_dispatcher(config, agent, sessions, spend)

    run(feed(dp, bot, "x" * 11))

    assert agent.calls == []
    assert MSG_TOO_LONG.format(limit=10) in sent_texts(session)
    assert not sessions.has(CHAT_ID)
    assert spend.total() == 0.0


def test_spend_cap_refuses_without_agent_call(tmp_path) -> None:
    config, agent, sessions, spend = build_setup(tmp_path, spend_cap_usd=0.50)
    spend.append(OTHER_CHAT_ID, 0.50)
    bot, session = make_bot()
    dp = build_test_dispatcher(config, agent, sessions, spend)

    run(feed(dp, bot, "Вопрос?"))

    assert agent.calls == []
    assert MSG_CAP_REACHED in sent_texts(session)
    assert spend.total() == 0.50


def test_agent_error_journals_fallback_cost_then_apologizes(tmp_path) -> None:
    class BrokenAgent(FakeAgent):
        async def ask(self, question, history):
            raise RuntimeError("boom")

    config, agent, sessions, spend = build_setup(tmp_path)
    broken = BrokenAgent()
    bot, session = make_bot()
    dp = build_test_dispatcher(config, broken, sessions, spend)

    run(feed(dp, bot, "Вопрос?"))

    assert spend.total() == config.agent_error_fallback_cost_usd == 0.01
    assert MSG_AGENT_ERROR in sent_texts(session)
    assert [item.role for item in sessions.get(CHAT_ID)] == ["user"]


def test_agent_timeout_records_no_cost_and_no_answer(tmp_path) -> None:
    config, agent, sessions, spend = build_setup(tmp_path, agent_timeout_seconds=0.05)
    slow = FakeAgent(delay=1.0)
    bot, session = make_bot()
    dp = build_test_dispatcher(config, slow, sessions, spend)

    run(feed(dp, bot, "Вопрос?"))

    assert MSG_TIMEOUT in sent_texts(session)
    assert spend.total() == 0.0
    assert [item.role for item in sessions.get(CHAT_ID)] == ["user"]


def test_midflight_reset_reply_sends_and_session_stays_empty(tmp_path) -> None:
    config, agent, sessions, spend = build_setup(tmp_path)

    class ResettingAgent(FakeAgent):
        async def ask(self, question, history):
            sessions.delete(CHAT_ID)
            return await super().ask(question, history)

    resetting = ResettingAgent()
    bot, session = make_bot()
    dp = build_test_dispatcher(config, resetting, sessions, spend)

    run(feed(dp, bot, "Вопрос?"))

    assert sent_texts(session), "reply must still send"
    assert not sessions.has(CHAT_ID)
    assert sessions.get(CHAT_ID) == []
    assert spend.total() == 0.0


def test_unexpected_error_apologizes_and_process_survives(
    tmp_path, monkeypatch
) -> None:
    config, agent, sessions, spend = build_setup(tmp_path)
    bot, session = make_bot()
    dp = build_test_dispatcher(config, agent, sessions, spend)

    def boom(answer, limit=4096):
        raise RuntimeError("boom")

    monkeypatch.setattr(handlers_mod, "render_answer", boom)
    run(feed(dp, bot, "Вопрос?"))
    assert MSG_UNEXPECTED in sent_texts(session)

    monkeypatch.undo()
    run(feed(dp, bot, "Второй вопрос?"))
    texts = sent_texts(session)
    assert any("RFC 3261" in t for t in texts), "process must keep serving"


def test_rate_limiter_is_sliding_window_per_chat() -> None:
    limiter = RateLimitMiddleware(rate_limit_per_minute=2)
    assert limiter._allow(1, now=0.0)
    assert limiter._allow(1, now=1.0)
    assert not limiter._allow(1, now=2.0)
    assert limiter._allow(1, now=60.5)
    assert limiter._allow(2, now=2.0)
