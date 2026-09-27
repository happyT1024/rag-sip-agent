"""Offline pipeline tests: guards, agent-call stage, spend journal (AD-3/AD-7).

Hand-built Updates are fed through the same Dispatcher wiring as production
(``bot.__main__.build_dispatcher``), with FakeAgent, an in-memory
SessionStore and a tmp_path JSONL journal. No network, no keys, no LLM.
"""

from __future__ import annotations

import json
import sqlite3

from aiogram.methods import SendChatAction

import bot.handlers as handlers_mod
from bot.config import BotConfig, ConfigError
from bot.fakes import FakeAgent
from bot.middleware import (
    MSG_AGENT_ERROR,
    MSG_CAP_REACHED,
    MSG_RATE_LIMITED,
    MSG_TIMEOUT,
    MSG_TOO_LONG,
    MSG_UNEXPECTED,
    MAX_HISTORY_TURNS,
    RateLimitMiddleware,
)
from bot.port import HistoryItem
from bot.sessions import SqliteSessionStore
from conftest import (
    BOT_TOKEN,
    CHAT_ID,
    OTHER_CHAT_ID,
    build_setup,
    build_test_dispatcher,
    feed,
    make_bot,
    make_config,
    run,
    sent_texts,
)


def test_defaults_follow_bot_limits(monkeypatch) -> None:
    for name in (
        "TELEGRAM_BOT_TOKEN",
        "AGENT_BACKEND",
        "SPEND_CAP_USD",
        "RATE_LIMIT_PER_MINUTE",
        "MAX_QUESTION_CHARS",
        "AGENT_TIMEOUT_SECONDS",
        "AGENT_ERROR_FALLBACK_COST_USD",
        "AGENT_MODEL",
        "MILVUS_URI",
        "OPENROUTER_BUDGET_USD",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", BOT_TOKEN)
    config = BotConfig.from_env()
    assert config.spend_cap_usd == 0.50
    assert config.rate_limit_per_minute == 5
    assert config.max_question_chars == 1000
    assert config.agent_timeout_seconds == 120
    assert config.agent_error_fallback_cost_usd == 0.01
    assert config.agent_model == "openai/gpt-4o-mini"
    assert config.milvus_uri == "http://localhost:19530"
    assert config.llm_budget_usd == 1.50
    assert config.backend == "fake"


def test_missing_token_names_variable(monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    try:
        BotConfig.from_env()
    except Exception as exc:
        assert "TELEGRAM_BOT_TOKEN" in str(exc)
    else:
        raise AssertionError("missing token must fail fast")


def _keyed_env(monkeypatch) -> None:
    for name in (
        "AGENT_BACKEND",
        "SPEND_CAP_USD",
        "RATE_LIMIT_PER_MINUTE",
        "MAX_QUESTION_CHARS",
        "AGENT_TIMEOUT_SECONDS",
        "AGENT_ERROR_FALLBACK_COST_USD",
        "AGENT_MODEL",
        "MILVUS_URI",
        "OPENROUTER_BUDGET_USD",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", BOT_TOKEN)


def test_unparsable_limit_names_variable(monkeypatch) -> None:
    _keyed_env(monkeypatch)
    monkeypatch.setenv("SPEND_CAP_USD", "пятьдесят")
    try:
        BotConfig.from_env()
    except ConfigError as exc:
        assert "SPEND_CAP_USD" in str(exc)
    else:
        raise AssertionError("unparsable limit must fail fast")


def test_zero_limit_rejected(monkeypatch) -> None:
    _keyed_env(monkeypatch)
    monkeypatch.setenv("SPEND_CAP_USD", "0")
    try:
        BotConfig.from_env()
    except ConfigError as exc:
        assert "SPEND_CAP_USD" in str(exc)
    else:
        raise AssertionError("zero limit must fail fast")


def test_nan_limit_rejected(monkeypatch) -> None:
    _keyed_env(monkeypatch)
    monkeypatch.setenv("AGENT_TIMEOUT_SECONDS", "nan")
    try:
        BotConfig.from_env()
    except ConfigError as exc:
        assert "AGENT_TIMEOUT_SECONDS" in str(exc)
    else:
        raise AssertionError("NaN limit must fail fast")


def test_placeholder_token_rejected(monkeypatch) -> None:
    _keyed_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    try:
        BotConfig.from_env()
    except ConfigError as exc:
        assert "TELEGRAM_BOT_TOKEN" in str(exc)
    else:
        raise AssertionError("placeholder token must fail fast")


def test_unknown_backend_rejected(monkeypatch) -> None:
    _keyed_env(monkeypatch)
    monkeypatch.setenv("AGENT_BACKEND", "claude")
    try:
        BotConfig.from_env()
    except ConfigError as exc:
        assert "AGENT_BACKEND" in str(exc)
    else:
        raise AssertionError("unknown backend must fail fast")


def test_empty_backend_rejected(monkeypatch) -> None:
    _keyed_env(monkeypatch)
    monkeypatch.setenv("AGENT_BACKEND", "   ")
    try:
        BotConfig.from_env()
    except ConfigError as exc:
        assert "AGENT_BACKEND" in str(exc)
    else:
        raise AssertionError("set-but-empty AGENT_BACKEND must fail fast")


def test_agent_config_vars_parsed_from_env(monkeypatch) -> None:
    _keyed_env(monkeypatch)
    monkeypatch.setenv("AGENT_MODEL", "custom/model")
    monkeypatch.setenv("MILVUS_URI", "http://milvus:19530")
    monkeypatch.setenv("OPENROUTER_BUDGET_USD", "0.25")
    config = BotConfig.from_env()
    assert config.agent_model == "custom/model"
    assert config.milvus_uri == "http://milvus:19530"
    assert config.llm_budget_usd == 0.25


def test_empty_agent_model_rejected(monkeypatch) -> None:
    _keyed_env(monkeypatch)
    monkeypatch.setenv("AGENT_MODEL", "   ")
    try:
        BotConfig.from_env()
    except ConfigError as exc:
        assert "AGENT_MODEL" in str(exc)
    else:
        raise AssertionError("empty AGENT_MODEL must fail fast")


def test_empty_milvus_uri_rejected(monkeypatch) -> None:
    _keyed_env(monkeypatch)
    monkeypatch.setenv("MILVUS_URI", "")
    try:
        BotConfig.from_env()
    except ConfigError as exc:
        assert "MILVUS_URI" in str(exc)
    else:
        raise AssertionError("empty MILVUS_URI must fail fast")


def test_non_positive_llm_budget_rejected(monkeypatch) -> None:
    _keyed_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_BUDGET_USD", "0")
    try:
        BotConfig.from_env()
    except ConfigError as exc:
        assert "OPENROUTER_BUDGET_USD" in str(exc)
    else:
        raise AssertionError("non-positive budget must fail fast")


def test_build_agent_fake_ignores_missing_key(monkeypatch) -> None:
    from bot.__main__ import build_agent

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    agent = build_agent("fake", make_config(backend="fake"))
    assert isinstance(agent, FakeAgent)


def test_build_agent_openrouter_constructs_adapter_from_config(
    monkeypatch,
) -> None:
    import agent.adapter as adapter_mod
    from bot.__main__ import build_agent

    config = make_config(
        backend="openrouter",
        agent_model="test/model",
        milvus_uri="http://localhost:19530",
        llm_budget_usd=0.25,
    )
    constructed: dict = {}

    class RecordingAgent:
        def __init__(self, **kwargs) -> None:
            constructed.update(kwargs)

        def probe(self) -> bool:
            return True

    monkeypatch.setattr(adapter_mod, "OpenRouterAgent", RecordingAgent)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    agent = build_agent("openrouter", config)
    assert isinstance(agent, RecordingAgent)
    assert constructed == {
        "model": "test/model",
        "milvus_uri": "http://localhost:19530",
        "budget_usd": 0.25,
    }


def test_build_agent_openrouter_missing_key_names_variable(monkeypatch) -> None:
    from bot.__main__ import build_agent

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    try:
        build_agent("openrouter", make_config(backend="openrouter"))
    except ConfigError as exc:
        assert "OPENROUTER_API_KEY" in str(exc)
    else:
        raise AssertionError("missing OPENROUTER_API_KEY must fail fast")


def test_build_agent_openrouter_probe_failure_names_milvus_uri(
    monkeypatch,
) -> None:
    import agent.adapter as adapter_mod
    from bot.__main__ import build_agent

    class UnreachableAgent:
        def __init__(self, **kwargs) -> None:
            pass

        def probe(self) -> bool:
            raise RuntimeError("connection refused")

    monkeypatch.setattr(adapter_mod, "OpenRouterAgent", UnreachableAgent)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    try:
        build_agent("openrouter", make_config(backend="openrouter"))
    except ConfigError as exc:
        assert "MILVUS_URI" in str(exc)
        assert "OPENROUTER_API_KEY" in str(exc)
    else:
        raise AssertionError("unreachable Milvus must fail fast")


def test_build_agent_openrouter_missing_collection_names_milvus_uri(
    monkeypatch,
) -> None:
    import agent.adapter as adapter_mod
    from bot.__main__ import build_agent

    class EmptyAgent:
        def __init__(self, **kwargs) -> None:
            pass

        def probe(self) -> bool:
            return False

    monkeypatch.setattr(adapter_mod, "OpenRouterAgent", EmptyAgent)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    try:
        build_agent("openrouter", make_config(backend="openrouter"))
    except ConfigError as exc:
        assert "MILVUS_URI" in str(exc)
    else:
        raise AssertionError("missing collection must fail fast")


def test_happy_path_citation_history_and_one_journal_entry(tmp_path) -> None:
    config, agent, sessions, spend = build_setup(tmp_path)
    bot, session = make_bot()
    dp = build_test_dispatcher(config, agent, sessions, spend)

    run(feed(dp, bot, "Что такое INVITE?"))

    texts = sent_texts(session)
    assert texts, "user must receive a reply"
    assert any(isinstance(m, SendChatAction) for m in session.calls), (
        "typing indicator expected while the agent works"
    )
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
    assert not sessions.has(CHAT_ID), "timeout must roll the question back"
    assert sessions.get(CHAT_ID) == []


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


def test_rate_limiter_prunes_emptied_windows() -> None:
    limiter = RateLimitMiddleware(rate_limit_per_minute=1)
    assert limiter._allow(1, now=0.0)
    assert not limiter._allow(1, now=1.0)
    assert limiter._allow(1, now=61.0)
    assert list(limiter._recent[1]) == [61.0]
    assert not limiter._allow(1, now=61.5)


def test_guards_run_in_contract_order_length_before_spend(tmp_path) -> None:
    config, agent, sessions, spend = build_setup(
        tmp_path, max_question_chars=10, spend_cap_usd=0.50
    )
    spend.append(OTHER_CHAT_ID, 0.50)
    bot, session = make_bot()
    dp = build_test_dispatcher(config, agent, sessions, spend)

    run(feed(dp, bot, "x" * 11))

    texts = sent_texts(session)
    assert MSG_TOO_LONG.format(limit=10) in texts, "length guard fires first"
    assert MSG_CAP_REACHED not in texts
    assert agent.calls == []
    assert spend.total() == 0.50, "spend guard must not double-journal"


def test_history_passed_to_agent_truncated_to_last_20(tmp_path) -> None:
    config, agent, sessions, spend = build_setup(tmp_path)
    for i in range(25):
        sessions.append(CHAT_ID, HistoryItem(role="user", text=f"старый {i}"))
    bot, session = make_bot()
    dp = build_test_dispatcher(config, agent, sessions, spend)

    run(feed(dp, bot, "новый вопрос?"))

    assert len(agent.calls) == 1
    passed = agent.calls[0][1]
    assert len(passed) == MAX_HISTORY_TURNS == 20
    assert passed[0] == HistoryItem(role="user", text="старый 5")
    assert passed[-1] == HistoryItem(role="user", text="старый 24")
    assert all(item.text != "новый вопрос?" for item in passed), (
        "the question itself is passed as the ask() argument, not in history"
    )
    assert len(sessions.get(CHAT_ID)) == 27, "store keeps the full history"


def test_sqlite_store_roundtrip_delete_and_isolation(tmp_path) -> None:
    store = SqliteSessionStore(tmp_path / "sessions.sqlite3")
    assert store.get(CHAT_ID) == []
    assert not store.has(CHAT_ID)

    store.append(CHAT_ID, HistoryItem(role="user", text="вопрос"))
    store.append(CHAT_ID, HistoryItem(role="assistant", text="ответ"))
    store.append(OTHER_CHAT_ID, HistoryItem(role="user", text="чужой"))

    assert store.has(CHAT_ID)
    assert store.get(CHAT_ID) == [
        HistoryItem(role="user", text="вопрос"),
        HistoryItem(role="assistant", text="ответ"),
    ]
    assert store.get(OTHER_CHAT_ID) == [HistoryItem(role="user", text="чужой")]

    store.delete(CHAT_ID)
    assert not store.has(CHAT_ID)
    assert store.get(CHAT_ID) == []
    assert store.get(OTHER_CHAT_ID) == [HistoryItem(role="user", text="чужой")]


def test_sqlite_store_corrupt_row_yields_empty_history(tmp_path) -> None:
    path = tmp_path / "sessions.sqlite3"
    store = SqliteSessionStore(path)
    store.append(CHAT_ID, HistoryItem(role="user", text="ok"))

    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE history SET items = '{not json' WHERE chat_id = ?", (CHAT_ID,)
    )
    conn.commit()
    conn.close()

    assert store.get(CHAT_ID) == []
