"""Offline handler tests: onboarding commands and /reset semantics (AD-4/AD-7).

Feeds hand-built Updates through the production Dispatcher wiring with
FakeAgent and an in-memory SessionStore. No network, no keys, no LLM.
"""

from __future__ import annotations

from bot.fakes import FakeAgent
from bot.handlers import MSG_ONBOARDING, MSG_RESET
from bot.port import HistoryItem
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
from test_bot_pipeline import build_setup


def test_start_replies_onboarding_without_agent_call(tmp_path) -> None:
    config, agent, sessions, spend = build_setup(tmp_path)
    bot, session = make_bot()
    dp = build_test_dispatcher(config, agent, sessions, spend)

    run(feed(dp, bot, "/start"))

    texts = sent_texts(session)
    assert len(texts) == 1
    assert "RFC 3261" in texts[0]
    assert "/reset" in texts[0]
    assert agent.calls == []
    assert spend.total() == 0.0
    assert not sessions.has(CHAT_ID)


def test_help_replies_onboarding_without_agent_call(tmp_path) -> None:
    config, agent, sessions, spend = build_setup(tmp_path)
    bot, session = make_bot()
    dp = build_test_dispatcher(config, agent, sessions, spend)

    run(feed(dp, bot, "/help"))

    assert sent_texts(session) == [MSG_ONBOARDING]
    assert agent.calls == []


def test_reset_clears_only_that_chat(tmp_path) -> None:
    config, agent, sessions, spend = build_setup(tmp_path)
    sessions.append(CHAT_ID, HistoryItem(role="user", text="мой вопрос"))
    sessions.append(
        OTHER_CHAT_ID, HistoryItem(role="user", text="чужой вопрос")
    )
    bot, session = make_bot()
    dp = build_test_dispatcher(config, agent, sessions, spend)

    run(feed(dp, bot, "/reset"))

    assert MSG_RESET in sent_texts(session)
    assert not sessions.has(CHAT_ID)
    assert sessions.get(CHAT_ID) == []
    assert sessions.get(OTHER_CHAT_ID) == [
        HistoryItem(role="user", text="чужой вопрос")
    ]


def test_reset_of_unknown_chat_is_polite(tmp_path) -> None:
    config, agent, sessions, spend = build_setup(tmp_path)
    bot, session = make_bot()
    dp = build_test_dispatcher(config, agent, sessions, spend)

    run(feed(dp, bot, "/reset"))

    assert MSG_RESET in sent_texts(session)


def test_reset_drops_history_so_next_answer_starts_fresh(tmp_path) -> None:
    config, agent, sessions, spend = build_setup(tmp_path)
    bot, session = make_bot()
    dp = build_test_dispatcher(config, agent, sessions, spend)

    async def scenario() -> None:
        await feed(dp, bot, "Первый вопрос?")
        await feed(dp, bot, "/reset")
        await feed(dp, bot, "Второй вопрос?")

    run(scenario())

    assert agent.calls[1][1] == [], "history must be empty after /reset"


def test_unknown_command_never_reaches_agent(tmp_path) -> None:
    config, agent, sessions, spend = build_setup(tmp_path)
    bot, session = make_bot()
    dp = build_test_dispatcher(config, agent, sessions, spend)

    run(feed(dp, bot, "/unknowncommand"))

    assert agent.calls == []
    assert sent_texts(session) == []
