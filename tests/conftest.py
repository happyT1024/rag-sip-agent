"""Shared fixtures for the chunker and bot test suites.

Everything here is offline and keyless: synthetic text pages, no PDFs, no
network, no LLM. The real-corpus smoke test lives in test_corpus_smoke.py.
The bot fixtures feed hand-built aiogram Updates through Dispatcher with a
mocked Bot API session (AD-7).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from typing import Any, AsyncGenerator

import pytest
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.methods import SendChatAction, SendMessage
from aiogram.types import (
    Chat,
    Message,
    TelegramObject,
    Update,
    User,
)

from chunker.clean import clean_pages
from chunker.extract import clean_page_lines
from chunker.schemas import Page

from bot.config import BotConfig
from bot.fakes import FakeAgent
from bot.port import Source
from bot.sessions import MemorySessionStore
from bot.spend import SpendJournal

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mini_rfc.txt"
RFC = 4000
TITLE = "The Test Protocol"

BOT_TOKEN = "42:AbCdEfGhIjKlMnOpQrStUv"
CHAT_ID = 100
OTHER_CHAT_ID = 200

_MESSAGE_IDS = count(1)
_UPDATE_IDS = count(1)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class MockSession(BaseSession):
    """Offline Bot API double: records methods, fabricates responses."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[TelegramObject] = []
        self._message_id = count(1000)

    async def close(self) -> None:
        pass

    async def make_request(
        self, bot: Bot, method: TelegramObject, timeout: int | None = None
    ) -> Any:
        self.calls.append(method)
        if isinstance(method, SendMessage):
            return Message(
                message_id=next(self._message_id),
                date=datetime.now(timezone.utc),
                chat=Chat(id=method.chat_id, type="private"),
                text=method.text,
            )
        if isinstance(method, SendChatAction):
            return True
        return True

    async def stream_content(
        self,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: int = 30,
        chunk_size: int = 65536,
        raise_for_status: bool = True,
    ) -> AsyncGenerator[bytes, None]:
        yield b""


def make_bot() -> tuple[Bot, MockSession]:
    session = MockSession()
    return Bot(token=BOT_TOKEN, session=session), session


def make_message(text: str, chat_id: int = CHAT_ID) -> Message:
    return Message(
        message_id=next(_MESSAGE_IDS),
        date=datetime.now(timezone.utc),
        chat=Chat(id=chat_id, type="private"),
        from_user=User(id=chat_id, is_bot=False, first_name="Test"),
        text=text,
    )


def make_update(message: Message) -> Update:
    return Update(update_id=next(_UPDATE_IDS), message=message)


async def feed(dp: Dispatcher, bot: Bot, text: str, chat_id: int = CHAT_ID) -> None:
    await dp.feed_update(bot, make_update(make_message(text, chat_id)))


def sent_texts(session: MockSession) -> list[str]:
    return [m.text for m in session.calls if isinstance(m, SendMessage)]


def build_test_dispatcher(
    config: Any, agent: Any, sessions: Any, spend: Any
) -> Dispatcher:
    """Production wiring (bot.__main__.build_dispatcher) with injected stores."""
    from bot.__main__ import build_dispatcher

    return build_dispatcher(config, agent, sessions=sessions, spend=spend)


def make_config(**overrides) -> BotConfig:
    defaults = dict(
        token=BOT_TOKEN,
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
    """(config, agent, sessions, spend) wired like production, all in-memory."""
    config = make_config(**config_overrides)
    # Small delay so ChatActionSender's worker gets a loop iteration to send
    # the typing action before the (instant) fake answer resolves.
    agent = FakeAgent(sources=[Source(rfc=3261, section="4.1", page=12)], delay=0.01)
    sessions = MemorySessionStore()
    spend = SpendJournal(tmp_path / "spend.jsonl")
    return config, agent, sessions, spend


def _fixture_pages() -> list[str]:
    raw = FIXTURE.read_text(encoding="utf-8")
    return raw.split("%%PAGE%%\n")


@pytest.fixture
def mini_pages() -> list[Page]:
    """Fixture pages with headers/footers stripped, like extract_pages gives."""
    pages: list[Page] = []
    for idx, raw in enumerate(_fixture_pages()):
        kept, printed = clean_page_lines(raw.splitlines())
        pages.append(Page(text="\n".join(kept), pdf=idx, printed=printed))
    return pages


@pytest.fixture
def mini_cleaned(mini_pages: list[Page]) -> list[Page]:
    cleaned = clean_pages(mini_pages)
    assert cleaned, "cleaning the fixture must keep the body"
    return cleaned
