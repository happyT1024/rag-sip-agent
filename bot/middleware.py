"""The one guarded question pipeline (AD-3), in fixed contractual order:

    ErrorCatch (outer) → RateLimit → Length → SpendCap → AgentCall → handler

Rejecting middlewares send the Russian refusal themselves and skip the rest.
AgentCall is the sole call site of ``AgentPort.ask``: it snapshots history,
appends the question pre-ask, wraps ``ask`` in ``AGENT_TIMEOUT_SECONDS``,
journals the cost append-only after ``ask`` and before the reply is sent,
appends the answer post-ask (unless the chat was reset mid-flight) and hands
the ``Answer`` to the handler via DI data. Unexpected errors anywhere in the
chain are caught by the outer middleware: Russian apology, process survives.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject
from aiogram.utils.chat_action import ChatActionSender

from bot.config import BotConfig
from bot.port import AgentPort, Answer, HistoryItem
from bot.sessions import SessionStore
from bot.spend import SpendJournal

logger = logging.getLogger(__name__)

RATE_WINDOW_SECONDS = 60.0

# Context-cost guard: the agent sees only the last MAX_HISTORY_TURNS turns;
# the full history stays in the SessionStore.
MAX_HISTORY_TURNS = 20

MSG_RATE_LIMITED = "Слишком много вопросов. Подождите минуту и попробуйте ещё раз."
MSG_TOO_LONG = (
    "Вопрос слишком длинный (лимит — {limit} символов). "
    "Сократите его и попробуйте ещё раз."
)
MSG_CAP_REACHED = (
    "Исчерпан лимит расходов на запросы к языковой модели. "
    "Обратитесь к владельцу бота."
)
MSG_TIMEOUT = (
    "Не удалось получить ответ вовремя — превышено время ожидания. "
    "Вопрос не записан, попробуйте ещё раз позже."
)
MSG_AGENT_ERROR = (
    "Произошла ошибка при обработке вопроса. Попробуйте ещё раз позже."
)
MSG_UNEXPECTED = "Произошла непредвиденная ошибка. Попробуйте ещё раз позже."


async def _reply(event: TelegramObject, text: str) -> None:
    """Send ``text`` to the chat behind ``event`` (message or callback)."""
    if isinstance(event, CallbackQuery) and event.message is not None:
        await event.message.answer(text)
    elif isinstance(event, Message):
        await event.answer(text)


class ErrorCatchMiddleware(BaseMiddleware):
    """Outermost stage: never let an error kill the polling process."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> None:
        try:
            await handler(event, data)
        except Exception as exc:  # noqa: BLE001 — deliberate, AD-3
            chat = event.chat.id if isinstance(event, Message) else None
            logger.error(
                "chat_id=%s unhandled error: %s: %s",
                chat,
                type(exc).__name__,
                exc,
            )
            await _reply(event, MSG_UNEXPECTED)


class RateLimitMiddleware(BaseMiddleware):
    """Sliding 60-second window per chat; refused attempts are not counted."""

    def __init__(self, rate_limit_per_minute: int) -> None:
        self.rate_limit_per_minute = rate_limit_per_minute
        self._recent: dict[int, deque[float]] = {}

    def _allow(self, chat_id: int, now: float) -> bool:
        window = self._recent.get(chat_id)
        if window is not None:
            while window and now - window[0] >= RATE_WINDOW_SECONDS:
                window.popleft()
            if not window:
                del self._recent[chat_id]
                window = None
        if window is None:
            self._recent[chat_id] = deque((now,))
            return True
        if len(window) >= self.rate_limit_per_minute:
            return False
        window.append(now)
        return True

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> None:
        assert isinstance(event, Message)  # mounted on dp.message only
        now = time.monotonic()
        if not self._allow(event.chat.id, now):
            logger.info("chat_id=%s rate limited", event.chat.id)
            await _reply(event, MSG_RATE_LIMITED)
            return
        await handler(event, data)


class LengthMiddleware(BaseMiddleware):
    """Refuse questions longer than MAX_QUESTION_CHARS — context-cost guard."""

    def __init__(self, max_question_chars: int) -> None:
        self.max_question_chars = max_question_chars

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> None:
        assert isinstance(event, Message)  # mounted on dp.message only
        text = event.text or ""
        if len(text) > self.max_question_chars:
            logger.info(
                "chat_id=%s question too long: %d chars", event.chat.id, len(text)
            )
            await _reply(event, MSG_TOO_LONG.format(limit=self.max_question_chars))
            return
        await handler(event, data)


class SpendCapMiddleware(BaseMiddleware):
    """Refuse everything once the journal total reaches SPEND_CAP_USD."""

    def __init__(self, spend: SpendJournal, spend_cap_usd: float) -> None:
        self.spend = spend
        self.spend_cap_usd = spend_cap_usd

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> None:
        assert isinstance(event, Message)  # mounted on dp.message only
        total = self.spend.total()
        if total >= self.spend_cap_usd:
            logger.info(
                "chat_id=%s spend cap reached: %.4f >= %.4f USD",
                event.chat.id,
                total,
                self.spend_cap_usd,
            )
            await _reply(event, MSG_CAP_REACHED)
            return
        await handler(event, data)


class AgentCallMiddleware(BaseMiddleware):
    """Innermost stage: the sole ``ask()`` call site of the whole bot.

    Order of operations (contractual, AD-3/AD-4): snapshot history → append
    question → typing indicator + ``wait_for(ask)`` → journal cost → append
    answer (skipped if ``/reset`` deleted the key mid-flight) → hand the
    ``Answer`` to the handler as ``data["answer"]``. On timeout the pre-ask
    append is rolled back to the snapshot: no cost, no history append.
    """

    def __init__(
        self,
        agent: AgentPort,
        sessions: SessionStore,
        spend: SpendJournal,
        config: BotConfig,
    ) -> None:
        self.agent = agent
        self.sessions = sessions
        self.spend = spend
        self.config = config
        self._locks: dict[int, asyncio.Lock] = {}

    def _lock_for(self, chat_id: int) -> asyncio.Lock:
        """Serialize same-chat exchanges so snapshot/append/rollback never
        interleave (aiogram processes updates concurrently)."""
        if chat_id not in self._locks:
            self._locks[chat_id] = asyncio.Lock()
        return self._locks[chat_id]

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> None:
        assert isinstance(event, Message)  # mounted on dp.message only
        chat_id = event.chat.id
        question = event.text or ""

        async with self._lock_for(chat_id):
            full_history = self.sessions.get(chat_id)
            self.sessions.append(chat_id, HistoryItem(role="user", text=question))

            try:
                started = time.monotonic()
                async with ChatActionSender(bot=data["bot"], chat_id=chat_id):
                    answer = await asyncio.wait_for(
                        self.agent.ask(question, full_history[-MAX_HISTORY_TURNS:]),
                        timeout=self.config.agent_timeout_seconds,
                    )
            except asyncio.TimeoutError:
                logger.info("chat_id=%s agent timeout", chat_id)
                self._rollback(chat_id, full_history)
                await _reply(event, MSG_TIMEOUT)
                return
            except Exception as exc:  # noqa: BLE001 — fallback cost, then apology
                logger.error(
                    "chat_id=%s agent error: %s", chat_id, type(exc).__name__
                )
                self.spend.append(
                    chat_id, self.config.agent_error_fallback_cost_usd
                )
                await _reply(event, MSG_AGENT_ERROR)
                return

            latency = time.monotonic() - started
            self.spend.append(chat_id, answer.cost_usd)
            if self.sessions.has(chat_id):
                self.sessions.append(
                    chat_id, HistoryItem(role="assistant", text=answer.text)
                )
            else:
                logger.info(
                    "chat_id=%s session reset mid-flight; answer not re-appended",
                    chat_id,
                )
            logger.info(
                "chat_id=%s answered in %.2fs, cost=%.4f USD",
                chat_id,
                latency,
                answer.cost_usd,
            )

            data["answer"] = answer
            await handler(event, data)

    def _rollback(self, chat_id: int, snapshot: list[HistoryItem]) -> None:
        """Restore the pre-ask snapshot after a timeout (no history append).

        If the chat key vanished mid-flight (/reset), it stays deleted.
        """
        if not self.sessions.has(chat_id):
            return
        self.sessions.delete(chat_id)
        for item in snapshot:
            self.sessions.append(chat_id, item)
