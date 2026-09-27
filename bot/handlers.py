"""Telegram handlers (CAP-1, CAP-2, CAP-4): /start /help /reset + questions.

All dependencies arrive via aiogram DI (AD-2). Handlers never mutate chat
history — only ``/reset`` drops the whole key, as AD-4 prescribes; the
question handler only renders the ``Answer`` that AgentCallMiddleware placed
into DI data. Onboarding commands never call the agent.

Routers are built fresh per composition (aiogram allows a router to be
attached to a single parent), the composition root mounts the guards on the
question router and includes both into the Dispatcher.
"""

from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.types import Message

from bot.port import Answer
from bot.rendering import render_answer
from bot.sessions import SessionStore

logger = logging.getLogger(__name__)

MSG_ONBOARDING = (
    "Привет! Я ассистент по SIP-телефонии: отвечаю на вопросы по корпусу "
    "RFC с указанием источника.\n\n"
    "Покрываемые документы:\n"
    "• RFC 3261 — SIP: Session Initiation Protocol\n"
    "• RFC 3311 — метод UPDATE\n"
    "• RFC 3515 — метод REFER\n"
    "• RFC 3262 — надёжные предварительные ответы (PRACK)\n"
    "• RFC 3428 — метод MESSAGE\n"
    "• RFC 2976 — метод INFO\n\n"
    "Просто задайте вопрос по протоколу SIP в этом чате.\n"
    "Команды: /help — справка, /reset — очистить историю диалога."
)

MSG_RESET = "История диалога очищена. Задавайте новый вопрос!"


def build_commands_router() -> Router:
    router = Router(name="commands")

    @router.message(F.text.regexp(r"^/start\b"))
    async def start_handler(message: Message) -> None:
        logger.info("chat_id=%s /start", message.chat.id)
        await message.answer(MSG_ONBOARDING)

    @router.message(F.text.regexp(r"^/help\b"))
    async def help_handler(message: Message) -> None:
        logger.info("chat_id=%s /help", message.chat.id)
        await message.answer(MSG_ONBOARDING)

    @router.message(F.text.regexp(r"^/reset\b"))
    async def reset_handler(message: Message, sessions: SessionStore) -> None:
        chat_id = message.chat.id
        sessions.delete(chat_id)
        logger.info("chat_id=%s /reset", chat_id)
        await message.answer(MSG_RESET)

    return router


def build_question_router() -> Router:
    """Fresh router per composition: build_dispatcher mounts the guards on it."""
    router = Router(name="question")

    @router.message(F.text, ~F.text.startswith("/"))
    async def question_handler(message: Message, answer: Answer) -> None:
        for part in render_answer(answer):
            await message.answer(part)

    return router
