"""Composition root (AD-2) and long-polling entry point (AD-5).

The only place that loads .env (python-dotenv), builds the frozen BotConfig,
selects the AgentPort adapter from AGENT_BACKEND, wires dependencies into
aiogram DI and starts Dispatcher.start_polling in a single asyncio process.
Run with: ``python -m bot``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

from aiogram import Bot, Dispatcher
from dotenv import load_dotenv

from bot.config import BotConfig, ConfigError
from bot.fakes import FakeAgent
from bot.handlers import build_commands_router, build_question_router
from bot.middleware import (
    AgentCallMiddleware,
    ErrorCatchMiddleware,
    LengthMiddleware,
    RateLimitMiddleware,
    SpendCapMiddleware,
)
from bot.port import AgentPort
from bot.sessions import SqliteSessionStore, SessionStore
from bot.spend import SpendJournal

logger = logging.getLogger(__name__)

MSG_OPENROUTER_NEEDS_KEY = (
    "Ошибка: не задана переменная окружения OPENROUTER_API_KEY. "
    "Укажите ключ OpenRouter в файле .env и перезапустите бота."
)
MSG_OPENROUTER_INIT_FAILED = (
    "Ошибка: не удалось инициализировать бэкенд AGENT_BACKEND=openrouter. "
    "Проверьте, что Milvus запущен и доступен по адресу из MILVUS_URI "
    "({uri}), коллекция rag_structural загружена "
    "(.venv/bin/python -m vectorstore.ingest) и что ключ OPENROUTER_API_KEY "
    "корректен. Причина: {reason}"
)


def build_agent(backend: str, config: BotConfig) -> AgentPort:
    """Select the port adapter (AD-2). The only place that touches the
    agent core: the import stays function-local, so ``AGENT_BACKEND=fake``
    never imports pymilvus/sentence_transformers (AD-1)."""
    if backend == "fake":
        return FakeAgent()
    if backend == "openrouter":
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise ConfigError(MSG_OPENROUTER_NEEDS_KEY)
        try:
            from agent.adapter import OpenRouterAgent

            agent = OpenRouterAgent(
                model=config.agent_model,
                milvus_uri=config.milvus_uri,
                budget_usd=config.llm_budget_usd,
            )
            if not agent.probe():
                raise RuntimeError(
                    "коллекция rag_structural не найдена — выполните "
                    ".venv/bin/python -m vectorstore.ingest"
                )
        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(
                MSG_OPENROUTER_INIT_FAILED.format(
                    uri=config.milvus_uri,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            ) from exc
        return agent
    raise ConfigError(
        f"Ошибка: AGENT_BACKEND={backend!r} не поддерживается. "
        f"Допустимые значения: {', '.join(('fake', 'openrouter'))}."
    )


def build_dispatcher(
    config: BotConfig,
    agent: AgentPort,
    sessions: SessionStore | None = None,
    spend: SpendJournal | None = None,
) -> Dispatcher:
    sessions = sessions if sessions is not None else SqliteSessionStore()
    spend = spend if spend is not None else SpendJournal()
    dp = Dispatcher()
    dp["config"] = config
    dp["agent"] = agent
    dp["sessions"] = sessions
    dp["spend"] = spend

    dp.message.outer_middleware(ErrorCatchMiddleware())
    question_router = build_question_router()
    question_router.message.middleware(RateLimitMiddleware(config.rate_limit_per_minute))
    question_router.message.middleware(LengthMiddleware(config.max_question_chars))
    question_router.message.middleware(SpendCapMiddleware(spend, config.spend_cap_usd))
    question_router.message.middleware(
        AgentCallMiddleware(agent, sessions, spend, config)
    )
    dp.include_router(build_commands_router())
    dp.include_router(question_router)
    return dp


async def run(config: BotConfig, agent: AgentPort) -> Any:
    bot = Bot(token=config.token)
    dp = build_dispatcher(config, agent)
    logging.getLogger(__name__).info("starting long polling, backend=%s", config.backend)
    return await dp.start_polling(bot)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    load_dotenv()
    try:
        config = BotConfig.from_env()
        agent = build_agent(config.backend, config)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    try:
        asyncio.run(run(config, agent))
    except Exception as exc:
        logger.exception("bot failed: %s", type(exc).__name__)
        print(
            f"Ошибка при запуске бота: {type(exc).__name__}. Проверьте "
            "правильность TELEGRAM_BOT_TOKEN и подключение к сети.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
