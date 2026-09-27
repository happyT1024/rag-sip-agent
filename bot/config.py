"""Frozen bot configuration from environment variables (AD-6).

Limits and defaults follow ``_bmad-output/specs/spec-telegram-bot/bot-limits.md``.
Loading is fail-fast: a missing or unparsable variable aborts the start with a
Russian message naming the variable. ``load_dotenv()`` is called by the
composition root (``bot/__main__.py``), not here, so tests stay keyless.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

TOKEN_PATTERN = re.compile(r"\d+:[A-Za-z0-9_-]{20,}")

DEFAULT_SPEND_CAP_USD = 0.50
DEFAULT_RATE_LIMIT_PER_MINUTE = 5
DEFAULT_MAX_QUESTION_CHARS = 1000
DEFAULT_AGENT_TIMEOUT_SECONDS = 120
DEFAULT_AGENT_ERROR_FALLBACK_COST_USD = 0.01
DEFAULT_AGENT_MODEL = "openai/gpt-4o-mini"
DEFAULT_MILVUS_URI = "http://localhost:19530"
DEFAULT_LLM_BUDGET_USD = 1.50
BACKENDS = ("fake", "openrouter")


class ConfigError(Exception):
    """Fatal configuration problem; message is user-facing (Russian)."""


@dataclass(frozen=True)
class BotConfig:
    token: str
    backend: str
    spend_cap_usd: float
    rate_limit_per_minute: int
    max_question_chars: int
    agent_timeout_seconds: float
    agent_error_fallback_cost_usd: float
    agent_model: str = DEFAULT_AGENT_MODEL
    milvus_uri: str = DEFAULT_MILVUS_URI
    llm_budget_usd: float = DEFAULT_LLM_BUDGET_USD

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> BotConfig:
        env = os.environ if env is None else env

        token = env.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise ConfigError(
                "Ошибка: не задана переменная окружения TELEGRAM_BOT_TOKEN. "
                "Укажите токен бота от @BotFather в файле .env и перезапустите."
            )
        if not TOKEN_PATTERN.fullmatch(token):
            raise ConfigError(
                "Ошибка: TELEGRAM_BOT_TOKEN имеет неверный формат — ожидается "
                "строка вида <id_бота>:<ключ>. Проверьте токен от @BotFather "
                "в файле .env."
            )

        backend = _non_empty_str(env, "AGENT_BACKEND", "fake").lower()
        if backend not in BACKENDS:
            raise ConfigError(
                f"Ошибка: AGENT_BACKEND={backend!r} не поддерживается. "
                f"Допустимые значения: {', '.join(BACKENDS)}."
            )

        return cls(
            token=token,
            backend=backend,
            spend_cap_usd=_positive_float(
                env, "SPEND_CAP_USD", DEFAULT_SPEND_CAP_USD
            ),
            rate_limit_per_minute=_positive_int(
                env, "RATE_LIMIT_PER_MINUTE", DEFAULT_RATE_LIMIT_PER_MINUTE
            ),
            max_question_chars=_positive_int(
                env, "MAX_QUESTION_CHARS", DEFAULT_MAX_QUESTION_CHARS
            ),
            agent_timeout_seconds=_positive_float(
                env, "AGENT_TIMEOUT_SECONDS", DEFAULT_AGENT_TIMEOUT_SECONDS
            ),
            agent_error_fallback_cost_usd=_positive_float(
                env,
                "AGENT_ERROR_FALLBACK_COST_USD",
                DEFAULT_AGENT_ERROR_FALLBACK_COST_USD,
            ),
            agent_model=_non_empty_str(env, "AGENT_MODEL", DEFAULT_AGENT_MODEL),
            milvus_uri=_non_empty_str(env, "MILVUS_URI", DEFAULT_MILVUS_URI),
            llm_budget_usd=_positive_float(
                env, "OPENROUTER_BUDGET_USD", DEFAULT_LLM_BUDGET_USD
            ),
        )


def _non_empty_str(env: dict[str, str], name: str, default: str) -> str:
    raw = (env.get(name) or "").strip()
    if not raw:
        if name in env:
            raise ConfigError(
                f"Ошибка: переменная {name} не должна быть пустой."
            )
        return default
    return raw


def _int(env: dict[str, str], name: str, default: int) -> int:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(
            f"Ошибка: переменная {name} должна быть целым числом, получено {raw!r}."
        ) from None


def _positive_int(env: dict[str, str], name: str, default: int) -> int:
    value = _int(env, name, default)
    if value <= 0:
        raise ConfigError(
            f"Ошибка: переменная {name} должна быть положительным целым числом, "
            f"получено {value}."
        )
    return value


def _float(env: dict[str, str], name: str, default: float) -> float:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(
            f"Ошибка: переменная {name} должна быть числом, получено {raw!r}."
        ) from None


def _positive_float(env: dict[str, str], name: str, default: float) -> float:
    value = _float(env, name, default)
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(
            f"Ошибка: переменная {name} должна быть положительным конечным "
            f"числом, получено {value}."
        )
    return value
