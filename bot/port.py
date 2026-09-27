"""Agent port (AD-1): the only contract between the bot and any agent core.

The bot owns this port; the real agent (OpenRouter + RAG over the SIP RFC
corpus) will implement it from the outside. ``bot/`` never imports LLM,
embedding or Milvus clients — those live behind this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class HistoryItem:
    """One turn of per-chat history, opaque to the bot."""

    role: str  # "user" | "assistant"
    text: str


@dataclass(frozen=True)
class Source:
    """Citation for an answer. ``section`` and ``page`` may be absent."""

    rfc: int
    section: str | None = None
    page: int | None = None


@dataclass(frozen=True)
class Answer:
    """Agent reply. ``sources`` is non-empty for an answerable question."""

    text: str
    sources: list[Source]
    cost_usd: float


class AgentPort(Protocol):
    """What the bot needs from an agent: answer a question given history."""

    async def ask(self, question: str, history: list[HistoryItem]) -> Answer:
        """Answer ``question`` in the context of ``history`` (oldest first).

        Implementations must return the cost of the call in ``Answer.cost_usd``.
        """
        raise NotImplementedError
