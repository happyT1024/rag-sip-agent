"""FakeAgent (AD-1 adapter): AGENT_BACKEND=fake and the offline test double.

Replies with a configurable stub answer and a fixed stub cost, records every
ask() call so tests can assert that guards skipped the agent.
"""

from __future__ import annotations

import asyncio

from bot.port import AgentPort, Answer, HistoryItem, Source

DEFAULT_FAKE_ANSWER = (
    "Это тестовый бэкенд (AGENT_BACKEND=fake): реальный агент по корпусу "
    "SIP RFC пока не подключён. Вопрос получен и сохранён в истории чата."
)
DEFAULT_FAKE_SOURCES = [Source(rfc=3261, section="4", page=1)]
DEFAULT_FAKE_COST_USD = 0.0


class FakeAgent:
    """Stub AgentPort: fixed reply, fixed cost, call log for assertions."""

    def __init__(
        self,
        answer: str = DEFAULT_FAKE_ANSWER,
        sources: list[Source] | None = DEFAULT_FAKE_SOURCES,
        cost_usd: float = DEFAULT_FAKE_COST_USD,
        delay: float = 0.0,
    ) -> None:
        self.answer = answer
        self.sources = list(sources) if sources is not None else []
        self.cost_usd = cost_usd
        self.delay = delay
        self.calls: list[tuple[str, list[HistoryItem]]] = []

    async def ask(self, question: str, history: list[HistoryItem]) -> Answer:
        self.calls.append((question, list(history)))
        if self.delay:
            await asyncio.sleep(self.delay)
        return Answer(
            text=self.answer,
            sources=list(self.sources),
            cost_usd=self.cost_usd,
        )
