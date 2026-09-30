"""OpenRouterAgent: the real agent core behind ``bot.port.AgentPort`` (AD-1).

Implements the bot-owned port from the outside: wraps the tool-use loop
(``agent.loop.run_agent``), the ``knowledge_base`` search over Milvus and the
blocking ``OpenRouterClient``. All blocking LLM/retrieval work runs in a
worker thread (``asyncio.to_thread``) so the event loop keeps polling
Telegram; the middleware's ``wait_for`` timeout still aborts on time, while
an orphaned thread's cost stays unknowable (review precedent BH12c).

Citations: kb hits are recorded per tool call and the model's ``[n]`` markers
are numbered within a call — the most recent **non-empty** call wins (the
final grounding context the model actually saw). Out-of-range markers are
dropped; an answer with no valid marker falls back to the top-1 hit; no
recorded hits at all → the answer is sent without citations (logged, not
fatal). The exact refusal marker yields ``sources=[]`` so the renderer skips
the citation block.
"""

from __future__ import annotations

import asyncio
import logging
import re

from pymilvus import MilvusClient

from agent.loop import run_agent
from bot.port import AgentPort, Answer, HistoryItem, Source
from llm.openrouter import OpenRouterClient
from rag.answer import COLLECTION, REFUSAL, TOP_K, format_context
from vectorstore.embed import Embedder
from vectorstore.ingest import search

logger = logging.getLogger(__name__)

PROBE_COLLECTION = COLLECTION
KB_NO_RESULTS = "Ничего не найдено по этому запросу."
_CITATION_RE = re.compile(r"\[(\d+)\]")


class OpenRouterAgent:
    """Real AgentPort adapter: OpenRouter + RAG over the SIP RFC corpus."""

    def __init__(
        self,
        model: str,
        milvus_uri: str,
        budget_usd: float,
        api_key: str | None = None,
        max_steps: int = 4,
    ) -> None:
        self.model = model
        self.max_steps = max_steps
        self._client = MilvusClient(milvus_uri)
        self._embedder = Embedder()
        self._llm = OpenRouterClient(api_key=api_key, budget_usd=budget_usd)

    def probe(self) -> bool:
        """Startup check: the ingested collection must be reachable and non-empty."""
        if not self._client.has_collection(PROBE_COLLECTION):
            return False
        stats = self._client.get_collection_stats(PROBE_COLLECTION)
        return int(stats.get("row_count", 0)) > 0

    async def ask(self, question: str, history: list[HistoryItem]) -> Answer:
        return await asyncio.to_thread(self._ask_sync, question, list(history))

    def _ask_sync(self, question: str, history: list[HistoryItem]) -> Answer:
        # Per-call state lives in locals: two chats may ask concurrently
        # (the middleware lock is per-chat), so nothing mutable is shared.
        calls: list[list[dict]] = []
        out = run_agent(
            self._llm,
            self.model,
            self._make_knowledge_base(calls),
            question,
            max_steps=self.max_steps,
            history=[{"role": h.role, "content": h.text} for h in history],
        )
        text = out["answer"]
        # The model may wrap the refusal phrase in its own words; normalize so
        # a refusal never carries a contradictory "Источники" block.
        if REFUSAL in text:
            text = REFUSAL
        return Answer(
            text=text,
            sources=self._sources_for(text, calls),
            cost_usd=out["cost_usd"],
        )

    def _make_knowledge_base(self, calls: list[list[dict]]):
        """The kb tool with per-call hit recording (last non-empty call wins)."""

        def knowledge_base(query: str, rfc: int | None = None) -> str:
            filter_expr = f"rfc == {int(rfc)}" if rfc is not None else None
            hits = search(
                self._client,
                self._embedder,
                COLLECTION,
                query,
                limit=TOP_K,
                filter_expr=filter_expr,
            )
            calls.append(hits)
            if not hits:
                return KB_NO_RESULTS
            return format_context(hits)

        return knowledge_base

    def _sources_for(self, text: str, calls: list[list[dict]]) -> list[Source]:
        if text == REFUSAL:
            return []
        hits = next((c for c in reversed(calls) if c), [])
        if not hits:
            logger.info("no kb hits recorded; answering without citations")
            return []
        picked: list[Source] = []
        seen: set[str] = set()
        for raw in _CITATION_RE.findall(text):
            n = int(raw)
            if not 1 <= n <= len(hits):
                continue
            hit = hits[n - 1]
            if hit["chunk_id"] in seen:
                continue
            seen.add(hit["chunk_id"])
            picked.append(_source_for_hit(hit))
        if not picked:
            picked = [_source_for_hit(hits[0])]
        return picked


def _source_for_hit(hit: dict) -> Source:
    return Source(
        rfc=int(hit["rfc"]),
        section=hit.get("section_path"),
        page=hit.get("page"),
    )
