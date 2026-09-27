"""The ``knowledge_base`` tool for the agent: schema, description, callable.

This is the piece the spec requires tests for ("тест на три случая"); the
tests run on Milvus Lite with a fake embedder — no LLM, no key, no docker.
"""

from __future__ import annotations

import json

from rag.answer import format_context
from vectorstore.ingest import Client as MilvusLike
from vectorstore.ingest import EmbedsPassages, search

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "knowledge_base",
        "description": (
            "Поиск по корпусу SIP-документов: RFC 3261, 3311, 3262, 3428, 2976, 3515. "
            "Возвращает до 5 релевантных кусков текста с указанием RFC, раздела и страницы. "
            "Вызывай, когда для ответа нужны факты из этих документов."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Поисковый запрос: тема вопроса ключевыми словами (русскими или английскими терминами).",
                },
                "rfc": {
                    "type": "integer",
                    "description": "Опционально: ограничить поиск одним RFC (3261, 3311, 3262, 3428, 2976 или 3515).",
                },
            },
            "required": ["query"],
        },
    },
}


def make_knowledge_base(
    client: MilvusLike,
    embedder: EmbedsPassages,
    collection: str = "rag_structural",
    k: int = 5,
):
    """Returns the tool callable ``knowledge_base(query, rfc=None) -> str``."""

    def knowledge_base(query: str, rfc: int | None = None) -> str:
        filter_expr = f"rfc == {int(rfc)}" if rfc is not None else None
        hits = search(client, embedder, collection, query, limit=k, filter_expr=filter_expr)
        if not hits:
            return "Ничего не найдено по этому запросу."
        return format_context(hits)

    return knowledge_base


def parse_tool_args(reply_tool_call: dict) -> dict:
    """Safely parse a tool_call's JSON arguments."""
    raw = reply_tool_call.get("function", {}).get("arguments") or "{}"
    try:
        args = json.loads(raw)
        return args if isinstance(args, dict) else {}
    except json.JSONDecodeError:
        return {}
