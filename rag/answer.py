"""Grounded RAG answering: top-k chunks in, cited short answer out.

Answers come ONLY from the retrieved chunks; when the answer is absent the
model must reply with the exact refusal marker, which the benchmark counts
as an honest refusal. Chunks (not whole pages) go into the context and
answers are capped at 400 tokens — the two cost guards from the spec.
"""

from __future__ import annotations

from vectorstore.ingest import Client as MilvusLike
from vectorstore.ingest import EmbedsPassages, search

COLLECTION = "rag_structural"
TOP_K = 5
REFUSAL = "В корпусе нет ответа на этот вопрос."

SYSTEM_PROMPT = (
    "Ты ассистент по протоколу SIP. Отвечай ТОЛЬКО по приведённым кускам документов. "
    "К каждому факту добавляй ссылку [n] на номер куска. "
    f"Если ответа в кусках нет, ответь ровно одной фразой: «{REFUSAL}» "
    "— не пытайся угадывать из собственных знаний. Отвечай кратко, по-русски."
)

NO_TOOLS_SYSTEM_PROMPT = (
    "Ты ассистент по протоколу SIP. Ответь на вопрос кратко, по-русски. "
    "Если не знаешь ответа, ответь ровно одной фразой: «В корпусе нет ответа на этот вопрос»."
)


def format_context(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, start=1):
        loc = f"RFC {h['rfc']}"
        if h.get("section_path"):
            loc += f" · §{h['section_path']}"
        elif h.get("page") is not None:
            loc += f" · стр. {h['page']}"
        parts.append(f"[{i}] {loc}\n{h['text']}")
    return "\n\n".join(parts)


def retrieve(
    client: MilvusLike,
    embedder: EmbedsPassages,
    question: str,
    collection: str = COLLECTION,
    k: int = TOP_K,
    filter_expr: str | None = None,
) -> list[dict]:
    return search(client, embedder, collection, question, limit=k, filter_expr=filter_expr)


def build_messages(question: str, hits: list[dict]) -> list[dict]:
    context = format_context(hits) if hits else "(пусто)"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Куски документов:\n\n{context}\n\nВопрос: {question}"},
    ]


def rag_answer(
    llm,
    client: MilvusLike,
    embedder: EmbedsPassages,
    question: str,
    model: str,
    collection: str = COLLECTION,
    k: int = TOP_K,
) -> dict:
    """Retrieve-then-answer. Returns the answer plus cost/provenance bookkeeping."""
    hits = retrieve(client, embedder, question, collection=collection, k=k)
    reply = llm.chat(model=model, messages=build_messages(question, hits), max_tokens=400)
    return {
        "answer": reply.content.strip(),
        "chunk_ids": [h["chunk_id"] for h in hits],
        "cost_usd": reply.usage.cost_usd,
        "prompt_tokens": reply.usage.prompt_tokens,
        "completion_tokens": reply.usage.completion_tokens,
    }
