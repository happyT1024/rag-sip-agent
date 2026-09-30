"""Agent loop: the model decides when to call knowledge_base.

Cheap-model tool-use loop with a hard step cap; if the model keeps calling
tools, one final forced call without tools produces the answer.
"""

from __future__ import annotations

import json

from rag.answer import REFUSAL
from rag.tool import TOOL_SCHEMA, parse_tool_args

AGENT_SYSTEM_PROMPT = (
    "Ты ассистент по протоколу SIP с инструментом knowledge_base — поиском по корпусу "
    "SIP-документов (RFC 3261, 3311, 3262, 3428, 2976, 3515). "
    "Если для ответа нужны факты, вызови инструмент (можно несколько раз с разными запросами), "
    "затем ответь ТОЛЬКО по найденным кускам со ссылками [n] на них. "
    "Разделы называй строго так, как они записаны в шапках кусков (например, «раздел 17 Transactions»); "
    "не выдумывай и не соединяй номера подразделов. "
    "Если просят определение — приведи его только дословной английской цитатой из куска (слово в слово, "
    "без изменений), с указанием раздела из шапки, а затем отдельно дай свой перевод цитаты на русский. "
    "Если в кусках нет определения самого термина, а описаны лишь смежные вещи, — не выдавай их за "
    "определение: перед отказом сделай ещё 1–2 вызова инструмента с другими формулировками "
    "(в том числе английскими терминами, например «SIP transaction definition»); "
    f"если определения и тогда нет, ответь ровно одной фразой: «{REFUSAL}». "
    f"Если поиск не дал ответа, ответь ровно одной фразой: «{REFUSAL}» — не выдумывай. "
    "Не используй Markdown-разметку (звёздочки, решётки, обратные кавычки): "
    "сообщения отправляются простым текстом. "
    "Отвечай кратко, по-русски."
)


def run_agent(
    llm,
    model: str,
    knowledge_base,
    question: str,
    max_steps: int = 4,
    history: list[dict] | None = None,
) -> dict:
    messages: list[dict] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    for turn in history or []:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})
    total_cost = 0.0
    tool_calls_made: list[dict] = []
    steps = 0

    for _ in range(max_steps):
        steps += 1
        reply = llm.chat(model=model, messages=messages, tools=[TOOL_SCHEMA], max_tokens=500)
        total_cost += reply.usage.cost_usd
        if not reply.tool_calls:
            return {
                "answer": (reply.content or "").strip(),
                "cost_usd": total_cost,
                "tool_calls": tool_calls_made,
                "steps": steps,
            }
        messages.append({"role": "assistant", "content": reply.content or None, "tool_calls": reply.tool_calls})
        for tc in reply.tool_calls:
            args = parse_tool_args(tc)
            try:
                try:
                    result = knowledge_base(**args)
                except TypeError:
                    result = knowledge_base(args.get("query", ""))
            except Exception as err:  # a broken tool must not kill the run
                result = f"Инструмент недоступен: {err}"
            tool_calls_made.append(args)
            messages.append(
                {"role": "tool", "tool_call_id": tc.get("id", ""), "content": str(result)}
            )

    # Step cap reached: force a final answer without tools.
    messages.append(
        {"role": "user", "content": "Хватит искать. Дай финальный ответ по уже найденным кускам."}
    )
    reply = llm.chat(model=model, messages=messages, max_tokens=400)
    total_cost += reply.usage.cost_usd
    return {
        "answer": (reply.content or "").strip(),
        "cost_usd": total_cost,
        "tool_calls": tool_calls_made,
        "steps": steps + 1,
    }
