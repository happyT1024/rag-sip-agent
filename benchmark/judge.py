"""LLM judge: is the system's answer correct / wrong / an honest refusal?

Judging uses the cheap model and returns strict JSON. Refusals are detected
semantically (not by exact marker match), because the no-tools configs have
no marker instruction and can refuse in their own words.
"""

from __future__ import annotations

import json
import re

JUDGE_SYSTEM_PROMPT = (
    "Ты строгий проверяющий ответов RAG-системы по документам SIP. "
    "Дан вопрос, эталонный ответ (может отсутствовать) и ответ системы. "
    'Верни ТОЛЬКО JSON вида {"verdict": "...", "reason": "кратко по-русски"}, где verdict:\n'
    "- \"correct\" — ядро ответа совпадает с эталоном по смыслу; перефраз допустим, "
    "незначительные детали или лишние верные подробности не портят вердикт;\n"
    "- \"refused\" — система честно сообщила, что не знает/нет данных, вместо выдумывания;\n"
    "- \"wrong\" — ответ содержит неверное утверждение о сути вопроса, отвечает не на то, "
    "или это отказ там, где есть эталонный ответ.\n"
    "Если эталонного ответа нет (null): \"correct\" запрещён; \"refused\" — если система отказалась; "
    "\"wrong\" — если она что-то утвердила."
)


def build_judge_messages(question: str, gold: str | None, answer: str) -> list[dict]:
    gold_block = gold if gold is not None else "(эталонного ответа нет — проверяй только, отказалась система или выдумала)"
    user = (
        f"Вопрос: {question}\n\n"
        f"Эталонный ответ: {gold_block}\n\n"
        f"Ответ системы: {answer}\n\n"
        "Верди JSON."
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


_VERDICTS = ("correct", "wrong", "refused")


def parse_verdict(content: str) -> str:
    """Extract the verdict from the judge's reply (tolerates code fences)."""
    text = content.strip()
    fence = re.search(r"\{.*\}", text, re.DOTALL)
    if fence:
        try:
            data = json.loads(fence.group(0))
            verdict = str(data.get("verdict", "")).lower()
            if verdict in _VERDICTS:
                return verdict
        except json.JSONDecodeError:
            pass
    lowered = text.lower()
    for v in _VERDICTS:
        if v in lowered:
            return v
    return "wrong"


def judge_answer(llm, model: str, question: str, gold: str | None, answer: str) -> tuple[str, float]:
    """Returns (verdict, judge_cost_usd)."""
    reply = llm.chat(
        model=model,
        messages=build_judge_messages(question, gold, answer),
        max_tokens=120,
    )
    return parse_verdict(reply.content), reply.usage.cost_usd
