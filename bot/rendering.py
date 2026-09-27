"""Answer rendering (CAP-2): citations + Telegram-safe splitting.

Citations render as ``RFC NNNN §<path> p.<N>`` (section or page may be
absent). Answers longer than the 4096-char Telegram limit are split into
whole parts at paragraph/line/word boundaries; the citation block is appended
to the last part (or sent as its own part if it does not fit), so a citation
is never cut in half.
"""

from __future__ import annotations

from bot.port import Answer, Source

TELEGRAM_MESSAGE_LIMIT = 4096


def format_source(source: Source) -> str:
    out = [f"RFC {source.rfc}"]
    if source.section:
        out.append(f"§{source.section}")
    if source.page is not None:
        out.append(f"p.{source.page}")
    return " ".join(out)


def format_citations(sources: list[Source]) -> str:
    return "Источники:\n" + "\n".join(f"— {format_source(s)}" for s in sources)


def split_text(text: str, limit: int) -> list[str]:
    """Split into parts of at most ``limit`` chars, never mid-word if avoidable."""
    text = text.strip()
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    while len(text) > limit:
        cut = 0
        for sep in ("\n\n", "\n", " "):
            cut = text.rfind(sep, 1, limit + 1)
            if cut > 0:
                break
        if cut <= 0:
            cut = limit
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


def render_answer(answer: Answer, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Render an answer into whole message parts, citations on the last one.

    Empty body parts are dropped; if nothing remains, the citations alone
    (or a placeholder) are sent — Telegram rejects empty messages.
    """
    parts = [part for part in split_text(answer.text, limit) if part]
    if answer.sources:
        citation_parts = split_text(format_citations(answer.sources), limit)
        if parts and len(parts[-1]) + 2 + len(citation_parts[0]) <= limit:
            parts[-1] = f"{parts[-1]}\n\n{citation_parts[0]}"
            parts.extend(citation_parts[1:])
        else:
            parts.extend(citation_parts)
    if not parts:
        parts = ["(получен пустой ответ)"]
    return parts
