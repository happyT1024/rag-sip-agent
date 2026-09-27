"""Offline rendering tests: citation format and 4096-safe splitting (CAP-2).

Pure functions, no Telegram objects. Verifies that citations render whole
(``RFC NNNN §<path> p.<N>``) and are always grouped on the last part.
"""

from __future__ import annotations

from bot.port import Answer, Source
from bot.rendering import (
    TELEGRAM_MESSAGE_LIMIT,
    format_citations,
    format_source,
    render_answer,
    split_text,
)

LONG = TELEGRAM_MESSAGE_LIMIT


def test_format_source_full() -> None:
    assert format_source(Source(rfc=3261, section="4.1", page=12)) == (
        "RFC 3261 §4.1 p.12"
    )


def test_format_source_section_only() -> None:
    assert format_source(Source(rfc=3311, section="3")) == "RFC 3311 §3"


def test_format_source_page_only() -> None:
    assert format_source(Source(rfc=2976, page=4)) == "RFC 2976 p.4"


def test_format_source_neither() -> None:
    assert format_source(Source(rfc=3428)) == "RFC 3428"


def test_format_citations_groups_all_sources() -> None:
    block = format_citations(
        [Source(rfc=3261, section="4.1", page=12), Source(rfc=3311, section="3")]
    )
    lines = block.splitlines()
    assert lines[0] == "Источники:"
    assert lines[1] == "— RFC 3261 §4.1 p.12"
    assert lines[2] == "— RFC 3311 §3"


def test_split_text_short_text_unchanged() -> None:
    assert split_text("привет", 100) == ["привет"]


def test_split_text_respects_limit() -> None:
    text = "\n\n".join(f"абзац {i} " + "слово " * 20 for i in range(50))
    parts = split_text(text, LONG)
    assert len(parts) > 1
    assert all(len(p) <= LONG for p in parts)
    assert " ".join("\n".join(parts).split()) == " ".join(text.split())


def test_split_text_hard_cut_when_no_separators() -> None:
    text = "x" * (LONG + 500)
    parts = split_text(text, LONG)
    assert all(len(p) <= LONG for p in parts)
    assert "".join(parts) == text


def test_render_answer_short_puts_citations_on_last_part() -> None:
    answer = Answer(
        text="Короткий ответ.", sources=[Source(rfc=3261, section="4.1", page=12)],
        cost_usd=0.0,
    )
    parts = render_answer(answer)
    assert len(parts) == 1
    assert parts[0].startswith("Короткий ответ.")
    assert "— RFC 3261 §4.1 p.12" in parts[0]


def test_render_answer_long_splits_and_keeps_citations_whole() -> None:
    body = "\n\n".join(f"Абзац номер {i}: " + "текст " * 150 for i in range(30))
    sources = [
        Source(rfc=3261, section="4.1", page=12),
        Source(rfc=3311, section="3", page=2),
    ]
    answer = Answer(text=body, sources=sources, cost_usd=0.0)

    parts = render_answer(answer)

    assert len(parts) > 1
    assert all(len(p) <= LONG for p in parts)
    joined = "\n\n".join(parts[:-1])
    assert parts[0] in body
    for source_line in (
        "— RFC 3261 §4.1 p.12",
        "— RFC 3311 §3 p.2",
    ):
        assert source_line in parts[-1], "citation must stay whole on the last part"
        assert source_line not in joined, "citations must not leak to earlier parts"


def test_render_answer_citations_get_own_part_when_last_part_is_full() -> None:
    body = "x" * LONG
    answer = Answer(text=body, sources=[Source(rfc=2976, page=4)], cost_usd=0.0)

    parts = render_answer(answer)

    assert len(parts) == 2
    assert parts[0] == body
    assert parts[1] == "Источники:\n— RFC 2976 p.4"
    assert all(len(p) <= LONG for p in parts)


def test_render_answer_no_sources_omits_citation_block() -> None:
    answer = Answer(text="Ответ без источника.", sources=[], cost_usd=0.0)
    assert render_answer(answer) == ["Ответ без источника."]
