"""Naive fixed-window chunking — the baseline arm.

Concatenates the cleaned pages (the same text the structural chunker
consumes) into one stream, then cuts ~1000-char windows with 200-char
overlap. Windows are cut at whitespace: the cut lands on the last whitespace
inside the window, so a word is never split (a chunk stays at or under the
window; only a pathological word longer than the window forces overshoot).
Every chunk records the pdf + printed pages its text spans.
"""

from __future__ import annotations

import re
from typing import Optional

from .schemas import Chunk, Page

WINDOW = 1000
OVERLAP = 200  # chunking-rules: configurable 150–200

_LAST_WS_RE = re.compile(r"(?s).*\s")


class _PageMap:
    """char index -> (pdf, printed) for the concatenated stream."""

    def __init__(self, pages: list[Page]) -> None:
        self.page_of: list[tuple[int, Optional[int]]] = []
        for page in pages:
            for _ in page.text:
                self.page_of.append((page.pdf, page.printed))
            self.page_of.append((page.pdf, page.printed))  # the joining "\n"

    def at(self, idx: int) -> tuple[int, Optional[int]]:
        return self.page_of[min(max(idx, 0), len(self.page_of) - 1)]


def _last_whitespace(text: str, start: int, end: int) -> int:
    """Index just past the last whitespace char in text[start:end], or -1."""
    m = _LAST_WS_RE.match(text, start, end)
    return m.end() if m else -1


def chunk_naive(
    pages: list[Page], rfc: int, title: Optional[str], window: int = WINDOW,
    overlap: int = OVERLAP,
) -> list[Chunk]:
    text = "\n".join(page.text for page in pages)
    if not text.strip():
        return []
    page_map = _PageMap(pages)

    chunks: list[Chunk] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + window, n)
        if end < n:
            cut = _last_whitespace(text, start, end)
            if cut > start:
                end = cut
            # No whitespace in the whole window: extend to the tail word's end
            # (overshoot by at most that single word).
            else:
                while end < n and not text[end].isspace():
                    end += 1
        piece = text[start:end]
        if piece.strip():
            lead = 0
            while lead < len(piece) and piece[lead].isspace():
                lead += 1
            trail = len(piece)
            while trail > lead and piece[trail - 1].isspace():
                trail -= 1
            pdf_start, printed_start = page_map.at(start + lead)
            pdf_end, printed_end = page_map.at(start + trail - 1)
            chunks.append(
                Chunk(
                    chunk_id="",  # assigned by the CLI
                    method="naive",
                    rfc=rfc,
                    title=title,
                    text=piece,
                    section_id=None,
                    section_path=None,
                    page_start={"pdf": pdf_start, "printed": printed_start},
                    page_end={"pdf": pdf_end, "printed": printed_end},
                )
            )
        if end >= n:
            break
        start = max(end - overlap, start + 1)
        # Never start a chunk mid-word.
        while start < n and not text[start].isspace():
            start += 1
        while start < n and text[start].isspace():
            start += 1
    return chunks
