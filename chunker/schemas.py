"""Data types shared by the chunker pipeline."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Page:
    """One PDF page's text plus its two page numbers.

    ``pdf`` is the 0-based PDF page index. ``printed`` is the 1-based page
    number parsed from the ``[Page N]`` footer, or None when the page carries
    no parsable footer (covers, blanks).
    """

    text: str
    pdf: int
    printed: Optional[int] = None

    def lines(self) -> list[str]:
        return self.text.splitlines()


@dataclass
class Chunk:
    """One retrieval chunk with its provenance metadata."""

    chunk_id: str
    method: str  # "naive" | "structural"
    rfc: int
    title: Optional[str]
    text: str
    section_id: Optional[str] = None  # "9.1" | "A" | None for naive
    section_path: Optional[str] = None  # "9.1 Client Behavior" | None
    page_start: Optional[dict] = None  # {"pdf": int, "printed": int|None}
    page_end: Optional[dict] = None

    def __post_init__(self) -> None:
        self.n_chars = len(self.text)

    def to_record(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "method": self.method,
            "rfc": self.rfc,
            "title": self.title,
            "text": self.text,
            "n_chars": self.n_chars,
            "section_id": self.section_id,
            "section_path": self.section_path,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }
