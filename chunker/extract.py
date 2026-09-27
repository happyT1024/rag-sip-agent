"""Per-page text extraction from the RFC PDFs.

One ``page.get_text()`` call per page (pymupdf). Running headers and footers
are removed wherever they appear in the extracted line list (MuPDF reading
order is not reliable: footers/headers can sit mid-page), and the printed RFC
page number is captured from the footer. Both wrapped forms observed in the
corpus are handled:

* header year wrapped to its own line: ``... September`` + ``2002``
* footer page number wrapped to its own line: ``... [Page`` + ``12]``
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Optional

import pymupdf

from .schemas import Page

# Running headers are typeset with wide gaps after the RFC number
# ("RFC 3261            SIP: Session Initiation Protocol           June 2002").
# Body references ("RFC 2543-compliant systems.", "RFC 3261 [1], ...") have at
# most one space after the number and never match.
_HEADER_RE = re.compile(r"^RFC \d+\s{2,}\S")
_TRAILING_YEAR_RE = re.compile(r"\d{4}\s*$")
_BARE_YEAR_RE = re.compile(r"^\s*\d{4}\s*$")

# Footer: "<author> Standards Track [Page N]" possibly wrapped after "[Page".
_FOOTER_RE = re.compile(r"Standards Track\s+\[Page\s*(\d+)\]\s*$")
_FOOTER_WRAPPED_RE = re.compile(r"Standards Track\s+\[Page\s*$")
_BARE_PAGENUM_RE = re.compile(r"^\s*(\d+)\]\s*$")

# Title parsed out of the running header: RFC NNNN <gaps> TITLE <gaps> Month Year
_TITLE_RE = re.compile(r"^RFC \d+\s{2,}(.*?)\s{2,}[A-Za-z]+ \d{4}\s*$")

RFC_FILENAME_RE = re.compile(r"RFC\s*(\d+)\.pdf$", re.IGNORECASE)


def rfc_number_from_path(path: str | Path) -> int:
    m = RFC_FILENAME_RE.search(str(path))
    if not m:
        raise ValueError(f"cannot parse RFC number from {path!r}")
    return int(m.group(1))


def _merge_wrapped(lines: list[str]) -> list[str]:
    """Join header/footer pairs that pymupdf split across two lines."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        if (
            nxt is not None
            and _HEADER_RE.match(line)
            and not _TRAILING_YEAR_RE.search(line)
            and _BARE_YEAR_RE.match(nxt)
        ):
            out.append(line.rstrip() + " " + nxt.strip())
            i += 2
            continue
        if (
            nxt is not None
            and _FOOTER_WRAPPED_RE.search(line)
            and _BARE_PAGENUM_RE.match(nxt)
        ):
            out.append(line.rstrip() + nxt.strip())
            i += 2
            continue
        out.append(line)
        i += 1
    return out


def clean_page_lines(raw_lines: list[str]) -> tuple[list[str], Optional[int]]:
    """Strip running header/footer lines wherever they occur.

    Returns the kept lines (whitespace-stripped at the edges) and the printed
    page number parsed from the footer, or None if the page has no parsable
    footer.
    """
    lines = _merge_wrapped(list(raw_lines))
    kept: list[str] = []
    printed: Optional[int] = None
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _FOOTER_RE.search(line)
        if m:
            printed = int(m.group(1))
            i += 1
            continue
        if _HEADER_RE.match(line):
            # If the year sits alone on the next line, drop it too.
            if (
                not _TRAILING_YEAR_RE.search(line)
                and i + 1 < len(lines)
                and _BARE_YEAR_RE.match(lines[i + 1])
            ):
                i += 2
            else:
                i += 1
            continue
        kept.append(line.strip())
        i += 1
    return kept, printed


def extract_pages(pdf_path: str | Path) -> list[Page]:
    """Extract all pages of one PDF with headers/footers removed."""
    pages: list[Page] = []
    with pymupdf.open(str(pdf_path)) as doc:
        for idx in range(doc.page_count):
            raw = doc[idx].get_text()
            kept, printed = clean_page_lines(raw.splitlines())
            pages.append(Page(text="\n".join(kept), pdf=idx, printed=printed))
    return pages


def parse_title(pdf_path: str | Path) -> Optional[str]:
    """Parse the RFC title from the raw running header (most common variant).

    Opens the PDF itself because ``extract_pages`` strips header lines — the
    title lives in exactly those lines.
    """
    counter: Counter[str] = Counter()
    with pymupdf.open(str(pdf_path)) as doc:
        for idx in range(doc.page_count):
            for line in _merge_wrapped(doc[idx].get_text().splitlines()):
                m = _TITLE_RE.match(line)
                if m:
                    counter[m.group(1).strip()] += 1
    if counter:
        return counter.most_common(1)[0][0]
    return None
