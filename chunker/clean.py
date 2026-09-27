"""Cleaning: one service-part-free body text per RFC.

Line-level state machine over the extracted pages:

1. **Front matter** — everything before the first body ``1`` heading is
   dropped (cover, abstract, Status of this Memo, the whole TOC block). The
   body-start anchor is the first ``1``-heading that carries no TOC leader
   dots and is not followed by a bare (wrapped) TOC page number, so TOC
   entries like ``1    Introduction .....  8`` never fire it, whatever the
   leader style (``.....`` or ``. . . .``).
2. **Tail cut** — the document is truncated at the first service-part anchor
   (``Authors' Addresses`` / ``Full Copyright Statement`` / ``Intellectual
   Property`` / ``Acknowledg...``) that (a) is heading-shaped, (b) sits in the
   last quarter of the pages, and (c) comes after the last ``References``
   heading, so sandwiched References sections survive (RFC 3515: §8
   Acknowledgments → §9 References → §10 Intellectual Property).

The output keeps the page map: a list of cleaned ``Page`` objects, the same
shape ``extract_pages`` produced, ready for both chunkers.
"""

from __future__ import annotations

import re
from typing import Optional

from .schemas import Page

# Body start: a flush-or-indented "1" heading with an uppercase title.
_BODY1_RE = re.compile(r"^\s*1\.?\s+([A-Z].*)$")
# TOC leader styles: "......." and ". . . . ."
_LEADER_RE = re.compile(r"\.{3,}|(?:\.\s){3,}")
_BARE_PAGENUM_RE = re.compile(r"^\s*\d{1,3}\s*$")

# Service-part anchors, optionally preceded by a section number
# ("13 Acknowledgements", "17. Authors' Addresses", "Full Copyright Statement").
_TAIL_ANCHOR_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(Authors?' Addresses?|Full Copyright Statement|Intellectual Property|Acknowledg)"
)
# "9. References" (3515) and "15.Normative References" (3428, no space after
# the dot in the PDF text layer) alike.
_REFERENCES_RE = re.compile(
    r"^\s*\d+(?:\.\d+)*\.?\s*(?:Normative\s+|Informative\s+)?References\b"
)


def _is_heading_shaped(line: str, max_len: int) -> bool:
    stripped = line.strip()
    return (
        bool(stripped)
        and len(stripped) <= max_len
        and not _LEADER_RE.search(stripped)
    )


def _is_body_start(line: str, following: list[str]) -> bool:
    """True if this line is the first real body section heading."""
    if not _BODY1_RE.match(line):
        return False
    if not _is_heading_shaped(line, max_len=80):
        return False
    # Guard against a TOC entry whose leader dots / page number wrapped onto
    # the next lines: the next non-blank line must not be a bare page number
    # or a leader-dots line.
    for nxt in following:
        if not nxt.strip():
            continue
        if _BARE_PAGENUM_RE.match(nxt) or _LEADER_RE.search(nxt):
            return False
        break
    return True


def _find_body_start(pages: list[Page]) -> Optional[tuple[int, int]]:
    """First (page_index, line_index) that starts the RFC body."""
    for p_idx, page in enumerate(pages):
        lines = page.lines()
        for l_idx, line in enumerate(lines):
            if _is_body_start(line, lines[l_idx + 1 :]):
                return p_idx, l_idx
    return None


def _last_quarter_start(n_pages: int) -> int:
    return n_pages - n_pages // 4


def _find_tail_cut(
    pages: list[Page], body_start: tuple[int, int]
) -> Optional[tuple[int, int]]:
    """Position of the tail cut: the first service-part anchor after the
    last References heading.

    Candidates are heading-shaped anchor lines anywhere from the body start
    on. Preference order: (1) first candidate after the last References
    heading — so References sandwiched between service sections is kept
    (RFC 3515: §8 Acknowledgments → §9 References → §10 Intellectual
    Property) and early boilerplate *mentions* never cut (they precede the
    References heading); (2) when no References heading exists or no anchor
    follows it, the first in-quarter candidate; (3) otherwise the first
    candidate overall (short documents whose service tail starts before the
    last quarter begins — RFC 2976).
    """
    quarter = _last_quarter_start(len(pages))
    refs_pos: Optional[tuple[int, int]] = None
    candidates: list[tuple[int, int]] = []
    for p_idx in range(body_start[0], len(pages)):
        from_line = body_start[1] if p_idx == body_start[0] else 0
        lines = pages[p_idx].lines()
        for l_idx in range(from_line, len(lines)):
            line = lines[l_idx]
            if _REFERENCES_RE.match(line) and _is_heading_shaped(line, 80):
                refs_pos = (p_idx, l_idx)
            if _TAIL_ANCHOR_RE.match(line) and _is_heading_shaped(line, 60):
                candidates.append((p_idx, l_idx))
    if not candidates:
        return None
    if refs_pos is not None:
        for cand in candidates:
            if cand > refs_pos:
                return cand
    in_quarter = [c for c in candidates if c[0] >= quarter]
    if in_quarter:
        return in_quarter[0]
    return candidates[0]


def _collapse_blanks(lines: list[str]) -> list[str]:
    """Collapse runs of blank lines to a single blank; drop leading blanks."""
    out: list[str] = []
    for line in lines:
        if not line.strip():
            if out and out[-1] != "":
                out.append("")
            continue
        out.append(line)
    while out and out[-1] == "":
        out.pop()
    return out


def _dehyphenate(lines: list[str]) -> list[str]:
    """Join hyphenated line wraps: "case-" + "insensitive" -> "case-insensitive".

    A line is joined directly (no space, hyphen kept) with the next line when
    it ends with "-" preceded by a letter or digit (a word continuation, so
    table borders and dash-rule example lines never join) and the next line
    starts with a lower- or uppercase letter ("Record-" + "Route" ->
    "Record-Route"). Chained wraps merge in one pass.
    """
    out: list[str] = []
    for line in lines:
        prev = out[-1] if out else None
        first = line.lstrip()[:1]
        if (
            prev is not None
            and prev.endswith("-")
            and len(prev) >= 2
            and prev[-2].isalnum()
            and (first.islower() or first.isupper())
        ):
            out[-1] = prev + line.lstrip()
        else:
            out.append(line)
    return out


def clean_pages(pages: list[Page]) -> list[Page]:
    """Drop front matter and the service tail; keep the page map."""
    body_start = _find_body_start(pages)
    if body_start is None:
        return []
    tail_cut = _find_tail_cut(pages, body_start)

    def after_cut(p_idx: int, l_idx: int) -> bool:
        return tail_cut is not None and (p_idx, l_idx) >= tail_cut

    cleaned: list[Page] = []
    for p_idx in range(body_start[0], len(pages)):
        from_line = body_start[1] if p_idx == body_start[0] else 0
        if after_cut(p_idx, from_line):
            break
        kept: list[str] = []
        stop = False
        for l_idx in range(from_line, len(pages[p_idx].lines())):
            if after_cut(p_idx, l_idx):
                stop = True
                break
            line = pages[p_idx].lines()[l_idx].rstrip()
            kept.append(line)
        kept = _collapse_blanks(_dehyphenate(kept))
        text = "\n".join(kept)
        if text.strip():
            cleaned.append(Page(text=text, pdf=pages[p_idx].pdf, printed=pages[p_idx].printed))
        if stop:
            break
    return cleaned
