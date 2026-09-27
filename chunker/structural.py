"""RFC-section-aware structural chunking — the main arm.

Splits the shared cleaned text at section boundaries detected with a
sequence-aware heading parser (indentation is advisory only: corpus headings
are both flush-left and indented). Each section is then sized:

* ``MAX = 2500`` chars: split by paragraphs; adjacent chunks overlap by
  10–15% of the emitted chunk length so a boundary sentence survives.
* ``MIN = 200`` chars: merged into the neighboring chunk (next section; if
  none, the previous).
* ABNF grammar rules (``name = ...`` plus continuation lines) and table
  blocks (3+ consecutive lines with 2+ multi-space column gaps) are atomic —
  never split internally, even when that makes a chunk exceed MAX.
* Every chunk carries a context prefix as its first line:
  ``RFC 3261 — SIP: Session Initiation Protocol · §9.1 Client Behavior``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .schemas import Chunk, Page

MAX = 2500
MIN = 200
OVERLAP_MIN = 0.10  # 10–15% of the emitted chunk length
OVERLAP_MAX = 0.15

_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s*([A-Z].*)$")
_APPENDIX_RE = re.compile(r"^Appendix ([A-Z])(?:\.(\d+))?\.?\s+([A-Z].*)$")
_ABNF_START_RE = re.compile(r"^\s*[A-Za-z][A-Za-z0-9-]*\s*=\s")
# Column gaps are 3+ spaces; RFC prose separates sentences with exactly two.
_TABLE_GAP_RE = re.compile(r"\S\s{3,}\S")
_LEADER_RE = re.compile(r"\.{3,}|(?:\.\s){3,}")
# Matches up to and including the last whitespace char of a string slice.
_LAST_WS_RE = re.compile(r"(?s).*\s")


def _is_heading_candidate(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return False
    if _LEADER_RE.search(stripped):
        return False
    if _ABNF_START_RE.match(line):
        return False
    return bool(_HEADING_RE.match(stripped) or _APPENDIX_RE.match(stripped))


def _is_table_line(line: str) -> bool:
    return len(_TABLE_GAP_RE.findall(line)) >= 2


class _HeadingTracker:
    """Sequence-aware acceptance: rejects numbered-list false positives.

    A top-level heading ``n`` is accepted only if ``n`` continues the section
    sequence (first must be 1). A subheading ``a.b.c`` is accepted only if its
    parent ``a.b`` is open and ``c`` continues the sibling sequence (or is 1).
    Rejected candidates are demoted to body text.
    """

    def __init__(self) -> None:
        self.stack: list[tuple[int, ...]] = []
        self.tops_seen = 0
        self.last_appendix: Optional[str] = None
        self.last_appendix_sub: Optional[int] = None

    def accept_number(self, num: tuple[int, ...]) -> bool:
        if len(num) == 1:
            if num[0] != self.tops_seen + 1:
                return False
            self.tops_seen = num[0]
            self.stack = [num]
            return True
        parent = num[:-1]
        if parent not in self.stack:
            return False
        if num[-1] != 1 and parent + (num[-1] - 1,) not in self.stack:
            return False
        self.stack = [t for t in self.stack if len(t) < len(num)]
        self.stack.append(num)
        return True

    def accept_appendix(self, letter: str, sub: Optional[int]) -> bool:
        if sub is None:
            # New appendix letter: the first must be A, later ones continue
            # the alphabet (A, B, C ...).
            if self.last_appendix is None:
                ok = letter == "A"
            else:
                ok = ord(letter) == ord(self.last_appendix) + 1
            if ok:
                self.last_appendix = letter
                self.last_appendix_sub = None
            return ok
        # Subsection A.n: must belong to the currently open appendix letter
        # and continue its subsection sequence (1, 2, 3 ...).
        if self.last_appendix != letter:
            return False
        expected = 1 if self.last_appendix_sub is None else self.last_appendix_sub + 1
        if sub != expected:
            return False
        self.last_appendix_sub = sub
        return True


@dataclass
class _Block:
    kind: str  # "para" | "abnf" | "table"
    text: str
    page_start: tuple[int, Optional[int]]  # (pdf, printed)
    page_end: tuple[int, Optional[int]]


@dataclass
class _Section:
    section_id: str
    title: str
    heading_line: str
    heading_page: tuple[int, Optional[int]]
    blocks: list[_Block] = field(default_factory=list)
    page_start: Optional[tuple[int, Optional[int]]] = None
    page_end: Optional[tuple[int, Optional[int]]] = None

    def add(self, block: _Block) -> None:
        self.blocks.append(block)
        if self.page_start is None:
            self.page_start = block.page_start
        self.page_end = block.page_end

    def n_chars(self) -> int:
        return len(self.heading_line) + sum(len(b.text) + 1 for b in self.blocks)


def _parse_sections(pages: list[Page]) -> list[_Section]:
    tracker = _HeadingTracker()
    sections: list[_Section] = []
    pending: list[_Block] = []  # blocks seen before the first accepted heading
    current: Optional[_Section] = None
    paragraph: list[tuple[str, tuple[int, Optional[int]]]] = []
    table_run: list[tuple[str, tuple[int, Optional[int]]]] = []

    def flush_paragraph() -> None:
        if not paragraph:
            return
        text = " ".join(l for l, _ in paragraph).strip()
        if text:
            block = _Block("para", text, paragraph[0][1], paragraph[-1][1])
            _emit_block(block)
        paragraph.clear()

    def flush_table_run() -> None:
        if not table_run:
            return
        if len(table_run) >= 3:
            block = _Block(
                "table",
                "\n".join(l for l, _ in table_run),
                table_run[0][1],
                table_run[-1][1],
            )
            _emit_block(block)
        else:
            for line, pos in table_run:
                paragraph.append((line, pos))
        table_run.clear()

    def _emit_block(block: _Block) -> None:
        if current is None:
            pending.append(block)
        else:
            current.add(block)

    for page in pages:
        lines = page.lines()
        i = 0
        while i < len(lines):
            line = lines[i]
            pos = (page.pdf, page.printed)
            stripped = line.strip()

            if not stripped:
                flush_paragraph()
                flush_table_run()
                i += 1
                continue

            if _is_heading_candidate(line):
                flush_paragraph()
                flush_table_run()
                m = _HEADING_RE.match(stripped)
                a = _APPENDIX_RE.match(stripped)
                accepted = False
                if m:
                    num = tuple(int(p) for p in m.group(1).split("."))
                    accepted = tracker.accept_number(num)
                    if accepted:
                        current = _Section(
                            section_id=m.group(1),
                            title=m.group(2).strip(),
                            heading_line=stripped,
                            heading_page=pos,
                        )
                        sections.append(current)
                        for block in pending:
                            current.add(block)
                        pending.clear()
                elif a:
                    sub = int(a.group(2)) if a.group(2) else None
                    accepted = tracker.accept_appendix(a.group(1), sub)
                    if accepted:
                        sid = a.group(1) + (f".{sub}" if sub else "")
                        current = _Section(
                            section_id=sid,
                            title=a.group(3).strip(),
                            heading_line=stripped,
                            heading_page=pos,
                        )
                        sections.append(current)
                        for block in pending:
                            current.add(block)
                        pending.clear()
                if not accepted:
                    paragraph.append((stripped, pos))
                i += 1
                continue

            if _ABNF_START_RE.match(line):
                flush_paragraph()
                flush_table_run()
                block_lines = [(stripped, pos)]
                j = i + 1
                while j < len(lines):
                    nxt = lines[j]
                    s = nxt.strip()
                    if (
                        not s
                        or _ABNF_START_RE.match(nxt)
                        or _is_heading_candidate(nxt)
                        or _is_table_line(nxt)
                        or not (nxt[:1].isspace() or s[:1] in "/;")
                    ):
                        break
                    block_lines.append((s, (page.pdf, page.printed)))
                    j += 1
                block = _Block(
                    "abnf",
                    "\n".join(l for l, _ in block_lines),
                    block_lines[0][1],
                    block_lines[-1][1],
                )
                _emit_block(block)
                i = j
                continue

            if _is_table_line(line):
                flush_paragraph()
                table_run.append((stripped, pos))
                blanks = 0
                j = i + 1
                while j < len(lines):
                    nxt = lines[j]
                    if not nxt.strip():
                        blanks += 1
                        if blanks > 1:
                            break
                        table_run.append(("", (page.pdf, page.printed)))
                        j += 1
                        continue
                    if not _is_table_line(nxt):
                        break
                    blanks = 0
                    table_run.append((nxt.strip(), (page.pdf, page.printed)))
                    j += 1
                while table_run and not table_run[-1][0]:
                    table_run.pop()
                i = j
                flush_table_run()
                continue

            paragraph.append((stripped, pos))
            i += 1

        flush_paragraph()
        flush_table_run()

    flush_paragraph()
    flush_table_run()
    return sections


def _carry_overlap(blocks: list[_Block], emitted_len: int) -> list[_Block]:
    """Trailing blocks forming a 10–15% overlap for the next chunk.

    Takes whole trailing paragraphs until the 10% floor is met; never exceeds
    the 15% ceiling with whole blocks. When even the last paragraph alone
    bursts the ceiling (huge paragraph), a ~12.5% whitespace-cut suffix of it
    is carried instead so the boundary sentence still survives.
    """
    target = OVERLAP_MIN * emitted_len
    ceiling = OVERLAP_MAX * emitted_len
    carried: list[_Block] = []
    total = 0
    for block in reversed(blocks):
        step = len(block.text) + 1
        if total + step > ceiling:
            break
        carried.insert(0, block)
        total += step
        if total >= target:
            return carried
    if carried:
        # Under the floor, but the next block would burst the ceiling:
        # a small overlap beats none.
        return carried
    last = blocks[-1]
    want = int((OVERLAP_MIN + OVERLAP_MAX) / 2 * emitted_len)
    if want >= len(last.text):
        return [last]
    cut = _LAST_WS_RE.match(last.text, 0, len(last.text) - want)
    start = cut.end() if cut else 0
    suffix = last.text[start:].strip()
    if suffix:
        return [_Block("para", suffix, last.page_start, last.page_end)]
    return [last]


def _unit_bounds(blocks: list[_Block]) -> tuple[list[_Block], tuple, tuple]:
    """Blocks plus the min/max (pdf, printed) span they cover.

    Merge-appended sections can put earlier-page blocks after later-page
    ones, so the unit's true page span is a min/max, not first/last.
    """
    start = min(blocks, key=lambda b: b.page_start[0]).page_start
    end = max(blocks, key=lambda b: b.page_end[0]).page_end
    return (list(blocks), start, end)


def _split_section(
    section: _Section, max_chars: int
) -> list[tuple[list[_Block], tuple, tuple]]:
    """Pack a section's blocks into <= max_chars units with overlap."""
    if section.n_chars() <= max_chars:
        start = section.page_start or section.heading_page
        end = section.page_end or section.heading_page
        return [(section.blocks, start, end)]

    # Oversized paragraphs are pre-split; ABNF/table blocks stay atomic.
    flat: list[_Block] = []
    for block in section.blocks:
        if block.kind == "para" and len(block.text) > max_chars:
            flat.extend(_hard_split_paragraph(block, max_chars))
        else:
            flat.append(block)

    heading_len = len(section.heading_line) + 1
    units: list[tuple[list[_Block], tuple, tuple]] = []
    acc: list[_Block] = []
    acc_len = heading_len
    carried_only = False  # acc holds just the overlap carry, no fresh content

    def emit() -> None:
        nonlocal acc, acc_len, carried_only
        if not acc or carried_only:
            return
        units.append((_unit_bounds(acc)))
        carry = _carry_overlap(acc, acc_len)
        acc = list(carry)
        acc_len = heading_len + sum(len(b.text) + 1 for b in acc)
        carried_only = bool(acc)

    for block in flat:
        blen = len(block.text) + 1
        if blen > max_chars and block.kind in ("abnf", "table"):
            # Atomic and oversized: flush acc, then emit it alone; keep the
            # overlap carry so the following unit still starts with context.
            emit()
            units.append(
                (
                    [block],
                    (block.page_start[0], block.page_start[1]),
                    (block.page_end[0], block.page_end[1]),
                )
            )
            carried_only = bool(acc)
            continue
        if acc and not carried_only and acc_len + blen > max_chars:
            emit()
        # After an emit the acc holds only the carry: the first fresh block
        # always joins it (unit may exceed max by at most the carry size).
        acc.append(block)
        acc_len += blen
        carried_only = False
    emit()
    return units


def _hard_split_paragraph(block: _Block, max_chars: int) -> list[_Block]:
    """Fallback for a single paragraph over MAX: whitespace windows, 12% overlap."""
    pieces: list[_Block] = []
    text = block.text
    overlap = int(max_chars * 0.12)
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            ws = end
            while ws < len(text) and not text[ws].isspace():
                ws += 1
            end = ws
        piece = text[start:end].strip()
        if piece:
            pieces.append(_Block("para", piece, block.page_start, block.page_end))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        while start < len(text) and not text[start].isspace():
            start += 1
    return pieces or [block]


def _merge_short_sections(sections: list[_Section]) -> list[_Section]:
    """Sections under MIN are folded into the neighboring chunk.

    A short section precedes its host section, so its blocks are prepended
    (heading line first) to keep document order inside the merged chunk.
    """
    kept: list[_Section] = []
    buffer: list[_Section] = []
    for section in sections:
        if section.n_chars() < MIN:
            buffer.append(section)
            continue
        if buffer:
            section.blocks[:0] = [
                block for buffered in buffer for block in _buffered_blocks(buffered)
            ]
            for buffered in buffer:
                section.page_start = _min_pos(
                    section.page_start, buffered.page_start
                )
            buffer.clear()
        kept.append(section)
    if buffer:
        if kept:
            # Trailing buffer follows the last kept section: append in order.
            last = kept[-1]
            for buffered in buffer:
                last.blocks.extend(_buffered_blocks(buffered))
                last.page_end = _max_pos(last.page_end, buffered.page_end)
        else:
            # Whole RFC under MIN: one chunk holding everything.
            first = buffer[0]
            for other in buffer[1:]:
                first.blocks.extend(_buffered_blocks(other))
                first.page_end = _max_pos(first.page_end, other.page_end)
            kept.append(first)
    return kept


def _buffered_blocks(section: _Section) -> list[_Block]:
    """A short section's heading line + blocks, ready to fold into a neighbor."""
    heading = _Block(
        "para", section.heading_line, section.heading_page, section.heading_page
    )
    return [heading] + section.blocks


def _min_pos(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a if a[0] <= b[0] else b


def _max_pos(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a if a[0] >= b[0] else b


def _prefix(rfc: int, title: Optional[str], section: _Section) -> str:
    t = title if title else f"RFC {rfc}"
    return f"RFC {rfc} — {t} · §{section.section_id} {section.title}"


def chunk_structural(
    pages: list[Page], rfc: int, title: Optional[str], max_chars: int = MAX,
    min_chars: int = MIN,
) -> list[Chunk]:
    sections = _parse_sections(pages)
    if not sections:
        return []
    sections = _merge_short_sections(sections)

    chunks: list[Chunk] = []
    for section in sections:
        units = _split_section(section, max_chars)
        for idx, (blocks, page_start, page_end) in enumerate(units):
            body_parts: list[str] = []
            if idx == 0:
                body_parts.append(section.heading_line)
            body_parts.extend(block.text for block in blocks)
            text = _prefix(rfc, title, section) + "\n" + "\n".join(body_parts)
            chunks.append(
                Chunk(
                    chunk_id="",
                    method="structural",
                    rfc=rfc,
                    title=title,
                    text=text,
                    section_id=section.section_id,
                    section_path=f"{section.section_id} {section.title}",
                    page_start={"pdf": page_start[0], "printed": page_start[1]},
                    page_end={"pdf": page_end[0], "printed": page_end[1]},
                )
            )

    # Final safety: no chunk under MIN unless unavoidable (single tiny chunk).
    def _end_key(page_end: dict) -> tuple[int, int]:
        return (
            page_end["pdf"] if page_end["pdf"] is not None else -1,
            page_end["printed"] if page_end["printed"] is not None else -1,
        )

    merged: list[Chunk] = []
    for chunk in chunks:
        if merged and len(chunk.text) < min_chars:
            prev = merged[-1]
            body = chunk.text
            first, sep, rest = body.partition("\n")
            if re.match(r"^RFC \d+ — ", first):
                body = rest if rest else body
            prev.text += "\n" + body
            prev.n_chars = len(prev.text)
            if _end_key(chunk.page_end) > _end_key(prev.page_end):
                prev.page_end = dict(chunk.page_end)
            continue
        merged.append(chunk)
    return merged
