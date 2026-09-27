"""Unit tests for the structural section-aware chunker (CAP-4)."""

import re

from chunker.structural import chunk_structural

PREFIX_RE = re.compile(r"^RFC \d+ — .+ · §")


def structural_chunks(mini_cleaned):
    return chunk_structural(mini_cleaned, 4000, "The Test Protocol")


def test_prefix_on_every_chunk(mini_cleaned):
    chunks = structural_chunks(mini_cleaned)
    for chunk in chunks:
        assert PREFIX_RE.match(chunk.text), chunk.text[:60]
        assert chunk.method == "structural"
        assert chunk.section_id and chunk.section_path
        assert chunk.n_chars == len(chunk.text)


def test_sections_detected(mini_cleaned):
    chunks = structural_chunks(mini_cleaned)
    paths = {c.section_path for c in chunks}
    assert "1 Introduction" in paths
    assert "2 The Foo Method" in paths
    assert "2.1 Request Body of Foo" in paths
    assert "3 References" in paths


def test_short_section_merged_into_neighbor(mini_cleaned):
    # 1.1 Scope is under 200 chars: no own chunk, text folded into the next
    # section's chunks.
    chunks = structural_chunks(mini_cleaned)
    paths = {c.section_path for c in chunks}
    assert "1.1 Scope" not in paths
    merged = [c for c in chunks if c.section_path == "2 The Foo Method"]
    assert merged
    body = "\n".join(c.text for c in merged)
    assert "1.1 Scope" in body
    assert "deliberately short" in body


def test_numbered_list_items_demoted(mini_cleaned):
    chunks = structural_chunks(mini_cleaned)
    ids = {c.section_id for c in chunks}
    assert "7" not in ids
    assert "2.99" not in ids
    # The demoted lines survive as body text somewhere.
    body = "\n".join(c.text for c in chunks)
    assert "Bogus list item" in body
    assert "Bogus subsection" in body


def test_quoted_heading_not_accepted(mini_cleaned):
    chunks = structural_chunks(mini_cleaned)
    ids = {c.section_id for c in chunks}
    assert "9" not in ids
    body = "\n".join(c.text for c in chunks)
    assert "9 This quoted example heading" in body


def test_long_section_split_with_overlap(mini_cleaned):
    chunks = [
        c for c in structural_chunks(mini_cleaned)
        if c.section_path == "2 The Foo Method"
    ]
    assert len(chunks) >= 2, "section 2 is long enough to split"
    for prev, nxt in zip(chunks, chunks[1:]):
        prev_body = prev.text.split("\n", 1)[1]
        nxt_body = nxt.text.split("\n", 1)[1]
        overlap_chars = 0
        for k in range(len(prev_body), 0, -1):
            if nxt_body.startswith(prev_body[-k:]):
                overlap_chars = k
                break
        assert overlap_chars > 0, "adjacent chunks must share boundary text"


def test_abnf_rule_never_split(mini_cleaned):
    chunks = structural_chunks(mini_cleaned)
    rule_lines = ["Foo =  ( \"FOO\" / \"f\" ) HCOLON foo-param", '/ "BAR"']
    for chunk in chunks:
        body = chunk.text
        if rule_lines[0] in body:
            assert rule_lines[1] in body, "ABNF rule continuation split off"
            assert "; the foo-param follows the HTTP rules" in body
            break
    else:
        raise AssertionError("ABNF rule not found in any chunk")


def test_table_never_split(mini_cleaned):
    chunks = structural_chunks(mini_cleaned)
    table_lines = ["Header field          where       proxy",
                   "Foo                   R           opt",
                   "Bar                   R           opt"]
    for chunk in chunks:
        if table_lines[0] in chunk.text:
            assert all(ln in chunk.text for ln in table_lines[1:])
            break
    else:
        raise AssertionError("table not found in any chunk")


def test_no_chunk_below_min(mini_cleaned):
    chunks = structural_chunks(mini_cleaned)
    for chunk in chunks:
        assert chunk.n_chars >= 200, (chunk.chunk_id, chunk.n_chars)


def test_page_metadata_sane(mini_cleaned):
    chunks = structural_chunks(mini_cleaned)
    for chunk in chunks:
        ps, pe = chunk.page_start, chunk.page_end
        assert ps["pdf"] is not None and pe["pdf"] is not None
        assert ps["pdf"] <= pe["pdf"]
        assert 1 <= ps["pdf"] < 8 and 1 <= pe["pdf"] < 8


def test_split_units_keep_paragraphs_whole(mini_cleaned):
    """Chunks are cut at whitespace boundaries: every token in a chunk body
    is a complete token of the source text — nothing is truncated mid-word."""
    chunks = structural_chunks(mini_cleaned)
    vocab = {
        token.strip(".,;:()")
        for page in mini_cleaned
        for token in page.text.split()
    }
    for chunk in chunks:
        body = chunk.text.split("\n", 1)[1]  # skip the prefix line
        for token in body.split():
            stripped = token.strip(".,;:()")
            assert not stripped or stripped in vocab, (chunk.section_path, token)


def test_appendix_sections_detected(mini_cleaned):
    chunks = structural_chunks(mini_cleaned)
    ids = {c.section_id for c in chunks}
    assert "A" in ids
    assert "A.1" in ids
    paths = {c.section_path for c in chunks}
    assert "A Test Appendices" in paths
    assert "A.1 Collecting Details" in paths
    for chunk in chunks:
        if chunk.section_id in ("A", "A.1"):
            assert chunk.text.split("\n", 1)[0].endswith(chunk.section_path)
