"""Unit tests for the naive fixed-window chunker (CAP-3)."""

from chunker.naive import chunk_naive
from chunker.schemas import Page

WINDOW = 1000


def stream_of(pages) -> str:
    return "\n".join(p.text for p in pages)


def test_chunks_are_nonempty_and_bounded(mini_cleaned):
    chunks = chunk_naive(mini_cleaned, 4000, "The Test Protocol")
    assert len(chunks) > 3
    for chunk in chunks:
        assert chunk.text.strip()
        assert chunk.n_chars == len(chunk.text)


def test_window_respected(mini_cleaned):
    chunks = chunk_naive(mini_cleaned, 4000, "The Test Protocol", window=WINDOW)
    # Cut happens at whitespace: a chunk stays at or under the window (the
    # only overshoot is a single word longer than the window — never here).
    assert max(c.n_chars for c in chunks) <= WINDOW


def test_never_splits_a_word(mini_cleaned):
    chunks = chunk_naive(mini_cleaned, 4000, "The Test Protocol")
    stream = stream_of(mini_cleaned)
    # Every chunk must be a verbatim substring of the stream AND start and
    # end on word boundaries.
    for chunk in chunks:
        start = stream.find(chunk.text[:40])
        assert start != -1
        end = start + len(chunk.text)
        assert chunk.text in stream
        if start > 0:
            assert stream[start - 1].isspace()
        if end < len(stream):
            assert stream[end].isspace() or stream[end - 1].isspace()


def test_overlap_between_consecutive_chunks(mini_cleaned):
    chunks = chunk_naive(mini_cleaned, 4000, "The Test Protocol", overlap=175)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.text[:100] in prev.text


def test_page_map_spans(mini_cleaned):
    chunks = chunk_naive(mini_cleaned, 4000, "The Test Protocol")
    for chunk in chunks:
        assert chunk.page_start["pdf"] <= chunk.page_end["pdf"]
        ps, pe = chunk.page_start, chunk.page_end
        assert isinstance(ps["pdf"], int) and isinstance(pe["pdf"], int)
        assert ps["printed"] is None or isinstance(ps["printed"], int)
        assert chunk.method == "naive"
        assert chunk.section_id is None and chunk.section_path is None


def test_chunk_on_late_page_reports_late_pdf_index(mini_cleaned):
    chunks = chunk_naive(mini_cleaned, 4000, "The Test Protocol")
    refs_chunks = [
        c for c in chunks if "The Test Protocol Base Specification" in c.text
    ]
    assert refs_chunks
    span = refs_chunks[0]
    assert span.page_start["pdf"] <= 6 <= span.page_end["pdf"]
    assert span.page_start["printed"] is not None


def test_empty_input():
    assert chunk_naive([], 1, None) == []
    assert chunk_naive([Page(text="   ", pdf=0, printed=None)], 1, None) == []


def test_unsplittable_long_word_overshoots():
    # A single word longer than the window forces a whole-word overshoot.
    long_word = "x" * 1200
    page = Page(text=f"short {long_word} tail", pdf=0, printed=1)
    chunks = chunk_naive([page], 1, None, window=1000, overlap=100)
    joined = "\n".join(c.text for c in chunks)
    assert long_word in joined, "the long word must survive intact"
