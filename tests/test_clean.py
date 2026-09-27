"""Unit tests for the cleaning state machine (CAP-2)."""

from chunker.clean import clean_pages
from chunker.schemas import Page


def joined(cleaned) -> str:
    return "\n".join(p.text for p in cleaned)


def test_front_matter_and_toc_dropped(mini_cleaned):
    text = joined(mini_cleaned)
    assert "Table of Contents" not in text
    assert "Status of this Memo" not in text
    assert "Abstract" not in text
    assert "Mini RFC 4000" not in text
    # No TOC leader lines of either style survive.
    assert "......." not in text
    assert ". . . ." not in text


def test_body_starts_at_first_numbered_heading(mini_cleaned):
    text = joined(mini_cleaned)
    assert "1 Introduction" in text
    body_pos = text.find("1 Introduction")
    intro_pos = text.find("The Test Protocol carries fictional")
    assert 0 <= body_pos < intro_pos


def test_toc_wrapped_pagenum_does_not_start_body(mini_pages):
    # The TOC's "1 Introduction" entry has its leader dots wrapped onto the
    # next line: the body-start anchor must skip it and wait for the real
    # heading on the body page.
    cleaned = clean_pages(mini_pages)
    text = joined(cleaned)
    assert text.count("1 Introduction") == 1


def test_early_fullcopyright_mention_does_not_cut(mini_cleaned):
    # p3 carries a heading-shaped "Full Copyright Statement" quote; only the
    # last quarter of pages may trigger the cut, so later pages survive.
    text = joined(mini_cleaned)
    assert "3 References" in text
    assert "Design Notes for Lightweight Signaling" in text


def test_tail_cut_removes_service_parts(mini_cleaned):
    text = joined(mini_cleaned)
    assert "4 Acknowledgments" not in text
    assert "Authors' Addresses" not in text
    assert "All Rights Reserved" not in text


def test_references_kept(mini_cleaned):
    text = joined(mini_cleaned)
    assert "3 References" in text
    assert "The Test Protocol Base Specification" in text


def test_page_map_preserved(mini_cleaned):
    # Body pages keep their pdf indices and printed numbers from the footer.
    pdfs = [p.pdf for p in mini_cleaned]
    assert pdfs == sorted(pdfs)
    printed = [p.printed for p in mini_cleaned]
    assert all(v is not None for v in printed)
    assert printed == sorted(printed)


def test_no_residue_in_cleaned(mini_cleaned):
    text = joined(mini_cleaned)
    assert "Standards Track" not in text
    assert "[Page" not in text
    assert "The Test Protocol                June" not in text


def test_clean_pages_returns_empty_without_body_start():
    pages = [
        Page(
            text="Cover page\nAbstract and boilerplate, but no numbered section.",
            pdf=0,
            printed=None,
        )
    ]
    assert clean_pages(pages) == []


def test_dehyphenation_joins_wrapped_words(mini_pages):
    from chunker.schemas import Page as P
    from chunker.clean import clean_pages as clean

    hyphen_page = P(
        text="1 Introduction\n\n   The protocol updates RFC 2543-\ncompliant "
             "systems with a case-\ninsensitive comparison rule and it "
             "reuses the Record-\nRoute logic of the base specification.",
        pdf=0,
        printed=1,
    )
    cleaned = clean([hyphen_page])
    text = "\n".join(p.text for p in cleaned)
    assert "RFC 2543-compliant" in text
    assert "case-insensitive" in text
    assert "Record-Route" in text
    assert "2543- compliant" not in text
    assert "case- insensitive" not in text
    assert "Record- Route" not in text
