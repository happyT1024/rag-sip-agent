"""Unit tests for header/footer stripping (CAP-1), no PDFs needed."""

from chunker.extract import (
    clean_page_lines,
    parse_title,
    rfc_number_from_path,
)


def lines_of(page_text: str):
    kept, printed = clean_page_lines(page_text.splitlines())
    return kept, printed


def test_footer_single_line_captures_printed():
    kept, printed = lines_of(
        "Some body text\n"
        "Bernstein                Standards Track                     [Page 12]\n"
        "More body\n"
    )
    assert printed == 12
    assert all("Standards Track" not in ln for ln in kept)


def test_footer_wrapped_captures_printed():
    kept, printed = lines_of(
        "Body line\n"
        "Bernstein                Standards Track                     [Page\n"
        "3]\n"
        "Another body line\n"
    )
    assert printed == 3
    assert not any("[Page" in ln for ln in kept)
    assert not any(ln.strip() == "3]" for ln in kept)


def test_footer_author_agnostic():
    for author in ("Rosenberg, et. al.", "Sparks", "Donovan", "Campbell, et. al."):
        kept, printed = lines_of(f"{author}     Standards Track     [Page 7]")
        assert printed == 7
        assert kept == []


def test_header_with_year_stripped():
    kept, printed = lines_of(
        "RFC 4000                    The Test Protocol                June 2002\n"
        "Body paragraph\n"
    )
    assert kept == ["Body paragraph"]
    assert printed is None


def test_header_wrapped_year_stripped():
    # Header mid-line-list with the year on its own line (3311/3515 style).
    kept, printed = lines_of(
        "Body before\n"
        "RFC 4000                    The Test Protocol                June\n"
        "2002\n"
        "Body after\n"
    )
    assert kept == ["Body before", "Body after"]
    assert printed is None


def test_body_rfc_mentions_kept():
    text = (
        "RFC 2543-compliant systems.\n"
        "   protocols defined in RFC 3261 [1], but generalized\n"
        "An RFC 1123 date is case-sensitive.\n"
    )
    kept, printed = lines_of(text)
    assert kept == [ln.strip() for ln in text.splitlines()]
    assert printed is None


def test_cover_category_line_kept():
    # "Category: Standards Track" has no [Page] part: it is not a footer.
    kept, printed = lines_of(
        "Request for Comments: 4000\n"
        "Category: Standards Track                                 June 2002\n"
    )
    assert len(kept) == 2
    assert printed is None


def test_footer_mid_page_stripped():
    # MuPDF reading order can put the footer before body lines (3311).
    kept, printed = lines_of(
        "Bernstein                Standards Track                     [Page\n"
        "5]\n"
        "Body comes after the shuffled footer\n"
    )
    assert printed == 5
    assert kept == ["Body comes after the shuffled footer"]


def test_rfc_number_from_path():
    assert rfc_number_from_path("/x/data/RFC 3261.pdf") == 3261
    assert rfc_number_from_path("RFC 2976.pdf") == 2976


def test_rfc_number_from_path_rejects_bad_filename():
    import pytest

    with pytest.raises(ValueError):
        rfc_number_from_path("draft-ietf-sip-update-05.pdf")


def test_parse_title_reads_raw_headers(tmp_path):
    # parse_title must look at RAW lines: extract strips header lines before
    # it would ever see them.
    pymupdf = __import__("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "RFC 4000      The Test Protocol      June 2002")
    page.insert_text((72, 92), "Body line")
    pdf_path = tmp_path / "RFC 4000.pdf"
    doc.save(str(pdf_path))
    doc.close()
    assert parse_title(pdf_path) == "The Test Protocol"
