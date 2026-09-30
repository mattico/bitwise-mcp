"""Search result formatting."""

from mcp_embedded_docs.retrieval import SearchResponse, SearchResult
from mcp_embedded_docs.retrieval.formatter import (
    ResultFormatter,
    clean_text,
    pages_label,
    split_prefix,
)


def test_pages_are_shown_one_based():
    assert pages_label(0, 0) == "p. 1"
    assert pages_label(1895, 1897) == "pp. 1896-1898"
    assert pages_label(-1, -1) == ""
    assert pages_label(None, None) == ""


def test_clean_text_drops_page_furniture_and_padding():
    raw = ("45.3.7   Debug mode\n\n\n\n     When the   device\n"
           "            RM0433 Rev 8                    1897/3353  \n"
           "302/357 DS12110 Rev 10\nkeep 3/4 of this line")
    assert clean_text(raw) == "45.3.7  Debug mode\n\nWhen the  device\n\nkeep 3/4 of this line"


def test_split_prefix():
    assert split_prefix("[Doc > A > B]\nbody") == (["Doc", "A", "B"], "body")
    assert split_prefix("no prefix") == ([], "no prefix")


def test_format_response_shows_ids_pages_and_snippets():
    result = SearchResult(
        chunk_id="d1_abc", score=1.0,
        text="[My Manual > 4 Clocks > 4.2 HSE]\nThe HSE bypass mode lets an external clock drive OSC_IN.",
        structured_data=None, metadata={}, doc_id="d1", page_start=41, page_end=41,
        section="4.2 HSE", doc_title="My Manual", more_in_section=2, channels=["keyword"],
    )
    out = ResultFormatter.format_response(SearchResponse(
        query="HSE bypass", results=[result], terms=["hse", "bypass"],
        keyword_mode="strict", semantic="on"))
    assert "## 1. 4 Clocks > 4.2 HSE" in out
    assert "p. 42" in out and "chunk `d1_abc`" in out and "doc `d1`" in out
    assert "**HSE** **bypass**" in out
    assert "+2 more matching chunks in this section" in out


def test_format_response_empty():
    out = ResultFormatter.format_response(SearchResponse(
        query="x", results=[], keyword_mode="no match", semantic="on"))
    assert "No results found" in out
