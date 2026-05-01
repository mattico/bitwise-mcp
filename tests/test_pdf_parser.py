"""Tests for PDFParser helpers that don't require an actual PDF."""

from __future__ import annotations

from mcp_embedded_docs.ingestion.pdf_parser import PDFParser


def test_trim_to_section_bounds_strips_leading_and_trailing():
    page_text = (
        "RM0433 Embedded flash memory (FLASH)\n"
        "4.9 FLASH registers\n"
        "4.9.1 FLASH access control register (FLASH_ACR)\n"
        "Address offset: 0x000\n"
        "Bits 5:4 WRHIGHFREQ: ...\n"
        "4.9.2 FLASH key register for bank 1 (FLASH_KEYR1)\n"
        "Address offset: 0x004\n"
    )
    trimmed = PDFParser._trim_to_section_bounds(
        page_text,
        title="4.9.1 FLASH access control register (FLASH_ACR)",
        next_title="4.9.2 FLASH key register for bank 1 (FLASH_KEYR1)",
    )
    assert trimmed.startswith("4.9.1 FLASH access control register (FLASH_ACR)")
    assert "FLASH_KEYR1" not in trimmed  # next section's title is excluded
    assert "Address offset: 0x000" in trimmed


def test_trim_to_section_bounds_preserves_content_when_titles_missing():
    text = "Free-form prose with no recognizable headings."
    out = PDFParser._trim_to_section_bounds(text, title="Foo", next_title="Bar")
    assert out == text


def test_trim_to_section_bounds_handles_no_next():
    text = (
        "Section heading here\n"
        "All the content of the last section.\n"
    )
    out = PDFParser._trim_to_section_bounds(
        text,
        title="Section heading here",
        next_title=None,
    )
    assert out.startswith("Section heading here")
    assert "All the content" in out


def test_trim_to_section_bounds_keeps_title_when_present_at_offset_zero():
    text = (
        "4.9.1 FLASH access control register (FLASH_ACR)\n"
        "Body.\n"
        "4.9.2 FLASH key register for bank 1 (FLASH_KEYR1)\n"
    )
    out = PDFParser._trim_to_section_bounds(
        text,
        title="4.9.1 FLASH access control register (FLASH_ACR)",
        next_title="4.9.2 FLASH key register for bank 1 (FLASH_KEYR1)",
    )
    assert out.startswith("4.9.1 FLASH access control register (FLASH_ACR)")
    assert "FLASH_KEYR1" not in out
