"""Tests for TableExtractor's caption-based register-name heuristic.

The ARM Cortex-M7 reference uses table captions like "Table 4. APSR bit
assignments" or "Table 10. Control register bit assignments" to identify
the register a bitfield table describes. _extract_register_name has to
handle both the all-caps abbrev form (APSR) and the prose form (Control).
"""

from __future__ import annotations

from mcp_embedded_docs.ingestion.table_extractor import TableExtractor


def test_extract_register_name_from_arm_caption_all_caps():
    ex = TableExtractor("dummy.pdf")
    assert ex._extract_register_name("Table 4. APSR bit assignments") == "APSR"


def test_extract_register_name_from_arm_caption_with_register_word():
    ex = TableExtractor("dummy.pdf")
    assert ex._extract_register_name(
        "Table 9. BASEPRI register bit assignments"
    ) == "BASEPRI"


def test_extract_register_name_uppercases_prose_caption():
    # "Control" -> "CONTROL" - ARM CMSIS symbol convention is all-caps.
    ex = TableExtractor("dummy.pdf")
    assert ex._extract_register_name(
        "Table 10. Control register bit assignments"
    ) == "CONTROL"


def test_extract_register_name_ignores_summary_table_caption():
    # No "bit assignments" suffix - a summary table like this should not
    # produce a phantom "FAULT" register.
    ex = TableExtractor("dummy.pdf")
    assert ex._extract_register_name(
        "Table 22. Fault status and fault address registers"
    ) != "FAULT"


def test_extract_register_name_falls_back_to_first_uppercase_word():
    # No ARM caption present - fall back to the original heuristic.
    ex = TableExtractor("dummy.pdf")
    assert ex._extract_register_name(
        "Section 4.5 The FLEXCAN module overview"
    ) == "FLEXCAN"


def test_extract_register_name_returns_unknown_for_empty_context():
    ex = TableExtractor("dummy.pdf")
    assert ex._extract_register_name("") == "Unknown"


def test_extract_register_name_from_usb334x_numbered_heading():
    # USB334x: "7.1.1.5  Function Control  Address = 04-06h (read), ..."
    ex = TableExtractor("dummy.pdf")
    ctx = "USB334x 7.1.1.5 Function Control Address = 04-06h (read), 04h (write)"
    assert ex._extract_register_name(ctx) == "Function Control"


def test_extract_register_name_from_ulpi_numbered_heading():
    # ULPI v1.1: "4.2.2 Function Control Address: 04h-06h (Read), ..."
    ex = TableExtractor("dummy.pdf")
    ctx = "4.2.2 Function Control Address: 04h-06h (Read), 04h (Write)"
    assert ex._extract_register_name(ctx) == "Function Control"


def test_extract_register_name_numbered_heading_handles_multi_word_names():
    ex = TableExtractor("dummy.pdf")
    ctx = "7.1.3.4 Vendor Rid Conversion Address = 36-38h (read)"
    assert ex._extract_register_name(ctx) == "Vendor Rid Conversion"


def test_extract_register_name_numbered_heading_doesnt_match_without_address():
    # Without an "Address" anchor we shouldn't pick up arbitrary numbered
    # headings (e.g. a table of contents fragment in the context).
    ex = TableExtractor("dummy.pdf")
    ctx = "1.2 List of abbreviations for registers SomethingElse"
    # Falls through to the generic uppercase-word fallback - returns
    # "SomethingElse" if it qualifies; here neither word does, so Unknown.
    assert ex._extract_register_name(ctx) == "Unknown"
