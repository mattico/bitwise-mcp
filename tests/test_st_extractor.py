"""Tests for the ST register-section extractor.

Fixture text is taken from the STM32H743 reference manual (RM0433):
- FLASH_ACR (section 4.9.1, page 200): two real bitfields plus a Reserved
  span. Exercises offset/reset extraction and Reserved skipping.
- FLASH_KEYR1 (section 4.9.2, page 201): a single 32-bit field (Bits 31:0).
  Exercises the single-field case where the bitfield grid spans the whole
  register.
- A synthetic non-register section title to confirm the parser declines.
"""

from __future__ import annotations

from mcp_embedded_docs.ingestion.st_extractor import (
    extract_bitfields,
    extract_offset,
    extract_register_abbrev,
    extract_reset_value,
    parse_register_section,
    parse_register_summary_table,
)


FLASH_ACR_TITLE = "4.9.1 FLASH access control register (FLASH_ACR)"

FLASH_ACR_BODY = """\
4.9.1 FLASH access control register (FLASH_ACR)
Address offset: 0x000 or 0x100
Reset value: 0x0000 0037
For more details, refer to Section 4.3.8: FLASH read operations and Section 4.3.9: FLASH
program operations.
31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16
Res. Res. Res. Res. Res. Res. Res. Res. Res. Res. Res. Res. Res. Res. Res. Res.
15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
Res. Res. Res. Res. Res. Res. Res. Res. Res. Res. WRHIGHFREQ LATENCY
rw rw rw rw rw rw
Bits 31:6 Reserved, must be kept at reset value.
Bits 5:4 WRHIGHFREQ: Flash signal delay
These bits are used to control the delay between non-volatile memory signals during
programming operations. The application software has to program them to the correct value
depending on the embedded flash memory interface frequency. Please refer to Table 17 for
details.
Note: No check is performed by hardware to verify that the configuration is correct.
Bits 3:0 LATENCY: Read latency
These bits are used to control the number of wait states used during read operations on both
non-volatile memory banks. The application software has to program them to the correct
value depending on the embedded flash memory interface frequency and voltage conditions.
0000: zero wait state used to read a word from non-volatile memory
0001: one wait state used to read a word from non-volatile memory
0010: two wait states used to read a word from non-volatile memory
0111: seven wait states used to read a word from non-volatile memory
Note: No check is performed by hardware to verify that the configuration is correct.
"""

FLASH_KEYR1_TITLE = "4.9.2 FLASH key register for bank 1 (FLASH_KEYR1)"

FLASH_KEYR1_BODY = """\
4.9.2 FLASH key register for bank 1 (FLASH_KEYR1)
Address offset: 0x004
Reset value: 0x0000 0000
FLASH_KEYR1 is a write-only register. The following values must be programmed
consecutively to unlock FLASH_CR1 register:
1. 1st key = 0x4567 0123
2. 2nd key = 0xCDEF 89AB
31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16
KEY1R
w w w w w w w w w w w w w w w w
15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
KEY1R
w w w w w w w w w w w w w w w w
Bits 31:0 KEY1R: Non-volatile memory bank 1 configuration access unlock key
"""


# ---------- title parsing ----------


def test_extract_register_abbrev_handles_typical_title():
    assert extract_register_abbrev(FLASH_ACR_TITLE) == "FLASH_ACR"


def test_extract_register_abbrev_handles_register_for_bank():
    assert extract_register_abbrev(FLASH_KEYR1_TITLE) == "FLASH_KEYR1"


def test_extract_register_abbrev_returns_none_for_non_register_section():
    assert extract_register_abbrev("4.9 FLASH registers") is None


def test_extract_register_abbrev_returns_none_when_no_parens():
    assert extract_register_abbrev("Embedded flash memory (FLASH)") is None


# ---------- offset / reset ----------


def test_extract_offset_handles_or_form():
    assert extract_offset(FLASH_ACR_BODY) == "0x000 or 0x100"


def test_extract_offset_handles_simple_form():
    assert extract_offset(FLASH_KEYR1_BODY) == "0x004"


def test_extract_reset_value_strips_inner_whitespace():
    # Reset value is "0x0000 0037" - the space is part of the canonical form
    # but we still want it normalized to a single space.
    assert extract_reset_value(FLASH_ACR_BODY) == "0x0000 0037"


# ---------- bitfields ----------


def test_extract_bitfields_skips_reserved():
    fields = extract_bitfields(FLASH_ACR_BODY)
    names = [f.name for f in fields]
    assert "Reserved" not in names
    assert names == ["WRHIGHFREQ", "LATENCY"]


def test_extract_bitfields_captures_bit_ranges():
    fields = extract_bitfields(FLASH_ACR_BODY)
    by_name = {f.name: f for f in fields}
    assert by_name["WRHIGHFREQ"].bits == "5:4"
    assert by_name["WRHIGHFREQ"].bit_range == (5, 4)
    assert by_name["LATENCY"].bits == "3:0"
    assert by_name["LATENCY"].bit_range == (3, 0)


def test_extract_bitfields_captures_multiline_descriptions():
    fields = extract_bitfields(FLASH_ACR_BODY)
    by_name = {f.name: f for f in fields}
    desc = by_name["LATENCY"].description
    # First line of description plus follow-on prose.
    assert desc.startswith("Read latency")
    assert "wait states" in desc
    # Continuation should reach the end-of-section enumerated values.
    assert "seven wait states" in desc


def test_extract_bitfields_handles_single_field_register():
    fields = extract_bitfields(FLASH_KEYR1_BODY)
    assert len(fields) == 1
    assert fields[0].name == "KEY1R"
    assert fields[0].bits == "31:0"
    assert fields[0].bit_range == (31, 0)


def test_extract_bitfields_strips_width_suffix_from_field_name():
    # ST often spells multi-bit field names with their width: "DIVM3[5:0]".
    # The bit position lives in "Bits 25:20" already, so the suffix is
    # redundant; strip it from the canonical name.
    body = (
        "Bits 25:20 DIVM3[5:0]: Prescaler for PLL3\n"
        "Set and reset by software to configure the prescaler.\n"
        "Bits 17:12 DIVM2[5:0]: Prescaler for PLL2\n"
        "Same as DIVM3.\n"
    )
    fields = extract_bitfields(body)
    by_name = {f.name: f for f in fields}
    assert "DIVM3" in by_name
    assert "DIVM2" in by_name
    assert by_name["DIVM3"].bit_range == (25, 20)
    assert by_name["DIVM3"].description.startswith("Prescaler for PLL3")


def test_extract_bitfields_tolerates_leading_indentation():
    # PyMuPDF's get_text(sort=True) preserves visual reading order but
    # also indents lines. The regex must not require Bits to be at column 0.
    indented = (
        "            Bits 31:6 Reserved, must be kept at reset value.\n"
        "\n"
        "             Bits 5:4 WRHIGHFREQ: Flash signal delay\n"
        "                   These bits control the delay.\n"
        "\n"
        "             Bits 3:0 LATENCY: Read latency\n"
        "                   These bits set wait states.\n"
    )
    fields = extract_bitfields(indented)
    assert [f.name for f in fields] == ["WRHIGHFREQ", "LATENCY"]


# ---------- top-level parse_register_section ----------


def test_parse_register_section_returns_register_table():
    table = parse_register_section(FLASH_ACR_TITLE, FLASH_ACR_BODY)
    assert table is not None
    assert table.peripheral == "FLASH"
    assert len(table.registers) == 1
    reg = table.registers[0]
    assert reg.name == "FLASH_ACR"
    assert reg.offset == "0x000 or 0x100"
    assert reg.reset_value == "0x0000 0037"
    assert [f.name for f in reg.fields] == ["WRHIGHFREQ", "LATENCY"]


def test_parse_register_section_uses_explicit_peripheral_over_inference():
    table = parse_register_section(FLASH_ACR_TITLE, FLASH_ACR_BODY, peripheral="Embedded Flash Memory")
    assert table is not None
    assert table.peripheral == "Embedded Flash Memory"


def test_parse_register_section_returns_none_without_bitfields():
    title = "4.9.2 FLASH key register for bank 1 (FLASH_KEYR1)"
    body_without_bits = "Address offset: 0x004\nReset value: 0x0000 0000\nNo bit prose here.\n"
    assert parse_register_section(title, body_without_bits) is None


def test_parse_register_section_returns_none_for_non_register_title():
    title = "4.9 FLASH registers"  # peripheral overview, not a register
    assert parse_register_section(title, FLASH_ACR_BODY) is None


# ---------- summary table parsing ----------


def test_parse_register_summary_table_rejects_unrelated_table():
    # Any random data table (not a register summary) should be rejected.
    data = [
        ["Mode", "Description"],
        ["Normal", "Default mode"],
        ["Sleep", "Low power"],
    ]
    assert parse_register_summary_table(data) is None


def test_clean_section_label_strips_section_numbers():
    from mcp_embedded_docs.ingestion.chunker import SemanticChunker

    assert SemanticChunker._clean_section_label("4.9 FLASH registers") == "FLASH registers"
    assert SemanticChunker._clean_section_label("8.7 RCC register description") == "RCC register description"
    assert SemanticChunker._clean_section_label("RCC") == "RCC"  # no leading number, unchanged
    assert SemanticChunker._clean_section_label("3.4.1 RAMECC interrupt enable register") == "RAMECC interrupt enable register"


def test_parse_register_summary_table_parses_st_summary_shape():
    # Mimics a real ST register summary header. The numeric columns are bit
    # positions; pdfplumber sometimes splits two-digit numbers across rows
    # which yields '13', '03', etc. — both forms must be accepted as
    # "this is a bit-position grid header".
    header = ["Offset", "Register name"] + [str(i) for i in range(31, -1, -1)]
    rows = [
        header,
        ["0x000", "RCC_CR"] + [""] * 32,
        ["0x008", "RCC_CFGR"] + [""] * 32,
    ]
    table = parse_register_summary_table(rows, peripheral="RCC")
    assert table is not None
    assert table.peripheral == "RCC"
    names = [r.name for r in table.registers]
    assert names == ["RCC_CR", "RCC_CFGR"]
    offsets = [r.offset for r in table.registers]
    assert offsets == ["0x000", "0x008"]
