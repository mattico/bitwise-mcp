"""Chunker tests focused on the ST-register-section + nested-TOC interaction.

ST reference manuals frequently put a "Table N. NAME address offset and
reset value" sub-entry under each register section. Without special
handling the chunker would treat the register section as non-leaf, recurse
into the table-only leaf, and drop the bitfield prose -- so
find_register('RCC_AHB4ENR') would return nothing even though the prose
exists in the source PDF.
"""

from __future__ import annotations

from mcp_embedded_docs.ingestion.chunker import SemanticChunker
from mcp_embedded_docs.ingestion.pdf_parser import Section


REGISTER_BODY = (
    "8.7.43 RCC AHB4 clock register (RCC_AHB4ENR)\n"
    "Address offset: 0x0E0\n"
    "Reset value: 0x0000 0000\n"
    "\n"
    "Bits 31:25 Reserved, must be kept at reset value.\n"
    "Bit 24 ADC3EN: ADC3 Peripheral Clocks Enable\n"
    "Set and reset by software.\n"
    "0: ADC3 peripheral clock disabled\n"
    "1: ADC3 peripheral clock enabled\n"
    "Bit 7 GPIOHEN: GPIOH peripheral clock enable\n"
    "Set and reset by software.\n"
)


def _make_section_with_table_subsection() -> Section:
    """Mimic the ST RM TOC layout: register section with a Table sub-entry."""
    table_sub = Section(
        title="Table 69. RCC_AHB4ENR address offset and reset value",
        level=4,
        start_page=457,
        end_page=457,
        content="",  # the table-only leaf has no useful content
        subsections=[],
    )
    return Section(
        title="8.7.43 RCC AHB4 clock register (RCC_AHB4ENR)",
        level=3,
        start_page=457,
        end_page=457,
        content=REGISTER_BODY,
        subsections=[table_sub],
    )


def _make_section_with_table_subsection_holding_body() -> Section:
    """The shape we actually see in H743 RM: page-range slicing puts the
    bitfield prose under the "Table N." subsection's content, not the
    register section's. The chunker has to reach into the subsection."""
    table_sub = Section(
        title="Table 69. RCC_AHB4ENR address offset and reset value",
        level=4,
        start_page=457,
        end_page=457,
        content=REGISTER_BODY,  # body lives here, not in the parent
        subsections=[],
    )
    return Section(
        title="8.7.43 RCC AHB4 clock register (RCC_AHB4ENR)",
        level=3,
        start_page=457,
        end_page=457,
        content="8.7.43 RCC AHB4 clock register (RCC_AHB4ENR)\nAddress offset: 0x0E0\nReset value: 0x0000 0000\n",
        subsections=[table_sub],
    )


def test_chunker_emits_register_chunk_even_when_section_has_subsection():
    chunker = SemanticChunker()
    section = _make_section_with_table_subsection()

    chunks = chunker.chunk_document(
        doc_id="test_doc",
        sections=[section],
        tables=[],
        doc_title="STM32H743 Reference Manual",
    )

    register_chunks = [c for c in chunks if c.chunk_type == "bitfield_definition"]
    assert len(register_chunks) == 1, (
        "expected exactly one bitfield_definition chunk for the register; "
        f"got chunk_types={[c.chunk_type for c in chunks]}"
    )
    rc = register_chunks[0]
    assert rc.structured_data is not None
    reg = rc.structured_data["registers"][0]
    assert reg["name"] == "RCC_AHB4ENR"
    field_names = [f["name"] for f in reg["fields"]]
    assert "ADC3EN" in field_names
    assert "GPIOHEN" in field_names


def test_chunker_merges_table_subsection_prose_when_parent_is_register():
    """Real H743 layout: the bitfield prose ends up in the "Table N."
    subsection's content slice, not the register section's. The chunker
    must merge them so the register still gets a bitfield_definition
    chunk -- otherwise find_register('RCC_AHB4ENR') breaks."""
    chunker = SemanticChunker()
    section = _make_section_with_table_subsection_holding_body()

    chunks = chunker.chunk_document(
        doc_id="test_doc",
        sections=[section],
        tables=[],
        doc_title="STM32H743 Reference Manual",
    )

    register_chunks = [c for c in chunks if c.chunk_type == "bitfield_definition"]
    assert len(register_chunks) == 1
    field_names = [f["name"] for f in register_chunks[0].structured_data["registers"][0]["fields"]]
    assert "ADC3EN" in field_names
    assert "GPIOHEN" in field_names


def test_chunker_does_not_also_emit_text_chunk_for_register_subsection():
    """When the register section is captured, the table sub-entry below it
    should not also produce a near-empty text chunk."""
    chunker = SemanticChunker()
    section = _make_section_with_table_subsection()

    chunks = chunker.chunk_document(
        doc_id="test_doc",
        sections=[section],
        tables=[],
        doc_title="STM32H743 Reference Manual",
    )

    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    assert text_chunks == [], (
        "register-as-non-leaf should suppress recursion into the Table N "
        f"subsection, but got text chunks: {[c.metadata.get('section_title') for c in text_chunks]}"
    )
