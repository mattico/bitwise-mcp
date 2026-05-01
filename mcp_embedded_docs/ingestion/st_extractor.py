"""ST-style register section extractor.

ST reference manuals (e.g. STM32H743 RM0433) describe each register as its
own TOC section laid out as linear text:

    4.9.1 FLASH access control register (FLASH_ACR)
    Address offset: 0x000 or 0x100
    Reset value: 0x0000 0037
    ...prose...
    31 30 29 28 ... 16
    Res. Res. ...
    15 14 ... 0
    ... WRHIGHFREQ LATENCY
    rw rw rw rw rw rw
    Bits 31:6 Reserved, must be kept at reset value.
    Bits 5:4 WRHIGHFREQ: Flash signal delay
    These bits control the delay between non-volatile memory signals...
    Bits 3:0 LATENCY: Read latency
    These bits are used to control the number of wait states...

This module extracts that prose into a Register record with BitField items.
The register summary tables (offset / name / bit columns) are also parsed
into a list of Register stubs so peripherals get a complete index.

Reading the prose sidesteps pdfplumber's main weakness on ST PDFs: long
field names are rendered with rotated glyphs and come back reversed
(``CRCRDERIE1`` -> ``1EIRREDRCRC``). The prose always uses the canonical
form, so we never have to guess a reversal.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .table_detector import TableType
from .table_extractor import BitField, Register, RegisterTable


REGISTER_TITLE_RE = re.compile(
    r"""
    (?P<long> [A-Za-z][\w \-/,]*? \s+ register s? )      # "FLASH access control register"
    (?: \s+ [^()\n]+? )?                                  # optional " for bank 1" etc.
    \s*
    \(
        \s* (?P<abbrev> [A-Z][A-Z0-9_]+ ) \s*            # "(FLASH_ACR)"
    \)
    """,
    re.VERBOSE,
)

ADDRESS_OFFSET_RE = re.compile(
    r"Address\s+offset\s*:\s*(?P<value>[^\n]+)",
    re.IGNORECASE,
)

RESET_VALUE_RE = re.compile(
    r"Reset\s+value\s*:\s*(?P<value>0x[0-9A-Fa-f][0-9A-Fa-f \t]*)",
    re.IGNORECASE,
)

# Lines that start a bitfield paragraph: "Bits 31:24 ..." or "Bit 7 ...".
# PyMuPDF's sort=True text emits indented lines, so allow leading whitespace.
BITS_LINE_RE = re.compile(
    r"""
    ^ [ \t]*
    (?P<bit_word> Bits? )
    [ \t]+
    (?P<bits> \d+ (?: : \d+ )? )
    [ \t]+
    (?P<rest> [^\n]* )
    $
    """,
    re.VERBOSE | re.MULTILINE,
)

NAME_COLON_RE = re.compile(
    # Allow an optional [N:M] or [N] width suffix on the field name, as
    # ST commonly uses for multi-bit fields ("DIVM3[5:0]: Prescaler...").
    # The suffix is dropped from the captured name so find_register('DIVM3')
    # works without forcing callers to know the exact width form.
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:\[\d+(?::\d+)?\])?\s*:\s*(?P<desc>.*)$"
)


def extract_register_abbrev(title: str) -> Optional[str]:
    """Return the parenthesized register abbreviation from a section title.

    Returns None if the title doesn't look like a register section.
    """
    m = REGISTER_TITLE_RE.search(title)
    if not m:
        return None
    return m.group("abbrev")


def extract_offset(content: str) -> Optional[str]:
    m = ADDRESS_OFFSET_RE.search(content)
    if not m:
        return None
    return _normalize_whitespace(m.group("value"))


def extract_reset_value(content: str) -> Optional[str]:
    m = RESET_VALUE_RE.search(content)
    if not m:
        return None
    return _normalize_whitespace(m.group("value"))


def extract_bitfields(content: str) -> List[BitField]:
    """Extract all bitfield paragraphs from a register section's text."""
    matches = list(BITS_LINE_RE.finditer(content))
    if not matches:
        return []

    fields: List[BitField] = []

    for i, m in enumerate(matches):
        bits = m.group("bits")
        rest = m.group("rest").strip()

        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        following = content[body_start:body_end].strip()

        bit_range = _parse_bit_range(bits)
        if bit_range is None:
            continue

        name, description = _split_name_and_description(rest, following)

        # Skip Reserved entries: they aren't useful targets for find_register
        # and would crowd out real fields in search results.
        if _is_reserved(name, rest):
            continue
        if not name:
            continue

        fields.append(
            BitField(
                name=name,
                bits=bits,
                bit_range=bit_range,
                access="",
                description=description,
                reset_value=None,
            )
        )

    return fields


def parse_register_section(
    title: str,
    content: str,
    peripheral: str = "",
) -> Optional[RegisterTable]:
    """Parse an ST register-section block into a RegisterTable.

    Args:
        title: Section title, e.g. "FLASH access control register (FLASH_ACR)".
        content: Section body text as returned by PyMuPDF's get_text().
        peripheral: Optional peripheral name (e.g. parent section title or
            the prefix of the register abbreviation).

    Returns:
        A RegisterTable holding a single Register, or None if the section
        does not look like an ST register description.
    """
    abbrev = extract_register_abbrev(title)
    if not abbrev:
        return None

    fields = extract_bitfields(content)
    if not fields:
        # No "Bits N:M ..." paragraphs found - this isn't an ST register
        # description (could be a list of registers, an overview, etc.).
        return None

    register = Register(
        name=abbrev,
        offset=extract_offset(content),
        reset_value=extract_reset_value(content),
        access="",
        description=title.strip(),
        fields=fields,
    )

    if not peripheral:
        peripheral = _infer_peripheral(abbrev)

    return RegisterTable(
        peripheral=peripheral,
        table_type=TableType.BITFIELD_DEFINITION,
        registers=[register],
        context=title.strip(),
    )


def parse_register_summary_table(
    table_data: List[List[Optional[str]]],
    peripheral: str = "",
) -> Optional[RegisterTable]:
    """Parse an ST register-summary table (offset | name | bit columns).

    These tables enumerate every register of a peripheral with its offset.
    We harvest just (name, offset) pairs - the per-bit cells are noisy
    (rotated text, merged cells) and we already get bitfield detail from
    the per-register prose sections.
    """
    if not table_data or len(table_data) < 2:
        return None

    header = [_norm(c) for c in (table_data[0] or [])]
    if not header:
        return None

    offset_col = _find_col(header, ("offset",))
    name_col = _find_col(header, ("register name", "register", "name"))
    if offset_col is None or name_col is None:
        return None

    if not _looks_like_bit_grid(header, offset_col, name_col):
        return None

    registers: List[Register] = []
    for row in table_data[1:]:
        if not row:
            continue
        offset = _clean(row[offset_col]) if offset_col < len(row) else ""
        name = _clean(row[name_col]) if name_col < len(row) else ""
        if not name or not offset:
            continue
        registers.append(
            Register(
                name=name,
                offset=offset,
                reset_value=None,
                access="",
                description="",
                fields=[],
            )
        )

    if not registers:
        return None

    return RegisterTable(
        peripheral=peripheral or "Unknown",
        table_type=TableType.REGISTER_MAP,
        registers=registers,
        context="",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_name_and_description(first_line: str, following: str) -> Tuple[str, str]:
    """Given the rest of a "Bits N:M ..." line plus any following lines,
    return (field_name, full_description).

    The first line typically reads ``NAME: short description`` or
    ``Reserved, must be kept at reset value.``. Multi-line descriptions
    continue on subsequent lines until the next bitfield marker.
    """
    first_line = first_line.strip()

    m = NAME_COLON_RE.match(first_line)
    if m:
        name = m.group("name")
        first_desc = m.group("desc").strip()
        if following:
            description = (first_desc + "\n" + following).strip() if first_desc else following
        else:
            description = first_desc
        return name, description

    # Reserved or narrative prose. Capture as-is for traceability.
    if following:
        description = (first_line + "\n" + following).strip()
    else:
        description = first_line
    return "", description


def _is_reserved(name: str, raw_first_line: str) -> bool:
    if name.lower() == "reserved":
        return True
    if not name and raw_first_line.lower().lstrip().startswith("reserved"):
        return True
    return False


def _parse_bit_range(bits: str) -> Optional[Tuple[int, int]]:
    if ":" in bits:
        msb_s, lsb_s = bits.split(":", 1)
        try:
            return (int(msb_s), int(lsb_s))
        except ValueError:
            return None
    try:
        b = int(bits)
        return (b, b)
    except ValueError:
        return None


def _infer_peripheral(abbrev: str) -> str:
    """Guess peripheral from a register abbreviation.

    ST register abbreviations are typically PERIPHERAL_NAME (e.g. FLASH_ACR,
    RCC_BDCR). Fall back to the whole name if there's no underscore.
    """
    if "_" in abbrev:
        return abbrev.split("_", 1)[0]
    return abbrev


def _normalize_whitespace(s: str) -> str:
    return " ".join(s.split())


def _norm(cell: Optional[str]) -> str:
    if not cell:
        return ""
    return " ".join(cell.split()).lower()


def _clean(cell: Optional[str]) -> str:
    if not cell:
        return ""
    return " ".join(cell.split())


def _find_col(header: List[str], keywords: Tuple[str, ...]) -> Optional[int]:
    for i, h in enumerate(header):
        for kw in keywords:
            if kw in h:
                return i
    return None


_BIT_HEADER_RE = re.compile(r"^\d{1,2}$")


def _looks_like_bit_grid(header: List[str], offset_col: int, name_col: int) -> bool:
    """True if most columns (other than offset/name) are bit-position numbers.

    pdfplumber sometimes splits a two-digit bit number into two characters
    on different rows; tolerate that by accepting digit-only headers of
    length 1-3 (e.g. '13', '03', '5').
    """
    other = [
        h for i, h in enumerate(header)
        if i not in (offset_col, name_col) and h
    ]
    if len(other) < 4:
        return False
    numeric = sum(1 for h in other if _BIT_HEADER_RE.match(h))
    return numeric >= len(other) * 0.6
