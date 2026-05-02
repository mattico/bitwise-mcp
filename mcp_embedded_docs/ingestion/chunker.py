"""Semantic chunking for documents."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from .pdf_parser import Section
from .st_extractor import parse_register_section
from .table_extractor import RegisterTable

# Sentence boundary pattern: period/question/exclamation followed by space or newline,
# or a double newline (paragraph break)
_SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+|\n\n+')

# Lines from a table-of-contents page have a leader of dots/spaces between
# the title and the page number. PyMuPDF preserves these as ". . . . ."
# (with intervening spaces) or as long runs of consecutive dots.
_TOC_LEADER_RE = re.compile(r'\.\s\.\s\.|\.{4,}')

# Section title for a free-standing data table (e.g. "Table 226. Bootloader
# device-dependent parameters"). Used to opt into a special chunking path
# that prepends column-header context so chunks split off the middle of a
# many-page table aren't a stream of headerless rows.
_TABLE_SECTION_TITLE_RE = re.compile(r'^\s*Table\s+\d+\.', re.IGNORECASE)


def _is_toc_chunk(text: str, threshold: float = 0.4) -> bool:
    """True when most non-blank lines in ``text`` look like TOC entries.

    A TOC line in PyMuPDF output is something like
    ``33.1   Bootloader configuration  . . . . . . . . .  158``: it ends
    with a leader of ". . ." or many consecutive dots. When that pattern
    dominates a chunk it's just TOC pollution -- e.g. AN2606's first TOC
    entry is "Table 1. Applicable products" at page 2 with the next entry
    at page 27, so the chunker sees 25 pages of TOC under that title.
    """
    body = text.split("]\n", 1)[-1]  # strip the [Doc > ...] hierarchy line
    lines = [line for line in body.splitlines() if line.strip()]
    if len(lines) < 4:
        return False
    toc_lines = sum(1 for line in lines if _TOC_LEADER_RE.search(line))
    return toc_lines / len(lines) >= threshold


@dataclass
class Chunk:
    """A chunk of document content."""
    id: str
    doc_id: str
    chunk_type: str  # "text", "register_definition", "memory_map"
    text: str
    structured_data: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    page_start: int
    page_end: int


class SemanticChunker:
    """Create semantic chunks from parsed documents."""

    def __init__(
        self,
        target_size: int = 2500,
        overlap: int = 200,
        preserve_tables: bool = True,
        pdf_path: Optional[Path] = None,
    ):
        """Initialize chunker.

        Args:
            target_size: Target chunk size in characters
            overlap: Overlap between adjacent text chunks (in characters, used as budget for trailing sentences)
            preserve_tables: Keep register tables intact (never split)
            pdf_path: Source PDF path. When provided, "Table N." sections
                use pdfplumber to recover the column-header row so the
                header gets prepended to every body chunk -- without it
                a search hit landing in the middle of a 30-page table
                gives only opaque rows.
        """
        self.target_size = target_size
        self.overlap = overlap
        self.preserve_tables = preserve_tables
        self.pdf_path = pdf_path

    def chunk_document(
        self,
        doc_id: str,
        sections: List[Section],
        tables: List[RegisterTable],
        doc_title: str = "",
        table_pages: Optional[Dict[int, int]] = None,
    ) -> List[Chunk]:
        """Create chunks from document sections and tables.

        Args:
            doc_id: Document identifier
            sections: Document sections
            tables: Extracted register tables
            doc_title: Document title for contextual prefixes
            table_pages: Mapping of table index -> page_num from detection phase

        Returns:
            List of chunks
        """
        chunks = []

        # First, create chunks for register tables (these are always kept intact)
        table_chunks = self._chunk_tables(doc_id, tables, doc_title, table_pages or {})
        chunks.extend(table_chunks)

        # Then, create chunks for text sections (starting with no ancestors)
        text_chunks = self._chunk_sections(doc_id, sections, doc_title, ancestors=[])
        chunks.extend(text_chunks)

        return chunks

    # ------------------------------------------------------------------
    # Contextual prefix
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context_prefix(doc_title: str, section_hierarchy: List[str]) -> str:
        """Build a contextual prefix like [Doc > Section > Subsection].

        Args:
            doc_title: Overall document title
            section_hierarchy: List of ancestor section titles, outermost first

        Returns:
            Prefix string (empty if no context available)
        """
        parts = []
        if doc_title:
            parts.append(doc_title)
        parts.extend(h for h in section_hierarchy if h)
        if not parts:
            return ""
        return "[" + " > ".join(parts) + "]\n"

    # ------------------------------------------------------------------
    # Table chunking
    # ------------------------------------------------------------------

    def _chunk_tables(
        self,
        doc_id: str,
        tables: List[RegisterTable],
        doc_title: str,
        table_pages: Dict[int, int],
    ) -> List[Chunk]:
        """Create chunks for register tables."""
        chunks = []

        for i, table in enumerate(tables):
            # Build hierarchy from peripheral + context
            hierarchy: List[str] = []
            if table.peripheral:
                hierarchy.append(table.peripheral)
            if table.context:
                hierarchy.append(table.context)

            prefix = self._build_context_prefix(doc_title, hierarchy)

            # Convert table to both text and structured format
            text = prefix + self._format_table_as_text(table)
            structured = self._format_table_as_json(table)

            # Content-based chunk ID
            chunk_hash = hashlib.md5(text.encode()).hexdigest()[:12]
            chunk_id = f"{doc_id}_{chunk_hash}"

            page_num = table_pages.get(i, 0)

            chunk = Chunk(
                id=chunk_id,
                doc_id=doc_id,
                chunk_type=table.table_type.value,
                text=text,
                structured_data=structured,
                metadata={
                    "peripheral": table.peripheral,
                    "table_type": table.table_type.value,
                    "context": table.context,
                    "register_names": [r.name for r in table.registers],
                },
                page_start=page_num,
                page_end=page_num,
            )

            chunks.append(chunk)

        return chunks

    # ------------------------------------------------------------------
    # Section chunking
    # ------------------------------------------------------------------

    def _chunk_sections(
        self,
        doc_id: str,
        sections: List[Section],
        doc_title: str,
        ancestors: List[str],
    ) -> List[Chunk]:
        """Create chunks for text sections.

        Only leaf sections (those with no subsections) get their content
        chunked, avoiding duplicate text between parent and child sections.
        """
        chunks = []

        for section in sections:
            current_hierarchy = ancestors + [section.title]
            prefix = self._build_context_prefix(doc_title, current_hierarchy)

            # Try the ST register parser first, working on the section's
            # content concatenated with any "Table N. ..." subsection
            # content. ST RMs nest a "Table N. NAME address offset and
            # reset value" entry under most register sections; the page-
            # range slicing in pdf_parser puts almost all the bitfield
            # prose under that subsection's content, leaving the parent
            # nearly empty. Concatenating recovers the prose so registers
            # like RCC_AHB4ENR get a real bitfield_definition chunk
            # instead of being lost as a generic text chunk.
            merged_content = self._merge_table_subsection_content(section)
            if merged_content.strip():
                merged = Section(
                    title=section.title,
                    level=section.level,
                    start_page=section.start_page,
                    end_page=section.end_page,
                    content=merged_content,
                    subsections=[],
                )
                st_chunk = self._try_st_register_chunk(
                    doc_id, merged, ancestors, prefix
                )
                if st_chunk is not None:
                    chunks.append(st_chunk)
                    continue

            if section.subsections:
                # Non-leaf, non-register: recurse into subsections only --
                # skip own content to avoid duplicating text already covered
                # by children.
                sub_chunks = self._chunk_sections(
                    doc_id, section.subsections, doc_title, current_hierarchy
                )
                chunks.extend(sub_chunks)
            else:
                content = section.content.strip()
                if not content:
                    continue

                # For "Table N." sections, prepend column-header context to
                # the prefix so each split chunk carries the header. Falls
                # back to bare prefix if pdf_path isn't set or pdfplumber
                # can't find a sensible table on the section's pages.
                effective_prefix = prefix
                if _TABLE_SECTION_TITLE_RE.match(section.title):
                    header = self._extract_table_header(section)
                    if header:
                        effective_prefix = prefix + header + "\n"

                if len(effective_prefix) + len(content) <= self.target_size:
                    chunk_text = effective_prefix + content
                    if _is_toc_chunk(chunk_text):
                        continue
                    chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:12]

                    chunk = Chunk(
                        id=f"{doc_id}_{chunk_hash}",
                        doc_id=doc_id,
                        chunk_type="text",
                        text=chunk_text,
                        structured_data=None,
                        metadata={
                            "section_title": section.title,
                            "section_level": section.level,
                        },
                        page_start=section.start_page,
                        page_end=section.end_page,
                    )
                    chunks.append(chunk)
                else:
                    # Large section — split with sentence-aware boundaries
                    split_chunks = self._split_section(
                        doc_id, section, effective_prefix
                    )
                    # Drop any split chunks dominated by TOC dot-leader
                    # patterns. The bootloader doc, for instance, has the
                    # very first TOC entry "Table 1. Applicable products"
                    # at page 2 with the next entry at page 27 -- so the
                    # section's content slice is 25 pages of TOC plus the
                    # actual table. The TOC chunks contain only entries
                    # like "33.1 Bootloader configuration ........ 158".
                    chunks.extend(c for c in split_chunks if not _is_toc_chunk(c.text))

        return chunks

    # ------------------------------------------------------------------
    # ST-style register section
    # ------------------------------------------------------------------

    _SECTION_NUMBER_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+")

    @classmethod
    def _clean_section_label(cls, title: str) -> str:
        """Strip leading section numbers (e.g. '4.9 FLASH registers' -> 'FLASH registers')."""
        return cls._SECTION_NUMBER_RE.sub("", title).strip()

    def _extract_table_header(self, section: Section) -> Optional[str]:
        """Pull the column-header row of a "Table N." section via pdfplumber.

        Returns a one-line summary like ``Columns: A | B | C`` or None if
        no PDF path is configured, the page can't be parsed, or no table
        is found. Errors are swallowed -- a missing header just means the
        body chunks ship without the column hint, which is no worse than
        the prior behavior.
        """
        if self.pdf_path is None:
            return None
        try:
            import pdfplumber
        except ImportError:
            return None

        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                # Probe the first 1-2 pages of the section. A multi-page
                # table's header lives on the first page; later pages
                # often repeat it but pdfplumber can mis-detect there.
                last = min(section.start_page + 1, len(pdf.pages) - 1)
                for page_idx in range(section.start_page, last + 1):
                    page = pdf.pages[page_idx]
                    tables = page.extract_tables()
                    if not tables:
                        continue
                    # Pick the table with the most columns -- that's the
                    # one that most resembles a labeled data table rather
                    # than an inline two-column note.
                    best = max(tables, key=lambda t: len(t[0]) if t and t[0] else 0)
                    if not best or not best[0]:
                        continue
                    header_cells = [
                        " ".join(str(c).split()) for c in best[0] if c and str(c).strip()
                    ]
                    if len(header_cells) < 2:
                        continue
                    return "Columns: " + " | ".join(header_cells)
        except Exception:
            return None
        return None

    @staticmethod
    def _merge_table_subsection_content(section: Section) -> str:
        """Return ``section.content`` plus any "Table N." subsection content.

        ST RM TOC layout puts each register's bitfield prose under a
        "Table N. NAME address offset and reset value" sub-entry, leaving
        the parent register section's content slice nearly empty. The
        register parser needs the prose, so we concatenate.

        Other (non-table) subsections are left alone -- merging genuine
        peripheral chapters would create huge chunks and double-count text.
        """
        parts = [section.content or ""]
        for sub in section.subsections:
            if sub.title.lstrip().startswith("Table "):
                parts.append(sub.content or "")
        return "\n".join(p for p in parts if p)

    def _try_st_register_chunk(
        self,
        doc_id: str,
        section: Section,
        ancestors: List[str],
        prefix: str,
    ) -> Optional[Chunk]:
        """Emit a register_definition chunk for an ST-style register section.

        Returns None if the section doesn't look like an ST register
        description (no parenthesized abbrev in the title, or no
        "Bits N:M ..." prose).
        """
        peripheral = self._clean_section_label(ancestors[-1]) if ancestors else ""
        table = parse_register_section(section.title, section.content, peripheral)
        if table is None:
            return None

        text = prefix + self._format_table_as_text(table)
        structured = self._format_table_as_json(table)

        chunk_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        return Chunk(
            id=f"{doc_id}_{chunk_hash}",
            doc_id=doc_id,
            chunk_type=table.table_type.value,
            text=text,
            structured_data=structured,
            metadata={
                "section_title": section.title,
                "section_level": section.level,
                "peripheral": table.peripheral,
                "register_names": [r.name for r in table.registers],
            },
            page_start=section.start_page,
            page_end=section.end_page,
        )

    # ------------------------------------------------------------------
    # Sentence-aware splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _find_sentence_boundaries(text: str) -> List[int]:
        """Return a sorted list of positions right *after* sentence boundaries.

        Each value is the index of the first character of the next sentence.
        """
        boundaries = []
        for m in _SENTENCE_BOUNDARY_RE.finditer(text):
            boundaries.append(m.end())
        return boundaries

    def _split_section(self, doc_id: str, section: Section, prefix: str) -> List[Chunk]:
        """Split a large section into multiple chunks using sentence boundaries."""
        chunks: List[Chunk] = []
        content = section.content.strip()
        boundaries = self._find_sentence_boundaries(content)

        # If no sentence boundaries found, fall back to splitting on double-newlines
        if not boundaries:
            boundaries = [m.end() for m in re.finditer(r'\n\n+', content)]

        start = 0
        budget = self.target_size - len(prefix)  # room for actual text per chunk

        while start < len(content):
            end = start + budget

            if end >= len(content):
                # Remaining text fits
                end = len(content)
            else:
                # Find the last sentence boundary before `end`
                best = None
                for b in boundaries:
                    if b <= start:
                        continue
                    if b <= end:
                        best = b
                    else:
                        break
                if best is not None:
                    end = best
                # else: no boundary found — take the full budget (hard cut at word boundary)
                else:
                    # Try to avoid cutting mid-word: back up to last space
                    space_pos = content.rfind(' ', start, end)
                    if space_pos > start:
                        end = space_pos + 1

            chunk_text = prefix + content[start:end].strip()
            if chunk_text.strip():
                chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
                chunks.append(Chunk(
                    id=f"{doc_id}_{chunk_hash}",
                    doc_id=doc_id,
                    chunk_type="text",
                    text=chunk_text,
                    structured_data=None,
                    metadata={
                        "section_title": section.title,
                        "section_level": section.level,
                    },
                    page_start=section.start_page,
                    page_end=section.end_page,
                ))

            # Compute overlap: grab the last 1-2 sentences from the chunk we just created
            overlap_start = self._compute_overlap_start(content, start, end)
            start = overlap_start if overlap_start < end else end

        return chunks

    def _compute_overlap_start(self, content: str, chunk_start: int, chunk_end: int) -> int:
        """Find where the next chunk should start so that it overlaps the last 1-2 sentences."""
        # Look for sentence boundaries in the region [chunk_end - overlap .. chunk_end]
        region_start = max(chunk_start, chunk_end - self.overlap)
        boundaries_in_region = []
        for m in _SENTENCE_BOUNDARY_RE.finditer(content, region_start, chunk_end):
            boundaries_in_region.append(m.end())

        if boundaries_in_region:
            # Start the next chunk from the beginning of the last sentence in the overlap zone
            # Use the second-to-last boundary if available (gives ~1-2 sentences of overlap)
            if len(boundaries_in_region) >= 2:
                return boundaries_in_region[-2]
            return boundaries_in_region[-1]

        # No sentence boundaries in overlap region — just advance to chunk_end (no overlap)
        return chunk_end

    # ------------------------------------------------------------------
    # Table formatting (unchanged)
    # ------------------------------------------------------------------

    def _format_table_as_text(self, table: RegisterTable) -> str:
        """Format register table as compact text."""
        lines = []

        lines.append(f"# {table.peripheral} - {table.table_type.value.replace('_', ' ').title()}")
        lines.append("")

        if table.context:
            lines.append(f"Context: {table.context}")
            lines.append("")

        for register in table.registers:
            lines.append(f"## {register.name}")

            details = []
            if register.address:
                details.append(f"Address: {register.address}")
            if register.offset:
                details.append(f"Offset: {register.offset}")
            details.append(f"Width: {register.width}-bit")
            if register.reset_value:
                details.append(f"Reset: {register.reset_value}")
            details.append(f"Access: {register.access}")

            lines.append(" | ".join(details))

            if register.description:
                lines.append(f"Description: {register.description}")

            if register.fields:
                lines.append("\nFields:")
                for field in register.fields:
                    lines.append(f"  - {field.name} [{field.bits}]: {field.description} ({field.access})")

            lines.append("")

        return "\n".join(lines)

    def _format_table_as_json(self, table: RegisterTable) -> Dict[str, Any]:
        """Format register table as structured JSON."""
        return {
            "peripheral": table.peripheral,
            "table_type": table.table_type.value,
            "context": table.context,
            "registers": [
                {
                    "name": reg.name,
                    "address": reg.address,
                    "offset": reg.offset,
                    "width": reg.width,
                    "reset_value": reg.reset_value,
                    "access": reg.access,
                    "description": reg.description,
                    "fields": [
                        {
                            "name": field.name,
                            "bits": field.bits,
                            "bit_range": field.bit_range,
                            "access": field.access,
                            "description": field.description,
                            "reset_value": field.reset_value
                        }
                        for field in reg.fields
                    ]
                }
                for reg in table.registers
            ]
        }
