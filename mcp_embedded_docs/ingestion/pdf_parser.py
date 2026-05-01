"""PDF parsing with layout preservation."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re

import fitz  # PyMuPDF


def _find_title(content: str, title: str, start: int = 0) -> Optional[int]:
    """Locate a TOC title in page text, tolerating whitespace differences.

    PyMuPDF often expands the single space between section number and
    title text into multiple spaces or a newline in the rendered page
    output, so a literal substring search misses. Build a regex that
    treats every whitespace run in ``title`` as ``\\s+`` and search.
    """
    if not title:
        return None
    parts = title.split()
    if not parts:
        return None
    pattern = r"\s+".join(re.escape(p) for p in parts)
    m = re.search(pattern, content[start:])
    if m is None:
        return None
    return start + m.start()


@dataclass
class TextBlock:
    """A block of text with position information."""
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    font_size: float
    font_name: str
    page_num: int


@dataclass
class Page:
    """A PDF page with extracted content."""
    page_num: int
    width: float
    height: float
    blocks: List[TextBlock]
    raw_text: str


@dataclass
class TOCEntry:
    """Table of contents entry."""
    level: int
    title: str
    page_num: int


@dataclass
class Section:
    """A document section."""
    title: str
    level: int
    start_page: int
    end_page: int
    content: str
    subsections: List["Section"]


class PDFParser:
    """PDF parser using PyMuPDF."""

    def __init__(self, pdf_path: Path):
        """Initialize parser with PDF path."""
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def close(self):
        """Close the PDF document."""
        if self.doc:
            self.doc.close()

    def extract_text_with_layout(self) -> List[Page]:
        """Extract text preserving layout structure."""
        pages = []

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            blocks = []

            # Extract text blocks with formatting
            text_dict = page.get_text("dict")

            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                blocks.append(TextBlock(
                                    text=text,
                                    bbox=tuple(span.get("bbox", [0, 0, 0, 0])),
                                    font_size=span.get("size", 0),
                                    font_name=span.get("font", ""),
                                    page_num=page_num
                                ))

            # sort=True orders blocks by visual position (top-to-bottom,
            # left-to-right) rather than PDF stream order. Without it, ST
            # reference manuals interleave register headers and bitfield
            # prose because the underlying PDF places them in different
            # block groups - which makes per-section content slicing wrong.
            raw_text = page.get_text(sort=True)

            pages.append(Page(
                page_num=page_num,
                width=page.rect.width,
                height=page.rect.height,
                blocks=blocks,
                raw_text=raw_text
            ))

        return pages

    def extract_toc(self) -> List[TOCEntry]:
        """Extract table of contents from PDF."""
        toc_entries = []
        toc = self.doc.get_toc()

        for entry in toc:
            level, title, page_num = entry
            toc_entries.append(TOCEntry(
                level=level,
                title=title,
                page_num=page_num - 1  # Convert to 0-indexed
            ))

        return toc_entries

    def detect_sections(self, pages: List[Page], toc: List[TOCEntry]) -> List[Section]:
        """Identify section boundaries using font sizes and TOC."""
        sections = []

        # If we have TOC, use it to define sections
        if toc:
            for i, entry in enumerate(toc):
                next_entry = toc[i + 1] if i + 1 < len(toc) else None

                # Concatenate page text from this entry's start to the page
                # before the next entry. When the next entry shares this
                # entry's start page (very common in dense reference manuals
                # like ST RMs), include that shared page so neither section
                # ends up with empty content; we then trim by title below.
                if next_entry is not None:
                    span_end = max(entry.page_num, next_entry.page_num)
                    end_page = max(entry.page_num, next_entry.page_num - 1)
                else:
                    span_end = len(pages) - 1
                    end_page = span_end

                content_pages = []
                for page_num in range(entry.page_num, min(span_end + 1, len(pages))):
                    content_pages.append(pages[page_num].raw_text)
                content = "\n".join(content_pages)

                # Trim leading text before this section's title and trailing
                # text from the next section's title onward. Falls back to
                # the raw concatenation if either title can't be located
                # (e.g. PDF text extraction reflowed it).
                content = self._trim_to_section_bounds(
                    content, entry.title, next_entry.title if next_entry else None
                )

                sections.append(Section(
                    title=entry.title,
                    level=entry.level,
                    start_page=entry.page_num,
                    end_page=end_page,
                    content=content,
                    subsections=[]
                ))
        else:
            # Fallback: detect sections using font size and patterns
            sections = self._detect_sections_heuristic(pages)

        # Build hierarchy
        return self._build_section_hierarchy(sections)

    @staticmethod
    def _trim_to_section_bounds(
        content: str,
        title: str,
        next_title: Optional[str],
    ) -> str:
        """Slice page-concatenated text down to a single section.

        Locates ``title`` near the start and trims everything before it,
        then locates ``next_title`` after that and trims everything from
        it onward. Match is whitespace-tolerant: PyMuPDF often emits a
        TOC title like "4.9.1 FLASH access control register (FLASH_ACR)"
        as "4.9.1    FLASH access control register (FLASH_ACR)" in the
        page text (multiple spaces or a newline between tokens).

        Falls back to the unmodified content if a marker can't be found.
        """
        if not content:
            return content

        if title:
            m = _find_title(content, title, 0)
            if m is not None and m > 0:
                content = content[m:]

        if next_title:
            search_start = len(title) if title else 0
            m = _find_title(content, next_title, search_start)
            if m is not None and m > 0:
                content = content[:m]

        return content

    def _detect_sections_heuristic(self, pages: List[Page]) -> List[Section]:
        """Detect sections using heuristics when no TOC available."""
        sections = []
        current_section = None
        section_pattern = re.compile(r'^(\d+\.)+\d*\s+[A-Z]')  # Match "45.3.2 Title"

        for page in pages:
            for block in page.blocks:
                # Look for large font sizes or section number patterns
                if block.font_size > 12 or section_pattern.match(block.text):
                    # Start new section
                    if current_section:
                        current_section.end_page = page.page_num - 1
                        sections.append(current_section)

                    # Determine level from numbering
                    level = block.text.count('.') + 1 if '.' in block.text else 1

                    current_section = Section(
                        title=block.text,
                        level=level,
                        start_page=page.page_num,
                        end_page=page.page_num,
                        content="",
                        subsections=[]
                    )

            if current_section:
                current_section.content += page.raw_text + "\n"

        # Add last section
        if current_section:
            current_section.end_page = len(pages) - 1
            sections.append(current_section)

        return sections

    def _build_section_hierarchy(self, sections: List[Section]) -> List[Section]:
        """Build hierarchical structure from flat section list."""
        if not sections:
            return []

        root_sections = []
        stack = []

        for section in sections:
            # Pop sections from stack that are not parents
            while stack and stack[-1].level >= section.level:
                stack.pop()

            if stack:
                # Add as subsection to parent
                stack[-1].subsections.append(section)
            else:
                # Add as root section
                root_sections.append(section)

            stack.append(section)

        return root_sections

    def extract_page_range(self, start_page: int, end_page: int) -> List[Page]:
        """Extract a specific range of pages."""
        all_pages = self.extract_text_with_layout()
        return all_pages[start_page:end_page + 1]