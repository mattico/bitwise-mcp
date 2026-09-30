"""Format search results compactly."""

import re
from typing import Any, Dict, List, Optional, Tuple

from . import SearchResponse, SearchResult
from .query import matched_terms, snippets

# Field names listed per register in a search hit; find_register shows them all.
_MAX_FIELD_NAMES = 16


def split_prefix(text: str) -> Tuple[List[str], str]:
    """Split a chunk into its '[Doc > Section > ...]' hierarchy and its body."""
    if text.startswith("["):
        end = text.find("]\n")
        if end == -1 and text.endswith("]"):
            end = len(text) - 1
        if end != -1:
            return [p.strip() for p in text[1:end].split(" > ")], text[end + 1:].lstrip("\n")
    return [], text


def pages_label(page_start: Optional[int], page_end: Optional[int]) -> str:
    """'p. 12', 'pp. 12-14', or '' when the chunk has no usable page.

    Chunks store 0-based PDF page indices; labels are 1-based, matching PDF
    viewers and what read_pages accepts. Negative indices come from TOC
    entries without a page.
    """
    if page_start is None or page_start < 0:
        return ""
    first = page_start + 1
    if page_end is not None and page_end > page_start:
        return f"pp. {first}-{page_end + 1}"
    return f"p. {first}"


# ST-style running headers/footers: 'RM0433 Rev 8   1897/3353', '302/357 DS12110 Rev 10'.
_DOC_CODE = r"[A-Z]{2}\d{3,6}\s+Rev\s+\d+"
_PAGE_FURNITURE = re.compile(
    rf"^[ \t]*(?:{_DOC_CODE}[ \t]+\d+/\d+|\d+/\d+[ \t]+{_DOC_CODE}|{_DOC_CODE})[ \t]*$",
    re.M,
)


def clean_text(text: str) -> str:
    """Drop page headers/footers and layout padding that only cost tokens.

    Runs of 3+ spaces become two (still visibly a column break), line
    indentation and trailing spaces go, and blank-line runs shrink to one.
    """
    text = _PAGE_FURNITURE.sub("", text)
    text = re.sub(r"[ \t]{3,}", "  ", text)
    text = re.sub(r"^[ \t]+|[ \t]+$", "", text, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n")


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())


def _excerpt(body: str, max_length: int = 320) -> str:
    text = _collapse_ws(body)
    if len(text) <= max_length:
        return text
    cut = text[:max_length]
    space = cut.rfind(" ")
    return (cut[:space] if space > max_length * 0.7 else cut) + " …"


class ResultFormatter:
    """Format search results for minimal token usage."""

    @staticmethod
    def format_response(response: SearchResponse) -> str:
        """Format a hybrid search response as compact markdown."""
        results = response.results
        header = f"Keyword: {response.keyword_mode} · Semantic: {response.semantic}"
        lines = [header]
        for note in response.notes:
            lines.append(f"Note: {note}")
        if not results:
            lines.append("")
            lines.append("No results found. Try fewer or different terms, the manual's own "
                         "wording, or find_register for a register name.")
            return "\n".join(lines)
        lines.append("")

        for i, result in enumerate(results, 1):
            lines.extend(ResultFormatter._format_hit(i, result, response.terms))
            lines.append("")

        lines.append("Follow up with read_section(chunk_id) for a full section, "
                     "read_pages(doc, pages) for raw pages, find_register(name) for bitfields.")
        return "\n".join(lines)

    @staticmethod
    def _format_hit(i: int, result: SearchResult, terms: List[str]) -> List[str]:
        hierarchy, body = split_prefix(result.text)
        body = clean_text(body)
        doc_title = result.doc_title or (hierarchy[0] if hierarchy else result.doc_id)
        path = hierarchy[1:] if len(hierarchy) > 1 else [result.section or "(untitled)"]
        # Parents first, leaf last; long breadcrumbs keep only the last three.
        crumb = " > ".join(path[-3:])

        meta = [doc_title]
        pages = pages_label(result.page_start, result.page_end)
        if pages:
            meta.append(pages)
        meta.append(f"doc `{result.doc_id}`")
        meta.append(f"chunk `{result.chunk_id}`")
        found = matched_terms(result.text, terms) if terms else []
        if found:
            meta.append("matched: " + ", ".join(found))
        if result.channels == ["semantic"]:
            meta.append("semantic match")
        if result.more_in_section:
            n = result.more_in_section
            meta.append(f"+{n} more matching chunk{'s' if n > 1 else ''} in this section")

        lines = [f"## {i}. {crumb}", " · ".join(meta)]
        if result.structured_data and result.structured_data.get("registers"):
            lines.append(ResultFormatter._register_summary(result.structured_data))
        snips = snippets(body, terms) if terms else []
        for s in snips or [_excerpt(body)]:
            lines.append(f"> {s}")
        return lines

    @staticmethod
    def _register_summary(data: Dict[str, Any]) -> str:
        out = []
        for reg in data.get("registers", [])[:4]:
            bits = []
            where = reg.get("address") or reg.get("offset")
            if where:
                bits.append(("addr " if reg.get("address") else "offset ") + str(where))
            if reg.get("reset_value"):
                bits.append(f"reset {reg['reset_value']}")
            names = [f["name"] for f in reg.get("fields", []) if f.get("name")]
            head = f"**{reg['name']}**" + (f" ({', '.join(bits)})" if bits else "")
            if names:
                more = len(names) - _MAX_FIELD_NAMES
                head += " fields: " + ", ".join(names[:_MAX_FIELD_NAMES]) + (
                    f", +{more} more" if more > 0 else "")
            out.append(head)
        extra = len(data.get("registers", [])) - 4
        if extra > 0:
            out.append(f"+{extra} more registers")
        return "\n".join(out)

    @staticmethod
    def format_results(results: List[SearchResult], max_results: int = 5) -> str:
        """Format bare results (no query context) as compact markdown."""
        response = SearchResponse(query="", results=results[:max_results],
                                  keyword_mode="n/a", semantic="n/a")
        return ResultFormatter.format_response(response)

    @staticmethod
    def _format_structured_data(data: dict) -> str:
        """Format structured register data in full."""
        lines = []

        peripheral = data.get("peripheral", "Unknown")
        table_type = data.get("table_type", "").replace("_", " ").title()

        lines.append(f"**{peripheral}** - {table_type}")
        lines.append("")

        for register in data.get("registers", []):
            reg_name = register["name"]
            address = register.get("address", "")
            offset = register.get("offset", "")

            header_parts = [f"### {reg_name}"]
            if address:
                header_parts.append(f"({address})")
            elif offset:
                header_parts.append(f"(Offset: {offset})")

            lines.append(" ".join(header_parts))

            details = [f"**Width:** {register.get('width', 32)}-bit"]
            if register.get("reset_value"):
                details.append(f"**Reset:** {register['reset_value']}")
            if register.get("access"):
                details.append(f"**Access:** {register['access']}")
            lines.append(" | ".join(details))

            if register.get("description"):
                lines.append(f"\n{register['description']}")

            if register.get("fields"):
                lines.append("\n**Fields:**")
                for field in register["fields"]:
                    lines.append(
                        f"- **{field['name']}** [{field['bits']}]: "
                        f"{field.get('description', '')} ({field.get('access', '')})"
                    )

            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _create_excerpt(text: str, max_length: int = 500) -> str:
        """Create an excerpt from text."""
        return _excerpt(text, max_length)

    @staticmethod
    def source_line(result: SearchResult) -> str:
        """'Source: <doc> · p. N · section · doc `id` · chunk `id`'."""
        hierarchy, _ = split_prefix(result.text)
        parts = [result.doc_title or (hierarchy[0] if hierarchy else result.doc_id)]
        pages = pages_label(result.page_start, result.page_end)
        if pages:
            parts.append(pages)
        if result.section:
            parts.append(result.section)
        parts.append(f"doc `{result.doc_id}`")
        parts.append(f"chunk `{result.chunk_id}`")
        return "**Source:** " + " · ".join(parts)

    @staticmethod
    def format_register(result: SearchResult) -> str:
        """Format a single register result.

        Args:
            result: Search result containing register data

        Returns:
            Formatted markdown string
        """
        if result.structured_data and result.structured_data.get("registers"):
            body = ResultFormatter._format_structured_data(result.structured_data)
        else:
            _, text = split_prefix(result.text)
            body = clean_text(text)
        return body + "\n\n" + ResultFormatter.source_line(result)
