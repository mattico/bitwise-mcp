"""Read tools: a whole section by chunk id, or raw PDF pages."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import Config
from ..indexing.metadata_store import MetadataStore
from ..retrieval.formatter import clean_text, pages_label, split_prefix
from ..retrieval.hybrid_search import HybridSearch

logger = logging.getLogger(__name__)

MAX_PAGES_PER_CALL = 10


def _strip_overlap(prev: str, nxt: str, max_overlap: int = 600) -> str:
    """`nxt` without the text it repeats from the end of `prev` (chunk overlap)."""
    limit = min(len(prev), len(nxt), max_overlap)
    for k in range(limit, 19, -1):
        if prev.endswith(nxt[:k]):
            return nxt[k:]
    return nxt


def section_text(store: MetadataStore, chunk_id: str, max_chars: int = 20000,
                 offset: int = 0,
                 chunks: Optional[List[Dict]] = None) -> Tuple[str, bool]:
    """Full text of the section containing `chunk_id`, chunks stitched in order.

    Args:
        chunks: The section's chunks, if the caller already fetched them

    Returns:
        (text, truncated)
    """
    if chunks is None:
        chunks = store.get_section_chunks(chunk_id)
    if not chunks:
        return "", False
    bodies: List[str] = []
    prev = ""
    for c in chunks:
        _, body = split_prefix(c["text"])
        body = clean_text(body)
        piece = _strip_overlap(prev, body) if prev else body
        bodies.append(piece)
        prev = body
    text = "\n".join(bodies)
    end = offset + max_chars if max_chars else len(text)
    return text[offset:end], end < len(text)


def read_section(search: HybridSearch, chunk_id: str, max_chars: int = 20000,
                 offset: int = 0) -> str:
    """Markdown for the whole section a search hit came from."""
    chunk_id = chunk_id.strip().strip("`")
    chunks = search.metadata_store.get_section_chunks(chunk_id)
    chunk = next((c for c in chunks if c["id"] == chunk_id), None)
    if chunk is None:
        return f"No chunk '{chunk_id}'. Chunk ids come from search_docs results."
    hierarchy, _ = split_prefix(chunk["text"])
    text, truncated = section_text(search.metadata_store, chunk_id, max_chars, offset, chunks)
    first = min((c["page_start"] for c in chunks if c["page_start"] is not None), default=None)
    last = max((c["page_end"] for c in chunks if c["page_end"] is not None), default=None)
    title = " > ".join(hierarchy) if hierarchy else (chunk["section_hierarchy"] or chunk_id)
    meta = [f"doc `{chunk['doc_id']}`"]
    pages = pages_label(first, last)
    if pages:
        meta.insert(0, pages)
    lines = [f"# {title}", " · ".join(meta), "", text]
    if truncated:
        lines.append("")
        lines.append(f"(truncated; read_section('{chunk_id}', offset={offset + max_chars}) continues)")
    return "\n".join(lines)


def parse_pages(pages: str) -> Tuple[int, int]:
    """'12' or '12-14' (1-based, inclusive) -> (first, last)."""
    m = re.fullmatch(r"\s*(\d+)\s*(?:[-–]\s*(\d+)\s*)?", pages or "")
    if not m:
        raise ValueError(f"pages must look like '12' or '12-14', got {pages!r}")
    first = int(m.group(1))
    last = int(m.group(2) or first)
    if first < 1 or last < first:
        raise ValueError(f"invalid page range {pages!r}")
    return first, last


_pdf_path_cache: Dict[str, Path] = {}


def resolve_pdf(doc: Dict, config: Config) -> Optional[Path]:
    """Where the source PDF of an indexed document lives, if it can be found.

    Only hits are cached: a PDF copied into doc_dirs later is still found.
    """
    if doc.get("path") and Path(doc["path"]).is_file():
        return Path(doc["path"])
    name = doc["filename"]
    cached = _pdf_path_cache.get(name)
    if cached is not None and cached.is_file():
        return cached
    found = None
    for d in config.doc_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*.pdf"):
            if p.name.lower() == name.lower():
                found = p
                break
        if found:
            break
    if found:
        _pdf_path_cache[name] = found
    return found


def read_pages(search: HybridSearch, config: Config, doc: str, pages: str,
               max_chars: int = 20000) -> str:
    """Text of PDF pages (1-based), from the source PDF when available."""
    doc_id, error = search.resolve_doc(doc)
    if error or not doc_id:
        return error or "Give a document id or filename (see list_docs)."
    try:
        first, last = parse_pages(pages)
    except ValueError as exc:
        return str(exc)
    if last - first + 1 > MAX_PAGES_PER_CALL:
        return (f"{last - first + 1} pages requested; at most {MAX_PAGES_PER_CALL} per call. "
                f"Ask for pages='{first}-{first + MAX_PAGES_PER_CALL - 1}' and continue from there.")
    record = search.metadata_store.get_document(doc_id)
    title = (record or {}).get("title") or (record or {}).get("filename") or doc_id

    pdf = resolve_pdf(record, config) if record else None
    parts: List[str] = []
    if pdf is not None:
        import fitz  # PyMuPDF; imported lazily, it is a native extension

        with fitz.open(pdf) as pdf_doc:
            total = pdf_doc.page_count
            if first > total:
                return f"{title} has {total} pages; page {first} does not exist."
            for n in range(first, min(last, total) + 1):
                text = clean_text(pdf_doc[n - 1].get_text("text"))
                parts.append(f"[page {n}]\n{text}")
        source = f"source PDF `{pdf.name}` ({total} pages)"
    else:
        # No PDF on this machine: stitch the indexed chunks covering the range.
        chunks = search.metadata_store.get_page_chunks(doc_id, first - 1, last - 1)
        if not chunks:
            return f"No indexed text for pages {first}-{last} of {title}, and the source PDF was not found."
        prev = ""
        for c in chunks:
            _, body = split_prefix(c["text"])
            body = clean_text(body)
            parts.append(_strip_overlap(prev, body) if prev else body)
            prev = body
        source = "indexed chunks (source PDF not found; chunk page ranges are approximate)"

    text = "\n\n".join(parts)
    truncated = len(text) > max_chars
    lines = [f"# {title}, pages {first}-{last}", f"doc `{doc_id}` · {source}", "",
             text[:max_chars]]
    if truncated:
        lines.append("")
        lines.append(f"(truncated at {max_chars} chars; request fewer pages)")
    return "\n".join(lines)
