"""List documents tool."""

import logging
from typing import Optional

from ..indexing.metadata_store import MetadataStore
from ..config import Config

logger = logging.getLogger(__name__)


def list_docs(config: Optional[Config] = None) -> str:
    """List indexed documents and PDFs available for ingestion.

    Args:
        config: Configuration object

    Returns:
        Formatted list of documents as markdown
    """
    if config is None:
        config = Config.load()

    # Lightweight DB check for indexed status
    db_path = config.index.directory / config.index.metadata_db
    indexed: dict = {}
    if db_path.exists():
        store = MetadataStore(db_path)
        try:
            for doc in store.list_documents():
                stats = store.get_document_stats(doc["id"]) or {}
                indexed[doc["filename"].lower()] = {**doc, **stats}
        finally:
            store.close()

    # Scan doc directories for PDF files
    available: dict = {}
    for doc_dir in config.doc_dirs:
        if not doc_dir.exists():
            continue
        for pdf_path in doc_dir.glob("**/*.pdf"):
            available.setdefault(pdf_path.name.lower(), pdf_path)

    if not indexed and not available:
        return f"No PDF files found in: {', '.join(str(d) for d in config.doc_dirs)}"

    lines = ["# Documentation", ""]
    lines.append(f"**{len(indexed)}** indexed · **{len(set(available) - set(indexed))}** "
                 f"more PDFs available in {', '.join(str(d) for d in config.doc_dirs)}")
    lines.append("")

    if indexed:
        lines.append("## Indexed (use the id as doc_filter / read_pages doc)")
        for key in sorted(indexed, key=lambda k: indexed[k]["filename"].lower()):
            doc = indexed[key]
            label = doc["title"] or doc["filename"]
            extra = f" — {doc['filename']}" if doc["title"] else ""
            lines.append(f"- `{doc['id']}` **{label}**{extra} ({doc.get('chunks', 0)} chunks)")
        lines.append("")

    not_indexed = sorted((p for k, p in available.items() if k not in indexed),
                         key=lambda p: p.name.lower())
    if not_indexed:
        lines.append("## Available, not indexed (ingest_docs with the filename)")
        for p in not_indexed:
            lines.append(f"- {p.name} ({p.stat().st_size / (1024 * 1024):.1f} MB)")

    return "\n".join(lines)
