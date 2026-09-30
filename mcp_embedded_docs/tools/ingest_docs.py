"""Ingest documentation tool."""

import logging
import os
from pathlib import Path
from typing import Any, Optional

from ..config import Config

logger = logging.getLogger(__name__)


def ingest_docs(
    doc_path: str,
    title: Optional[str] = None,
    version: Optional[str] = None,
    config: Optional[Config] = None,
    embedder: Any = None,
    detect_tables: bool = True,
) -> str:
    """Ingest a PDF into the index, replacing any previous version of it.

    Blocking; the server runs it on a worker thread.

    Args:
        doc_path: Path to the PDF, or a filename/relative path under one of
            config.doc_dirs (as shown by list_docs)
        title: Optional document title
        version: Optional document version
        config: Configuration object
        embedder: Already-loaded embedder to reuse (loaded on demand if None)
        detect_tables: Run register-table detection

    Returns:
        Status message as markdown
    """
    from ..ingestion.pipeline import ingest_pdf

    if config is None:
        config = Config.load()

    path = resolve_doc_path(doc_path, config)
    if path is None:
        dirs = ", ".join(str(d) for d in config.doc_dirs) or "(none configured)"
        return (
            f"❌ Error: Documentation file not found: {doc_path}\n\n"
            f"Searched as given and under doc_dirs: {dirs}"
        )

    if path.suffix.lower() != ".pdf":
        return f"❌ Error: Currently only PDF files are supported. Got: {path.suffix or path.name}"

    try:
        report = ingest_pdf(
            path,
            config,
            title=title,
            version=version,
            detect_tables=detect_tables,
            embedder=embedder,
            progress=logger.info,
        )
    except Exception as e:
        logger.exception("Ingestion failed for %s", path)
        return f"**Error during ingestion:** {e}"

    if config.embeddings.enabled:
        vectors = str(report.vectors)
    else:
        vectors = "0 (embeddings disabled; keyword search only)"
    timing = ", ".join(f"{k} {v:.1f}s" for k, v in report.timings.items())

    lines = [
        f"✅ **Successfully indexed {report.filename}**",
        "",
        f"- **Document ID:** `{report.doc_id}`",
        f"- **Path:** {report.path}",
        f"- **Pages:** {report.pages} ({report.sections} sections)",
        f"- **Chunks:** {report.chunks}",
        f"- **Register tables:** {report.tables}"
        + ("" if detect_tables else " (detection skipped)"),
        f"- **Vectors:** {vectors}",
        f"- **Replaced chunks:** {report.replaced_chunks}"
        + (" (previous version removed)" if report.replaced_chunks else ""),
        f"- **Time:** {report.seconds:.1f}s ({timing})",
    ]
    if report.unembedded_chunks:
        lines += [
            "",
            f"⚠️ {report.unembedded_chunks} chunks in the index (from this or other documents) "
            "have no vector, so semantic search cannot find them. Run "
            "`mcp-embedded-docs rebuild-vectors` to embed them.",
        ]
    return "\n".join(lines)


def resolve_doc_path(doc_path: str, config: Config) -> Optional[Path]:
    """Find the file an agent meant by `doc_path`.

    Tries the path as given, then relative to each doc dir, then a
    case-insensitive recursive filename search of the doc dirs (a name with
    no extension also matches `<name>.pdf`).
    """
    given = Path(doc_path).expanduser()
    if given.is_file():
        return given.resolve()

    for doc_dir in config.doc_dirs:
        candidate = Path(doc_dir) / given
        if not given.is_absolute() and candidate.is_file():
            return candidate.resolve()

    wanted = {given.name.lower()}
    if not given.suffix:
        wanted.add(given.name.lower() + ".pdf")
    for doc_dir in config.doc_dirs:
        if not Path(doc_dir).is_dir():
            continue
        for root, dirs, files in os.walk(doc_dir):
            dirs.sort()
            for name in sorted(files):
                if name.lower() in wanted:
                    return (Path(root) / name).resolve()
    return None
