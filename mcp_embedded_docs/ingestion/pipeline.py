"""The ingest and remove pipelines shared by the CLI and the MCP tools.

Ingest does all the slow work (parse, table detection, chunking, embedding)
before touching the index, then swaps the document's chunks and vectors in
one short step: the previous version's chunks and their vectors are deleted,
the new ones written, and the FAISS file saved atomically.

Nothing here imports torch at module level. The embedder is only constructed
when embeddings are enabled and no embedder was passed in, and removal never
needs it.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    import numpy as np

    from ..config import Config
    from ..indexing.vector_store import VectorStore

logger = logging.getLogger(__name__)

ProgressFn = Optional[Callable[[str], None]]

# Pages between table-detection progress lines.
TABLE_PROGRESS_EVERY = 200
# Chunks handed to the embedder per call; each call reports progress.
EMBED_GROUP = 512


@dataclass
class IngestReport:
    """What an ingest did."""

    doc_id: str
    filename: str
    path: str
    pages: int
    sections: int
    tables: int
    chunks: int
    vectors: int
    replaced_chunks: int
    seconds: float
    timings: Dict[str, float] = field(default_factory=dict)
    # Chunks of any document that have no vector (ingested while embeddings
    # were off, or before vectors.faiss was lost); rebuild-vectors fills them in.
    unembedded_chunks: int = 0


@dataclass
class RemoveReport:
    """What a removal did."""

    doc_id: str
    filename: str
    chunks: int
    vectors: int


def doc_id_for(pdf_path: Path) -> str:
    """Document id for a file. Existing indexes depend on this staying stable."""
    return hashlib.md5(pdf_path.name.encode()).hexdigest()[:16]


def ingest_pdf(
    pdf_path: Path,
    config: "Config",
    *,
    title: Optional[str] = None,
    version: Optional[str] = None,
    detect_tables: bool = True,
    embedder: Any = None,
    progress: ProgressFn = None,
) -> IngestReport:
    """Index a PDF, replacing any previous version of the same document.

    Args:
        pdf_path: The PDF to index
        config: Loaded configuration (index location, chunking, embeddings)
        title: Document title (defaults to the file stem in chunk prefixes)
        version: Document version
        detect_tables: Run pdfplumber register-table detection
        embedder: An already-loaded embedder (anything with `dimension` and
            `embed_batch(texts)`); one is loaded when None and embeddings
            are enabled
        progress: Receives short status lines

    Returns:
        IngestReport

    Raises:
        FileNotFoundError: pdf_path does not exist
        ValueError: pdf_path is not a PDF, or the existing vector index has a
            different dimension than the embedder
    """
    from ..indexing.metadata_store import MetadataStore
    from .chunker import SemanticChunker
    from .pdf_parser import PDFParser

    say = progress or (lambda _msg: None)
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Only PDF files are supported, got: {pdf_path.suffix or pdf_path.name}"
        )

    use_vectors = config.embeddings.enabled
    doc_id = doc_id_for(pdf_path)
    timings: Dict[str, float] = {}
    overall_start = time.perf_counter()

    say(f"Ingesting {pdf_path.name}...")

    # 1. Parse
    say("Parsing PDF...")
    t0 = time.perf_counter()
    with PDFParser(pdf_path) as parser:
        pages = parser.extract_text_with_layout()
        toc = parser.extract_toc()
        sections = parser.detect_sections(pages, toc)
    timings["parse"] = time.perf_counter() - t0
    say(f"  Extracted {len(pages)} pages, {len(sections)} sections")

    # 2. Register tables
    all_tables: List[Any] = []
    table_pages: Dict[int, int] = {}
    t0 = time.perf_counter()
    if detect_tables:
        all_tables, table_pages = _detect_tables(pdf_path, pages, say)
        say(f"  Found {len(all_tables)} register tables")
    else:
        say("Skipping register-table detection.")
    timings["tables"] = time.perf_counter() - t0

    # 3. Chunk
    say("Creating semantic chunks...")
    t0 = time.perf_counter()
    chunker = SemanticChunker(
        target_size=config.chunking.target_size,
        overlap=config.chunking.overlap,
        preserve_tables=config.chunking.preserve_tables,
        pdf_path=pdf_path,
    )
    chunks = chunker.chunk_document(
        doc_id,
        sections,
        all_tables,
        doc_title=title or pdf_path.stem,
        table_pages=table_pages,
    )
    # Identical text yields identical ids; keep the first so rows and vectors
    # stay one-to-one.
    unique: Dict[str, Any] = {}
    for chunk in chunks:
        unique.setdefault(chunk.id, chunk)
    if len(unique) != len(chunks):
        logger.debug("Dropped %d duplicate chunks", len(chunks) - len(unique))
    chunks = list(unique.values())
    timings["chunk"] = time.perf_counter() - t0
    say(f"  Created {len(chunks)} chunks")

    # 4. Embed
    embeddings = None
    t0 = time.perf_counter()
    if use_vectors and chunks:
        if embedder is None:
            say("Loading embedding model...")
            from ..indexing.embedder import LocalEmbedder

            embedder = LocalEmbedder(
                model_name=config.embeddings.model,
                device=config.embeddings.device,
                batch_size=config.embeddings.batch_size,
                max_seq_length=config.embeddings.max_seq_length,
            )
            timings["load_model"] = time.perf_counter() - t0
            t0 = time.perf_counter()
        say(f"Embedding {len(chunks)} chunks...")
        embeddings = _embed(embedder, [c.text for c in chunks], say)
    elif not use_vectors:
        say("Embeddings disabled; skipping vectors.")
    timings["embed"] = time.perf_counter() - t0

    # 5. Swap into the index
    say("Updating index...")
    t0 = time.perf_counter()
    index_dir = config.index.directory
    vector_path = index_dir / config.index.vector_file
    index_dir.mkdir(parents=True, exist_ok=True)

    metadata_store = MetadataStore(index_dir / config.index.metadata_db)
    unembedded = 0
    try:
        # One transaction for rows and vectors: SQLite's write lock is held
        # across processes, so the FAISS read-modify-write below cannot race
        # another ingest or remove, and a failure leaves the old copy intact.
        with metadata_store.write_transaction():
            vector_store = None
            if vector_path.exists():
                vector_store = _load_vectors(vector_path)
            if embeddings is not None:
                if vector_store is None:
                    from ..indexing.vector_store import VectorStore

                    vector_store = VectorStore(dimension=int(embeddings.shape[1]))
                elif vector_store.dimension != embeddings.shape[1]:
                    raise ValueError(
                        f"Existing vector index {vector_path} has dimension "
                        f"{vector_store.dimension} but the embedder produces "
                        f"{embeddings.shape[1]}; run `mcp-embedded-docs rebuild-vectors`."
                    )
                model = getattr(embedder, "model_name", None)
                if vector_store.model and model and vector_store.model != model:
                    raise ValueError(
                        f"Existing vector index {vector_path} was built with "
                        f"{vector_store.model} but the embedder is {model}; "
                        "run `mcp-embedded-docs rebuild-vectors`."
                    )
                vector_store.model = vector_store.model or model

            old_ids = metadata_store.delete_document_chunks(doc_id)
            metadata_store.add_document(
                doc_id=doc_id,
                filename=pdf_path.name,
                title=title,
                version=version,
                path=str(pdf_path),
            )
            metadata_store.add_chunks(
                {
                    "chunk_id": c.id,
                    "doc_id": c.doc_id,
                    "chunk_type": c.chunk_type,
                    "text": c.text,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "structured_data": c.structured_data,
                    "metadata": c.metadata,
                }
                for c in chunks
            )

            if vector_store is not None:
                removed = vector_store.remove_ids(old_ids)
                if embeddings is not None:
                    vector_store.add_vectors(embeddings, [c.id for c in chunks])
                removed += vector_store.dedupe()
                # Vectors whose chunk is gone (left by older ingest/remove code).
                live = set(metadata_store.all_chunk_ids())
                orphans = [cid for cid in vector_store.ids if cid not in live]
                removed += vector_store.remove_ids(orphans)
                vector_store.save(vector_path)
                logger.debug("Removed %d stale vectors (%d orphans)", removed, len(orphans))
                if use_vectors:
                    have = set(vector_store.ids)
                    unembedded = sum(1 for cid in live if cid not in have)
            elif use_vectors:
                unembedded = len(metadata_store.all_chunk_ids())
    finally:
        metadata_store.close()
    timings["store"] = time.perf_counter() - t0

    report = IngestReport(
        doc_id=doc_id,
        filename=pdf_path.name,
        path=str(pdf_path),
        pages=len(pages),
        sections=len(sections),
        tables=len(all_tables),
        chunks=len(chunks),
        vectors=0 if embeddings is None else len(embeddings),
        replaced_chunks=len(old_ids),
        seconds=time.perf_counter() - overall_start,
        timings=timings,
        unembedded_chunks=unembedded,
    )
    if unembedded:
        logger.warning("%d chunks in the index have no vector; run rebuild-vectors",
                       unembedded)
    logger.info(
        "Ingested %s: %d chunks, %d vectors, %d replaced in %.1fs (%s)",
        report.filename,
        report.chunks,
        report.vectors,
        report.replaced_chunks,
        report.seconds,
        ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()),
    )
    return report


def remove_document(doc_id: str, config: "Config") -> Optional[RemoveReport]:
    """Delete a document's rows and vectors from the index.

    Never loads the embedding model.

    Returns:
        RemoveReport, or None if the document is not indexed
    """
    from ..indexing.metadata_store import MetadataStore

    db_path = config.index.directory / config.index.metadata_db
    if not db_path.exists():
        return None

    vector_path = config.index.directory / config.index.vector_file
    vectors = 0
    metadata_store = MetadataStore(db_path)
    try:
        with metadata_store.write_transaction():
            doc = metadata_store.get_document(doc_id)
            if doc is None:
                return None
            chunk_ids = metadata_store.get_chunk_ids(doc_id)
            metadata_store.delete_document(doc_id)
            if vector_path.exists():
                vector_store = _load_vectors(vector_path)
                vectors = vector_store.remove_ids(chunk_ids)
                if vectors:
                    vector_store.save(vector_path)
    finally:
        metadata_store.close()

    return RemoveReport(
        doc_id=doc_id,
        filename=doc.get("filename") or "",
        chunks=len(chunk_ids),
        vectors=vectors,
    )


def _detect_tables(pdf_path: Path, pages: List[Any], say: Callable[[str], None]):
    """Run register-table detection over every page."""
    from .table_detector import TableDetector
    from .table_extractor import TableExtractor

    say(f"Detecting register tables across {len(pages)} pages...")
    extractor = TableExtractor(str(pdf_path))
    tables: List[Any] = []
    table_pages: Dict[int, int] = {}
    with TableDetector(str(pdf_path)) as detector:
        for i, page in enumerate(pages):
            if i % TABLE_PROGRESS_EVERY == 0:
                say(f"  page {i}/{len(pages)} (tables found: {len(tables)})")
            for region, table_data in detector.detect_register_tables(page):
                context = detector.detect_table_context(page, region)
                table = extractor.extract_register_table(region, table_data, context)
                if table:
                    table_pages[len(tables)] = region.page_num
                    tables.append(table)
    return tables, table_pages


def _embed(embedder: Any, texts: List[str], say: Callable[[str], None]) -> "np.ndarray":
    """Embed texts in groups, reporting progress between groups."""
    import numpy as np

    parts = []
    for start in range(0, len(texts), EMBED_GROUP):
        group = texts[start : start + EMBED_GROUP]
        parts.append(
            np.asarray(embedder.embed_batch(group), dtype=np.float32)
        )
        if len(texts) > EMBED_GROUP:
            say(f"  embedded {min(start + EMBED_GROUP, len(texts))}/{len(texts)}")
    return np.ascontiguousarray(np.vstack(parts))


def _load_vectors(vector_path: Path) -> "VectorStore":
    from ..indexing.vector_store import VectorStore

    store = VectorStore()
    store.load(vector_path)
    return store
