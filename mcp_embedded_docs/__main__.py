"""CLI entry point for MCP Embedded Docs."""

import logging
import os
import sys


def _run_server():
    """Run MCP server directly, bypassing Click to avoid stdin/stdout interference."""
    # Route logs to stderr so the MCP host (Claude Code, VSCode) can surface
    # them. stdout is reserved for the JSON-RPC protocol -- logging there
    # would corrupt the stream. BITWISE_MCP_DEBUG=1 raises the level to DEBUG
    # for deeper investigation.
    level = logging.DEBUG if os.getenv("BITWISE_MCP_DEBUG") else logging.INFO
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from .server import mcp
    mcp.run(transport="stdio")


def cli():
    """Entry point - serves MCP by default, or delegates to Click CLI for other commands."""
    if len(sys.argv) <= 1 or (len(sys.argv) > 1 and sys.argv[1] == "serve"):
        _run_server()
    else:
        # Lazy import Click and heavy deps only when needed for CLI commands
        _cli_group()(standalone_mode=True)


def _cli_group():
    """Build the Click CLI group with heavy imports deferred."""
    import logging
    import click
    from pathlib import Path
    import hashlib

    from .config import Config
    from .ingestion.pdf_parser import PDFParser
    from .ingestion.table_detector import TableDetector
    from .ingestion.table_extractor import TableExtractor
    from .ingestion.chunker import SemanticChunker
    from .indexing.embedder import LocalEmbedder
    from .indexing.vector_store import VectorStore
    from .indexing.metadata_store import MetadataStore

    logger = logging.getLogger(__name__)

    @click.group()
    def _cli():
        """MCP Embedded Documentation Server CLI."""
        pass

    @_cli.command()
    @click.argument('pdf_path', type=click.Path(exists=True))
    @click.option('--title', help='Document title')
    @click.option('--version', help='Document version')
    @click.option(
        '--no-tables',
        is_flag=True,
        help=(
            "Skip pdfplumber-based register-table detection. ST reference "
            "manuals get richer per-register data from the section-text "
            "parser; the table pass mostly duplicates that and is the "
            "slowest, most memory-hungry phase of ingestion."
        ),
    )
    def ingest(pdf_path: str, title: str = None, version: str = None, no_tables: bool = False):
        """Index a PDF document."""
        pdf_path = Path(pdf_path)
        config = Config.load()

        click.echo(f"Ingesting {pdf_path.name}...", err=True)

        doc_id = hashlib.md5(pdf_path.name.encode()).hexdigest()[:16]

        click.echo("Parsing PDF...", err=True)
        with PDFParser(pdf_path) as parser:
            pages = parser.extract_text_with_layout()
            toc = parser.extract_toc()
            sections = parser.detect_sections(pages, toc)

        click.echo(f"  Extracted {len(pages)} pages, {len(sections)} sections", err=True)

        all_tables = []
        table_pages = {}
        if no_tables:
            click.echo("Skipping register-table detection (--no-tables).", err=True)
        else:
            click.echo(f"Detecting register tables across {len(pages)} pages...", err=True)
            extractor = TableExtractor(str(pdf_path))

            with TableDetector(str(pdf_path)) as detector:
                for i, page in enumerate(pages):
                    if i % 200 == 0:
                        click.echo(f"  page {i}/{len(pages)} (tables found: {len(all_tables)})", err=True)
                    detected = detector.detect_register_tables(page)
                    for region, table_data in detected:
                        context = detector.detect_table_context(page, region)
                        table = extractor.extract_register_table(region, table_data, context)
                        if table:
                            table_pages[len(all_tables)] = region.page_num
                            all_tables.append(table)

            click.echo(f"  Found {len(all_tables)} register tables", err=True)

        click.echo("Creating semantic chunks...", err=True)
        chunker = SemanticChunker(
            target_size=config.chunking.target_size,
            overlap=config.chunking.overlap,
            preserve_tables=config.chunking.preserve_tables
        )

        doc_title = title or pdf_path.stem
        chunks = chunker.chunk_document(
            doc_id, sections, all_tables,
            doc_title=doc_title,
            table_pages=table_pages,
        )
        click.echo(f"  Created {len(chunks)} chunks", err=True)

        click.echo("Indexing...", err=True)
        embedder = LocalEmbedder(
            model_name=config.embeddings.model,
            device=config.embeddings.device
        )

        vector_store = VectorStore(dimension=embedder.dimension)
        # Load any existing FAISS index before adding new vectors so we
        # accumulate across ingests instead of overwriting every time.
        vector_path = config.index.directory / config.index.vector_file
        if vector_path.exists():
            vector_store.load(vector_path)
        metadata_store = MetadataStore(config.index.directory / config.index.metadata_db)

        metadata_store.add_document(
            doc_id=doc_id,
            filename=pdf_path.name,
            title=title,
            version=version
        )

        chunk_texts = [chunk.text for chunk in chunks]
        chunk_ids = [chunk.id for chunk in chunks]

        click.echo("  Creating embeddings...", err=True)
        embeddings = embedder.embed_batch(chunk_texts, show_progress=True)

        vector_store.add_vectors(embeddings, chunk_ids)

        click.echo("  Storing metadata...", err=True)
        for chunk in chunks:
            metadata_store.add_chunk(
                chunk_id=chunk.id,
                doc_id=chunk.doc_id,
                chunk_type=chunk.chunk_type,
                text=chunk.text,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                structured_data=chunk.structured_data,
                metadata=chunk.metadata
            )

        config.index.directory.mkdir(parents=True, exist_ok=True)
        vector_store.save(config.index.directory / config.index.vector_file)
        metadata_store.close()

        click.echo(f"Successfully indexed {pdf_path.name}", err=True)
        click.echo(f"  Document ID: {doc_id}", err=True)
        click.echo(f"  Total chunks: {len(chunks)}", err=True)
        click.echo(f"  Register tables: {len(all_tables)}", err=True)

    @_cli.command()
    def serve():
        """Start MCP server on stdio."""
        _run_server()

    @_cli.command(name="rebuild-vectors")
    def rebuild_vectors():
        """Rebuild the FAISS index from chunks already in the metadata DB.

        Useful when the vector file is missing, corrupted, or was clobbered
        by a prior bug (the older ingest path overwrote the index per
        document instead of accumulating). Doesn't re-parse PDFs.
        """
        import sqlite3
        config = Config.load()

        embedder = LocalEmbedder(
            model_name=config.embeddings.model,
            device=config.embeddings.device
        )

        db_path = config.index.directory / config.index.metadata_db
        if not db_path.exists():
            click.echo(f"No metadata DB at {db_path}", err=True)
            return

        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        rows = list(con.execute("SELECT id, text FROM chunks ORDER BY rowid"))
        con.close()

        if not rows:
            click.echo("No chunks in metadata DB to embed.", err=True)
            return

        click.echo(f"Re-embedding {len(rows)} chunks...", err=True)
        chunk_ids = [r["id"] for r in rows]
        chunk_texts = [r["text"] for r in rows]
        embeddings = embedder.embed_batch(chunk_texts, show_progress=True)

        vector_store = VectorStore(dimension=embedder.dimension)
        vector_store.add_vectors(embeddings, chunk_ids)

        config.index.directory.mkdir(parents=True, exist_ok=True)
        vector_store.save(config.index.directory / config.index.vector_file)
        click.echo(f"Wrote {len(chunk_ids)} vectors to {config.index.directory / config.index.vector_file}", err=True)

    @_cli.command(name="list")
    def list_cmd():
        """List indexed documents."""
        config = Config.load()
        metadata_store = MetadataStore(config.index.directory / config.index.metadata_db)

        try:
            docs = metadata_store.list_documents()

            if not docs:
                click.echo("No documents indexed yet.", err=True)
                return

            click.echo("Indexed Documents:", err=True)
            click.echo("", err=True)

            for doc in docs:
                click.echo(f"  {doc['filename']}", err=True)
                if doc['title']:
                    click.echo(f"    Title: {doc['title']}", err=True)
                if doc['version']:
                    click.echo(f"    Version: {doc['version']}", err=True)
                click.echo(f"    ID: {doc['id']}", err=True)
                click.echo(f"    Indexed: {doc['index_date']}", err=True)
                click.echo("", err=True)
        finally:
            metadata_store.close()

    return _cli


if __name__ == "__main__":
    if len(sys.argv) <= 1 or sys.argv[1] == "serve":
        _run_server()
    else:
        _cli_group()()
