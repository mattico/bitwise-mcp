"""CLI entry point for MCP Embedded Docs."""

import logging
import os
import sys
import time


def _configure_logging():
    """Configure stderr logging for CLI and MCP server runs."""
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("mcp_embedded_docs").setLevel(
        logging.DEBUG if os.getenv("BITWISE_MCP_DEBUG") else logging.INFO
    )


def _run_server():
    """Run MCP server directly, bypassing Click to avoid stdin/stdout interference."""
    # Route logs to stderr so the MCP host (Claude Code, VSCode) can surface
    # them. stdout is reserved for the JSON-RPC protocol.
    _configure_logging()
    from .server import mcp, start_warmup
    from .stdio import run_stdio
    start_warmup()
    run_stdio(mcp)


def cli():
    """Entry point - serves MCP by default, or delegates to Click CLI for other commands."""
    if len(sys.argv) <= 1 or (len(sys.argv) > 1 and sys.argv[1] == "serve"):
        _run_server()
    else:
        # Lazy import Click and heavy deps only when needed for CLI commands
        _cli_group()(standalone_mode=True)


def _cli_group():
    """Build the Click CLI group with heavy imports deferred."""
    import click
    from pathlib import Path

    from .config import Config
    from .indexing.metadata_store import MetadataStore

    def echo(message: str):
        click.echo(message, err=True)

    @click.group()
    def _cli():
        """MCP Embedded Documentation Server CLI."""
        _configure_logging()

    @_cli.command()
    @click.argument('pdf_path', type=click.Path(exists=True, dir_okay=False))
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
        """Index a PDF document (replacing any previous version of it)."""
        from .ingestion.pipeline import ingest_pdf

        config = Config.load()
        try:
            report = ingest_pdf(
                Path(pdf_path), config,
                title=title, version=version,
                detect_tables=not no_tables,
                progress=echo,
            )
        except (FileNotFoundError, ValueError) as e:
            raise click.ClickException(str(e))

        echo(f"Successfully indexed {report.filename}")
        echo(f"  Document ID: {report.doc_id}")
        echo(f"  Total chunks: {report.chunks}")
        echo(f"  Register tables: {report.tables}")
        if config.embeddings.enabled:
            echo(f"  Vectors: {report.vectors}")
        else:
            echo("  Vectors: 0 (embeddings disabled)")
        if report.replaced_chunks:
            echo(f"  Replaced chunks: {report.replaced_chunks} (previous version)")
        timing = ", ".join(f"{k} {v:.1f}s" for k, v in report.timings.items())
        echo(f"  Time: {report.seconds:.1f}s ({timing})")
        if report.unembedded_chunks:
            echo(f"Warning: {report.unembedded_chunks} chunks in the index have no vector, "
                 "so semantic search cannot find them. Run `mcp-embedded-docs rebuild-vectors`.")

    @_cli.command()
    @click.argument('doc_id')
    def remove(doc_id: str):
        """Remove a document (by ID, see `list`) and its vectors from the index."""
        from .ingestion.pipeline import remove_document

        report = remove_document(doc_id, Config.load())
        if report is None:
            raise click.ClickException(f"Document not found: {doc_id}")
        echo(f"Removed {report.filename} (ID: {report.doc_id}): "
             f"{report.chunks} chunks, {report.vectors} vectors")

    @_cli.command()
    def serve():
        """Start MCP server on stdio."""
        _run_server()

    @_cli.command(name="rebuild-vectors")
    def rebuild_vectors():
        """Rebuild the FTS5 keyword index and the FAISS index from the metadata DB.

        Useful when the vector file is missing, corrupted, or was clobbered
        by a prior bug (the older ingest path overwrote the index per
        document instead of accumulating), or after changing the embedding
        model. Doesn't re-parse PDFs.
        """
        import numpy as np

        config = Config.load()
        db_path = config.index.directory / config.index.metadata_db
        if not db_path.exists():
            echo(f"No metadata DB at {db_path}")
            return

        store = MetadataStore(db_path)
        try:
            # External-content FTS5 can end up with stale rowid pointers
            # ("fts5: missing row N from content table 'chunks'"), which
            # silently empties keyword search. Rebuilding repairs it.
            echo("Rebuilding FTS5 keyword index...")
            store.rebuild_fts()

            if not config.embeddings.enabled:
                echo("Embeddings are disabled in config; rebuilt the keyword index only.")
                return

            # Embed document by document so progress is readable, then
            # pick up any chunks whose document row is missing.
            groups = []
            seen = set()
            for doc in sorted(store.list_documents(), key=lambda d: d["filename"] or ""):
                ids = store.get_chunk_ids(doc["id"])
                seen.update(ids)
                if ids:
                    groups.append((doc["filename"], ids))
            leftover = [cid for cid in store.all_chunk_ids() if cid not in seen]
            if leftover:
                groups.append(("(chunks without a document row)", leftover))

            total = sum(len(ids) for _, ids in groups)
            if not total:
                echo("No chunks in metadata DB to embed.")
                return

            from .indexing.embedder import LocalEmbedder
            from .indexing.vector_store import VectorStore

            echo("Loading embedding model...")
            embedder = LocalEmbedder(
                model_name=config.embeddings.model,
                device=config.embeddings.device,
                batch_size=config.embeddings.batch_size,
            )
            vector_store = VectorStore(dimension=embedder.dimension)

            echo(f"Re-embedding {total} chunks from {len(groups)} documents...")
            for name, ids in groups:
                chunks = store.get_chunks(ids)
                ids = [cid for cid in ids if cid in chunks]
                echo(f"  {name}: {len(ids)} chunks")
                embeddings = embedder.embed_batch(
                    [chunks[cid]["text"] for cid in ids], show_progress=True)
                vector_store.add_vectors(np.asarray(embeddings, dtype=np.float32), ids)

            # Embedding took a while without the write lock; an ingest or
            # remove may have committed since. Reconcile and save under the
            # lock so neither side's changes are lost.
            vector_path = config.index.directory / config.index.vector_file
            with store.write_transaction():
                live = set(store.all_chunk_ids())
                vector_store.remove_ids(set(vector_store.ids) - live)
                have = set(vector_store.ids)
                added = [cid for cid in live if cid not in have]
                if added:
                    echo(f"  {len(added)} chunks were added meanwhile; embedding them")
                    chunks = store.get_chunks(added)
                    added = [cid for cid in added if cid in chunks]
                    embeddings = embedder.embed_batch([chunks[cid]["text"] for cid in added])
                    vector_store.add_vectors(np.asarray(embeddings, dtype=np.float32), added)
                vector_store.save(vector_path)
        finally:
            store.close()

        echo(f"Wrote {vector_store.size} vectors to {vector_path}")

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
