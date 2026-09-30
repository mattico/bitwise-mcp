# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                           # Install dependencies
uv run mcp-embedded-docs serve                    # Start MCP server (stdio)
uv run mcp-embedded-docs ingest PATH --title "Title"  # Ingest a PDF
uv run mcp-embedded-docs list                     # List indexed documents
uv run mcp-embedded-docs remove DOC_ID            # Remove a document (rows + vectors)
uv run mcp-embedded-docs rebuild-vectors          # Rebuild FTS and re-embed all chunks
uv run pytest                                     # Run tests
uv run pytest tests/test_chunker.py -k "test_name"  # Single test
uv run black mcp_embedded_docs/                   # Format
uv run mypy mcp_embedded_docs/                    # Type check
```

## Architecture

FastMCP server (`server.py`) exposing 7 tools: `search_docs`, `find_register`, `read_section`, `read_pages`, `list_docs`, `ingest_docs`, `remove_docs`. Tool bodies in `tools/` are synchronous; `server.py` runs each on a worker thread (`anyio.to_thread`) and appends it to the JSONL query log (`query_log.py`, `<index>/logs/queries.jsonl`). At startup `start_warmup()` opens the indexes and loads the embedding model on a background thread.

`stdio.py` replaces the MCP SDK's stdin reader on Windows. The SDK leaves a blocking `ReadFile` pending on stdin, and while it is pending, loading a native extension (numpy, torch, faiss, PyMuPDF) blocks until more input arrives, so lazy imports inside tool calls used to hang until the client sent something else. The replacement polls `PeekNamedPipe` and only reads bytes that are already there. Keep `run_stdio` as the server entry point.

### Ingestion Pipeline

```
ingestion/pipeline.py (ingest_pdf / remove_document, shared by CLI and MCP tools):
PDF → pdf_parser.py (PyMuPDF: text, TOC, section hierarchy)
    → table_detector.py (pdfplumber: find register tables on pages)
    → table_extractor.py (parse tables into Register/BitField structures)
    → chunker.py (semantic chunking with context prefixes)
    → embedder.py (granite-embedding-english-r2, 768-dim, normalized, float32 on CPU)
    → vector_store.py (FAISS IndexFlatL2) + metadata_store.py (SQLite FTS5)
```

Key chunking rules:
- Only leaf sections are chunked (parents with subsections are skipped to avoid duplication)
- Every chunk gets a hierarchy prefix: `[Doc > Section > Subsection]`
- Text splits on sentence boundaries (`. `, `.\n`, `\n\n`), never mid-word
- Register tables are never split — kept as whole chunks with both text and structured JSON
- Chunk IDs are `{doc_id}_{md5(text)[:12]}` to prevent collisions (semantic doc filtering relies on the prefix)
- Chunk `page_start`/`page_end` are 0-based PDF page indices; everything shown to agents is 1-based
- Re-ingest replaces: the store step deletes the doc's old chunks and vectors, writes new ones, and saves FAISS inside one `MetadataStore.write_transaction()` (SQLite's write lock also serializes the FAISS read-modify-write across processes)

### Search Pipeline

```
Query → HybridSearch.search_ex
        ├─ keyword_search_ex() → SQLite FTS5, porter-stemmed, bm25 with section title ×5
        │    query plan (retrieval/query.py): strict AND → relaxed OR → prefix OR,
        │    stops once ≥3 hits; explicit FTS5 syntax runs verbatim
        └─ _semantic_search() → FAISS (waits ≤20s for the background-loaded model)
        → weighted reciprocal-rank fusion (k=60, weights 0.5/0.5)
        → collapse chunks of the same section into one result
        → optional cross-encoder rerank of the top sections (retrieval/reranker.py)
        → ResultFormatter → markdown with highlighted snippets
```

A running server picks up CLI ingests/removes/rebuilds on its own: `HybridSearch.refresh_if_stale()` runs before each search and register lookup, and reloads the vectors when `vectors.faiss`/`.ids` change (mtime/size) and doc titles when SQLite's `PRAGMA data_version` changes. Its semantic status reads `partial (…)` when some chunks have no vector.

`tests/` has unit tests, but ranking changes should be checked against a real index with a known-item query set; keep the numbers in the change description.

### Storage

- `index/vectors.faiss` — FAISS flat index (cosine similarity via normalized L2)
- `index/metadata.db` — SQLite (WAL) with FTS5 external-content table; triggers keep FTS in sync using the FTS5 `'delete'` command (a plain `DELETE FROM chunks_fts` corrupts external-content tables). `MetadataStore` migrates older FTS schemas automatically on open.
- `index/logs/queries.jsonl` — tool call log
- `docs/` — PDF input directory (gitignored, per-project)

## Config

`config.yaml` (optional, falls back to defaults), or `$BITWISE_MCP_CONFIG`; relative paths resolve against the config file's directory, `$BITWISE_MCP_INDEX_DIR` overrides the index dir. Pydantic models in `config.py`:
- `chunking.target_size`: 2500 chars, `overlap`: 200 chars
- `search.keyword_weight`: 0.5, `semantic_weight`: 0.5 (rank-fusion weights)
- `search.rerank`: false, `rerank_model`: `cross-encoder/ms-marco-MiniLM-L6-v2`, `rerank_depth`: 20 (cross-encoder pass over the fused top sections)
- `embeddings.enabled`: true (false = keyword-only, no torch), `model`: `ibm-granite/granite-embedding-english-r2`, `device`: `cpu`
- `embeddings.query_prefix`: unset = the model's retrieval prefix from `embedder.QUERY_PREFIXES`, `""` = none (query-side only, no re-index)
- `embeddings.max_seq_length`: 512 (null = the model's limit); caps tokens per chunk (needs rebuild-vectors)
- The vector index records the model it was built with (`.ids` file); search turns semantic off with a rebuild-vectors hint if the configured model differs, and ingest refuses to mix models

## Plugin

`plugins/bitwise-embedded-docs/` contains the Claude Code plugin with `.mcp.json` entry point and two skills (`/ingest-docs`, `/search-docs`). Bump the version by changing `pyproject.toml` version field.

## Adding a New Tool

1. Create `tools/new_tool.py` with a synchronous function returning a markdown string
2. Register in `server.py` with `@mcp.tool()` on an async wrapper that calls it through `_call(...)` (worker thread + query log); the docstring becomes the tool description
3. Use lazy imports inside the tool function to keep server startup fast
