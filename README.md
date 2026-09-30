# bitwise-mcp

MCP server for embedded developers. Ingests PDF reference manuals (1000+ pages), extracts register definitions, and provides fast semantic search. Built with [FastMCP](https://github.com/jlowin/fastmcp) and available as a Claude Code plugin.

## Features

- **PDF Ingestion** - Parses large reference manuals preserving structure
- **Register Table Extraction** - Detects and converts register definitions to structured JSON
- **Hybrid Search** - Stemmed keyword search (SQLite FTS5) fused with semantic similarity (FAISS) by reciprocal rank; plain queries start strict (all terms) and widen automatically when that finds too little
- **Agent-Friendly Results** - Highlighted snippets, 1-based PDF pages, doc and chunk ids, and follow-up tools to read a whole section or exact pages
- **Context-Aware Chunking** - Chunks include section hierarchy prefixes (e.g. `[Manual > FlexCAN > MCR Register]`) for better retrieval
- **Sentence-Aware Splitting** - Text splits on sentence boundaries with 1-2 sentence overlap, never mid-word
- **Compact Output** - Formats responses to minimize token usage

## Installation

### Option 1: Claude Code Plugin (Recommended)

Install directly from the Claude Code plugin marketplace:

```bash
claude plugin add bitwise-embedded-docs
```

This registers the MCP server and adds `/ingest-docs` and `/search-docs` slash commands.

### Option 2: Global Install

Install once, use across all projects:

```bash
# From this repository directory
pip install -e .

# Then in ANY project directory where you want to use it
claude mcp add --scope project embedded-docs python -m mcp_embedded_docs
```

Each project maintains its own isolated documentation index. When you run the server in a project, it only indexes and searches PDFs in that project's `docs/` directory.

### Option 3: uv Install (Development)

```bash
uv sync

# Add to Claude Code
claude mcp add embedded-docs --command uv --args "run" "mcp-embedded-docs" "serve" --cwd "<path-to-this-repo>"
```

Restart Claude Code after adding the server.

## Usage

Place PDFs in a `docs/` directory, then in Claude Code:

```
What PDFs are available?
Ingest any files that haven't been ingested yet
What's the base address for FlexCAN0?
```

### Example: Checking Available PDFs

![Listing available PDFs](images/Screenshot%20(10).PNG)

### Example: Searching Documentation

The MCP server automatically queries the indexed documentation when you ask questions:

![Documentation search in action](images/Screenshot%20(11).PNG)

### CLI Usage

```bash
uv run mcp-embedded-docs ingest docs/manual.pdf --title "MCU Manual"  # re-ingest replaces
uv run mcp-embedded-docs list                # View indexed documents
uv run mcp-embedded-docs remove <doc_id>     # Remove a document (rows + vectors)
uv run mcp-embedded-docs rebuild-vectors     # Rebuild keyword index and re-embed everything
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_docs` | Hybrid keyword + semantic search; optional `doc_filter` (id or filename fragment) |
| `find_register` | Register definition by name; case-insensitive, peripheral prefix optional (`GUSBCFG` → `OTG_GUSBCFG`) |
| `read_section` | Whole section a search hit came from, by chunk id |
| `read_pages` | Text of specific 1-based PDF pages (max 10 per call) |
| `list_docs` | Indexed documents (with ids) and PDFs available for ingestion |
| `ingest_docs` | Ingest or re-ingest a PDF (path or bare filename from `doc_dirs`) |
| `remove_docs` | Remove a document's chunks, registers and vectors |

### Query semantics

A plain query's terms are AND-ed first; if that matches fewer than three chunks the same terms are re-run as a ranked OR, then a prefix OR, with strict matches kept on top. Words are porter-stemmed and identifiers split on `_`, so `OTG_GUSBCFG`, `GUSBCFG` and `timings`/`timing` all match. Queries using FTS5 syntax (`"exact phrase"`, `OR`, `NOT`, `NEAR`, `prefix*`) run verbatim. The result header says which mode ran and whether semantic search took part.

## Configuration

`config.yaml` in the working directory, or the file named by `$BITWISE_MCP_CONFIG`. Relative paths in it resolve against the file's own directory. `$BITWISE_MCP_INDEX_DIR` overrides the index directory. See `config.yaml.example`; notable settings:

- `embeddings.enabled` (default `true`): set `false` for keyword-only search: no torch import, no model in memory, faster ingest.
- `search.keyword_weight` / `search.semantic_weight`: weights in rank fusion (default 0.5 / 0.5).

Every tool call is appended to `<index>/logs/queries.jsonl` (arguments, duration, output preview) so slow or empty searches can be found afterwards. `BITWISE_MCP_LOG=0` disables it.

On Windows the server reads stdin without leaving a blocking read pending (`mcp_embedded_docs/stdio.py`); the MCP SDK's default reader deadlocks lazy imports of native modules such as numpy and torch inside tool calls.

## Architecture

Built on [FastMCP](https://github.com/jlowin/fastmcp) for the MCP server layer. The ingestion pipeline:

1. **PDF Parsing** (PyMuPDF) - Extracts text with layout, TOC, and section hierarchy
2. **Table Detection** (pdfplumber) - Identifies register maps, bitfield definitions, memory maps
3. **Semantic Chunking** - Leaf-only section chunking with contextual hierarchy prefixes, sentence-aware splitting, and content-based deduplication
4. **Embedding** (sentence-transformers, bge-small-en-v1.5) - Local embeddings, no API calls; the model is read from the local Hugging Face cache after the first download
5. **Indexing** (FAISS + SQLite FTS5) - Rows and vectors are written in one transaction; re-ingesting a document replaces its previous chunks and vectors

## Tech Stack

Python 3.10+ | FastMCP | PyMuPDF | pdfplumber | sentence-transformers | FAISS | SQLite FTS5

## Performance

**Tested:** S32K144 Reference Manual (2,179 pages, 14MB)
**Results:** 3min indexing, <500ms search, ~500MB memory

**STM32H7 set** (10 PDFs, ~9,700 chunks): searches take 5-40 ms. The server answers the MCP handshake immediately and loads the embedding model (~3 s) in the background. A search in those first seconds waits for the model for up to 20 s, then answers from keywords alone. The server uses ~530 MB with the model loaded, or well under 100 MB with `embeddings.enabled: false`.

## License

[MIT](LICENSE)
