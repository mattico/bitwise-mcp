"""MCP server for embedded documentation using FastMCP."""

import functools
import logging
import threading
import time
from typing import Any, Callable, Optional, TYPE_CHECKING

import anyio
from mcp.server.fastmcp import FastMCP

from . import query_log
from .config import Config

if TYPE_CHECKING:
    from .retrieval.hybrid_search import HybridSearch

logger = logging.getLogger(__name__)

# Globals cached for the life of the server process: the keyword index, the
# vector file and the embedding model are loaded once, not per tool call.
_config: Optional[Config] = None
_search: Optional["HybridSearch"] = None
_search_lock = threading.Lock()


def get_config() -> Config:
    """Get or create config instance."""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def get_search() -> "HybridSearch":
    """Get or create the shared HybridSearch instance."""
    global _search
    if _search is None:
        with _search_lock:
            if _search is None:
                from .retrieval.hybrid_search import HybridSearch
                t0 = time.perf_counter()
                logger.info("Initializing HybridSearch...")
                _search = HybridSearch(get_config())
                logger.info("HybridSearch ready in %.2fs (semantic: %s; model loading in background)",
                            time.perf_counter() - t0, _search.semantic_status)
    return _search


def start_warmup() -> None:
    """Open the indexes and start loading the embedding model in the background,
    so the first search does not pay for it. Only safe because stdio.run_stdio
    never leaves a blocking stdin read pending (see stdio.py)."""
    def _warm() -> None:
        try:
            get_search()
        except Exception:  # noqa: BLE001 - the first tool call will report it
            logger.exception("warm-up failed")

    threading.Thread(target=_warm, name="warmup", daemon=True).start()


def _reload_search() -> None:
    """Make a running server see what an ingest or remove just wrote."""
    if _search is not None:
        _search.reload()


async def _call(tool: str, fn: Callable[..., str], **kwargs: Any) -> str:
    """Run a blocking tool body on a worker thread and log the call.

    Tool bodies do SQLite, FAISS, model inference or whole-PDF parsing; run on
    the event loop they would stall pings and cancellations for their duration.
    """
    t0 = time.monotonic()
    try:
        result = await anyio.to_thread.run_sync(functools.partial(fn, **kwargs))
    except Exception as exc:
        ms = (time.monotonic() - t0) * 1000
        logger.exception("%s failed after %.0f ms", tool, ms)
        query_log.record(get_config().index.directory, tool, kwargs, ms=ms,
                         error=f"{type(exc).__name__}: {exc}")
        raise
    ms = (time.monotonic() - t0) * 1000
    logger.info("%s completed in %.0f ms", tool, ms)
    query_log.record(get_config().index.directory, tool, kwargs, ms=ms, result=result)
    return result


mcp = FastMCP(
    "mcp-embedded-docs",
    instructions=(
        "Search indexed embedded-systems documentation (reference manuals, datasheets, "
        "errata, app notes). Typical flow: list_docs to see what is indexed; search_docs "
        "for topics; find_register for a register's bitfields; read_section(chunk_id) to "
        "read a hit's whole section; read_pages(doc, pages) for exact PDF pages. Queries "
        "work best in the manual's own vocabulary (peripheral and register names, bit "
        "names, 'errata', 'initialization sequence'); if a search misses, rephrase with "
        "synonyms rather than repeating it."
    ),
)


@mcp.tool()
async def search_docs(
    query: str,
    top_k: int = 5,
    doc_filter: str | None = None,
) -> str:
    """Search the indexed documentation (keyword + semantic, fused).

    Plain queries: all terms are required first; if that finds too few
    chunks the search widens to any-term and prefix matching, strict matches
    ranked first (the header line says which ran). Words are stemmed
    ('timing' matches 'timings'); identifiers split on '_' so 'OTG_GUSBCFG'
    and 'GUSBCFG' both work. FTS5 syntax ("exact phrase", OR, NOT, NEAR,
    prefix*) is honoured when used. A semantic channel also matches by
    meaning, so a plain-language question can work, but manual terminology
    ranks best.

    Each hit shows its section path, 1-based PDF pages, doc id, chunk id,
    matched terms and highlighted snippets. Several matching chunks of one
    section are merged into one hit.

    Args:
        query: Keywords or a question, e.g. 'OTG TX FIFO errata' or
            'HSE bypass startup'
        top_k: Number of results (1-25)
        doc_filter: Restrict to one document: its id from list_docs, or a
            fragment of its filename/title such as 'errata'
    """
    from .tools.search_docs import search_docs as _search_docs

    return await _call("search_docs", lambda **kw: _search_docs(get_search(), **kw),
                       query=query, top_k=top_k, doc_filter=doc_filter)


@mcp.tool()
async def find_register(
    name: str,
    peripheral: str | None = None,
) -> str:
    """Find a hardware register by name and return its full description.

    Case-insensitive, and the peripheral prefix is optional: 'GUSBCFG'
    finds OTG_GUSBCFG, 'bdcr' finds RCC_BDCR. An ambiguous short name
    ('CR') returns the candidate names to choose from. Returns address or
    offset, reset value and bitfields.

    Args:
        name: Register name, e.g. 'RCC_BDCR', 'FLASH_ACR', 'GUSBCFG'
        peripheral: Optional peripheral to disambiguate, e.g. 'RCC'
    """
    from .tools.find_register import find_register as _find

    return await _call("find_register", lambda **kw: _find(get_search(), **kw),
                       name=name, peripheral=peripheral)


@mcp.tool()
async def read_section(
    chunk_id: str,
    max_chars: int = 20000,
    offset: int = 0,
) -> str:
    """Read the whole section a search_docs hit came from.

    Stitches every chunk of that section together in document order, so a
    hit's snippet can be read in full context without guessing pages.

    Args:
        chunk_id: The chunk id shown in a search_docs result
        max_chars: Maximum characters to return
        offset: Character offset to continue a truncated section
    """
    from .tools.read_docs import read_section as _read_section

    return await _call("read_section", lambda **kw: _read_section(get_search(), **kw),
                       chunk_id=chunk_id, max_chars=max_chars, offset=offset)


@mcp.tool()
async def read_pages(
    doc: str,
    pages: str,
    max_chars: int = 20000,
) -> str:
    """Read the text of specific PDF pages of an indexed document.

    Page numbers are 1-based PDF pages, as shown in search_docs results.
    At most 10 pages per call. Read from the source PDF when it is on this
    machine, otherwise from the indexed text.

    Args:
        doc: Document id from list_docs, or a fragment of its filename/title
        pages: One page ('1897') or an inclusive range ('1896-1898')
        max_chars: Maximum characters to return
    """
    from .tools.read_docs import read_pages as _read_pages

    return await _call(
        "read_pages", lambda **kw: _read_pages(get_search(), get_config(), **kw),
        doc=doc, pages=pages, max_chars=max_chars)


@mcp.tool()
async def list_docs() -> str:
    """List indexed documents (with the ids used by doc_filter and read_pages)
    and PDFs in the configured doc directories that are not indexed yet."""
    from .tools.list_docs import list_docs as _list

    return await _call("list_docs", lambda: _list(get_config()))


@mcp.tool()
async def ingest_docs(
    doc_path: str,
    title: str | None = None,
    version: str | None = None,
    detect_tables: bool = True,
) -> str:
    """Ingest a PDF into the search index (or re-ingest it, replacing the old copy).

    Parses text and section structure, extracts register tables, chunks,
    embeds and indexes it. Takes minutes for a 2000-page manual; other
    tools keep working meanwhile.

    Args:
        doc_path: Path to the PDF, or just its filename as shown by list_docs
        title: Optional document title
        version: Optional document version
        detect_tables: Run table-based register detection (slowest phase;
            ST reference manuals get register data from section text anyway)
    """
    from .tools.ingest_docs import ingest_docs as _ingest

    def _run(**kw: Any) -> str:
        embedder = _search.embedder if _search is not None else None
        try:
            return _ingest(config=get_config(), embedder=embedder, **kw)
        finally:
            _reload_search()

    return await _call("ingest_docs", _run, doc_path=doc_path, title=title,
                       version=version, detect_tables=detect_tables)


@mcp.tool()
async def remove_docs(doc_id: str) -> str:
    """Remove a document (its chunks, registers and vectors) from the index.

    Args:
        doc_id: Document id to remove (see list_docs)
    """
    from .tools.remove_docs import remove_docs as _remove

    def _run(**kw: Any) -> str:
        try:
            return _remove(config=get_config(), **kw)
        finally:
            _reload_search()

    return await _call("remove_docs", _run, doc_id=doc_id)
