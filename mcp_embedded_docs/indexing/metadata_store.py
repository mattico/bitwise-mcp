"""SQLite metadata store with full-text search."""

import json
import logging
import re
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from ..retrieval import query as fts_query

logger = logging.getLogger(__name__)

# Porter-stemmed so 'timing' matches 'timings'; unicode61 splits identifiers on
# '_', '-', '.', so 'OTG_GUSBCFG' indexes as otg + gusbcfg. The section title is
# its own column so bm25 can weight it (see KEYWORD_BM25_WEIGHTS). Changing
# either needs an FTS rebuild, which _migrate_fts does automatically.
FTS_TOKENIZE = "porter unicode61"
FTS_COLUMNS = ("section_hierarchy", "text")
KEYWORD_BM25_WEIGHTS = (5.0, 1.0)

_CHUNK_COLUMNS = (
    "id, doc_id, chunk_type, section_hierarchy, page_start, page_end, "
    "text, structured_data, metadata"
)


@dataclass
class KeywordResult:
    """Keyword search hits plus a description of how the query was run."""
    hits: List[Tuple[str, float]] = field(default_factory=list)
    mode: str = ""
    note: str = ""
    terms: List[str] = field(default_factory=list)


class MetadataStore:
    """SQLite database for chunk metadata and keyword search."""

    def __init__(self, db_path: Path):
        """Initialize metadata store.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # The MCP server runs tool calls on worker threads; `lock` serializes
        # access to this one connection.
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self._txn_depth = 0
        self._create_schema()

    @contextmanager
    def write_transaction(self) -> Iterator["MetadataStore"]:
        """Group writes into one transaction holding SQLite's write lock.

        The lock is exclusive across processes, so it also serializes other
        read-modify-write steps done inside it (the ingest pipeline updates the
        FAISS file here), e.g. a CLI ingest racing a server-side remove.
        Methods called inside commit nothing; everything commits on exit, or
        rolls back if the block raises.
        """
        with self.lock:
            outer = self._txn_depth == 0
            if outer:
                self.conn.execute("BEGIN IMMEDIATE")
            self._txn_depth += 1
            try:
                yield self
            except BaseException:
                self._txn_depth -= 1
                if outer:
                    self.conn.rollback()
                raise
            self._txn_depth -= 1
            if outer:
                self.conn.commit()

    def _commit(self):
        if self._txn_depth == 0:
            self.conn.commit()

    def _rollback(self):
        # Inside write_transaction the outermost block decides.
        if self._txn_depth == 0:
            self.conn.rollback()

    def _create_schema(self):
        """Create database schema, migrating older layouts in place."""
        cursor = self.conn.cursor()
        # WAL lets several MCP server processes read while an ingest writes.
        cursor.execute("PRAGMA journal_mode=WAL")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                title TEXT,
                version TEXT,
                index_date TEXT NOT NULL,
                path TEXT
            )
        """)
        # `path` was added with read_pages; older DBs lack it.
        doc_cols = {r["name"] for r in cursor.execute("PRAGMA table_info(documents)")}
        if "path" not in doc_cols:
            cursor.execute("ALTER TABLE documents ADD COLUMN path TEXT")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                section_hierarchy TEXT,
                page_start INTEGER,
                page_end INTEGER,
                text TEXT NOT NULL,
                structured_data TEXT,
                metadata TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS registers (
                name TEXT NOT NULL,
                peripheral TEXT,
                address TEXT,
                offset TEXT,
                chunk_id TEXT NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_registers_name ON registers(name COLLATE NOCASE)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_registers_peripheral ON registers(peripheral)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_registers_chunk ON registers(chunk_id)")
        self.conn.commit()

        self._migrate_fts()

    def _fts_is_current(self) -> bool:
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        ).fetchone()
        if not row:
            return False
        sql = row["sql"]
        return FTS_TOKENIZE in sql and "section_hierarchy" in sql

    def _migrate_fts(self):
        """(Re)create the FTS table and its triggers when missing or outdated.

        Pre-0.4 indexes used the default tokenizer, had no title column, and kept
        FTS in sync with a plain `DELETE FROM chunks_fts`, which an external-content
        table does not support -- that is what left stale rowids behind. Rebuilding
        from `chunks` takes well under a second for ~10k chunks.
        """
        if self._fts_is_current() and self._triggers_are_current():
            return
        cursor = self.conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        try:
            # Another process may have migrated while we waited for the lock.
            if not (self._fts_is_current() and self._triggers_are_current()):
                has_chunks = self.conn.execute("SELECT 1 FROM chunks LIMIT 1").fetchone()
                if has_chunks:
                    logger.info("Migrating FTS index to %s with title column", FTS_TOKENIZE)
                for trig in ("chunks_ai", "chunks_ad", "chunks_au"):
                    cursor.execute(f"DROP TRIGGER IF EXISTS {trig}")
                cursor.execute("DROP TABLE IF EXISTS chunks_fts")
                cursor.execute(f"""
                    CREATE VIRTUAL TABLE chunks_fts USING fts5(
                        section_hierarchy,
                        text,
                        content='chunks',
                        content_rowid='rowid',
                        tokenize='{FTS_TOKENIZE}'
                    )
                """)
                cursor.execute("""
                    CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
                        INSERT INTO chunks_fts(rowid, section_hierarchy, text)
                        VALUES (new.rowid, new.section_hierarchy, new.text);
                    END
                """)
                cursor.execute("""
                    CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
                        INSERT INTO chunks_fts(chunks_fts, rowid, section_hierarchy, text)
                        VALUES ('delete', old.rowid, old.section_hierarchy, old.text);
                    END
                """)
                cursor.execute("""
                    CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
                        INSERT INTO chunks_fts(chunks_fts, rowid, section_hierarchy, text)
                        VALUES ('delete', old.rowid, old.section_hierarchy, old.text);
                        INSERT INTO chunks_fts(rowid, section_hierarchy, text)
                        VALUES (new.rowid, new.section_hierarchy, new.text);
                    END
                """)
                cursor.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def _triggers_are_current(self) -> bool:
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND name='chunks_ad'"
        ).fetchone()
        return bool(row) and "'delete'" in row["sql"]

    def rebuild_fts(self):
        """Rebuild the keyword index from the chunks table."""
        with self.lock:
            self.conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
            self._commit()

    def add_document(self, doc_id: str, filename: str, title: Optional[str] = None,
                     version: Optional[str] = None, path: Optional[str] = None):
        """Add or update a document record.

        Args:
            doc_id: Document identifier
            filename: Document filename
            title: Document title
            version: Document version
            path: Absolute path of the source file, used by read_pages
        """
        with self.lock:
            self.conn.execute("""
                INSERT INTO documents (id, filename, title, version, index_date, path)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    filename = excluded.filename, title = excluded.title,
                    version = excluded.version, index_date = excluded.index_date,
                    path = excluded.path
            """, (doc_id, filename, title, version, datetime.now().isoformat(), path))
            self._commit()

    def add_chunk(self, chunk_id: str, doc_id: str, chunk_type: str, text: str,
                  page_start: int, page_end: int,
                  structured_data: Optional[Dict[str, Any]] = None,
                  metadata: Optional[Dict[str, Any]] = None):
        """Add (or replace) a single chunk. Prefer add_chunks for bulk loads."""
        self.add_chunks([{
            "chunk_id": chunk_id, "doc_id": doc_id, "chunk_type": chunk_type,
            "text": text, "page_start": page_start, "page_end": page_end,
            "structured_data": structured_data, "metadata": metadata,
        }])

    def add_chunks(self, chunks: Iterable[Dict[str, Any]]) -> int:
        """Add (or replace) chunks in a single transaction.

        Each item has the keyword arguments of add_chunk. A chunk whose id already
        exists is deleted first (with its register rows) so the FTS delete trigger
        fires -- INSERT OR REPLACE skips delete triggers and corrupts the index.

        Returns:
            Number of chunks written
        """
        count = 0
        with self.lock:
            cursor = self.conn.cursor()
            try:
                for c in chunks:
                    chunk_id = c["chunk_id"]
                    metadata = c.get("metadata")
                    structured_data = c.get("structured_data")
                    section_hierarchy = (metadata or {}).get("section_title")

                    cursor.execute("DELETE FROM registers WHERE chunk_id = ?", (chunk_id,))
                    cursor.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
                    cursor.execute(f"""
                        INSERT INTO chunks ({_CHUNK_COLUMNS})
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        chunk_id, c["doc_id"], c["chunk_type"], section_hierarchy,
                        c["page_start"], c["page_end"], c["text"],
                        json.dumps(structured_data) if structured_data else None,
                        json.dumps(metadata) if metadata else None,
                    ))

                    if structured_data and "registers" in structured_data:
                        peripheral = structured_data.get("peripheral", "Unknown")
                        seen = set()
                        for register in structured_data["registers"]:
                            if register["name"] in seen:
                                continue
                            seen.add(register["name"])
                            cursor.execute("""
                                INSERT INTO registers (name, peripheral, address, offset, chunk_id)
                                VALUES (?, ?, ?, ?, ?)
                            """, (register["name"], peripheral, register.get("address"),
                                  register.get("offset"), chunk_id))
                    count += 1
                self._commit()
            except Exception:
                self._rollback()
                raise
        return count

    def keyword_search(self, query: str, top_k: int = 10, doc_filter: Optional[str] = None) -> List[Tuple[str, float]]:
        """Search using keyword matching (FTS5).

        Args:
            query: Search query
            top_k: Number of results
            doc_filter: Optional document ID to filter by

        Returns:
            List of (chunk_id, score) tuples, best first
        """
        return self.keyword_search_ex(query, top_k, doc_filter).hits

    def keyword_search_ex(self, query: str, top_k: int = 10,
                          doc_filter: Optional[str] = None, min_hits: int = 3) -> KeywordResult:
        """Keyword search with progressive relaxation (see retrieval/query.py).

        Plain queries are AND-ed, then widened to a ranked OR and a prefix OR when
        the strict pass finds fewer than `min_hits` chunks; strict hits keep their
        place on top. Queries using FTS5 syntax run verbatim. Scores are
        positive, higher is better, and only comparable within one call.
        """
        sql = (
            "SELECT c.id, bm25(chunks_fts, {w}) AS score FROM chunks_fts "
            "JOIN chunks c ON c.rowid = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ?{f} ORDER BY score LIMIT ?"
        ).format(
            w=", ".join(str(x) for x in KEYWORD_BM25_WEIGHTS),
            f=" AND c.doc_id = ?" if doc_filter else "",
        )
        extra: Tuple[Any, ...] = (doc_filter,) if doc_filter else ()

        def run(match: str, limit: int) -> List[Tuple[str, float]]:
            try:
                rows = self.conn.execute(sql, (match, *extra, limit)).fetchall()
            except sqlite3.OperationalError as exc:
                # Only query-text problems are the caller's to rephrase; a locked
                # or corrupt index must surface, not read as "no match".
                if fts_query.is_syntax_error(str(exc)):
                    raise fts_query.SearchSyntaxError(str(exc)) from exc
                raise
            # bm25() is negative; flip so higher is better.
            return [(r["id"], -float(r["score"])) for r in rows]

        with self.lock:
            plan = fts_query.run_plans(query, top_k, min_hits, run)
            result = KeywordResult(hits=plan.hits, mode=plan.mode, note=plan.note,
                                   terms=plan.terms)

            # Last resort for literal tokens FTS cannot see as one term (odd
            # punctuation inside addresses and part numbers): a substring scan
            # requiring every content term.
            terms = fts_query.substring_terms(query)
            if not result.hits and terms:
                result.hits = self._literal_fallback(terms, top_k, doc_filter)
                if result.hits:
                    result.mode = (result.mode + "+" if result.mode else "") + "substring"
        return result

    def _literal_fallback(self, tokens: Sequence[str], top_k: int,
                          doc_filter: Optional[str]) -> List[Tuple[str, float]]:
        clauses = " AND ".join("text LIKE ? ESCAPE '\\'" for _ in tokens)
        params: List[Any] = [f"%{_like_escape(t)}%" for t in tokens]
        sql = f"SELECT id, text FROM chunks WHERE {clauses}"
        if doc_filter:
            sql += " AND doc_id = ?"
            params.append(doc_filter)
        # Every row contains every token, so there is nothing to rank by;
        # document order at least keeps the result stable.
        sql += " ORDER BY rowid LIMIT ?"
        params.append(top_k)
        rows = self.conn.execute(sql, params).fetchall()
        return [(r["id"], 1.0) for r in rows]

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk by ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Chunk data as dictionary or None
        """
        with self.lock:
            row = self.conn.execute(
                f"SELECT {_CHUNK_COLUMNS} FROM chunks WHERE id = ?", (chunk_id,)
            ).fetchone()
        return _row_to_chunk(row) if row else None

    def get_chunks(self, chunk_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch several chunks at once, keyed by id (missing ids are omitted)."""
        out: Dict[str, Dict[str, Any]] = {}
        ids = list(dict.fromkeys(chunk_ids))
        with self.lock:
            for i in range(0, len(ids), 500):
                batch = ids[i:i + 500]
                rows = self.conn.execute(
                    f"SELECT {_CHUNK_COLUMNS} FROM chunks WHERE id IN ({','.join('?' * len(batch))})",
                    batch,
                ).fetchall()
                for row in rows:
                    out[row["id"]] = _row_to_chunk(row)
        return out

    def get_section_chunks(self, chunk_id: str) -> List[Dict[str, Any]]:
        """All chunks sharing `chunk_id`'s document and section, in document order."""
        with self.lock:
            target = self.conn.execute(
                f"SELECT {_CHUNK_COLUMNS} FROM chunks WHERE id = ?", (chunk_id,)
            ).fetchone()
            if not target:
                return []
            if target["section_hierarchy"] is None:
                return [_row_to_chunk(target)]
            path = section_path(target["text"])
            rows = self.conn.execute(
                f"SELECT {_CHUNK_COLUMNS} FROM chunks "
                "WHERE doc_id = ? AND section_hierarchy = ? ORDER BY rowid",
                (target["doc_id"], target["section_hierarchy"]),
            ).fetchall()
        return [_row_to_chunk(r) for r in rows if section_path(r["text"]) == path]

    def get_page_chunks(self, doc_id: str, first: int, last: int) -> List[Dict[str, Any]]:
        """Chunks of `doc_id` whose page range overlaps [first, last], in document order."""
        with self.lock:
            rows = self.conn.execute(
                f"SELECT {_CHUNK_COLUMNS} FROM chunks "
                "WHERE doc_id = ? AND page_start <= ? AND page_end >= ? ORDER BY rowid",
                (doc_id, last, first),
            ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def find_register(self, name: str, peripheral: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Find a register by exact name (case-insensitive).

        Args:
            name: Register name
            peripheral: Optional peripheral name to filter

        Returns:
            Chunk containing the register or None
        """
        matches = self.find_register_matches(name, peripheral)
        if matches["kind"] != "exact":
            return None
        return matches["chunks"][0]

    def find_register_matches(self, name: str, peripheral: Optional[str] = None,
                              max_candidates: int = 25) -> Dict[str, Any]:
        """Resolve a register name, tolerating case, missing prefixes and instances.

        Candidates come from the parsed register tables and from register
        sections titled '... register (NAME)' (most register descriptions are
        prose sections, not parsed tables). A candidate whose name equals `name`
        wins; otherwise names ending in '_<name>' ('GUSBCFG' -> 'OTG_GUSBCFG');
        otherwise a generic name matching an instance ('GPIOA_IDR' ->
        'GPIOx_IDR'). Several distinct names are reported as ambiguous.

        Returns:
            {"kind": "exact"|"ambiguous"|"none", "name": resolved name,
             "chunks": [chunk dicts, best first], "candidates": [names]}
        """
        name = name.strip()
        key = name.upper()
        with self.lock:
            cands = self._register_candidates(name, peripheral)
            exact = [c for c in cands if c.upper() == key]
            if exact:
                chosen = exact
            else:
                chosen = [c for c in cands if c.upper().endswith("_" + key)]
                if not chosen and "_" in name:
                    cands = self._register_candidates(name.rsplit("_", 1)[1], peripheral)
                    chosen = [c for c in cands
                              if "x" in c and _instance_pattern(c).fullmatch(key)]
            names = list(dict.fromkeys(chosen))
            if len(names) > 1:
                return {"kind": "ambiguous", "name": name, "chunks": [],
                        "candidates": names[:max_candidates]}
            if names:
                chunk = self._get_chunk_unlocked(cands[names[0]])
                if chunk:
                    return {"kind": "exact", "name": names[0], "chunks": [chunk],
                            "candidates": []}
        return {"kind": "none", "name": name, "chunks": [], "candidates": []}

    def _register_candidates(self, name: str, peripheral: Optional[str]) -> Dict[str, str]:
        """{register name: best chunk id} for names equal to or ending in `name`.

        Names are deduplicated case-insensitively. For each name the richest
        chunk wins: a parsed bitfield definition, then a register map, then
        the prose section.
        """
        found: Dict[str, Tuple[int, int, str]] = {}  # UPPER -> (rank, rowid, chunk id)
        spelled: Dict[str, str] = {}

        def add(reg: str, rank: int, rowid: int, chunk_id: str, periph_col: str = "") -> None:
            if peripheral and not _peripheral_matches(peripheral, reg, periph_col):
                return
            up = reg.upper()
            if up not in found or (rank, rowid) < found[up][:2]:
                found[up] = (rank, rowid, chunk_id)
                spelled.setdefault(up, reg)

        rank_of = {"bitfield_definition": 0, "register_map": 1}
        for r in self.conn.execute("""
            SELECT r.name, r.peripheral, r.chunk_id, c.chunk_type, c.rowid FROM registers r
            JOIN chunks c ON c.id = r.chunk_id
            WHERE r.name = ? COLLATE NOCASE OR r.name LIKE ? ESCAPE '\\'
        """, (name, "%\\_" + _like_escape(name))):
            add(r["name"], rank_of.get(r["chunk_type"], 2), r["rowid"], r["chunk_id"],
                r["peripheral"] or "")

        # Register sections: '... register (RCC_BDCR)', '... (GPIOx_IDR) (x = A to K)'.
        for r in self.conn.execute("""
            SELECT c.id, c.section_hierarchy, c.rowid FROM chunks c
            WHERE c.section_hierarchy LIKE ? ESCAPE '\\'
            ORDER BY c.rowid
        """, ("%" + _like_escape(name) + ")%",)):
            for group in re.findall(r"\(([^()]*)\)", r["section_hierarchy"]):
                reg = group.strip()
                up = reg.upper()
                if re.fullmatch(r"[A-Za-z0-9_]+", reg) and (
                        up == name.upper() or up.endswith("_" + name.upper())):
                    add(reg, 3, r["rowid"], r["id"])

        return {spelled[up]: v[2] for up, v in sorted(found.items(), key=lambda kv: kv[1][:2])}

    def _get_chunk_unlocked(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            f"SELECT {_CHUNK_COLUMNS} FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        return _row_to_chunk(row) if row else None

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all indexed documents.

        Returns:
            List of document info dictionaries
        """
        with self.lock:
            rows = self.conn.execute("SELECT * FROM documents ORDER BY index_date DESC").fetchall()
        return [
            {
                "id": row["id"],
                "filename": row["filename"],
                "title": row["title"],
                "version": row["version"],
                "index_date": row["index_date"],
                "path": row["path"],
            }
            for row in rows
        ]

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get one document record by id."""
        for doc in self.list_documents():
            if doc["id"] == doc_id:
                return doc
        return None

    def get_document_stats(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a document.

        Args:
            doc_id: Document identifier

        Returns:
            Dictionary with chunk count and table count, or None if document not found
        """
        with self.lock:
            chunk_count = self.conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)
            ).fetchone()[0]
            table_count = self.conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE doc_id = ? AND structured_data IS NOT NULL",
                (doc_id,),
            ).fetchone()[0]

        return {
            "chunks": chunk_count,
            "tables": table_count
        }

    def get_chunk_ids(self, doc_id: str) -> List[str]:
        """All chunk ids belonging to a document."""
        with self.lock:
            return [r[0] for r in self.conn.execute(
                "SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))]

    def data_version(self) -> int:
        """SQLite's data_version: changes whenever another connection commits."""
        with self.lock:
            return int(self.conn.execute("PRAGMA data_version").fetchone()[0])

    def all_chunk_ids(self) -> List[str]:
        """Every chunk id in the index."""
        with self.lock:
            return [r[0] for r in self.conn.execute("SELECT id FROM chunks")]

    def delete_document_chunks(self, doc_id: str) -> List[str]:
        """Delete a document's chunks and register rows, keeping the document row.

        Returns:
            The ids of the deleted chunks (for removing their vectors)
        """
        with self.lock:
            try:
                ids = [r[0] for r in self.conn.execute(
                    "SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))]
                self.conn.execute("""
                    DELETE FROM registers
                    WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?)
                """, (doc_id,))
                # The chunks_ad trigger keeps the FTS index in sync.
                self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                self._commit()
            except Exception:
                self._rollback()
                raise
        return ids

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks.

        Args:
            doc_id: Document identifier to delete

        Returns:
            True if document was deleted, False if not found
        """
        with self.write_transaction():
            if not self.conn.execute(
                "SELECT id FROM documents WHERE id = ?", (doc_id,)
            ).fetchone():
                return False
            self.delete_document_chunks(doc_id)
            self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        return True

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def _peripheral_matches(peripheral: str, reg: str, periph_col: str = "") -> bool:
    """Whether a register belongs to `peripheral`, matched at a name boundary:
    'SPI' matches SPI_CR1 and SPIx_CR1 but not QUADSPI_CR; 'DMA' matches
    DMA_SxCR but not DMA2D_CR."""
    p = re.escape(peripheral.strip().upper())
    if re.match(rf"{p}(?:X|\d+)?_", reg.upper()):
        return True
    return bool(periph_col) and bool(re.match(rf"{p}(?![A-Z0-9])", periph_col.upper()))


def _instance_pattern(generic: str) -> "re.Pattern[str]":
    """'GPIOx_IDR' -> regex matching 'GPIOA_IDR', 'GPIOK_IDR' (upper-cased input);
    a lowercase 'x' in a register name stands for an instance letter or number."""
    parts = [re.escape(part.upper()) for part in generic.split("x")]
    return re.compile("[A-Z0-9]{1,2}".join(parts))


def section_path(text: str) -> Optional[str]:
    """The '[Doc > ... > Leaf]' line a chunk starts with, which identifies its
    section. The stored section_hierarchy is only the leaf title, and leaf
    titles repeat ('Signals', 'Overview' under many parents)."""
    # The prefix ends with ']\n' (see formatter.split_prefix); a bare ']' can
    # occur inside a title ('Bits [31:0] config').
    if text.startswith("["):
        end = text.find("]\n")
        if end == -1 and text.endswith("]"):
            end = len(text) - 1
        if end != -1:
            return text[:end + 1]
    return None


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _row_to_chunk(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "doc_id": row["doc_id"],
        "chunk_type": row["chunk_type"],
        "section_hierarchy": row["section_hierarchy"],
        "page_start": row["page_start"],
        "page_end": row["page_end"],
        "text": row["text"],
        "structured_data": json.loads(row["structured_data"]) if row["structured_data"] else None,
        "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
    }
