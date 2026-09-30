"""Hybrid search combining keyword and semantic search."""

import logging
import sqlite3
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..config import Config
from ..indexing.metadata_store import KeywordResult, MetadataStore, section_path
from . import SearchResponse, SearchResult
from .query import terms_of

if TYPE_CHECKING:
    from ..indexing.embedder import LocalEmbedder
    from ..indexing.vector_store import VectorStore
    from .reranker import Reranker

logger = logging.getLogger(__name__)

# Reciprocal-rank-fusion constant; 60 is the customary value and keeps one
# channel's top hit from drowning out agreement between the two.
RRF_K = 60
# How long a search waits for the embedding model before answering from the
# keyword index alone. Loading starts at server start, so only a search in
# the first seconds (or a cold, slow disk) ever waits.
EMBEDDER_WAIT_SECONDS = 20.0


class HybridSearch:
    """Hybrid search engine combining keyword and semantic search."""

    def __init__(self, config: Config, load_embedder: bool = True):
        """Initialize hybrid search.

        Opens the keyword index and vector file immediately; the embedding
        model loads on a background thread (see ensure_embedder).

        Args:
            config: Configuration object
            load_embedder: Start loading the embedding model right away
        """
        self.config = config
        index_dir = config.index.directory

        t0 = time.perf_counter()
        self.metadata_store = MetadataStore(index_dir / config.index.metadata_db)
        logger.info("metadata store opened in %.2fs (path=%s)",
                    time.perf_counter() - t0, index_dir / config.index.metadata_db)

        self.vector_store: Optional["VectorStore"] = None
        self.embedder: Optional["LocalEmbedder"] = None
        self._embedder_error: Optional[str] = None
        self._embedder_ready = threading.Event()
        self._embedder_thread: Optional[threading.Thread] = None
        self._embed_lock = threading.Lock()
        self.reranker: Optional["Reranker"] = None
        self._reranker_error: Optional[str] = None
        self._reranker_ready = threading.Event()
        self._reranker_thread: Optional[threading.Thread] = None
        self._doc_titles: Dict[str, str] = {}
        self._reload_lock = threading.Lock()
        # What the loaded state was read from; see refresh_if_stale.
        self._vector_sig: Tuple[Any, ...] = ()
        self._data_version = -1
        self._vectors_status = ""
        self.semantic_status = ""
        self._auto_load_embedder = load_embedder

        self.reload()

    # ------------------------------------------------------------ lifecycle

    def _vector_signature(self) -> Tuple[Any, ...]:
        """(mtime, size) of the FAISS file and its id file; changes on every save."""
        vector_path = self.config.index.directory / self.config.index.vector_file
        sig: List[Any] = []
        for path in (vector_path, vector_path.with_suffix(".ids")):
            try:
                st = path.stat()
                sig.append((st.st_mtime_ns, st.st_size))
            except OSError:
                sig.append(None)
        return tuple(sig)

    def _load_vectors(self) -> str:
        """Load the FAISS file; returns 'on' or why semantic search is off."""
        # Taken before reading, so a save that lands mid-load is seen next time.
        self._vector_sig = self._vector_signature()
        if not self.config.embeddings.enabled:
            self.vector_store = None
            return "off (embeddings.enabled is false)"
        vector_path = self.config.index.directory / self.config.index.vector_file
        if not vector_path.exists():
            logger.warning("vector store not found at %s; semantic search disabled", vector_path)
            self.vector_store = None
            return "off (no vector index; run rebuild-vectors)"
        try:
            from ..indexing.vector_store import VectorStore

            t0 = time.perf_counter()
            store = VectorStore()
            store.load(vector_path)
            if store.model and store.model != self.config.embeddings.model:
                self.vector_store = None
                return (f"off (vectors are from {store.model} but embeddings.model is "
                        f"{self.config.embeddings.model}; run rebuild-vectors)")
            self.vector_store = store
            logger.info("vector store loaded in %.2fs (%d vectors)",
                        time.perf_counter() - t0, len(store))
            return "on"
        except Exception as exc:  # noqa: BLE001 - keyword search still works
            logger.exception("could not load vector store")
            self.vector_store = None
            return f"off (vector index unreadable: {exc})"

    def _refresh_metadata(self):
        """Doc titles, and how many chunks the loaded vectors leave out."""
        self._data_version = self.metadata_store.data_version()
        self._doc_titles = {
            d["id"]: d["title"] or d["filename"] for d in self.metadata_store.list_documents()
        }
        status = self._vectors_status
        store = self.vector_store
        if store is not None:
            chunk_ids = self.metadata_store.all_chunk_ids()
            have = set(store.ids)
            missing = sum(1 for cid in chunk_ids if cid not in have)
            if missing:
                status = (f"partial ({missing} of {len(chunk_ids)} chunks have no vector; "
                          "run rebuild-vectors)")
        self.semantic_status = status

    def reload(self):
        """Pick up changes written by an ingest, remove or rebuild-vectors."""
        with self._reload_lock:
            self._vectors_status = self._load_vectors()
            self._refresh_metadata()
        self._maybe_start_embedder()

    def refresh_if_stale(self):
        """Reload whatever another process changed since it was loaded.

        A CLI ingest, remove or rebuild-vectors writes the index files directly;
        this notices via the vector files' mtime/size and SQLite's data_version
        (which changes whenever another connection commits). Costs two stats
        and a pragma per call.
        """
        vectors_changed = self._vector_signature() != self._vector_sig
        if not vectors_changed and self.metadata_store.data_version() == self._data_version:
            return
        with self._reload_lock:
            # Another thread may have reloaded while this one waited.
            if self._vector_signature() != self._vector_sig:
                logger.info("vector index changed on disk; reloading")
                self._vectors_status = self._load_vectors()
            elif self.metadata_store.data_version() == self._data_version:
                return
            self._refresh_metadata()
        self._maybe_start_embedder()

    def _maybe_start_embedder(self):
        """Start loading the models in the background once there is something to query."""
        if self._auto_load_embedder and self.vector_store is not None:
            self.ensure_embedder(wait=0)
        if self._auto_load_embedder and self.config.search.rerank:
            self.ensure_reranker(wait=0)

    def ensure_embedder(self, wait: float = EMBEDDER_WAIT_SECONDS) -> bool:
        """Start loading the embedding model if needed; wait up to `wait` seconds.

        Returns:
            True when the model is ready
        """
        if self._embedder_ready.is_set():
            return self.embedder is not None
        with self._embed_lock:
            if self._embedder_thread is None:
                self._embedder_thread = threading.Thread(
                    target=self._load_embedder, name="embedder-load", daemon=True)
                self._embedder_thread.start()
        if wait:
            self._embedder_ready.wait(wait)
        return self.embedder is not None

    def _load_embedder(self):
        try:
            t0 = time.perf_counter()
            from ..indexing.embedder import LocalEmbedder

            self.embedder = LocalEmbedder(
                model_name=self.config.embeddings.model,
                device=self.config.embeddings.device,
                batch_size=self.config.embeddings.batch_size,
                max_seq_length=self.config.embeddings.max_seq_length,
                query_prefix=self.config.embeddings.query_prefix,
            )
            logger.info("embedder loaded in %.2fs (model=%s)",
                        time.perf_counter() - t0, self.config.embeddings.model)
        except Exception as exc:  # noqa: BLE001 - keyword search still works
            logger.exception("could not load embedding model")
            self._embedder_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._embedder_ready.set()

    def ensure_reranker(self, wait: float = EMBEDDER_WAIT_SECONDS) -> bool:
        """Start loading the cross-encoder if needed; wait up to `wait` seconds."""
        if self._reranker_ready.is_set():
            return self.reranker is not None
        with self._embed_lock:
            if self._reranker_thread is None:
                self._reranker_thread = threading.Thread(
                    target=self._load_reranker, name="reranker-load", daemon=True)
                self._reranker_thread.start()
        if wait:
            self._reranker_ready.wait(wait)
        return self.reranker is not None

    def _load_reranker(self):
        try:
            t0 = time.perf_counter()
            from .reranker import Reranker

            self.reranker = Reranker(self.config.search.rerank_model,
                                     device=self.config.embeddings.device)
            logger.info("reranker loaded in %.2fs (model=%s)",
                        time.perf_counter() - t0, self.config.search.rerank_model)
        except Exception as exc:  # noqa: BLE001 - fused order still works
            logger.exception("could not load reranker")
            self._reranker_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._reranker_ready.set()

    # ------------------------------------------------------------ search

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """Perform hybrid search.

        Args:
            query: Search query
            top_k: Number of results to return
            doc_filter: Optional document ID to filter results

        Returns:
            List of search results sorted by relevance
        """
        return self.search_ex(query, top_k, doc_filter).results

    def search_ex(self, query: str, top_k: int = 5,
                  doc_filter: Optional[str] = None) -> SearchResponse:
        """Hybrid search returning results plus how the query was run."""
        t_start = time.perf_counter()
        top_k = max(1, top_k)
        # Over-fetch: several chunks of one section collapse into one result.
        # A fixed floor keeps ranking independent of top_k for typical calls.
        pool = max(30, top_k * 3)
        response = SearchResponse(query=query, results=[])
        self.refresh_if_stale()

        t0 = time.perf_counter()
        try:
            kw = self.metadata_store.keyword_search_ex(query, pool, doc_filter)
        except sqlite3.Error as exc:
            logger.exception("keyword search failed")
            kw = KeywordResult(mode="error", terms=terms_of(query))
            response.notes.append(
                f"keyword index error ({exc}); results are semantic only. "
                "`mcp-embedded-docs rebuild-vectors` rebuilds the keyword index.")
        t_kw = time.perf_counter() - t0
        response.terms = kw.terms
        response.keyword_mode = kw.mode or "no match"
        if kw.note:
            response.notes.append(kw.note)

        t0 = time.perf_counter()
        semantic_ids, response.semantic = self._semantic_search(query, pool, doc_filter)
        t_sem = time.perf_counter() - t0

        if not kw.hits and semantic_ids:
            response.notes.append(
                "no chunk contains these terms; the hits below are nearest neighbours by "
                "meaning only and may be unrelated. Try the manual's own wording.")

        t0 = time.perf_counter()
        fused = self._fuse([cid for cid, _ in kw.hits], semantic_ids)
        rerank = self.config.search.rerank
        candidates = self._collapse(
            fused, max(top_k, self.config.search.rerank_depth) if rerank else top_k)
        t_fetch = time.perf_counter() - t0

        t0 = time.perf_counter()
        if rerank:
            candidates = self._rerank(query, candidates, response)
        response.results = candidates[:top_k]
        t_rerank = time.perf_counter() - t0

        logger.info(
            "search %r done in %.3fs (keyword=%.3fs/%d %s, semantic=%.3fs/%d %s, fetch=%.3fs, "
            "rerank=%.3fs)",
            query, time.perf_counter() - t_start, t_kw, len(kw.hits), response.keyword_mode,
            t_sem, len(semantic_ids), response.semantic, t_fetch, t_rerank,
        )
        return response

    def _rerank(self, query: str, candidates: List[SearchResult],
                response: SearchResponse) -> List[SearchResult]:
        """Order candidates by cross-encoder score; unchanged if the model is unavailable."""
        if len(candidates) < 2:
            return candidates
        if not self.ensure_reranker() or self.reranker is None:
            why = (f"failed to load: {self._reranker_error}" if self._reranker_error
                   else "still loading")
            response.notes.append(f"reranker {why}; results are in fused order.")
            return candidates
        try:
            scores = self.reranker.score(query, [r.text for r in candidates])
        except Exception as exc:  # noqa: BLE001 - fused order is still a ranking
            logger.exception("rerank failed")
            response.notes.append(f"reranker failed ({exc}); results are in fused order.")
            return candidates
        for r, s in zip(candidates, scores):
            r.score = s
        # sorted() is stable: ties keep their fused order.
        return sorted(candidates, key=lambda r: r.score, reverse=True)

    def _semantic_search(self, query: str, top_k: int,
                         doc_filter: Optional[str]) -> Tuple[List[str], str]:
        """Chunk ids by embedding similarity, and the semantic channel's status."""
        store = self.vector_store
        if store is None:
            return [], self.semantic_status
        if not self.ensure_embedder() or self.embedder is None:
            if self._embedder_error:
                return [], f"off (model failed to load: {self._embedder_error})"
            return [], "skipped (model still loading; keyword results only)"
        if store.dimension != self.embedder.dimension:
            # An index from before the model was recorded, built with another model.
            return [], (f"off (vectors have dimension {store.dimension} but "
                        f"{self.embedder.model_name} produces {self.embedder.dimension}; "
                        "run rebuild-vectors)")
        try:
            vector = self.embedder.embed_query(query)  # thread-safe
            prefix = f"{doc_filter}_" if doc_filter else None
            ids = [cid for cid, _ in store.search(vector, top_k, id_prefix=prefix)]
            return ids, self.semantic_status
        except Exception as exc:  # noqa: BLE001 - keyword results still useful
            logger.exception("semantic search failed")
            return [], f"failed ({exc})"

    def _fuse(self, keyword_ids: List[str], semantic_ids: List[str]) -> List[Tuple[str, float, List[str]]]:
        """Weighted reciprocal-rank fusion: [(chunk_id, score, channels)], best first."""
        weights = (("keyword", keyword_ids, self.config.search.keyword_weight),
                   ("semantic", semantic_ids, self.config.search.semantic_weight))
        scores: Dict[str, float] = {}
        channels: Dict[str, List[str]] = {}
        for name, ids, weight in weights:
            for rank, cid in enumerate(ids, 1):
                scores[cid] = scores.get(cid, 0.0) + weight / (RRF_K + rank)
                channels.setdefault(cid, []).append(name)
        ordered = sorted(scores, key=lambda c: scores[c], reverse=True)
        return [(cid, scores[cid], channels[cid]) for cid in ordered]

    def _collapse(self, fused: List[Tuple[str, float, List[str]]], top_k: int) -> List[SearchResult]:
        """Materialize results, merging chunks of the same section into the best one."""
        chunks = self.metadata_store.get_chunks([cid for cid, _, _ in fused])
        results: List[SearchResult] = []
        by_section: Dict[Tuple[str, str], SearchResult] = {}
        for cid, score, chans in fused:
            chunk = chunks.get(cid)
            if chunk is None:  # a vector whose chunk was removed
                continue
            section = section_path(chunk["text"]) or chunk.get("section_hierarchy")
            key = (chunk["doc_id"], section) if section else (chunk["doc_id"], cid)
            if key in by_section:
                best = by_section[key]
                best.more_in_section += 1
                for ch in chans:
                    if ch not in best.channels:
                        best.channels.append(ch)
                continue
            if len(results) >= top_k:
                continue  # still counting collapsed siblings of kept results
            result = self._to_result(chunk, score)
            result.channels = list(chans)
            by_section[key] = result
            results.append(result)
        return results

    def _to_result(self, chunk: Dict[str, Any], score: float) -> SearchResult:
        return SearchResult(
            chunk_id=chunk["id"],
            score=score,
            text=chunk["text"],
            structured_data=chunk.get("structured_data"),
            metadata=chunk.get("metadata") or {},
            doc_id=chunk["doc_id"],
            page_start=chunk["page_start"],
            page_end=chunk["page_end"],
            chunk_type=chunk.get("chunk_type") or "text",
            section=chunk.get("section_hierarchy"),
            doc_title=self._doc_titles.get(chunk["doc_id"]),
        )

    # ------------------------------------------------------------ lookups

    def resolve_doc(self, doc: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """Map a doc id, filename or title fragment to a doc id.

        Returns:
            (doc_id, None) on success, (None, error message) otherwise
        """
        if not doc:
            return None, None
        docs = self.metadata_store.list_documents()
        needle = doc.strip().lower()
        for d in docs:
            if d["id"] == doc.strip():
                return d["id"], None
        matches = [d for d in docs
                   if needle in d["filename"].lower() or needle in (d["title"] or "").lower()
                   or d["id"].startswith(needle)]
        if len(matches) == 1:
            return matches[0]["id"], None
        listing = ", ".join(f"`{d['id']}` ({d['filename']})" for d in (matches or docs))
        if matches:
            return None, f"'{doc}' matches several documents: {listing}"
        return None, f"no indexed document matches '{doc}'. Indexed: {listing}"

    def find_register(
        self,
        name: str,
        peripheral: Optional[str] = None
    ) -> Optional[SearchResult]:
        """Find a specific register by name.

        Args:
            name: Register name
            peripheral: Optional peripheral name to filter

        Returns:
            Search result containing the register or None
        """
        matches = self.find_register_ex(name, peripheral)
        return matches["results"][0] if matches["kind"] == "exact" else None

    def find_register_ex(self, name: str, peripheral: Optional[str] = None) -> Dict[str, Any]:
        """Register lookup tolerant of case and missing prefixes.

        Returns:
            {"kind": "exact"|"ambiguous"|"none", "name", "results", "candidates"}
        """
        self.refresh_if_stale()
        m = self.metadata_store.find_register_matches(name, peripheral)
        return {
            "kind": m["kind"],
            "name": m["name"],
            "results": [self._to_result(c, 1.0) for c in m["chunks"]],
            "candidates": m["candidates"],
        }

    def list_documents(self) -> List[Dict]:
        """List all indexed documents.

        Returns:
            List of document information
        """
        return self.metadata_store.list_documents()

    def close(self):
        """Close connections."""
        self.metadata_store.close()
