"""FAISS vector store for similarity search."""

import hashlib
import os
import pickle
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import faiss
import numpy as np


class VectorStore:
    """FAISS-based vector storage for semantic search."""

    def __init__(self, dimension: int = 384, model: Optional[str] = None):
        """Initialize vector store.

        Args:
            dimension: Dimension of embedding vectors
            model: Embedding model the vectors come from (saved with them)
        """
        self.dimension = dimension
        # None for files saved before the model was recorded.
        self.model = model
        # Use L2 distance (with normalized embeddings, equivalent to cosine similarity)
        self.index = faiss.IndexFlatL2(dimension)
        self.ids: List[str] = []  # Map from FAISS index position to chunk ID

    def add_vectors(self, vectors: np.ndarray, ids: List[str]):
        """Add embedding vectors to the index.

        Args:
            vectors: Array of shape (n, dimension)
            ids: List of chunk IDs corresponding to vectors
        """
        if len(ids) != len(vectors):
            raise ValueError("Number of IDs must match number of vectors")
        if not ids:
            return

        # Ensure vectors are float32 and contiguous
        vectors = np.ascontiguousarray(vectors.astype(np.float32))

        # Add to FAISS index
        self.index.add(vectors)

        # Store IDs
        self.ids.extend(ids)

    def remove_ids(self, ids: Iterable[str]) -> int:
        """Drop every vector whose chunk id is in `ids`.

        Returns:
            Number of vectors removed
        """
        drop = set(ids)
        if not drop or not self.ids:
            return 0
        keep = [i for i, cid in enumerate(self.ids) if cid not in drop]
        removed = len(self.ids) - len(keep)
        if removed:
            self._keep_positions(keep)
        return removed

    def dedupe(self) -> int:
        """Keep only the newest vector per chunk id (re-ingests used to append).

        Returns:
            Number of vectors removed
        """
        last = {cid: i for i, cid in enumerate(self.ids)}
        keep = sorted(last.values())
        removed = len(self.ids) - len(keep)
        if removed:
            self._keep_positions(keep)
        return removed

    def _keep_positions(self, keep: List[int]):
        vectors = self.index.reconstruct_n(0, self.index.ntotal)
        index = faiss.IndexFlatL2(self.dimension)
        if keep:
            index.add(np.ascontiguousarray(vectors[keep]))
        self.index = index
        self.ids = [self.ids[i] for i in keep]

    def search(self, query_vector: np.ndarray, top_k: int = 10,
               id_prefix: Optional[str] = None) -> List[Tuple[str, float]]:
        """Search for similar vectors.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            id_prefix: Only return chunk ids starting with this prefix (chunk
                ids are '{doc_id}_{hash}', so this filters by document)

        Returns:
            List of (chunk_id, distance) tuples, sorted by similarity. A chunk id
            appears at most once.
        """
        if self.index.ntotal == 0 or top_k <= 0:
            return []
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)

        # Ensure query is float32 and contiguous
        query_vector = np.ascontiguousarray(query_vector.astype(np.float32))

        # A flat index scores every vector anyway, so a filtered search simply
        # asks for all of them; ~10k x 768 floats is a few milliseconds.
        k = self.index.ntotal if id_prefix else min(self.index.ntotal, top_k * 2)
        distances, indices = self.index.search(query_vector, k)

        results: List[Tuple[str, float]] = []
        seen = set()
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.ids):
                continue
            cid = self.ids[idx]
            if cid in seen or (id_prefix and not cid.startswith(id_prefix)):
                continue
            seen.add(cid)
            results.append((cid, float(dist)))
            if len(results) >= top_k:
                break

        return results

    def save(self, filepath: Path):
        """Save index to disk.

        The index goes to `filepath` in FAISS's own format and the chunk ids to
        `<stem>.ids`, together with a digest of the index bytes: the two files
        cannot be replaced atomically as a pair, so load() checks they belong
        together. Each file is written to a temporary and swapped in.

        Args:
            filepath: Path to save the index
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        id_file = filepath.with_suffix('.ids')
        data = faiss.serialize_index(self.index).tobytes()
        meta = {"ids": self.ids, "ntotal": self.index.ntotal, "digest": _digest(data),
                "model": self.model}

        tmp_index = filepath.with_name(filepath.name + ".tmp")
        tmp_ids = id_file.with_name(id_file.name + ".tmp")
        tmp_index.write_bytes(data)
        with open(tmp_ids, 'wb') as f:
            pickle.dump(meta, f)
        _replace(tmp_index, filepath)
        _replace(tmp_ids, id_file)

    def load(self, filepath: Path, wait: float = 5.0):
        """Load index from disk.

        Retries for up to `wait` seconds while the index and id files do not
        match (another process is between its two replaces).

        Args:
            filepath: Path to the saved index
            wait: Seconds to keep retrying on a mismatched pair

        Raises:
            ValueError: the files still do not match after `wait`
        """
        id_file = filepath.with_suffix('.ids')
        deadline = time.monotonic() + wait
        while True:
            data = filepath.read_bytes()
            with open(id_file, 'rb') as f:
                meta = pickle.load(f)
            if isinstance(meta, list):  # pre-0.4 files carry no digest
                ids, ok, model = meta, True, None
            else:
                ids, ok = meta["ids"], meta.get("digest") == _digest(data)
                model = meta.get("model")
            if ok:
                break
            if time.monotonic() > deadline:
                raise ValueError(f"{filepath.name} and {id_file.name} do not belong together; "
                                 "run `mcp-embedded-docs rebuild-vectors`")
            time.sleep(0.1)

        index = faiss.deserialize_index(np.frombuffer(data, dtype=np.uint8))
        if index.ntotal != len(ids):
            raise ValueError(f"{filepath.name} holds {index.ntotal} vectors but "
                             f"{id_file.name} lists {len(ids)} ids; run `mcp-embedded-docs rebuild-vectors`")
        self.index = index
        self.dimension = index.d
        self.ids = list(ids)
        self.model = model

    @property
    def size(self) -> int:
        """Get number of vectors in the index."""
        return self.index.ntotal

    def __len__(self) -> int:
        """Get number of vectors in the index."""
        return self.size


def _digest(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _replace(src: Path, dst: Path, wait: float = 10.0):
    """os.replace, retried while Windows refuses because a reader has `dst` open."""
    deadline = time.monotonic() + wait
    while True:
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)
