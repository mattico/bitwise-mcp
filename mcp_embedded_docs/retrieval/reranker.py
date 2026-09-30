"""Cross-encoder reranking of search candidates."""

import logging
import threading
from typing import List

logger = logging.getLogger(__name__)


class Reranker:
    """Scores (query, passage) pairs with a sentence-transformers CrossEncoder."""

    # Tokens per pair. Chunks run to ~2500 chars (~600 tokens); the section
    # prefix and opening sentences carry most of the relevance signal, and
    # cost grows with length on CPU.
    MAX_LENGTH = 512

    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        from ..indexing.embedder import model_kwargs

        self.model_name = model_name
        self._lock = threading.Lock()
        # Cache first, as in LocalEmbedder: no Hub round trip on every start.
        try:
            self.model = CrossEncoder(model_name, device=device, max_length=self.MAX_LENGTH,
                                      local_files_only=True, model_kwargs=model_kwargs(device))
        except Exception as exc:  # noqa: BLE001 - any cache miss falls back to download
            logger.info("model %s not in local cache (%s); downloading", model_name, exc)
            self.model = CrossEncoder(model_name, device=device, max_length=self.MAX_LENGTH,
                                      model_kwargs=model_kwargs(device))

    def score(self, query: str, passages: List[str]) -> List[float]:
        """Relevance of each passage to the query; higher is better."""
        if not passages:
            return []
        # Shared between search threads; see LocalEmbedder.embed_batch.
        with self._lock:
            scores = self.model.predict([(query, p) for p in passages],
                                        show_progress_bar=False, convert_to_numpy=True)
        return [float(s) for s in scores]
