"""Text embedding using sentence-transformers."""

import logging
import threading
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """Wrapper for sentence-transformers embeddings."""

    # Texts encoded per lock acquisition in embed_batch.
    LOCK_SLICE = 32

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu",
                 batch_size: int = 32):
        """Initialize embedder.

        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run on ("cpu" or "cuda")
            batch_size: Texts per forward pass in embed_batch
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = max(1, batch_size)
        self._lock = threading.Lock()
        # Load from the local Hugging Face cache first: otherwise every start
        # asks the Hub for model metadata (~1s, and it stalls when the network
        # is flaky). Only a model that was never downloaded goes online.
        try:
            self.model = SentenceTransformer(model_name, device=device, local_files_only=True)
        except Exception as exc:  # noqa: BLE001 - any cache miss falls back to download
            logger.info("model %s not in local cache (%s); downloading", model_name, exc)
            self.model = SentenceTransformer(model_name, device=device)
        get_dim = getattr(self.model, "get_embedding_dimension", None) \
            or self.model.get_sentence_embedding_dimension
        self.dimension = get_dim()

    def embed_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Embed a batch of texts.

        Args:
            texts: List of texts to embed
            show_progress: Show progress bar

        Returns:
            Array of embeddings with shape (len(texts), dimension)
        """
        # One model serves search queries and server-side ingests on different
        # threads. Fast tokenizers have been known to raise "Already borrowed"
        # under concurrent use; torch already spreads one encode across cores,
        # so serializing costs little. The lock is taken per slice so a search
        # waits for at most one slice of a long ingest, not the whole batch.
        # A slice smaller than batch_size would cap the forward-pass size.
        step = max(self.LOCK_SLICE, self.batch_size)
        if show_progress or len(texts) <= step:
            with self._lock:
                return self._encode(texts, show_progress)
        parts = []
        for i in range(0, len(texts), step):
            with self._lock:
                parts.append(self._encode(texts[i:i + step], False))
        return np.vstack(parts)

    def _encode(self, texts: List[str], show_progress: bool) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return self.embed_batch([text])[0]

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query (same as text embedding for this model).

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        return self.embed_single(query)
