from typing import Any, Optional
import hashlib
import math
import logging

import sentence_transformers
import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """Wrapper around SentenceTransformers that gracefully falls back to a
    deterministic, normalized hash-based embedder when the real model can't be
    loaded (e.g., offline CI or incompatible package versions).

    The fallback produces unit-norm vectors so downstream distance normalization
    (e.g., Euclidean -> [0,1]) behaves as expected in tests and in production.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_folder: Optional[str] = None, dim: int = 384, **kwargs: Any):
        self._model_name = model_name
        self._dim = dim
        self.client = None
        try:
            # Try to load the real model
            self.client = sentence_transformers.SentenceTransformer(model_name, cache_folder=cache_folder, **kwargs)
        except Exception as e:  # pragma: no cover - exercised by tests when offline
            logger.warning(
                "Failed to load SentenceTransformer '%s' (falling back to hash-based embedder): %s",
                model_name,
                e,
            )
            self.client = None

    def _hash_to_vector(self, text: str) -> list[float]:
        # Deterministic hashing -> pseudo-random vector.
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Expand or repeat hash bytes to reach required dim
        repeats = math.ceil(self._dim / len(h))
        buf = (h * repeats)[: self._dim]
        arr = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        # Map bytes 0-255 to range [-1,1]
        arr = (arr / 255.0) * 2.0 - 1.0
        # Normalize to unit length
        norm = np.linalg.norm(arr)
        if norm == 0:
            return [0.0] * self._dim
        arr = arr / norm
        return arr.tolist()

    def embed_documents(self, texts: list[str], multi_process: bool = False, **encode_kwargs: Any) -> list[list[float]]:
        texts = [t.replace("\n", " ") for t in texts]
        if self.client is None:
            # Fallback deterministic embeddings (unit-norm)
            return [self._hash_to_vector(t) for t in texts]

        if multi_process:
            pool = self.client.start_multi_process_pool()
            embeddings = self.client.encode_multi_process(texts, pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self.client.encode(texts, show_progress_bar=False, **encode_kwargs)

        # Ensure output is list[list[float]] and normalized to unit length to avoid
        # large Euclidean distances that break relevance normalization.
        emb = np.array(embeddings)
        # If model returns 1D vector for a single input, force 2D
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        # Normalize rows to unit norm
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
        return emb.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
