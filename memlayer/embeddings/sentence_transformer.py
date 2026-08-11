"""SentenceTransformerEmbedder — lazy-loaded local embedder, the project default.

See docs/design/02-lld-memlayer.md §3. `sentence-transformers` (and its
`torch` dependency) is NOT a core dependency of this project — it lives
behind the optional `[local]` extra in pyproject.toml. That means both the
`import sentence_transformers` statement and the construction of the
underlying model must be deferred until the first `.embed()` call:
`import memlayer` (and even `import memlayer.embeddings.sentence_transformer`
or constructing `SentenceTransformerEmbedder`) must never require the extra
to be installed or trigger a ~90MB model download.
"""

from __future__ import annotations

import numpy as np

from memlayer.embeddings.base import EmbeddingBase

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_DIMS = 384

MISSING_DEPENDENCY_MESSAGE = "sentence-transformers is not installed. Run: uv sync --extra local"


class SentenceTransformerEmbedder(EmbeddingBase):
    """Default embedder: wraps sentence-transformers' `all-MiniLM-L6-v2` model.

    Free, local, offline, 384-dim. The real `SentenceTransformer` model is
    loaded lazily on the first `.embed()` call rather than in `__init__` —
    constructing it downloads/loads the model from disk, which must never
    happen just from instantiating this class.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: object | None = None
        self._dims = DEFAULT_DIMS

    @property
    def dims(self) -> int:
        """384 by default; updated to the loaded model's actual dimension
        (via `get_sentence_embedding_dimension()`) once `_ensure_loaded()`
        has run at least once.
        """
        return self._dims

    def embed(self, text: str) -> list[float]:
        self._ensure_loaded()
        raw_vector = self._model.encode(text)
        return np.asarray(raw_vector, dtype=float).tolist()

    def _ensure_loaded(self) -> None:
        """Import sentence_transformers and construct the model, exactly once."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(MISSING_DEPENDENCY_MESSAGE) from exc
        self._model = SentenceTransformer(self.model_name)
        self._dims = self._model.get_sentence_embedding_dimension()
