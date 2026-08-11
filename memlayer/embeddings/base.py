"""EmbeddingBase — the abstract interface every embedding provider implements.

See docs/design/02-lld-memlayer.md §3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingBase(ABC):
    dims: int

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a fixed-length embedding vector for text. `dims` must match
        len(embed(...)) exactly — LocalVectorStore uses `dims` to size its
        per-user partition matrices at collection-creation time, and a
        mismatch here is the classic "embedder/store dimension mismatch"
        misconfiguration this attribute exists to prevent.
        """
        raise NotImplementedError
