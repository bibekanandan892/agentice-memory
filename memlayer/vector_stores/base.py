"""VectorStoreBase — the abstract interface every vector store provider implements.

See docs/design/02-lld-memlayer.md §4.

Score contract: search() results MUST rank higher = more similar, always in
[-1, 1] for cosine similarity. Every method is user-scoped except get()/reset(),
which operate by memory id / globally respectively.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ScoredPoint:
    id: str
    score: float
    payload: dict[str, Any]


class VectorStoreBase(ABC):
    @abstractmethod
    def insert(self, id: str, vector: list[float], payload: dict[str, Any]) -> None:
        """payload must contain a "user_id" key — the store uses it to decide
        which partition this row belongs to."""
        raise NotImplementedError

    @abstractmethod
    def search(self, vector: list[float], user_id: str, top_k: int = 5) -> list[ScoredPoint]:
        """MUST only ever search within `user_id`'s own partition — this is
        the enforced user-isolation boundary, not a post-hoc filter."""
        raise NotImplementedError

    @abstractmethod
    def get(self, id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def get_all(self, user_id: str, top_k: int = 100) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def update(self, id: str, vector: list[float], payload: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_all(self, user_id: str) -> int:
        """Returns the number of rows removed."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Drop everything — used by Memory.reset()."""
        raise NotImplementedError
