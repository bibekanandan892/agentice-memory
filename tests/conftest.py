"""Shared test fixtures and fakes.

FakeLLM and FakeEmbedder are the two test doubles every test in this project
depends on. See docs/design/03-lld-memento.md §7 for their design contract.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest


class FakeLLM:
    """A scripted LLMBase double.

    Construct with an ordered list of canned response strings (or feed them via
    .queue()); each generate_response() call pops the next one and records the
    full prompt it was given, so tests can assert what was sent (e.g. "the
    reconciler prompt contained integer ids, never a UUID").
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses: list[str] = list(responses or [])
        self.prompts_seen: list[list[dict]] = []

    def queue(self, response: str) -> None:
        self._responses.append(response)

    def generate_response(self, messages: list[dict], response_format: str = "json") -> str:
        self.prompts_seen.append(messages)
        if not self._responses:
            raise AssertionError(
                "FakeLLM.generate_response() called but no more responses were queued — "
                "the test provided fewer FakeLLM responses than were consumed."
            )
        return self._responses.pop(0)


class FakeEmbedder:
    """A deterministic EmbeddingBase double.

    Identical text always produces an identical vector (seeded from md5(text)),
    and different text produces different vectors — reproducible cosine-similarity
    assertions without downloading a real embedding model.
    """

    def __init__(self, dims: int = 8) -> None:
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        return self._deterministic_vector(text)

    def _deterministic_vector(self, text: str) -> list[float]:
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=self.dims)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder(dims=8)


@pytest.fixture
def mem_config(tmp_path: Path) -> dict:
    """A realistic memlayer config dict (real provider names) pointing at temp DB
    files, for from_config()/MemoryConfig parsing tests. Pipeline tests that need
    a working Memory instance use the `memory_with_fakes` fixture instead — fakes
    are injected directly via Memory's constructor, never through the provider
    factory, so test doubles never leak into production config parsing.
    """
    return {
        "llm": {"provider": "gemini", "config": {"model": "gemini-flash-latest"}},
        "embedder": {"provider": "sentence_transformer", "config": {}},
        "vector_store": {"provider": "local", "config": {"db_path": str(tmp_path / "vectors.db")}},
        "history_db_path": str(tmp_path / "history.db"),
    }


@pytest.fixture
def memory_with_fakes(fake_llm: FakeLLM, fake_embedder: FakeEmbedder, tmp_path: Path):
    """A fully wired Memory instance with FakeLLM/FakeEmbedder injected and real
    (but temp-path) LocalVectorStore + SQLiteHistoryStore — for add()/search()
    pipeline tests that need real storage/search behavior without any network
    or model download.
    """
    from memlayer.memory import Memory
    from memlayer.storage.history import SQLiteHistoryStore
    from memlayer.vector_stores.local import LocalVectorStore

    vector_store = LocalVectorStore(db_path=tmp_path / "vectors.db")
    history_store = SQLiteHistoryStore(db_path=tmp_path / "history.db")
    return Memory(
        llm=fake_llm,
        embedder=fake_embedder,
        vector_store=vector_store,
        history_store=history_store,
    )
