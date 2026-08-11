"""Live smoke test hitting the real Gemini API — the ONLY test in this
project allowed to touch the network. Everything else is mocked.

Run with:
    uv run pytest -m live
Skipped automatically (not failed) if GEMINI_API_KEY isn't set. Also requires
the [local] extra (`uv sync --extra local`) for the real embedder.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live


@pytest.fixture(autouse=True)
def _skip_without_api_key():
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is not set; skipping live smoke test.")


def test_add_then_search_against_real_gemini_and_real_embeddings(tmp_path):
    from memlayer.embeddings.sentence_transformer import SentenceTransformerEmbedder
    from memlayer.llms.gemini import GeminiLLM
    from memlayer.memory import Memory
    from memlayer.storage.history import SQLiteHistoryStore
    from memlayer.vector_stores.local import LocalVectorStore

    memory = Memory(
        llm=GeminiLLM(),
        embedder=SentenceTransformerEmbedder(),
        vector_store=LocalVectorStore(db_path=tmp_path / "v.db"),
        history_store=SQLiteHistoryStore(db_path=tmp_path / "h.db"),
    )

    add_result = memory.add(
        "I love filter coffee and I'm learning about AI memory systems.",
        user_id="live_test_user",
    )
    assert isinstance(add_result["results"], list)
    assert len(add_result["results"]) > 0

    search_result = memory.search(
        "What does the user like to drink?", user_id="live_test_user", limit=5
    )
    assert isinstance(search_result["results"], list)
    assert len(search_result["results"]) > 0
