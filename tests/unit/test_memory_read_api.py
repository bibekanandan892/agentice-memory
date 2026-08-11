"""Tests for Memory's constructor, from_config(), and read APIs
(search/get/get_all/history) — the scope guard and payload promotion.

See docs/design/02-lld-memlayer.md §7.
"""

from unittest.mock import patch

import pytest

from memlayer.errors import ScopeError
from memlayer.memory import Memory


class TestConstructorAndFromConfig:
    def test_direct_construction_with_injected_collaborators(
        self, fake_llm, fake_embedder, tmp_path
    ):
        from memlayer.storage.history import SQLiteHistoryStore
        from memlayer.vector_stores.local import LocalVectorStore

        memory = Memory(
            llm=fake_llm,
            embedder=fake_embedder,
            vector_store=LocalVectorStore(db_path=tmp_path / "v.db"),
            history_store=SQLiteHistoryStore(db_path=tmp_path / "h.db"),
        )
        assert memory.llm is fake_llm
        assert memory.embedder is fake_embedder

    def test_from_config_wires_gemini_and_sentence_transformer_by_default(self, tmp_path):
        config = {
            "llm": {"provider": "gemini", "config": {"api_key": "fake-key-for-test"}},
            "embedder": {"provider": "sentence_transformer", "config": {}},
            "vector_store": {"provider": "local", "config": {"db_path": str(tmp_path / "v.db")}},
            "history_db_path": str(tmp_path / "h.db"),
        }
        with patch("memlayer.llms.gemini.genai.Client"):
            memory = Memory.from_config(config)

        from memlayer.embeddings.sentence_transformer import SentenceTransformerEmbedder
        from memlayer.llms.gemini import GeminiLLM
        from memlayer.vector_stores.local import LocalVectorStore

        assert isinstance(memory.llm, GeminiLLM)
        assert isinstance(memory.embedder, SentenceTransformerEmbedder)
        assert isinstance(memory.vector_store, LocalVectorStore)

    def test_from_config_defaults_vector_store_path_next_to_history_db(self, tmp_path):
        config = {
            "llm": {"provider": "gemini", "config": {"api_key": "fake-key-for-test"}},
            "history_db_path": str(tmp_path / "sub" / "h.db"),
        }
        with patch("memlayer.llms.gemini.genai.Client"):
            memory = Memory.from_config(config)
        assert memory.vector_store.db_path.parent == (tmp_path / "sub")


class TestScopeGuard:
    def test_search_without_any_scope_id_raises_scope_error(self, memory_with_fakes):
        memory_with_fakes.llm.queue('{"facts": []}')
        with pytest.raises(ScopeError):
            memory_with_fakes.search("anything")

    def test_get_all_without_scope_id_raises_scope_error(self, memory_with_fakes):
        with pytest.raises(ScopeError):
            memory_with_fakes.get_all()

    def test_search_with_user_id_does_not_raise(self, memory_with_fakes):
        result = memory_with_fakes.search("anything", user_id="alice")
        assert result == {"results": []}


class TestSearch:
    def test_search_returns_empty_results_for_new_user(self, memory_with_fakes):
        assert memory_with_fakes.search("hello", user_id="alice") == {"results": []}

    def test_search_finds_a_directly_inserted_memory(self, memory_with_fakes):
        memory_with_fakes.vector_store.insert(
            "mem-1",
            memory_with_fakes.embedder.embed("likes filter coffee"),
            {
                "user_id": "alice",
                "data": "Likes filter coffee",
                "hash": "abc",
                "memory_category": "semantic",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            },
        )
        result = memory_with_fakes.search("likes filter coffee", user_id="alice", limit=5)
        assert len(result["results"]) == 1
        item = result["results"][0]
        assert item["id"] == "mem-1"
        assert item["memory"] == "Likes filter coffee"
        assert item["user_id"] == "alice"
        assert item["memory_category"] == "semantic"
        assert "score" in item

    def test_search_never_crosses_user_boundary(self, memory_with_fakes):
        memory_with_fakes.vector_store.insert(
            "mem-bob",
            memory_with_fakes.embedder.embed("bob's secret"),
            {
                "user_id": "bob",
                "data": "bob's secret",
                "hash": "x",
                "memory_category": "semantic",
                "created_at": "t",
                "updated_at": "t",
            },
        )
        result = memory_with_fakes.search("bob's secret", user_id="alice", limit=5)
        assert result["results"] == []


class TestGet:
    def test_get_unknown_id_returns_none(self, memory_with_fakes):
        assert memory_with_fakes.get("does-not-exist") is None

    def test_get_returns_formatted_result_without_score(self, memory_with_fakes):
        memory_with_fakes.vector_store.insert(
            "mem-1",
            memory_with_fakes.embedder.embed("x"),
            {
                "user_id": "alice",
                "data": "some fact",
                "hash": "h",
                "memory_category": "episodic",
                "created_at": "t1",
                "updated_at": "t1",
            },
        )
        result = memory_with_fakes.get("mem-1")
        assert result["memory"] == "some fact"
        assert result["user_id"] == "alice"
        assert result["memory_category"] == "episodic"
        assert "score" not in result


class TestGetAll:
    def test_get_all_empty_for_new_user(self, memory_with_fakes):
        assert memory_with_fakes.get_all(user_id="alice") == {"results": []}

    def test_get_all_returns_every_memory_for_user(self, memory_with_fakes):
        for i in range(3):
            memory_with_fakes.vector_store.insert(
                f"mem-{i}",
                memory_with_fakes.embedder.embed(f"fact {i}"),
                {
                    "user_id": "alice",
                    "data": f"fact {i}",
                    "hash": str(i),
                    "memory_category": "semantic",
                    "created_at": "t",
                    "updated_at": "t",
                },
            )
        result = memory_with_fakes.get_all(user_id="alice")
        assert len(result["results"]) == 3
        assert all("score" not in item for item in result["results"])


class TestHistory:
    def test_history_of_unknown_memory_is_empty(self, memory_with_fakes):
        assert memory_with_fakes.history("nope") == []

    def test_history_reflects_history_store_records(self, memory_with_fakes):
        memory_with_fakes.history_store.add_history("mem-1", None, "hello", "ADD")
        records = memory_with_fakes.history("mem-1")
        assert len(records) == 1
        assert records[0]["event"] == "ADD"
