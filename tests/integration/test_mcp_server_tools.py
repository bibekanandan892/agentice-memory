"""Integration tests for the MCP server's tools (mcp_server/server.py).

MCPServer's @app.tool() decorator registers the function as a side effect
and returns the same plain callable (verified empirically), so these tests
call the tool functions directly — no MCP protocol/transport involved —
with `get_memory` monkeypatched to a real Memory built from
FakeLLM/FakeEmbedder, matching the testing approach used for the FastAPI
server in tests/integration/test_server_app.py.
"""

import json

import pytest


def _make_memory(tmp_path, fake_llm, fake_embedder):
    from memlayer.memory import Memory
    from memlayer.storage.history import SQLiteHistoryStore
    from memlayer.vector_stores.local import LocalVectorStore

    return Memory(
        llm=fake_llm,
        embedder=fake_embedder,
        vector_store=LocalVectorStore(db_path=tmp_path / "v.db"),
        history_store=SQLiteHistoryStore(db_path=tmp_path / "h.db"),
    )


@pytest.fixture
def memory(tmp_path, fake_llm, fake_embedder, monkeypatch):
    instance = _make_memory(tmp_path, fake_llm, fake_embedder)
    monkeypatch.setattr("mcp_server.server.get_memory", lambda: instance)
    return instance


def _seed(memory, memory_id, user_id, text, category="semantic"):
    memory.vector_store.insert(
        memory_id,
        memory.embedder.embed(text),
        {
            "user_id": user_id,
            "data": text,
            "hash": "h",
            "memory_category": category,
            "created_at": "t0",
            "updated_at": "t0",
        },
    )


class TestSaveMemory:
    def test_save_memory_extracts_and_adds(self, memory):
        from mcp_server.server import save_memory

        memory.llm.queue(
            json.dumps({"facts": [{"text": "Likes filter coffee", "category": "semantic"}]})
        )
        memory.llm.queue(json.dumps({"memory": [{"text": "Likes filter coffee", "event": "ADD"}]}))

        result = save_memory("I love filter coffee", user_id="alice")

        assert result["results"][0]["event"] == "ADD"
        assert result["results"][0]["memory"] == "Likes filter coffee"
        assert memory.get_all(user_id="alice")["results"]


class TestSearchMemory:
    def test_search_memory_finds_seeded_fact(self, memory):
        from mcp_server.server import search_memory

        _seed(memory, "mem-1", "alice", "Likes tea")

        result = search_memory("Likes tea", user_id="alice", limit=5)

        assert len(result["results"]) == 1
        assert result["results"][0]["memory"] == "Likes tea"

    def test_search_memory_never_crosses_users(self, memory):
        from mcp_server.server import search_memory

        _seed(memory, "mem-1", "bob", "bob's secret")

        result = search_memory("bob's secret", user_id="alice", limit=5)

        assert result["results"] == []


class TestListMemories:
    def test_list_memories_returns_all_for_user(self, memory):
        from mcp_server.server import list_memories

        _seed(memory, "mem-1", "alice", "fact one")
        _seed(memory, "mem-2", "alice", "fact two")

        result = list_memories(user_id="alice")

        assert len(result["results"]) == 2

    def test_list_memories_empty_for_new_user(self, memory):
        from mcp_server.server import list_memories

        assert list_memories(user_id="nobody")["results"] == []


class TestForgetMemory:
    def test_forget_memory_deletes_it(self, memory):
        from mcp_server.server import forget_memory

        _seed(memory, "mem-1", "alice", "fact one")

        result = forget_memory("mem-1")

        assert "message" in result
        assert memory.get("mem-1") is None

    def test_forget_memory_unknown_id_returns_error_message_not_a_crash(self, memory):
        from mcp_server.server import forget_memory

        result = forget_memory("does-not-exist")

        assert "error" in result
