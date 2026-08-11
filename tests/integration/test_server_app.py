"""Integration tests for the FastAPI server (server/app.py) via TestClient,
with the `get_memory` dependency overridden to inject a Memory built from
FakeLLM/FakeEmbedder — no real network or model download.
"""

import json

import pytest
from fastapi.testclient import TestClient

from server.app import app
from server.dependencies import get_memory


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
def client(tmp_path, fake_llm, fake_embedder):
    memory = _make_memory(tmp_path, fake_llm, fake_embedder)
    app.dependency_overrides[get_memory] = lambda: memory
    with TestClient(app) as test_client:
        yield test_client, memory
    app.dependency_overrides.clear()


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


class TestHealthCheck:
    def test_root_reports_ok(self, client):
        test_client, _memory = client
        response = test_client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestAddEndpoint:
    def test_add_returns_add_event(self, client):
        test_client, memory = client
        memory.llm.queue(
            json.dumps({"facts": [{"text": "Likes filter coffee", "category": "semantic"}]})
        )
        memory.llm.queue(json.dumps({"memory": [{"text": "Likes filter coffee", "event": "ADD"}]}))

        response = test_client.post(
            "/memories", json={"messages": "I love filter coffee", "user_id": "alice"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["results"][0]["event"] == "ADD"
        assert body["results"][0]["memory"] == "Likes filter coffee"

    def test_add_infer_false_never_touches_the_llm(self, client):
        test_client, _memory = client
        response = test_client.post(
            "/memories",
            json={"messages": "hi there", "user_id": "alice", "infer": False},
        )
        assert response.status_code == 200
        assert response.json()["results"][0]["event"] == "ADD"

    def test_add_missing_user_id_is_rejected(self, client):
        test_client, _memory = client
        response = test_client.post("/memories", json={"messages": "hi"})
        assert response.status_code == 422


class TestListEndpoint:
    def test_list_empty_for_new_user(self, client):
        test_client, _memory = client
        response = test_client.get("/memories", params={"user_id": "alice"})
        assert response.status_code == 200
        assert response.json()["results"] == []

    def test_list_returns_seeded_memories(self, client):
        test_client, memory = client
        _seed(memory, "mem-1", "alice", "Likes filter coffee")
        response = test_client.get("/memories", params={"user_id": "alice"})
        results = response.json()["results"]
        assert len(results) == 1
        assert results[0]["memory"] == "Likes filter coffee"

    def test_list_requires_user_id(self, client):
        test_client, _memory = client
        response = test_client.get("/memories")
        assert response.status_code == 422

    def test_list_never_leaks_across_users(self, client):
        test_client, memory = client
        _seed(memory, "mem-1", "bob", "bob's secret")
        response = test_client.get("/memories", params={"user_id": "alice"})
        assert response.json()["results"] == []


class TestSearchEndpoint:
    def test_search_finds_inserted_memory(self, client):
        test_client, memory = client
        _seed(memory, "mem-1", "alice", "Likes tea")

        response = test_client.post(
            "/search", json={"query": "Likes tea", "user_id": "alice", "limit": 5}
        )

        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == 1
        assert results[0]["memory"] == "Likes tea"
        assert "score" in results[0]

    def test_search_missing_query_is_rejected(self, client):
        test_client, _memory = client
        response = test_client.post("/search", json={"user_id": "alice"})
        assert response.status_code == 422


class TestDeleteEndpoint:
    def test_delete_removes_memory(self, client):
        test_client, memory = client
        _seed(memory, "mem-1", "alice", "x")

        response = test_client.delete("/memories/mem-1")

        assert response.status_code == 200
        assert response.json()["message"]
        assert memory.get("mem-1") is None

    def test_delete_unknown_id_returns_404(self, client):
        test_client, _memory = client
        response = test_client.delete("/memories/does-not-exist")
        assert response.status_code == 404


class TestHistoryEndpoint:
    def test_history_returns_events(self, client):
        test_client, memory = client
        memory.history_store.add_history("mem-1", None, "hello", "ADD")

        response = test_client.get("/memories/mem-1/history")

        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["event"] == "ADD"

    def test_history_of_unknown_id_is_an_empty_list_not_an_error(self, client):
        test_client, _memory = client
        response = test_client.get("/memories/does-not-exist/history")
        assert response.status_code == 200
        assert response.json() == []


class TestOpenApiDocs:
    def test_openapi_schema_is_served(self, client):
        test_client, _memory = client
        response = test_client.get("/openapi.json")
        assert response.status_code == 200
        assert response.json()["info"]["title"]
