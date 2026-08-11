"""Tests for LocalVectorStore — NumPy cosine search sharded by user_id, SQLite-persisted.

See docs/design/02-lld-memlayer.md §4. The headline test is
test_search_never_returns_another_users_memories: search() must only ever
touch its own user's partition — an enforced isolation boundary, not a
post-hoc filter over a shared pool.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from memlayer.vector_stores.base import ScoredPoint
from memlayer.vector_stores.local import LocalVectorStore


def _payload(user_id: str, **extra: object) -> dict:
    payload = {"user_id": user_id, "data": "some memory text"}
    payload.update(extra)
    return payload


@pytest.fixture
def store(tmp_path: Path) -> LocalVectorStore:
    return LocalVectorStore(db_path=tmp_path / "vectors.db")


class TestInsertAndSearch:
    def test_insert_then_search_returns_the_inserted_item(self, store):
        store.insert("id-1", [1.0, 0.0, 0.0], _payload("user-a"))

        results = store.search([1.0, 0.0, 0.0], user_id="user-a", top_k=5)

        assert len(results) == 1
        assert isinstance(results[0], ScoredPoint)
        assert results[0].id == "id-1"
        assert results[0].payload["data"] == "some memory text"

    def test_search_on_missing_partition_returns_empty_list(self, store):
        assert store.search([1.0, 0.0], user_id="ghost", top_k=5) == []

    def test_search_on_empty_store_returns_empty_list(self, store):
        assert store.search([1.0, 0.0, 0.0], user_id="user-a", top_k=5) == []

    def test_cosine_ranking_matches_hand_computed_scores(self, store):
        # query = [1, 0]
        # a = [1, 0]              -> cosine = 1.0
        # b = [1, 1] normalized   -> cosine with query = 1/sqrt(2) ~= 0.7071
        # c = [0, 1]              -> cosine = 0.0
        store.insert("a", [1.0, 0.0], _payload("u1"))
        store.insert("b", [1.0, 1.0], _payload("u1"))
        store.insert("c", [0.0, 1.0], _payload("u1"))

        results = store.search([1.0, 0.0], user_id="u1", top_k=3)

        assert [r.id for r in results] == ["a", "b", "c"]
        expected = {"a": 1.0, "b": 1 / math.sqrt(2), "c": 0.0}
        for result in results:
            assert result.score == pytest.approx(expected[result.id], abs=1e-6)

    def test_scores_are_within_valid_cosine_range(self, store):
        store.insert("a", [1.0, 0.0], _payload("u1"))
        store.insert("b", [-1.0, 0.0], _payload("u1"))

        results = store.search([1.0, 0.0], user_id="u1", top_k=2)

        for result in results:
            assert -1.0 <= result.score <= 1.0

    def test_top_k_limits_number_of_results(self, store):
        for i in range(5):
            store.insert(f"id-{i}", [1.0, float(i)], _payload("u1"))

        results = store.search([1.0, 0.0], user_id="u1", top_k=2)

        assert len(results) == 2

    def test_works_with_8_dim_vectors(self, store):
        store.insert("id-1", [1.0] * 8, _payload("u1"))
        results = store.search([1.0] * 8, user_id="u1", top_k=1)
        assert results[0].id == "id-1"

    def test_works_with_384_dim_vectors(self, store):
        store.insert("id-1", [1.0] * 384, _payload("u1"))
        results = store.search([1.0] * 384, user_id="u1", top_k=1)
        assert results[0].id == "id-1"

    def test_zero_vector_does_not_crash(self, store):
        store.insert("id-1", [0.0, 0.0], _payload("u1"))
        results = store.search([0.0, 0.0], user_id="u1", top_k=5)
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.0)

    def test_insert_without_user_id_in_payload_raises(self, store):
        with pytest.raises(KeyError):
            store.insert("id-1", [1.0, 0.0], {"data": "no user id here"})


class TestUserIsolation:
    def test_search_never_returns_another_users_memories(self, store):
        """The headline test.

        User B's vector is an EXACT match for the query (more similar than
        anything user A owns), yet a search scoped to user A must never
        surface it. If search() ever degraded to a global matrix with a
        post-hoc filter, user B's row would win on similarity and leak
        through — this test exists to make that leak impossible to miss.
        """
        store.insert("a-1", [1.0, 0.0, 0.0], _payload("user-a"))
        store.insert("b-1", [0.0, 1.0, 0.0], _payload("user-b"))

        results = store.search([0.0, 1.0, 0.0], user_id="user-a", top_k=5)

        ids = [r.id for r in results]
        assert "b-1" not in ids
        assert ids == ["a-1"]

    def test_get_all_only_returns_scoped_user(self, store):
        store.insert("a-1", [1.0, 0.0], _payload("user-a"))
        store.insert("b-1", [0.0, 1.0], _payload("user-b"))

        results = store.get_all("user-a", top_k=100)

        assert [row["id"] for row in results] == ["a-1"]

    def test_delete_all_leaves_other_users_untouched(self, store):
        store.insert("a-1", [1.0, 0.0], _payload("user-a"))
        store.insert("b-1", [0.0, 1.0], _payload("user-b"))

        removed = store.delete_all("user-a")

        assert removed == 1
        assert store.search([1.0, 0.0], user_id="user-a", top_k=5) == []
        assert len(store.search([0.0, 1.0], user_id="user-b", top_k=5)) == 1


class TestGet:
    def test_get_returns_payload_with_id_included(self, store):
        store.insert("id-1", [1.0, 0.0], _payload("user-a", data="hello"))

        fetched = store.get("id-1")

        assert fetched["id"] == "id-1"
        assert fetched["data"] == "hello"
        assert fetched["user_id"] == "user-a"

    def test_get_on_never_existed_id_returns_none_gracefully(self, store):
        assert store.get("does-not-exist") is None


class TestGetAll:
    def test_get_all_respects_top_k(self, store):
        for i in range(5):
            store.insert(f"id-{i}", [1.0, float(i)], _payload("user-a"))

        results = store.get_all("user-a", top_k=2)

        assert len(results) == 2

    def test_get_all_on_missing_user_returns_empty_list(self, store):
        assert store.get_all("ghost", top_k=100) == []


class TestUpdate:
    def test_update_changes_vector_and_payload_and_search_reflects_it(self, store):
        store.insert("id-1", [1.0, 0.0], _payload("user-a", data="old"))

        store.update("id-1", [0.0, 1.0], _payload("user-a", data="new"))

        results = store.search([0.0, 1.0], user_id="user-a", top_k=5)
        assert len(results) == 1
        assert results[0].id == "id-1"
        assert results[0].payload["data"] == "new"
        assert results[0].score == pytest.approx(1.0, abs=1e-6)

        # the old vector no longer matches as strongly
        old_direction_results = store.search([1.0, 0.0], user_id="user-a", top_k=5)
        assert old_direction_results[0].score == pytest.approx(0.0, abs=1e-6)

    def test_update_on_unknown_id_raises_key_error(self, store):
        with pytest.raises(KeyError):
            store.update("nope", [1.0, 0.0], _payload("user-a"))

    def test_update_persists_across_reload(self, tmp_path):
        db_path = tmp_path / "vectors.db"
        store1 = LocalVectorStore(db_path=db_path)
        store1.insert("id-1", [1.0, 0.0], _payload("user-a", data="old"))
        store1.update("id-1", [0.0, 1.0], _payload("user-a", data="new"))
        del store1

        store2 = LocalVectorStore(db_path=db_path)
        fetched = store2.get("id-1")
        assert fetched["data"] == "new"
        results = store2.search([0.0, 1.0], user_id="user-a", top_k=5)
        assert results[0].score == pytest.approx(1.0, abs=1e-6)


class TestDelete:
    def test_delete_removes_row_from_search_and_get_all(self, store):
        store.insert("id-1", [1.0, 0.0], _payload("user-a"))
        store.insert("id-2", [0.0, 1.0], _payload("user-a"))

        store.delete("id-1")

        search_ids = [r.id for r in store.search([1.0, 0.0], user_id="user-a", top_k=5)]
        assert "id-1" not in search_ids
        remaining_ids = [row["id"] for row in store.get_all("user-a", top_k=100)]
        assert remaining_ids == ["id-2"]

    def test_get_after_delete_returns_none(self, store):
        store.insert("id-1", [1.0, 0.0], _payload("user-a"))
        store.delete("id-1")
        assert store.get("id-1") is None

    def test_delete_on_unknown_id_raises_key_error(self, store):
        with pytest.raises(KeyError):
            store.delete("nope")

    def test_delete_persists_across_reload(self, tmp_path):
        db_path = tmp_path / "vectors.db"
        store1 = LocalVectorStore(db_path=db_path)
        store1.insert("id-1", [1.0, 0.0], _payload("user-a"))
        store1.delete("id-1")
        del store1

        store2 = LocalVectorStore(db_path=db_path)
        assert store2.get("id-1") is None
        assert store2.search([1.0, 0.0], user_id="user-a", top_k=5) == []


class TestDeleteAll:
    def test_delete_all_returns_count_removed(self, store):
        store.insert("a-1", [1.0, 0.0], _payload("user-a"))
        store.insert("a-2", [0.0, 1.0], _payload("user-a"))

        removed = store.delete_all("user-a")

        assert removed == 2

    def test_delete_all_on_missing_user_returns_zero(self, store):
        assert store.delete_all("ghost") == 0

    def test_delete_all_persists_across_reload(self, tmp_path):
        db_path = tmp_path / "vectors.db"
        store1 = LocalVectorStore(db_path=db_path)
        store1.insert("a-1", [1.0, 0.0], _payload("user-a"))
        store1.delete_all("user-a")
        del store1

        store2 = LocalVectorStore(db_path=db_path)
        assert store2.search([1.0, 0.0], user_id="user-a", top_k=5) == []


class TestReset:
    def test_reset_empties_everything(self, store):
        store.insert("a-1", [1.0, 0.0], _payload("user-a"))
        store.insert("b-1", [0.0, 1.0], _payload("user-b"))

        store.reset()

        assert store.search([1.0, 0.0], user_id="user-a", top_k=5) == []
        assert store.search([0.0, 1.0], user_id="user-b", top_k=5) == []
        assert store.get("a-1") is None

    def test_reset_persists_across_reload(self, tmp_path):
        db_path = tmp_path / "vectors.db"
        store1 = LocalVectorStore(db_path=db_path)
        store1.insert("a-1", [1.0, 0.0], _payload("user-a"))
        store1.reset()
        del store1

        store2 = LocalVectorStore(db_path=db_path)
        assert store2.search([1.0, 0.0], user_id="user-a", top_k=5) == []

    def test_insert_after_reset_works(self, store):
        store.insert("a-1", [1.0, 0.0], _payload("user-a"))
        store.reset()
        store.insert("a-2", [1.0, 0.0], _payload("user-a"))

        results = store.search([1.0, 0.0], user_id="user-a", top_k=5)

        assert [r.id for r in results] == ["a-2"]


class TestReloadAndPersistence:
    def test_reload_from_same_db_path_resumes_state(self, tmp_path):
        db_path = tmp_path / "vectors.db"
        store1 = LocalVectorStore(db_path=db_path)
        store1.insert("id-1", [1.0, 0.0, 0.0], _payload("user-a", data="hello"))
        del store1

        store2 = LocalVectorStore(db_path=db_path)
        results = store2.search([1.0, 0.0, 0.0], user_id="user-a", top_k=5)

        assert len(results) == 1
        assert results[0].id == "id-1"
        assert results[0].payload["data"] == "hello"

    def test_reload_preserves_multiple_users_partitions(self, tmp_path):
        db_path = tmp_path / "vectors.db"
        store1 = LocalVectorStore(db_path=db_path)
        store1.insert("a-1", [1.0, 0.0], _payload("user-a"))
        store1.insert("b-1", [0.0, 1.0], _payload("user-b"))
        del store1

        store2 = LocalVectorStore(db_path=db_path)
        assert len(store2.search([1.0, 0.0], user_id="user-a", top_k=5)) == 1
        assert len(store2.search([0.0, 1.0], user_id="user-b", top_k=5)) == 1
        # isolation must still hold after reload
        a_ids = [r.id for r in store2.search([1.0, 0.0], user_id="user-a", top_k=5)]
        assert "b-1" not in a_ids

    def test_constructing_with_nonexistent_parent_dir_creates_it(self, tmp_path):
        db_path = tmp_path / "nested" / "dir" / "vectors.db"
        store_instance = LocalVectorStore(db_path=db_path)
        assert store_instance.search([1.0, 0.0], user_id="user-a", top_k=5) == []
        assert db_path.parent.exists()
