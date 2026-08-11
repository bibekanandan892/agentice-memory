"""Tests for memlayer.storage.history.SQLiteHistoryStore — the mutation audit log.

See docs/design/02-lld-memlayer.md §5.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pytest

from memlayer.storage.history import SQLiteHistoryStore

HISTORY_COLUMNS = {
    "id",
    "memory_id",
    "old_memory",
    "new_memory",
    "event",
    "created_at",
    "updated_at",
    "is_deleted",
    "actor_id",
    "role",
}


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "history.db"


@pytest.fixture
def store(db_path):
    return SQLiteHistoryStore(db_path)


class TestSchemaCreation:
    def test_creates_history_table_on_fresh_file(self, db_path):
        assert not db_path.exists()

        SQLiteHistoryStore(db_path)

        assert db_path.exists()
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='history'"
            )
            assert cursor.fetchone() is not None
        finally:
            conn.close()

    def test_history_table_has_expected_columns(self, db_path):
        SQLiteHistoryStore(db_path)

        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.execute("PRAGMA table_info(history)")
            columns = {row[1] for row in cursor.fetchall()}
        finally:
            conn.close()

        assert columns == HISTORY_COLUMNS

    def test_reopening_existing_db_does_not_error(self, db_path):
        SQLiteHistoryStore(db_path)
        # Second store instance pointed at the same file must not blow up
        # (CREATE TABLE IF NOT EXISTS semantics).
        second_store = SQLiteHistoryStore(db_path)
        assert second_store.get_history("nonexistent") == []


class TestAddHistoryRoundtrip:
    def test_add_event_roundtrips_with_none_old_memory(self, store):
        store.add_history("mem-1", None, "the new fact", "ADD")

        rows = store.get_history("mem-1")

        assert len(rows) == 1
        row = rows[0]
        assert row["memory_id"] == "mem-1"
        assert row["old_memory"] is None
        assert row["new_memory"] == "the new fact"
        assert row["event"] == "ADD"
        assert row["is_deleted"] is False
        assert row["actor_id"] is None
        assert row["role"] is None

    def test_add_event_generates_uuid4_id(self, store):
        store.add_history("mem-1", None, "the new fact", "ADD")

        row = store.get_history("mem-1")[0]

        # Must not raise — id should be a valid uuid4 hex string.
        parsed = uuid.UUID(row["id"])
        assert parsed.version == 4

    def test_add_event_sets_created_at_and_updated_at_internally(self, store):
        store.add_history("mem-1", None, "the new fact", "ADD")

        row = store.get_history("mem-1")[0]

        assert row["created_at"] is not None
        assert row["updated_at"] is not None
        # Must be parseable ISO-8601.
        datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
        datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00"))

    def test_update_event_stores_both_old_and_new_memory(self, store):
        store.add_history("mem-1", None, "original fact", "ADD")
        store.add_history("mem-1", "original fact", "revised fact", "UPDATE")

        rows = store.get_history("mem-1")

        assert len(rows) == 2
        update_row = rows[1]
        assert update_row["event"] == "UPDATE"
        assert update_row["old_memory"] == "original fact"
        assert update_row["new_memory"] == "revised fact"
        assert update_row["is_deleted"] is False

    def test_delete_event_sets_is_deleted_true_and_new_memory_none(self, store):
        store.add_history("mem-1", None, "original fact", "ADD")
        store.add_history("mem-1", "original fact", None, "DELETE")

        rows = store.get_history("mem-1")

        assert len(rows) == 2
        delete_row = rows[1]
        assert delete_row["event"] == "DELETE"
        assert delete_row["old_memory"] == "original fact"
        assert delete_row["new_memory"] is None
        assert delete_row["is_deleted"] is True

    def test_actor_id_and_role_are_stored(self, store):
        store.add_history(
            "mem-1", None, "a fact", "ADD", actor_id="user-42", role="user"
        )

        row = store.get_history("mem-1")[0]

        assert row["actor_id"] == "user-42"
        assert row["role"] == "user"


class TestGetHistoryOrdering:
    def test_multiple_rows_returned_in_created_at_ascending_order(self, store):
        # Sleep between inserts to guarantee strictly increasing created_at
        # timestamps regardless of the host clock's resolution, so ordering
        # is verified by real timestamp order rather than insertion-order luck.
        store.add_history("mem-1", None, "fact one", "ADD")
        time.sleep(0.01)
        store.add_history("mem-1", "fact one", "fact two", "UPDATE")
        time.sleep(0.01)
        store.add_history("mem-1", "fact two", None, "DELETE")

        rows = store.get_history("mem-1")

        assert [row["new_memory"] for row in rows] == ["fact one", "fact two", None]
        assert [row["event"] for row in rows] == ["ADD", "UPDATE", "DELETE"]
        created_timestamps = [row["created_at"] for row in rows]
        assert created_timestamps == sorted(created_timestamps)

    def test_unknown_memory_id_returns_empty_list(self, store):
        assert store.get_history("does-not-exist") == []

    def test_only_matching_memory_id_rows_are_returned(self, store):
        store.add_history("mem-1", None, "fact for mem-1", "ADD")
        store.add_history("mem-2", None, "fact for mem-2", "ADD")

        rows = store.get_history("mem-1")

        assert len(rows) == 1
        assert rows[0]["memory_id"] == "mem-1"


class TestReset:
    def test_reset_empties_all_history(self, store):
        store.add_history("mem-1", None, "fact one", "ADD")
        store.add_history("mem-2", None, "fact two", "ADD")

        store.reset()

        assert store.get_history("mem-1") == []
        assert store.get_history("mem-2") == []

    def test_store_is_usable_after_reset(self, store):
        store.add_history("mem-1", None, "fact one", "ADD")
        store.reset()

        store.add_history("mem-1", None, "fact after reset", "ADD")

        rows = store.get_history("mem-1")
        assert len(rows) == 1
        assert rows[0]["new_memory"] == "fact after reset"


class TestThreadSafety:
    def test_concurrent_add_history_calls_do_not_raise_and_all_land(self, store):
        memory_id = "concurrent-mem"
        call_count = 20

        def write(index: int) -> None:
            store.add_history(memory_id, None, f"fact {index}", "ADD")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(write, i) for i in range(call_count)]
            # Propagate any exception raised inside a worker thread.
            for future in as_completed(futures):
                future.result()

        rows = store.get_history(memory_id)
        assert len(rows) == call_count
        # Every row got a distinct uuid4 id — no lost or duplicated writes.
        assert len({row["id"] for row in rows}) == call_count
