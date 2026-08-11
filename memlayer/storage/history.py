"""SQLiteHistoryStore — the mutation audit log. Implemented in Phase 1 Task 1.4.

See docs/design/02-lld-memlayer.md §5.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from pathlib import Path

from memlayer.utils import utc_now_iso

_CREATE_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS history (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    old_memory TEXT,
    new_memory TEXT,
    event TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    is_deleted INTEGER NOT NULL DEFAULT 0,
    actor_id TEXT,
    role TEXT
);
"""

_INSERT_HISTORY_ROW = """
INSERT INTO history (
    id, memory_id, old_memory, new_memory, event,
    created_at, updated_at, is_deleted, actor_id, role
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_HISTORY_BY_MEMORY_ID = """
SELECT id, memory_id, old_memory, new_memory, event,
       created_at, updated_at, is_deleted, actor_id, role
FROM history
WHERE memory_id = ?
ORDER BY created_at ASC
"""

_HISTORY_ROW_COLUMNS = (
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
)


class SQLiteHistoryStore:
    """Audit log of every ADD/UPDATE/DELETE mutation made to the vector store.

    Backed by a single SQLite table (see module docstring / LLD §5). All
    writes are serialized through a `threading.Lock` since the underlying
    sqlite3 connection is shared across threads (`check_same_thread=False`).
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock:
            self._conn.execute(_CREATE_HISTORY_TABLE)
            self._conn.commit()

    def add_history(
        self,
        memory_id: str,
        old_memory: str | None,
        new_memory: str | None,
        event: str,
        *,
        actor_id: str | None = None,
        role: str | None = None,
    ) -> None:
        """Append one audit row for a vector-store mutation.

        `id`, `created_at`, and `updated_at` are generated internally —
        callers never pass them in.
        """
        row_id = uuid.uuid4().hex
        timestamp = utc_now_iso()
        is_deleted = 1 if event == "DELETE" else 0

        with self._lock:
            self._conn.execute(
                _INSERT_HISTORY_ROW,
                (
                    row_id,
                    memory_id,
                    old_memory,
                    new_memory,
                    event,
                    timestamp,
                    timestamp,
                    is_deleted,
                    actor_id,
                    role,
                ),
            )
            self._conn.commit()

    def get_history(self, memory_id: str) -> list[dict]:
        """Return all audit rows for `memory_id`, ordered oldest to newest.

        Returns an empty list for an unknown `memory_id` (not an error).
        """
        cursor = self._conn.execute(_SELECT_HISTORY_BY_MEMORY_ID, (memory_id,))
        rows = cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]

    def _row_to_dict(self, row: tuple) -> dict:
        record = dict(zip(_HISTORY_ROW_COLUMNS, row, strict=True))
        record["is_deleted"] = bool(record["is_deleted"])
        return record

    def reset(self) -> None:
        """Drop and recreate the `history` table, discarding all rows."""
        with self._lock:
            self._conn.execute("DROP TABLE IF EXISTS history")
            self._conn.execute(_CREATE_HISTORY_TABLE)
            self._conn.commit()
