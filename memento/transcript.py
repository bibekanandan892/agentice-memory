"""TranscriptStore — the raw, verbatim chat log (the episodic store, owned by
memento and never touched by memlayer — HLD §7's two-store separation).

See docs/design/03-lld-memento.md §1.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from memlayer.utils import utc_now_iso

logger = logging.getLogger(__name__)

_CREATE_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

_INSERT_MESSAGE = """
INSERT INTO messages (id, session_id, user_id, role, content, created_at)
VALUES (?, ?, ?, ?, ?, ?)
"""

_SELECT_RECENT = """
SELECT role, content, created_at FROM messages
WHERE user_id = ?
ORDER BY created_at DESC, rowid DESC
LIMIT ?
"""

_DEFAULT_RECENT_COUNT = 6


class TranscriptStore:
    """Verbatim audit log of every message a user has ever sent or received.

    Never searched at read time (that's what the distilled memlayer.Memory
    vector store is for) — only ever read back as recent context for prompt
    assembly, or for a human to scroll through later.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.execute(_CREATE_MESSAGES_TABLE)
        self._conn.commit()

    def _execute(self, sql: str, params: tuple[Any, ...]) -> None:
        self._conn.execute(sql, params)
        self._conn.commit()

    def log(self, session_id: str, user_id: str, role: str, content: str) -> None:
        """Append one message. Never raises — a transcript-write failure is
        logged, never surfaced to the user; it's an audit trail, not the
        critical path.
        """
        try:
            self._execute(
                _INSERT_MESSAGE,
                (uuid.uuid4().hex, session_id, user_id, role, content, utc_now_iso()),
            )
        except Exception:
            logger.warning(
                "Failed to log transcript message for user %r.", user_id, exc_info=True
            )

    def recent(self, user_id: str, n: int = _DEFAULT_RECENT_COUNT) -> list[dict[str, str]]:
        """Last n messages for this user, oldest-first, for prompt assembly."""
        cursor = self._conn.execute(_SELECT_RECENT, (user_id, n))
        rows = cursor.fetchall()
        return [{"role": role, "content": content} for role, content, _created_at in reversed(rows)]

    def reset(self) -> None:
        self._conn.execute("DELETE FROM messages")
        self._conn.commit()
