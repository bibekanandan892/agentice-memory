"""LocalVectorStore — NumPy cosine search sharded by user_id, SQLite-blob persisted.

Implemented in Phase 1 Task 1.3. See docs/design/02-lld-memlayer.md §4.

Internal structure (the "shard-by-user" design, HLD §10):

    _partitions: dict[user_id, tuple[ids: list[str],
                                      matrix: np.ndarray[N, dims],
                                      payloads: list[dict]]]

search() only ever reads `_partitions[user_id]` — this is the enforced
user-isolation boundary, not a post-hoc filter over a shared pool.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from memlayer.vector_stores.base import ScoredPoint, VectorStoreBase

_DEFAULT_SEARCH_TOP_K = 5
_DEFAULT_GET_ALL_TOP_K = 100

_CREATE_VECTORS_TABLE = """
CREATE TABLE IF NOT EXISTS vectors (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    embedding BLOB,
    payload TEXT
);
"""

_UPSERT_VECTOR_ROW = """
INSERT OR REPLACE INTO vectors (id, user_id, embedding, payload) VALUES (?, ?, ?, ?)
"""

_SELECT_ALL_VECTOR_ROWS = "SELECT id, user_id, embedding, payload FROM vectors"

Partition = tuple[list[str], np.ndarray, list[dict[str, Any]]]


class LocalVectorStore(VectorStoreBase):
    """In-RAM cosine-similarity vector store, sharded per user_id, write-through to SQLite.

    Every mutating call (insert/update/delete/delete_all/reset) updates the
    in-RAM partition and the SQLite row(s) in the same call — there is no
    separate flush step, so a crash never loses a committed write.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._create_schema()

        self._partitions: dict[str, Partition] = {}
        self._id_to_user: dict[str, str] = {}
        self._load_partitions()

    # -- schema / reload ----------------------------------------------------

    def _create_schema(self) -> None:
        self._conn.execute(_CREATE_VECTORS_TABLE)
        self._conn.commit()

    def _load_partitions(self) -> None:
        """Rebuild the in-RAM partitions from whatever is already in SQLite.

        Called once at construction, so a new LocalVectorStore pointed at an
        existing db_path resumes exactly where a previous instance left off.
        """
        rows = self._conn.execute(_SELECT_ALL_VECTOR_ROWS).fetchall()

        grouped: dict[str, list[tuple[str, np.ndarray, dict[str, Any]]]] = {}
        for row_id, user_id, embedding_blob, payload_json in rows:
            vector = np.frombuffer(embedding_blob, dtype=np.float32)
            payload = json.loads(payload_json)
            grouped.setdefault(user_id, []).append((row_id, vector, payload))
            self._id_to_user[row_id] = user_id

        for user_id, entries in grouped.items():
            ids = [row_id for row_id, _vector, _payload in entries]
            matrix = np.vstack([vector for _row_id, vector, _payload in entries])
            payloads = [payload for _row_id, _vector, payload in entries]
            self._partitions[user_id] = (ids, matrix, payloads)

    # -- write path -----------------------------------------------------------

    def insert(self, id: str, vector: list[float], payload: dict[str, Any]) -> None:
        if "user_id" not in payload:
            raise KeyError(
                f"insert() payload must contain 'user_id' (got keys={list(payload.keys())})"
            )
        user_id = payload["user_id"]
        new_row = np.asarray(vector, dtype=np.float32).reshape(1, -1)

        ids, matrix, payloads = self._partitions.get(user_id, ([], None, []))
        new_matrix = new_row if matrix is None else np.vstack([matrix, new_row])
        self._rebuild_partition(user_id, [*ids, id], new_matrix, [*payloads, dict(payload)])

        self._id_to_user[id] = user_id
        self._write_row(id, user_id, new_row[0], payload)

    def update(self, id: str, vector: list[float], payload: dict[str, Any]) -> None:
        user_id = self._require_user_of(id)
        ids, matrix, payloads = self._partitions[user_id]
        index = ids.index(id)

        updated_matrix = matrix.copy()
        updated_matrix[index] = np.asarray(vector, dtype=np.float32)
        updated_payloads = list(payloads)
        updated_payloads[index] = dict(payload)
        self._rebuild_partition(user_id, ids, updated_matrix, updated_payloads)

        self._write_row(id, user_id, updated_matrix[index], payload)

    def delete(self, id: str) -> None:
        user_id = self._require_user_of(id)
        ids, matrix, payloads = self._partitions[user_id]
        index = ids.index(id)

        new_ids = ids[:index] + ids[index + 1 :]
        new_matrix = np.delete(matrix, index, axis=0)
        new_payloads = payloads[:index] + payloads[index + 1 :]
        self._rebuild_partition(user_id, new_ids, new_matrix, new_payloads)

        del self._id_to_user[id]
        self._conn.execute("DELETE FROM vectors WHERE id = ?", (id,))
        self._conn.commit()

    def delete_all(self, user_id: str) -> int:
        partition = self._partitions.pop(user_id, None)
        if partition is None:
            return 0

        ids, _matrix, _payloads = partition
        for row_id in ids:
            del self._id_to_user[row_id]

        self._conn.execute("DELETE FROM vectors WHERE user_id = ?", (user_id,))
        self._conn.commit()
        return len(ids)

    def reset(self) -> None:
        self._partitions = {}
        self._id_to_user = {}
        self._conn.execute("DELETE FROM vectors")
        self._conn.commit()

    # -- read path ------------------------------------------------------------

    def search(
        self, vector: list[float], user_id: str, top_k: int = _DEFAULT_SEARCH_TOP_K
    ) -> list[ScoredPoint]:
        """Only ever reads `_partitions[user_id]` — the enforced isolation boundary."""
        partition = self._partitions.get(user_id)
        if not partition:
            return []

        ids, matrix, payloads = partition
        query = np.asarray(vector, dtype=np.float32)
        return self._cosine_topk(query, matrix, ids, payloads, top_k)

    def get(self, id: str) -> dict[str, Any] | None:
        user_id = self._id_to_user.get(id)
        if user_id is None:
            return None

        ids, _matrix, payloads = self._partitions[user_id]
        index = ids.index(id)
        return {"id": id, **payloads[index]}

    def get_all(
        self, user_id: str, top_k: int = _DEFAULT_GET_ALL_TOP_K
    ) -> list[dict[str, Any]]:
        partition = self._partitions.get(user_id)
        if not partition:
            return []

        ids, _matrix, payloads = partition
        rows = [{"id": row_id, **payload} for row_id, payload in zip(ids, payloads, strict=True)]
        return rows[:top_k]

    # -- helpers ----------------------------------------------------------

    def _rebuild_partition(
        self, user_id: str, ids: list[str], matrix: np.ndarray, payloads: list[dict[str, Any]]
    ) -> None:
        """Replace a partition wholesale rather than mutating it in place."""
        if not ids:
            self._partitions.pop(user_id, None)
            return
        self._partitions[user_id] = (ids, matrix, payloads)

    def _require_user_of(self, id: str) -> str:
        user_id = self._id_to_user.get(id)
        if user_id is None:
            raise KeyError(f"No vector found with id={id!r}")
        return user_id

    def _write_row(
        self, id: str, user_id: str, vector: np.ndarray, payload: dict[str, Any]
    ) -> None:
        embedding_blob = np.asarray(vector, dtype=np.float32).tobytes()
        payload_json = json.dumps(payload)
        self._conn.execute(_UPSERT_VECTOR_ROW, (id, user_id, embedding_blob, payload_json))
        self._conn.commit()

    @staticmethod
    def _cosine_topk(
        query: np.ndarray,
        matrix: np.ndarray,
        ids: list[str],
        payloads: list[dict[str, Any]],
        top_k: int,
    ) -> list[ScoredPoint]:
        """L2-normalize `matrix` and `query`, score via matrix @ query, return top-K
        descending. Score is always in [-1, 1]; higher = more similar."""
        query_norm = np.linalg.norm(query)
        normalized_query = query / query_norm if query_norm > 0 else query

        row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe_row_norms = np.where(row_norms > 0, row_norms, 1.0)
        normalized_matrix = matrix / safe_row_norms

        scores = normalized_matrix @ normalized_query
        top_indices = np.argsort(-scores)[:top_k]

        return [
            ScoredPoint(id=ids[i], score=float(scores[i]), payload=dict(payloads[i]))
            for i in top_indices
        ]
