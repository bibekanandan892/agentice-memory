"""The Memory facade — public API, provider wiring, and (in later Phase 1
tasks) the two-phase add() pipeline.

See docs/design/02-lld-memlayer.md §7-8. This module is implemented across
three tasks: 1.7 (constructor, from_config, read APIs — this pass), 1.8
(add()), 1.9 (update/delete/delete_all/reset).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from memlayer.config import EmbedderConfig, LlmConfig, MemoryConfig, VectorStoreConfig
from memlayer.embeddings.base import EmbeddingBase
from memlayer.errors import ConfigError, ScopeError
from memlayer.llms.base import LLMBase
from memlayer.storage.history import SQLiteHistoryStore
from memlayer.vector_stores.base import VectorStoreBase

_DEFAULT_SEARCH_LIMIT = 5
_DEFAULT_GET_ALL_LIMIT = 100

# Payload keys promoted to the top level of a formatted read result (alongside
# the always-present id/memory/created_at/updated_at). Everything else in the
# payload is nested under "metadata". `memory_category` is promoted here even
# though it's not part of Mem0's own contract, per the class-notes extension —
# see the Mem0 parity table in docs/design/02-lld-memlayer.md §10.
_PROMOTED_PAYLOAD_KEYS = ("user_id", "agent_id", "run_id", "memory_category")
_NON_METADATA_KEYS = frozenset(
    {"id", "data", "hash", "created_at", "updated_at", *_PROMOTED_PAYLOAD_KEYS}
)


class Memory:
    """The public memlayer API. Collaborators are injected directly (for tests
    and custom wiring) or built for you by `from_config()`.
    """

    def __init__(
        self,
        llm: LLMBase,
        embedder: EmbeddingBase,
        vector_store: VectorStoreBase,
        history_store: SQLiteHistoryStore,
        config: MemoryConfig | None = None,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store
        self.history_store = history_store
        self.config = config or MemoryConfig()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> Memory:
        """Parse `config` into a MemoryConfig and build concrete providers for
        each plug-in seam via the small factory functions below.
        """
        parsed = MemoryConfig.from_dict(config)
        default_vector_store_path = Path(parsed.history_db_path).parent / "vectors.db"

        return cls(
            llm=_build_llm(parsed.llm),
            embedder=_build_embedder(parsed.embedder),
            vector_store=_build_vector_store(parsed.vector_store, default_vector_store_path),
            history_store=SQLiteHistoryStore(db_path=parsed.history_db_path),
            config=parsed,
        )

    # -- read APIs ------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = _DEFAULT_SEARCH_LIMIT,
    ) -> dict[str, Any]:
        scope_key, _identity = self._scope_or_raise(user_id, agent_id, run_id)
        query_vector = self.embedder.embed(query)
        scored_points = self.vector_store.search(query_vector, user_id=scope_key, top_k=limit)
        results = [
            self._format_result({"id": point.id, **point.payload}, score=point.score)
            for point in scored_points
        ]
        return {"results": results}

    def get(self, memory_id: str) -> dict[str, Any] | None:
        payload = self.vector_store.get(memory_id)
        if payload is None:
            return None
        return self._format_result(payload)

    def get_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = _DEFAULT_GET_ALL_LIMIT,
    ) -> dict[str, Any]:
        scope_key, _identity = self._scope_or_raise(user_id, agent_id, run_id)
        payloads = self.vector_store.get_all(scope_key, top_k=limit)
        return {"results": [self._format_result(payload) for payload in payloads]}

    def history(self, memory_id: str) -> list[dict[str, Any]]:
        return self.history_store.get_history(memory_id)

    # -- internal helpers -------------------------------------------------

    def _scope_or_raise(
        self, user_id: str | None, agent_id: str | None, run_id: str | None
    ) -> tuple[str, dict[str, str]]:
        """Validate at least one scope id is present (the user-isolation safety
        rail — HLD F5) and compute the single shard/partition key the vector
        store uses.

        Design decision (Phase 1.7): the project's storage layer shards by a
        single key (VectorStoreBase's `user_id` parameter). When a caller
        provides more than one of user_id/agent_id/run_id, user_id wins as the
        shard key, then agent_id, then run_id — but every identity field that
        WAS provided is still faithfully stored in the payload and promoted
        back out on read. This mirrors Mem0's API surface (all three
        dimensions are accepted) while keeping this project's storage layer
        as simple as the class notes' single-axis "shard by user_id" design.
        """
        if not (user_id or agent_id or run_id):
            raise ScopeError(
                "At least one of user_id, agent_id, or run_id is required for this operation."
            )
        identity = {
            key: value
            for key, value in (("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id))
            if value is not None
        }
        scope_key = user_id or agent_id or run_id
        return scope_key, identity

    def _format_result(self, payload: dict[str, Any], score: float | None = None) -> dict[str, Any]:
        """payload must already include "id" (LocalVectorStore's get/get_all
        merge it in; search() results are normalized to include it by the
        caller before this is invoked).
        """
        result: dict[str, Any] = {
            "id": payload["id"],
            "memory": payload.get("data", ""),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }
        for key in _PROMOTED_PAYLOAD_KEYS:
            if key in payload:
                result[key] = payload[key]
        if score is not None:
            result["score"] = score

        result["metadata"] = {
            key: value for key, value in payload.items() if key not in _NON_METADATA_KEYS
        }
        return result


def _build_llm(llm_config: LlmConfig) -> LLMBase:
    if llm_config.provider == "gemini":
        from memlayer.llms.gemini import GeminiLLM

        return GeminiLLM(**llm_config.config)
    raise ConfigError(f"Unknown llm provider {llm_config.provider!r}")


def _build_embedder(embedder_config: EmbedderConfig) -> EmbeddingBase:
    if embedder_config.provider == "sentence_transformer":
        from memlayer.embeddings.sentence_transformer import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(**embedder_config.config)
    if embedder_config.provider == "gemini":
        from memlayer.embeddings.gemini import GeminiEmbedder

        return GeminiEmbedder(**embedder_config.config)
    raise ConfigError(f"Unknown embedder provider {embedder_config.provider!r}")


def _build_vector_store(
    vector_store_config: VectorStoreConfig, default_db_path: Path
) -> VectorStoreBase:
    if vector_store_config.provider == "local":
        from memlayer.vector_stores.local import LocalVectorStore

        db_path = vector_store_config.config.get("db_path", default_db_path)
        return LocalVectorStore(db_path=db_path)
    raise ConfigError(f"Unknown vector_store provider {vector_store_config.provider!r}")
