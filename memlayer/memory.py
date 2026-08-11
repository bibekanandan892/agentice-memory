"""The Memory facade — public API, provider wiring, and (in later Phase 1
tasks) the two-phase add() pipeline.

See docs/design/02-lld-memlayer.md §7-8. This module is implemented across
three tasks: 1.7 (constructor, from_config, read APIs — this pass), 1.8
(add()), 1.9 (update/delete/delete_all/reset).
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from memlayer.config import EmbedderConfig, LlmConfig, MemoryConfig, VectorStoreConfig
from memlayer.embeddings.base import EmbeddingBase
from memlayer.errors import ConfigError, LLMResponseError, ScopeError
from memlayer.llms.base import LLMBase
from memlayer.prompts import build_fact_retrieval_messages, build_update_memory_messages
from memlayer.storage.history import SQLiteHistoryStore
from memlayer.utils import md5_hash, parse_messages, safe_json_loads, utc_now_iso
from memlayer.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)

_DEFAULT_SEARCH_LIMIT = 5
_DEFAULT_GET_ALL_LIMIT = 100
_NEIGHBOR_SEARCH_TOP_K = 5
_DEFAULT_MEMORY_CATEGORY = "semantic"
_IDENTITY_KEYS = ("user_id", "agent_id", "run_id")

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

    # -- write API: add() ------------------------------------------------

    def add(
        self,
        messages: str | list[dict[str, Any]],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
    ) -> dict[str, Any]:
        scope_key, identity = self._scope_or_raise(user_id, agent_id, run_id)
        clean_metadata = _strip_identity_keys(metadata or {})

        if not infer:
            return self._add_infer_false(messages, scope_key, identity, clean_metadata)
        return self._add_infer_true(messages, scope_key, identity, clean_metadata)

    def _add_infer_false(
        self,
        messages: str | list[dict[str, Any]],
        scope_key: str,
        identity: dict[str, str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Store each non-system message verbatim, with no LLM extraction call."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        results = []
        for message in messages:
            if message.get("role") == "system":
                continue
            text = message["content"]
            payload = self._build_payload(
                text, _DEFAULT_MEMORY_CATEGORY, identity, metadata, extra={"role": message["role"]}
            )
            if "name" in message:
                payload["actor_id"] = message["name"]
            results.append(self._create_memory(text, self.embedder.embed(text), payload))
        return {"results": results}

    def _add_infer_true(
        self,
        messages: str | list[dict[str, Any]],
        scope_key: str,
        identity: dict[str, str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        facts = self._extract_facts(messages)
        if not facts:
            return {"results": []}

        embeddings, existing_memories = self._retrieve_neighbors(facts, scope_key)
        temp_uuid_map, existing_view = self._build_uuid_remap(existing_memories)

        memory_items = self._reconcile(existing_view, facts)
        category_by_text = {fact["text"]: fact["category"] for fact in facts}

        results = self._apply_events(
            memory_items, temp_uuid_map, embeddings, identity, metadata, category_by_text
        )
        return {"results": results}

    def _extract_facts(self, messages: str | list[dict[str, Any]]) -> list[dict[str, str]]:
        """LLM call 1: pull candidate facts out of the transcript.

        Graceful degradation: any parse failure (or an explicitly empty
        "facts" list) returns [] rather than raising — a failed guess about
        what's worth remembering should never crash the caller's chat turn.
        """
        transcript = parse_messages(messages)
        raw_response = self.llm.generate_response(
            build_fact_retrieval_messages(transcript), response_format="json"
        )
        try:
            parsed = safe_json_loads(raw_response)
        except LLMResponseError:
            logger.warning("Fact extraction returned unparseable JSON; treating as no facts.")
            return []

        raw_facts = parsed.get("facts", []) if isinstance(parsed, dict) else []
        return [_normalize_fact(fact) for fact in raw_facts]

    def _retrieve_neighbors(
        self, facts: list[dict[str, str]], scope_key: str
    ) -> tuple[dict[str, list[float]], list[dict[str, str]]]:
        """Embed each fact once (cached for reuse at write time) and collect
        the deduplicated set of existing memories that might need reconciling.
        """
        embeddings: dict[str, list[float]] = {}
        neighbors_by_id: dict[str, dict[str, str]] = {}

        for fact in facts:
            text = fact["text"]
            vector = self.embedder.embed(text)
            embeddings[text] = vector
            for point in self.vector_store.search(vector, scope_key, top_k=_NEIGHBOR_SEARCH_TOP_K):
                neighbors_by_id[point.id] = {"id": point.id, "text": point.payload.get("data", "")}

        return embeddings, list(neighbors_by_id.values())

    def _build_uuid_remap(
        self, existing_memories: list[dict[str, str]]
    ) -> tuple[dict[str, str], list[dict[str, str]]]:
        """Anti-hallucination mechanism: the reconciliation LLM only ever sees
        integer-string ids, never real UUIDs, so it cannot invent a
        plausible-looking id that happens to collide with a real one.
        """
        temp_uuid_map: dict[str, str] = {}
        existing_view: list[dict[str, str]] = []
        for index, memory in enumerate(existing_memories):
            int_id = str(index)
            temp_uuid_map[int_id] = memory["id"]
            existing_view.append({"id": int_id, "text": memory["text"]})
        return temp_uuid_map, existing_view

    def _reconcile(
        self, existing_view: list[dict[str, str]], facts: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """LLM call 2: decide ADD/UPDATE/DELETE/NONE for each fact against the
        (integer-remapped) existing memories. Graceful degradation on parse
        failure, same rationale as _extract_facts.
        """
        fact_payload = [{"text": f["text"], "category": f["category"]} for f in facts]
        raw_response = self.llm.generate_response(
            build_update_memory_messages(existing_view, fact_payload), response_format="json"
        )
        try:
            parsed = safe_json_loads(raw_response)
        except LLMResponseError:
            logger.warning("Reconciliation returned unparseable JSON; applying no changes.")
            return []
        return parsed.get("memory", []) if isinstance(parsed, dict) else []

    def _apply_events(
        self,
        memory_items: list[dict[str, Any]],
        temp_uuid_map: dict[str, str],
        embeddings: dict[str, list[float]],
        identity: dict[str, str],
        metadata: dict[str, Any],
        category_by_text: dict[str, str],
    ) -> list[dict[str, Any]]:
        results = []
        for item in memory_items:
            text = item.get("text")
            event = item.get("event")
            if not text or event == "NONE":
                continue
            try:
                applied = self._apply_single_event(
                    item, text, event, temp_uuid_map, embeddings, identity, metadata,
                    category_by_text,
                )
            except Exception:
                logger.warning("Skipping event %r for text %r due to an error.", event, text,
                                exc_info=True)
                continue
            if applied is not None:
                results.append(applied)
        return results

    def _apply_single_event(
        self,
        item: dict[str, Any],
        text: str,
        event: str,
        temp_uuid_map: dict[str, str],
        embeddings: dict[str, list[float]],
        identity: dict[str, str],
        metadata: dict[str, Any],
        category_by_text: dict[str, str],
    ) -> dict[str, Any] | None:
        vector = embeddings.get(text) or self.embedder.embed(text)
        category = category_by_text.get(text, _DEFAULT_MEMORY_CATEGORY)

        if event == "ADD":
            payload = self._build_payload(text, category, identity, metadata)
            return self._create_memory(text, vector, payload)

        real_id = temp_uuid_map.get(item.get("id"))
        if real_id is None:
            logger.warning("Skipping %s event referencing unknown id %r.", event, item.get("id"))
            return None

        if event == "UPDATE":
            return self._update_memory(real_id, text, vector, category)
        if event == "DELETE":
            return self._delete_memory(real_id)
        return None

    def _create_memory(
        self, text: str, vector: list[float], payload: dict[str, Any]
    ) -> dict[str, Any]:
        new_id = uuid.uuid4().hex
        self.vector_store.insert(new_id, vector, payload)
        self.history_store.add_history(new_id, None, text, "ADD")
        return {"id": new_id, "memory": text, "event": "ADD"}

    def _update_memory(
        self, memory_id: str, text: str, vector: list[float], category: str
    ) -> dict[str, Any]:
        existing = self.vector_store.get(memory_id)
        old_text = existing.get("data") if existing else None
        payload = {k: v for k, v in (existing or {}).items() if k != "id"}
        payload.update(
            data=text,
            hash=md5_hash(text),
            memory_category=category,
            updated_at=utc_now_iso(),
        )
        self.vector_store.update(memory_id, vector, payload)
        self.history_store.add_history(memory_id, old_text, text, "UPDATE")
        return {"id": memory_id, "memory": text, "event": "UPDATE", "previous_memory": old_text}

    def _delete_memory(self, memory_id: str) -> dict[str, Any]:
        existing = self.vector_store.get(memory_id)
        old_text = existing.get("data") if existing else None
        self.vector_store.delete(memory_id)
        self.history_store.add_history(memory_id, old_text, None, "DELETE")
        return {"id": memory_id, "memory": old_text, "event": "DELETE"}

    def _build_payload(
        self,
        text: str,
        category: str,
        identity: dict[str, str],
        metadata: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        timestamp = utc_now_iso()
        payload = {
            "data": text,
            "hash": md5_hash(text),
            "memory_category": category,
            "created_at": timestamp,
            "updated_at": timestamp,
            **identity,
            **metadata,
        }
        if extra:
            payload.update(extra)
        return payload

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


_VALID_CATEGORIES = frozenset({"semantic", "episodic", "procedural"})


def _normalize_fact(fact: Any) -> dict[str, str]:
    """Tolerate both {"text": ..., "category": ...} and a bare string from the
    extraction LLM (smaller/older models sometimes drop the category field or
    return a flat list of strings despite the prompt's instructions).
    """
    if isinstance(fact, str):
        return {"text": fact, "category": _DEFAULT_MEMORY_CATEGORY}

    text = fact.get("text", "")
    category = fact.get("category", _DEFAULT_MEMORY_CATEGORY)
    if category not in _VALID_CATEGORIES:
        category = _DEFAULT_MEMORY_CATEGORY
    return {"text": text, "category": category}


def _strip_identity_keys(metadata: dict[str, Any]) -> dict[str, Any]:
    """Identity is immutable after creation and can never be smuggled in
    through caller-supplied metadata (must-not-skip mechanism #5)."""
    return {key: value for key, value in metadata.items() if key not in _IDENTITY_KEYS}


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
