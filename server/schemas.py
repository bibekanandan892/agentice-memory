"""Pydantic request/response models mirroring memlayer.Memory's API contract
(docs/design/02-lld-memlayer.md §7).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AddRequest(BaseModel):
    messages: str | list[dict[str, Any]]
    user_id: str
    metadata: dict[str, Any] | None = None
    infer: bool = True


class MemoryEventResult(BaseModel):
    id: str
    memory: str
    event: str
    previous_memory: str | None = None


class AddResponse(BaseModel):
    results: list[MemoryEventResult]


class SearchRequest(BaseModel):
    query: str
    user_id: str
    limit: int = 5


class MemoryRecord(BaseModel):
    id: str
    memory: str
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    memory_category: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryListResponse(BaseModel):
    results: list[MemoryRecord]


class MessageResponse(BaseModel):
    message: str


class HistoryEntry(BaseModel):
    id: str
    memory_id: str
    old_memory: str | None
    new_memory: str | None
    event: str
    created_at: str
    updated_at: str | None
    is_deleted: bool
    actor_id: str | None = None
    role: str | None = None
