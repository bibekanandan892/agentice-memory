"""FastAPI app wrapping one memlayer.Memory instance.

Endpoints:
    POST   /memories               -> Memory.add()
    GET    /memories?user_id=...   -> Memory.get_all()
    POST   /search                 -> Memory.search()
    DELETE /memories/{memory_id}   -> Memory.delete()
    GET    /memories/{memory_id}/history -> Memory.history()

Run with: uv run uvicorn server.app:app --reload
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from memlayer.memory import Memory
from server.dependencies import get_memory
from server.schemas import (
    AddRequest,
    AddResponse,
    HistoryEntry,
    MemoryListResponse,
    MessageResponse,
    SearchRequest,
)

app = FastAPI(
    title="memlayer server",
    description="HTTP surface over the memlayer memory library.",
    version="1.0.0",
)


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "memlayer-server"}


@app.post("/memories", response_model=AddResponse)
def add_memory(payload: AddRequest, memory: Memory = Depends(get_memory)) -> dict:
    return memory.add(
        payload.messages,
        user_id=payload.user_id,
        metadata=payload.metadata,
        infer=payload.infer,
    )


@app.get("/memories", response_model=MemoryListResponse)
def list_memories(user_id: str, memory: Memory = Depends(get_memory)) -> dict:
    return memory.get_all(user_id=user_id)


@app.post("/search", response_model=MemoryListResponse)
def search_memories(payload: SearchRequest, memory: Memory = Depends(get_memory)) -> dict:
    return memory.search(payload.query, user_id=payload.user_id, limit=payload.limit)


@app.delete("/memories/{memory_id}", response_model=MessageResponse)
def delete_memory(memory_id: str, memory: Memory = Depends(get_memory)) -> dict:
    try:
        return memory.delete(memory_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/memories/{memory_id}/history", response_model=list[HistoryEntry])
def memory_history(memory_id: str, memory: Memory = Depends(get_memory)) -> list[dict]:
    return memory.history(memory_id)
