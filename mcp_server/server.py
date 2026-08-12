"""MCP server exposing memlayer as four tools: save_memory, search_memory,
list_memories, forget_memory.

Reuses server.dependencies.get_memory() (the same lazily-built Memory
singleton the FastAPI server uses) so the CLI, the REST server, and this MCP
server all operate on the same memlayer.Memory public API — no duplicated
wiring logic.

Run with: uv run python -m mcp_server.server
Claude Desktop config: see the README's "Optional: MCP server" section.
"""

from __future__ import annotations

from typing import Any

# mcp SDK v2.0 renamed the older FastMCP class to MCPServer and moved it to
# mcp.server.mcpserver — verified against the installed 2.0.0 package since
# most tutorials/examples online still reference `from mcp.server.fastmcp
# import FastMCP`, which no longer exists in this version.
from mcp.server.mcpserver import MCPServer

from memlayer.errors import MemLayerError
from server.dependencies import get_memory

app = MCPServer(
    name="memlayer",
    version="1.0.0",
    instructions=(
        "Tools for saving, searching, listing, and forgetting a user's personal memories, "
        "backed by the memlayer library's two-phase extract/reconcile pipeline."
    ),
)


@app.tool()
def save_memory(text: str, user_id: str) -> dict[str, Any]:
    """Save a memory for a user. The text is run through memlayer's extraction
    pipeline, which pulls out durable facts and reconciles them against
    existing memories (adding, updating, or deleting as appropriate) rather
    than storing the raw text verbatim.
    """
    return get_memory().add(text, user_id=user_id)


@app.tool()
def search_memory(query: str, user_id: str, limit: int = 5) -> dict[str, Any]:
    """Search a user's saved memories for facts relevant to a query."""
    return get_memory().search(query, user_id=user_id, limit=limit)


@app.tool()
def list_memories(user_id: str) -> dict[str, Any]:
    """List every memory currently saved for a user."""
    return get_memory().get_all(user_id=user_id)


@app.tool()
def forget_memory(memory_id: str) -> dict[str, Any]:
    """Delete a specific memory by its id."""
    try:
        return get_memory().delete(memory_id)
    except (ValueError, MemLayerError) as exc:
        return {"error": str(exc)}


def main() -> None:
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
