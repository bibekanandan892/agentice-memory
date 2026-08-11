"""CommandRegistry and slash-command handlers (/memories /forget /history
/user /help /exit).

See docs/design/03-lld-memento.md §5.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rich.table import Table

_SHORT_ID_LENGTH = 8


@dataclass
class CommandContext:
    """Mutable per-session state handlers read and write. `active_user_id`
    is the CLI's single source of truth for the current user — /user writes
    it directly, and the CLI loop just reads context.active_user_id back.
    """

    memory: Any
    active_user_id: str


@dataclass
class CommandResult:
    handled: bool
    render: Any = None
    should_exit: bool = False


CommandHandler = Callable[[str, CommandContext], CommandResult]


class CommandRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, CommandHandler] = {}

    def register(self, name: str, handler: CommandHandler) -> None:
        self._handlers[name] = handler

    def dispatch(self, line: str, context: CommandContext) -> CommandResult:
        parts = line.strip().split(maxsplit=1)
        name = parts[0] if parts else ""
        args = parts[1] if len(parts) > 1 else ""

        handler = self._handlers.get(name)
        if handler is None:
            return CommandResult(handled=True, render=f"Unknown command: {name}. Try /help")
        return handler(args, context)


def _resolve_id(prefix: str, candidates: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    """Resolve a short-id prefix against a list of memory rows.

    Returns (resolved_id, matches). Exactly one match -> (id, []). Zero or
    multiple matches -> (None, matching_ids) so the caller can report
    "no match" or "ambiguous, did you mean one of these?".
    """
    matches = [row["id"] for row in candidates if row["id"].startswith(prefix)]
    if len(matches) == 1:
        return matches[0], []
    return None, matches


def _handle_memories(_args: str, context: CommandContext) -> CommandResult:
    results = context.memory.get_all(user_id=context.active_user_id)["results"]
    if not results:
        return CommandResult(handled=True, render=f"No memories yet for {context.active_user_id}.")

    table = Table(title=f"Memories for {context.active_user_id}")
    table.add_column("id")
    table.add_column("memory")
    table.add_column("category")
    table.add_column("created")
    for row in results:
        table.add_row(
            row["id"][:_SHORT_ID_LENGTH],
            row["memory"],
            row.get("memory_category", ""),
            row.get("created_at", ""),
        )
    return CommandResult(handled=True, render=table)


def _handle_forget(args: str, context: CommandContext) -> CommandResult:
    prefix = args.strip()
    if not prefix:
        return CommandResult(handled=True, render="Usage: /forget <id-prefix>")

    candidates = context.memory.get_all(user_id=context.active_user_id)["results"]
    resolved_id, matches = _resolve_id(prefix, candidates)
    if resolved_id is None:
        return CommandResult(handled=True, render=_no_match_or_ambiguous_message(prefix, matches))

    result = context.memory.delete(resolved_id)
    return CommandResult(handled=True, render=result["message"])


def _handle_history(args: str, context: CommandContext) -> CommandResult:
    prefix = args.strip()
    if not prefix:
        return CommandResult(handled=True, render="Usage: /history <id-prefix>")

    candidates = context.memory.get_all(user_id=context.active_user_id)["results"]
    resolved_id, matches = _resolve_id(prefix, candidates)
    if resolved_id is None:
        return CommandResult(handled=True, render=_no_match_or_ambiguous_message(prefix, matches))

    rows = context.memory.history(resolved_id)
    if not rows:
        return CommandResult(handled=True, render="No history for this memory.")

    table = Table(title=f"History for {resolved_id[:_SHORT_ID_LENGTH]}")
    table.add_column("event")
    table.add_column("old")
    table.add_column("new")
    table.add_column("when")
    for row in rows:
        table.add_row(
            row["event"],
            row.get("old_memory") or "",
            row.get("new_memory") or "",
            row["created_at"],
        )
    return CommandResult(handled=True, render=table)


def _handle_user(args: str, context: CommandContext) -> CommandResult:
    name = args.strip()
    if not name:
        return CommandResult(handled=True, render=f"Current user: {context.active_user_id}")
    context.active_user_id = name
    return CommandResult(handled=True, render=f"Switched to user: {name}")


def _handle_help(_args: str, _context: CommandContext) -> CommandResult:
    lines = [
        "Available commands:",
        "  /memories        list your saved memories",
        "  /forget <id>     delete a memory by id prefix",
        "  /history <id>    show a memory's change history",
        "  /user <name>     switch the active user",
        "  /help            show this help",
        "  /exit            quit",
    ]
    return CommandResult(handled=True, render="\n".join(lines))


def _handle_exit(_args: str, _context: CommandContext) -> CommandResult:
    return CommandResult(handled=True, render="Goodbye!", should_exit=True)


def _no_match_or_ambiguous_message(prefix: str, matches: list[str]) -> str:
    if matches:
        shown = ", ".join(m[:_SHORT_ID_LENGTH] for m in matches)
        return f"Ambiguous id prefix {prefix!r} — matches: {shown}"
    return f"No memory found matching {prefix!r}."


def build_default_registry() -> CommandRegistry:
    registry = CommandRegistry()
    registry.register("/memories", _handle_memories)
    registry.register("/forget", _handle_forget)
    registry.register("/history", _handle_history)
    registry.register("/user", _handle_user)
    registry.register("/help", _handle_help)
    registry.register("/exit", _handle_exit)
    return registry
