"""Tests for memento.commands — CommandRegistry, CommandContext, and the
slash-command handlers (/memories /forget /history /user /help /exit).

See docs/design/03-lld-memento.md §5.
"""

import io

from rich.console import Console
from rich.table import Table

from memento.commands import CommandContext, build_default_registry


def render_to_text(render) -> str:
    if isinstance(render, Table):
        buf = io.StringIO()
        Console(file=buf, width=200).print(render)
        return buf.getvalue()
    return str(render)


class FakeMemory:
    def __init__(self):
        self._rows: dict[str, dict] = {}
        self._history: dict[str, list[dict]] = {}
        self.deleted: list[str] = []

    def seed(self, memory_id: str, user_id: str, text: str, category: str = "semantic"):
        self._rows[memory_id] = {
            "id": memory_id,
            "memory": text,
            "user_id": user_id,
            "memory_category": category,
            "created_at": "2026-01-01T00:00:00+00:00",
        }

    def get_all(self, *, user_id):
        return {"results": [row for row in self._rows.values() if row["user_id"] == user_id]}

    def delete(self, memory_id):
        self.deleted.append(memory_id)
        self._rows.pop(memory_id, None)
        return {"message": "Memory deleted successfully!"}

    def history(self, memory_id):
        return self._history.get(memory_id, [])


def make_context(user_id="alice"):
    memory = FakeMemory()
    context = CommandContext(memory=memory, active_user_id=user_id)
    return context, memory


class TestDispatch:
    def test_unknown_command_returns_help_hint(self):
        registry = build_default_registry()
        context, _memory = make_context()
        result = registry.dispatch("/nope", context)
        assert result.handled is True
        assert "Unknown command" in render_to_text(result.render)


class TestMemoriesCommand:
    def test_empty_memories_shows_friendly_message(self):
        registry = build_default_registry()
        context, _memory = make_context()
        result = registry.dispatch("/memories", context)
        assert "No memories yet" in render_to_text(result.render)

    def test_lists_memories_for_active_user(self):
        registry = build_default_registry()
        context, memory = make_context()
        memory.seed("mem-1", "alice", "Likes filter coffee")
        result = registry.dispatch("/memories", context)
        text = render_to_text(result.render)
        assert "Likes filter coffee" in text

    def test_does_not_show_other_users_memories(self):
        registry = build_default_registry()
        context, memory = make_context(user_id="alice")
        memory.seed("mem-1", "bob", "bob's secret")
        result = registry.dispatch("/memories", context)
        assert "bob's secret" not in render_to_text(result.render)


class TestForgetCommand:
    def test_usage_message_when_no_arg_given(self):
        registry = build_default_registry()
        context, _memory = make_context()
        result = registry.dispatch("/forget", context)
        assert "Usage" in render_to_text(result.render)

    def test_unique_prefix_deletes_the_memory(self):
        registry = build_default_registry()
        context, memory = make_context()
        memory.seed("abcdef12345", "alice", "some fact")
        result = registry.dispatch("/forget abcdef", context)
        assert memory.deleted == ["abcdef12345"]
        assert "deleted" in render_to_text(result.render).lower()

    def test_unknown_prefix_reports_no_match(self):
        registry = build_default_registry()
        context, _memory = make_context()
        result = registry.dispatch("/forget zzzzzz", context)
        assert "No memory found" in render_to_text(result.render)

    def test_ambiguous_prefix_lists_candidates(self):
        registry = build_default_registry()
        context, memory = make_context()
        memory.seed("abc111", "alice", "fact one")
        memory.seed("abc222", "alice", "fact two")
        result = registry.dispatch("/forget abc", context)
        text = render_to_text(result.render)
        assert "Ambiguous" in text
        assert memory.deleted == []


class TestHistoryCommand:
    def test_usage_message_when_no_arg_given(self):
        registry = build_default_registry()
        context, _memory = make_context()
        result = registry.dispatch("/history", context)
        assert "Usage" in render_to_text(result.render)

    def test_shows_history_for_resolved_id(self):
        registry = build_default_registry()
        context, memory = make_context()
        memory.seed("abcdef12345", "alice", "some fact")
        memory._history["abcdef12345"] = [
            {"event": "ADD", "old_memory": None, "new_memory": "some fact", "created_at": "t0"}
        ]
        result = registry.dispatch("/history abcdef", context)
        text = render_to_text(result.render)
        assert "ADD" in text

    def test_no_history_shows_friendly_message(self):
        registry = build_default_registry()
        context, memory = make_context()
        memory.seed("abcdef12345", "alice", "some fact")
        result = registry.dispatch("/history abcdef", context)
        assert "No history" in render_to_text(result.render)


class TestUserCommand:
    def test_switches_active_user(self):
        registry = build_default_registry()
        context, _memory = make_context(user_id="alice")
        registry.dispatch("/user bob", context)
        assert context.active_user_id == "bob"

    def test_no_arg_shows_current_user(self):
        registry = build_default_registry()
        context, _memory = make_context(user_id="alice")
        result = registry.dispatch("/user", context)
        assert "alice" in render_to_text(result.render)

    def test_memories_isolated_after_switching_user(self):
        registry = build_default_registry()
        context, memory = make_context(user_id="alice")
        memory.seed("mem-1", "alice", "alice's fact")
        memory.seed("mem-2", "bob", "bob's fact")

        registry.dispatch("/user bob", context)
        result = registry.dispatch("/memories", context)

        text = render_to_text(result.render)
        assert "bob's fact" in text
        assert "alice's fact" not in text


class TestHelpAndExit:
    def test_help_lists_all_commands(self):
        registry = build_default_registry()
        context, _memory = make_context()
        result = registry.dispatch("/help", context)
        text = render_to_text(result.render)
        for command in ("/memories", "/forget", "/history", "/user", "/help", "/exit"):
            assert command in text

    def test_exit_sets_should_exit(self):
        registry = build_default_registry()
        context, _memory = make_context()
        result = registry.dispatch("/exit", context)
        assert result.should_exit is True
