"""Tests for Memory.update/delete/delete_all/reset — completing the public API.

See docs/design/02-lld-memlayer.md §7 (API contract table) and §9
(error taxonomy: delete_all refuses without a scope id).
"""

import pytest

from memlayer.errors import ScopeError


def _insert(memory, memory_id, user_id, text, category="semantic"):
    memory.vector_store.insert(
        memory_id,
        memory.embedder.embed(text),
        {
            "user_id": user_id,
            "data": text,
            "hash": "h",
            "memory_category": category,
            "created_at": "t0",
            "updated_at": "t0",
        },
    )


class TestUpdate:
    def test_update_changes_text_and_returns_success_message(self, memory_with_fakes):
        _insert(memory_with_fakes, "mem-1", "alice", "Likes tea")

        result = memory_with_fakes.update("mem-1", "Loves green tea")

        assert result == {"message": "Memory updated successfully!"}
        assert memory_with_fakes.get("mem-1")["memory"] == "Loves green tea"

    def test_update_preserves_created_at_and_category(self, memory_with_fakes):
        _insert(memory_with_fakes, "mem-1", "alice", "Likes tea", category="episodic")

        memory_with_fakes.update("mem-1", "Loves green tea")

        stored = memory_with_fakes.get("mem-1")
        assert stored["created_at"] == "t0"
        assert stored["memory_category"] == "episodic"

    def test_update_writes_a_history_row(self, memory_with_fakes):
        _insert(memory_with_fakes, "mem-1", "alice", "Likes tea")
        memory_with_fakes.update("mem-1", "Loves green tea")

        history = memory_with_fakes.history("mem-1")
        assert len(history) == 1
        assert history[0]["event"] == "UPDATE"
        assert history[0]["old_memory"] == "Likes tea"
        assert history[0]["new_memory"] == "Loves green tea"

    def test_update_unknown_id_raises_clear_error(self, memory_with_fakes):
        with pytest.raises(ValueError, match="not found"):
            memory_with_fakes.update("does-not-exist", "new text")


class TestDelete:
    def test_delete_removes_memory_and_returns_success_message(self, memory_with_fakes):
        _insert(memory_with_fakes, "mem-1", "alice", "Likes tea")

        result = memory_with_fakes.delete("mem-1")

        assert result == {"message": "Memory deleted successfully!"}
        assert memory_with_fakes.get("mem-1") is None

    def test_delete_writes_a_history_row(self, memory_with_fakes):
        _insert(memory_with_fakes, "mem-1", "alice", "Likes tea")
        memory_with_fakes.delete("mem-1")

        history = memory_with_fakes.history("mem-1")
        assert len(history) == 1
        assert history[0]["event"] == "DELETE"
        assert history[0]["is_deleted"] is True

    def test_delete_unknown_id_raises_clear_error(self, memory_with_fakes):
        with pytest.raises(ValueError, match="not found"):
            memory_with_fakes.delete("does-not-exist")


class TestDeleteAll:
    def test_delete_all_requires_a_scope_id(self, memory_with_fakes):
        with pytest.raises(ScopeError):
            memory_with_fakes.delete_all()

    def test_delete_all_removes_every_memory_for_that_user(self, memory_with_fakes):
        _insert(memory_with_fakes, "mem-1", "alice", "fact one")
        _insert(memory_with_fakes, "mem-2", "alice", "fact two")

        result = memory_with_fakes.delete_all(user_id="alice")

        assert result == {"message": "Memories deleted successfully!"}
        assert memory_with_fakes.get_all(user_id="alice")["results"] == []

    def test_delete_all_leaves_other_users_untouched(self, memory_with_fakes):
        _insert(memory_with_fakes, "mem-1", "alice", "alice's fact")
        _insert(memory_with_fakes, "mem-2", "bob", "bob's fact")

        memory_with_fakes.delete_all(user_id="alice")

        assert memory_with_fakes.get_all(user_id="bob")["results"] != []

    def test_delete_all_writes_one_history_row_per_deleted_memory(self, memory_with_fakes):
        _insert(memory_with_fakes, "mem-1", "alice", "fact one")
        _insert(memory_with_fakes, "mem-2", "alice", "fact two")

        memory_with_fakes.delete_all(user_id="alice")

        assert len(memory_with_fakes.history("mem-1")) == 1
        assert memory_with_fakes.history("mem-1")[0]["event"] == "DELETE"
        assert len(memory_with_fakes.history("mem-2")) == 1

    def test_delete_all_on_empty_user_is_a_no_op(self, memory_with_fakes):
        result = memory_with_fakes.delete_all(user_id="nobody-home")
        assert result == {"message": "Memories deleted successfully!"}


class TestReset:
    def test_reset_clears_all_vector_data_and_history(self, memory_with_fakes):
        _insert(memory_with_fakes, "mem-1", "alice", "fact one")
        memory_with_fakes.history_store.add_history("mem-1", None, "fact one", "ADD")

        result = memory_with_fakes.reset()

        assert result == {"message": "All memories reset."}
        assert memory_with_fakes.get_all(user_id="alice")["results"] == []
        assert memory_with_fakes.history("mem-1") == []
