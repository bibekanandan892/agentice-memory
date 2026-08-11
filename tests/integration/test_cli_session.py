"""Integration test: a scripted REPL session driven through
memento.cli.run_prompt_loop with fakes for both LLMs and real (tmp_path)
SQLite-backed stores.

Chat-triggered background writes are asynchronous by design (HLD N2/N3), so
assertions that depend on a write having completed always call
assistant.shutdown() first (exactly what memento.cli.main()'s finally block
does) rather than racing the writer thread — see docs/design/03-lld-memento.md
§3's threading invariants.
"""

import io
import json
from unittest.mock import patch

from rich.console import Console

from memento.assistant import Assistant
from memento.cli import run_prompt_loop
from memento.commands import CommandContext, build_default_registry
from memento.transcript import TranscriptStore
from tests.conftest import FakeLLM


def _run_scripted_session(assistant, context, scripted_inputs):
    registry = build_default_registry()
    buf = io.StringIO()
    console = Console(file=buf, width=200, force_terminal=False)

    inputs = iter(scripted_inputs)

    def fake_input(*_args, **_kwargs):
        try:
            return next(inputs)
        except StopIteration as exc:
            raise EOFError from exc

    with patch.object(Console, "input", fake_input):
        run_prompt_loop(console, assistant, registry, context)

    return buf.getvalue()


def test_chat_turn_reply_is_shown_and_memory_persists_after_shutdown(tmp_path, fake_embedder):
    from memlayer.memory import Memory
    from memlayer.storage.history import SQLiteHistoryStore
    from memlayer.vector_stores.local import LocalVectorStore

    memory_llm = FakeLLM()
    chat_llm = FakeLLM()
    memory = Memory(
        llm=memory_llm,
        embedder=fake_embedder,
        vector_store=LocalVectorStore(db_path=tmp_path / "v.db"),
        history_store=SQLiteHistoryStore(db_path=tmp_path / "h.db"),
    )
    transcript = TranscriptStore(db_path=tmp_path / "t.db")
    assistant = Assistant(memory=memory, transcript=transcript, chat_llm=chat_llm)
    context = CommandContext(memory=memory, active_user_id="alice")

    chat_llm.queue("Nice to meet you!")
    memory_llm.queue(
        json.dumps({"facts": [{"text": "Likes filter coffee", "category": "semantic"}]})
    )
    memory_llm.queue(json.dumps({"memory": [{"text": "Likes filter coffee", "event": "ADD"}]}))

    output = _run_scripted_session(assistant, context, ["I love filter coffee", "/exit"])
    assistant.shutdown(timeout=2.0)

    assert "Nice to meet you!" in output
    assert "Goodbye!" in output

    stored = memory.get_all(user_id="alice")["results"]
    assert len(stored) == 1
    assert stored[0]["memory"] == "Likes filter coffee"


def test_commands_session_lists_forgets_and_isolates_by_user(tmp_path, fake_embedder):
    from memlayer.memory import Memory
    from memlayer.storage.history import SQLiteHistoryStore
    from memlayer.vector_stores.local import LocalVectorStore

    memory = Memory(
        llm=FakeLLM(),
        embedder=fake_embedder,
        vector_store=LocalVectorStore(db_path=tmp_path / "v.db"),
        history_store=SQLiteHistoryStore(db_path=tmp_path / "h.db"),
    )
    memory.vector_store.insert(
        "alice-mem",
        fake_embedder.embed("Likes filter coffee"),
        {
            "user_id": "alice",
            "data": "Likes filter coffee",
            "hash": "h1",
            "memory_category": "semantic",
            "created_at": "t0",
            "updated_at": "t0",
        },
    )
    memory.vector_store.insert(
        "bob-mem",
        fake_embedder.embed("Likes green tea"),
        {
            "user_id": "bob",
            "data": "Likes green tea",
            "hash": "h2",
            "memory_category": "semantic",
            "created_at": "t0",
            "updated_at": "t0",
        },
    )

    transcript = TranscriptStore(db_path=tmp_path / "t.db")
    assistant = Assistant(memory=memory, transcript=transcript, chat_llm=FakeLLM())
    context = CommandContext(memory=memory, active_user_id="alice")

    output = _run_scripted_session(
        assistant,
        context,
        ["/memories", "/user bob", "/memories", "/forget bob-mem", "/memories", "/exit"],
    )

    assert "Likes filter coffee" in output
    assert "Switched to user: bob" in output
    assert "Likes green tea" in output
    assert "Memory deleted successfully" in output
    assert "No memories yet for bob" in output

    # alice's memory was never touched by any of bob's commands.
    assert memory.get_all(user_id="alice")["results"][0]["memory"] == "Likes filter coffee"
