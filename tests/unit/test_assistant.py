"""Tests for memento.assistant.Assistant — read path, prompt assembly, and
the threaded write path contract.

See docs/design/03-lld-memento.md §2-3. Uses a small FakeMemory double (not
a real memlayer.Memory) so these tests exercise Assistant's own orchestration
logic in isolation — the real add()/search() pipeline is already covered by
memlayer's own test suite.
"""

from memento.assistant import Assistant
from memento.transcript import TranscriptStore


class FakeMemory:
    def __init__(self, search_results=None, add_side_effect=None):
        self.search_results = search_results or []
        self.add_calls = []
        self.add_side_effect = add_side_effect

    def search(self, query, *, user_id, limit=5):
        return {"results": self.search_results}

    def add(self, messages, *, user_id, metadata=None, infer=True):
        self.add_calls.append({"messages": messages, "user_id": user_id})
        if self.add_side_effect is not None:
            raise self.add_side_effect
        return {"results": [{"id": "new-1", "memory": "something", "event": "ADD"}]}


def make_assistant(tmp_path, fake_llm, search_results=None, add_side_effect=None):
    transcript = TranscriptStore(db_path=tmp_path / "transcript.db")
    memory = FakeMemory(search_results=search_results, add_side_effect=add_side_effect)
    assistant = Assistant(memory=memory, transcript=transcript, chat_llm=fake_llm)
    return assistant, memory, transcript


class TestChatReadPath:
    def test_chat_returns_the_llm_reply(self, tmp_path, fake_llm):
        fake_llm.queue("Hello! How can I help?")
        assistant, _memory, _transcript = make_assistant(tmp_path, fake_llm)

        reply = assistant.chat("hi there", user_id="alice")
        assistant.shutdown(timeout=2.0)

        assert reply == "Hello! How can I help?"

    def test_chat_logs_both_turns_to_transcript(self, tmp_path, fake_llm):
        fake_llm.queue("Hello!")
        assistant, _memory, transcript = make_assistant(tmp_path, fake_llm)

        assistant.chat("hi there", user_id="alice")
        assistant.shutdown(timeout=2.0)

        recent = transcript.recent("alice", n=10)
        assert [m["role"] for m in recent] == ["user", "assistant"]
        assert recent[0]["content"] == "hi there"
        assert recent[1]["content"] == "Hello!"

    def test_system_prompt_includes_retrieved_memories(self, tmp_path, fake_llm):
        fake_llm.queue("reply")
        results = [{"id": "1", "memory": "Likes filter coffee", "memory_category": "semantic"}]
        assistant, _memory, _transcript = make_assistant(tmp_path, fake_llm, search_results=results)

        assistant.chat("what do I like?", user_id="alice")
        assistant.shutdown(timeout=2.0)

        sent_messages = fake_llm.prompts_seen[0]
        system_message = next(m for m in sent_messages if m["role"] == "system")
        assert "Likes filter coffee" in system_message["content"]
        assert "semantic" in system_message["content"]

    def test_system_prompt_omits_memory_block_when_no_results(self, tmp_path, fake_llm):
        fake_llm.queue("reply")
        assistant, _memory, _transcript = make_assistant(tmp_path, fake_llm, search_results=[])

        assistant.chat("hello", user_id="alice")
        assistant.shutdown(timeout=2.0)

        system_message = next(m for m in fake_llm.prompts_seen[0] if m["role"] == "system")
        assert "What I remember" not in system_message["content"]


class TestWritePath:
    def test_add_is_called_with_exactly_the_two_new_turns_after_reply(self, tmp_path, fake_llm):
        fake_llm.queue("Hello!")
        assistant, memory, _transcript = make_assistant(tmp_path, fake_llm)

        assistant.chat("hi there", user_id="alice")
        assistant.shutdown(timeout=2.0)

        assert len(memory.add_calls) == 1
        call = memory.add_calls[0]
        assert call["user_id"] == "alice"
        roles = [m["role"] for m in call["messages"]]
        contents = [m["content"] for m in call["messages"]]
        assert roles == ["user", "assistant"]
        assert contents == ["hi there", "Hello!"]

    def test_chat_returns_before_write_necessarily_completes(self, tmp_path, fake_llm):
        # chat() itself must not block on the writer thread — we can't assert
        # timing directly in a unit test, but we CAN assert that chat()
        # returns a value without requiring shutdown() to have been called
        # first (i.e. it doesn't join the writer internally).
        fake_llm.queue("Hello!")
        assistant, _memory, _transcript = make_assistant(tmp_path, fake_llm)

        reply = assistant.chat("hi there", user_id="alice")
        assert reply == "Hello!"
        assistant.shutdown(timeout=2.0)  # cleanup

    def test_successful_write_result_reaches_poll_write_result(self, tmp_path, fake_llm):
        fake_llm.queue("Hello!")
        assistant, _memory, _transcript = make_assistant(tmp_path, fake_llm)

        assistant.chat("hi there", user_id="alice")
        assistant.shutdown(timeout=2.0)

        status, result = assistant.poll_write_result()
        assert status == "ok"
        assert result["results"][0]["event"] == "ADD"

    def test_poll_write_result_returns_none_when_nothing_ready(self, tmp_path, fake_llm):
        assistant, _memory, _transcript = make_assistant(tmp_path, fake_llm)
        assert assistant.poll_write_result() is None

    def test_write_failure_is_caught_and_never_propagates_out_of_chat_or_shutdown(
        self, tmp_path, fake_llm
    ):
        fake_llm.queue("Hello!")
        assistant, _memory, _transcript = make_assistant(
            tmp_path, fake_llm, add_side_effect=RuntimeError("simulated storage failure")
        )

        reply = assistant.chat("hi there", user_id="alice")  # must not raise
        assistant.shutdown(timeout=2.0)  # must not raise

        assert reply == "Hello!"
        status, message = assistant.poll_write_result()
        assert status == "error"
        assert "simulated storage failure" in message

    def test_shutdown_joins_all_pending_writer_threads(self, tmp_path, fake_llm):
        fake_llm.queue("first")
        fake_llm.queue("second")
        assistant, memory, _transcript = make_assistant(tmp_path, fake_llm)

        assistant.chat("message one", user_id="alice")
        assistant.chat("message two", user_id="alice")
        assistant.shutdown(timeout=2.0)

        assert all(not t.is_alive() for t in assistant._pending_writers)
        assert len(memory.add_calls) == 2
