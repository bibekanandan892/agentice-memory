"""Assistant — read path (search -> prompt assembly -> reply) and threaded
write path (add() after the reply is already shown to the user).

See docs/design/03-lld-memento.md §2-3. The threading invariants documented
there are load-bearing: chat() never blocks on the writer thread, a writer
exception is always caught and queued rather than silently swallowed, and
queue.Queue is the only state shared across threads.
"""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from typing import Any

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are Memento, a helpful personal assistant with memory of past conversations. "
    "Answer naturally using the facts below when relevant. Do not mention \"memory\" or "
    "\"database\" mechanics to the user — just use the facts as things you already know."
)

_DEFAULT_SEARCH_LIMIT = 5
_DEFAULT_RECENT_TURNS = 6


class Assistant:
    def __init__(self, memory: Any, transcript: Any, chat_llm: Any, session_id: str | None = None):
        self.memory = memory
        self.transcript = transcript
        self._chat_llm = chat_llm
        self.session_id = session_id or uuid.uuid4().hex
        self._write_results: queue.Queue = queue.Queue()
        self._pending_writers: list[threading.Thread] = []

    def chat(self, text: str, user_id: str) -> str:
        """The read path — synchronous, blocks until the reply is ready."""
        self.transcript.log(self.session_id, user_id, "user", text)

        retrieved = self.memory.search(text, user_id=user_id, limit=_DEFAULT_SEARCH_LIMIT)
        system_prompt = self._build_system_prompt(retrieved["results"])
        recent_turns = self.transcript.recent(user_id, n=_DEFAULT_RECENT_TURNS)
        messages = [{"role": "system", "content": system_prompt}, *recent_turns]

        reply = self._chat_llm.generate_response(messages, response_format="text")
        self.transcript.log(self.session_id, user_id, "assistant", reply)

        self._spawn_writer(
            [{"role": "user", "content": text}, {"role": "assistant", "content": reply}], user_id
        )
        return reply

    def poll_write_result(self) -> tuple[str, Any] | None:
        """Non-blocking check for a completed background write.

        Returns None if nothing has finished since the last poll, otherwise
        ("ok", add_result) or ("error", message_str).
        """
        try:
            return self._write_results.get_nowait()
        except queue.Empty:
            return None

    def shutdown(self, timeout: float = 5.0) -> None:
        """Join every still-running writer thread — called on /exit, Ctrl+C,
        or EOF so the last turn's write isn't silently dropped.
        """
        for thread in self._pending_writers:
            thread.join(timeout=timeout)

    def _build_system_prompt(self, retrieved: list[dict[str, Any]]) -> str:
        if not retrieved:
            return SYSTEM_PROMPT

        lines = [SYSTEM_PROMPT, "", "What I remember about this user:"]
        for item in retrieved:
            category = item.get("memory_category", "semantic")
            lines.append(f"- {item['memory']}  ({category})")
        return "\n".join(lines)

    def _spawn_writer(self, messages: list[dict[str, str]], user_id: str) -> None:
        thread = threading.Thread(
            target=self._writer_target, args=(messages, user_id), daemon=False
        )
        self._pending_writers.append(thread)
        thread.start()

    def _writer_target(self, messages: list[dict[str, str]], user_id: str) -> None:
        """Runs on the background thread. A raw Thread swallows uncaught
        exceptions silently by default, so we catch here and queue an
        ("error", ...) tuple instead — write failures must stay visible.
        """
        try:
            result = self.memory.add(messages, user_id=user_id)
            self._write_results.put(("ok", result))
        except Exception as exc:
            logger.warning("Background memory write failed for user %r.", user_id, exc_info=True)
            self._write_results.put(("error", str(exc)))
