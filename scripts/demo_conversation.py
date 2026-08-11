"""Scripted demo conversation exercising ADD / UPDATE / DELETE / (short-circuit)
"nothing to remember" end-to-end, printing the events memlayer.Memory.add()
produces after each turn.

Usage:
    uv run python scripts/demo_conversation.py          # live, needs GEMINI_API_KEY in .env
    uv run python scripts/demo_conversation.py --fake    # offline, deterministic, no network

--fake mode uses a small scripted/keyword-driven fake LLM (defined below) so
the four event types are demonstrated deterministically without depending on
a real model's exact wording. --live mode sends the same five conversational
turns to the real Gemini API and prints whatever it actually decides —
useful to sanity-check the real pipeline, but not asserted against (LLM
output isn't deterministic).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

USER_ID = "demo_user"
LIVE_SLEEP_BETWEEN_TURNS_SECONDS = 3.0
_EXISTING_MEMORY_PATTERN = re.compile(r'"id":\s*"(\d+)",\s*"text":\s*"([^"]*)"')


class _DeterministicFakeEmbedder:
    """Same md5-seeded deterministic embedder as tests/conftest.py's
    FakeEmbedder, reimplemented here so this script has no dependency on the
    tests/ package (which isn't installed/importable outside pytest)."""

    dims = 8

    def embed(self, text: str) -> list[float]:
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=self.dims)
        norm = np.linalg.norm(vector)
        return (vector / norm if norm > 0 else vector).tolist()


class _ScriptedReconciliationLLM:
    """Fake LLM for --fake mode.

    Extraction responses are scripted per turn. Reconciliation responses are
    computed DYNAMICALLY by regex-scanning the prompt for the existing
    memories the pipeline actually retrieved (via the anti-hallucination
    integer-id remap), then applying each turn's keyword rule — so the demo
    stays correct regardless of which integer id gets assigned to which
    memory, without needing a real LLM's judgment.
    """

    def __init__(self, turns: list[dict]) -> None:
        self._turns = list(turns)
        self._current: dict | None = None
        self._expecting_extraction = True

    def generate_response(self, messages: list[dict], response_format: str = "json") -> str:
        if self._expecting_extraction:
            self._current = self._turns.pop(0)
            self._expecting_extraction = False
            return self._current["extraction_response"]

        turn = self._current
        self._expecting_extraction = True
        prompt_text = messages[0]["content"]
        existing = _EXISTING_MEMORY_PATTERN.findall(prompt_text)

        memory_items = []
        for mem_id, text in existing:
            event = turn["existing_rule"](text)
            item = {"id": mem_id, "text": text, "event": event}
            if event == "UPDATE":
                item["old_memory"] = text
                item["text"] = turn["update_text"]
            memory_items.append(item)

        for new_text, _category in turn["new_facts"]:
            memory_items.append({"text": new_text, "event": "ADD"})

        return json.dumps({"memory": memory_items})


def _always_none(_text: str) -> str:
    return "NONE"


def _fake_turns() -> list[dict]:
    return [
        {
            "user_says": "Hi, I'm Bibek, doing an AIML course. My address is 42 MG Road.",
            "extraction_response": json.dumps(
                {
                    "facts": [
                        {"text": "Name is Bibek", "category": "semantic"},
                        {"text": "Is doing an AIML course", "category": "semantic"},
                        {"text": "Address is 42 MG Road", "category": "episodic"},
                    ]
                }
            ),
            "new_facts": [
                ("Name is Bibek", "semantic"),
                ("Is doing an AIML course", "semantic"),
                ("Address is 42 MG Road", "episodic"),
            ],
            "existing_rule": _always_none,
            "update_text": None,
        },
        {
            "user_says": "I love filter coffee.",
            "extraction_response": json.dumps(
                {"facts": [{"text": "Likes filter coffee", "category": "semantic"}]}
            ),
            "new_facts": [("Likes filter coffee", "semantic")],
            "existing_rule": _always_none,
            "update_text": None,
        },
        {
            "user_says": "Actually, I've switched to green tea instead of coffee.",
            "extraction_response": json.dumps(
                {"facts": [{"text": "Drinks green tea instead of coffee", "category": "semantic"}]}
            ),
            "new_facts": [],
            "existing_rule": lambda text: "UPDATE" if "coffee" in text.lower() else "NONE",
            "update_text": "Drinks green tea instead of coffee",
        },
        {
            "user_says": "I moved out of 42 MG Road — that address is no longer valid.",
            "extraction_response": json.dumps(
                {"facts": [{"text": "No longer lives at 42 MG Road", "category": "episodic"}]}
            ),
            "new_facts": [],
            "existing_rule": lambda text: (
                "DELETE" if "mg road" in text.lower() or "address" in text.lower() else "NONE"
            ),
            "update_text": None,
        },
        {
            "user_says": "What's 2 + 2?",
            "extraction_response": json.dumps({"facts": []}),
            "new_facts": [],
            "existing_rule": _always_none,
            "update_text": None,
        },
    ]


def build_memory(fake: bool, data_dir: Path):
    from memlayer.memory import Memory
    from memlayer.storage.history import SQLiteHistoryStore
    from memlayer.vector_stores.local import LocalVectorStore

    vector_store = LocalVectorStore(db_path=data_dir / "demo_vectors.db")
    history_store = SQLiteHistoryStore(db_path=data_dir / "demo_history.db")

    if fake:
        turns_spec = _fake_turns()
        llm = _ScriptedReconciliationLLM(turns_spec)
        embedder = _DeterministicFakeEmbedder()
        turns = [t["user_says"] for t in turns_spec]
        memory = Memory(
            llm=llm, embedder=embedder, vector_store=vector_store, history_store=history_store
        )
        return memory, turns

    from memlayer.embeddings.sentence_transformer import SentenceTransformerEmbedder
    from memlayer.llms.gemini import GeminiLLM

    llm = GeminiLLM()
    embedder = SentenceTransformerEmbedder()
    turns = [t["user_says"] for t in _fake_turns()]
    memory = Memory(
        llm=llm, embedder=embedder, vector_store=vector_store, history_store=history_store
    )
    return memory, turns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true", help="Run offline with a scripted fake LLM.")
    args = parser.parse_args()

    if not args.fake:
        from dotenv import load_dotenv

        load_dotenv()

    data_dir = Path("./data/demo")
    data_dir.mkdir(parents=True, exist_ok=True)
    memory, turns = build_memory(fake=args.fake, data_dir=data_dir)
    memory.reset()  # each run starts fresh — this script's db files persist across runs otherwise

    print(f"=== Memento demo conversation ({'fake' if args.fake else 'live'} mode) ===\n")
    for i, user_text in enumerate(turns, start=1):
        print(f"Turn {i} — you: {user_text}")
        result = memory.add(user_text, user_id=USER_ID)
        if not result["results"]:
            print("  -> nothing worth remembering (extraction returned no facts, or all NONE)")
        for item in result["results"]:
            print(f"  -> {item['event']}: {item['memory']}")
        print()
        if not args.fake:
            time.sleep(LIVE_SLEEP_BETWEEN_TURNS_SECONDS)

    print("=== Final memory state ===")
    for row in memory.get_all(user_id=USER_ID)["results"]:
        print(f"- [{row['memory_category']}] {row['memory']}")


if __name__ == "__main__":
    main()
