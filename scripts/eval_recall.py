"""Retrieval evaluation: seed known facts for two users via infer=False
(bypassing the extraction/reconciliation LLM entirely, since this script
evaluates embedding + vector-search quality in isolation), then query for
them and report top-K recall plus a cross-user leakage check.

Usage:
    uv run python scripts/eval_recall.py         # real embeddings, needs `uv sync --extra local`
    uv run python scripts/eval_recall.py --fake  # deterministic offline smoke test of the script

Note: with only 5 facts per user and TOP_K=5, --fake mode trivially returns
every fact regardless of relevance (there's nothing to rank out), so its
recall number is not meaningful — it only validates the plumbing and the
leakage check, which is why the 80% recall gate is only enforced in real mode.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

TOP_K = 5
RECALL_TARGET_PERCENT = 80

FACTS_ALICE = [
    "Alice's favorite color is blue.",
    "Alice works as a software engineer.",
    "Alice lives in Bangalore.",
    "Alice enjoys playing badminton on weekends.",
    "Alice is allergic to peanuts.",
]
FACTS_BOB = [
    "Bob's favorite color is green.",
    "Bob works as a data scientist.",
    "Bob lives in Mumbai.",
    "Bob enjoys hiking in the mountains.",
    "Bob is a vegetarian.",
]

QUERIES = [
    ("alice", "What is Alice's favorite color?", "Alice's favorite color is blue."),
    ("alice", "Where does Alice work?", "Alice works as a software engineer."),
    ("alice", "Where does Alice live?", "Alice lives in Bangalore."),
    ("alice", "What does Alice do on weekends?", "Alice enjoys playing badminton on weekends."),
    ("alice", "Does Alice have any allergies?", "Alice is allergic to peanuts."),
    ("bob", "What is Bob's favorite color?", "Bob's favorite color is green."),
    ("bob", "Where does Bob work?", "Bob works as a data scientist."),
    ("bob", "Where does Bob live?", "Bob lives in Mumbai."),
    ("bob", "What does Bob enjoy doing?", "Bob enjoys hiking in the mountains."),
    ("bob", "What are Bob's dietary preferences?", "Bob is a vegetarian."),
]


class _UnusedLLM:
    """This script only ever calls add(infer=False) and search(), neither of
    which touches the LLM — so no real GeminiLLM (and no API key) is needed."""

    def generate_response(self, messages: list[dict], response_format: str = "json") -> str:
        raise RuntimeError("eval_recall.py never calls the LLM.")


class _DeterministicFakeEmbedder:
    dims = 8

    def embed(self, text: str) -> list[float]:
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=self.dims)
        norm = np.linalg.norm(vector)
        return (vector / norm if norm > 0 else vector).tolist()


def build_memory(fake: bool, data_dir: Path):
    from memlayer.memory import Memory
    from memlayer.storage.history import SQLiteHistoryStore
    from memlayer.vector_stores.local import LocalVectorStore

    if fake:
        embedder = _DeterministicFakeEmbedder()
    else:
        from memlayer.embeddings.sentence_transformer import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()

    return Memory(
        llm=_UnusedLLM(),
        embedder=embedder,
        vector_store=LocalVectorStore(db_path=data_dir / "eval_vectors.db"),
        history_store=SQLiteHistoryStore(db_path=data_dir / "eval_history.db"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true", help="Use deterministic fake embeddings.")
    args = parser.parse_args()

    data_dir = Path("./data/eval")
    data_dir.mkdir(parents=True, exist_ok=True)
    memory = build_memory(fake=args.fake, data_dir=data_dir)
    memory.reset()

    for text in FACTS_ALICE:
        memory.add(text, user_id="alice", infer=False)
    for text in FACTS_BOB:
        memory.add(text, user_id="bob", infer=False)

    print(f"=== Retrieval evaluation ({'fake' if args.fake else 'real'} embeddings) ===\n")
    hits = 0
    leaks = 0
    for user_id, query, expected in QUERIES:
        results = memory.search(query, user_id=user_id, limit=TOP_K)["results"]
        found = any(row["memory"] == expected for row in results)
        other_user = "bob" if user_id == "alice" else "alice"
        leaked = any(row.get("user_id") == other_user for row in results)
        hits += int(found)
        leaks += int(leaked)
        status = "HIT " if found else "MISS"
        print(f"[{status}] ({user_id}) {query!r}")

    recall_percent = 100 * hits / len(QUERIES)
    print(f"\nTop-{TOP_K} recall: {hits}/{len(QUERIES)} ({recall_percent:.0f}%)")
    print(f"Cross-user leakage: {leaks} quer{'y' if leaks == 1 else 'ies'}")

    if leaks > 0:
        raise SystemExit(f"FAILED: {leaks} cross-user leakage(s) detected.")
    if not args.fake and recall_percent < RECALL_TARGET_PERCENT:
        raise SystemExit(
            f"FAILED: recall {recall_percent:.0f}% is below the {RECALL_TARGET_PERCENT}% target."
        )

    print("\nPASSED: zero cross-user leakage" + ("" if args.fake else " and recall >= 80%") + ".")


if __name__ == "__main__":
    main()
