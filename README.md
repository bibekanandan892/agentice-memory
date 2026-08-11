# Memento + MemLayer

A personal AI assistant (`memento`) built on a from-scratch, Mem0-style memory library (`memlayer`) — a portfolio implementation of the *"Design Memory for a Personal AI Assistant"* AIML course class, reproducing the classic Mem0 `v0.1.118` two-phase extract/reconcile pipeline entirely from scratch.

**Design docs (read these first):**
[HLD — system context, read/write path, data architecture](docs/design/01-hld.md) ·
[memlayer LLD — class diagrams, the add() pipeline](docs/design/02-lld-memlayer.md) ·
[memento LLD — CLI, threading model, test doubles](docs/design/03-lld-memento.md)

## What this demonstrates

- **The two-phase memory pipeline**: an extractor LLM call turns a conversation into candidate facts; a reconciler LLM call compares them against existing memories and decides `ADD` / `UPDATE` / `DELETE` / `NONE` — with an anti-hallucination integer-id remap so the model can never invent a plausible-looking memory id.
- **Read path vs write path**: `search()` is synchronous and fast (in-process cosine search); `add()` runs on a background thread *after* the reply is already shown, because the write path is never latency-critical.
- **Two separate stores**: a verbatim chat transcript (`transcript.db`, owned by `memento`) is kept entirely separate from the distilled, searched memory store (`vectors.db`, owned by `memlayer`) — plus a `history.db` audit trail of every mutation.
- **User isolation**: the vector store is sharded by `user_id` — a search physically cannot see another user's rows, not just filtered post-hoc.
- **Semantic / episodic / procedural** categorization of every extracted fact, an extension beyond Mem0's own contract, directly implementing the class notes' memory taxonomy.

## Project layout

```
memlayer/     the memory library — see docs/design/02-lld-memlayer.md
  memory.py       Memory facade + the two-phase add() pipeline
  prompts.py      the frozen extraction/reconciliation LLM prompts
  llms/           LLMBase + GeminiLLM
  embeddings/     EmbeddingBase + SentenceTransformerEmbedder (default) + GeminiEmbedder
  vector_stores/  VectorStoreBase + LocalVectorStore (shard-by-user, SQLite-backed)
  storage/        SQLiteHistoryStore (mutation audit log)

memento/      the CLI assistant — see docs/design/03-lld-memento.md
  assistant.py    read path + threaded write path
  commands.py     /memories /forget /history /user /help /exit
  cli.py          the REPL loop

scripts/      demo_conversation.py, eval_recall.py (both support --fake for offline runs)
tests/        unit + integration (all mocked) + live (needs a real GEMINI_API_KEY)
docs/design/  the frozen HLD/LLD spec — the "paper API" the code implements
```

## Class-notes concept map

| Class notes concept | Where it lives here |
|---|---|
| Short-term vs long-term memory | `TranscriptStore` (session-only, no login needed) vs `memlayer.Memory` (cross-session, requires `user_id`) |
| Semantic / episodic / procedural | `memory_category` on every stored fact (`memlayer/prompts.py`, `memlayer/memory.py`) |
| Read path = personal-scale RAG | `Assistant.chat()`: search → inject top-5 into system prompt → generate |
| Write path: extract → decide | `Memory.add()`: extractor call → reconciler call → apply ADD/UPDATE/DELETE/NONE |
| Never leak memory across users | `LocalVectorStore` shards by `user_id`; `delete_all`/`search`/`get_all` refuse without a scope id |
| Shard by user, no global index | `LocalVectorStore._partitions: dict[user_id -> vectors]` |
| Read latency matters, write doesn't | Search is synchronous; `add()` runs on a background thread |
| Two separate stores | `transcript.db` (raw) vs `vectors.db` (distilled) vs `history.db` (audit) |

Full mapping with file/function references: [`docs/design/01-hld.md` §11](docs/design/01-hld.md#11-class-notes-concept-design-element-mapping).

## Mem0 parity

Targets Mem0's classic `v0.1.118` semantics (current Mem0 `main`/v2 replaced this with a single-call, ADD-only pipeline — research confirmed v0.1.118 is the pedagogically interesting architecture and the one this class actually describes). Full parity table: [`docs/design/02-lld-memlayer.md` §10](docs/design/02-lld-memlayer.md#10-mem0-parity-table). Deliberately out of scope: graph memory, rerankers, telemetry, an async/await twin, expiration/decay.

## Setup (Windows / PowerShell)

```powershell
# Requires Python 3.11+ and uv (https://docs.astral.sh/uv/)
uv sync --extra dev              # core deps + test tooling (no model download)
uv sync --extra dev --extra local  # add this when you want the real local embedder

Copy-Item .env.example .env
# then edit .env and set GEMINI_API_KEY (free key: https://aistudio.google.com/)
```

Without `uv`, a plain `pip install -e ".[dev,local]"` inside a venv works too — the package is a standard `pyproject.toml` project.

The `[local]` extra installs `sentence-transformers` (and `torch`), used for the default embedder. It's optional: the core library and all offline tests never require it (the embedder is lazy-loaded on first real use, so `import memlayer` and the whole test suite work without it). The first time you actually call the real embedder it downloads the `all-MiniLM-L6-v2` model (~90MB) and caches it under your Hugging Face cache dir.

## Running the tests

```powershell
uv run pytest -m "not live" --cov --cov-report=term-missing   # offline suite, no API key needed
uv run ruff check .
uv run pytest -m live          # 1-2 real Gemini calls, needs GEMINI_API_KEY in .env
```

The offline suite (250+ tests) mocks every LLM and, for the default embedder path, mocks `sentence-transformers` too — nothing in CI ever hits the network or downloads a model.

## Try it

```powershell
# Scripted, deterministic demo — no API key needed, walks through ADD/UPDATE/DELETE
uv run python scripts/demo_conversation.py --fake

# Retrieval quality check across two users (zero cross-user leakage is a hard gate)
uv run python scripts/eval_recall.py --fake      # deterministic offline smoke test
uv run python scripts/eval_recall.py             # real embeddings, needs --extra local

# The interactive assistant (needs GEMINI_API_KEY in .env)
uv run memento
```

Sample `--fake` demo output (`scripts/demo_conversation.py --fake`):

```
Turn 1 — you: Hi, I'm Bibek, doing an AIML course. My address is 42 MG Road.
  -> ADD: Name is Bibek
  -> ADD: Is doing an AIML course
  -> ADD: Address is 42 MG Road

Turn 2 — you: I love filter coffee.
  -> ADD: Likes filter coffee

Turn 3 — you: Actually, I've switched to green tea instead of coffee.
  -> UPDATE: Drinks green tea instead of coffee

Turn 4 — you: I moved out of 42 MG Road — that address is no longer valid.
  -> DELETE: Address is 42 MG Road

Turn 5 — you: What's 2 + 2?
  -> nothing worth remembering (extraction returned no facts, or all NONE)

=== Final memory state ===
- [semantic] Name is Bibek
- [semantic] Is doing an AIML course
- [semantic] Drinks green tea instead of coffee
```

A sample `uv run memento` session:

```
you> Hi, I'm Bibek. I'm taking an AIML course and I love filter coffee.
memento> Nice to meet you, Bibek! Good luck with the AIML course — and filter coffee is a great choice.
Memory updated ✓

you> /memories
                 Memories for bibek
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
┃ id       ┃ memory                      ┃ category  ┃ created  ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
│ a1b2c3d4 │ Name is Bibek               │ semantic  │ ...      │
│ e5f6g7h8 │ Likes filter coffee         │ semantic  │ ...      │
└──────────┴─────────────────────────────┴───────────┴──────────┘

you> /user alice
Switched to user: alice

you> /memories
No memories yet for alice.

you> /user bibek
Switched to user: bibek

you> /history e5f6g7h8
...event timeline...

you> /forget e5f6g7h8
Memory deleted successfully!

you> /exit
Goodbye!
```

## Architecture at a glance

See [`docs/design/01-hld.md`](docs/design/01-hld.md) for the full diagram set (system context, component diagram, read/write path sequence diagrams, two-store data architecture, threading model, ER diagrams). Short version:

```
User <-> memento CLI <-> memlayer.Memory <-> {Gemini API, local embedder, SQLite files}
```

Only conversation text and extracted facts ever leave the machine (to Gemini); embeddings, vector search, and all persistence stay local.
