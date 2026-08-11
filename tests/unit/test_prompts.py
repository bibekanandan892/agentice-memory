"""Tests for memlayer.prompts — the frozen LLM contract.

Per docs/design/02-lld-memlayer.md, these prompt builders are tested for
STRUCTURE (what they must contain / how the messages are shaped), not for
LLM behavior — that's what FakeLLM-driven tests in test_memory_add.py cover.
"""

import json

from memlayer.prompts import (
    DEFAULT_UPDATE_MEMORY_PROMPT,
    FACT_RETRIEVAL_PROMPT,
    build_fact_retrieval_messages,
    build_update_memory_messages,
)


class TestFactRetrievalPrompt:
    def test_mentions_all_three_memory_categories(self):
        for category in ("semantic", "episodic", "procedural"):
            assert category in FACT_RETRIEVAL_PROMPT.lower()

    def test_requires_facts_json_key(self):
        assert '"facts"' in FACT_RETRIEVAL_PROMPT

    def test_restricts_extraction_to_user_messages(self):
        assert "user message" in FACT_RETRIEVAL_PROMPT.lower()

    def test_has_a_negative_example_returning_empty_facts(self):
        assert '"facts": []' in FACT_RETRIEVAL_PROMPT

    def test_mentions_same_language_rule(self):
        assert "language" in FACT_RETRIEVAL_PROMPT.lower()


class TestBuildFactRetrievalMessages:
    def test_returns_system_and_user_messages(self):
        messages = build_fact_retrieval_messages("user: I love coffee")
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user"]

    def test_system_message_is_the_frozen_prompt(self):
        messages = build_fact_retrieval_messages("user: hi")
        assert messages[0]["content"] == FACT_RETRIEVAL_PROMPT

    def test_user_message_contains_the_transcript(self):
        messages = build_fact_retrieval_messages("user: I love filter coffee")
        assert "I love filter coffee" in messages[1]["content"]


class TestUpdateMemoryPrompt:
    def test_describes_all_four_events(self):
        for event in ("ADD", "UPDATE", "DELETE", "NONE"):
            assert event in DEFAULT_UPDATE_MEMORY_PROMPT

    def test_instructs_reuse_of_input_ids_only(self):
        assert "do not generate" in DEFAULT_UPDATE_MEMORY_PROMPT.lower()

    def test_mentions_old_memory_field_for_update(self):
        assert "old_memory" in DEFAULT_UPDATE_MEMORY_PROMPT

    def test_instructs_every_existing_memory_must_appear(self):
        assert "every" in DEFAULT_UPDATE_MEMORY_PROMPT.lower()


class TestBuildUpdateMemoryMessages:
    def test_returns_single_user_message(self):
        messages = build_update_memory_messages([], [{"text": "likes tea", "category": "semantic"}])
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_empty_existing_memory_is_valid_json_empty_list(self):
        messages = build_update_memory_messages([], [{"text": "likes tea", "category": "semantic"}])
        content = messages[0]["content"]
        # The existing-memory block must be parseable JSON, not a Python repr.
        assert "[]" in content

    def test_existing_memories_embedded_as_valid_json(self):
        existing = [{"id": "0", "text": "likes coffee"}]
        messages = build_update_memory_messages(existing, [])
        content = messages[0]["content"]
        assert json.dumps(existing) in content or json.dumps(existing, indent=2) in content

    def test_new_facts_embedded_in_prompt(self):
        facts = [{"text": "likes green tea", "category": "semantic"}]
        messages = build_update_memory_messages([], facts)
        assert "likes green tea" in messages[0]["content"]

    def test_only_integer_string_ids_appear_never_uuid_shaped(self):
        # Anti-hallucination: existing memories passed in must already be
        # remapped to integer-string ids by the caller (Memory.add) — this
        # test guards that the prompt builder itself never invents or
        # forwards a UUID-shaped id.
        existing = [{"id": "0", "text": "likes coffee"}, {"id": "1", "text": "lives in Pune"}]
        messages = build_update_memory_messages(existing, [])
        content = messages[0]["content"]
        assert '"id": "0"' in content
        assert '"id": "1"' in content
        assert "-" not in content.split('"text": "likes coffee"')[0].split('"id"')[-1][:10]
