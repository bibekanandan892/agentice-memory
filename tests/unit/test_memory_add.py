"""Tests for Memory.add() — the two-phase extract/reconcile pipeline.

See docs/design/02-lld-memlayer.md §8 and the "must-not-skip mechanisms"
table. Each test in this file is deliberately named after the mechanism or
scenario it proves, so a reviewer can check off the LLD's checklist directly
against test names.
"""

import json

FACT_EXTRACTION_EMPTY = json.dumps({"facts": []})


def facts_response(*facts):
    return json.dumps({"facts": list(facts)})


def reconcile_response(*items):
    return json.dumps({"memory": list(items)})


class TestAddEventTypes:
    def test_add_event_creates_a_new_memory(self, memory_with_fakes):
        memory_with_fakes.llm.queue(
            facts_response({"text": "Likes filter coffee", "category": "semantic"})
        )
        memory_with_fakes.llm.queue(
            reconcile_response({"text": "Likes filter coffee", "event": "ADD"})
        )

        result = memory_with_fakes.add("I love filter coffee", user_id="alice")

        assert len(result["results"]) == 1
        item = result["results"][0]
        assert item["event"] == "ADD"
        assert item["memory"] == "Likes filter coffee"
        assert "id" in item

        stored = memory_with_fakes.get_all(user_id="alice")["results"]
        assert len(stored) == 1
        assert stored[0]["memory"] == "Likes filter coffee"

        history = memory_with_fakes.history(item["id"])
        assert len(history) == 1
        assert history[0]["event"] == "ADD"
        assert history[0]["old_memory"] is None

    def test_update_event_updates_an_existing_memory_preserving_id(self, memory_with_fakes):
        existing_id = "existing-1"
        memory_with_fakes.vector_store.insert(
            existing_id,
            memory_with_fakes.embedder.embed("Likes tea"),
            {
                "user_id": "alice",
                "data": "Likes tea",
                "hash": "h1",
                "memory_category": "semantic",
                "created_at": "t0",
                "updated_at": "t0",
            },
        )
        memory_with_fakes.llm.queue(
            facts_response({"text": "Loves green tea with friends", "category": "semantic"})
        )
        memory_with_fakes.llm.queue(
            reconcile_response(
                {
                    "id": "0",
                    "text": "Loves green tea with friends",
                    "event": "UPDATE",
                    "old_memory": "Likes tea",
                }
            )
        )

        result = memory_with_fakes.add("I love green tea with friends", user_id="alice")

        assert len(result["results"]) == 1
        item = result["results"][0]
        assert item["event"] == "UPDATE"
        assert item["id"] == existing_id  # same id preserved, not a new uuid
        assert item["memory"] == "Loves green tea with friends"
        assert item["previous_memory"] == "Likes tea"

        stored = memory_with_fakes.get(existing_id)
        assert stored["memory"] == "Loves green tea with friends"

        history = memory_with_fakes.history(existing_id)
        assert len(history) == 1
        assert history[0]["event"] == "UPDATE"
        assert history[0]["old_memory"] == "Likes tea"
        assert history[0]["new_memory"] == "Loves green tea with friends"

    def test_delete_event_removes_the_memory_and_marks_history(self, memory_with_fakes):
        existing_id = "existing-1"
        memory_with_fakes.vector_store.insert(
            existing_id,
            memory_with_fakes.embedder.embed("Loves cheese pizza"),
            {
                "user_id": "alice",
                "data": "Loves cheese pizza",
                "hash": "h1",
                "memory_category": "semantic",
                "created_at": "t0",
                "updated_at": "t0",
            },
        )
        memory_with_fakes.llm.queue(
            facts_response({"text": "Dislikes cheese pizza", "category": "semantic"})
        )
        memory_with_fakes.llm.queue(
            reconcile_response({"id": "0", "text": "Dislikes cheese pizza", "event": "DELETE"})
        )

        result = memory_with_fakes.add("I actually dislike cheese pizza now", user_id="alice")

        assert len(result["results"]) == 1
        assert result["results"][0]["event"] == "DELETE"
        assert result["results"][0]["id"] == existing_id

        assert memory_with_fakes.get(existing_id) is None

        history = memory_with_fakes.history(existing_id)
        assert len(history) == 1
        assert history[0]["event"] == "DELETE"
        assert history[0]["is_deleted"] is True
        assert history[0]["new_memory"] is None

    def test_none_event_is_excluded_from_results_and_leaves_memory_untouched(
        self, memory_with_fakes
    ):
        existing_id = "existing-1"
        memory_with_fakes.vector_store.insert(
            existing_id,
            memory_with_fakes.embedder.embed("Name is John"),
            {
                "user_id": "alice",
                "data": "Name is John",
                "hash": "h1",
                "memory_category": "semantic",
                "created_at": "t0",
                "updated_at": "t0",
            },
        )
        memory_with_fakes.llm.queue(
            facts_response({"text": "Name is John", "category": "semantic"})
        )
        memory_with_fakes.llm.queue(
            reconcile_response({"id": "0", "text": "Name is John", "event": "NONE"})
        )

        result = memory_with_fakes.add("My name is John", user_id="alice")

        assert result["results"] == []
        # untouched: no history row was ever written for this memory.
        assert memory_with_fakes.history(existing_id) == []
        assert memory_with_fakes.get(existing_id)["memory"] == "Name is John"


class TestEmbeddingCache:
    def test_embedder_is_called_exactly_once_per_extracted_fact(self, memory_with_fakes):
        from unittest.mock import Mock

        spy = Mock(wraps=memory_with_fakes.embedder.embed)
        memory_with_fakes.embedder.embed = spy

        memory_with_fakes.llm.queue(
            facts_response(
                {"text": "Likes filter coffee", "category": "semantic"},
                {"text": "Works as an engineer", "category": "semantic"},
            )
        )
        memory_with_fakes.llm.queue(
            reconcile_response(
                {"text": "Likes filter coffee", "event": "ADD"},
                {"text": "Works as an engineer", "event": "ADD"},
            )
        )
        memory_with_fakes.add("coffee and work talk", user_id="alice")

        # One embed() call per fact during retrieval, and the cached vector
        # (not a fresh embed call) must be reused when writing each ADD.
        assert spy.call_count == 2


class TestHallucinationGuard:
    def test_update_referencing_unknown_id_is_skipped_others_still_apply(self, memory_with_fakes):
        memory_with_fakes.llm.queue(
            facts_response(
                {"text": "A hallucinated update target", "category": "semantic"},
                {"text": "A brand new fact", "category": "semantic"},
            )
        )
        memory_with_fakes.llm.queue(
            reconcile_response(
                {"id": "99", "text": "A hallucinated update target", "event": "UPDATE"},
                {"text": "A brand new fact", "event": "ADD"},
            )
        )

        result = memory_with_fakes.add("some text", user_id="alice")

        events = {item["event"] for item in result["results"]}
        assert events == {"ADD"}
        assert len(result["results"]) == 1
        assert result["results"][0]["memory"] == "A brand new fact"

    def test_delete_referencing_unknown_id_is_skipped(self, memory_with_fakes):
        memory_with_fakes.llm.queue(facts_response({"text": "irrelevant", "category": "semantic"}))
        memory_with_fakes.llm.queue(
            reconcile_response({"id": "42", "text": "irrelevant", "event": "DELETE"})
        )

        result = memory_with_fakes.add("some text", user_id="alice")
        assert result["results"] == []


class TestGracefulJsonDegradation:
    def test_unparseable_extraction_response_yields_empty_results_no_crash(self, memory_with_fakes):
        memory_with_fakes.llm.queue("this is not valid json at all")

        result = memory_with_fakes.add("some text", user_id="alice")
        assert result == {"results": []}

    def test_empty_facts_short_circuits_before_reconciliation_call(self, memory_with_fakes):
        memory_with_fakes.llm.queue(FACT_EXTRACTION_EMPTY)
        # Only one response queued — if the implementation tried to make a
        # second (reconciliation) call, FakeLLM would raise AssertionError.
        result = memory_with_fakes.add("nothing worth remembering", user_id="alice")
        assert result == {"results": []}

    def test_unparseable_reconciliation_response_yields_empty_results_no_crash(
        self, memory_with_fakes
    ):
        memory_with_fakes.llm.queue(facts_response({"text": "a fact", "category": "semantic"}))
        memory_with_fakes.llm.queue("garbage, not json")

        result = memory_with_fakes.add("some text", user_id="alice")
        assert result == {"results": []}
        assert memory_with_fakes.get_all(user_id="alice")["results"] == []


class TestFencedAndThinkWrappedResponses:
    def test_extraction_response_wrapped_in_think_and_fence_is_parsed(self, memory_with_fakes):
        wrapped = (
            "<think>let me consider what's worth remembering</think>\n"
            "```json\n" + facts_response({"text": "Likes filter coffee", "category": "semantic"})
            + "\n```"
        )
        memory_with_fakes.llm.queue(wrapped)
        memory_with_fakes.llm.queue(
            reconcile_response({"text": "Likes filter coffee", "event": "ADD"})
        )

        result = memory_with_fakes.add("I love filter coffee", user_id="alice")
        assert result["results"][0]["event"] == "ADD"

    def test_reconciliation_response_wrapped_in_fence_is_parsed(self, memory_with_fakes):
        memory_with_fakes.llm.queue(facts_response({"text": "a fact", "category": "semantic"}))
        wrapped = "```json\n" + reconcile_response({"text": "a fact", "event": "ADD"}) + "\n```"
        memory_with_fakes.llm.queue(wrapped)

        result = memory_with_fakes.add("some text", user_id="alice")
        assert result["results"][0]["event"] == "ADD"


class TestPerEventIsolation:
    def test_one_failing_event_does_not_abort_the_rest_of_the_batch(self, memory_with_fakes):
        memory_with_fakes.llm.queue(
            facts_response(
                {"text": "fact one", "category": "semantic"},
                {"text": "fact two", "category": "semantic"},
                {"text": "fact three", "category": "semantic"},
            )
        )
        memory_with_fakes.llm.queue(
            reconcile_response(
                {"text": "fact one", "event": "ADD"},
                {"text": "fact two", "event": "ADD"},
                {"text": "fact three", "event": "ADD"},
            )
        )

        call_count = {"n": 0}
        real_insert = memory_with_fakes.vector_store.insert

        def flaky_insert(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("simulated storage failure")
            return real_insert(*args, **kwargs)

        memory_with_fakes.vector_store.insert = flaky_insert

        result = memory_with_fakes.add("three facts", user_id="alice")

        # Item 2 failed and was skipped; items 1 and 3 still succeeded.
        memories = {item["memory"] for item in result["results"]}
        assert memories == {"fact one", "fact three"}
        assert len(result["results"]) == 2

        stored = {row["memory"] for row in memory_with_fakes.get_all(user_id="alice")["results"]}
        assert stored == {"fact one", "fact three"}


class TestIdentityKeyStripping:
    def test_caller_cannot_smuggle_a_different_user_id_via_metadata(self, memory_with_fakes):
        memory_with_fakes.llm.queue(facts_response({"text": "a fact", "category": "semantic"}))
        memory_with_fakes.llm.queue(reconcile_response({"text": "a fact", "event": "ADD"}))

        memory_with_fakes.add(
            "some text",
            user_id="alice",
            metadata={"user_id": "evil", "topic": "cooking"},
        )

        stored = memory_with_fakes.get_all(user_id="alice")["results"]
        assert len(stored) == 1
        assert stored[0]["user_id"] == "alice"
        assert stored[0]["metadata"]["topic"] == "cooking"
        assert "user_id" not in stored[0]["metadata"]

        # And it must NOT have leaked into the "evil" partition either.
        evil_stored = memory_with_fakes.get_all(user_id="evil")["results"]
        assert evil_stored == []


class TestTolerantFactParsing:
    def test_bare_string_facts_are_tolerated_with_default_category(self, memory_with_fakes):
        memory_with_fakes.llm.queue(json.dumps({"facts": ["Likes filter coffee"]}))
        memory_with_fakes.llm.queue(
            reconcile_response({"text": "Likes filter coffee", "event": "ADD"})
        )

        result = memory_with_fakes.add("I love filter coffee", user_id="alice")

        assert result["results"][0]["event"] == "ADD"
        stored = memory_with_fakes.get_all(user_id="alice")["results"]
        assert stored[0]["memory_category"] == "semantic"

    def test_invalid_category_falls_back_to_semantic(self, memory_with_fakes):
        memory_with_fakes.llm.queue(
            facts_response({"text": "a fact", "category": "not-a-real-category"})
        )
        memory_with_fakes.llm.queue(reconcile_response({"text": "a fact", "event": "ADD"}))

        memory_with_fakes.add("some text", user_id="alice")

        stored = memory_with_fakes.get_all(user_id="alice")["results"]
        assert stored[0]["memory_category"] == "semantic"


class TestInferFalse:
    def test_infer_false_stores_each_message_verbatim(self, memory_with_fakes):
        messages = [
            {"role": "user", "content": "hello there"},
            {"role": "assistant", "content": "hi, how can I help?"},
        ]
        result = memory_with_fakes.add(messages, user_id="alice", infer=False)

        assert len(result["results"]) == 2
        assert {item["event"] for item in result["results"]} == {"ADD"}
        memories = {item["memory"] for item in result["results"]}
        assert memories == {"hello there", "hi, how can I help?"}

        stored = memory_with_fakes.get_all(user_id="alice")["results"]
        assert len(stored) == 2

    def test_infer_false_skips_system_messages(self, memory_with_fakes):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        result = memory_with_fakes.add(messages, user_id="alice", infer=False)
        assert len(result["results"]) == 1
        assert result["results"][0]["memory"] == "hi"

    def test_infer_false_never_calls_the_llm(self, memory_with_fakes):
        # No responses queued at all — if infer=False touched the LLM, FakeLLM
        # would raise AssertionError.
        result = memory_with_fakes.add("hello", user_id="alice", infer=False)
        assert len(result["results"]) == 1

    def test_infer_false_records_actor_id_when_message_has_a_name(self, memory_with_fakes):
        messages = [{"role": "user", "content": "hi", "name": "alice-device-1"}]
        result = memory_with_fakes.add(messages, user_id="alice", infer=False)

        stored = memory_with_fakes.get(result["results"][0]["id"])
        assert stored["metadata"]["actor_id"] == "alice-device-1"
