"""Integration test: add() -> search() -> history(), through real SQLite-backed
LocalVectorStore and SQLiteHistoryStore in tmp_path, with only the LLM faked.

See docs/design/03-lld-memento.md testing-strategy map.
"""

import json


def test_add_then_search_then_history_roundtrip(memory_with_fakes):
    memory_with_fakes.llm.queue(
        json.dumps({"facts": [{"text": "Likes filter coffee", "category": "semantic"}]})
    )
    memory_with_fakes.llm.queue(
        json.dumps({"memory": [{"text": "Likes filter coffee", "event": "ADD"}]})
    )

    add_result = memory_with_fakes.add("I love filter coffee", user_id="bibek")
    memory_id = add_result["results"][0]["id"]

    search_result = memory_with_fakes.search("filter coffee", user_id="bibek", limit=5)
    assert len(search_result["results"]) == 1
    assert search_result["results"][0]["id"] == memory_id
    assert search_result["results"][0]["memory"] == "Likes filter coffee"

    history = memory_with_fakes.history(memory_id)
    assert len(history) == 1
    assert history[0]["event"] == "ADD"


def test_add_update_then_search_reflects_latest_text(memory_with_fakes):
    memory_with_fakes.llm.queue(
        json.dumps({"facts": [{"text": "Likes tea", "category": "semantic"}]})
    )
    memory_with_fakes.llm.queue(
        json.dumps({"memory": [{"text": "Likes tea", "event": "ADD"}]})
    )
    first = memory_with_fakes.add("I like tea", user_id="bibek")
    memory_id = first["results"][0]["id"]

    memory_with_fakes.llm.queue(
        json.dumps({"facts": [{"text": "Loves green tea", "category": "semantic"}]})
    )
    memory_with_fakes.llm.queue(
        json.dumps(
            {
                "memory": [
                    {
                        "id": "0",
                        "text": "Loves green tea",
                        "event": "UPDATE",
                        "old_memory": "Likes tea",
                    }
                ]
            }
        )
    )
    memory_with_fakes.add("Actually I love green tea", user_id="bibek")

    search_result = memory_with_fakes.search("green tea", user_id="bibek", limit=5)
    assert search_result["results"][0]["id"] == memory_id
    assert search_result["results"][0]["memory"] == "Loves green tea"

    history = memory_with_fakes.history(memory_id)
    assert [row["event"] for row in history] == ["ADD", "UPDATE"]
