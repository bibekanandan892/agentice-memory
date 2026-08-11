"""Tests for memlayer.utils — pure helpers used throughout the add() pipeline.

See docs/design/02-lld-memlayer.md must-not-skip mechanism #3.
"""

import json

import pytest

from memlayer.errors import LLMResponseError
from memlayer.utils import (
    md5_hash,
    parse_messages,
    remove_code_blocks,
    safe_json_loads,
    utc_now_iso,
)


class TestRemoveCodeBlocks:
    def test_plain_text_passthrough(self):
        assert remove_code_blocks('{"a": 1}') == '{"a": 1}'

    def test_strips_json_fence(self):
        text = '```json\n{"a": 1}\n```'
        assert remove_code_blocks(text).strip() == '{"a": 1}'

    def test_strips_bare_fence(self):
        text = '```\n{"a": 1}\n```'
        assert remove_code_blocks(text).strip() == '{"a": 1}'

    def test_strips_think_block(self):
        text = "<think>reasoning about the answer</think>\n{\"a\": 1}"
        assert remove_code_blocks(text).strip() == '{"a": 1}'

    def test_strips_think_block_and_fence_together(self):
        text = '<think>let me think</think>\n```json\n{"a": 1}\n```'
        assert remove_code_blocks(text).strip() == '{"a": 1}'

    def test_strips_multiline_think_block(self):
        text = "<think>\nline one\nline two\n</think>\n{\"a\": 1}"
        assert remove_code_blocks(text).strip() == '{"a": 1}'


class TestSafeJsonLoads:
    def test_parses_clean_json(self):
        assert safe_json_loads('{"facts": []}') == {"facts": []}

    def test_parses_fenced_json(self):
        assert safe_json_loads('```json\n{"facts": ["x"]}\n```') == {"facts": ["x"]}

    def test_parses_json_after_think_block(self):
        text = '<think>reasoning</think>```json\n{"facts": []}\n```'
        assert safe_json_loads(text) == {"facts": []}

    def test_raises_llm_response_error_on_garbage(self):
        with pytest.raises(LLMResponseError):
            safe_json_loads("this is not json at all")

    def test_error_message_includes_snippet_of_offending_text(self):
        with pytest.raises(LLMResponseError, match="not json"):
            safe_json_loads("not json whatsoever")


class TestParseMessages:
    def test_bare_string_becomes_single_user_line(self):
        assert parse_messages("hello") == "user: hello"

    def test_list_of_messages_formatted_per_role(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello there"},
        ]
        assert parse_messages(messages) == "user: hi\nassistant: hello there"

    def test_system_messages_are_skipped(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        assert parse_messages(messages) == "user: hi"

    def test_empty_list_returns_empty_string(self):
        assert parse_messages([]) == ""


class TestHashAndTimestamp:
    def test_md5_hash_is_deterministic(self):
        assert md5_hash("hello") == md5_hash("hello")

    def test_md5_hash_differs_for_different_text(self):
        assert md5_hash("hello") != md5_hash("world")

    def test_md5_hash_matches_stdlib(self):
        import hashlib

        assert md5_hash("hello") == hashlib.md5(b"hello").hexdigest()

    def test_utc_now_iso_is_iso_parseable(self):
        from datetime import datetime

        stamp = utc_now_iso()
        # Must be parseable and carry explicit UTC offset info.
        datetime.fromisoformat(stamp.replace("Z", "+00:00"))

    def test_utc_now_iso_format_is_json_serializable(self):
        # Sanity: it's just a string, always safe to json.dumps.
        assert isinstance(json.dumps({"t": utc_now_iso()}), str)
