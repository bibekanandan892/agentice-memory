"""Pure helper functions: JSON/fence cleanup, message parsing, hashing, timestamps.

See docs/design/02-lld-memlayer.md §8, must-not-skip mechanism #3.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any

from memlayer.errors import LLMResponseError

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*\n)?(.*?)```", re.DOTALL)


def remove_code_blocks(text: str) -> str:
    """Strip ```-fenced code blocks (with or without a language tag) and
    <think>...</think> reasoning blocks that small/reasoning-tuned LLMs
    frequently wrap JSON output in.

    If a fenced block is present, only the innermost fenced content is
    returned (fences take priority over any surrounding prose). Otherwise
    the think-stripped text is returned as-is.
    """
    cleaned = _THINK_BLOCK_RE.sub("", text)

    fence_match = _CODE_FENCE_RE.search(cleaned)
    if fence_match:
        return fence_match.group(1).strip()

    return cleaned.strip()


def safe_json_loads(text: str) -> Any:
    """remove_code_blocks() + json.loads(), raising a clear LLMResponseError
    (with a snippet of the offending text) on failure instead of a bare
    JSONDecodeError.

    Callers that want graceful degradation (e.g. Memory._extract_facts)
    catch LLMResponseError themselves and fall back to an empty result —
    this function's job is only to fail loudly and clearly, not to decide
    whether that failure is fatal.
    """
    cleaned = remove_code_blocks(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        snippet = cleaned[:200]
        raise LLMResponseError(
            f"Could not parse LLM response as JSON ({exc}). Response was not json: {snippet!r}"
        ) from exc


def parse_messages(messages: str | list[dict]) -> str:
    """Flatten a message list (or a bare string) into 'role: content' lines,
    one per message, skipping system messages. A bare string is treated as
    a single user message.
    """
    if isinstance(messages, str):
        return f"user: {messages}"

    lines = [
        f"{message['role']}: {message['content']}"
        for message in messages
        if message.get("role") != "system"
    ]
    return "\n".join(lines)


def md5_hash(text: str) -> str:
    """MD5 hex digest of text, used for cheap exact-duplicate detection."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with an explicit offset."""
    return datetime.now(UTC).isoformat()
