"""GeminiLLM provider. Implemented in Phase 1 Task 1.6.

See docs/design/02-lld-memlayer.md §2.
"""

from __future__ import annotations

import os
import random
import re
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from memlayer.errors import ConfigError, LLMResponseError
from memlayer.llms.base import LLMBase

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRIES = 3

JSON_RESPONSE_FORMAT = "json"
JSON_MIME_TYPE = "application/json"

# OpenAI-style "assistant" -> Gemini's "model". Any role not listed here
# (currently just "user") passes through unchanged.
GEMINI_ROLE_MAP = {"assistant": "model"}

RATE_LIMIT_HTTP_CODE = 429
RATE_LIMIT_STATUS = "RESOURCE_EXHAUSTED"

# Exponential backoff used only when the server didn't hand us a retry_delay:
# BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + uniform(0, BACKOFF_JITTER_SECONDS)
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_JITTER_SECONDS = 0.5

# Matches a protobuf Duration string like "19s" or "1.5s", the shape Gemini
# uses for google.rpc.RetryInfo.retryDelay.
_RETRY_DELAY_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)s$")


class GeminiLLM(LLMBase):
    """Gemini provider backed by the google-genai SDK (`google.genai.Client`).

    API key resolution order: the `api_key` constructor argument, then the
    `GEMINI_API_KEY` environment variable. If neither is set, `ConfigError`
    is raised immediately at construction time (fail fast, before any network
    call could happen) — the same fail-fast-on-bad-config posture
    `MemoryConfig.from_dict()` uses for unknown provider names.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        resolved_api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_api_key:
            raise ConfigError(
                "GeminiLLM requires an API key: pass api_key= to the constructor "
                "or set the GEMINI_API_KEY environment variable."
            )
        self._client = genai.Client(api_key=resolved_api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    def generate_response(self, messages: list[dict], response_format: str = "json") -> str:
        system_instruction, contents = self._map_messages(messages)
        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self.temperature,
            response_mime_type=JSON_MIME_TYPE if response_format == JSON_RESPONSE_FORMAT else None,
        )

        response = self._call_with_backoff(
            lambda: self._client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        )
        return response.text

    def _map_messages(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Split the leading run of `system`-role messages into a single
        `system_instruction` string, and map everything else into Gemini
        `contents` turns (`"assistant"` -> `"model"`).

        If more than one leading system message is present, their `content`
        values are concatenated with a blank line between them into one
        `system_instruction` string — Gemini's API takes a single string (or
        None), not a list.
        """
        leading_system_count = 0
        while (
            leading_system_count < len(messages)
            and messages[leading_system_count]["role"] == "system"
        ):
            leading_system_count += 1

        system_messages = messages[:leading_system_count]
        system_instruction = (
            "\n\n".join(message["content"] for message in system_messages)
            if system_messages
            else None
        )

        contents = [
            {
                "role": GEMINI_ROLE_MAP.get(message["role"], message["role"]),
                "parts": [{"text": message["content"]}],
            }
            for message in messages[leading_system_count:]
        ]
        return system_instruction, contents

    def _call_with_backoff(self, call):
        """Call `call()`, retrying on Gemini rate-limit errors.

        `self.max_retries` is the total number of attempts allowed (the
        initial try counts as attempt 1) — with the default of 3, the client
        is called at most 3 times total, not 3 retries on top of a first try.
        Any exception other than a rate-limit-shaped `ClientError` propagates
        immediately, uncaught, on the first attempt.
        """
        last_error: genai_errors.ClientError | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return call()
            except genai_errors.ClientError as exc:
                if not _is_rate_limit_error(exc):
                    raise
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(_retry_delay_seconds(exc, attempt))

        raise LLMResponseError(
            f"Gemini rate limit exceeded after {self.max_retries} attempt(s): {last_error}"
        ) from last_error


def _is_rate_limit_error(exc: genai_errors.ClientError) -> bool:
    return exc.code == RATE_LIMIT_HTTP_CODE or exc.status == RATE_LIMIT_STATUS


def _retry_delay_seconds(exc: genai_errors.ClientError, attempt: int) -> float:
    server_delay = _extract_server_retry_delay(exc)
    if server_delay is not None:
        return server_delay
    return BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, BACKOFF_JITTER_SECONDS)


def _extract_server_retry_delay(exc: genai_errors.ClientError) -> float | None:
    """Best-effort extraction of a `google.rpc.RetryInfo.retryDelay` (e.g.
    "19s") from the raw error body the SDK stores on `exc.details`.

    The installed google-genai SDK (2.17.0) does not parse this into a
    dedicated attribute on `ClientError` — it only keeps the raw JSON error
    body around as `.details`. We defensively dig for a RetryInfo entry
    there; if it's absent or unparseable (the common case observed against
    the free-tier quota error shape), the caller falls back to its own
    exponential backoff.
    """
    details = getattr(exc, "details", None)
    error_body = details.get("error", details) if isinstance(details, dict) else None
    if not isinstance(error_body, dict):
        return None

    for detail in error_body.get("details") or []:
        if isinstance(detail, dict) and "RetryInfo" in str(detail.get("@type", "")):
            match = _RETRY_DELAY_PATTERN.match(str(detail.get("retryDelay", "")))
            if match:
                return float(match.group(1))
    return None
