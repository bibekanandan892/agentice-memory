"""LLMBase — the abstract interface every LLM provider implements.

See docs/design/02-lld-memlayer.md §2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMBase(ABC):
    """Deliberately dumb: returns raw text, never parses JSON itself. JSON
    parsing (and graceful degradation on failure) is Memory's responsibility,
    keeping this boundary trivial to fake in tests.
    """

    @abstractmethod
    def generate_response(self, messages: list[dict], response_format: str = "json") -> str:
        """messages: [{"role": "system"|"user"|"assistant", "content": str}, ...].
        response_format: a hint ("json" or "text") for providers that support
        structured-output modes; providers that don't support it may ignore it.
        Returns the raw text response.
        """
        raise NotImplementedError
