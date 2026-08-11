"""Exception hierarchy for memlayer.

See docs/design/02-lld-memlayer.md §6.
"""


class MemLayerError(Exception):
    """Base class for all memlayer errors."""


class ConfigError(MemLayerError):
    """Raised when MemoryConfig.from_dict() sees an unknown provider or malformed shape."""


class LLMResponseError(MemLayerError):
    """Raised when an LLM provider exhausts its retries or returns an unusable response."""


class ScopeError(MemLayerError):
    """Raised when a scoped operation (search/get_all/delete_all) is called with no
    user_id, agent_id, or run_id — the user-isolation safety rail."""
