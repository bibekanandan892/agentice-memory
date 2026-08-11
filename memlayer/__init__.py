"""memlayer — a from-scratch, Mem0-v0.1.118-style memory library.

See docs/design/02-lld-memlayer.md for the full design. Public API (Memory,
MemoryConfig) is exported here once implemented in Phase 1.
"""

from memlayer.errors import ConfigError, LLMResponseError, MemLayerError, ScopeError

__all__ = [
    "MemLayerError",
    "ConfigError",
    "LLMResponseError",
    "ScopeError",
]
