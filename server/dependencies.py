"""FastAPI dependency wiring — a lazily-built, process-wide Memory singleton.

Tests never call get_memory() for real: they override it via
`app.dependency_overrides[get_memory] = lambda: fake_memory_instance`
(FastAPI TestClient), so this module's real construction path (which needs
GEMINI_API_KEY) never runs under pytest.
"""

from __future__ import annotations

from memlayer.memory import Memory

_memory_instance: Memory | None = None


def get_memory() -> Memory:
    global _memory_instance
    if _memory_instance is None:
        from memento.config import AppConfig

        config = AppConfig.load_from_env()
        _memory_instance = Memory.from_config(config.to_memlayer_config())
    return _memory_instance
