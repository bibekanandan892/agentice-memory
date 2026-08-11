"""Tests for server.dependencies.get_memory() — the real (mocked) wiring path,
distinct from the dependency-override path exercised in
tests/integration/test_server_app.py.
"""

from unittest.mock import patch

import server.dependencies as deps


def test_get_memory_returns_the_same_instance_on_repeated_calls(monkeypatch, tmp_path):
    deps._memory_instance = None
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.setenv("MEMENTO_DATA_DIR", str(tmp_path))

    with patch("memlayer.llms.gemini.genai.Client"):
        first = deps.get_memory()
        second = deps.get_memory()

    assert first is second
    deps._memory_instance = None
