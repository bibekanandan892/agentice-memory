"""Tests for memlayer.config — MemoryConfig and from_config() parsing.

See docs/design/02-lld-memlayer.md §1.
"""

from pathlib import Path

import pytest

from memlayer.config import MemoryConfig
from memlayer.errors import ConfigError


class TestDefaults:
    def test_empty_dict_uses_all_defaults(self):
        config = MemoryConfig.from_dict({})
        assert config.llm.provider == "gemini"
        assert config.embedder.provider == "sentence_transformer"
        assert config.vector_store.provider == "local"

    def test_none_is_treated_like_empty_dict(self):
        config = MemoryConfig.from_dict(None)
        assert config.llm.provider == "gemini"

    def test_default_history_db_path(self):
        config = MemoryConfig.from_dict({})
        assert "history.db" in str(config.history_db_path)


class TestPartialOverride:
    def test_overriding_llm_provider_only(self, tmp_path: Path):
        config = MemoryConfig.from_dict(
            {"llm": {"provider": "gemini", "config": {"model": "gemini-flash-latest"}}}
        )
        assert config.llm.provider == "gemini"
        assert config.llm.config["model"] == "gemini-flash-latest"
        # Untouched sections still fall back to defaults.
        assert config.embedder.provider == "sentence_transformer"

    def test_overriding_history_db_path(self, tmp_path: Path):
        custom_path = tmp_path / "custom_history.db"
        config = MemoryConfig.from_dict({"history_db_path": str(custom_path)})
        assert Path(config.history_db_path) == custom_path

    def test_vector_store_config_dict_is_preserved(self, tmp_path: Path):
        config = MemoryConfig.from_dict(
            {"vector_store": {"provider": "local", "config": {"db_path": str(tmp_path)}}}
        )
        assert config.vector_store.config["db_path"] == str(tmp_path)


class TestValidation:
    def test_unknown_llm_provider_raises_config_error(self):
        with pytest.raises(ConfigError, match="llm"):
            MemoryConfig.from_dict({"llm": {"provider": "not-a-real-provider"}})

    def test_unknown_embedder_provider_raises_config_error(self):
        with pytest.raises(ConfigError, match="embedder"):
            MemoryConfig.from_dict({"embedder": {"provider": "not-a-real-provider"}})

    def test_unknown_vector_store_provider_raises_config_error(self):
        with pytest.raises(ConfigError, match="vector_store"):
            MemoryConfig.from_dict({"vector_store": {"provider": "not-a-real-provider"}})

    def test_error_message_lists_allowed_providers(self):
        with pytest.raises(ConfigError, match="gemini"):
            MemoryConfig.from_dict({"llm": {"provider": "bogus"}})


class TestPathNormalization:
    def test_windows_style_path_round_trips(self):
        config = MemoryConfig.from_dict({"history_db_path": r"C:\data\history.db"})
        assert isinstance(config.history_db_path, Path)

    def test_pathlib_path_input_accepted(self, tmp_path: Path):
        config = MemoryConfig.from_dict({"history_db_path": tmp_path / "history.db"})
        assert Path(config.history_db_path) == tmp_path / "history.db"
