"""Tests for memento.config.AppConfig and memento.transcript.TranscriptStore.

See docs/design/03-lld-memento.md §1.
"""

from pathlib import Path

import pytest

from memento.config import AppConfig, MissingApiKeyError
from memento.transcript import TranscriptStore


class TestAppConfigLoadFromEnv:
    def test_loads_required_and_optional_values(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
        monkeypatch.setenv("MEMENTO_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("MEMENTO_DEFAULT_USER_ID", "bibek")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-flash-latest")

        config = AppConfig.load_from_env(dotenv_path=tmp_path / "does-not-exist.env")

        assert config.gemini_api_key == "test-key-123"
        assert config.data_dir == tmp_path
        assert config.default_user_id == "bibek"
        assert config.gemini_model == "gemini-flash-latest"

    def test_missing_api_key_raises_clear_error(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(MissingApiKeyError, match="GEMINI_API_KEY"):
            AppConfig.load_from_env(dotenv_path=tmp_path / "does-not-exist.env")

    def test_defaults_applied_when_optional_vars_absent(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
        monkeypatch.delenv("MEMENTO_DATA_DIR", raising=False)
        monkeypatch.delenv("MEMENTO_DEFAULT_USER_ID", raising=False)
        monkeypatch.delenv("GEMINI_MODEL", raising=False)

        config = AppConfig.load_from_env(dotenv_path=tmp_path / "does-not-exist.env")

        assert config.data_dir == Path("./data")
        assert config.default_user_id == "default_user"
        assert config.gemini_model == "gemini-flash-latest"

    def test_derives_db_paths_under_data_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
        monkeypatch.setenv("MEMENTO_DATA_DIR", str(tmp_path))

        config = AppConfig.load_from_env(dotenv_path=tmp_path / "does-not-exist.env")

        assert config.transcript_db_path == tmp_path / "transcript.db"
        assert config.vectors_db_path == tmp_path / "vectors.db"
        assert config.history_db_path == tmp_path / "history.db"


class TestAppConfigToMemlayerConfig:
    def test_produces_a_dict_memory_config_can_parse(self, tmp_path):
        config = AppConfig(
            gemini_api_key="test-key",
            data_dir=tmp_path,
            transcript_db_path=tmp_path / "transcript.db",
            vectors_db_path=tmp_path / "vectors.db",
            history_db_path=tmp_path / "history.db",
            default_user_id="default_user",
            gemini_model="gemini-flash-latest",
        )
        memlayer_config = config.to_memlayer_config()

        assert memlayer_config["llm"]["provider"] == "gemini"
        assert memlayer_config["llm"]["config"]["api_key"] == "test-key"
        assert memlayer_config["embedder"]["provider"] == "sentence_transformer"
        assert memlayer_config["vector_store"]["config"]["db_path"] == str(tmp_path / "vectors.db")
        assert memlayer_config["history_db_path"] == str(tmp_path / "history.db")

        from memlayer.config import MemoryConfig

        MemoryConfig.from_dict(memlayer_config)  # must not raise


class TestTranscriptStore:
    def test_log_and_recent_round_trip(self, tmp_path):
        store = TranscriptStore(db_path=tmp_path / "transcript.db")
        store.log("session-1", "alice", "user", "hello")
        store.log("session-1", "alice", "assistant", "hi there")

        recent = store.recent("alice", n=6)
        assert [m["content"] for m in recent] == ["hello", "hi there"]
        assert [m["role"] for m in recent] == ["user", "assistant"]

    def test_recent_respects_n_limit_keeping_most_recent(self, tmp_path):
        store = TranscriptStore(db_path=tmp_path / "transcript.db")
        for i in range(5):
            store.log("session-1", "alice", "user", f"message {i}")

        recent = store.recent("alice", n=2)
        assert [m["content"] for m in recent] == ["message 3", "message 4"]

    def test_recent_is_scoped_to_user(self, tmp_path):
        store = TranscriptStore(db_path=tmp_path / "transcript.db")
        store.log("session-1", "alice", "user", "alice says hi")
        store.log("session-1", "bob", "user", "bob says hi")

        assert [m["content"] for m in store.recent("alice", n=10)] == ["alice says hi"]
        assert [m["content"] for m in store.recent("bob", n=10)] == ["bob says hi"]

    def test_recent_for_unknown_user_is_empty(self, tmp_path):
        store = TranscriptStore(db_path=tmp_path / "transcript.db")
        assert store.recent("nobody", n=6) == []

    def test_reset_clears_all_messages(self, tmp_path):
        store = TranscriptStore(db_path=tmp_path / "transcript.db")
        store.log("session-1", "alice", "user", "hello")
        store.reset()
        assert store.recent("alice", n=10) == []

    def test_log_never_raises_on_write_failure(self, tmp_path, monkeypatch):
        store = TranscriptStore(db_path=tmp_path / "transcript.db")

        def boom(*args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(store, "_conn", None)  # force an AttributeError-ish failure path
        store._execute = boom  # type: ignore[method-assign]
        store.log("session-1", "alice", "user", "hello")  # must not raise
