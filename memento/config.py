"""AppConfig — env/dotenv loading, builds the memlayer config dict.

See docs/design/03-lld-memento.md §1.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

DEFAULT_DATA_DIR = "./data"
DEFAULT_USER_ID = "default_user"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"

GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"


class MissingApiKeyError(RuntimeError):
    """Raised when GEMINI_API_KEY is not set anywhere load_from_env looks.

    cli.main() catches this and prints a friendly one-line fix instead of a
    stack trace (LLD §1's fail-fast-but-friendly requirement).
    """


@dataclass
class AppConfig:
    gemini_api_key: str
    data_dir: Path
    transcript_db_path: Path
    vectors_db_path: Path
    history_db_path: Path
    default_user_id: str
    gemini_model: str

    @classmethod
    def load_from_env(cls, dotenv_path: Path | str | None = None) -> AppConfig:
        load_dotenv(dotenv_path=dotenv_path)

        api_key = os.environ.get(GEMINI_API_KEY_ENV_VAR)
        if not api_key:
            raise MissingApiKeyError(
                f"{GEMINI_API_KEY_ENV_VAR} is not set. Get a free key at "
                "https://aistudio.google.com/ and add it to your .env file "
                "(see .env.example)."
            )

        data_dir = Path(os.environ.get("MEMENTO_DATA_DIR", DEFAULT_DATA_DIR))
        default_user_id = os.environ.get("MEMENTO_DEFAULT_USER_ID", DEFAULT_USER_ID)
        gemini_model = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

        return cls(
            gemini_api_key=api_key,
            data_dir=data_dir,
            transcript_db_path=data_dir / "transcript.db",
            vectors_db_path=data_dir / "vectors.db",
            history_db_path=data_dir / "history.db",
            default_user_id=default_user_id,
            gemini_model=gemini_model,
        )

    def to_memlayer_config(self) -> dict[str, Any]:
        """Shape MemoryConfig.from_dict() expects — see memlayer/config.py."""
        return {
            "llm": {
                "provider": "gemini",
                "config": {"api_key": self.gemini_api_key, "model": self.gemini_model},
            },
            "embedder": {"provider": "sentence_transformer", "config": {}},
            "vector_store": {"provider": "local", "config": {"db_path": str(self.vectors_db_path)}},
            "history_db_path": str(self.history_db_path),
        }
