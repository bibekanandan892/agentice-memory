"""Config dataclasses and MemoryConfig.from_dict() parsing/validation.

See docs/design/02-lld-memlayer.md §1. This is the highest-leverage design
review in the project: these shapes freeze the plug-in seams that Phase 1
tasks 1.3-1.6 implement against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memlayer.errors import ConfigError

KNOWN_LLM_PROVIDERS = {"gemini"}
KNOWN_EMBEDDER_PROVIDERS = {"sentence_transformer", "gemini"}
KNOWN_VECTOR_STORE_PROVIDERS = {"local"}

DEFAULT_HISTORY_DB_PATH = "./data/history.db"


@dataclass
class LlmConfig:
    provider: str = "gemini"
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbedderConfig:
    provider: str = "sentence_transformer"
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorStoreConfig:
    provider: str = "local"
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    llm: LlmConfig = field(default_factory=LlmConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    history_db_path: Path = field(default_factory=lambda: Path(DEFAULT_HISTORY_DB_PATH))
    custom_instructions: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> MemoryConfig:
        """Parse a plain config dict into a MemoryConfig, applying defaults for
        any missing section and raising ConfigError (fail fast) for an unknown
        provider name in any section.
        """
        data = data or {}

        llm = _parse_component(
            data.get("llm"), LlmConfig, "llm", KNOWN_LLM_PROVIDERS
        )
        embedder = _parse_component(
            data.get("embedder"), EmbedderConfig, "embedder", KNOWN_EMBEDDER_PROVIDERS
        )
        vector_store = _parse_component(
            data.get("vector_store"),
            VectorStoreConfig,
            "vector_store",
            KNOWN_VECTOR_STORE_PROVIDERS,
        )

        history_db_path = Path(data.get("history_db_path", DEFAULT_HISTORY_DB_PATH))

        return cls(
            llm=llm,
            embedder=embedder,
            vector_store=vector_store,
            history_db_path=history_db_path,
            custom_instructions=data.get("custom_instructions"),
        )


def _parse_component(
    section: dict[str, Any] | None,
    component_cls: type,
    section_name: str,
    allowed_providers: set[str],
) -> Any:
    section = section or {}
    provider = section.get("provider", component_cls().provider)
    if provider not in allowed_providers:
        allowed = ", ".join(sorted(allowed_providers))
        raise ConfigError(
            f"Unknown {section_name} provider {provider!r}. Allowed providers: {allowed}."
        )
    return component_cls(provider=provider, config=section.get("config", {}))
