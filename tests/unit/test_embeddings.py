"""Tests for the two concrete EmbeddingBase providers.

`SentenceTransformerEmbedder` (default, local) and `GeminiEmbedder` (optional,
remote) both implement the same `EmbeddingBase` interface (see
docs/design/02-lld-memlayer.md §3).

`sentence-transformers` is NOT installed in this environment (it lives behind
the optional `[local]` extra, see pyproject.toml), so its tests mock
`sentence_transformers.SentenceTransformer` entirely via a fake module
injected into `sys.modules` — this exercises the real lazy-import code path
without requiring the actual (heavy) dependency.

`GeminiEmbedder` tests mock `genai.Client` entirely; no real network calls are
ever made.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from unittest import mock

import numpy as np
import pytest

from memlayer.embeddings.gemini import GeminiEmbedder
from memlayer.embeddings.sentence_transformer import SentenceTransformerEmbedder

# ---------------------------------------------------------------------------
# SentenceTransformerEmbedder
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    """Inject a fake `sentence_transformers` module into sys.modules so the
    lazy `from sentence_transformers import SentenceTransformer` inside
    `_ensure_loaded()` succeeds without the real (optional) library installed.

    Returns the mocked `SentenceTransformer` class; `.return_value` is the
    mocked model instance handed back by every construction call.
    """
    fake_module = types.ModuleType("sentence_transformers")
    mock_model_cls = mock.MagicMock(name="SentenceTransformer")
    mock_model_instance = mock.MagicMock(name="model_instance")
    mock_model_instance.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_model_instance.get_sentence_embedding_dimension.return_value = 384
    mock_model_cls.return_value = mock_model_instance
    fake_module.SentenceTransformer = mock_model_cls

    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    return mock_model_cls


class TestSentenceTransformerEmbedder:
    def test_construction_does_not_raise_and_does_not_load_model(self):
        embedder = SentenceTransformerEmbedder()
        assert embedder._model is None

    def test_first_embed_call_constructs_model_exactly_once(self, fake_sentence_transformers):
        embedder = SentenceTransformerEmbedder()
        embedder.embed("hello world")
        fake_sentence_transformers.assert_called_once_with(embedder.model_name)

    def test_second_embed_call_reuses_model_instance(self, fake_sentence_transformers):
        embedder = SentenceTransformerEmbedder()
        embedder.embed("first call")
        embedder.embed("second call")
        assert fake_sentence_transformers.call_count == 1

    def test_embed_returns_list_of_floats_from_model_encode(self, fake_sentence_transformers):
        embedder = SentenceTransformerEmbedder()
        result = embedder.embed("hello world")
        assert result == pytest.approx([0.1, 0.2, 0.3])
        assert isinstance(result, list)

    def test_dims_is_384_before_and_after_load(self, fake_sentence_transformers):
        embedder = SentenceTransformerEmbedder()
        assert embedder.dims == 384
        embedder.embed("hello world")
        assert embedder.dims == 384

    def test_embed_raises_clear_import_error_when_library_missing(self):
        if importlib.util.find_spec("sentence_transformers") is not None:
            pytest.skip("sentence-transformers is installed in this environment")
        embedder = SentenceTransformerEmbedder()
        with pytest.raises(ImportError, match="uv sync --extra local"):
            embedder.embed("hello")


# ---------------------------------------------------------------------------
# GeminiEmbedder
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_genai_client():
    with mock.patch("memlayer.embeddings.gemini.genai.Client") as mock_client_cls:
        yield mock_client_cls


class TestGeminiEmbedder:
    def test_missing_api_key_raises_clear_error_at_construction(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            GeminiEmbedder()

    def test_constructor_arg_api_key_is_used_over_env_var(self, monkeypatch, mock_genai_client):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        GeminiEmbedder(api_key="explicit-key")
        mock_genai_client.assert_called_once_with(api_key="explicit-key")

    def test_env_var_api_key_used_when_no_constructor_arg(self, monkeypatch, mock_genai_client):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        GeminiEmbedder()
        mock_genai_client.assert_called_once_with(api_key="env-key")

    def test_embed_sends_text_to_embed_content_and_maps_response(
        self, monkeypatch, mock_genai_client
    ):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        mock_client = mock_genai_client.return_value
        mock_embedding = mock.MagicMock()
        mock_embedding.values = [0.1, 0.2, 0.3]
        mock_client.models.embed_content.return_value = mock.MagicMock(
            embeddings=[mock_embedding]
        )

        embedder = GeminiEmbedder()
        result = embedder.embed("hello world")

        mock_client.models.embed_content.assert_called_once_with(
            model="text-embedding-004", contents="hello world"
        )
        assert result == [0.1, 0.2, 0.3]

    def test_dims_is_768_by_default(self, monkeypatch, mock_genai_client):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        embedder = GeminiEmbedder()
        assert embedder.dims == 768

    def test_embed_raises_when_response_has_no_embeddings(self, monkeypatch, mock_genai_client):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        mock_client = mock_genai_client.return_value
        mock_client.models.embed_content.return_value = mock.MagicMock(embeddings=[])

        embedder = GeminiEmbedder()
        with pytest.raises(RuntimeError, match="empty embeddings"):
            embedder.embed("hello world")
