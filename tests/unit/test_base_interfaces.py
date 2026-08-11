"""Tests pinning the three plug-in seam ABCs (LLMBase, EmbeddingBase, VectorStoreBase).

See docs/design/02-lld-memlayer.md §2-4. These ABCs are the frozen contract
every provider (real or fake) must satisfy.
"""

import pytest

from memlayer.embeddings.base import EmbeddingBase
from memlayer.llms.base import LLMBase
from memlayer.vector_stores.base import ScoredPoint, VectorStoreBase


def test_llm_base_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        LLMBase()


def test_embedding_base_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        EmbeddingBase()


def test_vector_store_base_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        VectorStoreBase()


def test_fake_llm_satisfies_llm_base_via_duck_typing(fake_llm):
    # FakeLLM deliberately doesn't subclass LLMBase (keeps tests/conftest.py
    # dependency-free) but must expose the same call signature.
    assert callable(fake_llm.generate_response)


def test_fake_embedder_satisfies_embedding_base_via_duck_typing(fake_embedder):
    assert callable(fake_embedder.embed)
    assert isinstance(fake_embedder.dims, int)


def test_scored_point_is_a_simple_data_holder():
    point = ScoredPoint(id="abc", score=0.9, payload={"data": "hello"})
    assert point.id == "abc"
    assert point.score == 0.9
    assert point.payload["data"] == "hello"
