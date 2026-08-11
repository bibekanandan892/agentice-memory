"""Tests for the FakeLLM / FakeEmbedder test doubles themselves.

These fakes shape every later test in the project, so their behavior is
pinned down explicitly here.
"""

import pytest

from tests.conftest import FakeEmbedder, FakeLLM


class TestFakeLLM:
    def test_returns_queued_responses_in_order(self):
        llm = FakeLLM(["first", "second"])
        assert llm.generate_response([{"role": "user", "content": "a"}]) == "first"
        assert llm.generate_response([{"role": "user", "content": "b"}]) == "second"

    def test_queue_appends_additional_responses(self):
        llm = FakeLLM(["first"])
        llm.queue("second")
        assert llm.generate_response([]) == "first"
        assert llm.generate_response([]) == "second"

    def test_records_prompts_seen(self):
        llm = FakeLLM(["ok"])
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        llm.generate_response(messages)
        assert llm.prompts_seen == [messages]

    def test_raises_clear_error_when_exhausted(self):
        llm = FakeLLM([])
        with pytest.raises(AssertionError, match="fewer FakeLLM responses"):
            llm.generate_response([])


class TestFakeEmbedder:
    def test_identical_text_produces_identical_vector(self):
        embedder = FakeEmbedder(dims=8)
        v1 = embedder.embed("hello world")
        v2 = embedder.embed("hello world")
        assert v1 == v2

    def test_different_text_produces_different_vectors(self):
        embedder = FakeEmbedder(dims=8)
        v1 = embedder.embed("hello world")
        v2 = embedder.embed("goodbye world")
        assert v1 != v2

    def test_vector_has_configured_dims(self):
        embedder = FakeEmbedder(dims=16)
        assert len(embedder.embed("anything")) == 16
        assert embedder.dims == 16

    def test_vector_is_unit_normalized(self):
        import numpy as np

        embedder = FakeEmbedder(dims=8)
        vector = np.array(embedder.embed("normalize me"))
        assert np.isclose(np.linalg.norm(vector), 1.0)
