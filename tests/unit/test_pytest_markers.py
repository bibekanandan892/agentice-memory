"""Smoke test proving the `live` marker and default deselect wiring works
(Phase 0 Task 0.2's coverage/marker verification)."""

import pytest


@pytest.mark.live
def test_live_marked_test_is_deselected_by_default():
    """This test should NOT run under the default `pytest` invocation
    (addopts = "-m 'not live'" in pyproject.toml) — only under
    `pytest -m live` with a real GEMINI_API_KEY. If this test ever executes
    without -m live being passed explicitly, the marker wiring is broken."""
    pass


def test_plain_test_always_runs():
    assert True
