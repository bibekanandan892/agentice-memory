"""Unit tests for memlayer.llms.gemini.GeminiLLM.

CRITICAL: these tests must never make a real network call. The google-genai
`Client` class is always replaced with a mock before a `GeminiLLM` is
constructed (see `_make_llm` below). `time.sleep` is monkeypatched to a no-op
in the retry tests so exponential backoff never actually blocks the suite.

See docs/design/02-lld-memlayer.md §2.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from google.genai import errors as genai_errors

from memlayer.errors import ConfigError, LLMResponseError
from memlayer.llms import gemini as gemini_module

FAKE_API_KEY = "test-api-key"


def _rate_limit_error(retry_delay: str | None = None) -> genai_errors.ClientError:
    """Build a real ClientError shaped like Gemini's 429 RESOURCE_EXHAUSTED body."""
    error_body: dict = {
        "code": 429,
        "message": "Quota exceeded",
        "status": "RESOURCE_EXHAUSTED",
    }
    if retry_delay is not None:
        error_body["details"] = [
            {
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": retry_delay,
            }
        ]
    return genai_errors.ClientError(429, {"error": error_body})


def _bad_request_error() -> genai_errors.ClientError:
    """A non-rate-limit 4xx error — must never trigger a retry."""
    return genai_errors.ClientError(
        400, {"error": {"code": 400, "message": "Invalid argument", "status": "INVALID_ARGUMENT"}}
    )


def _response(text: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(text=text)


@pytest.fixture
def mock_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch google.genai.Client so GeminiLLM() never touches the network,
    and return the mock client instance that will be constructed.
    """
    client_instance = MagicMock()
    client_class = MagicMock(return_value=client_instance)
    monkeypatch.setattr(gemini_module.genai, "Client", client_class)
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_API_KEY)
    return client_instance


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace time.sleep with a recorder so retry tests never actually block."""
    recorded_delays: list[float] = []
    monkeypatch.setattr(
        gemini_module.time, "sleep", lambda seconds: recorded_delays.append(seconds)
    )
    return recorded_delays


class TestConstruction:
    def test_missing_api_key_raises_before_any_client_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        client_class = MagicMock()
        monkeypatch.setattr(gemini_module.genai, "Client", client_class)

        with pytest.raises((ConfigError, ValueError), match="(?i)api.?key"):
            gemini_module.GeminiLLM()

        client_class.assert_not_called()

    def test_env_var_api_key_is_used_when_no_constructor_arg(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "from-env")
        client_class = MagicMock()
        monkeypatch.setattr(gemini_module.genai, "Client", client_class)

        gemini_module.GeminiLLM()

        client_class.assert_called_once_with(api_key="from-env")

    def test_constructor_arg_api_key_takes_precedence_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "from-env")
        client_class = MagicMock()
        monkeypatch.setattr(gemini_module.genai, "Client", client_class)

        gemini_module.GeminiLLM(api_key="from-arg")

        client_class.assert_called_once_with(api_key="from-arg")


class TestMapMessages:
    def test_leading_system_message_becomes_system_instruction(self) -> None:
        llm = gemini_module.GeminiLLM.__new__(gemini_module.GeminiLLM)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]

        system_instruction, contents = llm._map_messages(messages)

        assert system_instruction == "You are helpful."
        assert contents == [{"role": "user", "parts": [{"text": "Hi"}]}]

    def test_assistant_role_maps_to_model(self) -> None:
        llm = gemini_module.GeminiLLM.__new__(gemini_module.GeminiLLM)
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]

        _, contents = llm._map_messages(messages)

        assert contents == [
            {"role": "user", "parts": [{"text": "Hi"}]},
            {"role": "model", "parts": [{"text": "Hello!"}]},
        ]

    def test_no_system_message_yields_none_system_instruction(self) -> None:
        llm = gemini_module.GeminiLLM.__new__(gemini_module.GeminiLLM)
        messages = [{"role": "user", "content": "Hi"}]

        system_instruction, contents = llm._map_messages(messages)

        assert system_instruction is None
        assert contents == [{"role": "user", "parts": [{"text": "Hi"}]}]

    def test_multiple_leading_system_messages_are_concatenated(self) -> None:
        llm = gemini_module.GeminiLLM.__new__(gemini_module.GeminiLLM)
        messages = [
            {"role": "system", "content": "Rule one."},
            {"role": "system", "content": "Rule two."},
            {"role": "user", "content": "Hi"},
        ]

        system_instruction, contents = llm._map_messages(messages)

        assert system_instruction == "Rule one.\n\nRule two."
        assert contents == [{"role": "user", "parts": [{"text": "Hi"}]}]

    def test_empty_message_list(self) -> None:
        llm = gemini_module.GeminiLLM.__new__(gemini_module.GeminiLLM)

        system_instruction, contents = llm._map_messages([])

        assert system_instruction is None
        assert contents == []


class TestResponseFormat:
    def test_json_response_format_sets_json_mime_type(self, mock_client: MagicMock) -> None:
        mock_client.models.generate_content.return_value = _response('{"facts": []}')
        llm = gemini_module.GeminiLLM()

        result = llm.generate_response(
            [{"role": "user", "content": "extract facts"}], response_format="json"
        )

        assert result == '{"facts": []}'
        _, kwargs = mock_client.models.generate_content.call_args
        assert kwargs["config"].response_mime_type == "application/json"
        assert kwargs["model"] == gemini_module.DEFAULT_MODEL
        assert kwargs["contents"] == [{"role": "user", "parts": [{"text": "extract facts"}]}]

    def test_text_response_format_does_not_set_json_mime_type(
        self, mock_client: MagicMock
    ) -> None:
        mock_client.models.generate_content.return_value = _response("plain text")
        llm = gemini_module.GeminiLLM()

        result = llm.generate_response(
            [{"role": "user", "content": "say hi"}], response_format="text"
        )

        assert result == "plain text"
        _, kwargs = mock_client.models.generate_content.call_args
        assert kwargs["config"].response_mime_type is None

    def test_system_instruction_and_temperature_are_forwarded(
        self, mock_client: MagicMock
    ) -> None:
        mock_client.models.generate_content.return_value = _response("ok")
        llm = gemini_module.GeminiLLM(temperature=0.7)

        llm.generate_response(
            [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Hi"},
            ]
        )

        _, kwargs = mock_client.models.generate_content.call_args
        assert kwargs["config"].system_instruction == "Be terse."
        assert kwargs["config"].temperature == 0.7


class TestRetryBehavior:
    def test_succeeds_after_two_rate_limit_errors(self, mock_client: MagicMock) -> None:
        mock_client.models.generate_content.side_effect = [
            _rate_limit_error(),
            _rate_limit_error(),
            _response("finally"),
        ]
        llm = gemini_module.GeminiLLM(max_retries=3)

        result = llm.generate_response([{"role": "user", "content": "Hi"}])

        assert result == "finally"
        assert mock_client.models.generate_content.call_count == 3

    def test_retries_exhausted_raises_llm_response_error(self, mock_client: MagicMock) -> None:
        mock_client.models.generate_content.side_effect = _rate_limit_error()
        llm = gemini_module.GeminiLLM(max_retries=3)

        with pytest.raises(LLMResponseError):
            llm.generate_response([{"role": "user", "content": "Hi"}])

        assert mock_client.models.generate_content.call_count == 3

    def test_falls_back_to_exponential_backoff_when_error_body_is_malformed(
        self, mock_client: MagicMock, no_real_sleep: list[float]
    ) -> None:
        # error_body["error"] is a string, not a dict -> the RetryInfo digger
        # must bail out gracefully and fall back to exponential backoff,
        # rather than raising or misbehaving on a shape it can't parse.
        malformed = genai_errors.ClientError(429, {"error": "not a dict"})
        mock_client.models.generate_content.side_effect = [malformed, _response("ok")]
        llm = gemini_module.GeminiLLM(max_retries=3)

        result = llm.generate_response([{"role": "user", "content": "Hi"}])

        assert result == "ok"
        assert len(no_real_sleep) == 1
        assert no_real_sleep[0] >= gemini_module.BACKOFF_BASE_SECONDS

    def test_non_rate_limit_error_propagates_without_retry(self, mock_client: MagicMock) -> None:
        mock_client.models.generate_content.side_effect = _bad_request_error()
        llm = gemini_module.GeminiLLM(max_retries=3)

        with pytest.raises(genai_errors.ClientError):
            llm.generate_response([{"role": "user", "content": "Hi"}])

        assert mock_client.models.generate_content.call_count == 1

    def test_unexpected_exception_type_propagates_without_retry(
        self, mock_client: MagicMock
    ) -> None:
        mock_client.models.generate_content.side_effect = RuntimeError("boom")
        llm = gemini_module.GeminiLLM(max_retries=3)

        with pytest.raises(RuntimeError, match="boom"):
            llm.generate_response([{"role": "user", "content": "Hi"}])

        assert mock_client.models.generate_content.call_count == 1

    def test_backoff_sleeps_between_attempts_but_not_after_final_failure(
        self, mock_client: MagicMock, no_real_sleep: list[float]
    ) -> None:
        mock_client.models.generate_content.side_effect = _rate_limit_error()
        llm = gemini_module.GeminiLLM(max_retries=3)

        with pytest.raises(LLMResponseError):
            llm.generate_response([{"role": "user", "content": "Hi"}])

        # 3 attempts total -> sleeps between attempt 1->2 and 2->3, not after the 3rd.
        assert len(no_real_sleep) == 2

    def test_server_provided_retry_delay_is_honored(
        self, mock_client: MagicMock, no_real_sleep: list[float]
    ) -> None:
        mock_client.models.generate_content.side_effect = [
            _rate_limit_error(retry_delay="2.5s"),
            _response("ok"),
        ]
        llm = gemini_module.GeminiLLM(max_retries=3)

        llm.generate_response([{"role": "user", "content": "Hi"}])

        assert no_real_sleep == [2.5]


class TestPublicApiSurface:
    def test_gemini_llm_implements_llm_base(self) -> None:
        assert issubclass(gemini_module.GeminiLLM, gemini_module.LLMBase)
