"""A hosted provider's failure reaches the user as what stopped and whose side it is on (#1630).

Each case is a response the provider is documented to send -- a retired Gemini model, a
revoked Anthropic key, a spent OpenAI quota -- fed through the connector's real request
path by `httpx.MockTransport`, on both `generate` and `stream`. The two halves of the P8
plain-copy rule are asserted separately: that the message names the provider (and the
model, where the model is the cause), and that it carries none of the response.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import httpx
import pytest

from uclone_x.agent.models import ProviderFailure
from uclone_x.errors import (
    LLMProviderError,
    ModelNotAvailableError,
    ProviderAuthError,
    ProviderFailureError,
    ProviderFailureKind,
    ProviderOutageError,
    ProviderQuotaError,
    ProviderResponseError,
    ProviderUnreachableError,
)
from uclone_x.llm import ChatMessage, LLMRequest, MessageRole
from uclone_x.llm.connectors.anthropic import AnthropicConnector
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.connectors.failures import failed_status
from uclone_x.llm.connectors.gemini import GeminiConnector
from uclone_x.llm.connectors.openai import OpenAIConnector
from uclone_x.llm.connectors.vllm import VLLMConnector

_FAILURES_LOGGER = "uclone_x.llm.connectors.failures"

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _gemini(handler: Handler) -> BaseLLMConnector:
    return GeminiConnector(api_key="key", http_client=_client(handler))


def _anthropic(handler: Handler) -> BaseLLMConnector:
    return AnthropicConnector(api_key="key", http_client=_client(handler))


def _openai(handler: Handler) -> BaseLLMConnector:
    return OpenAIConnector(api_key="key", http_client=_client(handler))


def _vllm(handler: Handler) -> BaseLLMConnector:
    return VLLMConnector(base_url="http://vllm.invalid:8000/v1", http_client=_client(handler))


@dataclass(frozen=True)
class Case:
    """One documented provider response and the failure it must become."""

    name: str
    connector: Callable[[Handler], BaseLLMConnector]
    provider: str
    model: str
    status: int
    body: str
    expected: type[ProviderFailureError]
    #: A fragment of `body` that identifies it, which the user-facing message must not carry.
    leak: str


CASES = (
    Case(
        "gemini-retired-model",
        _gemini,
        "Google",
        "gemini-1.5-pro",
        404,
        '{"error": {"code": 404, "message": "models/gemini-1.5-pro is not found for API '
        'version v1beta, or is not supported for generateContent.", "status": "NOT_FOUND"}}',
        ModelNotAvailableError,
        "NOT_FOUND",
    ),
    Case(
        "gemini-bad-key",
        _gemini,
        "Google",
        "gemini-2.5-flash",
        400,
        '{"error": {"code": 400, "message": "API key not valid. Please pass a valid API '
        'key.", "status": "INVALID_ARGUMENT", "details": [{"reason": "API_KEY_INVALID"}]}}',
        ProviderAuthError,
        "INVALID_ARGUMENT",
    ),
    Case(
        "gemini-quota",
        _gemini,
        "Google",
        "gemini-2.5-flash",
        429,
        '{"error": {"code": 429, "message": "You exceeded your current quota.", '
        '"status": "RESOURCE_EXHAUSTED"}}',
        ProviderQuotaError,
        "RESOURCE_EXHAUSTED",
    ),
    Case(
        "gemini-overloaded",
        _gemini,
        "Google",
        "gemini-2.5-flash",
        503,
        '{"error": {"code": 503, "message": "The model is overloaded. Please try again '
        'later.", "status": "UNAVAILABLE"}}',
        ProviderOutageError,
        "UNAVAILABLE",
    ),
    Case(
        "gemini-unrecognised",
        _gemini,
        "Google",
        "gemini-2.5-flash",
        400,
        '{"error": {"code": 400, "message": "Invalid JSON payload received.", '
        '"status": "INVALID_ARGUMENT"}}',
        ProviderResponseError,
        "Invalid JSON payload",
    ),
    Case(
        "anthropic-retired-model",
        _anthropic,
        "Anthropic",
        "claude-3-5-sonnet-20241022",
        404,
        '{"type": "error", "error": {"type": "not_found_error", "message": "model: '
        'claude-3-5-sonnet-20241022"}}',
        ModelNotAvailableError,
        "not_found_error",
    ),
    Case(
        "anthropic-bad-key",
        _anthropic,
        "Anthropic",
        "claude-sonnet-4-5",
        401,
        '{"type": "error", "error": {"type": "authentication_error", "message": '
        '"invalid x-api-key"}}',
        ProviderAuthError,
        "authentication_error",
    ),
    Case(
        "anthropic-no-credit",
        _anthropic,
        "Anthropic",
        "claude-sonnet-4-5",
        400,
        '{"type": "error", "error": {"type": "invalid_request_error", "message": "Your '
        "credit balance is too low to access the Anthropic API. Please go to Plans & "
        'Billing to upgrade or purchase credits."}}',
        ProviderQuotaError,
        "invalid_request_error",
    ),
    Case(
        "anthropic-overloaded",
        _anthropic,
        "Anthropic",
        "claude-sonnet-4-5",
        529,
        '{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}',
        ProviderOutageError,
        "overloaded_error",
    ),
    Case(
        "openai-retired-model",
        _openai,
        "OpenAI",
        "gpt-4-32k",
        404,
        '{"error": {"message": "The model `gpt-4-32k` does not exist or you do not have '
        'access to it.", "type": "invalid_request_error", "code": "model_not_found"}}',
        ModelNotAvailableError,
        "model_not_found",
    ),
    Case(
        "openai-bad-key",
        _openai,
        "OpenAI",
        "gpt-5-mini",
        401,
        '{"error": {"message": "Incorrect API key provided: sk-abc.", "type": '
        '"invalid_request_error", "code": "invalid_api_key"}}',
        ProviderAuthError,
        "invalid_api_key",
    ),
    Case(
        "openai-quota",
        _openai,
        "OpenAI",
        "gpt-5-mini",
        429,
        '{"error": {"message": "You exceeded your current quota, please check your plan '
        'and billing details.", "type": "insufficient_quota", "code": "insufficient_quota"}}',
        ProviderQuotaError,
        "insufficient_quota",
    ),
    Case(
        "openai-server-error",
        _openai,
        "OpenAI",
        "gpt-5-mini",
        500,
        '{"error": {"message": "The server had an error while processing your request.", '
        '"type": "server_error"}}',
        ProviderOutageError,
        "server_error",
    ),
    # Anthropic's 403 for a key that may not use the resource is about the key.
    Case(
        "anthropic-key-without-permission",
        _anthropic,
        "Anthropic",
        "claude-opus-4-1",
        403,
        '{"type": "error", "error": {"type": "permission_error", "message": "Your API key '
        'does not have permission to use the specified resource."}}',
        ProviderAuthError,
        "permission_error",
    ),
    # OpenAI's 403 for an unsupported region is not about the key: a new key would not help,
    # so "did not accept the API key" would send the user to fix the wrong thing.
    Case(
        "openai-region-forbidden",
        _openai,
        "OpenAI",
        "gpt-5-mini",
        403,
        '{"error": {"code": "unsupported_country_region_territory", "message": "Country, '
        'region, or territory not supported", "param": null, "type": "request_forbidden"}}',
        ProviderResponseError,
        "unsupported_country_region_territory",
    ),
    # A mistyped Gemini endpoint: Google's own HTML 404 page echoes the URL, which holds
    # `/models/<id>:generateContent`. The page says "model" and the model is not the problem.
    Case(
        "gemini-wrong-endpoint",
        _gemini,
        "Google",
        "gemini-2.5-flash",
        404,
        "<html><title>Error 404 (Not Found)!!1</title><p>The requested URL "
        "<code>/v1betax/models/gemini-2.5-flash:generateContent</code> was not found on "
        "this server.</p></html>",
        ProviderResponseError,
        "/v1betax/",
    ),
    # A mistyped custom endpoint: a 404 that says nothing about a model is not a retired
    # model, and saying it was would send the user to fix the wrong thing.
    Case(
        "vllm-proxy-not-found",
        _vllm,
        "vLLM",
        "qwen2.5-coder-32b-instruct",
        404,
        "<html><body><h1>Not Found</h1></body></html>",
        ProviderResponseError,
        "<html>",
    ),
)

MODES = ("generate", "stream")


def _request(model: str) -> LLMRequest:
    return LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),), model=model)


async def _call(connector: BaseLLMConnector, mode: str, model: str) -> None:
    if mode == "generate":
        await connector.generate(_request(model))
        return
    async for _ in connector.stream(_request(model)):  # pragma: no branch
        pass


async def _failure(case: Case, mode: str) -> ProviderFailureError:
    connector = case.connector(lambda _request: httpx.Response(case.status, text=case.body))
    with pytest.raises(ProviderFailureError) as excinfo:
        await _call(connector, mode, case.model)
    return excinfo.value


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
@pytest.mark.asyncio
async def test_each_documented_response_becomes_the_failure_that_names_its_cause(
    case: Case, mode: str
) -> None:
    """The kind decides the remedy a head offers, so it is asserted exactly, not by base."""
    exc = await _failure(case, mode)

    assert type(exc) is case.expected
    assert isinstance(exc, LLMProviderError), "callers catching the old base still catch it"
    assert exc.provider == case.provider
    assert exc.model == case.model


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
@pytest.mark.asyncio
async def test_the_message_says_in_plain_words_whose_side_the_problem_is_on(
    case: Case, mode: str
) -> None:
    """Plain-copy half: the provider is named, and the model where the model is the cause."""
    message = str(await _failure(case, mode))

    assert case.provider in message
    if case.expected is ModelNotAvailableError:
        assert case.model in message
        assert "retired" in message


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
@pytest.mark.asyncio
async def test_the_message_carries_none_of_the_response(case: Case, mode: str) -> None:
    """No-internals half: no status, body, markup, URL or class name reaches the user.

    The model is removed before the digit check, because a model name may carry digits of
    its own and is the one identifier the message is meant to show.
    """
    exc = await _failure(case, mode)
    message = str(exc)
    without_model = message.replace(case.model, "")

    assert case.leak not in message
    assert str(case.status) not in without_model
    assert "{" not in message
    assert "<" not in message
    assert "http" not in message.lower()
    assert type(exc).__name__ not in message


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
@pytest.mark.asyncio
async def test_the_raw_response_is_kept_in_the_log(
    case: Case, caplog: pytest.LogCaptureFixture
) -> None:
    """The raw response is kept for whoever reads the log, since the message drops it (P6)."""
    with caplog.at_level(logging.INFO, logger=_FAILURES_LOGGER):
        await _failure(case, "generate")

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert case.leak in logged
    assert str(case.status) in logged


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
@pytest.mark.asyncio
async def test_the_raw_response_is_logged_below_the_level_that_reaches_the_terminal(
    case: Case, caplog: pytest.LogCaptureFixture
) -> None:
    """No head configures logging, so Python prints WARNING and above to stderr as it is.

    A raw response logged there reaches the user's terminal beside the plain sentence that
    was written to replace it -- the status and JSON this whole change keeps off it.

    Killed by: src/uclone_x/llm/connectors/failures.py :: _RAW_RESPONSE_LOG_LEVEL = logging.INFO
    Becomes: _RAW_RESPONSE_LOG_LEVEL = logging.WARNING
    """
    with caplog.at_level(logging.DEBUG, logger=_FAILURES_LOGGER):
        await _failure(case, "generate")

    loud = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not loud
    assert caplog.records, "the response must still be logged, only not loudly"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("connector", "provider"),
    ((_gemini, "Google"), (_anthropic, "Anthropic"), (_openai, "OpenAI"), (_vllm, "vLLM")),
)
@pytest.mark.asyncio
async def test_a_request_that_gets_no_answer_is_unreachable(
    connector: Callable[[Handler], BaseLLMConnector], provider: str, mode: str
) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 61] Connection refused", request=request)

    with pytest.raises(ProviderUnreachableError) as excinfo:
        await _call(connector(refuse), mode, "some-model")

    message = str(excinfo.value)
    assert provider in message
    assert "internet connection" in message
    assert "Errno" not in message
    assert "refused" not in message


@pytest.mark.parametrize(
    ("connector", "provider"),
    ((_gemini, "Google"), (_anthropic, "Anthropic"), (_openai, "OpenAI"), (_vllm, "vLLM")),
)
@pytest.mark.asyncio
async def test_a_200_that_is_not_a_reply_is_an_unrecognised_response(
    connector: Callable[[Handler], BaseLLMConnector], provider: str
) -> None:
    with pytest.raises(ProviderResponseError) as excinfo:
        await _call(
            connector(lambda _request: httpx.Response(200, text="<html>502 Bad Gateway</html>")),
            "generate",
            "some-model",
        )

    message = str(excinfo.value)
    assert provider in message
    assert "<html>" not in message
    assert "502" not in message


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    (
        (403, "Forbidden", ProviderResponseError),
        (403, '{"status": "PERMISSION_DENIED"}', ProviderAuthError),
        (402, "Payment Required", ProviderQuotaError),
        (502, "Bad Gateway", ProviderOutageError),
        (400, "The model is overloaded", ProviderOutageError),
        (400, '{"code": "model_not_found"}', ModelNotAvailableError),
        (404, "Not Found", ProviderResponseError),
        (
            404,
            "The requested URL /v1beta/models/m:generateContent was not found",
            ProviderResponseError,
        ),
        (
            404,
            '{"error": {"message": "models/m is not found for API version v1beta"}}',
            ModelNotAvailableError,
        ),
        (418, "I'm a teapot", ProviderResponseError),
    ),
)
def test_the_classifier_reads_the_status_first_and_the_body_only_where_it_must(
    status: int, body: str, expected: type[ProviderFailureError]
) -> None:
    """A 403 or 404 alone does not name the cause; the body must say key or model.

    Killed by: src/uclone_x/llm/connectors/failures.py :: or (status_code == 403 and any(phrase in text for phrase in _FORBIDDEN_KEY_PHRASES))
    Becomes: or status_code == 403
    Killed by: src/uclone_x/llm/connectors/failures.py :: if (status_code == 404 and any(phrase in text for phrase in _MODEL_404_PHRASES)) or any(
    Becomes: if (status_code == 404 and "model" in text) or any(
    """
    exc = failed_status(provider="Example", model="m", status_code=status, body=body)

    assert type(exc) is expected


def test_a_key_the_provider_refused_is_not_mistaken_for_a_missing_model() -> None:
    """Auth is checked before the model: a 401 whose body mentions the model is still a key."""
    exc = failed_status(provider="Example", model="m", status_code=401, body="no access to model m")

    assert type(exc) is ProviderAuthError


@pytest.mark.parametrize(
    ("exc", "kind", "retryable"),
    (
        (ModelNotAvailableError(provider="P", model="m"), "model_unavailable", False),
        (ProviderAuthError(provider="P", model="m"), "provider_auth", False),
        (ProviderQuotaError(provider="P", model="m"), "provider_quota", True),
        (ProviderUnreachableError(provider="P", model="m"), "provider_unreachable", True),
        (ProviderOutageError(provider="P", model="m"), "provider_outage", True),
        (ProviderResponseError(provider="P", model="m"), "provider_error", True),
    ),
)
def test_a_failure_is_carried_as_its_kind_its_message_and_whether_to_retry(
    exc: ProviderFailureError, kind: str, retryable: bool
) -> None:
    """A retired model and a rejected key cannot succeed by retrying, so no Retry is offered."""
    failure = ProviderFailure.of(exc)

    assert failure.kind == ProviderFailureKind(kind)
    assert failure.message == str(exc)
    assert failure.retryable is retryable
    assert failure.model_dump(mode="json")["kind"] == kind
    # The head names the provider's own key variable from this, not from the sentence.
    assert failure.provider == "P"
