"""Unit tests for LLM provider connectors (Principle 5 & Principle 6)."""

from __future__ import annotations

import ast
import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from uclone_x.core.provenance import ExecutionPath, ServiceRef
from uclone_x.errors import (
    LLMCredentialsNotConfiguredError,
    LLMError,
    LLMProviderError,
    LLMProviderNotConfiguredError,
    LLMTimeoutError,
    MalformedToolCallArgumentsError,
    UnmappableChatMessageError,
)
from uclone_x.llm import (
    AnthropicConnector,
    ChatMessage,
    FinishReason,
    GeminiConnector,
    LLMRequest,
    MessageRole,
    MockLLMConnector,
    OllamaConnector,
    OpenAIConnector,
    StreamChunk,
    TokenCountSource,
    TokenUsage,
    ToolCallRequest,
    ToolDefinition,
    create_llm_connector,
)
from uclone_x.llm.compactor import estimate_reply_tokens, estimate_request_tokens
from uclone_x.llm.connectors import openai as openai_module
from uclone_x.llm.connectors.base import BaseLLMConnector, parse_dict_payload
from uclone_x.llm.connectors.ollama import (
    DEFAULT_OLLAMA_KEEP_ALIVE,
    MODEL_MANAGEMENT_CONNECT_TIMEOUT_SECONDS,
    OLLAMA_ENDPOINT_ENV_VARS,
    PULL_SILENCE_TIMEOUT_SECONDS,
    PULL_TOTAL_BACKSTOP_SECONDS,
    delete_model,
    describe_transport_error,
    normalize_ollama_base_url,
    pull_model,
    resolve_ollama_base_url,
    resolve_ollama_model,
)
from uclone_x.llm.connectors.vllm import (
    VLLM_ENDPOINT_ENV_VARS,
    VLLMConnector,
)


def _make_mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    """Create an AsyncClient with custom MockTransport."""
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


# ======================================================================================
# OllamaConnector Tests
# ======================================================================================


def test_normalize_ollama_base_url() -> None:
    assert normalize_ollama_base_url("http://localhost:11434") == "http://localhost:11434"
    assert normalize_ollama_base_url("http://localhost:11434/") == "http://localhost:11434"
    assert normalize_ollama_base_url("http://localhost:11434///") == "http://localhost:11434"
    assert normalize_ollama_base_url("http://localhost:11434/v1") == "http://localhost:11434"
    assert normalize_ollama_base_url("http://localhost:11434/v1/") == "http://localhost:11434"
    assert normalize_ollama_base_url("http://localhost:11434/v1///") == "http://localhost:11434"
    assert normalize_ollama_base_url("http://localhost:11434/v1/v1/") == "http://localhost:11434"
    assert normalize_ollama_base_url("http://192.0.2.10:11434/v1") == "http://192.0.2.10:11434"
    assert normalize_ollama_base_url("   http://localhost:11434/v1/   ") == "http://localhost:11434"
    assert (
        normalize_ollama_base_url("http://custom-host:8080/prefix/v1")
        == "http://custom-host:8080/prefix"
    )


def test_resolve_ollama_base_url_hierarchy(monkeypatch: pytest.MonkeyPatch) -> None:
    # Clear all relevant environment variables
    for var in ("OLLAMA_BASE_URL", "OLLAMA_FAST_BASE_URL", "LOCAL_LLM_BASE_URL", "OLLAMA_HOST"):
        monkeypatch.delenv(var, raising=False)

    # 1. Default when nothing is set
    assert resolve_ollama_base_url() == "http://localhost:11434"
    assert resolve_ollama_base_url(None) == "http://localhost:11434"
    assert resolve_ollama_base_url("   ") == "http://localhost:11434"

    # 2. OLLAMA_HOST
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama-host:11434/v1/")
    assert resolve_ollama_base_url() == "http://ollama-host:11434"

    # 3. LOCAL_LLM_BASE_URL overrides OLLAMA_HOST
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://local-llm:11434/v1")
    assert resolve_ollama_base_url() == "http://local-llm:11434"

    # 4. OLLAMA_FAST_BASE_URL overrides LOCAL_LLM_BASE_URL
    monkeypatch.setenv("OLLAMA_FAST_BASE_URL", "http://192.0.2.10:11434/v1/")
    assert resolve_ollama_base_url() == "http://192.0.2.10:11434"

    # 5. OLLAMA_BASE_URL overrides OLLAMA_FAST_BASE_URL
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama-primary:11434/v1")
    assert resolve_ollama_base_url() == "http://ollama-primary:11434"

    # 6. Explicit base_url overrides OLLAMA_BASE_URL
    assert resolve_ollama_base_url("http://explicit-host:11434/v1/") == "http://explicit-host:11434"

    # 7. Empty environment variables are skipped in hierarchy
    monkeypatch.setenv("OLLAMA_BASE_URL", "")
    assert resolve_ollama_base_url() == "http://192.0.2.10:11434"


def test_resolve_ollama_model_hierarchy(monkeypatch: pytest.MonkeyPatch) -> None:
    # Clear all relevant environment variables
    for var in ("OLLAMA_MODEL", "OLLAMA_INDEPTH_MODEL", "OLLAMA_FAST_MODEL"):
        monkeypatch.delenv(var, raising=False)

    # 1. Default fallback when all envs are unset and model is None / "" / "default" / whitespace
    assert resolve_ollama_model() == "qwen3:8b"
    assert resolve_ollama_model(None) == "qwen3:8b"
    assert resolve_ollama_model("") == "qwen3:8b"
    assert resolve_ollama_model("default") == "qwen3:8b"
    assert resolve_ollama_model("   ") == "qwen3:8b"
    assert resolve_ollama_model("  default  ") == "qwen3:8b"

    # 2. Resolves OLLAMA_FAST_MODEL when higher envs are absent
    monkeypatch.setenv("OLLAMA_FAST_MODEL", "qwen2.5-coder:7b")
    assert resolve_ollama_model() == "qwen2.5-coder:7b"
    assert resolve_ollama_model(None) == "qwen2.5-coder:7b"
    assert resolve_ollama_model("") == "qwen2.5-coder:7b"
    assert resolve_ollama_model("default") == "qwen2.5-coder:7b"
    assert resolve_ollama_model("   ") == "qwen2.5-coder:7b"

    # 3. Resolves OLLAMA_INDEPTH_MODEL when OLLAMA_MODEL is absent (overrides FAST_MODEL)
    monkeypatch.setenv("OLLAMA_INDEPTH_MODEL", "  qwen3:8b  ")
    assert resolve_ollama_model() == "qwen3:8b"
    assert resolve_ollama_model(None) == "qwen3:8b"
    assert resolve_ollama_model("") == "qwen3:8b"
    assert resolve_ollama_model("default") == "qwen3:8b"

    # 4. Resolves OLLAMA_MODEL when set (overrides INDEPTH_MODEL and FAST_MODEL)
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5-coder:32b")
    assert resolve_ollama_model() == "qwen2.5-coder:32b"
    assert resolve_ollama_model(None) == "qwen2.5-coder:32b"
    assert resolve_ollama_model("") == "qwen2.5-coder:32b"
    assert resolve_ollama_model("default") == "qwen2.5-coder:32b"

    # 5. Explicit model name returns explicit name (overrides all envs)
    assert resolve_ollama_model("llama3:8b") == "llama3:8b"
    assert resolve_ollama_model("  custom-model:latest  ") == "custom-model:latest"

    # 6. Empty or whitespace env vars are skipped
    monkeypatch.setenv("OLLAMA_MODEL", "   ")
    monkeypatch.setenv("OLLAMA_INDEPTH_MODEL", "")
    assert resolve_ollama_model() == "qwen2.5-coder:7b"


@pytest.mark.asyncio
async def test_ollama_connector_strips_v1_in_request_url() -> None:
    called_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        called_urls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5-coder:7b",
                "message": {"role": "assistant", "content": "URL normalized"},
                "done": True,
            },
        )

    client = _make_mock_client(handler)
    # Instantiate connector with a /v1 suffix
    connector = OllamaConnector(base_url="http://192.0.2.10:11434/v1/", http_client=client)
    assert connector.base_url == "http://192.0.2.10:11434"

    llm_req = LLMRequest(
        model="qwen2.5-coder:7b",
        messages=(ChatMessage(role=MessageRole.USER, content="ping"),),
    )
    resp = await connector.generate(llm_req)
    assert resp.content == "URL normalized"
    assert called_urls == ["http://192.0.2.10:11434/api/chat"]


@pytest.mark.asyncio
async def test_ollama_connector_env_fast_base_url_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    called_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        called_urls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5-coder:7b",
                "message": {"role": "assistant", "content": "fast tier reply"},
                "done": True,
            },
        )

    for var in ("OLLAMA_BASE_URL", "LOCAL_LLM_BASE_URL", "OLLAMA_HOST"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OLLAMA_FAST_BASE_URL", "http://192.0.2.10:11434/v1")

    client = _make_mock_client(handler)
    connector = OllamaConnector(http_client=client)
    assert connector.base_url == "http://192.0.2.10:11434"

    llm_req = LLMRequest(
        model="qwen2.5-coder:7b",
        messages=(ChatMessage(role=MessageRole.USER, content="ping"),),
    )
    resp = await connector.generate(llm_req)
    assert resp.content == "fast tier reply"
    assert called_urls == ["http://192.0.2.10:11434/api/chat"]


@pytest.mark.asyncio
async def test_ollama_connector_generate_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        req_body: dict[str, Any] = json.loads(request.content.decode("utf-8"))
        assert req_body["model"] == "qwen2.5-coder:7b"
        assert req_body["messages"][0]["content"] == "Hello Ollama"
        assert req_body["stream"] is False

        response_data: dict[str, Any] = {
            "model": "qwen2.5-coder:7b",
            "message": {"role": "assistant", "content": "Hello human!"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 20,
            "eval_count": 15,
        }
        return httpx.Response(200, json=response_data)

    client = _make_mock_client(handler)
    connector = OllamaConnector(http_client=client)

    llm_req = LLMRequest(
        model="qwen2.5-coder:7b",
        messages=(ChatMessage(role=MessageRole.USER, content="Hello Ollama"),),
        temperature=0.5,
    )

    resp = await connector.generate(llm_req)
    assert resp.content == "Hello human!"
    assert resp.model_name == "qwen2.5-coder:7b"
    assert resp.finish_reason == FinishReason.STOP
    assert resp.usage.input_tokens == 20
    assert resp.usage.output_tokens == 15
    assert resp.usage.total_tokens == 35
    assert resp.usage.provider == "ollama"
    assert resp.provenance is not None
    assert resp.provenance.path == ExecutionPath.PRIMARY
    assert resp.provenance.requested.provider == "ollama"


@pytest.mark.asyncio
async def test_ollama_connector_generate_tool_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        response_data: dict[str, Any] = {
            "model": "qwen2.5-coder:7b",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "lookup_user",
                            "arguments": {"user_id": 42},
                        }
                    }
                ],
            },
            "done": True,
            "prompt_eval_count": 30,
            "eval_count": 10,
        }
        return httpx.Response(200, json=response_data)

    client = _make_mock_client(handler)
    connector = OllamaConnector(http_client=client)

    tool_def = ToolDefinition(
        name="lookup_user",
        description="Lookup user by ID",
        parameters={"type": "object", "properties": {"user_id": {"type": "integer"}}},
    )
    llm_req = LLMRequest(
        model="qwen2.5-coder:7b",
        messages=(ChatMessage(role=MessageRole.USER, content="Find user 42"),),
        tools=(tool_def,),
    )

    resp = await connector.generate(llm_req)
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "lookup_user"
    assert resp.tool_calls[0].arguments == {"user_id": 42}
    assert resp.finish_reason == FinishReason.TOOL_CALLS


@pytest.mark.asyncio
async def test_ollama_connector_streaming() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        lines = [
            json.dumps({"message": {"content": "Hel"}, "done": False}),
            json.dumps({"message": {"content": "lo"}, "done": False}),
            json.dumps(
                {
                    "message": {"content": "!"},
                    "done": True,
                    "prompt_eval_count": 10,
                    "eval_count": 3,
                }
            ),
        ]
        return httpx.Response(200, text="\n".join(lines) + "\n")

    client = _make_mock_client(handler)
    connector = OllamaConnector(http_client=client)

    chunks: list[StreamChunk] = []
    async for chunk in connector.stream(
        LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="Hi"),))
    ):
        chunks.append(chunk)

    assert len(chunks) == 3
    deltas: list[str] = [c.delta_content for c in chunks if c.delta_content is not None]
    assert "".join(deltas) == "Hello!"
    last_chunk = chunks[-1]
    assert last_chunk.usage is not None
    assert last_chunk.usage.total_tokens == 13


@pytest.mark.asyncio
async def test_ollama_connector_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Ollama Error")

    client = _make_mock_client(handler)
    connector = OllamaConnector(http_client=client)

    with pytest.raises(LLMProviderError, match="Ollama provider returned status 500"):
        await connector.generate(LLMRequest(messages=()))


@pytest.mark.asyncio
async def test_ollama_transport_error_message_names_a_cause_when_exc_is_blank() -> None:
    """A transport failure never raises a message that names nothing (P6, #212).

    `httpx.ConnectError()` carries no arguments, so it stringifies to the empty string
    and `f"Failed to connect to Ollama: {exc}"` used to produce, verbatim:

        LLMProviderError: Failed to connect to Ollama:

    P6 requires a failure to "surface immediately with its precise root cause"; a
    message with nothing after the colon does not. This is how the real defect
    presented -- it is what the traceback said when the unit gate's live inference call
    failed under load, and it is why the red run could not be attributed by reading the
    error alone. The case that presented it was a `ReadTimeout`, which since #1277 leaves
    `generate` as an `LLMTimeoutError` instead; the blank-`str()` hazard is a property of
    `httpx` exceptions generally, so it is asserted here on the branch that remains and
    below on the timeout branch.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("", request=request)

    connector = OllamaConnector(http_client=_make_mock_client(handler))

    with pytest.raises(LLMProviderError) as excinfo:
        await connector.generate(LLMRequest(messages=()))

    message = str(excinfo.value)
    # The class name is the root cause when the exception itself carries no text.
    assert message == "Failed to connect to Ollama: ConnectError"
    assert not message.rstrip().endswith(":")


@pytest.mark.asyncio
async def test_ollama_generate_says_a_ceiling_expired_rather_than_a_daemon_was_absent() -> None:
    """A chat that ran past the caller's ceiling is a different fact from an absent host.

    Both used to leave `generate` as a bare `LLMProviderError` reading `Failed to connect
    to Ollama: ReadTimeout`, so nothing downstream could tell "the daemon answered and we
    gave up waiting" from "there was no daemon". #1277 is the cost: `frontier_live` probes
    truncated at the connector's 60s default reached the report as `UNREACHABLE`, the same
    value an unplugged host produces, and the upper tiers were unmeasurable because of it.
    `pull_model` already drew this line in #1233; the chat path had not.

    The ceiling is carried on the exception rather than only written into the message,
    because the remedy is a number and a caller that has to parse prose to find it will
    not.

    Killed by: src/uclone_x/llm/connectors/ollama.py :: return LLMTimeoutError(_chat_timeout_message(seconds, exc), seconds=seconds)
    Becomes: return LLMProviderError(_chat_timeout_message(seconds, exc))
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("", request=request)

    connector = OllamaConnector(http_client=_make_mock_client(handler), timeout=42.0)

    with pytest.raises(LLMTimeoutError) as excinfo:
        await connector.generate(LLMRequest(messages=()))

    assert excinfo.value.seconds == 42.0
    message = str(excinfo.value)
    # The ceiling that expired is named, and so is the cause, which stringifies blank.
    assert "42s" in message
    assert "ReadTimeout" in message
    # Still an LLMProviderError, so every existing handler keeps catching it.
    assert isinstance(excinfo.value, LLMProviderError)


@pytest.mark.asyncio
async def test_ollama_generate_keeps_a_connect_timeout_on_the_unreachable_side() -> None:
    """Never reaching the host is not the caller's ceiling expiring, though both time out.

    `httpx.ConnectTimeout` is a subclass of `httpx.TimeoutException`, so an `except
    TimeoutException` written without this branch in front of it would relabel every
    unreachable daemon as a run that needed a longer deadline -- exactly inverting the
    distinction #1277 asks for, and sending readers to raise a number that will not help.

    Killed by: src/uclone_x/llm/connectors/ollama.py :: if isinstance(exc, httpx.TimeoutException) and not isinstance(exc, httpx.ConnectTimeout):
    Becomes: if isinstance(exc, httpx.TimeoutException):
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out reaching host", request=request)

    connector = OllamaConnector(http_client=_make_mock_client(handler), timeout=42.0)

    with pytest.raises(LLMProviderError) as excinfo:
        await connector.generate(LLMRequest(messages=()))

    assert not isinstance(excinfo.value, LLMTimeoutError)
    assert str(excinfo.value) == "Failed to connect to Ollama: timed out reaching host"


@pytest.mark.asyncio
async def test_ollama_stream_draws_the_same_line_generate_does() -> None:
    """The streaming path is the one live evals take, so it cannot be the one left out.

    A distinction the non-streaming call makes and the streaming call does not is worse
    than no distinction: a reader who learned to trust `LLMTimeoutError` would read every
    cut-off stream as an unreachable host.

    Killed by: src/uclone_x/llm/connectors/ollama.py :: self.timeout, exc, unreachable_prefix="Ollama stream connection error"
    Becomes: self.timeout, exc, unreachable_prefix="Failed to connect to Ollama"
    """

    def timed_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("", request=request)

    connector = OllamaConnector(http_client=_make_mock_client(timed_out), timeout=7.5)

    with pytest.raises(LLMTimeoutError) as excinfo:
        async for _ in connector.stream(LLMRequest(messages=())):  # pragma: no branch
            pass

    assert excinfo.value.seconds == 7.5
    assert "7.5s" in str(excinfo.value)

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out reaching host", request=request)

    connector = OllamaConnector(http_client=_make_mock_client(unreachable), timeout=7.5)

    with pytest.raises(LLMProviderError) as connect_info:
        async for _ in connector.stream(LLMRequest(messages=())):  # pragma: no branch
            pass

    assert not isinstance(connect_info.value, LLMTimeoutError)
    assert str(connect_info.value) == "Ollama stream connection error: timed out reaching host"


def test_describe_transport_error_prefers_the_exception_text_when_there_is_some() -> None:
    """The type-name fallback applies only to a blank exception, never over real text."""
    assert describe_transport_error(httpx.ConnectError("All connection attempts failed")) == (
        "All connection attempts failed"
    )
    assert describe_transport_error(httpx.ReadTimeout("")) == "ReadTimeout"
    # Whitespace-only is as uninformative as empty, and is treated the same.
    assert describe_transport_error(httpx.ReadTimeout("   ")) == "ReadTimeout"


def test_resolve_ollama_timeout_hierarchy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ollama timeout resolves via explicit argument, OLLAMA_TIMEOUT, LLM_TIMEOUT, or 180s default."""
    from uclone_x.llm.connectors.ollama import (
        DEFAULT_OLLAMA_TIMEOUT_SECONDS,
        resolve_ollama_timeout,
    )

    monkeypatch.delenv("OLLAMA_TIMEOUT", raising=False)
    monkeypatch.delenv("LLM_TIMEOUT", raising=False)

    # Default is 180.0s
    assert resolve_ollama_timeout() == DEFAULT_OLLAMA_TIMEOUT_SECONDS
    assert DEFAULT_OLLAMA_TIMEOUT_SECONDS == 180.0

    # Explicit argument wins
    assert resolve_ollama_timeout(45.0) == 45.0

    # OLLAMA_TIMEOUT env var
    monkeypatch.setenv("OLLAMA_TIMEOUT", "240.0")
    assert resolve_ollama_timeout() == 240.0

    # LLM_TIMEOUT env var fallback
    monkeypatch.delenv("OLLAMA_TIMEOUT")
    monkeypatch.setenv("LLM_TIMEOUT", "150.0")
    assert resolve_ollama_timeout() == 150.0


def test_ollama_connector_uses_resolved_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """OllamaConnector defaults to 180.0s timeout and respects OLLAMA_TIMEOUT."""
    from uclone_x.llm.connectors.ollama import DEFAULT_OLLAMA_TIMEOUT_SECONDS

    monkeypatch.delenv("OLLAMA_TIMEOUT", raising=False)
    monkeypatch.delenv("LLM_TIMEOUT", raising=False)

    conn_default = OllamaConnector()
    assert conn_default.timeout == DEFAULT_OLLAMA_TIMEOUT_SECONDS
    assert conn_default.timeout == 180.0

    monkeypatch.setenv("OLLAMA_TIMEOUT", "300")
    conn_env = OllamaConnector()
    assert conn_env.timeout == 300.0

    conn_explicit = OllamaConnector(timeout=45.0)
    assert conn_explicit.timeout == 45.0


# ======================================================================================
# pull_model / delete_model Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_pull_model_success() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "success"})

    client = _make_mock_client(handler)
    await pull_model("llama3.2:1b", http_client=client)

    assert seen["url"] == "http://localhost:11434/api/pull"
    assert seen["body"] == {"model": "llama3.2:1b", "stream": True}


@pytest.mark.asyncio
async def test_pull_model_raises_on_non_2xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Ollama Error")

    client = _make_mock_client(handler)

    with pytest.raises(LLMProviderError, match="Ollama provider returned status 500"):
        await pull_model("llama3.2:1b", http_client=client)


@pytest.mark.asyncio
async def test_pull_model_raises_when_a_200_reports_a_failed_pull() -> None:
    """Ollama puts the outcome in the body it streams, not in the HTTP status line.

    A pull that dies part-way still answers 200, carrying `{"error": ...}` on one of
    its progress lines or ending on a `status` that is not `"success"`. Reading only
    `resp.status_code` made the Settings modal say `Installed model "X".` about a
    model the daemon does not have — the one failure mode a local-first install path
    must not have (P6).

    Killed by: src/uclone_x/llm/connectors/ollama.py :: status = _raise_for_pull_line(model, line)
    Becomes: status = None
    """
    for body, expected in (
        ({"error": "pull model manifest: file does not exist"}, "file does not exist"),
        ({"status": "pulling manifest"}, "pulling manifest"),
    ):

        def handler(request: httpx.Request, body: dict[str, Any] = body) -> httpx.Response:
            return httpx.Response(200, json=body)

        client = _make_mock_client(handler)

        with pytest.raises(LLMProviderError) as excinfo:
            await pull_model("nope:latest", http_client=client)

        message = str(excinfo.value)
        assert expected in message
        assert "nope:latest" in message


@pytest.mark.asyncio
async def test_pull_model_lets_an_unreadable_200_body_through() -> None:
    """A body this function cannot parse is not evidence of failure, and is not reported as one.

    Ollama's `/api/pull` has answered 200 with an empty body in the past; turning
    that into an error would invent a failure the daemon never reported.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="")

    client = _make_mock_client(handler)
    await pull_model("llama3.2:1b", http_client=client)


@pytest.mark.asyncio
async def test_pull_model_does_not_give_connecting_the_downloading_budget() -> None:
    """`httpx` spends a bare float on connect, read, write and pool alike (#1233).

    So the budget the pull buys for *waiting on the stream* was also what a caller
    spent discovering the daemon is not there. On loopback that never showed —
    a closed port refuses at once — but `resolve_ollama_base_url` also resolves
    `OLLAMA_HOST` pointing at the README's second tier, and a machine that is
    asleep or firewalled drops the SYN instead of refusing it. The wait to *reach*
    the daemon is therefore pinned separately and short; asserting it off
    `request.extensions["timeout"]` reads the value the transport was actually
    handed rather than the one the call site meant to pass.

    Killed by: src/uclone_x/llm/connectors/ollama.py :: return httpx.Timeout(timeout, connect=MODEL_MANAGEMENT_CONNECT_TIMEOUT_SECONDS)
    Becomes: return httpx.Timeout(timeout)
    """
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["timeout"] = request.extensions["timeout"]
        return httpx.Response(200, json={"status": "success"})

    client = _make_mock_client(handler)
    await pull_model("llama3.2:1b", silence_timeout=900.0, http_client=client)

    assert seen["timeout"]["read"] == 900.0
    assert seen["timeout"]["connect"] == MODEL_MANAGEMENT_CONNECT_TIMEOUT_SECONDS
    assert MODEL_MANAGEMENT_CONNECT_TIMEOUT_SECONDS < 900.0


@pytest.mark.asyncio
async def test_pull_model_reports_a_socket_level_silence_as_a_stall_not_a_refusal() -> None:
    """A stream that went quiet and a daemon that was never there need different remedies.

    Both used to arrive as `Failed to connect to Ollama: ...`, so a pull whose
    ceiling expired read as a daemon that is not running — and the surface told the
    user to start Ollama, which was already running and still downloading. P6: the
    report names what happened.

    `httpx`'s read timeout measures the same quantity `pull_model`'s own loop does,
    one level down — time between *bytes* rather than between NDJSON lines — so it
    is the same fact and must produce the same report, not a second vocabulary for
    it.

    Killed by: src/uclone_x/llm/connectors/ollama.py :: except httpx.TimeoutException as exc:
    Becomes: except ValueError as exc:
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = _make_mock_client(handler)

    with pytest.raises(LLMTimeoutError) as excinfo:
        await pull_model("qwen3:8b", silence_timeout=12.0, http_client=client)

    assert excinfo.value.seconds == 12.0
    message = str(excinfo.value)
    assert "qwen3:8b" in message
    assert "nothing for 12s" in message
    assert "Failed to connect" not in message


@pytest.mark.asyncio
async def test_pull_model_reports_an_unreachable_host_as_a_refusal_not_a_deadline() -> None:
    """The connect ceiling expiring is not the pull ceiling expiring.

    `httpx.ConnectTimeout` is an `httpx.TimeoutException`, so the clause added for
    the case above would otherwise swallow it and report a 5-second failure to
    reach the daemon as a stream that went quiet for two minutes — a silence that
    never elapsed, about a download that never started.

    Killed by: src/uclone_x/llm/connectors/ollama.py :: except httpx.ConnectTimeout as exc:
    Becomes: except ValueError as exc:
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    client = _make_mock_client(handler)

    with pytest.raises(LLMProviderError) as excinfo:
        await pull_model("qwen3:8b", http_client=client)

    assert not isinstance(excinfo.value, LLMTimeoutError)
    assert "Failed to connect to Ollama" in str(excinfo.value)


@pytest.mark.asyncio
async def test_pull_model_raises_on_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("All connection attempts failed", request=request)

    client = _make_mock_client(handler)

    with pytest.raises(LLMProviderError) as excinfo:
        await pull_model("llama3.2:1b", http_client=client)

    message = str(excinfo.value)
    assert "All connection attempts failed" in message
    assert not message.rstrip().endswith(":")


class _TimedNdjsonStream(httpx.AsyncByteStream):
    """An `/api/pull` response body whose *timing* the test owns (#1243).

    The two facts this change is about are both about time, and neither can be
    asserted from a constant: that a pull which keeps reporting is never killed no
    matter how long it runs, and that a pull which stops reporting dies quickly. A
    test that read `PULL_SILENCE_TIMEOUT_SECONDS` and compared it to something would
    assert that a number is the number it is.

    So the timings are real and the scale is not: the windows passed to `pull_model`
    are hundredths of a second, and the gaps here are smaller still. Nothing sleeps
    for minutes and nothing is patched — `asyncio.sleep` between chunks is what the
    silence window is actually measured against, so the code under test runs its
    real loop against a real clock.
    """

    def __init__(self, chunks: Sequence[tuple[float, bytes]], *, then_silent: bool = False) -> None:
        self._chunks = tuple(chunks)
        self._then_silent = then_silent

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for gap, payload in self._chunks:
            await asyncio.sleep(gap)
            yield payload
        if self._then_silent:
            # Longer than any window this file passes, so what ends the pull is
            # `pull_model` giving up and never this stream running out.
            await asyncio.sleep(3600)


def _ndjson_client(stream: _TimedNdjsonStream) -> httpx.AsyncClient:
    """An `AsyncClient` whose `/api/pull` answers 200 with `stream`."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _progress(status: str, completed: int | None = None) -> bytes:
    line: dict[str, Any] = {"status": status}
    if completed is not None:
        line["completed"] = completed
        line["total"] = 45_961_741
    return (json.dumps(line) + "\n").encode()


@pytest.mark.asyncio
async def test_a_pull_that_keeps_reporting_is_not_killed_by_how_long_it_has_run() -> None:
    """The quantity that used to end a pull was elapsed time, and it ended working ones.

    `PULL_ROUTE_TIMEOUT_SECONDS = 900.0` was a ceiling on the whole pull. A 70B model
    on a slow link runs past it while working perfectly, and there is no value that
    fixes this — the number has to cover the slowest legitimate transfer, which is
    unbounded. Here the stream runs for many multiples of the window it is given and
    finishes, because every line resets what it is measured against.

    The assertion is on *elapsed against the window*, not on a wall-clock constant:
    `elapsed > silence * 4` is the same claim as "this outlived a 900 s ceiling",
    stated at a scale a test suite can afford.

    Killed by: src/uclone_x/llm/connectors/ollama.py :: backstop_at = loop.time() + total_timeout
    Becomes: backstop_at = loop.time() + silence_timeout
    """
    silence = 0.05
    chunks = [(0.0, _progress("pulling manifest"))]
    chunks += [
        (0.02, _progress("pulling 797b70c4edf8", completed=n * 2_000_000)) for n in range(20)
    ]
    chunks.append((0.02, _progress("success")))

    client = _ndjson_client(_TimedNdjsonStream(chunks))
    started = time.monotonic()
    await pull_model("qwen3:8b", silence_timeout=silence, total_timeout=10.0, http_client=client)
    elapsed = time.monotonic() - started

    assert elapsed > silence * 4, (
        f"the fake pull finished in {elapsed:.3f}s, which is not long enough past the "
        f"{silence}s window to show that elapsed time did not end it"
    )


@pytest.mark.asyncio
async def test_a_pull_that_goes_quiet_dies_on_the_silence_window_not_the_backstop() -> None:
    """A wedged pull used to hold everything it holds for the full 900 s.

    It holds a socket, an `httpx.AsyncClient` and — since #1233 — that model's
    single-flight entry, so for as long as it lives nobody can retry *that* model.
    Silence is what distinguishes it from a slow pull, and it is measured here
    against a backstop 40x larger so that the elapsed time says which of the two
    ended it: dying on the backstop and dying on the window are indistinguishable
    when the two are close together, which is how a test like this passes vacuously.

    Killed by: src/uclone_x/llm/connectors/ollama.py :: line = await asyncio.wait_for(anext(lines), min(silence_timeout, remaining))
    Becomes: line = await asyncio.wait_for(anext(lines), remaining)
    """
    silence, backstop = 0.05, 2.0
    stream = _TimedNdjsonStream(
        [(0.0, _progress("pulling manifest")), (0.01, _progress("pulling 797b70c4edf8"))],
        then_silent=True,
    )

    client = _ndjson_client(stream)
    started = time.monotonic()
    with pytest.raises(LLMTimeoutError) as excinfo:
        await pull_model(
            "qwen3:8b", silence_timeout=silence, total_timeout=backstop, http_client=client
        )
    elapsed = time.monotonic() - started

    assert elapsed < backstop / 4, (
        f"gave up after {elapsed:.3f}s with a {silence}s silence window and a "
        f"{backstop}s backstop — that is the backstop expiring, not the silence"
    )
    assert excinfo.value.seconds == silence

    message = str(excinfo.value)
    assert "pulling 797b70c4edf8" in message, "the report does not say where the stream stopped"
    assert "stalled pull, not a slow one" in message
    assert "nothing for 0.05s" in message


@pytest.mark.asyncio
async def test_a_stream_that_never_goes_quiet_is_ended_by_the_backstop_which_says_so() -> None:
    """Ollama's progress ticker is driven by time, not by bytes — so silence has a blind spot.

    Measured against a local daemon on Ollama 0.31.2 (2026-09-20): the stream emits a
    line about every 60 ms, and it kept emitting through six lines before a single
    byte had transferred. A download whose bytes stop while the daemon stays healthy
    therefore keeps talking, with `completed` frozen, and a pure silence ceiling
    would never end it — which the 900 s wall-clock ceiling did. `total_timeout`
    exists for exactly that case and for no other.

    The backstop here is set absurdly low (0.2 s against a 1 s window) because the
    real one is two hours. What is pinned is that it exists, that it fires, and that
    it reports itself as a backstop rather than borrowing the stall's wording — an
    operator told "Ollama stopped sending progress" about a daemon that never stopped
    sending progress would go and look in the wrong place.

    Killed by: src/uclone_x/llm/connectors/ollama.py :: remaining = backstop_at - loop.time()
    Becomes: remaining = float("inf")
    """
    frozen = [(0.01, _progress("pulling 797b70c4edf8", completed=1_000_000)) for _ in range(200)]
    client = _ndjson_client(_TimedNdjsonStream(frozen))

    started = time.monotonic()
    with pytest.raises(LLMTimeoutError) as excinfo:
        await pull_model("qwen3:8b", silence_timeout=1.0, total_timeout=0.2, http_client=client)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"{elapsed:.3f}s is long enough that the silence window may have ended it"
    assert excinfo.value.seconds == 0.2

    message = str(excinfo.value)
    assert "still streaming progress" in message
    assert "stopped sending" not in message


@pytest.mark.asyncio
async def test_a_stream_that_ends_before_success_is_not_reported_as_an_install() -> None:
    """A streamed pull can end without failing and without succeeding, and that is new.

    With `{"stream": false}` there was one body and one verdict in it. A stream can
    simply stop — the daemon exits, the connection drops — leaving the last word as
    `pulling <digest>`, which is neither an error nor a success. Reported as `ok` it
    is the same P6 hole #1183 fixed at the body level: `Installed model "X".` for a
    model that is not there.

    Killed by: src/uclone_x/llm/connectors/ollama.py :: _raise_for_pull_outcome(model, last_status)
    Becomes: pass
    """
    stream = _TimedNdjsonStream(
        [
            (0.0, _progress("pulling manifest")),
            (0.0, _progress("pulling 797b70c4edf8", completed=12)),
        ]
    )
    client = _ndjson_client(stream)

    with pytest.raises(LLMProviderError) as excinfo:
        await pull_model("qwen3:8b", silence_timeout=1.0, total_timeout=5.0, http_client=client)

    assert not isinstance(excinfo.value, LLMTimeoutError), (
        "the stream ended, it did not go quiet — a deadline is the wrong fact to report"
    )
    message = str(excinfo.value)
    assert "qwen3:8b" in message
    assert "last status was 'pulling 797b70c4edf8'" in message


def test_the_shipped_windows_are_the_ones_the_measurement_argued_for() -> None:
    """The two defaults are a decision, and a decision with no test is a number that drifts.

    Neither is asserted as "correct" — that is not a thing a unit test can know. What
    is asserted is the *relationship* the design rests on: the silence window is
    orders of magnitude above the ~60 ms tick and the 0.744 s cold-manifest gap
    measured on Ollama 0.31.2, and the backstop is far enough above the silence
    window that it can never be what ends a stalled pull.
    """
    assert PULL_SILENCE_TIMEOUT_SECONDS > 0.744 * 100
    assert PULL_TOTAL_BACKSTOP_SECONDS > PULL_SILENCE_TIMEOUT_SECONDS * 10


@pytest.mark.asyncio
async def test_delete_model_success() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={})

    client = _make_mock_client(handler)
    await delete_model("llama3.2:1b", http_client=client)

    assert seen["method"] == "DELETE"
    assert seen["url"] == "http://localhost:11434/api/delete"
    assert seen["body"] == {"model": "llama3.2:1b"}


@pytest.mark.asyncio
async def test_delete_model_raises_on_non_2xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="model not found")

    client = _make_mock_client(handler)

    with pytest.raises(LLMProviderError, match="Ollama provider returned status 404"):
        await delete_model("llama3.2:1b", http_client=client)


@pytest.mark.asyncio
async def test_delete_model_raises_on_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("", request=request)

    client = _make_mock_client(handler)

    with pytest.raises(LLMProviderError) as excinfo:
        await delete_model("llama3.2:1b", http_client=client)

    message = str(excinfo.value)
    assert message == "Failed to connect to Ollama: ReadTimeout"
    assert not message.rstrip().endswith(":")


# ======================================================================================
# OpenAIConnector Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_openai_connector_generate() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test_key"
        body: dict[str, Any] = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "gpt-4o"
        assert body["temperature"] == 0.7

        response_data: dict[str, Any] = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "OpenAI answer",
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "calculate",
                                    "arguments": '{"x": 10, "y": 20}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 25,
                "total_tokens": 75,
            },
        }
        return httpx.Response(200, json=response_data)

    client = _make_mock_client(handler)
    connector = OpenAIConnector(api_key="test_key", http_client=client)

    resp = await connector.generate(
        LLMRequest(
            model="gpt-4o",
            messages=(
                ChatMessage(role=MessageRole.SYSTEM, content="System instruction"),
                ChatMessage(role=MessageRole.USER, content="Compute sum"),
            ),
        )
    )

    assert resp.content == "OpenAI answer"
    assert resp.finish_reason == FinishReason.TOOL_CALLS
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].id == "call_abc"
    assert resp.tool_calls[0].name == "calculate"
    assert resp.tool_calls[0].arguments == {"x": 10, "y": 20}
    assert resp.usage.provider == "openai"
    assert resp.usage.total_tokens == 75
    assert resp.provenance is not None
    assert resp.provenance.requested.provider == "openai"


@pytest.mark.asyncio
async def test_openai_connector_streaming() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        sse_lines = [
            "data: "
            + json.dumps({"choices": [{"delta": {"content": "Part 1"}, "finish_reason": None}]}),
            "data: "
            + json.dumps(
                {
                    "choices": [{"delta": {"content": " Part 2"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            ),
            "data: [DONE]",
        ]
        return httpx.Response(200, text="\n\n".join(sse_lines) + "\n\n")

    client = _make_mock_client(handler)
    connector = OpenAIConnector(api_key="key", http_client=client)

    chunks: list[StreamChunk] = []
    async for chunk in connector.stream(
        LLMRequest(model="gpt-4o", messages=(ChatMessage(role=MessageRole.USER, content="Hello"),))
    ):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].delta_content == "Part 1"
    assert chunks[1].delta_content == " Part 2"
    assert chunks[1].finish_reason == FinishReason.STOP
    assert chunks[1].usage is not None
    assert chunks[1].usage.total_tokens == 15


@pytest.mark.asyncio
async def test_openai_connector_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="Unauthorized API key")

    client = _make_mock_client(handler)
    connector = OpenAIConnector(api_key="bad_key", http_client=client)

    with pytest.raises(LLMProviderError, match="OpenAI error 401"):
        await connector.generate(LLMRequest(messages=()))


# ======================================================================================
# AnthropicConnector Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_anthropic_connector_generate() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "ant_key"
        body: dict[str, Any] = json.loads(request.content.decode("utf-8"))
        assert body["system"] == "Be helpful"
        assert body["model"] == "claude-3-5-sonnet"

        response_data: dict[str, Any] = {
            "id": "msg_123",
            "model": "claude-3-5-sonnet",
            "content": [
                {"type": "text", "text": "I can help with that."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "read_file",
                    "input": {"path": "/tmp/test.txt"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 40,
                "output_tokens": 30,
            },
        }
        return httpx.Response(200, json=response_data)

    client = _make_mock_client(handler)
    connector = AnthropicConnector(api_key="ant_key", http_client=client)

    llm_req = LLMRequest(
        model="claude-3-5-sonnet",
        messages=(
            ChatMessage(role=MessageRole.SYSTEM, content="Be helpful"),
            ChatMessage(role=MessageRole.USER, content="Read /tmp/test.txt"),
        ),
        tools=(
            ToolDefinition(
                name="read_file",
                description="Read a file",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            ),
        ),
    )

    resp = await connector.generate(llm_req)
    assert resp.content == "I can help with that."
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "read_file"
    assert resp.tool_calls[0].arguments == {"path": "/tmp/test.txt"}
    assert resp.finish_reason == FinishReason.TOOL_CALLS
    assert resp.usage.provider == "anthropic"
    assert resp.usage.total_tokens == 70
    assert resp.provenance is not None


@pytest.mark.asyncio
async def test_anthropic_connector_streaming() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        sse_lines = [
            "data: "
            + json.dumps({"type": "message_start", "message": {"usage": {"input_tokens": 20}}}),
            "data: "
            + json.dumps(
                {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello "}}
            ),
            "data: "
            + json.dumps(
                {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Claude"}}
            ),
            "data: "
            + json.dumps(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 10},
                }
            ),
        ]
        return httpx.Response(200, text="\n\n".join(sse_lines) + "\n\n")

    client = _make_mock_client(handler)
    connector = AnthropicConnector(api_key="key", http_client=client)

    chunks: list[StreamChunk] = []
    async for chunk in connector.stream(
        LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="Hi"),))
    ):
        chunks.append(chunk)

    assert len(chunks) >= 3
    deltas: list[str] = [c.delta_content for c in chunks if c.delta_content is not None]
    assert "".join(deltas) == "Hello Claude"
    last_chunk = chunks[-1]
    assert last_chunk.usage is not None
    assert last_chunk.usage.total_tokens == 30
    assert last_chunk.finish_reason == FinishReason.STOP


@pytest.mark.asyncio
async def test_anthropic_connector_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="Rate limit exceeded")

    client = _make_mock_client(handler)
    connector = AnthropicConnector(api_key="key", http_client=client)

    with pytest.raises(LLMProviderError, match="Anthropic error 429"):
        await connector.generate(LLMRequest(messages=()))


# ======================================================================================
# GeminiConnector Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_gemini_connector_generate() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-goog-api-key"] == "gem_key"
        body: dict[str, Any] = json.loads(request.content.decode("utf-8"))
        assert "systemInstruction" in body

        response_data: dict[str, Any] = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Gemini response text."},
                            {
                                "functionCall": {
                                    "name": "search_db",
                                    "args": {"query": "test"},
                                }
                            },
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 60,
                "candidatesTokenCount": 30,
                "totalTokenCount": 90,
            },
            "modelVersion": "gemini-1.5-pro",
        }
        return httpx.Response(200, json=response_data)

    client = _make_mock_client(handler)
    connector = GeminiConnector(api_key="gem_key", http_client=client)

    resp = await connector.generate(
        LLMRequest(
            model="gemini-1.5-pro",
            messages=(
                ChatMessage(role=MessageRole.SYSTEM, content="System prompt"),
                ChatMessage(role=MessageRole.USER, content="Search test"),
            ),
        )
    )

    assert resp.content == "Gemini response text."
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "search_db"
    assert resp.tool_calls[0].arguments == {"query": "test"}
    assert resp.usage.provider == "gemini"
    assert resp.usage.total_tokens == 90
    assert resp.provenance is not None
    assert resp.provenance.served_by.provider == "gemini"


@pytest.mark.asyncio
async def test_gemini_connector_keeps_a_resolved_model_alias_visible() -> None:
    """A provider-side alias reaches the caller as `degraded` (issue #149).

    Gemini resolves `gemini-1.5-pro` to a dated build and reports it back in
    `modelVersion`. The connector used to read that field and pass it to
    `Provenance.primary(provider, model=...)`, which sets `requested` and `served_by` to
    the *same* value — the served one — so `degraded` computed `False` and the caller
    could not tell it was talking to a different model than it named. P6: "a model swap
    changes tool-calling behaviour, output formatting, and determinism."
    """

    def handler(request: httpx.Request) -> httpx.Response:
        # The alias is not in the URL: the caller asked for the floating name.
        assert "gemini-1.5-pro:generateContent" in str(request.url)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "aliased"}], "role": "model"},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 10,
                },
                "modelVersion": "gemini-1.5-pro-002",
            },
        )

    connector = GeminiConnector(api_key="gem_key", http_client=_make_mock_client(handler))
    resp = await connector.generate(
        LLMRequest(
            model="gemini-1.5-pro",
            messages=(ChatMessage(role=MessageRole.USER, content="hi"),),
        )
    )

    assert resp.provenance is not None
    assert resp.provenance.requested == ServiceRef(provider="gemini", model="gemini-1.5-pro")
    assert resp.provenance.served_by == ServiceRef(provider="gemini", model="gemini-1.5-pro-002")
    assert resp.provenance.degraded is True
    # Still the primary path: one call, nothing failed, nothing re-attempted — so P6's
    # check 4 does not apply and there is no failover event to announce.
    assert resp.provenance.path is ExecutionPath.PRIMARY
    assert resp.provenance.attempts == ()


def _openai_alias_handler(served: str) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "model": served,
            },
        )

    return handler


def _anthropic_alias_handler(served: str) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "stop_reason": "end_turn",
                "model": served,
            },
        )

    return handler


def _ollama_alias_handler(served: str) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {"content": "ok"},
                "prompt_eval_count": 1,
                "eval_count": 1,
                "done_reason": "stop",
                "model": served,
            },
        )

    return handler


@pytest.mark.asyncio
async def test_requested_provenance_names_the_model_actually_sent() -> None:
    """`provenance.requested` is the model on the wire, even when the caller named none.

    The default was written out once per call site — the request builder, the streaming
    path, and the `requested` reported in provenance — from the same literal. Folding
    them onto one `_requested_model` expression makes divergence unrepresentable rather
    than merely untested, but the invariant is what matters, so it is asserted directly:
    whatever went out in the payload is what provenance claims was requested. A future
    edit that reintroduces a second copy and lets it drift reports a `requested` model
    that was never sent, and `degraded` flips in the field #149 exists to make
    trustworthy.
    """
    sent: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "model": sent["model"],
            },
        )

    connector = OpenAIConnector(api_key="k", http_client=_make_mock_client(handler))
    # No model named: the connector's own default is what reaches the provider.
    resp = await connector.generate(
        LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),))
    )

    assert resp.provenance is not None
    assert resp.provenance.requested.model == sent["model"]
    assert resp.provenance.degraded is False


_LOCAL_MODEL = "qwen2.5-coder-32b-instruct"
"""A model name OpenAI does not serve, served by a local OpenAI-compatible endpoint."""


class _OpenAICompatibleConnector(OpenAIConnector):
    """An OpenAI-compatible endpoint that is not OpenAI.

    A local vLLM or LM Studio server speaks `/v1/chat/completions`, so the wire format
    belongs to `OpenAIConnector` and the connector for such a server is a subclass of it.
    Its *identity* is not OpenAI's, and `provider_name` is the one thing it should have to
    override to say so — provided the rest of the connector reads that property instead of
    restating the literal. It did not: `TokenUsage.provider` and `Provenance.primary`
    each carried `"openai"` written out, in five places, so a
    subclass could not report itself at all (#1304; #958's third item, from the other
    direction).
    """

    @property
    def provider_name(self) -> str:
        return "vllm"


def _local_endpoint_handler(request: httpx.Request) -> httpx.Response:
    """Answer as a local OpenAI-compatible server does: a model of its own, with usage."""
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 1000},
            "model": _LOCAL_MODEL,
        },
    )


@pytest.mark.asyncio
async def test_an_openai_compatible_connector_is_attributed_to_itself() -> None:
    """A subclass's turn is attributed to the subclass, not to OpenAI.

    `provenance` is P6's in-band attribution, and the question it answers is *which
    service produced this*. A connector pointed at `http://localhost:8000/v1` that stamps
    `openai` does not merely omit an answer — it supplies a wrong one, and every ledger,
    budget row and UI label downstream repeats it. The literal is unreachable by any
    override, which is what makes this the connector's defect rather than the operator's.

    Killed by: src/uclone_x/llm/connectors/openai.py :: provider=self.provider_name, model=requested_model, served_model=model_name
    Becomes: provider="openai", model=requested_model, served_model=model_name
    """
    connector = _OpenAICompatibleConnector(
        api_key="k", http_client=_make_mock_client(_local_endpoint_handler)
    )

    resp = await connector.generate(
        LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),))
    )

    assert resp.provenance is not None
    assert resp.provenance.requested.provider == "vllm"
    assert resp.provenance.served_by.provider == "vllm"
    assert resp.usage.provider == "vllm"


@pytest.mark.asyncio
async def test_a_streamed_openai_compatible_turn_is_attributed_to_itself() -> None:
    """The streaming path carries the same attribution as `generate`.

    It is a separate path with its own copy of the usage construction, and #397 records
    what happens when the two disagree: the stream is the one that reverts to reporting
    nothing. Here the disagreement would be a stream booked to OpenAI while `generate`
    is not, on the same connector.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            'data: {"model":"' + _LOCAL_MODEL + '",'
            '"usage":{"prompt_tokens":1000,"completion_tokens":1000}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=body)

    connector = _OpenAICompatibleConnector(api_key="k", http_client=_make_mock_client(handler))

    usages: list[TokenUsage] = []
    async for chunk in connector.stream(
        LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),))
    ):
        if chunk.usage is not None:
            usages.append(chunk.usage)

    assert len(usages) == 1
    assert usages[0].provider == "vllm"


def _openai_connector(client: httpx.AsyncClient) -> BaseLLMConnector:
    return OpenAIConnector(api_key="k", http_client=client)


def _anthropic_connector(client: httpx.AsyncClient) -> BaseLLMConnector:
    return AnthropicConnector(api_key="k", http_client=client)


def _ollama_connector(client: httpx.AsyncClient) -> BaseLLMConnector:
    return OllamaConnector(http_client=client)


_ALIAS_CASES: list[
    tuple[
        str,
        Callable[[httpx.AsyncClient], BaseLLMConnector],
        Callable[[str], Callable[[httpx.Request], httpx.Response]],
        str,
        str,
    ]
] = [
    ("openai", _openai_connector, _openai_alias_handler, "gpt-4o", "gpt-4o-2024-08-06"),
    (
        "anthropic",
        _anthropic_connector,
        _anthropic_alias_handler,
        "claude-3-5-sonnet",
        "claude-3-5-sonnet-20241022",
    ),
    (
        "ollama",
        _ollama_connector,
        _ollama_alias_handler,
        "qwen2.5-coder:14b",
        "qwen2.5-coder:14b-instruct-q4_K_M",
    ),
]


@pytest.mark.parametrize(
    ("provider", "make_connector", "make_handler", "requested", "served"), _ALIAS_CASES
)
@pytest.mark.asyncio
async def test_connectors_keep_a_resolved_model_alias_visible(
    provider: str,
    make_connector: Callable[[httpx.AsyncClient], BaseLLMConnector],
    make_handler: Callable[[str], Callable[[httpx.Request], httpx.Response]],
    requested: str,
    served: str,
) -> None:
    """Every connector reports a provider-side alias as `degraded` (issue #149).

    The sibling `test_connectors_report_an_unaliased_model_as_undegraded` cannot catch
    this defect and must not be mistaken for coverage of it: when the response model
    equals the requested one, the repaired `model=requested_model, served_model=served`
    and the defective `model=served` build **byte-identical** provenance. Only a case
    where the two differ can tell fixed from broken, and until this test existed the
    only such case was Gemini's — so three of the four connectors could be reverted to
    the #149 defect with a fully green gate.

    Mutation-checked: reverting any one of the three connectors to
    `Provenance.primary(provider, model=<served name>)` fails its parameter here.
    """
    client = _make_mock_client(make_handler(served))
    connector = make_connector(client)

    resp = await connector.generate(
        LLMRequest(
            model=requested,
            messages=(ChatMessage(role=MessageRole.USER, content="hi"),),
        )
    )

    assert resp.provenance is not None
    assert resp.provenance.requested == ServiceRef(provider=provider, model=requested)
    assert resp.provenance.served_by == ServiceRef(provider=provider, model=served)
    # The point of the repair: the caller can see it is not talking to the model it
    # named. The defective form collapses both refs onto `served` and reports False.
    assert resp.provenance.degraded is True
    # Still primary: one call, nothing failed, nothing re-attempted, so P6's check 4
    # does not apply and there is no failover to announce.
    assert resp.provenance.path is ExecutionPath.PRIMARY
    assert resp.provenance.attempts == ()


@pytest.mark.asyncio
async def test_connectors_report_an_unaliased_model_as_undegraded() -> None:
    """The ordinary case stays `degraded=False` across all four connectors (#149).

    The alias repair must not turn every response into a degraded one; `requested` and
    `served_by` coincide whenever the provider serves the model it was asked for.
    """

    def openai_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "model": "gpt-4o",
            },
        )

    def anthropic_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "stop_reason": "end_turn",
                "model": "claude-3-5-sonnet",
            },
        )

    def ollama_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {"content": "ok"},
                "prompt_eval_count": 1,
                "eval_count": 1,
                "done_reason": "stop",
                "model": "qwen2.5-coder:14b",
            },
        )

    cases = (
        (
            OpenAIConnector(api_key="k", http_client=_make_mock_client(openai_handler)),
            "gpt-4o",
        ),
        (
            AnthropicConnector(api_key="k", http_client=_make_mock_client(anthropic_handler)),
            "claude-3-5-sonnet",
        ),
        (
            OllamaConnector(http_client=_make_mock_client(ollama_handler)),
            "qwen2.5-coder:14b",
        ),
    )

    for connector, model in cases:
        resp = await connector.generate(
            LLMRequest(model=model, messages=(ChatMessage(role=MessageRole.USER, content="hi"),))
        )
        assert resp.provenance is not None, model
        assert resp.provenance.requested.model == model
        assert resp.provenance.served_by.model == model
        assert resp.provenance.degraded is False, model


@pytest.mark.asyncio
async def test_gemini_connector_streaming() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        sse_lines = [
            "data: "
            + json.dumps(
                {
                    "candidates": [{"content": {"parts": [{"text": "Chunk 1"}]}}],
                }
            ),
            "data: "
            + json.dumps(
                {
                    "candidates": [
                        {"content": {"parts": [{"text": " Chunk 2"}]}, "finishReason": "STOP"}
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 15,
                        "candidatesTokenCount": 8,
                        "totalTokenCount": 23,
                    },
                }
            ),
        ]
        return httpx.Response(200, text="\n\n".join(sse_lines) + "\n\n")

    client = _make_mock_client(handler)
    connector = GeminiConnector(api_key="key", http_client=client)

    chunks: list[StreamChunk] = []
    async for chunk in connector.stream(
        LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="Hi"),))
    ):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].delta_content == "Chunk 1"
    assert chunks[1].delta_content == " Chunk 2"
    assert chunks[1].finish_reason == FinishReason.STOP
    assert chunks[1].usage is not None
    assert chunks[1].usage.total_tokens == 23


@pytest.mark.asyncio
async def test_gemini_connector_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="Bad Request")

    client = _make_mock_client(handler)
    connector = GeminiConnector(api_key="key", http_client=client)

    with pytest.raises(LLMProviderError, match="Gemini error 400"):
        await connector.generate(LLMRequest(messages=()))


def test_parse_dict_payload_malformed_raises() -> None:
    # Truncated JSON string must raise MalformedToolCallArgumentsError, not return {}
    with pytest.raises(
        MalformedToolCallArgumentsError, match="Failed to parse tool call arguments"
    ):
        parse_dict_payload('{"path": "/etc/passwd", "recursi')

    # Non-mapping JSON values must raise MalformedToolCallArgumentsError
    with pytest.raises(MalformedToolCallArgumentsError, match="Expected JSON object"):
        parse_dict_payload('["item1", "item2"]')

    with pytest.raises(MalformedToolCallArgumentsError, match="Expected JSON object"):
        parse_dict_payload("12345")

    # Valid empty or valid mapping cases
    assert parse_dict_payload(None) == {}
    assert parse_dict_payload("") == {}
    assert parse_dict_payload("{}") == {}
    assert parse_dict_payload({"key": "val"}) == {"key": "val"}


@pytest.mark.asyncio
async def test_openai_connector_malformed_tool_args_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path": "/etc/passwd", "recursi',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    client = _make_mock_client(handler)
    connector = OpenAIConnector(api_key="key", http_client=client)

    with pytest.raises(MalformedToolCallArgumentsError):
        await connector.generate(LLMRequest(messages=()))


@pytest.mark.asyncio
async def test_ollama_connector_malformed_tool_args_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "delete_file",
                                "arguments": '{"target": "/var/log", "force":',
                            }
                        }
                    ],
                },
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        )

    client = _make_mock_client(handler)
    connector = OllamaConnector(base_url="http://localhost:11434", http_client=client)

    with pytest.raises(MalformedToolCallArgumentsError):
        await connector.generate(LLMRequest(messages=()))


@pytest.mark.asyncio
async def test_anthropic_connector_malformed_tool_args_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_ant_1",
                        "name": "calc",
                        "input": "invalid json not a dict [1, 2",
                    }
                ],
                "role": "assistant",
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    client = _make_mock_client(handler)
    connector = AnthropicConnector(api_key="key", http_client=client)

    with pytest.raises(MalformedToolCallArgumentsError):
        await connector.generate(LLMRequest(messages=()))


@pytest.mark.asyncio
async def test_gemini_connector_malformed_tool_args_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "calc",
                                        "args": "not a valid dict json [",
                                    }
                                }
                            ]
                        }
                    }
                ],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 5},
            },
        )

    client = _make_mock_client(handler)
    connector = GeminiConnector(api_key="key", http_client=client)

    with pytest.raises(MalformedToolCallArgumentsError):
        await connector.generate(LLMRequest(messages=()))


@pytest.mark.asyncio
async def test_gemini_connector_streaming_malformed_tool_args_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        sse_lines = [
            "data: "
            + json.dumps(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "functionCall": {
                                            "name": "search",
                                            "args": "broken json {",
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            )
        ]
        return httpx.Response(200, text="\n\n".join(sse_lines) + "\n\n")

    client = _make_mock_client(handler)
    connector = GeminiConnector(api_key="key", http_client=client)

    with pytest.raises(MalformedToolCallArgumentsError):
        async for _ in connector.stream(LLMRequest(messages=())):
            pass


@pytest.mark.asyncio
async def test_openai_connector_streaming_malformed_tool_args_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        sse_lines = [
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "function": {
                                            "name": "exec",
                                            "arguments": '{"broken":',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            )
        ]
        return httpx.Response(200, text="\n\n".join(sse_lines) + "\n\n")

    client = _make_mock_client(handler)
    connector = OpenAIConnector(api_key="key", http_client=client)

    with pytest.raises(MalformedToolCallArgumentsError):
        async for _ in connector.stream(LLMRequest(messages=())):
            pass


@pytest.mark.asyncio
async def test_ollama_connector_streaming_malformed_tool_args_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        ndjson = json.dumps(
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "exec",
                                "arguments": '{"invalid":',
                            }
                        }
                    ]
                }
            }
        )
        return httpx.Response(200, text=ndjson + "\n")

    client = _make_mock_client(handler)
    connector = OllamaConnector(base_url="http://localhost:11434", http_client=client)

    with pytest.raises(MalformedToolCallArgumentsError):
        async for _ in connector.stream(LLMRequest(messages=())):
            pass


# ======================================================================================
# Factory Resolution & Auto-Detection Tests (Principle 5 & Principle 6)
# ======================================================================================


def test_factory_default_resolution_to_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    # Clear cloud API keys and custom endpoints
    for var in (
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OLLAMA_BASE_URL",
        "OLLAMA_FAST_BASE_URL",
        "LOCAL_LLM_BASE_URL",
        "OLLAMA_HOST",
    ):
        monkeypatch.delenv(var, raising=False)

    # An unconfigured environment is refused rather than answered with a default endpoint
    # (#533). Building an `OllamaConnector` here succeeds, because it has no credential to
    # validate, and defers the failure to the first turn as a refused TCP connection — a true
    # statement about a socket and a false account of the configuration defect that caused it.
    with pytest.raises(LLMProviderNotConfiguredError) as excinfo:
        create_llm_connector()

    # The message has to be actionable by someone who does not know what Ollama is (P0).
    message = str(excinfo.value)
    assert "LLM_PROVIDER" in message
    assert "OPENAI_API_KEY" in message
    assert "OLLAMA_BASE_URL" in message


def test_factory_resolves_ollama_from_an_endpoint_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An endpoint variable is a configuration choice, so it is honoured, not refused.

    This is step 4 of the documented precedence, which the factory did not previously
    perform: it fell through to `OllamaConnector` whether or not any variable was set, so
    the configured and unconfigured cases were indistinguishable. Asserting both sides is
    what makes the refusal above a detector rather than a blanket failure.
    """
    for var in (
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OLLAMA_BASE_URL",
        "OLLAMA_FAST_BASE_URL",
        "LOCAL_LLM_BASE_URL",
        "OLLAMA_HOST",
    ):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11500")
    connector = create_llm_connector()
    assert isinstance(connector, OllamaConnector)
    assert connector.base_url == "http://127.0.0.1:11500"

    # An explicit base_url is equally a choice, with no variable set.
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    explicit = create_llm_connector(base_url="http://127.0.0.1:11501")
    assert isinstance(explicit, OllamaConnector)

    # And naming the provider needs no endpoint at all.
    assert isinstance(create_llm_connector(provider="ollama"), OllamaConnector)


def test_the_guide_documented_indepth_endpoint_is_honoured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A developer who followed `docs/local-development-guide.md` is not refused.

    That guide documents `OLLAMA_INDEPTH_BASE_URL` as part of the pre-configured 2-Tier
    setup, and `cli/commands/llm.py` reads it. #539 omitted it from
    `OLLAMA_ENDPOINT_ENV_VARS`, so the refusal it added fired on an environment the
    repository itself tells people to create — a regression from working to broken, and the
    exact drift the shared-list docstring claims to prevent.

    Model resolution already consults `OLLAMA_MODEL`, `OLLAMA_INDEPTH_MODEL`,
    `OLLAMA_FAST_MODEL` in that order; endpoint resolution mirrors it, so the tier a name
    belongs to means the same thing on both axes.
    """
    for var in (
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        *OLLAMA_ENDPOINT_ENV_VARS,
    ):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("OLLAMA_INDEPTH_BASE_URL", "http://localhost:11434/v1")
    connector = create_llm_connector()
    assert isinstance(connector, OllamaConnector)

    # And the endpoint it resolves is the one that was configured, not the default.
    assert connector.base_url is not None
    assert connector.base_url.startswith("http://localhost:11434")


def test_endpoint_precedence_mirrors_model_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`OLLAMA_BASE_URL` > `OLLAMA_INDEPTH_BASE_URL` > `OLLAMA_FAST_BASE_URL`."""
    for var in ("LLM_PROVIDER", *OLLAMA_ENDPOINT_ENV_VARS):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("OLLAMA_FAST_BASE_URL", "http://fast.invalid:1")
    assert create_llm_connector().base_url == "http://fast.invalid:1"

    monkeypatch.setenv("OLLAMA_INDEPTH_BASE_URL", "http://indepth.invalid:2")
    assert create_llm_connector().base_url == "http://indepth.invalid:2"

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://base.invalid:3")
    assert create_llm_connector().base_url == "http://base.invalid:3"


def test_a_configured_endpoint_is_not_shadowed_by_fallback_to_mock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`fallback_to_mock` is the unconfigured-case default, so a configured host wins.

    The factory docstring puts endpoint detection at step 4 and `fallback_to_mock` at step
    5, and #539 added the detection *below* the flag, so the code ran them in the opposite
    order. With an endpoint configured, the caller received a mock in place of the provider
    they had chosen — a substituted implementation with no in-band attribution, which is
    what the surrounding docstring spends thirty lines forbidding.

    The sibling test asserts this invariant for `OPENAI_API_KEY`, the one case where it
    already held. This asserts it for the case where it did not.
    """
    for var in (
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        *OLLAMA_ENDPOINT_ENV_VARS,
    ):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://configured.invalid:11434")
    assert isinstance(create_llm_connector(fallback_to_mock=True), OllamaConnector)


def test_factory_cloud_key_auto_detection_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    # 1. OpenAI key present -> OpenAIConnector
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test")
    conn1 = create_llm_connector()
    assert isinstance(conn1, OpenAIConnector)

    # 2. ANTHROPIC_API_KEY without OpenAI -> AnthropicConnector
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    conn2 = create_llm_connector()
    assert isinstance(conn2, AnthropicConnector)

    # 3. GEMINI_API_KEY without OpenAI/Anthropic -> GeminiConnector
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    conn3 = create_llm_connector()
    assert isinstance(conn3, GeminiConnector)

    # 4. GOOGLE_API_KEY alias -> GeminiConnector
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test")
    conn4 = create_llm_connector()
    assert isinstance(conn4, GeminiConnector)


def test_factory_ollama_endpoint_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("OLLAMA_FAST_BASE_URL", "http://192.0.2.10:11434")
    connector = create_llm_connector()
    assert isinstance(connector, OllamaConnector)
    assert connector.base_url == "http://192.0.2.10:11434"


def test_factory_fallback_to_mock_and_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    # Default fallback_to_mock is False -> unsupported provider raises LLMProviderError
    with pytest.raises(LLMProviderError, match="Unsupported LLM provider"):
        create_llm_connector(provider="nonexistent_unknown")

    # Explicit fallback_to_mock=False -> raises LLMProviderError
    with pytest.raises(LLMProviderError, match="Unsupported LLM provider"):
        create_llm_connector(provider="nonexistent_unknown", fallback_to_mock=False)

    # Explicit fallback_to_mock=True -> STILL raises. An unmappable provider name is not
    # a case a mock repairs; see
    # `test_an_unsupported_provider_name_is_refused_naming_the_value_not_replaced_by_a_mock`
    # (#397, P6).
    with pytest.raises(LLMProviderError, match="Unsupported LLM provider"):
        create_llm_connector(provider="nonexistent_unknown", fallback_to_mock=True)

    # Explicit mock provider -> returns MockLLMConnector
    conn_explicit_mock = create_llm_connector(provider="mock")
    assert isinstance(conn_explicit_mock, MockLLMConnector)


# ======================================================================================
# Default Model Sentinel & Fallback Resolution Seam Tests (Issue #279)
# ======================================================================================


@pytest.mark.parametrize(
    "model_input",
    [None, "", "default", "   ", "  default  "],
)
@pytest.mark.asyncio
async def test_openai_connector_resolves_default_model(model_input: str | None) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = json.loads(request.content.decode("utf-8"))
        captured["model"] = body["model"]
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-123",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OpenAI answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    client = _make_mock_client(handler)
    connector = OpenAIConnector(api_key="test_key", http_client=client)
    resp = await connector.generate(
        LLMRequest(
            model=model_input,
            messages=(ChatMessage(role=MessageRole.USER, content="Hello"),),
        )
    )

    assert captured["model"] == "gpt-4o"
    assert resp.provenance is not None
    assert resp.provenance.requested.model == "gpt-4o"
    assert resp.provenance.served_by.model == "gpt-4o"
    assert resp.provenance.degraded is False


@pytest.mark.parametrize(
    "model_input",
    [None, "", "default", "   ", "  default  "],
)
@pytest.mark.asyncio
async def test_anthropic_connector_resolves_default_model(model_input: str | None) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = json.loads(request.content.decode("utf-8"))
        captured["model"] = body["model"]
        return httpx.Response(
            200,
            json={
                "id": "msg_123",
                "model": body["model"],
                "content": [{"type": "text", "text": "Anthropic answer"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    client = _make_mock_client(handler)
    connector = AnthropicConnector(api_key="ant_key", http_client=client)
    resp = await connector.generate(
        LLMRequest(
            model=model_input,
            messages=(ChatMessage(role=MessageRole.USER, content="Hello"),),
        )
    )

    assert captured["model"] == "claude-3-5-sonnet"
    assert resp.provenance is not None
    assert resp.provenance.requested.model == "claude-3-5-sonnet"
    assert resp.provenance.served_by.model == "claude-3-5-sonnet"
    assert resp.provenance.degraded is False


@pytest.mark.parametrize(
    "model_input",
    [None, "", "default", "   ", "  default  "],
)
@pytest.mark.asyncio
async def test_gemini_connector_resolves_default_model(model_input: str | None) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Gemini answer"}], "role": "model"},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                },
                "modelVersion": "gemini-1.5-pro",
            },
        )

    client = _make_mock_client(handler)
    connector = GeminiConnector(api_key="gem_key", http_client=client)
    resp = await connector.generate(
        LLMRequest(
            model=model_input,
            messages=(ChatMessage(role=MessageRole.USER, content="Hello"),),
        )
    )

    assert "models/gemini-1.5-pro:generateContent" in captured["url"]
    assert resp.provenance is not None
    assert resp.provenance.requested.model == "gemini-1.5-pro"
    assert resp.provenance.served_by.model == "gemini-1.5-pro"
    assert resp.provenance.degraded is False


@pytest.mark.parametrize(
    "model_input",
    [None, "", "default", "   ", "  default  "],
)
@pytest.mark.asyncio
async def test_ollama_connector_resolves_default_model(
    monkeypatch: pytest.MonkeyPatch, model_input: str | None
) -> None:
    for var in ("OLLAMA_MODEL", "OLLAMA_INDEPTH_MODEL", "OLLAMA_FAST_MODEL"):
        monkeypatch.delenv(var, raising=False)

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = json.loads(request.content.decode("utf-8"))
        captured["model"] = body["model"]
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "message": {"role": "assistant", "content": "Ollama answer"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        )

    client = _make_mock_client(handler)
    connector = OllamaConnector(http_client=client)
    resp = await connector.generate(
        LLMRequest(
            model=model_input,
            messages=(ChatMessage(role=MessageRole.USER, content="Hello"),),
        )
    )

    assert captured["model"] == "qwen3:8b"
    assert resp.provenance is not None
    assert resp.provenance.requested.model == "qwen3:8b"
    assert resp.provenance.served_by.model == "qwen3:8b"
    assert resp.provenance.degraded is False


@pytest.mark.parametrize(
    ("model_input", "env_var", "expected_model"),
    [
        (None, "llama3.2", "llama3.2"),
        ("", "llama3.2", "llama3.2"),
        ("default", "llama3.2", "llama3.2"),
        ("  default  ", "llama3.2", "llama3.2"),
    ],
)
@pytest.mark.asyncio
async def test_ollama_connector_resolves_default_model_from_env(
    monkeypatch: pytest.MonkeyPatch,
    model_input: str | None,
    env_var: str,
    expected_model: str,
) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", env_var)

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = json.loads(request.content.decode("utf-8"))
        captured["model"] = body["model"]
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "message": {"role": "assistant", "content": "Ollama answer"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        )

    client = _make_mock_client(handler)
    connector = OllamaConnector(http_client=client)
    resp = await connector.generate(
        LLMRequest(
            model=model_input,
            messages=(ChatMessage(role=MessageRole.USER, content="Hello"),),
        )
    )

    assert captured["model"] == expected_model
    assert resp.provenance is not None
    assert resp.provenance.requested.model == expected_model
    assert resp.provenance.served_by.model == expected_model
    assert resp.provenance.degraded is False


@pytest.mark.parametrize(
    "model_input",
    [None, "", "default"],
)
@pytest.mark.asyncio
async def test_connectors_streaming_resolves_default_model(
    model_input: str | None,
) -> None:
    def openai_stream_handler(request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "gpt-4o"
        sse_content = (
            "data: "
            + json.dumps(
                {
                    "choices": [{"delta": {"content": "chunk"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 5},
                }
            )
            + "\n\ndata: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            content=sse_content.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
        )

    def anthropic_stream_handler(request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "claude-3-5-sonnet"
        sse_content = (
            'event: content_block_delta\ndata: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "chunk"}}\n\n'
            'event: message_delta\ndata: {"type": "message_delta", "usage": {"output_tokens": 5}}\n\n'
        )
        return httpx.Response(
            200,
            content=sse_content.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
        )

    def gemini_stream_handler(request: httpx.Request) -> httpx.Response:
        assert "models/gemini-1.5-pro:streamGenerateContent" in str(request.url)
        sse_content = 'data: {"candidates": [{"content": {"parts": [{"text": "chunk"}]}}], "usageMetadata": {"candidatesTokenCount": 5}}\n\n'
        return httpx.Response(
            200,
            content=sse_content.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
        )

    def ollama_stream_handler(request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "qwen3:8b"
        ndjson = json.dumps({"message": {"content": "chunk"}, "done": False}) + "\n"
        ndjson += json.dumps({"done": True, "prompt_eval_count": 5, "eval_count": 5}) + "\n"
        return httpx.Response(
            200,
            content=ndjson.encode("utf-8"),
            headers={"Content-Type": "application/x-ndjson"},
        )

    req = LLMRequest(
        model=model_input,
        messages=(ChatMessage(role=MessageRole.USER, content="hi"),),
    )

    # 1. OpenAI
    c_oa = OpenAIConnector(api_key="k", http_client=_make_mock_client(openai_stream_handler))
    chunks_oa = [c async for c in c_oa.stream(req)]
    assert any(c.delta_content == "chunk" for c in chunks_oa)

    # 2. Anthropic
    c_ant = AnthropicConnector(api_key="k", http_client=_make_mock_client(anthropic_stream_handler))
    chunks_ant = [c async for c in c_ant.stream(req)]
    assert any(c.delta_content == "chunk" for c in chunks_ant)

    # 3. Gemini
    c_gem = GeminiConnector(api_key="k", http_client=_make_mock_client(gemini_stream_handler))
    chunks_gem = [c async for c in c_gem.stream(req)]
    assert any(c.delta_content == "chunk" for c in chunks_gem)

    # 4. Ollama
    c_oll = OllamaConnector(http_client=_make_mock_client(ollama_stream_handler))
    chunks_oll = [c async for c in c_oll.stream(req)]
    assert any(c.delta_content == "chunk" for c in chunks_oll)


@pytest.mark.parametrize(
    "model_input",
    [None, "", "default", "   ", "  default  "],
)
@pytest.mark.asyncio
async def test_mock_connector_resolves_default_model(model_input: str | None) -> None:
    connector = MockLLMConnector()
    resp = await connector.generate(
        LLMRequest(
            model=model_input,
            messages=(ChatMessage(role=MessageRole.USER, content="Hello"),),
        )
    )
    assert resp.model_name == "mock-model"
    assert resp.provenance is not None
    assert resp.provenance.requested.model == "mock-model"


@pytest.mark.asyncio
async def test_mock_connector_custom_default_model() -> None:
    connector = MockLLMConnector(default_model="mock-gpt-4o")
    resp = await connector.generate(
        LLMRequest(
            model=None,
            messages=(ChatMessage(role=MessageRole.USER, content="Hello"),),
        )
    )
    assert resp.model_name == "mock-gpt-4o"
    assert resp.provenance is not None
    assert resp.provenance.requested.model == "mock-gpt-4o"
    assert resp.provenance.served_by is not None
    assert resp.provenance.served_by.model == "mock-gpt-4o"


@pytest.mark.asyncio
async def test_mock_connector_labels_its_token_figures_as_the_shared_estimate() -> None:
    """The mock's token figures are estimates, from the shared estimator, and say so (#983).

    The mock calls no provider, so nothing counted its tokens. Before #983 it built them from
    `len // 4` and left `count_source` at its `PROVIDER` default, so a mock-backed dashboard
    showed them as counted and the budget booked them as provider counts. The rule is the
    design doc's §6.7 **[#939]**: a count the provider did not report is estimated with
    `estimate_request_tokens` / `estimate_reply_tokens` and labelled `ESTIMATE`.

    The prompt and reply are Hangul, three UTF-8 bytes a syllable, so the byte estimator and
    `len // 4` disagree; the request carries a tool definition and the reply a tool call,
    which only the shared estimator counts. Both endpoints are checked: `stream` hands out the
    same usage on its last chunk.

    Killed by: src/uclone_x/llm/connectors/mock.py :: count_source=TokenCountSource.ESTIMATE,
    Becomes: count_source=TokenCountSource.PROVIDER,
    Killed by: src/uclone_x/llm/connectors/mock.py :: in_tokens = estimate_request_tokens(request)
    Becomes: in_tokens = max(1, len(str([m.content for m in request.messages])) // 4)
    Killed by: src/uclone_x/llm/connectors/mock.py :: out_tokens = estimate_reply_tokens(content, tool_calls_to_return)
    Becomes: out_tokens = max(1, len(content) // 4)
    """
    reply = "좋아요, 파일을 먼저 읽어 보겠습니다"
    tool_call = ToolCallRequest(id="call_983", name="read_file", arguments={"path": "날씨.txt"})
    request = LLMRequest(
        messages=(ChatMessage(role=MessageRole.USER, content="안녕하세요, 오늘 날씨가 좋네요"),),
        tools=(
            ToolDefinition(
                name="read_file",
                description="Read a file from the workspace",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            ),
        ),
    )
    want_in = estimate_request_tokens(request)
    want_out = estimate_reply_tokens(reply, (tool_call,))
    # Not vacuous: the old heuristic gives different figures for this request and reply.
    assert (want_in, want_out) != (
        max(1, len(str([m.content for m in request.messages])) // 4),
        max(1, len(reply) // 4),
    )

    generated = await MockLLMConnector(default_response=reply, tool_calls=[tool_call]).generate(
        request
    )
    chunks = [
        c
        async for c in MockLLMConnector(default_response=reply, tool_calls=[tool_call]).stream(
            request
        )
    ]
    streamed = [c.usage for c in chunks if c.usage is not None]
    assert len(streamed) == 1, "the stream must report its usage exactly once"

    for usage in (generated.usage, streamed[0]):
        assert usage is not None
        assert usage.count_source is TokenCountSource.ESTIMATE
        assert (usage.input_tokens, usage.output_tokens) == (want_in, want_out)
        assert usage.total_tokens == want_in + want_out


# ======================================================================================
# GeminiConnector payload fidelity (#380, P6)
#
# Four `or`-defaulting / truthiness sites in one function each made a distinct input
# arrive at a paid provider as some other input's bytes. Every test below is a pair or a
# refusal: the pairs make `None` and `""` observably different on the wire, and the
# refusals pin that an unmappable message fails before the request rather than being
# invented into something sendable. The fixed and broken forms differ by one operator and
# both emit plausible payloads, so each test names the mutation it exists to catch.
#
# `contents` is read back off the captured request body rather than from
# `_build_payload`, so what is asserted is what the provider would receive.
# ======================================================================================


async def _gemini_sent_body(*messages: ChatMessage) -> dict[str, Any]:
    """The whole request body Gemini actually receives for `messages`."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "ok"}], "role": "model"},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 2,
                },
                "modelVersion": "gemini-1.5-pro",
            },
        )

    connector = GeminiConnector(api_key="k", http_client=_make_mock_client(handler))
    await connector.generate(LLMRequest(model="gemini-1.5-pro", messages=tuple(messages)))
    return cast("dict[str, Any]", captured["body"])


async def _gemini_sent_contents(*messages: ChatMessage) -> list[dict[str, Any]]:
    """The `contents` array Gemini actually receives for `messages`."""
    body = await _gemini_sent_body(*messages)
    return cast("list[dict[str, Any]]", body["contents"])


@pytest.mark.asyncio
async def test_gemini_refuses_a_nameless_tool_result_rather_than_naming_it_tool() -> None:
    """A `TOOL` message with no `name` is refused, not sent as a function called 'tool'.

    This is the one site of the four that **fabricates** rather than loses: the others
    make two inputs indistinguishable, while `msg.name or "tool"` invents a function that
    never ran, under a name no downstream reader can tell from a real tool named `tool`.

    Mutation this exists to catch: restore `"name": msg.name or "tool"`.
    """
    msg = ChatMessage(role=MessageRole.TOOL, content="42", tool_call_id="call_1")
    assert msg.name is None

    with pytest.raises(UnmappableChatMessageError) as excinfo:
        await _gemini_sent_contents(msg)

    text = str(excinfo.value)
    assert "name=None" in text, text
    assert "call_1" in text, text


@pytest.mark.asyncio
async def test_gemini_refuses_a_blank_tool_name_the_same_way_as_a_missing_one() -> None:
    """`name=""` and `name="   "` are refused too.

    `ToolCallRequest.name` is a bare `str` with no length constraint, so a blank name is
    constructible, and it is exactly as unusable as `functionResponse.name` as `None` is.
    Testing only `None` would leave the truthiness guard and a narrow `is None` guard
    indistinguishable, and the narrow one ships a blank function name to the provider.

    Mutation this exists to catch: narrow the guard to `if msg.name is None:`.
    """
    for blank in ("", "   "):
        with pytest.raises(UnmappableChatMessageError) as excinfo:
            await _gemini_sent_contents(
                ChatMessage(role=MessageRole.TOOL, name=blank, content="42")
            )
        assert repr(blank) in str(excinfo.value), str(excinfo.value)


@pytest.mark.asyncio
async def test_gemini_sends_a_real_tool_name_verbatim() -> None:
    """The guard refuses only unusable names; a real one is passed through unchanged.

    Paired with the two refusals above so that a mutation replacing the whole `TOOL`
    branch with an unconditional raise cannot pass the suite.
    """
    contents = await _gemini_sent_contents(
        ChatMessage(role=MessageRole.TOOL, name="read_file", content="42")
    )
    assert contents == [
        {
            "role": "user",
            "parts": [{"functionResponse": {"name": "read_file", "response": {"result": "42"}}}],
        }
    ]


@pytest.mark.asyncio
async def test_gemini_distinguishes_an_empty_tool_result_from_an_absent_one() -> None:
    """`content=""` is sent as `{"result": ""}`; `content=None` is refused.

    "the tool ran and returned nothing" and "no return value was recorded" are different
    facts about an invocation, and `msg.content or ""` sent both as `{"result": ""}`. The
    empty-string half of this pair passes under that mutation as well — that is the
    coincidence case — so the refusal half is what makes the test able to fail.

    `{"result": null}` is deliberately **not** the chosen representation for `None`:
    whether Gemini distinguishes it from `{"result": ""}` is unmeasured here, and
    guessing on a paid wire format is what P6 forbids. `adapters.uclone2.adk_content`
    does carry `None` through as `{"result": None}`; its contract is round-trip identity
    against an in-process model rather than an unobservable request.

    Mutation this exists to catch: restore `"result": msg.content or ""`.
    """
    empty = await _gemini_sent_contents(
        ChatMessage(role=MessageRole.TOOL, name="read_file", content="")
    )
    assert empty[0]["parts"][0]["functionResponse"]["response"] == {"result": ""}

    with pytest.raises(UnmappableChatMessageError) as excinfo:
        await _gemini_sent_contents(
            ChatMessage(role=MessageRole.TOOL, name="read_file", content=None)
        )
    assert "content=None" in str(excinfo.value), str(excinfo.value)


@pytest.mark.asyncio
async def test_gemini_keeps_an_empty_assistant_text_part_beside_its_tool_calls() -> None:
    """An `ASSISTANT` turn with `content=""` **and** tool calls keeps its text part.

    This is the conditional defect. With no tool calls the old
    `parts if parts else [{"text": ""}]` fallback restored a text part, so nothing looked
    dropped; with tool calls present `parts` was already non-empty, the fallback did not
    fire, and the empty text part vanished from the payload entirely. The distinguishing
    input is therefore empty content **together with** a tool call — a request carrying
    either one alone cannot tell the two forms apart.

    Mutation this exists to catch: restore `if msg.content:`.
    """
    contents = await _gemini_sent_contents(
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=(ToolCallRequest(id="call_1", name="read_file", arguments={"p": "a.txt"}),),
        )
    )
    assert contents == [
        {
            "role": "model",
            "parts": [
                {"text": ""},
                {"functionCall": {"name": "read_file", "args": {"p": "a.txt"}}},
            ],
        }
    ]


@pytest.mark.asyncio
async def test_gemini_omits_the_text_part_for_a_tool_calls_only_assistant_turn() -> None:
    """`content=None` with tool calls emits **no** text part — the other side of the pair.

    `content=None` here is the ordinary shape of a model turn that is only tool calls, so
    it has a faithful representation and must not raise. This test is what stops the fix
    for the previous one from becoming an unconditional
    `parts.append({"text": msg.content or ""})`, which would add a spurious empty text
    part to every such turn.

    Mutation this exists to catch: emit the text part unconditionally.
    """
    contents = await _gemini_sent_contents(
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=(ToolCallRequest(id="call_1", name="read_file"),),
        )
    )
    assert contents[0]["parts"] == [{"functionCall": {"name": "read_file", "args": {}}}]


@pytest.mark.asyncio
async def test_gemini_refuses_an_assistant_turn_with_neither_content_nor_tool_calls() -> None:
    """A message carrying nothing is refused, not backfilled with an empty text part.

    `parts if parts else [{"text": ""}]` turned "this turn said nothing" into "this turn
    said the empty string". Gemini does require a non-empty `parts`, which is why the
    fallback was written; the conclusion P6 draws from that is that the message has no
    representation, not that one should be invented.

    Mutation this exists to catch: restore `parts if parts else [{"text": ""}]`.
    """
    with pytest.raises(UnmappableChatMessageError) as excinfo:
        await _gemini_sent_contents(ChatMessage(role=MessageRole.ASSISTANT, content=None))
    assert "no tool_calls" in str(excinfo.value), str(excinfo.value)


@pytest.mark.asyncio
async def test_gemini_distinguishes_an_empty_user_turn_from_an_absent_one() -> None:
    """`content=""` is sent as `{"text": ""}`; `content=None` is refused.

    The same conflation as the `TOOL` branch, on the branch that carries the majority of
    traffic. As with that pair the empty-string half passes under the mutation too; the
    refusal half is the load-bearing assertion.

    Mutation this exists to catch: restore `[{"text": msg.content or ""}]`.
    """
    contents = await _gemini_sent_contents(ChatMessage(role=MessageRole.USER, content=""))
    assert contents == [{"role": "user", "parts": [{"text": ""}]}]

    with pytest.raises(UnmappableChatMessageError) as excinfo:
        await _gemini_sent_contents(ChatMessage(role=MessageRole.USER, content=None))
    assert "content=None" in str(excinfo.value), str(excinfo.value)
    assert "'user'" in str(excinfo.value), str(excinfo.value)


@pytest.mark.asyncio
async def test_gemini_payload_refusals_are_not_provider_errors() -> None:
    """These failures happen before any request, so they are not provider errors.

    `LLMProviderError` means "a provider returned an error or was unreachable", and a
    caller may reasonably retry or fail over on it. Nothing was sent here, so retrying
    the same message would fail identically; the distinct type keeps a caller's failover
    logic from treating a malformed message as a transient provider fault.
    """
    with pytest.raises(LLMError) as excinfo:
        await _gemini_sent_contents(ChatMessage(role=MessageRole.USER, content=None))
    assert isinstance(excinfo.value, UnmappableChatMessageError)
    assert not isinstance(excinfo.value, LLMProviderError)


@pytest.mark.asyncio
async def test_gemini_joins_an_empty_system_instruction_rather_than_filtering_it() -> None:
    """A `SYSTEM` message holding `""` contributes a segment to the join.

    The fifth site, found by a reviewer after the first four were fixed and
    graded too leniently by this author as "omits rather than substitutes". It is
    substitution: under `if msg.content:` a request carrying `["A", ""]`, one carrying
    `["A", None]` and one carrying no second system message at all emitted **identical**
    `systemInstruction` bytes, so two distinct inputs were reported as a third. What makes
    the difference observable is the trailing separator — a faithful join of `["A", ""]`
    is `"A\n\n"`, not `"A"`.

    Mutation this exists to catch: restore `if msg.content:` on the `SYSTEM` branch.
    """
    with_empty = await _gemini_sent_body(
        ChatMessage(role=MessageRole.SYSTEM, content="A"),
        ChatMessage(role=MessageRole.SYSTEM, content=""),
        ChatMessage(role=MessageRole.USER, content="hi"),
    )
    without = await _gemini_sent_body(
        ChatMessage(role=MessageRole.SYSTEM, content="A"),
        ChatMessage(role=MessageRole.USER, content="hi"),
    )

    assert with_empty["systemInstruction"] == {"parts": [{"text": "A\n\n"}]}
    assert without["systemInstruction"] == {"parts": [{"text": "A"}]}
    # The point of the test: the two requests must not be the same bytes.
    assert with_empty["systemInstruction"] != without["systemInstruction"]


@pytest.mark.asyncio
async def test_gemini_refuses_a_system_message_with_no_content() -> None:
    """`SYSTEM` `content=None` is refused rather than skipped.

    Skipping it made "there is no system instruction" and "this system instruction is
    absent from a message that exists" the same request. Paired with the join test above
    so that neither `is not None` nor a bare truthiness guard can satisfy both.

    Mutation this exists to catch: drop the guard and restore `if msg.content:`.
    """
    with pytest.raises(UnmappableChatMessageError) as excinfo:
        await _gemini_sent_body(
            ChatMessage(role=MessageRole.SYSTEM, content=None),
            ChatMessage(role=MessageRole.USER, content="hi"),
        )
    assert "content=None" in str(excinfo.value), str(excinfo.value)
    assert "system" in str(excinfo.value), str(excinfo.value)


@pytest.mark.asyncio
async def test_gemini_refuses_the_tool_shape_ui_history_rehydration_produces() -> None:
    """The `ui/app.py` session-rehydration path reaches both `TOOL` refusals.

    Disclosed rather than accommodated. `_load_session_history` in `uclone_x.ui.app`
    rebuilds a `ChatMessage` from a persisted record with **no `name` argument at all**,
    and with `content` explicitly `None` when the record has no `content` key. Its role
    comes from `MessageRole(role_str)` over a persisted `msg.role.value`, which is why a
    grep for the literal `MessageRole.TOOL` did not find it (§5.5 shape 3) and why this
    author reported "no caller relies on it" — a claim that was false.

    The old behaviour on this path was the defect in its purest form: the writer discards
    the tool name, and the reader then invented `name: "tool"` to replace it, so the
    provider was told a function ran under a name the UI had already thrown away. Raising
    is therefore the correct outcome here and the default must not be restored. The
    remaining defect is the lossy rehydration in `ui/app.py`, which is filed separately.

    Mutation this exists to catch: any restoration of the `"tool"` or `""` defaults.
    """
    # The mechanism that makes MessageRole.TOOL reachable from a persisted string. If
    # this ever stops holding, the exposure this test documents is gone and so is the
    # reason for the test.
    assert MessageRole("tool") is MessageRole.TOOL

    for content in ("", None):
        with pytest.raises(UnmappableChatMessageError) as excinfo:
            await _gemini_sent_contents(ChatMessage(role=MessageRole("tool"), content=content))
        assert "name=None" in str(excinfo.value), str(excinfo.value)


# ======================================================================================
# Issue #385 — the same `or`-defaulting class in every remaining connector
#
# PR #384 fixed `GeminiConnector._build_payload` and reported the siblings rather than
# widening its diff. These pin the siblings. Each test names the mutation that
# reintroduces the defect it covers, as a diff rather than as prose, so a mutant is
# identified by an operator rather than by a description (§6.9).
#
# Read alongside the request-capture helpers above: every assertion here reads the body
# **off the captured HTTP request**, not off `_build_payload`, so what is asserted is
# what the provider would actually receive.
# ======================================================================================


async def _anthropic_sent_body(
    *messages: ChatMessage,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """The JSON body Anthropic actually receives for `messages`."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-3-5-sonnet",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    connector = AnthropicConnector(api_key="k", http_client=_make_mock_client(handler))
    await connector.generate(
        LLMRequest(
            model="claude-3-5-sonnet",
            messages=tuple(messages),
            max_tokens=max_tokens,
        )
    )
    return cast("dict[str, Any]", captured["body"])


async def _openai_sent_messages(*messages: ChatMessage) -> list[dict[str, Any]]:
    """The `messages` array OpenAI actually receives for `messages`."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    connector = OpenAIConnector(api_key="k", http_client=_make_mock_client(handler))
    await connector.generate(LLMRequest(model="gpt-4o", messages=tuple(messages)))
    body = cast("dict[str, Any]", captured["body"])
    return cast("list[dict[str, Any]]", body["messages"])


async def _ollama_sent_messages(*messages: ChatMessage) -> list[dict[str, Any]]:
    """The `messages` array Ollama actually receives for `messages`."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5-coder:14b",
                "message": {"content": "ok"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 1,
                "eval_count": 1,
            },
        )

    connector = OllamaConnector(http_client=_make_mock_client(handler))
    await connector.generate(LLMRequest(messages=tuple(messages)))
    body = cast("dict[str, Any]", captured["body"])
    return cast("list[dict[str, Any]]", body["messages"])


# --------------------------------------------------------------------------------------
# anthropic.py:94 — the site that differs in kind: a fabricated *correlation key*
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_refuses_a_tool_result_with_no_tool_use_id() -> None:
    """`tool_use_id` is refused when absent, never sent as `""`.

    This is the site of the eleven that differs in kind, and the difference is not one of
    degree. Gemini's `msg.name or "tool"` fabricates a **label**; Anthropic correlates a
    `tool_result` block to the `tool_use` block that requested it **by id**, so
    `msg.tool_call_id or ""` fabricates the **key the association is made on**. The result
    is not delivered under a wrong name — it is matched against a block that does not
    exist, or against whichever block an empty id collides with. Neither the payload text
    nor the response shows it, which is why no amount of reading either would have found
    this one.

    Mutation this exists to catch:
        -   if msg.tool_call_id is None or not msg.tool_call_id.strip():
        -       raise UnmappableChatMessageError(...)
        -   "tool_use_id": msg.tool_call_id,
        +   "tool_use_id": msg.tool_call_id or "",
    """
    msg = ChatMessage(role=MessageRole.TOOL, name="read_file", content="42")
    assert msg.tool_call_id is None

    with pytest.raises(UnmappableChatMessageError) as excinfo:
        await _anthropic_sent_body(msg)

    text = str(excinfo.value)
    assert "tool_call_id=None" in text, text
    assert "read_file" in text, text
    # The mechanism, not just the field, so a reviewer skimming a uniform list of
    # `or ""` fixes reads why this one is worse.
    assert "correlate" in text, text


@pytest.mark.asyncio
async def test_anthropic_refuses_a_blank_tool_use_id_the_same_way_as_a_missing_one() -> None:
    """`tool_call_id=""` and `"  "` are refused too.

    `ChatMessage.tool_call_id` is an unconstrained `str | None`, so a blank id is
    constructible and is exactly as unusable as a correlation key as `None` is. Without
    this, a guard narrowed to `is None` is indistinguishable from the real fix while still
    sending an empty `tool_use_id`.

    Mutation this exists to catch:
        -   if msg.tool_call_id is None or not msg.tool_call_id.strip():
        +   if msg.tool_call_id is None:
    """
    for blank in ("", "   "):
        with pytest.raises(UnmappableChatMessageError) as excinfo:
            await _anthropic_sent_body(
                ChatMessage(role=MessageRole.TOOL, name="t", content="42", tool_call_id=blank)
            )
        assert repr(blank) in str(excinfo.value), str(excinfo.value)


@pytest.mark.asyncio
async def test_anthropic_refuses_a_tool_result_with_no_recorded_content() -> None:
    """`TOOL` `content=None` is refused; `content=""` is sent as `""`.

    The pair is the whole test. `msg.content or ""` maps both to `""`, so only asserting
    that `""` survives cannot distinguish the fix from the defect — the discriminating
    input is `None`.

    Mutation this exists to catch:
        -   if msg.content is None:
        -       raise UnmappableChatMessageError(...)
        -   "content": msg.content,
        +   "content": msg.content or "",
    """
    with pytest.raises(UnmappableChatMessageError) as excinfo:
        await _anthropic_sent_body(
            ChatMessage(role=MessageRole.TOOL, name="t", tool_call_id="c1", content=None)
        )
    assert "content=None" in str(excinfo.value), str(excinfo.value)

    body = await _anthropic_sent_body(
        ChatMessage(role=MessageRole.TOOL, name="t", tool_call_id="c1", content="")
    )
    block = cast("list[dict[str, Any]]", body["messages"][0]["content"])[0]
    assert block == {"type": "tool_result", "tool_use_id": "c1", "content": ""}


@pytest.mark.asyncio
async def test_anthropic_joins_an_empty_system_message_rather_than_filtering_it() -> None:
    """An empty `SYSTEM` message changes the request; filtering it made it invisible.

    Under `if msg.content:` a request carrying `["A", ""]` and one carrying only `["A"]`
    emitted a byte-identical `system` field, so two distinct inputs were reported as one.
    The trailing separator is what makes the difference observable: a faithful join of
    `["A", ""]` is `"A\\n\\n"`.

    Mutation this exists to catch:
        -   if msg.content is None:
        -       raise UnmappableChatMessageError(...)
        -   system_prompts.append(msg.content)
        +   if msg.content:
        +       system_prompts.append(msg.content)
    """
    with_empty = await _anthropic_sent_body(
        ChatMessage(role=MessageRole.SYSTEM, content="A"),
        ChatMessage(role=MessageRole.SYSTEM, content=""),
        ChatMessage(role=MessageRole.USER, content="hi"),
    )
    without = await _anthropic_sent_body(
        ChatMessage(role=MessageRole.SYSTEM, content="A"),
        ChatMessage(role=MessageRole.USER, content="hi"),
    )

    assert with_empty["system"] == "A\n\n"
    assert without["system"] == "A"
    assert with_empty["system"] != without["system"]


@pytest.mark.asyncio
async def test_anthropic_refuses_a_system_message_with_no_content() -> None:
    """`SYSTEM` `content=None` is refused rather than skipped.

    Paired with the join test above so that neither a bare truthiness guard nor a
    silently-skipping `is not None` can satisfy both.

    Mutation this exists to catch: restore `if msg.content:` on the `SYSTEM` branch.
    """
    with pytest.raises(UnmappableChatMessageError) as excinfo:
        await _anthropic_sent_body(
            ChatMessage(role=MessageRole.SYSTEM, content=None),
            ChatMessage(role=MessageRole.USER, content="hi"),
        )
    assert "content=None" in str(excinfo.value), str(excinfo.value)
    assert "system" in str(excinfo.value), str(excinfo.value)


@pytest.mark.asyncio
async def test_anthropic_sends_an_empty_assistant_text_block_rather_than_dropping_it() -> None:
    """An `ASSISTANT` turn whose text is `""` still emits a text block.

    `if msg.content:` dropped it, which mattered **only when tool calls were present** —
    with no tool calls the `content_blocks if content_blocks else (msg.content or "")`
    fallback put an empty string back. So the conditionality is what makes it observable,
    and this is the input that exposes it.

    Mutation this exists to catch:
        -   if msg.content is not None:
        +   if msg.content:
    """
    call = ToolCallRequest(id="t1", name="read_file", arguments={})
    blocks = cast(
        "list[dict[str, Any]]",
        (
            await _anthropic_sent_body(
                ChatMessage(role=MessageRole.ASSISTANT, content="", tool_calls=(call,))
            )
        )["messages"][0]["content"],
    )
    assert blocks[0] == {"type": "text", "text": ""}
    assert blocks[1]["type"] == "tool_use"

    # `content=None` with tool calls is faithfully representable: no text was said, so no
    # text part is emitted. That is an omission of something absent, not a substitution.
    none_blocks = cast(
        "list[dict[str, Any]]",
        (
            await _anthropic_sent_body(
                ChatMessage(role=MessageRole.ASSISTANT, content=None, tool_calls=(call,))
            )
        )["messages"][0]["content"],
    )
    assert [b["type"] for b in none_blocks] == ["tool_use"]


@pytest.mark.asyncio
async def test_anthropic_refuses_an_assistant_turn_with_nothing_to_send() -> None:
    """`ASSISTANT` with `content=None` and no tool calls is refused, not sent as `""`.

    The expression this replaces defaulted twice over: the empty block list fell back to
    the same `msg.content` that produced it, and that `None` was then coerced to `""`. So
    an assistant turn that said nothing arrived as one that said the empty string.

    Mutation this exists to catch:
        -   if not content_blocks:
        -       raise UnmappableChatMessageError(...)
        -   {"role": "assistant", "content": content_blocks}
        +   {"role": "assistant", "content": content_blocks or (msg.content or "")}
    """
    with pytest.raises(UnmappableChatMessageError) as excinfo:
        await _anthropic_sent_body(ChatMessage(role=MessageRole.ASSISTANT, content=None))
    assert "content=None" in str(excinfo.value), str(excinfo.value)
    assert "assistant" in str(excinfo.value), str(excinfo.value)


@pytest.mark.asyncio
async def test_anthropic_refuses_a_user_turn_with_no_content() -> None:
    """`USER` `content=None` is refused; `""` is sent as `""`.

    This branch carries the majority of traffic, so it is the one whose conflation is
    reached most often.

    Mutation this exists to catch:
        -   if msg.content is None:
        -       raise UnmappableChatMessageError(...)
        -   {"role": "user", "content": msg.content}
        +   {"role": "user", "content": msg.content or ""}
    """
    with pytest.raises(UnmappableChatMessageError) as excinfo:
        await _anthropic_sent_body(ChatMessage(role=MessageRole.USER, content=None))
    assert "content=None" in str(excinfo.value), str(excinfo.value)

    body = await _anthropic_sent_body(ChatMessage(role=MessageRole.USER, content=""))
    assert body["messages"][0] == {"role": "user", "content": ""}


@pytest.mark.asyncio
async def test_anthropic_sends_max_tokens_zero_as_zero_rather_than_4096() -> None:
    """`max_tokens=0` is passed through; only `None` takes the connector's 4096.

    `LLMRequest.max_tokens` is an unconstrained `int | None`, so `0` is constructible, and
    `request.max_tokens or 4096` billed a caller that asked for zero against 4096. The
    `None` default itself is kept and is not a substitution in P6's sense: nothing failed,
    Anthropic requires the field, and the value is the connector's declared policy stated
    in its docstring.

    Mutation this exists to catch:
        -   request.max_tokens if request.max_tokens is not None else 4096
        +   request.max_tokens or 4096
    """
    zero = await _anthropic_sent_body(
        ChatMessage(role=MessageRole.USER, content="hi"), max_tokens=0
    )
    assert zero["max_tokens"] == 0

    absent = await _anthropic_sent_body(ChatMessage(role=MessageRole.USER, content="hi"))
    assert absent["max_tokens"] == 4096


# --------------------------------------------------------------------------------------
# openai.py — the connector that contained both the correct and the incorrect form
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_refuses_a_tool_message_with_no_content_not_undoing_its_own_guard() -> None:
    """The `TOOL` branch no longer overwrites the `is not None` guard eight lines above.

    `if msg.content is not None` (the fix #380 applied) and
    `m_dict["content"] = msg.content or ""` sat eight lines apart in the same function, so
    a reader who checked the guard concluded the connector was safe. `git log -L 82,92`
    shows both arrived in the **same** commit (`62412fd`), so this was never drift: one
    constraint was encoded two ways. The `is not None` guard is the intended rule; the
    `TOOL` override existed because OpenAI **requires** `content` on a `tool` message, so
    omitting the key is not an option there. That constraint is real and is kept — by
    refusing, which is what a required field with no faithful value calls for.

    Mutation this exists to catch:
        -   if msg.content is None:
        -       raise UnmappableChatMessageError(...)
        -   m_dict["content"] = msg.content
        +   m_dict["content"] = msg.content or ""
    """
    with pytest.raises(UnmappableChatMessageError) as excinfo:
        await _openai_sent_messages(
            ChatMessage(role=MessageRole.TOOL, name="t", tool_call_id="c1", content=None)
        )
    text = str(excinfo.value)
    assert "content=None" in text, text
    assert "requires content" in text, text

    # `""` is a real value and is sent as one, so the key is present exactly as the
    # constraint demands, without inventing what it holds.
    sent = await _openai_sent_messages(
        ChatMessage(role=MessageRole.TOOL, name="t", tool_call_id="c1", content="")
    )
    assert sent[0]["content"] == ""
    assert sent[0]["tool_call_id"] == "c1"


@pytest.mark.asyncio
async def test_openai_refuses_a_tool_message_with_no_tool_call_id() -> None:
    """`tool_call_id` is refused when absent, never silently dropped.

    `if msg.tool_call_id:` omitted the key, which is not a result with a missing label —
    it is a result attached to nothing. Same class as `anthropic.py`'s `tool_use_id`,
    reached by omission rather than by fabrication.

    Mutation this exists to catch:
        -   if msg.tool_call_id is None or not msg.tool_call_id.strip():
        -       raise UnmappableChatMessageError(...)
        -   m_dict["tool_call_id"] = msg.tool_call_id
        +   if msg.tool_call_id:
        +       m_dict["tool_call_id"] = msg.tool_call_id
    """
    for absent in (None, "", "   "):
        with pytest.raises(UnmappableChatMessageError) as excinfo:
            await _openai_sent_messages(
                ChatMessage(role=MessageRole.TOOL, name="t", content="42", tool_call_id=absent)
            )
        assert "tool_call_id=" in str(excinfo.value), str(excinfo.value)


@pytest.mark.asyncio
async def test_openai_sends_an_empty_name_as_empty_rather_than_omitting_it() -> None:
    """`name=""` is emitted; only `name=None` omits the key.

    `if msg.name:` made a message holding `name=""` and one holding `name=None` produce
    the same request, which is the same conflation as the content sites in the field that
    identifies the speaker.

    Mutation this exists to catch:
        -   if msg.name is not None:
        +   if msg.name:
    """
    empty = await _openai_sent_messages(ChatMessage(role=MessageRole.USER, content="hi", name=""))
    assert empty[0]["name"] == ""

    absent = await _openai_sent_messages(ChatMessage(role=MessageRole.USER, content="hi"))
    assert "name" not in absent[0]


# --------------------------------------------------------------------------------------
# ollama.py — unconditional on every role, plus the enum default toward success
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ollama_refuses_a_message_with_no_content_on_every_role() -> None:
    """`content=None` is refused for every role; `""` is sent as `""`.

    `"content": msg.content or ""` was unconditional, so unlike the other two connectors
    there was no role for which this connector got it right. That a model runs locally
    changes who is billed, not whether the substitution is observable — it is not.

    Mutation this exists to catch:
        -   if msg.content is None:
        -       raise UnmappableChatMessageError(...)
        -   "content": msg.content,
        +   "content": msg.content or "",
    """
    for role in (MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT):
        with pytest.raises(UnmappableChatMessageError) as excinfo:
            await _ollama_sent_messages(ChatMessage(role=role, content=None))
        assert "content=None" in str(excinfo.value), str(excinfo.value)
        assert role.value in str(excinfo.value), str(excinfo.value)

    sent = await _ollama_sent_messages(ChatMessage(role=MessageRole.USER, content=""))
    assert sent[0] == {"role": "user", "content": ""}


# --------------------------------------------------------------------------------------
# The enum-mapping shape: an unrecognised value silently reported as *success*
# (issue #385's comment; not `or`-defaulting, same defect, same remedy)
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("stop", FinishReason.STOP),
        ("length", FinishReason.LENGTH),
        ("load", FinishReason.UNKNOWN),
        ("", FinishReason.UNKNOWN),
        (None, FinishReason.UNKNOWN),
    ],
)
async def test_ollama_generate_reports_an_unrecognised_done_reason_as_unknown(
    raw: str | None, expected: FinishReason
) -> None:
    """An unrecognised `done_reason` is `UNKNOWN`, not `STOP`.

    The substitution here ran toward **success**, which is the direction nothing prompts
    anyone to check: a response truncated for a reason Ollama names and this connector
    does not enumerate arrived at the caller indistinguishable from one that finished
    normally.

    Mutation this exists to catch:
        -   return FinishReason.UNKNOWN
        +   return FinishReason.STOP
      (equivalently, restoring the inline
       `FinishReason.LENGTH if data.get("done_reason") == "length" else FinishReason.STOP`)
    """
    payload: dict[str, Any] = {
        "model": "qwen2.5-coder:14b",
        "message": {"content": "ok"},
        "done": True,
        "prompt_eval_count": 1,
        "eval_count": 1,
    }
    if raw is not None:
        payload["done_reason"] = raw

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    connector = OllamaConnector(http_client=_make_mock_client(handler))
    resp = await connector.generate(
        LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),))
    )
    assert resp.finish_reason == expected


@pytest.mark.asyncio
async def test_ollama_stream_reads_done_reason_rather_than_always_reporting_stop() -> None:
    """The streaming path consults `done_reason`; it previously ignored it entirely.

    `FinishReason.TOOL_CALLS if tool_calls else FinishReason.STOP` in `stream` did not
    read `done_reason` **at all**, so every non-tool-call finish was reported as a clean
    stop with no input whatsoever. This is the same defect as the `generate` path with the
    substitution made unconditional, and it is not reachable by any test of `generate`.

    Mutation this exists to catch:
        -   else self._map_finish_reason(...)
        +   else FinishReason.STOP
    """

    def handler(request: httpx.Request) -> httpx.Response:
        lines = [
            json.dumps({"message": {"content": "part"}, "done": False}),
            json.dumps(
                {
                    "message": {"content": ""},
                    "done": True,
                    "done_reason": "load",
                    "prompt_eval_count": 1,
                    "eval_count": 1,
                }
            ),
        ]
        return httpx.Response(200, text="\n".join(lines) + "\n")

    connector = OllamaConnector(http_client=_make_mock_client(handler))
    chunks: list[StreamChunk] = []
    async for chunk in connector.stream(
        LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),))
    ):
        chunks.append(chunk)

    finals = [c for c in chunks if c.finish_reason is not None]
    assert len(finals) == 1, chunks
    assert finals[0].finish_reason == FinishReason.UNKNOWN


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("end_turn", FinishReason.STOP),
        ("stop_sequence", FinishReason.STOP),
        ("max_tokens", FinishReason.LENGTH),
        ("tool_use", FinishReason.TOOL_CALLS),
        ("refusal", FinishReason.UNKNOWN),
        ("pause_turn", FinishReason.UNKNOWN),
        (None, FinishReason.UNKNOWN),
    ],
)
async def test_anthropic_reports_an_unrecognised_stop_reason_as_unknown(
    raw: str | None, expected: FinishReason
) -> None:
    """Only enumerated `stop_reason` values are claimed; the rest are `UNKNOWN`.

    Mutation this exists to catch:
        -   return FinishReason.UNKNOWN
        +   return FinishReason.STOP
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "m1",
                "model": "claude-3-5-sonnet",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": raw,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    connector = AnthropicConnector(api_key="k", http_client=_make_mock_client(handler))
    resp = await connector.generate(
        LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),))
    )
    assert resp.finish_reason == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("stop", FinishReason.STOP),
        ("length", FinishReason.LENGTH),
        ("tool_calls", FinishReason.TOOL_CALLS),
        ("content_filter", FinishReason.CONTENT_FILTER),
        ("function_call", FinishReason.UNKNOWN),
        (None, FinishReason.UNKNOWN),
    ],
)
async def test_openai_reports_an_unrecognised_finish_reason_as_unknown(
    raw: str | None, expected: FinishReason
) -> None:
    """Only enumerated `finish_reason` values are claimed; the rest are `UNKNOWN`.

    Mutation this exists to catch:
        -   return FinishReason.UNKNOWN
        +   return FinishReason.STOP
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o",
                "choices": [{"message": {"content": "ok"}, "finish_reason": raw}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    connector = OpenAIConnector(api_key="k", http_client=_make_mock_client(handler))
    resp = await connector.generate(
        LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),))
    )
    assert resp.finish_reason == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("STOP", FinishReason.STOP),
        ("MAX_TOKENS", FinishReason.LENGTH),
        ("SAFETY", FinishReason.CONTENT_FILTER),
        ("RECITATION", FinishReason.UNKNOWN),
        ("PROHIBITED_CONTENT", FinishReason.UNKNOWN),
        (None, FinishReason.UNKNOWN),
    ],
)
async def test_gemini_reports_an_unrecognised_finish_reason_as_unknown(
    raw: str | None, expected: FinishReason
) -> None:
    """PR #384 fixed this connector's request path and left this mapper untouched.

    Recorded because it is the counter-example to "the Gemini connector is done": the same
    substitution survived on the response side of the file #380 fixed, which is the shape
    of mistake #385 exists to stop repeating.

    Mutation this exists to catch:
        -   return FinishReason.UNKNOWN
        +   return FinishReason.STOP
    """
    candidate: dict[str, Any] = {"content": {"parts": [{"text": "ok"}], "role": "model"}}
    if raw is not None:
        candidate["finishReason"] = raw

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [candidate],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 2,
                },
                "modelVersion": "gemini-1.5-pro",
            },
        )

    connector = GeminiConnector(api_key="k", http_client=_make_mock_client(handler))
    resp = await connector.generate(
        LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),))
    )
    assert resp.finish_reason == expected


# --------------------------------------------------------------------------------------
# The credential class: fail at construction, not at send
# --------------------------------------------------------------------------------------

_KEYED_CONNECTORS: tuple[tuple[type[BaseLLMConnector], tuple[str, ...]], ...] = (
    (OpenAIConnector, ("OPENAI_API_KEY",)),
    (AnthropicConnector, ("ANTHROPIC_API_KEY",)),
    (GeminiConnector, ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
)


@pytest.mark.parametrize(("connector_cls", "env_vars"), _KEYED_CONNECTORS)
def test_a_keyed_connector_refuses_construction_with_no_credential(
    connector_cls: type[BaseLLMConnector],
    env_vars: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No key means the connector does not exist, rather than existing and failing at send.

    `api_key or os.getenv(...) or ""` produced a connector whose auth header was empty, so
    a **configuration** defect surfaced as a provider-side `401` on the first billed call —
    at a different time, in a different subsystem, wearing the clothes of a transport
    fault a caller's declared retry or failover path may legitimately re-attempt. The
    remedy for this class is therefore not the same as for the payload sites: the point is
    *where* it fails, not what value it sends.

    Mutation this exists to catch:
        -   resolved_key = api_key if api_key is not None else os.getenv(...)
        -   if resolved_key is None or not resolved_key.strip():
        -       raise LLMCredentialsNotConfiguredError(...)
        +   resolved_key = api_key or os.getenv(...) or ""
    """
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(LLMCredentialsNotConfiguredError) as excinfo:
        connector_cls()

    text = str(excinfo.value)
    assert connector_cls.__name__ in text, text
    for var in env_vars:
        assert var in text, text


@pytest.mark.parametrize(("connector_cls", "env_vars"), _KEYED_CONNECTORS)
def test_a_keyed_connector_refuses_a_blank_credential_as_well_as_a_missing_one(
    connector_cls: type[BaseLLMConnector],
    env_vars: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`api_key=""` and an environment variable set to blanks are refused too.

    `""` is the exact value the old expression substituted, so a guard that accepted it
    would leave the defect reachable through the one input most likely to produce it — an
    unset variable that a shell wrapper exported as empty.

    Mutation this exists to catch:
        -   if resolved_key is None or not resolved_key.strip():
        +   if resolved_key is None:
    """
    for var in env_vars:
        monkeypatch.setenv(var, "   ")

    with pytest.raises(LLMCredentialsNotConfiguredError):
        connector_cls()
    with pytest.raises(LLMCredentialsNotConfiguredError):
        connector_cls(api_key="")


@pytest.mark.parametrize(("connector_cls", "env_vars"), _KEYED_CONNECTORS)
def test_a_keyed_connector_reads_its_credential_from_the_environment(
    connector_cls: type[BaseLLMConnector],
    env_vars: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The environment fallback still works; only the `or ""` terminator is gone.

    Asserted so that "it refuses when there is no key" cannot be satisfied by a connector
    that refuses when there *is* one — the instrument has to observe the accepting case
    too (§6.9 case 2).
    """
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(env_vars[0], "env-key")

    assert connector_cls().api_key == "env-key"


@pytest.mark.asyncio
async def test_an_api_key_blanked_after_construction_is_refused_at_request_time() -> None:
    """`api_key` is public and mutable, so the `__init__` check is not the only guard.

    `_require_api_key` exists because the auth header dicts have to narrow `api_key` from
    `str | None` to `str`, and the way they did it was `self.api_key or ""` — the
    substitution itself. A `cast` would have silenced the type error while leaving an
    empty credential reachable by anything that assigns to the attribute after
    construction, which is what this asserts.

    Mutation this exists to catch:
        -   "Authorization": f"Bearer {self._require_api_key()}",
        +   "Authorization": f"Bearer {self.api_key or ''}",
    """
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers.get("authorization", ""))
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    connector = OpenAIConnector(api_key="k", http_client=_make_mock_client(handler))
    connector.api_key = ""

    with pytest.raises(LLMCredentialsNotConfiguredError) as excinfo:
        await connector.generate(
            LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),))
        )
    assert "api_key is ''" in str(excinfo.value), str(excinfo.value)
    # The request must not have been sent: an empty credential never reaches the provider.
    assert calls == []


# --------------------------------------------------------------------------------------
# `_require_api_key` — all six call sites (#397)
#
# PR #393 introduced `_require_api_key` and a reviewer's mutation run found
# it pinned at exactly one of its six call sites: `OpenAIConnector.generate`. Anthropic
# generate and stream, Gemini generate and stream, and OpenAI stream were unpinned, so
# reverting any of those five to `self.api_key or ""` shipped a green gate. It is the
# §6.9 coincidence case rather than a weak test — the one pinned site made the family
# look covered.
#
# The inventory test below is what keeps this true as the file changes: it reads the call
# sites out of the source and asserts the pinned set is the whole set, so a seventh site
# added without a pin fails rather than passing silently.
# --------------------------------------------------------------------------------------


def _openai_success(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "gpt-4o",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )


def _openai_stream_success(request: httpx.Request) -> httpx.Response:
    body = (
        "data: "
        + json.dumps({"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]})
        + "\n\n"
        + "data: "
        + json.dumps(
            {
                "model": "gpt-4o",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )
        + "\n\ndata: [DONE]\n\n"
    )
    return httpx.Response(200, text=body)


def _anthropic_success(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "model": "claude-3-5-sonnet",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )


def _anthropic_stream_success(request: httpx.Request) -> httpx.Response:
    body = (
        "data: "
        + json.dumps({"type": "message_start", "message": {"usage": {"input_tokens": 1}}})
        + "\n\n"
        + "data: "
        + json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "ok"}})
        + "\n\n"
        + "data: "
        + json.dumps(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1},
            }
        )
        + "\n\n"
    )
    return httpx.Response(200, text=body)


def _gemini_success(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [
                {"content": {"parts": [{"text": "ok"}], "role": "model"}, "finishReason": "STOP"}
            ],
            "usageMetadata": {
                "promptTokenCount": 1,
                "candidatesTokenCount": 1,
                "totalTokenCount": 2,
            },
            "modelVersion": "gemini-1.5-pro",
        },
    )


def _gemini_stream_success(request: httpx.Request) -> httpx.Response:
    body = (
        "data: "
        + json.dumps(
            {
                "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
            }
        )
        + "\n\n"
    )
    return httpx.Response(200, text=body)


async def _drive_generate(connector: BaseLLMConnector) -> None:
    await connector.generate(
        LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),))
    )


async def _drive_stream(connector: BaseLLMConnector) -> None:
    # `stream` is an async generator, so the credential guard fires on the first
    # `__anext__` rather than at the call. Iterating is what makes the assertion real:
    # a test that only *called* `stream` would pass against a connector that never
    # checks anything (§6.9 case 2 — assert the instrument observed what it claims).
    async for _chunk in connector.stream(
        LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),))
    ):
        pass


_CREDENTIAL_PIN_SITES: tuple[
    tuple[
        str,
        type[BaseLLMConnector],
        Callable[[httpx.Request], httpx.Response],
        Callable[[BaseLLMConnector], Any],
    ],
    ...,
] = (
    # OpenAI builds its header in one place, `_auth_headers`, which both `generate` and
    # `stream` call — so the family has one call site behind two doors, and both doors are
    # still driven below. The anthropic and gemini connectors still build theirs inline,
    # one site per door (#1304).
    (
        "openai.OpenAIConnector._auth_headers",
        OpenAIConnector,
        _openai_success,
        _drive_generate,
    ),
    (
        "openai.OpenAIConnector._auth_headers",
        OpenAIConnector,
        _openai_stream_success,
        _drive_stream,
    ),
    (
        "anthropic.AnthropicConnector.generate",
        AnthropicConnector,
        _anthropic_success,
        _drive_generate,
    ),
    (
        "anthropic.AnthropicConnector.stream",
        AnthropicConnector,
        _anthropic_stream_success,
        _drive_stream,
    ),
    ("gemini.GeminiConnector.generate", GeminiConnector, _gemini_success, _drive_generate),
    ("gemini.GeminiConnector.stream", GeminiConnector, _gemini_stream_success, _drive_stream),
)


@pytest.mark.parametrize(
    ("site", "connector_cls", "handler", "drive"),
    _CREDENTIAL_PIN_SITES,
    # Two cases share the `_auth_headers` site, so the drive is part of the id.
    ids=[f"{entry[0]}-{entry[3].__name__}" for entry in _CREDENTIAL_PIN_SITES],
)
@pytest.mark.asyncio
async def test_a_credential_blanked_after_construction_is_refused_at_every_request_door(
    site: str,
    connector_cls: type[BaseLLMConnector],
    handler: Callable[[httpx.Request], httpx.Response],
    drive: Callable[[BaseLLMConnector], Any],
) -> None:
    """Every auth-header site refuses a blanked credential instead of sending an empty one.

    `api_key` is public and mutable, so the `__init__` check is not the only door: anything
    that assigns to the attribute after construction reaches the header builders directly.
    Every header builder calls `_require_api_key()`; five of the six doors had no test, and a
    revert of any of the five shipped green.

    Six doors, five call sites: OpenAI's `generate` and `stream` both build their header
    through `_auth_headers`, so that family has one site (#1304). The parametrisation stays
    per-door, because what a caller reaches is a door — a seam shared by two of them is a
    detail of the connector, not a reason to stop driving one of them.

    Mutation this exists to catch, at each of the sites named in
    `_CREDENTIAL_PIN_SITES` (`"Authorization"` / `"x-api-key"` / `"x-goog-api-key"`):
        -   self._require_api_key()
        +   self.api_key or ""

    Two assertions, not one. The refusal alone would be satisfied by a connector that
    refuses unconditionally, so the accepting case is driven through the *same* site with
    the *same* handler and must reach the provider (§6.9 case 2, and the coincidence case:
    an instrument has to observe both outcomes to distinguish them).
    """
    seen: list[str] = []

    def recording_handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return handler(request)

    refusing = connector_cls(api_key="k", http_client=_make_mock_client(recording_handler))
    refusing.api_key = ""

    with pytest.raises(LLMCredentialsNotConfiguredError) as excinfo:
        await drive(refusing)

    text = str(excinfo.value)
    assert connector_cls.__name__ in text, (site, text)
    assert "api_key is ''" in text, (site, text)
    # An empty credential never reaches the wire: the provider would answer 401 and the
    # caller would read a configuration defect as a transport fault.
    assert seen == [], (site, seen)

    accepting = connector_cls(api_key="k", http_client=_make_mock_client(recording_handler))
    await drive(accepting)
    assert len(seen) == 1, (site, seen)


@pytest.mark.parametrize(
    "blank",
    (None, "", "   ", "\t\n"),
    ids=("none", "empty", "spaces", "whitespace"),
)
@pytest.mark.asyncio
async def test_every_shape_of_absent_credential_is_refused_not_only_the_empty_string(
    blank: str | None,
) -> None:
    """`None` and whitespace-only are refused as well as `""`.

    `self.api_key or ""` conflated all four into an empty header. A guard written as
    `if key is None` would accept `"   "`, and a shell wrapper exporting an unset variable
    is the likeliest producer of exactly that.

    Mutation this exists to catch:
        -   if key is None or not key.strip():
        +   if key is None:
    """
    connector = OpenAIConnector(api_key="k", http_client=_make_mock_client(_openai_success))
    connector.api_key = blank
    with pytest.raises(LLMCredentialsNotConfiguredError):
        await _drive_generate(connector)


def _discover_require_api_key_sites() -> set[str]:
    """Every `self._require_api_key()` call in the connector package, as `module.Class.method`.

    Read out of the source rather than listed by hand, so the inventory cannot drift from
    the code it claims to enumerate. The caller asserts a non-zero file count before
    comparing, so a broken path cannot deliver an empty set that the comparison would then
    have to be wrong to reject (§6.9 case 2, `scanned > 50` worked example).
    """
    from pathlib import Path

    import uclone_x.llm.connectors as connectors_pkg

    package_dir = Path(str(connectors_pkg.__file__)).parent
    sites: set[str] = set()
    scanned = 0
    for path in sorted(package_dir.glob("*.py")):
        scanned += 1
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for cls in ast.walk(tree):
            if not isinstance(cls, ast.ClassDef):
                continue
            for func in cls.body:
                if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                for node in ast.walk(func):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "_require_api_key"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "self"
                    ):
                        sites.add(f"{path.stem}.{cls.name}.{func.name}")
    assert scanned >= 6, f"the sweep scanned {scanned} connector modules, so it read nothing"
    return sites


def test_every_require_api_key_call_site_is_named_by_a_pin() -> None:
    """The pinned set is the whole set, so a new call site cannot arrive unpinned.

    This is the assertion that makes the six parametrized cases above checkable rather
    than merely present. a reviewer's finding was not that a test was weak
    but that a family of six was covered at one member, and nothing in the suite said how
    many members there were. Now something does.

    The count is five, not six, because OpenAI's two request doors share `_auth_headers`.
    A subclass of `OpenAIConnector` that varies only its identity and its endpoint (the
    vLLM connector, #1304) inherits that seam and adds no site, which is the point of
    having it: a new OpenAI-compatible provider cannot introduce an unpinned credential
    door without editing the base class.

    The control is inside `_discover_require_api_key_sites`: it asserts it scanned at least
    six modules before returning, positioned so a broken glob cannot reach the emptiness
    that would otherwise satisfy a subset comparison.
    """
    discovered = _discover_require_api_key_sites()
    pinned = {site for site, _cls, _handler, _drive in _CREDENTIAL_PIN_SITES}

    assert discovered == pinned, (
        f"unpinned call sites: {sorted(discovered - pinned)}; "
        f"pins naming no call site: {sorted(pinned - discovered)}"
    )
    assert len(discovered) == 5, sorted(discovered)


# --------------------------------------------------------------------------------------
# F1 — `generate` and `stream` agree on finish-reason mapping (#397)
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_generate_and_stream_agree_that_an_empty_finish_reason_is_unknown() -> None:
    """`""` maps to `UNKNOWN` on both paths; only `null` leaves a chunk making no claim.

    A truthiness guard (`if raw_fr:`) sat in front of the mapper on the stream path, so a
    `finish_reason` of `""` became `None` there while `generate` mapped it to `UNKNOWN`.
    Two code paths answered differently about the same response property, and the stream
    path was the one that reverted to the pre-#385 shape of reporting nothing.

    `is not None` rather than dropping the guard: OpenAI sends `"finish_reason": null` on
    every intermediate chunk, and mapping that to `UNKNOWN` would stamp a finish reason on
    chunks that report none — the same substitution in the other direction.

    Mutation this exists to catch:
        -   if raw_fr is not None:
        +   if raw_fr:
    """

    def generate_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o",
                "choices": [{"message": {"content": "ok"}, "finish_reason": ""}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    def stream_handler(request: httpx.Request) -> httpx.Response:
        body = (
            "data: "
            + json.dumps({"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]})
            + "\n\ndata: "
            + json.dumps({"choices": [{"delta": {}, "finish_reason": ""}]})
            + "\n\ndata: [DONE]\n\n"
        )
        return httpx.Response(200, text=body)

    resp = await OpenAIConnector(
        api_key="k", http_client=_make_mock_client(generate_handler)
    ).generate(LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),)))
    assert resp.finish_reason is FinishReason.UNKNOWN

    chunks: list[StreamChunk] = []
    async for chunk in OpenAIConnector(
        api_key="k", http_client=_make_mock_client(stream_handler)
    ).stream(LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),))):
        chunks.append(chunk)

    reported = [c.finish_reason for c in chunks if c.finish_reason is not None]
    assert reported == [FinishReason.UNKNOWN], [c.finish_reason for c in chunks]
    # The `null` chunk still makes no claim, which is what `is not None` buys over
    # dropping the guard entirely.
    assert chunks[0].finish_reason is None, chunks[0]


@pytest.mark.asyncio
async def test_anthropic_generate_and_stream_agree_that_an_empty_stop_reason_is_unknown() -> None:
    """`stop_reason=""` maps to `UNKNOWN` on both paths.

    Mutation this exists to catch:
        -   if stop_reason is not None:
        +   if stop_reason:
    """

    def generate_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-3-5-sonnet",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    def stream_handler(request: httpx.Request) -> httpx.Response:
        body = (
            "data: "
            + json.dumps({"type": "message_start", "message": {"usage": {"input_tokens": 1}}})
            + "\n\ndata: "
            + json.dumps(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": ""},
                    "usage": {"output_tokens": 1},
                }
            )
            + "\n\n"
        )
        return httpx.Response(200, text=body)

    resp = await AnthropicConnector(
        api_key="k", http_client=_make_mock_client(generate_handler)
    ).generate(LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),)))
    assert resp.finish_reason is FinishReason.UNKNOWN

    reported: list[FinishReason] = []
    async for chunk in AnthropicConnector(
        api_key="k", http_client=_make_mock_client(stream_handler)
    ).stream(LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),))):
        if chunk.finish_reason is not None:
            reported.append(chunk.finish_reason)
    assert reported == [FinishReason.UNKNOWN], reported


@pytest.mark.asyncio
async def test_gemini_generate_and_stream_agree_that_an_empty_finish_reason_is_unknown() -> None:
    """`finishReason=""` maps to `UNKNOWN` on both paths.

    Mutation this exists to catch:
        -   if raw_fr is not None:
        +   if raw_fr:
    """

    def generate_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": "ok"}], "role": "model"}, "finishReason": ""}
                ],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
                "modelVersion": "gemini-1.5-pro",
            },
        )

    def stream_handler(request: httpx.Request) -> httpx.Response:
        body = (
            "data: "
            + json.dumps(
                {
                    "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": ""}],
                    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
                }
            )
            + "\n\n"
        )
        return httpx.Response(200, text=body)

    resp = await GeminiConnector(
        api_key="k", http_client=_make_mock_client(generate_handler)
    ).generate(LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),)))
    assert resp.finish_reason is FinishReason.UNKNOWN

    reported: list[FinishReason] = []
    async for chunk in GeminiConnector(
        api_key="k", http_client=_make_mock_client(stream_handler)
    ).stream(LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),))):
        if chunk.finish_reason is not None:
            reported.append(chunk.finish_reason)
    assert reported == [FinishReason.UNKNOWN], reported


@pytest.mark.asyncio
async def test_ollama_generate_and_stream_agree_that_an_empty_done_reason_is_unknown() -> None:
    """The connector #385 already fixed, asserted here so all four are covered by name.

    Without this the "all four agree" claim would rest on three measurements and an
    assumption about the fourth.
    """

    def generate_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "llama3",
                "message": {"content": "ok"},
                "done": True,
                "done_reason": "",
                "prompt_eval_count": 1,
                "eval_count": 1,
            },
        )

    def stream_handler(request: httpx.Request) -> httpx.Response:
        body = (
            json.dumps(
                {
                    "model": "llama3",
                    "message": {"content": "ok"},
                    "done": True,
                    "done_reason": "",
                    "prompt_eval_count": 1,
                    "eval_count": 1,
                }
            )
            + "\n"
        )
        return httpx.Response(200, text=body)

    resp = await OllamaConnector(http_client=_make_mock_client(generate_handler)).generate(
        LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),))
    )
    assert resp.finish_reason is FinishReason.UNKNOWN

    reported: list[FinishReason] = []
    async for chunk in OllamaConnector(http_client=_make_mock_client(stream_handler)).stream(
        LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),))
    ):
        if chunk.finish_reason is not None:
            reported.append(chunk.finish_reason)
    assert reported == [FinishReason.UNKNOWN], reported


# --------------------------------------------------------------------------------------
# F3 — `create_llm_connector` no longer converts a refusal into a mock (#397)
# --------------------------------------------------------------------------------------


def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize(
    ("env_var", "provider"),
    (
        ("OPENAI_API_KEY", "openai"),
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("GEMINI_API_KEY", "gemini"),
    ),
)
def test_a_missing_credential_is_not_convertible_into_a_mock_even_with_fallback_requested(
    env_var: str,
    provider: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`fallback_to_mock=True` does not turn a credential refusal into a `MockLLMConnector`.

    This is the reachable form of the defect, and it needs no explicit provider name: a
    credential variable set to blanks is *truthy*, so auto-detection selects the provider,
    the connector refuses to construct, and the removed `except Exception` returned a mock
    with a `logger.warning`. The caller asked for a real provider and received fabricated
    completions.

    A log line is not in-band attribution, and `MockLLMConnector` cannot supply the
    in-band form P6 requires: it stamps `path=PRIMARY`, `requested == served_by`, both
    naming `mock`, and `degraded=False` — so it overwrites the record of what was
    requested rather than merely omitting the substitution.

    Mutation this exists to catch:
        -   (no handler)
        +   except Exception as exc:
        +       if fallback_to_mock:
        +           logger.warning(...)
        +           return MockLLMConnector(**kwargs)
    """
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv(env_var, "   ")

    for flag in (True, False):
        with pytest.raises(LLMCredentialsNotConfiguredError):
            create_llm_connector(fallback_to_mock=flag)

    # And with the provider named explicitly, which is the path a CLI `--provider` takes.
    for flag in (True, False):
        with pytest.raises(LLMCredentialsNotConfiguredError):
            create_llm_connector(provider=provider, fallback_to_mock=flag)


@pytest.mark.parametrize("provider", ("openai", "anthropic", "gemini", "google"))
def test_the_factory_raises_the_class_the_connector_raised_not_its_sibling(
    provider: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing credential surfaces as `LLMCredentialsNotConfiguredError`, not `LLMProviderError`.

    `LLMCredentialsNotConfiguredError` and `LLMProviderError` are **siblings** under
    `LLMError`, not parent and child. Connector construction must propagate
    `LLMCredentialsNotConfiguredError` unaltered without re-raising or converting it to
    its sibling `LLMProviderError`.

    The sibling relationship is asserted here rather than assumed, because the whole
    finding rests on it and an `issubclass` reading either way changes what the test means.

    Mutation this exists to catch:
        -   return OpenAIConnector(api_key=api_key, base_url=base_url, **kwargs)
        +   if resolved_provider == "openai" and not key:
        +       raise LLMProviderError("... is required for 'openai' provider")
    """
    assert issubclass(LLMCredentialsNotConfiguredError, LLMError)
    assert issubclass(LLMProviderError, LLMError)
    assert not issubclass(LLMCredentialsNotConfiguredError, LLMProviderError)
    assert not issubclass(LLMProviderError, LLMCredentialsNotConfiguredError)

    _clear_provider_env(monkeypatch)

    with pytest.raises(LLMCredentialsNotConfiguredError) as excinfo:
        create_llm_connector(provider=provider, fallback_to_mock=False)
    assert not isinstance(excinfo.value, LLMProviderError)
    # The message still names the variables the connector consulted, which is the only
    # thing the removed pre-check contributed.
    text = str(excinfo.value)
    assert "_API_KEY" in text, text


def test_an_unsupported_provider_name_is_refused_naming_the_value_not_replaced_by_a_mock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unmappable provider name fails loudly whatever `fallback_to_mock` says.

    A mock returned here answers a request the factory could not understand, which is a
    substituted implementation rather than the unconfigured-environment default. The
    refusal names the offending value so the caller can see which string was rejected —
    a typo in `LLM_PROVIDER` was previously indistinguishable from a working mock setup.

    Mutation this exists to catch:
        -   raise LLMProviderError(f"Unsupported LLM provider: {resolved_provider}")
        +   if fallback_to_mock:
        +       return MockLLMConnector(...)
        +   raise LLMProviderError(f"Unsupported LLM provider: {resolved_provider}")
    """
    _clear_provider_env(monkeypatch)

    for flag in (True, False):
        with pytest.raises(LLMProviderError, match="Unsupported LLM provider") as excinfo:
            create_llm_connector(provider="opnai", fallback_to_mock=flag)
        assert "opnai" in str(excinfo.value), str(excinfo.value)

    # Reached through the environment as well, not only the argument.
    monkeypatch.setenv("LLM_PROVIDER", "opnai")
    with pytest.raises(LLMProviderError, match="Unsupported LLM provider"):
        create_llm_connector(fallback_to_mock=True)


def test_fallback_to_mock_survives_only_where_nothing_was_requested_and_nothing_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one surviving site: no provider named, no credential present, nothing failed.

    `fallback_to_mock` is not a fallback here. No operation failed and no value is
    substituted for one a failed operation should have produced: the environment names no
    provider and holds no credential, so the flag selects in advance which connector to
    build in place of the `OllamaConnector` default. Both outcomes are asserted, so
    "returns a mock" cannot be satisfied by a factory that always returns one.

    A caller wanting a mock unconditionally already has an honest door —
    `create_llm_connector(provider="mock")` — which is a *request* for a mock rather than
    a substitution for something else, and is asserted here beside it.
    """
    _clear_provider_env(monkeypatch)

    assert isinstance(create_llm_connector(fallback_to_mock=True), MockLLMConnector)
    # Without the flag the unconfigured case is now refused rather than defaulted (#533).
    # The flag therefore selects between "a mock, deliberately" and "an error", which is a
    # narrower and more honest choice than the "a mock or a dead Ollama connector" it made
    # before.
    with pytest.raises(LLMProviderNotConfiguredError):
        create_llm_connector(fallback_to_mock=False)
    with pytest.raises(LLMProviderNotConfiguredError):
        create_llm_connector()
    assert isinstance(
        create_llm_connector(provider="mock", fallback_to_mock=False), MockLLMConnector
    )

    # A present credential is honoured over the flag: the flag is the unconfigured-case
    # default, not an override, so it cannot shadow a provider the caller did configure.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    assert isinstance(create_llm_connector(fallback_to_mock=True), OpenAIConnector)


def test_the_factory_has_no_broad_exception_handler_and_imports_no_logger() -> None:
    """`except Exception` is gone from the factory, rather than narrowed.

    The handler was not recovering from anything: it caught construction errors, replaced
    an implementation or an exception class, and logged. There is no failure class at this
    site that a substitution repairs, so the breadth was removed rather than justified —
    every branch either constructs a connector or raises, and a connector's own error
    reaches the caller unaltered.

    The `logging` import is asserted absent too, because the `logger.warning` was the only
    thing making the substitution look accounted for. This is a source-level assertion for
    a source-level property; it is not a substitute for the behavioural tests above, which
    is why both exist.
    """
    from pathlib import Path

    import uclone_x.llm.connectors.factory as factory_module

    source = Path(str(factory_module.__file__)).read_text(encoding="utf-8")
    assert "def create_llm_connector" in source, "the sweep did not read the factory"

    tree = ast.parse(source)
    handlers = [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)]
    assert handlers == [], [ast.unparse(h) for h in handlers]

    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "logging" not in imported, sorted(imported)


# --------------------------------------------------------------------------------------
# F5 — the substitution sweep, with a control (#397)
# --------------------------------------------------------------------------------------


def _literal_names(tree: ast.AST) -> set[str]:
    """Names bound to a literal, e.g. `ANTHROPIC_REQUIRED_MAX_TOKENS_DEFAULT = 4096`.

    Collected so that `x or SOME_CONSTANT` is detected as the substitution it is. Naming a
    literal is the right way to make a site findable by symbol (#137), and it must not
    also be a way to make it invisible to the detector — which is exactly the divergence
    between the evidence and the fact that F5 is about, arriving a second time through the
    repair for the first. Measured: before this, mutating the `max_tokens` site to
    `request.max_tokens or ANTHROPIC_REQUIRED_MAX_TOKENS_DEFAULT` was caught only by
    `test_anthropic_sends_max_tokens_zero_as_zero_rather_than_4096` and **not** by the
    sweep below.

    Detects uppercase, lowercase, and mixed-case names bound to literals at both module
    scope and function scope (#402, #410).
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value = list(node.targets), node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        if value is None or not _is_literal(value, frozenset()):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _is_literal(node: ast.expr, literal_names: frozenset[str]) -> bool:
    """Whether `node` is a literal, a display literal of literals, or a named literal.

    `[{"text": ""}]` and `{}` count: #384's sixth removed site was
    `parts if parts else [{"text": ""}]`, which both of PR #393's stated text patterns
    missed, so a detector recognising only bare constants would inherit that blind spot.
    Names in `literal_names` count for the reason given in `_literal_names`.
    `None` is excluded because initializing an optional variable to `None` does not define
    a fallback default value.

    `self.<attr>` counts on the same terms, because a class attribute bound to a literal is
    a named literal reached through an instance. Without this, moving
    `'https://api.openai.com/v1'` from the `or` expression into a `ClassVar` and writing
    `or self._default_base_url` would have removed the site from the sweep while the
    literal default stood — the same divergence between the evidence and the fact that
    `_literal_names` was written for, arriving a third time through a refactor (#1304).
    """
    if isinstance(node, ast.Constant):
        return node.value is not None
    if isinstance(node, ast.Name):
        return node.id in literal_names
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return node.attr in literal_names
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return all(_is_literal(element, literal_names) for element in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            key is not None and _is_literal(key, literal_names) for key in node.keys
        ) and all(_is_literal(value, literal_names) for value in node.values)
    return False


def _literal_substitution_sites(source: str) -> list[str]:
    """`<expr> or <literal>` occurrences in `source`, as unparsed expressions.

    Covered substitution shapes:
    - Direct literal defaults: `<expr> or <literal>` (e.g. `x or 4096`, `x or ""`, `x or {}`)
    - Module-level named literal constants (uppercase, lowercase, MixedCase)
    - Function-scope named literal constants (uppercase, lowercase, MixedCase)
    - Conditional defaults with identical test and body: `<expr> if <expr> else <literal>`
      (both direct literals and named constants at module or function scope)
    - Class attributes bound to a literal, reached as `self.<attr>` (#1304)

    Documented blind spots (shapes the AST sweep does NOT cover):
    - `X if cond else <literal>` (where cond is not semantically equivalent to X)
    - `dict.get("k", <literal>)`
    - Parameter defaults in function signatures: `def f(param=<literal>)`
    - Multi-statement conditional assignment (e.g. `if not x: x = <literal>`)

    An AST detector rather than a text pattern, because the site this replaces —
    `request.max_tokens or 4096` — was rewritten into a conditional expression and became
    invisible to PR #393's `or <literal>` *text* search, so re-running that search reported
    the class absent while a literal default stood. A text pattern is defeated by
    reformatting; the AST is not.
    """
    tree = ast.parse(source)
    literal_names = frozenset(_literal_names(tree))
    sites: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.BoolOp)
            and isinstance(node.op, ast.Or)
            and _is_literal(node.values[-1], literal_names)
        ):
            sites.append(ast.unparse(node))
        elif (
            isinstance(node, ast.IfExp)
            and _is_literal(node.orelse, literal_names)
            and ast.unparse(node.test) == ast.unparse(node.body)
        ):
            sites.append(ast.unparse(node))
    return sites


def test_the_substitution_detector_matches_its_known_instances_before_reporting_absence() -> None:
    """The control. A negative finding from a blind instrument carries no information.

    #385's deliverable was *the absence* of a pattern, evidenced by a search — and this is
    the case where the evidence and the fact diverged: the `max_tokens` site was rewritten
    into a conditional expression, so the search stopped matching it while the literal
    default remained. §5.5's third unpinnable shape is exactly this, and the repair is a
    control proving the detector matches known instances first.

    Every string below is a form #380/#384/#385 actually removed from this package, so the
    control is not a synthetic pattern chosen to match the detector. The `[{"text": ""}]`
    case is #384's sixth site, which both of PR #393's stated text patterns missed.
    """
    known_instances = (
        # 1) Direct literal substitution (`X or <literal>`)
        'x = {"content": msg.content or ""}',
        "payload = request.max_tokens or 4096",
        'headers = {"Authorization": self.api_key or ""}',
        'y = {"tool_use_id": msg.tool_call_id or ""}',
        'parts = parts or [{"text": ""}]',
        "opts = extra or {}",
        "n = count or 0",
        # 2) Module-level named literal (uppercase, lowercase, MixedCase)
        "LIMIT = 4096\nv = request.max_tokens or LIMIT",
        "limit = 4096\nv = request.max_tokens or limit",
        "LimitVal = 4096\nv = request.max_tokens or LimitVal",
        # 3) Function-scope named literal (both uppercase and lowercase/MixedCase)
        "def f(request):\n    LIMIT = 4096\n    return request.max_tokens or LIMIT",
        "def f(request):\n    limit = 4096\n    return request.max_tokens or limit",
        "def f(request):\n    LimitVal = 4096\n    return request.max_tokens or LimitVal",
        # 4) Conditional substitution (`X if X else <literal>`)
        "v = request.max_tokens if request.max_tokens else 4096",
        "LIMIT = 4096\nv = request.max_tokens if request.max_tokens else LIMIT",
        "limit = 4096\nv = request.max_tokens if request.max_tokens else limit",
        "def f(request):\n    limit = 4096\n    return request.max_tokens if request.max_tokens else limit",
        # 5) Class attribute bound to a literal, reached through `self` (#1304)
        "class C:\n    LIMIT = 4096\n    def f(self, request):\n        return request.max_tokens or self.LIMIT",
        'class C:\n    _default: ClassVar[str] = "x"\n    def f(self, v):\n        return v or self._default',
    )
    for source in known_instances:
        assert _literal_substitution_sites(source), f"detector blind to: {source}"

    # And it does not fire on the forms that replaced them, or it would report every fix
    # as the defect and its absence would mean nothing either.
    for clean in (
        "v = request.max_tokens if request.max_tokens is not None else 4096",
        "v = ANTHROPIC_REQUIRED_MAX_TOKENS_DEFAULT",
        "LIMIT = 4096\nv = request.max_tokens if request.max_tokens is not None else LIMIT",
        "limit = 4096\nv = request.max_tokens if request.max_tokens is not None else limit",
        # A name that is *not* a literal stays uninteresting, or the detector
        # would fire on every `a or b`.
        "def f(b):\n    return a or b",
        "v = a or b",
        "v = a or compute()",
        "v = a or os.getenv('X')",
        # Initializing an optional variable to None does not count as a literal default
        "finish_reason: str | None = None\nif delta_content or tool_calls or usage or finish_reason: pass",
        # Conditional with non-identical test and body is a documented blind spot
        "v = a if condition else 4096",
        # An attribute whose name is not bound to a literal anywhere stays uninteresting,
        # or the detector would fire on `x or self.<anything>`.
        "class C:\n    def f(self, v):\n        return v or self._client",
    ):
        assert not _literal_substitution_sites(clean), f"detector fires on clean form: {clean}"


# Every `<expr> or <literal>` the detector finds in the connector package, each with the
# reason it is not the class #380/#384/#385 removed. Asserted in **both** directions
# below: an unlisted site fails, and a listed site that has disappeared fails too, so the
# list cannot quietly outlive the code it excuses.
_ALLOWED_LITERAL_DEFAULTS: dict[str, dict[str, str]] = {
    "openai.py": {
        "base_url or os.getenv(self._base_url_env_var) or self._default_base_url": "A provider's own documented public endpoint, resolved from configuration. Nothing has failed at this point and no result is being substituted: this is which URL to call, decided before any call is made. The endpoint itself is a ClassVar since #1304, so a subclass serving an OpenAI-compatible endpoint cannot silently inherit api.openai.com."
    },
    "anthropic.py": {
        "base_url or os.getenv('ANTHROPIC_BASE_URL') or 'https://api.anthropic.com/v1'": "A provider's own documented public endpoint, resolved from configuration. Nothing has failed at this point and no result is being substituted: this is which URL to call, decided before any call is made."
    },
    "gemini.py": {
        "base_url or os.getenv('GEMINI_BASE_URL') or "
        "'https://generativelanguage.googleapis.com/v1beta'": "A provider's own documented public endpoint, resolved from configuration. Nothing has failed at this point and no result is being substituted: this is which URL to call, decided before any call is made."
    },
    "mock.py": {
        "resp.content or ''": '`MockLLMConnector` fabricates its whole response by construction — that is what it is for, and it is reached only when a caller asks for `provider="mock"` or declares the unconfigured-environment default. It is a *requested* mock, never substituted for a provider that failed (#397). The two sites are its own echo of the prompt.',
        "request.messages[-1].content or ''": '`MockLLMConnector` fabricates its whole response by construction — that is what it is for, and it is reached only when a caller asks for `provider="mock"` or declares the unconfigured-environment default. It is a *requested* mock, never substituted for a provider that failed (#397). The two sites are its own echo of the prompt.',
    },
}


def test_no_connector_substitutes_a_literal_for_a_missing_value() -> None:
    """Every `<expr> or <literal>` in the connector package is one of the allowed defaults.

    Runs after the control above has established the detector is not blind, and asserts
    its own file count before comparing, so a broken glob cannot produce the emptiness
    this test is looking for (§6.9 case 2, `scanned > 50` worked example).

    `ANTHROPIC_REQUIRED_MAX_TOKENS_DEFAULT` is not on the list because the detector no
    longer sees it: it is a named constant behind an `is not None` conditional, carrying
    its own rationale, which is what makes it findable by symbol rather than by a text
    pattern reformatting defeats (#137).
    """
    from pathlib import Path

    import uclone_x.llm.connectors as connectors_pkg

    # Assert reasons
    for file_name, sites in _ALLOWED_LITERAL_DEFAULTS.items():
        for site, reason in sites.items():
            assert isinstance(reason, str) and reason.strip(), (
                f"Empty reason for {file_name}:{site}"
            )

    package_dir = Path(str(connectors_pkg.__file__)).parent
    scanned: list[str] = []
    found: dict[str, tuple[str, ...]] = {}
    for path in sorted(package_dir.glob("*.py")):
        scanned.append(path.name)
        sites = tuple(_literal_substitution_sites(path.read_text(encoding="utf-8")))
        if sites:
            found[path.name] = sites

    assert len(scanned) >= 7, f"the sweep scanned {scanned}, so it read almost nothing"
    for expected_file in ("factory.py", "anthropic.py", "openai.py", "gemini.py", "ollama.py"):
        assert expected_file in scanned, (expected_file, scanned)

    unlisted = {
        name: [site for site in sites if site not in _ALLOWED_LITERAL_DEFAULTS.get(name, {})]
        for name, sites in found.items()
    }
    assert {name: sites for name, sites in unlisted.items() if sites} == {}, unlisted

    # The other direction: an allow-list entry naming a site that no longer exists is a
    # stale excuse, and leaving it would let a future reintroduction land pre-approved.
    stale = {
        name: [site for site in sites if site not in found.get(name, ())]
        for name, sites in _ALLOWED_LITERAL_DEFAULTS.items()
    }
    assert {name: sites for name, sites in stale.items() if sites} == {}, stale


@pytest.mark.parametrize(
    "violating_source",
    (
        "payload = request.max_tokens or 4096",
        "LIMIT = 4096\nv = request.max_tokens or LIMIT",
        "limit = 4096\nv = request.max_tokens or limit",
        "LimitVal = 4096\nv = request.max_tokens or LimitVal",
        "def f(request):\n    LIMIT = 4096\n    return request.max_tokens or LIMIT",
        "def f(request):\n    limit = 4096\n    return request.max_tokens or limit",
        "def f(request):\n    LimitVal = 4096\n    return request.max_tokens or LimitVal",
        "v = request.max_tokens if request.max_tokens else 4096",
        "LIMIT = 4096\nv = request.max_tokens if request.max_tokens else LIMIT",
        "limit = 4096\nv = request.max_tokens if request.max_tokens else limit",
        "def f(request):\n    limit = 4096\n    return request.max_tokens if request.max_tokens else limit",
    ),
)
def test_each_violating_shape_fails_the_substitution_sweep(violating_source: str) -> None:
    """Every detected shape produces an unlisted site that fails the sweep when present in a connector."""
    sites = _literal_substitution_sites(violating_source)
    assert sites, f"sweep detector is blind to violating shape: {violating_source}"
    unlisted_sites = [
        site for site in sites if site not in _ALLOWED_LITERAL_DEFAULTS.get("ollama.py", {})
    ]
    assert unlisted_sites == sites


# ======================================================================================
# Ollama: an assistant turn whose content IS a tool call (#385 over-applied)
# ======================================================================================


async def _ollama_sent_body(*messages: ChatMessage) -> dict[str, Any]:
    """The whole request body Ollama actually receives for `messages`."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5-coder:14b",
                "message": {"role": "assistant", "content": "ok"},
                "done": True,
                "done_reason": "stop",
            },
        )

    connector = OllamaConnector(http_client=_make_mock_client(handler))
    await connector.generate(LLMRequest(model="qwen2.5-coder:14b", messages=messages))
    return cast(dict[str, Any], captured["body"])


@pytest.mark.asyncio
async def test_ollama_sends_an_assistant_tool_call_turn_with_no_content() -> None:
    """A tool-call-only assistant turn is representable and must not be refused.

    `BaseAgent` stores such a turn as `content=resp_content or None` with `tool_calls`
    set, so this message is replayed on every later turn of the session. Refusing it
    made one tool call poison the conversation permanently: every subsequent turn failed
    here, before any request was made, in ~15ms with
    `served_by=agent.core:error_handler`. `openai.py`, `anthropic.py` and `gemini.py`
    all accept this shape; only this connector refused it.

    Mutation this exists to catch: widen the guard back to an unconditional
    `if msg.content is None: raise`.
    """
    body = await _ollama_sent_body(
        ChatMessage(role=MessageRole.USER, content="find the news"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=(ToolCallRequest(id="c1", name="web_search", arguments={"q": "news"}),),
        ),
        ChatMessage(role=MessageRole.USER, content="where did you find it?"),
    )
    assistant = body["messages"][1]
    assert "content" not in assistant, assistant
    assert assistant["tool_calls"] == [
        {"function": {"name": "web_search", "arguments": {"q": "news"}}}
    ]


@pytest.mark.asyncio
async def test_ollama_distinguishes_absent_content_from_the_empty_string() -> None:
    """Omitting the key is what keeps #385's distinction true on the wire.

    Coercing the absent case to `""` would satisfy the test above while re-introducing
    exactly the substitution #385 removed, so the two shapes are asserted to differ.

    Mutation this exists to catch: `m_dict["content"] = msg.content or ""`.
    """
    call = (ToolCallRequest(id="c1", name="t", arguments={}),)
    absent = await _ollama_sent_body(
        ChatMessage(role=MessageRole.ASSISTANT, content=None, tool_calls=call)
    )
    empty = await _ollama_sent_body(
        ChatMessage(role=MessageRole.ASSISTANT, content="", tool_calls=call)
    )
    assert "content" not in absent["messages"][0]
    assert empty["messages"][0]["content"] == ""
    assert absent["messages"] != empty["messages"]


@pytest.mark.asyncio
async def test_ollama_still_refuses_content_none_where_it_is_a_missing_value() -> None:
    """The narrowing is only for the assistant-plus-tool_calls encoding.

    Every other role, and an assistant turn carrying nothing at all, still has no
    faithful representation and is still refused.

    Mutation this exists to catch: drop the role/tool_calls condition and accept every
    `None`.
    """
    for msg in (
        ChatMessage(role=MessageRole.USER, content=None),
        ChatMessage(role=MessageRole.SYSTEM, content=None),
        ChatMessage(role=MessageRole.ASSISTANT, content=None),
        ChatMessage(role=MessageRole.TOOL, content=None, tool_call_id="c1"),
    ):
        with pytest.raises(UnmappableChatMessageError) as excinfo:
            await _ollama_sent_body(msg)
        assert "content=None" in str(excinfo.value), str(excinfo.value)


@pytest.mark.asyncio
async def test_mock_llm_connector_virtual_latency_and_event_loop_yielding() -> None:
    """Verify MockLLMConnector virtual latency yields control to other event loop coroutines without wall-clock blocking."""
    interleaved_flag = False

    async def _background_task() -> None:
        nonlocal interleaved_flag
        await asyncio.sleep(0)  # yield once
        interleaved_flag = True

    # With a small virtual latency (e.g. 0.001s), background task has opportunity to execute
    connector = MockLLMConnector(default_response="Delayed answer", latency_seconds=0.005)
    assert connector.latency_seconds == 0.005
    assert connector.streaming_chunk_delay == 0.0

    task = asyncio.create_task(_background_task())
    req = LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="Hello"),))
    resp = await connector.generate(req)
    await task

    assert interleaved_flag is True
    assert resp.content == "Delayed answer"


@pytest.mark.asyncio
async def test_mock_llm_connector_streaming_chunk_delay() -> None:
    """Verify MockLLMConnector streaming delay yields between successive chunks."""
    connector = MockLLMConnector(
        default_response="hello world test",
        streaming_chunk_delay=0.002,
    )
    req = LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="Hello"),))
    chunks: list[StreamChunk] = []
    async for chunk in connector.stream(req):
        chunks.append(chunk)

    assert len(chunks) == 3
    assert [c.delta_content for c in chunks] == ["hello", " world", " test"]


# ======================================================================================
# Issue #665 — `dict(tc.arguments)` is a shallow unwrap of a recursively frozen mapping
#
# `ToolCallRequest.arguments` is `ImmutableJsonMapping`, and `freeze_mapping` freezes it
# **recursively**: every nested object inside it is a `MappingProxyType` too. `dict(...)`
# copies only the top level, so the nested proxies survive into the payload and the JSON
# encoder — `httpx`'s `json=`, or `json.dumps` directly — raises
# `TypeError: Object of type mappingproxy is not JSON serializable`.
#
# Every connector already imports `unwrap_immutable` and already applies it to
# `t.parameters` twenty lines from the defect, so the fix is the helper that was
# already there. The interesting part is why it survived: a **flat** argument mapping
# round-trips fine under the shallow copy, which is the shape every existing test and
# the mock connector produce. `_NESTED_ARGUMENTS` below is therefore load-bearing —
# swapping it for a flat mapping makes all four tests pass against the broken code.
# ======================================================================================


_NESTED_ARGUMENTS: dict[str, Any] = {
    "path": "README.md",
    "options": {"encoding": "utf-8", "limits": {"max_bytes": 4096}},
    "ranges": [{"start": 1, "end": 20}],
}
"""Tool-call arguments with an object nested inside an object and inside a list.

The nesting is the test. `freeze_mapping` leaves `MappingProxyType` at every level, and
only the top one is undone by `dict()`.
"""


def _nested_tool_call_turn() -> ChatMessage:
    """An assistant turn carrying one tool call with nested arguments."""
    msg = ChatMessage(
        role=MessageRole.ASSISTANT,
        tool_calls=(ToolCallRequest(id="call_1", name="read_file", arguments=_NESTED_ARGUMENTS),),
    )
    # Guard the premise: if pydantic ever stops freezing recursively, these tests would
    # keep passing while testing nothing. Both nestings are guarded, not just the first:
    # `_deep_freeze` descends into mappings and into sequences by separate branches, so a
    # change that stopped it descending into sequences would leave `options` a proxy and
    # silently evaporate the object-in-list coverage while this guard still passed.
    frozen = msg.tool_calls[0].arguments
    in_object = frozen["options"]
    assert not isinstance(in_object, dict), (
        f"expected a frozen proxy, got {type(in_object).__name__}"
    )
    in_list: object = cast(Sequence[object], frozen["ranges"])[0]
    in_list_type = type(in_list).__name__
    assert type(in_list) is not dict, f"expected a frozen proxy, got {in_list_type}"
    return msg


def test_ollama_serializes_nested_tool_call_arguments() -> None:
    """A tool call with a nested argument object survives `_build_payload` + `json.dumps`.

    This is the reported failure (#665): an Ollama turn carrying a previous structured
    tool call died inside `httpx`'s `json=` encoder, on the default local-model path.

    Killed by: src/uclone_x/llm/connectors/ollama.py :: "arguments": cast(dict[str, Any], unwrap_immutable(tc.arguments)),
    """
    connector = OllamaConnector()
    payload = connector._build_payload(  # pyright: ignore[reportPrivateUsage]
        LLMRequest(messages=(_nested_tool_call_turn(),))
    )

    round_tripped = json.loads(json.dumps(payload))
    arguments = round_tripped["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert arguments == _NESTED_ARGUMENTS


def test_anthropic_serializes_nested_tool_call_arguments() -> None:
    """The same shallow unwrap sat in Anthropic's `tool_use` block.

    Killed by: src/uclone_x/llm/connectors/anthropic.py :: "input": cast(dict[str, Any], unwrap_immutable(tc.arguments)),
    """
    connector = AnthropicConnector(api_key="k")
    payload = connector._build_payload(  # pyright: ignore[reportPrivateUsage]
        LLMRequest(model="claude-3-5-sonnet", messages=(_nested_tool_call_turn(),))
    )

    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped["messages"][0]["content"][0]["input"] == _NESTED_ARGUMENTS


def test_gemini_serializes_nested_tool_call_arguments() -> None:
    """The same shallow unwrap sat in Gemini's `functionCall` part.

    Killed by: src/uclone_x/llm/connectors/gemini.py :: "args": cast(dict[str, Any], unwrap_immutable(tc.arguments)),
    """
    connector = GeminiConnector(api_key="k")
    payload = connector._build_payload(  # pyright: ignore[reportPrivateUsage]
        LLMRequest(model="gemini-1.5-pro", messages=(_nested_tool_call_turn(),))
    )

    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped["contents"][0]["parts"][0]["functionCall"]["args"] == _NESTED_ARGUMENTS


def test_openai_serializes_nested_tool_call_arguments() -> None:
    """OpenAI fails one step earlier: it `json.dumps` the arguments inside `_build_payload`.

    The wire shape differs — OpenAI carries the arguments as a JSON *string* — so the
    assertion parses that string rather than reading a nested object off the payload.

    Killed by: src/uclone_x/llm/connectors/openai.py :: "arguments": json.dumps(unwrap_immutable(tc.arguments)),
    """
    connector = OpenAIConnector(api_key="k")
    payload = connector._build_payload(  # pyright: ignore[reportPrivateUsage]
        LLMRequest(model="gpt-4o", messages=(_nested_tool_call_turn(),))
    )

    round_tripped = json.loads(json.dumps(payload))
    raw = round_tripped["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(raw) == _NESTED_ARGUMENTS


@pytest.mark.asyncio
async def test_ollama_connector_captures_thinking_channel_in_generate() -> None:
    """Ollama models emitting `thinking` in message payload must be preserved in ModelResponse (#695).

    Killed by: src/uclone_x/llm/connectors/ollama.py :: raw_thinking = msg_data.get("thinking")
    Becomes: raw_thinking = None
    """

    def handler(request: httpx.Request) -> httpx.Response:
        response_data: dict[str, Any] = {
            "model": "qwen3:8b",
            "message": {
                "role": "assistant",
                "content": "The final answer is 42.",
                "thinking": "Step 1: Compute 6 * 7 = 42.",
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 25,
            "eval_count": 30,
        }
        return httpx.Response(200, json=response_data)

    client = _make_mock_client(handler)
    connector = OllamaConnector(http_client=client)

    llm_req = LLMRequest(
        model="qwen3:8b",
        messages=(ChatMessage(role=MessageRole.USER, content="What is 6 * 7?"),),
    )

    resp = await connector.generate(llm_req)
    assert resp.content == "The final answer is 42."
    assert resp.thinking == "Step 1: Compute 6 * 7 = 42."
    assert resp.finish_reason == FinishReason.STOP


@pytest.mark.asyncio
async def test_ollama_connector_captures_delta_thinking_channel_in_stream() -> None:
    """Ollama streaming chunks emitting `thinking` must yield StreamChunk with delta_thinking (#695).

    Killed by: src/uclone_x/llm/connectors/ollama.py :: raw_delta_thinking = msg_data.get("thinking")
    Becomes: raw_delta_thinking = None
    """

    def handler(request: httpx.Request) -> httpx.Response:
        lines = [
            json.dumps({"message": {"thinking": "Thinking step 1..."}, "done": False}),
            json.dumps({"message": {"thinking": "Thinking step 2..."}, "done": False}),
            json.dumps({"message": {"content": "Here is"}, "done": False}),
            json.dumps(
                {
                    "message": {"content": " the answer."},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 10,
                    "eval_count": 20,
                }
            ),
        ]
        return httpx.Response(200, text="\n".join(lines) + "\n")

    client = _make_mock_client(handler)
    connector = OllamaConnector(http_client=client)

    chunks: list[StreamChunk] = []
    async for chunk in connector.stream(
        LLMRequest(
            model="qwen3:8b",
            messages=(ChatMessage(role=MessageRole.USER, content="Solve problem"),),
        )
    ):
        chunks.append(chunk)

    assert len(chunks) == 4
    thinking_deltas = [c.delta_thinking for c in chunks if c.delta_thinking is not None]
    assert thinking_deltas == ["Thinking step 1...", "Thinking step 2..."]
    content_deltas = [c.delta_content for c in chunks if c.delta_content is not None]
    assert "".join(content_deltas) == "Here is the answer."
    assert chunks[-1].finish_reason == FinishReason.STOP


# ======================================================================================
# A count the provider did not report is estimated and labelled, never a zero (#939)
# ======================================================================================

_UNREPORTED_REPLY = "맑고 따뜻합니다. Sunny and warm."
# Ollama's `eval_count` counts a thinking model's reasoning, so its output estimate does too.
_UNREPORTED_THINKING = "날씨 도구 없이 답한다."


def _reply_text(provider: str) -> str:
    """Everything the reply said that its output count covers."""
    return _UNREPORTED_THINKING + _UNREPORTED_REPLY if provider == "ollama" else _UNREPORTED_REPLY


_UNREPORTED_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-haiku",
    "gemini": "gemini-1.5-flash",
    "ollama": "qwen2.5-coder:7b",
}


def _unreported_request(provider: str) -> LLMRequest:
    return LLMRequest(
        model=_UNREPORTED_MODELS[provider],
        messages=(
            ChatMessage(role=MessageRole.SYSTEM, content="Answer in one sentence."),
            ChatMessage(role=MessageRole.USER, content="서울 날씨 어때요?"),
        ),
        tools=(
            ToolDefinition(
                name="weather",
                description="Current weather for a city",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            ),
        ),
    )


def _generate_body(provider: str, input_tokens: int | None, output_tokens: int | None) -> Any:
    """A `generate` response carrying only the counts given; `None` leaves a count out."""
    model = _UNREPORTED_MODELS[provider]
    if provider == "openai":
        usage = {
            key: value
            for key, value in (
                ("prompt_tokens", input_tokens),
                ("completion_tokens", output_tokens),
            )
            if value is not None
        }
        body: dict[str, Any] = {
            "choices": [{"message": {"content": _UNREPORTED_REPLY}, "finish_reason": "stop"}],
            "model": model,
        }
        if usage:
            body["usage"] = usage
        return body
    if provider == "anthropic":
        usage = {
            key: value
            for key, value in (("input_tokens", input_tokens), ("output_tokens", output_tokens))
            if value is not None
        }
        body = {
            "content": [{"type": "text", "text": _UNREPORTED_REPLY}],
            "stop_reason": "end_turn",
            "model": model,
        }
        if usage:
            body["usage"] = usage
        return body
    if provider == "gemini":
        usage = {
            key: value
            for key, value in (
                ("promptTokenCount", input_tokens),
                ("candidatesTokenCount", output_tokens),
            )
            if value is not None
        }
        body = {
            "candidates": [
                {"content": {"parts": [{"text": _UNREPORTED_REPLY}]}, "finishReason": "STOP"}
            ],
            "modelVersion": model,
        }
        if usage:
            body["usageMetadata"] = usage
        return body
    body = {
        "message": {"thinking": _UNREPORTED_THINKING, "content": _UNREPORTED_REPLY},
        "done": True,
        "done_reason": "stop",
        "model": model,
    }
    if input_tokens is not None:
        body["prompt_eval_count"] = input_tokens
    if output_tokens is not None:
        body["eval_count"] = output_tokens
    return body


def _connector_for(provider: str, client: httpx.AsyncClient) -> BaseLLMConnector:
    if provider == "openai":
        return OpenAIConnector(api_key="k", http_client=client)
    if provider == "anthropic":
        return AnthropicConnector(api_key="k", http_client=client)
    if provider == "gemini":
        return GeminiConnector(api_key="k", http_client=client)
    return OllamaConnector(http_client=client)


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini", "ollama"])
@pytest.mark.parametrize(
    ("reported_in", "reported_out"),
    [(12, 7), (12, None), (None, 7), (None, None)],
    ids=["both-reported", "output-missing", "input-missing", "usage-missing"],
)
@pytest.mark.asyncio
async def test_generate_estimates_and_labels_a_count_the_provider_did_not_report(
    provider: str, reported_in: int | None, reported_out: int | None
) -> None:
    """A missing count becomes a labelled estimate, and a reported one is kept.

    Every connector read a missing usage block as `0` tokens with the default
    `count_source=PROVIDER`: a claim that the provider counted zero. A provider that
    omitted usage was therefore uncharged on the headless path, and after #935 charged
    only when a head was watching, because the stream path estimates. The figure now is
    the shared estimate (`llm/compactor.py`), marked `ESTIMATE`, so the budget and the room report can tell it from a count.

    Gemini is the one wire format where an absent field inside a present block *is* a
    count: its JSON omits zero-valued fields, so only an absent `usageMetadata` reports
    nothing.

    Killed by: src/uclone_x/llm/connectors/openai.py :: in_tokens = reported_count(usage_data, "prompt_tokens")
    Becomes: in_tokens = usage_data.get("prompt_tokens", 0)
    Killed by: src/uclone_x/llm/connectors/anthropic.py :: out_tokens = reported_count(usage_data, "output_tokens")
    Becomes: out_tokens = usage_data.get("output_tokens", 0)
    Killed by: src/uclone_x/llm/connectors/gemini.py :: if usage_meta is None:
    Becomes: if False:
    Killed by: src/uclone_x/llm/connectors/ollama.py :: in_tokens = reported_count(data, "prompt_eval_count")
    Becomes: in_tokens = data.get("prompt_eval_count", 0)
    Killed by: src/uclone_x/llm/connectors/ollama.py :: reply="".join(part for part in (thinking, content) if part),
    Becomes: reply=content,
    Killed by: src/uclone_x/llm/connectors/base.py :: source = TokenCountSource.ESTIMATE if estimated else TokenCountSource.PROVIDER
    Becomes: source = TokenCountSource.PROVIDER
    """
    request = _unreported_request(provider)
    body = _generate_body(provider, reported_in, reported_out)
    client = _make_mock_client(lambda _: httpx.Response(200, json=body))

    usage = (await _connector_for(provider, client).generate(request)).usage

    fully_reported = reported_in is not None and reported_out is not None
    if provider == "gemini" and (reported_in is not None or reported_out is not None):
        expected_in, expected_out = reported_in or 0, reported_out or 0
        expected_source = TokenCountSource.PROVIDER
    else:
        expected_in = reported_in if reported_in is not None else estimate_request_tokens(request)
        expected_out = (
            reported_out
            if reported_out is not None
            else estimate_reply_tokens(_reply_text(provider))
        )
        expected_source = TokenCountSource.PROVIDER if fully_reported else TokenCountSource.ESTIMATE
    assert (usage.input_tokens, usage.output_tokens) == (expected_in, expected_out)
    assert usage.total_tokens == expected_in + expected_out
    assert usage.count_source is expected_source


def _stream_body(provider: str, input_tokens: int | None, output_tokens: int | None) -> str:
    """A stream whose usage carries only the counts given, over two text deltas."""
    first, second = _UNREPORTED_REPLY[:6], _UNREPORTED_REPLY[6:]
    if provider == "openai":
        # Usage arrives, as `include_usage` sends it, with only the counts given.
        usage = {
            key: value
            for key, value in (
                ("prompt_tokens", input_tokens),
                ("completion_tokens", output_tokens),
            )
            if value is not None
        }
        events: list[dict[str, Any]] = [
            {"choices": [{"delta": {"content": first}}]},
            {"choices": [{"delta": {"content": second}, "finish_reason": "stop"}]},
            {"choices": [], "usage": usage, "model": "gpt-4o-mini"},
        ]
        return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    if provider == "anthropic":
        # The input count belongs to `message_start`, the output count to `message_delta`.
        start: dict[str, Any] = {}
        if input_tokens is not None:
            start["usage"] = {"input_tokens": input_tokens}
        delta: dict[str, Any] = {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}
        if output_tokens is not None:
            delta["usage"] = {"output_tokens": output_tokens}
        events = [
            {"type": "message_start", "message": start},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": first}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": second}},
            delta,
        ]
        return "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    done: dict[str, Any] = {"message": {"content": second}, "done": True, "done_reason": "stop"}
    if input_tokens is not None:
        done["prompt_eval_count"] = input_tokens
    if output_tokens is not None:
        done["eval_count"] = output_tokens
    lines = [
        {"message": {"thinking": _UNREPORTED_THINKING}, "done": False},
        {"message": {"content": first}, "done": False},
        done,
    ]
    return "".join(json.dumps(line) + "\n" for line in lines)


@pytest.mark.parametrize(
    ("provider", "reported_in", "reported_out"),
    [
        ("openai", 12, None),
        ("anthropic", None, 7),
        ("anthropic", 12, None),
        ("ollama", None, None),
    ],
)
@pytest.mark.asyncio
async def test_a_stream_estimates_and_labels_a_count_its_usage_left_out(
    provider: str, reported_in: int | None, reported_out: int | None
) -> None:
    """A stream's usage with a count missing is completed from an estimate, not a zero.

    A stream that sends no usage at all yields none, and `BaseAgent._invoke_model` labels
    its estimate (#935). These streams do send usage, with a count left out, and the
    Anthropic and Ollama streams read the missing count as `0`, as the OpenAI stream did
    field by field. The reported count is kept, and the missing one is estimated: an input
    from the request, an output from what the stream said.

    Killed by: src/uclone_x/llm/connectors/openai.py :: out_tok = reported_count(stream_usage, "completion_tokens")
    Becomes: out_tok = stream_usage.get("completion_tokens", 0)
    Killed by: src/uclone_x/llm/connectors/openai.py :: streamed.append(delta_content)
    Becomes: pass
    Killed by: src/uclone_x/llm/connectors/anthropic.py :: input_tokens = reported_count(u, "input_tokens")
    Becomes: input_tokens = 0
    Killed by: src/uclone_x/llm/connectors/anthropic.py :: streamed.append(delta_content)
    Becomes: pass
    Killed by: src/uclone_x/llm/connectors/ollama.py :: out_tok = reported_count(data, "eval_count")
    Becomes: out_tok = data.get("eval_count", 0)
    Killed by: src/uclone_x/llm/connectors/ollama.py :: streamed.append(delta_content)
    Becomes: pass
    Killed by: src/uclone_x/llm/connectors/ollama.py :: streamed.append(delta_thinking)
    Becomes: pass
    """
    body = _stream_body(provider, reported_in, reported_out)
    client = _make_mock_client(lambda _: httpx.Response(200, content=body.encode("utf-8")))
    request = _unreported_request(provider)

    chunks = [c async for c in _connector_for(provider, client).stream(request)]

    usages = [c.usage for c in chunks if c.usage is not None]
    assert len(usages) == 1, usages
    usage = usages[0]
    want_in = reported_in if reported_in is not None else estimate_request_tokens(request)
    want_out = (
        reported_out if reported_out is not None else estimate_reply_tokens(_reply_text(provider))
    )
    assert (usage.input_tokens, usage.output_tokens) == (want_in, want_out)
    assert usage.count_source is TokenCountSource.ESTIMATE


# ======================================================================================
# vLLM — an OpenAI-compatible endpoint that is not OpenAI (#1304)
# ======================================================================================

_VLLM_SERVED_MODEL = "qwen2.5-coder-32b-instruct"
"""A model name of the kind a `vllm serve --model` argument names."""


def _vllm_endpoint(
    recorder: dict[str, Any] | None = None,
    served_model: str = _VLLM_SERVED_MODEL,
) -> Callable[[httpx.Request], httpx.Response]:
    """A handler answering like vLLM's `/v1/chat/completions`, recording what it received.

    The body is OpenAI's, because vLLM's is: the same `choices`/`message`/`finish_reason`
    shape and the same `usage` block. That is the fact `VLLMConnector` is built on, and a
    handler that invented a different shape would test the test instead.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if recorder is not None:
            recorder["path"] = request.url.path
            recorder["headers"] = dict(request.headers)
            recorder["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 1000},
                "model": served_model,
            },
        )

    return handler


@pytest.mark.asyncio
async def test_a_vllm_turn_is_attributed_to_vllm_and_not_to_openai() -> None:
    """Provenance and usage name the service that answered, not the wire format it speaks.

    Reaching vLLM as `OpenAIConnector(base_url=...)` — the arrangement this connector
    replaces — recorded `provider="openai"` in `Provenance.requested`, in
    `Provenance.served_by` and on the ledger. Every one of those is a statement about a
    service that was never involved, in the fields #149 added to make attribution
    trustworthy, and the ledger's is also the figure #958's third item is about.

    Killed by: src/uclone_x/llm/connectors/vllm.py :: return "vllm"
    Becomes: return "openai"
    """
    connector = VLLMConnector(
        base_url="http://localhost:8000", http_client=_make_mock_client(_vllm_endpoint())
    )
    resp = await connector.generate(
        LLMRequest(
            messages=(ChatMessage(role=MessageRole.USER, content="hi"),),
            model=_VLLM_SERVED_MODEL,
        )
    )

    assert resp.provenance is not None
    assert resp.provenance.requested.provider == "vllm"
    assert resp.provenance.served_by.provider == "vllm"
    assert resp.usage is not None
    assert resp.usage.provider == "vllm"


@pytest.mark.asyncio
async def test_a_server_with_no_api_key_is_sent_no_authorization_header() -> None:
    """No credential means no `Authorization` header — not an empty bearer, not a refusal.

    `vllm serve` takes `--api-key` and does not require it, so demanding one would be
    demanding a value the operator has to invent. The other direction is asserted in the
    same test: a key that *is* configured is still sent, or "no header" would be
    indistinguishable from a connector that never authenticates and #385's empty bearer
    would have been replaced by a silently unauthenticated request.

    Killed by: src/uclone_x/llm/connectors/vllm.py :: _requires_api_key: ClassVar[bool] = False
    Becomes: _requires_api_key: ClassVar[bool] = True
    """
    anonymous: dict[str, Any] = {}
    connector = VLLMConnector(
        base_url="http://localhost:8000",
        http_client=_make_mock_client(_vllm_endpoint(anonymous)),
    )
    await connector.generate(
        LLMRequest(
            messages=(ChatMessage(role=MessageRole.USER, content="hi"),),
            model=_VLLM_SERVED_MODEL,
        )
    )
    assert "authorization" not in {name.lower() for name in anonymous["headers"]}

    keyed: dict[str, Any] = {}
    authenticated = VLLMConnector(
        api_key="served-with-a-key",
        base_url="http://localhost:8000",
        http_client=_make_mock_client(_vllm_endpoint(keyed)),
    )
    await authenticated.generate(
        LLMRequest(
            messages=(ChatMessage(role=MessageRole.USER, content="hi"),),
            model=_VLLM_SERVED_MODEL,
        )
    )
    assert keyed["headers"]["authorization"] == "Bearer served-with-a-key"


def test_an_unnamed_vllm_endpoint_is_refused_at_construction_naming_the_variable() -> None:
    """No endpoint is a refusal here, not a localhost guess failing at the first turn.

    Ollama defaults to `http://localhost:11434` because that is where its daemon installs
    itself. A vLLM server is launched per model on a port chosen at the command line, so
    `http://localhost:8000` would name a process that exists only if somebody started it
    with those arguments — and a connector built on that guess succeeds, then reports the
    missing configuration as a refused TCP connection on the first turn (P6, the shape #533
    fixed in the factory).

    Killed by: src/uclone_x/llm/connectors/vllm.py :: raise LLMProviderNotConfiguredError(_UNCONFIGURED_ENDPOINT_MESSAGE)
    Becomes: return "http://localhost:8000/v1"
    """
    with pytest.raises(LLMProviderNotConfiguredError) as excinfo:
        VLLMConnector()
    assert "VLLM_BASE_URL" in str(excinfo.value)

    # And through the factory, where naming the provider is what asks for the refusal.
    with pytest.raises(LLMProviderNotConfiguredError) as from_factory:
        create_llm_connector(provider="vllm")
    assert "VLLM_BASE_URL" in str(from_factory.value)


@pytest.mark.asyncio
async def test_no_failure_from_this_connector_names_openai_to_a_vllm_operator() -> None:
    """Every refusal `OpenAIConnector` raises says which server actually answered.

    `_display_name` exists so a subclass inherits the request path without inheriting the
    vendor's name, and it is only worth having if *every* message uses it. One did not: the
    malformed-body branch said "Invalid JSON from OpenAI" regardless, so an operator whose
    own vLLM server returned an HTML error page was told OpenAI had misbehaved — a sentence
    naming a vendor they are not talking to, on the one code path where they have the least
    information. Found by review, not by the gate, because no test read that string.

    Swept rather than pinned at the fixed site: the defect was a family covered at four of
    its five members, so the assertion is over every `raise LLMProviderError` in the module.

    Killed by: src/uclone_x/llm/connectors/openai.py :: f"Invalid JSON from {self._display_name}: {exc}"
    Becomes: f"Invalid JSON from OpenAI: {exc}"
    """
    source = Path(openai_module.__file__ or "").read_text(encoding="utf-8")
    raises = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "LLMProviderError"
    ]
    # The control: a parse that found nothing would satisfy every assertion below.
    assert len(raises) >= 5, f"only {len(raises)} LLMProviderError raises found in openai.py"
    for node in raises:
        rendered = ast.get_source_segment(source, node) or ""
        assert "OpenAI" not in rendered, f"line {node.lineno} hardcodes the vendor: {rendered}"

    # And on the wire, where the operator reads it: a body that is not JSON at all.
    connector = VLLMConnector(
        base_url="http://vllm.invalid:8000/v1",
        http_client=_make_mock_client(
            lambda _request: httpx.Response(200, text="<html>502 Bad Gateway</html>")
        ),
    )
    with pytest.raises(LLMProviderError) as excinfo:
        await connector.generate(
            LLMRequest(
                messages=(ChatMessage(role=MessageRole.USER, content="hi"),),
                model=_VLLM_SERVED_MODEL,
            )
        )
    assert "vLLM" in str(excinfo.value)
    assert "OpenAI" not in str(excinfo.value)


@pytest.mark.parametrize(
    "configured",
    (
        "http://vllm.invalid:8000",
        "http://vllm.invalid:8000/",
        "http://vllm.invalid:8000/v1",
        "http://vllm.invalid:8000/v1/",
        "http://vllm.invalid:8000/v1/v1",
    ),
)
@pytest.mark.asyncio
async def test_every_spelling_of_the_endpoint_reaches_the_same_v1_surface(
    configured: str,
) -> None:
    """The five ways an operator writes the endpoint are one endpoint.

    vLLM's startup banner prints `http://0.0.0.0:8000` and its curl examples print
    `/v1/chat/completions`, so both spellings are what a person will paste — and the
    difference between them must not be the difference between a working connector and a
    404 on every turn.

    The `/v1/v1` case is the one that distinguishes the `while` in
    `normalize_vllm_base_url` from an `if`: with a single strip, a value that already
    carries two segments keeps two, and this is the only spelling that notices.

    Killed by: src/uclone_x/llm/connectors/vllm.py :: return f"{cleaned}/v1"
    Becomes: return cleaned
    """
    seen: dict[str, Any] = {}
    connector = VLLMConnector(
        base_url=configured, http_client=_make_mock_client(_vllm_endpoint(seen))
    )
    assert connector.base_url == "http://vllm.invalid:8000/v1"

    await connector.generate(
        LLMRequest(
            messages=(ChatMessage(role=MessageRole.USER, content="hi"),),
            model=_VLLM_SERVED_MODEL,
        )
    )
    assert seen["path"] == "/v1/chat/completions"


@pytest.mark.asyncio
async def test_a_request_naming_no_model_asks_for_the_configured_one_not_gpt_4o(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`VLLM_MODEL` is what a request that names no model asks for.

    The inherited default is `gpt-4o`, which no vLLM server serves: it answers `The model
    gpt-4o does not exist`, reporting the operator's missing configuration as a fact about
    a model nobody chose. What reaches the wire is also what provenance must claim was
    requested, so both are asserted from the same recording (#149).

    Killed by: src/uclone_x/llm/connectors/vllm.py :: configured = resolve_vllm_model()
    Becomes: configured = None
    """
    monkeypatch.setenv("VLLM_MODEL", _VLLM_SERVED_MODEL)
    seen: dict[str, Any] = {}
    connector = VLLMConnector(
        base_url="http://localhost:8000", http_client=_make_mock_client(_vllm_endpoint(seen))
    )

    resp = await connector.generate(
        LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),))
    )

    assert seen["body"]["model"] == _VLLM_SERVED_MODEL
    assert resp.provenance is not None
    assert resp.provenance.requested.model == _VLLM_SERVED_MODEL


@pytest.mark.asyncio
async def test_a_model_named_on_the_request_wins_over_the_configured_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caller's model is sent as the caller wrote it.

    A server can be restarted on another model without the variable being updated, and a
    caller naming one explicitly is the case where the variable is stale. Reading the
    variable first would silently send the wrong name and report a 404 about it.

    Killed by: src/uclone_x/llm/connectors/vllm.py :: named = named_model(request)
    Becomes: named = None
    """
    monkeypatch.setenv("VLLM_MODEL", "stale-from-a-previous-launch")
    seen: dict[str, Any] = {}
    connector = VLLMConnector(
        base_url="http://localhost:8000", http_client=_make_mock_client(_vllm_endpoint(seen))
    )

    await connector.generate(
        LLMRequest(
            messages=(ChatMessage(role=MessageRole.USER, content="hi"),),
            model=_VLLM_SERVED_MODEL,
        )
    )

    assert seen["body"]["model"] == _VLLM_SERVED_MODEL


@pytest.mark.asyncio
async def test_a_turn_with_no_model_anywhere_is_refused_rather_than_guessed() -> None:
    """Neither the request nor the environment naming a model is a refusal.

    There is no model string this connector could supply that is not a guess at somebody
    else's `--model` argument, and the guess arrives as a 404 naming a model the caller
    never chose. The refusal names the variable to set instead.

    Killed by: src/uclone_x/llm/connectors/vllm.py :: raise LLMProviderNotConfiguredError(_UNCONFIGURED_MODEL_MESSAGE)
    Becomes: return "gpt-4o"
    """
    connector = VLLMConnector(
        base_url="http://localhost:8000", http_client=_make_mock_client(_vllm_endpoint())
    )

    with pytest.raises(LLMProviderNotConfiguredError) as excinfo:
        await connector.generate(
            LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),))
        )
    assert "VLLM_MODEL" in str(excinfo.value)


def test_the_factory_resolves_vllm_by_name_and_from_its_endpoint_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Step 4 of the documented precedence, for vLLM, including what it does not decide.

    A configured endpoint is a choice the factory honours; an environment that configures
    both Ollama and vLLM is an ambiguity it does not resolve by guessing. Ollama keeps the
    answer it has always given there, and `LLM_PROVIDER` is how the other one is named —
    asserted here so that the precedence is a decision on record rather than a consequence
    of which line was added last.

    Killed by: src/uclone_x/llm/connectors/factory.py :: if has_configured_vllm_endpoint(base_url):
    Becomes: if False:
    """
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.invalid:8000")
    detected = create_llm_connector()
    assert isinstance(detected, VLLMConnector)
    assert detected.base_url == "http://vllm.invalid:8000/v1"

    named = create_llm_connector(provider="vllm")
    assert isinstance(named, VLLMConnector)

    # Both configured: Ollama, as before this provider existed.
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    assert isinstance(create_llm_connector(), OllamaConnector)
    assert isinstance(create_llm_connector(provider="vllm"), VLLMConnector)


def test_the_unconfigured_refusal_names_vllm_among_the_ways_to_configure_one() -> None:
    """The refusal a person actually reads mentions the provider they could have set.

    `create_llm_connector`'s message is the one place an unconfigured installation is told
    what to do, and a provider absent from it is a provider nobody discovers. The list and
    the code that reads it are the pair `VLLM_ENDPOINT_ENV_VARS` exists to keep together.

    Killed by: src/uclone_x/llm/connectors/factory.py :: "LLM_PROVIDER=openai|anthropic|gemini|ollama|vllm; "
    Becomes: "LLM_PROVIDER=openai|anthropic|gemini|ollama; "
    """
    with pytest.raises(LLMProviderNotConfiguredError) as excinfo:
        create_llm_connector()

    message = str(excinfo.value)
    assert "vllm" in message
    for name in VLLM_ENDPOINT_ENV_VARS:
        assert name in message


# --- #1372: `keep_alive` and `num_ctx` on the Ollama request ---------------------------


def _ollama_body(connector: OllamaConnector, **request: Any) -> dict[str, Any]:
    msgs = (ChatMessage(role=MessageRole.USER, content="hi"),)
    return connector._build_payload(LLMRequest(messages=msgs, **request))  # pyright: ignore[reportPrivateUsage]


def test_ollama_sends_keep_alive_with_the_stated_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without it the daemon unloads the model after five idle minutes and the next turn
    re-reads the whole conversation cold. The default is thirty minutes.

    Killed by: src/uclone_x/llm/connectors/ollama.py :: "keep_alive": self.keep_alive,
    Becomes: "keep_alive_unused": self.keep_alive,
    Killed by: src/uclone_x/llm/connectors/ollama.py :: DEFAULT_OLLAMA_KEEP_ALIVE = "30m"
    Becomes: DEFAULT_OLLAMA_KEEP_ALIVE = "5m"
    """
    monkeypatch.delenv("OLLAMA_KEEP_ALIVE", raising=False)

    body = _ollama_body(OllamaConnector(base_url="http://localhost:11434"))

    assert DEFAULT_OLLAMA_KEEP_ALIVE == "30m"
    assert body["keep_alive"] == "30m"


def test_ollama_keep_alive_follows_the_daemons_variable_and_then_the_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request's `keep_alive` overrides the daemon's, so an operator's
    `OLLAMA_KEEP_ALIVE=-1` is sent rather than replaced by the default. A bare number goes
    as a number: the daemon reads a string as a duration, and `"-1"` has no unit.

    Killed by: src/uclone_x/llm/connectors/ollama.py :: return int(text)
    Becomes: return str(text)
    """
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "-1")
    from_env = _ollama_body(OllamaConnector(base_url="http://localhost:11434"))
    explicit = _ollama_body(OllamaConnector(base_url="http://localhost:11434", keep_alive="1h"))

    assert from_env["keep_alive"] == -1
    assert isinstance(from_env["keep_alive"], int)
    assert explicit["keep_alive"] == "1h"


def test_ollama_sends_the_configured_window_as_num_ctx_and_keeps_sending_it() -> None:
    """A request naming no window, to a model that was sent one, is sent the same one:
    Ollama reloads a model whose options change, and a summary request without `num_ctx`
    would reload it at the daemon's default and the next turn would reload it back.

    Killed by: src/uclone_x/llm/connectors/ollama.py :: options["num_ctx"] = num_ctx
    Becomes: options["num_ctx_unused"] = num_ctx
    Killed by: src/uclone_x/llm/connectors/ollama.py :: num_ctx = self._num_ctx.get(window_key)
    Becomes: num_ctx = request.context_window
    """
    connector = OllamaConnector(base_url="http://localhost:11434")

    first = _ollama_body(connector, model="llama3.2:1b", context_window=16_384)
    later = _ollama_body(connector, model="llama3.2:1b")
    other = _ollama_body(connector, model="qwen3:8b")

    assert first["options"]["num_ctx"] == 16_384
    assert later["options"]["num_ctx"] == 16_384
    assert "num_ctx" not in other["options"]
