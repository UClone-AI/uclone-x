"""A connection that could not be reached is said as that, never as a missing model (#2418).

The owner's screenshot (2026-10-29): a turn on `remote-gpu/qwen3:14b` said "remote-gpu
doesn't offer the model this turn asked for", while the Dell lists `qwen3:14b`. What failed
was the tunnel the connection points at. Two paths told it wrong:

* a ref whose connection is not there (the tunnel's connection is removed when the tunnel
  goes) was a `model_unavailable` failure -- the kind a provider that lacks the model gets;
* an Ollama that refuses the connection was no provider failure at all, so the row said
  only that something went wrong.

Both are now `provider_unreachable`. A provider that answers that it lacks the model is still
`model_unavailable`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from uclone_x.agent.models import AgentLLMConfig
from uclone_x.agent.turn_executor import (
    _turn_failure,  # pyright: ignore[reportPrivateUsage]
)
from uclone_x.errors import ProviderFailureKind
from uclone_x.llm.connectors.failures import failed_status
from uclone_x.llm.connectors.ollama import OllamaConnector
from uclone_x.llm.gateway import ModelGateway
from uclone_x.llm.models import ChatMessage, LLMRequest, MessageRole

_REQUEST = LLMRequest(
    model="qwen3:14b", messages=(ChatMessage(role=MessageRole.USER, content="hi"),)
)

#: What a sentence for a person must never carry (#1408, #2167).
_INTERNALS = ("Error", "http", "127.0.0.1", "Traceback", "agent.core", "error_handler")


def _refused(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("All connection attempts failed", request=request)


def _ollama() -> OllamaConnector:
    return OllamaConnector(
        base_url="http://127.0.0.1:61669",
        model="qwen3:14b",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_refused)),
    )


async def _stream(connector: OllamaConnector) -> None:
    async for _ in connector.stream(_REQUEST):
        pass


def test_a_ref_whose_connection_is_gone_could_not_be_reached(tmp_path: Path) -> None:
    """The tunnel's connection is removed with the tunnel; the model is not the cause.

    Killed by: src/uclone_x/llm/gateway.py :: kind = ProviderFailureKind.PROVIDER_UNREACHABLE
    Becomes: kind = ProviderFailureKind.MODEL_UNAVAILABLE
    """
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"connections": [{"id": "mock", "kind": "mock"}]}))
    seat = ModelGateway(path).bind(AgentLLMConfig(model_name="remote-gpu/qwen3:14b"))
    assert seat.llm is not None

    with pytest.raises(Exception) as raised:
        asyncio.run(seat.llm.generate(_REQUEST))
    stop, _error, failure = _turn_failure(raised.value, "not_started", "composer")

    assert failure is not None
    assert failure.kind is ProviderFailureKind.PROVIDER_UNREACHABLE
    assert stop == "provider_unreachable"
    assert failure.provider == "remote-gpu"
    # A connection that comes back answers the next try (the seat is re-bound then).
    assert failure.retryable


@pytest.mark.parametrize("call", ["generate", "stream"])
def test_an_ollama_that_refuses_the_connection_could_not_be_reached(call: str) -> None:
    """A refused connect is a provider failure the row can name, not "something went wrong".

    Killed by: src/uclone_x/llm/connectors/ollama.py :: return OllamaUnreachableError(
    Becomes: return LLMProviderError(
    """
    connector = _ollama()
    with pytest.raises(Exception) as raised:
        asyncio.run(connector.generate(_REQUEST) if call == "generate" else _stream(connector))
    stop, _error, failure = _turn_failure(raised.value, "not_started", "composer")

    assert failure is not None, "a refused connection was no provider failure"
    assert failure.kind is ProviderFailureKind.PROVIDER_UNREACHABLE
    assert stop == "provider_unreachable"
    assert failure.provider == "Ollama"


def test_the_failure_a_person_reads_carries_no_internals() -> None:
    """The provider and the cause, in plain words: no class, address or handler name."""
    connector = _ollama()
    with pytest.raises(Exception) as raised:
        asyncio.run(connector.generate(_REQUEST))
    _stop, _error, failure = _turn_failure(raised.value, "not_started", "composer")

    assert failure is not None
    for internal in _INTERNALS:
        assert internal not in failure.message, failure.message
        assert internal not in (failure.provider or ""), failure.provider


def test_a_provider_that_lacks_the_model_still_says_so() -> None:
    """The listing was read and the model is not in it: that alone is `model_unavailable`."""
    error = failed_status(
        provider="Ollama",
        model="qwen3:14b",
        status_code=404,
        body='{"error": "model \\"qwen3:14b\\" does not exist"}',
    )
    _stop, _error, failure = _turn_failure(error, "not_started", "composer")

    assert failure is not None
    assert failure.kind is ProviderFailureKind.MODEL_UNAVAILABLE
