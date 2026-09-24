"""A streamed turn names the model that answered it, and never invents one (#1447, FR-13.4).

A room turn is streamed, so it is attributed by `BaseAgent._invoke_model`'s stream branch,
not by a connector's `generate`. That branch named the model from the request, else from a
`_default_model`/`model` attribute the connector might carry, else the literal `"default"`.
The room's requests name no model and `OllamaConnector` carried neither attribute, so every
streamed Ollama turn was stored as `{"provider": "ollama", "model": "default"}` and shown as
`ollama:default` while `OLLAMA_MODEL` answered.

The Ollama tests run the real `OllamaConnector` over an `httpx.MockTransport` that answers
with the NDJSON lines Ollama's `/api/chat` streams -- each naming the model it ran -- so the
path from the wire to the stored provenance is the production one; a fake connector would
have supplied the attribute the real one lacked and hidden the bug. The other test uses a
connector with no model to name at all, which is the case the literal came from.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
import pytest

from uclone_x.agent import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig, TurnResult
from uclone_x.core.provenance import ServiceRef
from uclone_x.llm import OllamaConnector, OpenAIConnector
from uclone_x.llm.models import (
    ChatMessage,
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
)

_ENV_MODEL = "qwen3:8b"


@pytest.fixture(autouse=True)
def ollama_model_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """`OLLAMA_MODEL` names the model, as in the report; the lower tiers are cleared."""
    monkeypatch.setenv("OLLAMA_MODEL", _ENV_MODEL)
    monkeypatch.delenv("OLLAMA_INDEPTH_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_FAST_MODEL", raising=False)


def _ollama_lines(served: str) -> list[str]:
    """What `/api/chat` streams: one JSON object per line, each naming the model that ran,
    `done` on the last."""
    named: dict[str, Any] = {"model": served}
    return [
        json.dumps({**named, "message": {"role": "assistant", "content": "Hel"}, "done": False}),
        json.dumps({**named, "message": {"role": "assistant", "content": "lo"}, "done": False}),
        json.dumps(
            {
                **named,
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 12,
                "eval_count": 2,
            }
        ),
    ]


async def _streamed_turn(llm: Any, model_name: str | None = None) -> TurnResult:
    """One watched turn; by default from an agent whose config names no model, as a room
    seat's does."""
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="clone",
            name="Clone",
            system_prompt="",
            llm_config=AgentLLMConfig(model_name=model_name),
        ),
        llm=llm,
        context=AgentContext(session_id="sess_1447", agent_id="clone"),
    )

    async def listener(event: str, data: dict[str, Any]) -> None:
        return None

    return await agent.execute_turn("hi", stream_callback=listener)


async def _ollama_turn(served: str) -> tuple[TurnResult, list[dict[str, Any]]]:
    """A streamed turn against the real connector; also returns the bodies it sent."""
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            text="\n".join(_ollama_lines(served)) + "\n",
            headers={"Content-Type": "application/x-ndjson"},
        )

    connector = OllamaConnector(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    return await _streamed_turn(connector), sent


@pytest.mark.asyncio
async def test_a_streamed_ollama_turn_records_what_it_asked_for_and_what_answered() -> None:
    """The report's case: the request names no model, `OLLAMA_MODEL` is what is sent, and
    the stream names what ran. Both are recorded; neither is `default`.

    The stream names a different tag from the one sent, so each half of the attribution has
    one source: `requested` can only come from the connector's resolution and `served_by`
    only from the stream. Copying one onto the other would hide a substitution P6 wants
    visible, the way `generate` keeps them apart (#149).

    Killed by: src/uclone_x/agent/base.py :: served_model = chunk.model
    Becomes: pass
    Killed by: src/uclone_x/llm/connectors/ollama.py :: model=served_model,
    Becomes:
    Killed by: src/uclone_x/llm/connectors/ollama.py :: return resolve_ollama_model(None)
    Becomes: return "default"
    """
    result, sent = await _ollama_turn(served="qwen3:8b-q4_K_M")

    assert result.error is None
    assert [body["model"] for body in sent] == [_ENV_MODEL]
    assert result.provenance is not None
    assert result.provenance.requested == ServiceRef(provider="ollama", model=_ENV_MODEL)
    assert result.provenance.served_by == ServiceRef(provider="ollama", model="qwen3:8b-q4_K_M")


class _NamelessStreamConnector:
    """A connector with no model to name: no `_default_model`, no `model`, and chunks
    that carry none. Nothing in the turn can know which model answered."""

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def generate(self, request: LLMRequest) -> ModelResponse:  # pragma: no cover
        raise AssertionError("a watched turn streams; generate must not be reached")

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        chunks: Sequence[StreamChunk] = (
            StreamChunk(delta_content="Hello"),
            StreamChunk(finish_reason=FinishReason.STOP),
        )
        for chunk in chunks:
            yield chunk


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [None, "default"])
async def test_a_turn_whose_model_cannot_be_known_records_none_rather_than_default(
    configured: str | None,
) -> None:
    """No model is knowable, so none is recorded -- `None`, which the room renders as the
    provider alone -- and the placeholder `default` is not presented as one.

    Killed by: src/uclone_x/agent/base.py :: return stripped if stripped and stripped != "default" else None
    Becomes: return stripped or None
    Killed by: src/uclone_x/agent/base.py :: model=requested_model or served_model,
    Becomes: model=requested_model or served_model or "default",
    """
    result = await _streamed_turn(_NamelessStreamConnector(), model_name=configured)

    assert result.error is None
    assert result.content == "Hello"
    assert result.provenance is not None
    assert result.provenance.requested == ServiceRef(provider="ollama", model=None)
    assert result.provenance.served_by == ServiceRef(provider="ollama", model=None)


@pytest.mark.asyncio
async def test_an_openai_compatible_stream_carries_the_model_its_chunks_name() -> None:
    """OpenAI and vLLM name the model on every SSE chunk, and the chunk passes it on.

    vLLM declares an empty `_default_model`, so a streamed vLLM turn whose request named no
    model had nothing but the stream to be attributed from, and recorded `default` too.

    Killed by: src/uclone_x/llm/connectors/openai.py :: model=served_model,
    Becomes:
    """

    def handler(request: httpx.Request) -> httpx.Response:
        events: list[dict[str, Any]] = [
            {"model": "gpt-4o-2024-08-06", "choices": [{"delta": {"content": "Hi"}}]},
            {"model": "gpt-4o-2024-08-06", "choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})

    connector = OpenAIConnector(
        api_key="test_key", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    chunks = [
        chunk
        async for chunk in connector.stream(
            LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),))
        )
    ]

    assert [chunk.model for chunk in chunks] == ["gpt-4o-2024-08-06", "gpt-4o-2024-08-06"]
