"""Anthropic prompt-cache breakpoints, and the cache counts every usage record carries (#1371).

Anthropic caches a request's prefix up to each block marked `cache_control`, in the order
tools, system, messages. A mark only pays for itself if a later request sends the same
bytes up to it, so these tests pin three things:

*   where the marks go: the last tool, the system block, and the last block of the last
    message the model or a tool wrote -- never the trailing `USER` message, which carries
    the turn context that changes every step;
*   that across the steps of one real agent turn, and into the next turn, the bytes up to
    the previous request's conversation mark are sent again unchanged, so a step can read
    what the one before it wrote;
*   that the cache counts a provider reports reach `TokenUsage`, and a usage that reports
    none serializes byte for byte as it did before the fields existed.

No request leaves the process: the connector is given a mock transport, and the agent a
recording fake.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import BaseModel, Field

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig, AgentState
from uclone_x.agent.session import SessionStore
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.llm.connectors.anthropic import AnthropicConnector
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.connectors.gemini import GeminiConnector
from uclone_x.llm.connectors.openai import OpenAIConnector
from uclone_x.llm.models import (
    ChatMessage,
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
    ToolDefinition,
    aggregate_token_usages,
)
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry

_MODEL = "claude-opus-5"
_MARK = {"type": "ephemeral"}
_FACT = "postgres"
"""The remembered fact that puts a turn-context block at the tail of every request."""


def _connector(
    handler: Callable[[httpx.Request], httpx.Response] | None = None,
) -> AnthropicConnector:
    transport = httpx.MockTransport(handler or (lambda _: httpx.Response(500)))
    return AnthropicConnector(
        model=_MODEL, api_key="test-key", http_client=httpx.AsyncClient(transport=transport)
    )


def _payload(request: LLMRequest) -> dict[str, Any]:
    """The body the connector builds for `request`, before it is sent."""
    return _connector()._build_payload(request)  # pyright: ignore[reportPrivateUsage]


def _marks(node: object, path: str = "") -> list[str]:
    """Every path in `node` whose dict carries `cache_control`."""
    found: list[str] = []
    if isinstance(node, dict):
        mapping = cast("dict[str, object]", node)
        if "cache_control" in mapping:
            found.append(path)
        for key, value in mapping.items():
            found.extend(_marks(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for i, value in enumerate(cast("list[object]", node)):
            found.extend(_marks(value, f"{path}[{i}]"))
    return found


def _unmarked(node: object) -> object:
    """`node` with every `cache_control` key removed, which is what the cache key hashes."""
    if isinstance(node, dict):
        mapping = cast("dict[str, object]", node)
        return {k: _unmarked(v) for k, v in mapping.items() if k != "cache_control"}
    if isinstance(node, list):
        return [_unmarked(v) for v in cast("list[object]", node)]
    return node


def _marked_message(payload: dict[str, Any]) -> int | None:
    """The index of the one message holding a mark, or `None` when none does."""
    marked = [
        i for i, m in enumerate(payload["messages"]) if isinstance(m["content"], list) and _marks(m)
    ]
    assert len(marked) <= 1, marked
    return marked[0] if marked else None


_TOOL = ToolDefinition(
    name="count_tool",
    description="Adds one",
    parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
)


# ======================================================================================
# Where the marks go
# ======================================================================================


def test_the_marks_sit_on_the_last_tool_the_system_and_the_last_model_or_tool_message() -> None:
    """Three marks, one per stable layer, and none on the trailing user message.

    Killed by: src/uclone_x/llm/connectors/anthropic.py :: payload["tools"][-1]["cache_control"] = _CACHE_BREAKPOINT
    Becomes: payload["tools"][0]["cache_control"] = _CACHE_BREAKPOINT
    Killed by: src/uclone_x/llm/connectors/anthropic.py :: [{"type": "text", "text": system_text, "cache_control": _CACHE_BREAKPOINT}]
    Becomes: [{"type": "text", "text": system_text}]
    Killed by: src/uclone_x/llm/connectors/anthropic.py :: stable = [i for i, m in enumerate(messages_payload) if isinstance(m["content"], list)]
    Becomes: stable = [i for i, m in enumerate(messages_payload) if m["role"] == "assistant"]
    """
    second = _TOOL.model_copy(update={"name": "other_tool"})
    request = LLMRequest(
        messages=(
            ChatMessage(role=MessageRole.SYSTEM, content="You are a careful counter."),
            ChatMessage(role=MessageRole.USER, content="count"),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=(ToolCallRequest(id="c1", name="count_tool", arguments={"x": 1}),),
            ),
            ChatMessage(role=MessageRole.TOOL, content='{"result": 2}', tool_call_id="c1"),
            ChatMessage(role=MessageRole.USER, content=f"Turn context: the project uses {_FACT}."),
        ),
        tools=(_TOOL, second),
    )

    payload = _payload(request)

    assert sorted(_marks(payload)) == [
        ".messages[2].content[0]",
        ".system[0]",
        ".tools[1]",
    ]
    assert payload["system"] == [
        {"type": "text", "text": "You are a careful counter.", "cache_control": _MARK}
    ]
    assert payload["messages"][2]["content"][0]["type"] == "tool_result"
    assert payload["messages"][-1] == {
        "role": "user",
        "content": f"Turn context: the project uses {_FACT}.",
    }


def test_a_conversation_that_is_only_a_user_message_has_no_conversation_mark() -> None:
    """The first request of a session has nothing stable after the system prompt to mark."""
    payload = _payload(
        LLMRequest(
            messages=(
                ChatMessage(role=MessageRole.SYSTEM, content="Be brief."),
                ChatMessage(role=MessageRole.USER, content=f"hello; remember {_FACT}"),
            )
        )
    )

    assert _marks(payload) == [".system[0]"]


def test_an_assistant_answer_is_marked_on_its_text_and_a_blank_one_is_not() -> None:
    """Anthropic refuses `cache_control` on a blank text block, so a blank answer goes unmarked.

    Killed by: src/uclone_x/llm/connectors/anthropic.py :: if last_block.get("type") != "text" or last_block["text"].strip():
    Becomes: if True:
    """

    def payload_for(answer: str) -> dict[str, Any]:
        return _payload(
            LLMRequest(
                messages=(
                    ChatMessage(role=MessageRole.USER, content="hi"),
                    ChatMessage(role=MessageRole.ASSISTANT, content=answer),
                    ChatMessage(role=MessageRole.USER, content="again"),
                )
            )
        )

    assert _marks(payload_for("hello")) == [".messages[1].content[0]"]
    assert _marks(payload_for("  ")) == []


def test_a_blank_system_prompt_goes_out_as_the_plain_string_it_was() -> None:
    """A blank system prompt cannot carry a mark, so it keeps its old unmarked shape.

    Killed by: src/uclone_x/llm/connectors/anthropic.py :: if system_text.strip()
    Becomes: if True
    """
    payload = _payload(
        LLMRequest(
            messages=(
                ChatMessage(role=MessageRole.SYSTEM, content=""),
                ChatMessage(role=MessageRole.USER, content="hi"),
            )
        )
    )

    assert payload["system"] == ""
    assert _marks(payload) == []


# ======================================================================================
# The marked prefix is the same bytes from step to step
# ======================================================================================


_PROV = Provenance(
    path=ExecutionPath.PRIMARY,
    requested=ServiceRef(provider="agent", model="dummy"),
    served_by=ServiceRef(provider="agent", model="dummy"),
    attempts=(),
)
_USAGE = TokenUsage(provider="dummy", model="dummy", input_tokens=0, output_tokens=0)


def _answer(text: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP, content=text, tool_calls=(), usage=_USAGE, provenance=_PROV
    )


def _call(call_id: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        content=None,
        tool_calls=(ToolCallRequest(id=call_id, name="count_tool", arguments={"x": 1}),),
        usage=_USAGE,
        provenance=_PROV,
    )


class _RecordingLLM(BaseLLMConnector):
    """Replays a script, then answers `"done"`, and records every request it is sent."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        super().__init__()
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return "dummy"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0) if self.responses else _answer("done")

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:  # pragma: no cover
        yield StreamChunk(delta_content="")


class _CountParams(BaseModel):
    x: int = Field(default=0)


class _CountTool(BaseTool[_CountParams]):
    name = "count_tool"
    description = "Adds one"

    def run(self, params: _CountParams, context: ToolContext) -> dict[str, Any]:
        return {"result": params.x + 1}


def _agent(tmp_path: Path, llm: _RecordingLLM) -> BaseAgent:
    registry = ToolRegistry()
    registry.register(_CountTool())
    memory = CrossSessionMemory()
    memory.record_fact(
        subject="project",
        predicate="uses",
        object_value=_FACT,
        provenance=_PROV,
        source_session_id="earlier",
    )
    return BaseAgent(
        config=AgentConfig(
            agent_id="agent_cache",
            name="Agent",
            system_prompt="You are a careful counter.",
            llm_config=AgentLLMConfig(model_name="dummy", temperature=0.3, max_tokens=512),
            max_steps=8,
        ),
        llm=llm,
        tools=registry,
        store=SessionStore(tmp_path),
        context=AgentContext(
            session_id="sess_cache", agent_id="agent_cache", current_state=AgentState.IDLE
        ),
        memory=memory,
    )


@pytest.mark.asyncio
async def test_each_step_resends_the_previous_steps_marked_prefix_byte_for_byte(
    tmp_path: Path,
) -> None:
    """Two turns of a real agent, five requests: every step can read what the last one wrote.

    The requests are the ones `BaseAgent` actually assembled, turned into Anthropic bodies by
    the connector. For each pair of consecutive requests the tools, the system block and the
    messages up to the earlier request's conversation mark are the same bytes, apart from
    where the marks sit -- the cache key does not hash them. And no mark ever lands on a
    message holding the turn context, which is the one part that changes per step.

    Killed by: src/uclone_x/llm/connectors/anthropic.py :: last_block = messages_payload[stable[-1]]["content"][-1]
    Becomes: last_block = messages_payload[stable[0]]["content"][-1]
    """
    llm = _RecordingLLM([_call("c1"), _call("c2"), _answer("counted"), _call("c3")])
    agent = _agent(tmp_path, llm)
    await agent.start()
    assert (await agent.execute_turn("count twice")).is_completed
    assert (await agent.execute_turn("count once more")).is_completed
    assert len(llm.requests) == 5

    payloads = [_payload(r) for r in llm.requests]

    for payload in payloads:
        assert len(_marks(payload)) <= 4
        assert ".system[0]" in _marks(payload)
        assert _marks(payload["tools"]) == [f"[{len(payload['tools']) - 1}]"]
        # The turn context is present, and it is after the conversation mark, unmarked.
        holding = [i for i, m in enumerate(payload["messages"]) if _FACT in json.dumps(m)]
        assert holding, "the remembered fact never reached the request"
        assert not _marks([payload["messages"][i] for i in holding])
        marked = _marked_message(payload)
        if marked is not None:
            assert marked < min(holding)

    # Only the very first request, one user message, has no conversation mark.
    assert _marked_message(payloads[0]) is None
    assert all(_marked_message(p) is not None for p in payloads[1:])

    for earlier, later in zip(payloads, payloads[1:], strict=False):
        assert later["tools"] == earlier["tools"]
        assert later["system"] == earlier["system"]
        end = _marked_message(earlier)
        if end is None:
            continue
        later_end = _marked_message(later)
        assert later_end is not None and later_end > end
        assert _unmarked(later["messages"][: end + 1]) == _unmarked(earlier["messages"][: end + 1])


# ======================================================================================
# Cache counts on the usage record
# ======================================================================================


def _anthropic_reply(usage: dict[str, int]) -> Callable[[httpx.Request], httpx.Response]:
    return lambda _: httpx.Response(
        200,
        json={
            "id": "msg_1",
            "model": _MODEL,
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": usage,
        },
    )


_HI = LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="hi"),))


@pytest.mark.asyncio
async def test_anthropic_cache_counts_are_recorded_and_the_input_is_the_whole_prompt() -> None:
    """Anthropic's `input_tokens` is only the uncached rest; `TokenUsage.input_tokens` is all of it.

    Killed by: src/uclone_x/llm/connectors/anthropic.py :: return uncached + sum(c for c in (cache_creation, cache_read) if c is not None)
    Becomes: return uncached
    Killed by: src/uclone_x/llm/connectors/anthropic.py :: cache_creation, cache_read = _cache_counts(usage_data)
    Becomes: cache_creation, cache_read = None, None
    """
    reply = _anthropic_reply(
        {
            "input_tokens": 10,
            "cache_creation_input_tokens": 200,
            "cache_read_input_tokens": 3000,
            "output_tokens": 5,
        }
    )

    usage = (await _connector(reply).generate(_HI)).usage

    assert (usage.input_tokens, usage.output_tokens) == (3210, 5)
    assert (usage.cache_creation_input_tokens, usage.cache_read_input_tokens) == (200, 3000)


@pytest.mark.asyncio
async def test_anthropic_usage_without_cache_counts_records_none_not_zero() -> None:
    """A reply that says nothing about the cache leaves both counts unset, and out of the dump."""
    usage = (
        await _connector(_anthropic_reply({"input_tokens": 10, "output_tokens": 5})).generate(_HI)
    ).usage

    assert usage.input_tokens == 10
    assert usage.cache_creation_input_tokens is None
    assert usage.cache_read_input_tokens is None
    assert "cache_read_input_tokens" not in usage.model_dump_json()


def _sse(*events: dict[str, Any]) -> str:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


@pytest.mark.asyncio
async def test_an_anthropic_stream_reads_the_cache_counts_from_message_start() -> None:
    """The stream reports the prompt's counts once, on `message_start`.

    Killed by: src/uclone_x/llm/connectors/anthropic.py :: cache_creation, cache_read = _cache_counts(u)
    Becomes: cache_creation, cache_read = None, None
    """
    body = _sse(
        {
            "type": "message_start",
            "message": {
                "usage": {
                    "input_tokens": 4,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 1500,
                    "output_tokens": 1,
                }
            },
        },
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 2},
        },
        {"type": "message_stop"},
    )
    connector = _connector(lambda _: httpx.Response(200, content=body.encode("utf-8")))

    usages = [c.usage async for c in connector.stream(_HI) if c.usage is not None]

    assert len(usages) == 1
    usage = usages[0]
    assert usage.input_tokens == 1504
    assert (usage.cache_creation_input_tokens, usage.cache_read_input_tokens) == (0, 1500)


@pytest.mark.parametrize(
    ("details", "want"),
    [({"cached_tokens": 1024}, 1024), ({"cached_tokens": 0}, 0), ({}, None), (None, None)],
)
@pytest.mark.asyncio
async def test_openai_cached_tokens_are_recorded_when_reported(
    details: dict[str, int] | None, want: int | None
) -> None:
    """OpenAI's `prompt_tokens_details.cached_tokens` is already a subset of `prompt_tokens`.

    Killed by: src/uclone_x/llm/connectors/openai.py :: return reported_count(cast("dict[str, Any]", details), "cached_tokens")
    Becomes: return None
    """
    usage_body: dict[str, Any] = {"prompt_tokens": 2000, "completion_tokens": 3}
    if details is not None:
        usage_body["prompt_tokens_details"] = details
    reply = {
        "id": "chatcmpl-1",
        "model": "gpt-4o",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
        "usage": usage_body,
    }
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=reply))
    )
    connector = OpenAIConnector(model="gpt-4o", api_key="test-key", http_client=client)

    usage = (await connector.generate(_HI)).usage

    assert usage.input_tokens == 2000
    assert usage.cache_read_input_tokens == want
    assert usage.cache_creation_input_tokens is None


def test_a_usage_without_cache_counts_serializes_exactly_as_before() -> None:
    """Records written before #1371 and records from providers that report no cache match (#1844).

    Killed by: src/uclone_x/llm/models.py :: exclude_if=lambda created: created is None,
    Becomes: exclude_if=lambda created: False,
    Killed by: src/uclone_x/llm/models.py :: exclude_if=lambda read: read is None,
    Becomes: exclude_if=lambda read: False,
    """
    usage = TokenUsage(provider="anthropic", model=_MODEL, input_tokens=3, output_tokens=4)
    before = {
        "provider": "anthropic",
        "model": _MODEL,
        "input_tokens": 3,
        "output_tokens": 4,
    }

    dumped = usage.model_dump(mode="json")

    assert {k: dumped[k] for k in before} == before
    assert "cache_creation_input_tokens" not in dumped
    assert "cache_read_input_tokens" not in dumped
    assert TokenUsage.model_validate_json(usage.model_dump_json()) == usage
    with_cache = usage.model_copy(update={"cache_read_input_tokens": 0})
    assert with_cache.model_dump(mode="json")["cache_read_input_tokens"] == 0


def test_an_aggregate_sums_a_cache_count_only_when_every_usage_reported_it() -> None:
    """A sum over some unreported counts would read as a smaller cache hit than there was.

    Killed by: src/uclone_x/llm/models.py :: return sum(known) if len(known) == len(counts) else None
    Becomes: return sum(known)
    """

    def usage(read: int | None) -> TokenUsage:
        return TokenUsage(
            provider="anthropic",
            model=_MODEL,
            input_tokens=100,
            output_tokens=1,
            cache_read_input_tokens=read,
        )

    both = aggregate_token_usages([usage(40), usage(60)])
    partial = aggregate_token_usages([usage(40), usage(None)])

    assert both is not None and both.cache_read_input_tokens == 100
    assert partial is not None and partial.cache_read_input_tokens is None


def test_the_token_usage_schema_names_every_field_with_its_type() -> None:
    """The OpenAPI schema describes `TokenUsage`, not an untyped object (#1371 review).

    A wrap `model_serializer` returning `dict[str, object]` made pydantic publish the
    serialization schema as `{"type": "object", "additionalProperties": true}`, so every
    client generated from it lost the fields. The mutation below reinstates such a
    serializer, return annotation included, since pydantic types the schema from it.

    Killed by: src/uclone_x/llm/models.py :: count_source: TokenCountSource = Field(
    Becomes: omit_unreported = __import__("pydantic").model_serializer(mode="wrap")((lambda f: (setattr(f, "__annotations__", {"return": dict}), f)[1])(lambda self, handler: handler(self))); count_source: TokenCountSource = Field(
    """
    schema = TokenUsage.model_json_schema(mode="serialization")
    count = {"anyOf": [{"minimum": 0, "type": "integer"}, {"type": "null"}]}

    properties = schema.get("properties", {})

    assert set(properties) == set(TokenUsage.model_fields)
    for field in ("cache_creation_input_tokens", "cache_read_input_tokens"):
        assert {k: properties[field][k] for k in count} == count
    assert properties["input_tokens"]["type"] == "integer"


_GEMINI_CACHED_USAGE = {
    "promptTokenCount": 5000,
    "cachedContentTokenCount": 4096,
    "candidatesTokenCount": 7,
    "totalTokenCount": 5007,
}


def _gemini(handler: Callable[[httpx.Request], httpx.Response]) -> GeminiConnector:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GeminiConnector(model="gemini-2.5-flash", api_key="test-key", http_client=client)


def _gemini_body(usage_meta: dict[str, int]) -> dict[str, Any]:
    return {
        "candidates": [
            {"content": {"parts": [{"text": "ok"}], "role": "model"}, "finishReason": "STOP"}
        ],
        "usageMetadata": usage_meta,
    }


@pytest.mark.asyncio
async def test_gemini_cached_content_tokens_are_a_cache_read_inside_the_prompt() -> None:
    """Google's `promptTokenCount` already includes `cachedContentTokenCount`, so it is not re-added.

    Killed by: src/uclone_x/llm/connectors/gemini.py :: return reported_count(usage_meta, "cachedContentTokenCount")
    Becomes: return None
    """
    connector = _gemini(lambda _: httpx.Response(200, json=_gemini_body(_GEMINI_CACHED_USAGE)))

    usage = (await connector.generate(_HI)).usage

    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (5000, 7, 5007)
    assert usage.cache_read_input_tokens == 4096
    assert usage.cache_creation_input_tokens is None


@pytest.mark.asyncio
async def test_a_gemini_stream_records_the_same_cache_read_as_generate() -> None:
    """The stream's final usage carries the cache read `generate` does.

    Killed by: src/uclone_x/llm/connectors/gemini.py :: return reported_count(usage_meta, "cachedContentTokenCount")
    Becomes: return None
    """
    line = "data: " + json.dumps(_gemini_body(_GEMINI_CACHED_USAGE))
    connector = _gemini(lambda _: httpx.Response(200, text=line + "\n\n"))

    usages = [c.usage async for c in connector.stream(_HI) if c.usage is not None]

    assert [(u.input_tokens, u.cache_read_input_tokens) for u in usages] == [(5000, 4096)]


@pytest.mark.asyncio
async def test_a_gemini_usage_without_a_cached_count_records_none_not_zero() -> None:
    """Gemini's JSON omits a zero, so an absent count cannot be read as "no cache hit" (P6).

    Killed by: src/uclone_x/llm/connectors/gemini.py :: return reported_count(usage_meta, "cachedContentTokenCount")
    Becomes: return reported_count(usage_meta, "cachedContentTokenCount") or 0
    """
    plain = {k: v for k, v in _GEMINI_CACHED_USAGE.items() if k != "cachedContentTokenCount"}
    connector = _gemini(lambda _: httpx.Response(200, json=_gemini_body(plain)))

    usage = (await connector.generate(_HI)).usage

    assert usage.input_tokens == 5000
    assert usage.cache_read_input_tokens is None
    assert "cache_read_input_tokens" not in usage.model_dump(mode="json")
