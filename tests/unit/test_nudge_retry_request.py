"""The request that re-asks after a nudge must carry the answer the nudge is about (#1420).

The evidence and grounding nudges used to be appended to `req`, the request that
*produced* the answer. The answer itself went to `_history` and never into `req.messages`,
so the retry asked the model to re-check an answer it could not see, and ended in two
consecutive `USER` messages (the span, then the nudge) -- which chat templates that
require alternating roles refuse or silently merge.

The retry is now rebuilt from history the way a tool step is, with the nudge in the
turn-context tail (the request-layering design, §5.5). These tests pin the three
things the issue asks for:

*   the retry request holds the rejected answer, immediately followed by the nudge, on
    both the evidence and the grounding path;
*   no request the agent sends on those paths has two adjacent `USER` messages;
*   the persisted shape of a nudged turn. The rejected answer leaves history only when the
    retry's own message takes its place -- the nudge never enters history (#702), so a rejected
    answer left behind would sit next to the retry's message as two `ASSISTANT` turns in
    a row, in the record and in every later request. It is still emitted as its
    `ASSISTANT_MESSAGE` durable event. A retry that writes nothing replaces nothing: the
    rejected answer stays the turn's record, so history never ends on the user's message.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import BaseModel, Field

from uclone_x.agent.base import (
    EVIDENCE_REQUIRED_NUDGE,
    GROUNDING_REQUIRED_NUDGE_PREFIX,
    BaseAgent,
    _is_evidence_nudge_declined,  # pyright: ignore[reportPrivateUsage]
)
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.models import (
    ChatMessage,
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry

_PROV = Provenance(
    path=ExecutionPath.PRIMARY,
    requested=ServiceRef(provider="agent", model="dummy"),
    served_by=ServiceRef(provider="agent", model="dummy"),
    attempts=(),
)
_USAGE = TokenUsage(provider="dummy", model="dummy", input_tokens=0, output_tokens=0)

#: Names a figure nothing in the turn contains, so the grounding check flags it; with no
#: tool call before it, the evidence check flags it first.
_REJECTED = "The configured maximum is 4096, inferred from common defaults."
_RETRIED = "I could not confirm the maximum; the tool returned 2."


def _answer(text: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        content=text,
        tool_calls=(),
        usage=_USAGE,
        provenance=_PROV,
    )


def _empty() -> ModelResponse:
    """A reply with no text and no call: what a reasoning-only reply reaches the loop as."""
    return _answer("")


def _call(call_id: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        content=None,
        tool_calls=(ToolCallRequest(id=call_id, name="my_tool", arguments={"x": 1}),),
        usage=_USAGE,
        provenance=_PROV,
    )


class RecordingLLM(BaseLLMConnector):
    """Replays a script, repeating its last response, and records every request."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        super().__init__()
        self.responses = responses
        self.requests: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return "dummy"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses[min(len(self.requests) - 1, len(self.responses) - 1)]

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:  # pragma: no cover
        yield StreamChunk(delta_content="")


class EchoParams(BaseModel):
    x: int = Field(default=0)


class EchoTool(BaseTool[EchoParams]):
    name = "my_tool"
    description = "Echo tool"

    def run(self, params: EchoParams, context: ToolContext) -> dict[str, Any]:
        return {"result": params.x + 1}


def _agent(llm: RecordingLLM, *, tail: bool = False) -> BaseAgent:
    """`tail` gives the turn a turn-context block (a memory fact) from its first step."""
    config = AgentConfig(
        agent_id="agent_nudge_retry",
        name="Agent",
        llm_config=AgentLLMConfig(model_name="dummy"),
        require_evidence_before_answer=True,
    )
    registry = ToolRegistry()
    registry.register(EchoTool())
    memory = None
    if tail:
        memory = CrossSessionMemory()
        memory.record_fact(
            subject="project",
            predicate="uses",
            object_value="postgres",
            provenance=_PROV,
            source_session_id="earlier",
        )
    return BaseAgent(config=config, llm=llm, tools=registry, memory=memory)


def _the_nudged_request(llm: RecordingLLM, marker: str) -> LLMRequest:
    carrying = [r for r in llm.requests if any(marker in str(m.content or "") for m in r.messages)]
    assert len(carrying) == 1, f"expected one request carrying the nudge, saw {len(carrying)}"
    return carrying[0]


def _adjacent_users(messages: tuple[ChatMessage, ...]) -> list[int]:
    return [
        i
        for i in range(1, len(messages))
        if messages[i].role is MessageRole.USER and messages[i - 1].role is MessageRole.USER
    ]


def _adjacent_assistants(messages: tuple[ChatMessage, ...] | list[ChatMessage]) -> list[int]:
    return [
        i
        for i in range(1, len(messages))
        if messages[i].role is MessageRole.ASSISTANT
        and messages[i - 1].role is MessageRole.ASSISTANT
    ]


@pytest.mark.asyncio
async def test_the_evidence_retry_carries_the_answer_it_rejects() -> None:
    """The model is asked to re-check an answer it can see, and the nudge follows it.

    Built from the stale request, the retry ended `USER(span) · USER(nudge)`: the answer
    being critiqued was absent. Built from history it ends
    `ASSISTANT(rejected) · USER(turn context + nudge)`.

    Killed by: src/uclone_x/agent/base.py :: retry_messages = assemble_request_messages(layers)
    Becomes: retry_messages = [*req.messages, ChatMessage(role=MessageRole.USER, content=nudge)]
    """
    llm = RecordingLLM([_answer(_REJECTED), _answer(_RETRIED)])
    agent = _agent(llm)
    await agent.start()

    await agent.execute_turn("what is the configured maximum?")

    retry = _the_nudged_request(llm, EVIDENCE_REQUIRED_NUDGE)
    assert retry.messages[-1].role is MessageRole.USER
    assert EVIDENCE_REQUIRED_NUDGE in str(retry.messages[-1].content)
    assert retry.messages[-2].role is MessageRole.ASSISTANT
    assert retry.messages[-2].content == _REJECTED, "the retry lacks the answer it critiques"


@pytest.mark.asyncio
async def test_the_grounding_retry_carries_the_answer_it_rejects() -> None:
    """The same, for the answer that named a specific no tool output contained.

    Here the rejected answer follows a tool call. The stale request ended in the tool
    result, so the nudge followed a `TOOL` message and the answer it names was never in the
    request. (Two adjacent `USER` messages arise on this path only when the turn carries a
    turn-context tail -- a plan, memory or a scoping notice -- which this agent has none of.)

    Killed by: src/uclone_x/agent/base.py :: retry_messages = assemble_request_messages(layers)
    Becomes: retry_messages = [*req.messages, ChatMessage(role=MessageRole.USER, content=nudge)]
    """
    llm = RecordingLLM([_call("tc_1"), _answer(_REJECTED), _answer(_RETRIED)])
    agent = _agent(llm)
    await agent.start()

    await agent.execute_turn("what is the configured maximum?")

    retry = _the_nudged_request(llm, GROUNDING_REQUIRED_NUDGE_PREFIX)
    assert retry.messages[-1].role is MessageRole.USER
    assert GROUNDING_REQUIRED_NUDGE_PREFIX in str(retry.messages[-1].content)
    assert retry.messages[-2].role is MessageRole.ASSISTANT
    assert retry.messages[-2].content == _REJECTED, "the retry lacks the answer it critiques"


_EVIDENCE = [_answer(_REJECTED), _answer(_RETRIED)]
_GROUNDING = [_call("tc_1"), _answer(_REJECTED), _answer(_RETRIED)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("script", "tail"),
    [
        pytest.param(_EVIDENCE, False, id="evidence"),
        pytest.param(_GROUNDING, False, id="grounding"),
        pytest.param(
            [_answer(_REJECTED), _call("tc_1"), _answer(_REJECTED), _answer(_RETRIED)],
            False,
            id="both",
        ),
        # With a turn-context block, step 1 carries it merged into the user's message and
        # later steps carry it as a `USER` tail of its own. Appended after either, the old
        # nudge made `USER · USER` on both paths -- the case the issue names.
        pytest.param(_EVIDENCE, True, id="evidence-with-turn-context"),
        pytest.param(_GROUNDING, True, id="grounding-with-turn-context"),
        pytest.param([_answer(_REJECTED), _empty()], False, id="evidence-empty-retry"),
        pytest.param([_answer(_REJECTED), _empty()], True, id="evidence-empty-retry-tail"),
        pytest.param([_call("tc_1"), _answer(_REJECTED), _empty()], False, id="grounding-empty"),
    ],
)
async def test_no_request_on_a_nudged_turn_has_two_adjacent_user_messages(
    script: list[ModelResponse], tail: bool
) -> None:
    """Strict chat templates refuse `USER · USER`; every request of the turn is checked,
    and the next turn's first request, which is where the persisted shape shows.

    Killed by: src/uclone_x/agent/base.py :: retry_messages = assemble_request_messages(layers)
    Becomes: retry_messages = [*req.messages, ChatMessage(role=MessageRole.USER, content=nudge)]
    """
    llm = RecordingLLM([*script, _answer("next turn")])
    agent = _agent(llm, tail=tail)
    await agent.start()

    await agent.execute_turn("what is the configured maximum?")
    assert len(llm.requests) >= 2, "no nudge fired, so nothing was tested"
    llm.responses = [_answer("next turn")]
    await agent.execute_turn("and the minimum?")

    for n, request in enumerate(llm.requests):
        assert _adjacent_users(request.messages) == [], f"request {n} has USER · USER"
        assert _adjacent_assistants(request.messages) == [], f"request {n} has ASSISTANT twice"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "script",
    [
        pytest.param([_answer(_REJECTED), _answer(_RETRIED)], id="evidence"),
        pytest.param([_call("tc_1"), _answer(_REJECTED), _answer(_RETRIED)], id="grounding"),
    ],
)
async def test_a_nudged_turn_persists_one_answer_the_one_it_returned(
    script: list[ModelResponse],
) -> None:
    """The persisted shape: the rejected answer is gone from history, and logged.

    Before, history kept both answers back to back, so the saved session replayed a turn
    answered twice, as two consecutive `ASSISTANT` messages, in every later request. The
    next turn's request is checked too: that is where a left-behind answer would surface.

    Killed by: src/uclone_x/agent/base.py :: self._history.pop()
    Becomes: pass
    """
    llm = RecordingLLM([*script, _answer("next turn")])
    agent = _agent(llm)
    await agent.start()

    result = await agent.execute_turn("what is the configured maximum?")

    assert result.content == _RETRIED
    answers = [
        m.content for m in agent.history if m.role is MessageRole.ASSISTANT and not m.tool_calls
    ]
    assert answers == [_RETRIED], f"persisted answers: {answers}"
    assert _adjacent_assistants(agent.history) == []
    logged = [
        e["content"] for e in agent.pending_durable_events if e.get("type") == "ASSISTANT_MESSAGE"
    ]
    assert _REJECTED in logged, "the rejected answer left no record"

    llm.responses = [_answer("next turn")]
    before = len(llm.requests)
    await agent.execute_turn("and the minimum?")
    next_request = llm.requests[before]
    assert _adjacent_assistants(next_request.messages) == []
    assert _REJECTED not in [m.content for m in next_request.messages]


@pytest.mark.asyncio
async def test_an_evidence_retry_that_writes_nothing_leaves_the_rejected_answer_standing() -> None:
    """An empty retry has no message to take the rejected answer's place, so it keeps it.

    Removed as soon as the nudge fired, the answer left history ending on the user's
    message, and every later request in the session opened with `USER · USER`. An empty
    reply is not hypothetical: a reasoning-only reply reaches the loop as one.

    Killed by: src/uclone_x/agent/base.py :: superseded = first_assistant_msg
    Becomes: superseded = first_assistant_msg and self._history.pop()
    """
    llm = RecordingLLM([_answer(_REJECTED), _empty()])
    agent = _agent(llm)
    await agent.start()

    await agent.execute_turn("what is the configured maximum?")

    assert agent.history[-1].role is MessageRole.ASSISTANT
    assert agent.history[-1].content == _REJECTED
    llm.responses = [_answer("next turn")]
    before = len(llm.requests)
    await agent.execute_turn("and the minimum?")
    assert _adjacent_users(llm.requests[before].messages) == []


@pytest.mark.asyncio
async def test_a_grounding_retry_that_writes_nothing_leaves_the_rejected_answer_standing() -> None:
    """The same on the grounding path, where the loss was quieter: history ended on the
    tool result, and the turn's answer was missing from the record altogether.

    Killed by: src/uclone_x/agent/base.py :: superseded = (
    Becomes: superseded = self._history.pop() if resp_content else None or (
    """
    llm = RecordingLLM([_call("tc_1"), _answer(_REJECTED), _empty()])
    agent = _agent(llm)
    await agent.start()

    await agent.execute_turn("what is the configured maximum?")

    answers = [
        m.content for m in agent.history if m.role is MessageRole.ASSISTANT and not m.tool_calls
    ]
    assert answers == [_REJECTED], f"persisted answers: {answers}"
    assert agent.history[-1].role is MessageRole.ASSISTANT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retried",
    [
        pytest.param(_RETRIED, id="plain"),
        pytest.param(
            "This is answerable from what was given: the tool returned 2.", id="decline-phrase"
        ),
    ],
)
async def test_after_an_empty_first_answer_the_retrys_answer_stands(retried: str) -> None:
    """An empty first answer leaves nothing to fall back to, so the retry cannot "decline".

    Treated as declined, the retry's answer was swapped for the empty first one: the turn
    returned "", nothing entered history, and every later request opened `USER · USER`.

    Killed by: src/uclone_x/agent/base.py :: and first_answer
    Becomes: and first_answer is not None
    """
    llm = RecordingLLM([_empty(), _answer(retried)])
    agent = _agent(llm)
    await agent.start()

    result = await agent.execute_turn("what is the configured maximum?")

    assert any(EVIDENCE_REQUIRED_NUDGE in str(m.content) for m in llm.requests[1].messages)
    assert result.content == retried
    answers = [
        m.content for m in agent.history if m.role is MessageRole.ASSISTANT and not m.tool_calls
    ]
    assert answers == [retried], f"persisted answers: {answers}"
    assert _adjacent_users(tuple(agent.history)) == []
    llm.responses = [_answer("next turn")]
    before = len(llm.requests)
    await agent.execute_turn("and the minimum?")
    assert _adjacent_users(llm.requests[before].messages) == []


@pytest.mark.parametrize("first", ["", "...", "**"])
def test_a_first_answer_with_no_words_matches_no_retry(first: str) -> None:
    """Stripped to nothing, the first answer made the phrase pattern `\\b\\b`, which
    matches every retry, so any retry read as a restatement of it.

    Killed by: src/uclone_x/agent/base.py :: if clean_first and re.search(pattern, lowered_second):
    Becomes: if re.search(pattern, lowered_second):
    """
    assert (
        _is_evidence_nudge_declined("I could not confirm it; the tool returned 2.", first) is False
    )
