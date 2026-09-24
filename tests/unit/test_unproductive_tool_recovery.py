"""A turn whose tools found nothing must get the same second chance as one that ran none.

The first live baseline: 28 problems of 100 made no tool call, and **38 stopped at exactly
one** — the largest bucket. A search missed, the model read the miss as an absence, and the
turn ended (#698). The evidence nudge (#697) fired only when `tool_executions` was empty,
so the 28 got a second chance and the 38 did not. The grounding nudge (#739) covers the
other side — an answer that outran what was read — and needs the answer to state a
specific, which "the line does not appear in the file" does not.

So the largest bucket in the measurement got neither.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import BaseModel

from uclone_x.agent.base import EVIDENCE_REQUIRED_NUDGE, BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)
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


class _Params(BaseModel):
    pass


class _EmptySearch(BaseTool[_Params]):
    """The measured shape: succeeds, matches nothing."""

    name = "empty_search"
    description = "matches nothing"

    def run(self, params: _Params, context: ToolContext) -> dict[str, Any]:
        return {"query": "anything", "total_matches": 0, "matches": []}


class _FindingSearch(BaseTool[_Params]):
    name = "finding_search"
    description = "matches something"

    def run(self, params: _Params, context: ToolContext) -> dict[str, Any]:
        return {"query": "anything", "total_matches": 1, "matches": [{"line": 7}]}


class _Scripted(BaseLLMConnector):
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


def _answer(text: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        content=text,
        tool_calls=(),
        usage=_USAGE,
        provenance=_PROV,
    )


def _call(name: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        content=None,
        tool_calls=(ToolCallRequest(id="tc_1", name=name, arguments={}),),
        usage=_USAGE,
        provenance=_PROV,
    )


def _agent(llm: _Scripted, tool: BaseTool[Any]) -> BaseAgent:
    registry = ToolRegistry()
    registry.register(tool)
    return BaseAgent(
        config=AgentConfig(
            agent_id="a_recovery",
            name="Agent",
            llm_config=AgentLLMConfig(model_name="dummy"),
            require_evidence_before_answer=True,
        ),
        llm=llm,
        tools=registry,
    )


def _nudges(llm: _Scripted) -> list[str]:
    return [
        str(m.content)
        for req in llm.requests
        for m in req.messages
        if m.content and EVIDENCE_REQUIRED_NUDGE in str(m.content)
    ]


@pytest.mark.asyncio
async def test_a_search_that_found_nothing_earns_a_second_chance() -> None:
    """The 38-problem bucket, which previously got no nudge at all.

    One call, empty result, then an answer. `tool_executions` is non-empty, so the evidence
    nudge used not to fire; the answer states no specific, so the grounding nudge does not
    either.

    Killed by: src/uclone_x/agent/base.py :: and nothing_found  # evidence nudge when tools found nothing
    Becomes: and False  # evidence nudge when tools found nothing
    """
    llm = _Scripted([_call("empty_search"), _answer("The line does not appear in the file.")])
    agent = _agent(llm, _EmptySearch())
    await agent.start()

    result = await agent.execute_turn("find the value")

    assert _nudges(llm), "a turn whose only tool found nothing was accepted as answered"
    assert result.steps_taken >= 3, result.steps_taken


@pytest.mark.asyncio
async def test_a_search_that_found_something_is_left_alone() -> None:
    """Evidence was produced. Asking again would punish the behaviour the setting wants.

    This is the other side of the disjointness: once a tool has produced something, an
    answer that goes past it is the *grounding* nudge's business, not this one's.

    Killed by: src/uclone_x/agent/base.py :: _tool_outcome_of(record) == ToolOutcome.PRODUCTIVE.value
    Becomes: False
    """
    llm = _Scripted([_call("finding_search"), _answer("The value is on line 7.")])
    agent = _agent(llm, _FindingSearch())
    await agent.start()

    await agent.execute_turn("find the value")

    assert not _nudges(llm), "a turn with a productive tool result was nudged for evidence"


@pytest.mark.asyncio
async def test_the_unproductive_stop_is_named_in_the_log() -> None:
    """Three endings that look alike from outside must not look alike in the log.

    "answered", "gave up having found nothing", and "gave up having looked at nothing" are
    different diagnoses, and only the log can carry the difference to whoever reads the run
    afterwards.

    Killed by: src/uclone_x/agent/base.py :: stop_reason = "model_stopped_after_nudge"
    Becomes: stop_reason = "model_stopped"
    """
    llm = _Scripted(
        [
            _call("empty_search"),
            _answer("still nothing"),
            _answer("still nothing"),
            _answer("still nothing"),
        ]
    )
    agent = _agent(llm, _EmptySearch())
    await agent.start()

    await agent.execute_turn("find the value")
    ends = [e for e in agent.pending_durable_events if e.get("type") == "TURN_END"]

    assert ends
    assert ends[-1]["stop_reason"] == "model_stopped_after_nudge"


@pytest.mark.asyncio
async def test_the_nudge_still_fires_once() -> None:
    """Widening what counts as "nothing found" must not widen how often it asks.

    Killed by: src/uclone_x/agent/base.py :: and not evidence_nudged
    Becomes:
    """
    llm = _Scripted([_call("empty_search"), _answer("nothing"), _answer("still nothing")])
    agent = _agent(llm, _EmptySearch())
    await agent.start()

    await agent.execute_turn("find the value")

    assert len(_nudges(llm)) == 1, f"expected one nudge, saw {len(_nudges(llm))}"


@pytest.mark.asyncio
async def test_the_specific_nudge_goes_first() -> None:
    """When both apply to the same answer, grounding fires and evidence defers.

    A turn that searched, found nothing, and answered with a figure satisfies both
    conditions. The tie goes to grounding because it can name the thing: *"the figure 42
    appears in nothing you read"* is strictly more useful than *"your search found
    nothing"*.

    Both may still fire across a turn -- that is #739's decision, recorded by
    `model_stopped_after_both_nudges` -- but not for the same answer, and not in the order
    that would make the generic one pre-empt the specific one.

    Killed by: src/uclone_x/agent/base.py :: and not defer_to_grounding
    Becomes:
    """
    llm = _Scripted(
        [
            _call("empty_search"),
            _answer("The value is 42, on line 137 of models.py."),
            _answer("I could not verify it."),
        ]
    )
    agent = _agent(llm, _EmptySearch())
    await agent.start()

    await agent.execute_turn("find the value")

    fired = [
        str(e["type"])
        for e in agent.pending_durable_events
        if str(e.get("type")) in {"EVIDENCE_NUDGE", "GROUNDING_NUDGE"}
    ]
    assert fired[0] == "GROUNDING_NUDGE", (
        f"the answer named a specific and the generic nudge pre-empted it: {fired}"
    )
