"""One tool call is not evidence: the loop must notice an answer that outran what it read.

## The half of #697 that #702 could not reach

#702 made the loop refuse an answer that consulted **nothing**. On the baseline that
produced the issue that rule covers 28 of 100 problems. The other 59 are the harder shape
and the one the issue's title is about: against a median declared `horizon` of **six**
dependent steps the median turn executed **one** tool call, so it passes a not-zero test
and still ends six steps early. The runtime saw a healthy tool call and an answer, and had
nothing to say about the distance between them.

`horizon` is evaluation metadata -- a production agent has none -- so the signal has to come
from the turn. It does: at the moment the model falls silent the runtime holds the answer
and everything the turn was shown. A path or a multi-digit number in the first that appears
nowhere in the second was not read from anything. `agent/grounding.py` extracts them.

## The three decisions these tests pin

*   **What is noticed** -- an unsupported specific, not a step count and not a horizon.
*   **What is done** -- the question goes back once, naming the specifics and offering two
    exits: check them, or say they are unchecked. Not a refusal, which would block a
    legitimately one-step task; not a log line, which would change nothing, and the whole
    defect is that the runtime *accepted* the answer.
*   **Where it stops** -- one latch per reason, two reasons, both set before the retry and
    never cleared inside a turn. A turn can therefore spend at most two extra steps
    whatever the model does, which `test_the_two_nudges_together_add_at_most_two_steps`
    measures against a model that never changes its answer.

The scripted connector below repeats its **last** response forever rather than running out.
A fake that exhausts a queue would end the loop by itself, which is the exact defect under
test, and the bound would read as proved when nothing had been bounded.
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
)
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


def _answer(text: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        content=text,
        tool_calls=(),
        usage=_USAGE,
        provenance=_PROV,
    )


def _call(call_id: str, value: int = 1) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        content=None,
        tool_calls=(ToolCallRequest(id=call_id, name="my_tool", arguments={"x": value}),),
        usage=_USAGE,
        provenance=_PROV,
    )


class ScriptedLLM(BaseLLMConnector):
    """Replays a script and then repeats its last response for as long as it is asked.

    The repetition is the point. A connector that raised or returned nothing once its queue
    emptied would terminate the loop on its own, and every bound asserted below would be
    the fixture's bound rather than the runtime's.
    """

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


def _agent(llm: ScriptedLLM, *, require_evidence: bool = True) -> BaseAgent:
    config = AgentConfig(
        agent_id="agent_grounding",
        name="Agent",
        llm_config=AgentLLMConfig(model_name="dummy"),
        require_evidence_before_answer=require_evidence,
    )
    registry = ToolRegistry()
    registry.register(EchoTool())
    return BaseAgent(config=config, llm=llm, tools=registry)


def _nudges_seen(llm: ScriptedLLM) -> list[str]:
    return [
        str(m.content)
        for req in llm.requests
        for m in req.messages
        if m.content and GROUNDING_REQUIRED_NUDGE_PREFIX in str(m.content)
    ]


@pytest.mark.asyncio
async def test_an_answer_whose_figure_was_never_read_is_sent_back_once() -> None:
    """The median-one-call shape: a tool ran, and the answer came from somewhere else.

    The echo tool returns `{"result": 2}`. The answer claims `100`, which appears in no
    tool output and in no prompt, so the turn did not read it. The existing evidence rule
    is silent here -- a tool executed -- which is why this needed its own condition.

    Killed by: src/uclone_x/agent/base.py :: grounding_nudge,
    Becomes: "",
    """
    llm = ScriptedLLM(
        [
            _call("tc_1"),
            _answer("The default is 100."),
            _answer("I could not verify that."),
        ]
    )
    agent = _agent(llm)
    await agent.start()

    result = await agent.execute_turn("what is the default?")

    assert _nudges_seen(llm), "the model was never asked about the unread figure"
    assert "100" in _nudges_seen(llm)[0], "the nudge did not name what was unsupported"
    assert result.steps_taken == 3, result.steps_taken
    assert result.content == "I could not verify that."


@pytest.mark.asyncio
async def test_an_answer_that_quotes_what_the_tool_returned_is_not_sent_back() -> None:
    """Evidence that the answer rests on: nothing to ask for.

    Without this the check taxes exactly the turns that did the work, and a nudge on every
    answer carrying a figure is a nudge that distinguishes nothing.

    Killed by: src/uclone_x/agent/base.py :: unsupported = unsupported_specifics(
    Becomes: unsupported = ("42",) or unsupported_specifics(
    """
    llm = ScriptedLLM([_call("tc_1", value=41), _answer("The tool returned 42.")])
    agent = _agent(llm)
    await agent.start()

    result = await agent.execute_turn("what does the tool return for 41?")

    assert _nudges_seen(llm) == [], _nudges_seen(llm)
    assert result.steps_taken == 2
    assert result.unsupported_claims == ()


@pytest.mark.asyncio
async def test_the_grounding_nudge_is_sent_once_and_then_the_answer_stands() -> None:
    """A model that will not change its answer must not be asked forever.

    It repeats `The default is 100.` for as long as it is asked. Unlatched, this runs to
    `max_steps` -- turning a cheap wrong answer into fifty round-trips of the same wrong
    answer.

    Killed by: src/uclone_x/agent/base.py :: grounding_nudged = True
    """
    llm = ScriptedLLM([_call("tc_1"), _answer("The default is 100.")])
    agent = _agent(llm)
    await agent.start()

    result = await agent.execute_turn("what is the default?")

    assert len(_nudges_seen(llm)) == 1, _nudges_seen(llm)
    assert result.steps_taken == 3, result.steps_taken
    assert result.content == "The default is 100."
    assert result.is_completed is True


@pytest.mark.asyncio
async def test_the_two_nudges_together_add_at_most_two_steps() -> None:
    """The terminating condition, measured rather than argued.

    Two reasons, one latch each, both set before the retry and never cleared within a turn.
    So a turn that would have taken two steps takes at most four, against a model that
    concedes nothing: it answers unevidenced, is nudged, calls a tool, answers with the
    same unread figure, is nudged again, and repeats it. `max_steps` is 50 and is not what
    stops this.

    Killed by: src/uclone_x/agent/base.py :: and not grounding_nudged
    """
    llm = ScriptedLLM(
        [
            _answer("The default is 100."),
            _call("tc_1"),
            _answer("The default is 100."),
        ]
    )
    agent = _agent(llm)
    await agent.start()

    result = await agent.execute_turn("what is the default?")

    evidence = [
        m
        for req in llm.requests
        for m in req.messages
        if m.content and EVIDENCE_REQUIRED_NUDGE in str(m.content)
    ]
    assert len(evidence) == 1, evidence
    assert len(_nudges_seen(llm)) == 1, _nudges_seen(llm)
    assert result.steps_taken == 4, result.steps_taken
    assert result.is_completed is True


@pytest.mark.asyncio
async def test_the_unsupported_specifics_are_reported_even_when_nothing_is_required() -> None:
    """Reported always, acted on only when configured.

    The observation is what #700 compares and what #733 needs after trap-based detection
    left the eval set: a turn that can say which of its own claims rest on nothing is a
    fabrication signal that costs no extra problem and cannot be gamed by an answer shaped
    to miss a trap string. Gating the *field* on the setting would make that unavailable to
    every agent that has not opted into the behaviour.

    Killed by: src/uclone_x/agent/base.py :: unsupported_claims=tuple(unsupported),
    """
    llm = ScriptedLLM([_call("tc_1"), _answer("The default is 100.")])
    agent = _agent(llm, require_evidence=False)
    await agent.start()

    result = await agent.execute_turn("what is the default?")

    assert result.steps_taken == 2, "the turn was nudged with the setting off"
    assert _nudges_seen(llm) == []
    assert result.unsupported_claims == ("100",), result.unsupported_claims


@pytest.mark.asyncio
async def test_the_models_own_earlier_answer_does_not_support_its_later_one() -> None:
    """A claim cannot be its own evidence.

    Three steps: it calls a tool, answers `100`, is nudged, and answers `100` again. The
    retry request carries the first answer -- it is rebuilt from history so the model can
    see what the nudge is about (#1420) -- and the second answer is checked against that
    request. With assistant turns left in the support set, the second `100` is found in the
    first and the claim certifies itself.

    This test used to take four steps, with a tool call after the nudge, because the retry
    was then built from the stale request and the first answer reached a request only
    through the tool step's rebuild from history. Since #1420 the rejected answer leaves
    history only when the retry's own message replaces it, so that four-step script no
    longer puts it in front of the check. An earlier two-step version had the same flaw: in each, the declaration named
    a line the script did not exercise.

    Killed by: src/uclone_x/agent/base.py :: if m.role is MessageRole.ASSISTANT or not m.content:
    Becomes: if not m.content:
    """
    llm = ScriptedLLM(
        [
            _call("tc_1"),
            _answer("The default is 100."),
            _answer("The default is 100."),
        ]
    )
    agent = _agent(llm)
    await agent.start()

    result = await agent.execute_turn("what is the default?")

    assert result.steps_taken == 3, result.steps_taken
    assert result.unsupported_claims == ("100",), result.unsupported_claims
    assert len(_nudges_seen(llm)) == 1


@pytest.mark.asyncio
async def test_the_nudge_quoting_a_specific_does_not_then_ground_it() -> None:
    """The check must not disarm itself one step after firing.

    The nudge *quotes the unsupported specifics back*, so it arrives in the request
    carrying exactly the tokens that were missing. Left in the support set it grounds them,
    and the second look at a repeated answer comes back clean: `unsupported_claims` empties
    on the turn that most needs it, and any later rule built on that field reads a
    fabrication as evidenced.

    Killed by: src/uclone_x/agent/base.py :: text = text.replace(nudge, "")
    Becomes: pass
    """
    llm = ScriptedLLM([_call("tc_1"), _answer("The default is 100.")])
    agent = _agent(llm)
    await agent.start()

    result = await agent.execute_turn("what is the default?")

    assert len(_nudges_seen(llm)) == 1
    assert "100" in _nudges_seen(llm)[0], "the nudge did not quote the specific back"
    assert result.unsupported_claims == ("100",), result.unsupported_claims


@pytest.mark.asyncio
async def test_the_grounding_nudge_never_enters_the_conversation_history() -> None:
    """Same reason as the evidence nudge: it is a runtime artifact, not a user line.

    In `_history` it persists through `persist_session`, reaches `agent.history` and the
    CLI transcript, and accumulates one synthetic user turn per turn for the rest of the
    session.

    No `Killed by:` line: the claim is an absence, and an absence has no single-substring
    edit that restores it -- the mutation that breaks this test is adding an `append`,
    which is a new line rather than a changed one. The positive half is pinned by
    `test_an_answer_whose_figure_was_never_read_is_sent_back_once`.
    """
    llm = ScriptedLLM([_call("tc_1"), _answer("The default is 100.")])
    agent = _agent(llm)
    await agent.start()

    await agent.execute_turn("what is the default?")

    in_history = [
        m for m in agent.history if m.content and GROUNDING_REQUIRED_NUDGE_PREFIX in str(m.content)
    ]
    assert in_history == [], "the nudge was written into the conversation history"
    assert _nudges_seen(llm), "the model never saw it"
