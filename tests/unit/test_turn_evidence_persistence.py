"""The turn loop must not accept an unevidenced answer when evidence was required.

Measured on the first `frontier_live` baseline (#697): against problems declaring a median
`horizon` of **6** dependent steps, the agent took a median of **1** tool call, and reached
the declared horizon on **5 of 92** problems. Twenty-eight problems were answered with no
tool call at all — one of them stating outright that it had *inferred* a value "from common
system defaults" rather than reading the file it had a workspace for.

The loop ended because the model stopped asking for tools. Nothing compared that to what
the task needed, so a first silence was read as an answer.

These tests pin the two halves of the repair: the turn reports how many steps it took, and
an agent configured to require evidence gets one more round rather than an answer built on
nothing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import BaseModel, Field

from uclone_x.agent.base import EVIDENCE_REQUIRED_NUDGE, BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    MessageRole,
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


def _call(call_id: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        content=None,
        tool_calls=(ToolCallRequest(id=call_id, name="my_tool", arguments={"x": 1}),),
        usage=_USAGE,
        provenance=_PROV,
    )


class ScriptedLLM(BaseLLMConnector):
    """Replays a fixed script and records every request it was given."""

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


class DummyToolParams(BaseModel):
    x: int = Field(default=0)


class DummyEchoTool(BaseTool[DummyToolParams]):
    name = "my_tool"
    description = "Echo tool"

    def run(self, params: DummyToolParams, context: ToolContext) -> dict[str, Any]:
        return {"result": params.x + 1}


def _agent(llm: ScriptedLLM, *, require_evidence: bool, with_tools: bool = True) -> BaseAgent:
    config = AgentConfig(
        agent_id="agent_evidence",
        name="Agent",
        llm_config=AgentLLMConfig(model_name="dummy"),
        require_evidence_before_answer=require_evidence,
    )
    registry: ToolRegistry | None = None
    if with_tools:
        registry = ToolRegistry()
        registry.register(DummyEchoTool())
    return BaseAgent(config=config, llm=llm, tools=registry)


@pytest.mark.asyncio
async def test_the_turn_reports_how_many_steps_it_took() -> None:
    """`steps_taken` is the figure #697 had to reconstruct by hand.

    The count lived only in `_run_steps`, a private attribute, so a caller could see the
    tool executions a turn produced but not how many model round-trips produced them. The
    eval harness needs both to compare effort against a task's declared horizon.

    Killed by: src/uclone_x/agent/base.py :: self._run_steps = step
    Becomes: self._run_steps = step + 1
    """
    llm = ScriptedLLM([_call("tc_1"), _answer("done")])
    agent = _agent(llm, require_evidence=False)
    await agent.start()

    result = await agent.execute_turn("check something")

    assert result.steps_taken == 2, result.steps_taken
    assert len(result.tool_executions) == 1


@pytest.mark.asyncio
async def test_an_unevidenced_answer_is_sent_back_for_evidence() -> None:
    """The defect itself: one silence accepted as an answer.

    The model answers immediately, having called nothing. With evidence required the loop
    must put the question again rather than return it, and the nudge must reach the model
    as a message — a continuation the model never sees is the #584 failure in another
    costume.

    Killed by: src/uclone_x/agent/base.py :: EVIDENCE_REQUIRED_NUDGE,
    Becomes: "",
    """
    llm = ScriptedLLM([_answer("It is 100, inferred from common defaults."), _call("tc_1")])
    agent = _agent(llm, require_evidence=True)
    await agent.start()

    result = await agent.execute_turn("what is the configured maximum?")

    assert result.steps_taken >= 2, "the loop accepted the first unevidenced answer"
    assert len(result.tool_executions) >= 1, "the nudge did not lead to a tool call"
    sent = [m.content for req in llm.requests for m in req.messages if m.content]
    assert any(EVIDENCE_REQUIRED_NUDGE in str(c) for c in sent), "the model never saw the nudge"


@pytest.mark.asyncio
async def test_the_nudge_is_sent_once_and_then_the_answer_stands() -> None:
    """A model that will not use tools must not be asked forever.

    Without a bound this is an infinite loop against a model that always answers in prose,
    terminated only by the step budget — turning a cheap wrong answer into an expensive
    one. One nudge, then the second answer is returned as it is.

    Killed by: src/uclone_x/agent/base.py :: evidence_nudged = True
    """
    llm = ScriptedLLM([_answer("still no tools")])
    agent = _agent(llm, require_evidence=True)
    await agent.start()

    result = await agent.execute_turn("what is the configured maximum?")

    assert result.steps_taken == 2, result.steps_taken
    assert result.content == "still no tools"
    assert result.is_completed is True
    nudges = [
        m
        for req in llm.requests
        for m in req.messages
        if m.content and EVIDENCE_REQUIRED_NUDGE in str(m.content)
    ]
    assert len(nudges) == 1, f"expected exactly one nudge, saw {len(nudges)}"


@pytest.mark.asyncio
async def test_an_answer_that_used_a_tool_is_not_nudged() -> None:
    """Evidence was produced, so there is nothing to ask for.

    The condition is about the *turn*, not the step: a model that searched, then answered
    in a later step with no call, has grounded its answer. Nudging there would punish the
    behaviour the setting exists to produce.

    The answer quotes the tool's own output (`{"result": 2}`) rather than a figure of its
    own. It said `the value is 50` until the grounding check landed beside this one, and
    `50` appears in no tool output and in no prompt — so the turn was sent back, correctly,
    by a *different* rule than the one under test here, and the step count moved. The
    subject of this test is the evidence nudge; the data is changed so that it measures
    only that. What the old data actually demonstrated now has its own test,
    `test_an_answer_whose_figure_was_never_read_is_sent_back_once`.

    Killed by: src/uclone_x/agent/base.py :: and not tool_executions
    """
    llm = ScriptedLLM([_call("tc_1"), _answer("the tool returned result 2")])
    agent = _agent(llm, require_evidence=True)
    await agent.start()

    result = await agent.execute_turn("what is the configured maximum?")

    assert result.steps_taken == 2
    assert result.content == "the tool returned result 2"
    sent = [m.content for req in llm.requests for m in req.messages if m.content]
    assert not any(EVIDENCE_REQUIRED_NUDGE in str(c) for c in sent)


@pytest.mark.asyncio
async def test_an_agent_with_no_tools_is_not_nudged() -> None:
    """Asking for evidence an agent cannot gather is a loop with no exit but the budget.

    A toolless agent has nothing to call, so the nudge would be answered in prose every
    time. The guard has to be on the means, not only on the setting.

    Killed by: src/uclone_x/agent/base.py :: and nothing_found  # evidence nudge when tools found nothing
    Becomes: or nothing_found  # evidence nudge when tools found nothing
    """
    llm = ScriptedLLM([_answer("no tools here")])
    agent = _agent(llm, require_evidence=True, with_tools=False)
    await agent.start()

    result = await agent.execute_turn("what is the configured maximum?")

    assert result.steps_taken == 1
    assert result.content == "no tools here"


@pytest.mark.asyncio
async def test_the_default_leaves_the_loop_unchanged() -> None:
    """Off by default: a chat turn must not be told to go and read something.

    `require_evidence_before_answer` describes a benchmark or a verification agent, not
    every agent. A greeting answered without tools is correct behaviour and the loop must
    still end on it.

    Built without naming the flag at all. An earlier version of this test passed
    `require_evidence=False` explicitly, which exercises the argument and never the
    default -- so flipping the default to `True` left all six tests green. That is the
    shape of false `Killed by:` declaration this repository keeps finding in other
    people's work, and it was in mine.

    Killed by: src/uclone_x/agent/models.py :: require_evidence_before_answer: bool = Field(
    Becomes: require_evidence_before_answer: bool = Field(default=True) if True else Field(
    """
    llm = ScriptedLLM([_answer("hello")])
    config = AgentConfig(
        agent_id="agent_default",
        name="Agent",
        llm_config=AgentLLMConfig(model_name="dummy"),
    )
    assert config.require_evidence_before_answer is False, "the default is not off"

    registry = ToolRegistry()
    registry.register(DummyEchoTool())
    agent = BaseAgent(config=config, llm=llm, tools=registry)
    await agent.start()

    result = await agent.execute_turn("hi")

    assert result.steps_taken == 1
    assert result.content == "hello"
    assert len(llm.requests) == 1


@pytest.mark.asyncio
async def test_a_blocked_turn_does_not_report_the_previous_turns_steps() -> None:
    """The counter must be zeroed before any return can read it, not beside the loop.

    Two returns sit above the loop — a PRE_TURN hook block and a pre-loop exception — and
    both reported the *previous* turn's count while `_run_steps = 0` lived at the top of
    the loop body. A turn that never ran a step claiming three of them is worse than no
    figure at all, and #700 is about to compare exactly this number against each problem's
    declared horizon.

    Found in review of #702, after four of five return paths turned out to be unpinned.

    Killed by: src/uclone_x/agent/base.py :: stop_reason: TurnStopReason = "not_started"
    Becomes: stop_reason: TurnStopReason = "not_started"; self._run_steps = 3
    """
    llm = ScriptedLLM([_call("tc_1"), _call("tc_2"), _answer("done")])
    agent = _agent(llm, require_evidence=False)
    await agent.start()

    first = await agent.execute_turn("do some work")
    assert first.steps_taken >= 2, "the first turn needs steps for the bug to be visible"

    async def _raise_before_the_loop(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("blocked before any step ran")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(agent, "_prepare_turn_layers", _raise_before_the_loop)
    try:
        second = await agent.execute_turn("blocked")
    finally:
        monkeypatch.undo()

    assert second.is_completed is False
    assert second.steps_taken == 0, (
        f"a turn that ran no step reported {second.steps_taken}, which is the previous turn's count"
    )


@pytest.mark.asyncio
async def test_the_nudge_never_enters_the_conversation_history() -> None:
    """It is a runtime artifact, not something the user said.

    In `_history` the nudge persists through `persist_session`, reaches `agent.history` and
    the CLI transcript, and accumulates one synthetic user turn per turn for the rest of
    the session. The model still has to see it — it is turn-taking, so a system message
    will not do — so it is appended to the request and nowhere else.

    No `Killed by:` line, deliberately. The claim is an *absence* -- that nothing was
    written to `_history` -- and an absence has no single-substring edit that restores it:
    the mutation that breaks this test is adding an `append` back, which is a new line
    rather than a changed one. Recording that here rather than naming a substring that
    survives, which is what a first attempt at this docstring did.

    The positive half (the model does see it) is killed by
    `EVIDENCE_REQUIRED_NUDGE,` and is pinned by
    `test_an_unevidenced_answer_is_sent_back_for_evidence`.
    """
    llm = ScriptedLLM([_answer("no tools"), _answer("still none")])
    agent = _agent(llm, require_evidence=True)
    await agent.start()

    await agent.execute_turn("what is the configured maximum?")

    in_history = [
        m for m in agent.history if m.content and EVIDENCE_REQUIRED_NUDGE in str(m.content)
    ]
    assert in_history == [], "the nudge was written into the conversation history"

    sent = [m.content for req in llm.requests for m in req.messages if m.content]
    assert any(EVIDENCE_REQUIRED_NUDGE in str(c) for c in sent), "the model never saw it"


@pytest.mark.asyncio
async def test_a_self_contained_answer_survives_when_evidence_nudge_is_declined() -> None:
    """A self-contained question answered without tools must not become an abstention (#756).

    When require_evidence_before_answer is set, a turn that makes zero tool calls receives
    the evidence nudge. For self-contained questions (e.g. arithmetic, logic, reading
    comprehension from the prompt), the model declines the nudge by affirming that the
    question is answerable from what was given. The loop must accept the first answer and
    restore it, rather than turning it into an abstention or leaving duplicate assistant
    messages in history.

    Killed by: src/uclone_x/agent/base.py :: resp_content = first_answer
    Becomes: pass
    """
    llm = ScriptedLLM(
        [
            _answer("42"),
            _answer("This question is answerable from what you were given."),
        ]
    )
    agent = _agent(llm, require_evidence=True)
    await agent.start()

    result = await agent.execute_turn("What is 40 + 2?")

    assert result.content == "42"
    assert result.steps_taken == 2
    assert len(result.tool_executions) == 0
    assert result.is_completed is True
    assistant_msgs = [m.content for m in agent.history if m.role == MessageRole.ASSISTANT]
    assert assistant_msgs == ["42"]


@pytest.mark.asyncio
async def test_a_materially_equivalent_answer_declines_evidence_nudge() -> None:
    """When the post-nudge answer is materially the same, the first answer survives (#784).

    Small models often do not echo specific decline phrases. If the model reaffirms its
    answer (materially equivalent in prose without tool use), the nudge cost a turn and had
    nothing to add: the first answer survives and an EVIDENCE_NUDGE_DECLINED event is recorded.

    Killed by: src/uclone_x/agent/base.py :: "type": "EVIDENCE_NUDGE_DECLINED",
    Becomes: "type": "EVIDENCE_NUDGE",
    """
    llm = ScriptedLLM(
        [
            _answer("no"),
            _answer("The build does not run because of lockfile mismatch.\n\nno"),
        ]
    )
    agent = _agent(llm, require_evidence=True)
    await agent.start()

    result = await agent.execute_turn("Does the build run? yes or no.")

    assert result.content == "no"
    assert result.steps_taken == 2
    assert len(result.tool_executions) == 0
    assert result.is_completed is True
    assistant_msgs = [m.content for m in agent.history if m.role == MessageRole.ASSISTANT]
    assert assistant_msgs == ["no"]
    events = [e for e in agent.pending_durable_events if e.get("type") == "EVIDENCE_NUDGE_DECLINED"]
    assert len(events) == 1
