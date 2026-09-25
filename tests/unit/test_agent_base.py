"""Unit tests for BaseAgent and TurnResult persona attribution (FR-13.4, Issue #442)."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Sequence, Sized
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field, ValidationError

from uclone_x.agent import BaseAgent, TurnBudgetExceededError
from uclone_x.agent.base import TURN_CONTEXT_HEADER
from uclone_x.agent.hooks import (
    BaseHook,
    HookContext,
    HookDecision,
)
from uclone_x.agent.models import (
    BASE_PERSONA_TOOLS,
    AgentConfig,
    AgentContext,
    AgentLLMConfig,
    AgentState,
    PersonaDefinition,
    TurnResult,
)
from uclone_x.agent.prompts import (
    HERMES_STEERABILITY_POLICY,
    QWEN_STEERABILITY_POLICY,
    STEERABILITY_POLICY,
)
from uclone_x.agent.session import AnchorAuthor, AnchorProvenance, SessionState, SessionStore
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.engine.event_bus import EventBus, EventType
from uclone_x.errors import BudgetExceededError, LLMStreamInterruptedError
from uclone_x.llm.budget import TokenBudgetManager
from uclone_x.llm.compactor import estimate_reply_tokens
from uclone_x.llm.connectors.base import resolve_token_counts
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import (
    ChatMessage,
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
    TokenBudget,
    TokenCountSource,
    TokenUsage,
    ToolCallRequest,
    ToolDefinition,
)
from uclone_x.llm.protocols import LLMProviderProtocol
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext, ToolResultStatus
from uclone_x.tools.registry import ToolRegistry
from uclone_x.tools.tool_scoper import LexicalToolScoper


def test_turn_result_persona_defaults_to_none() -> None:
    """Verify TurnResult.persona defaults to None when omitted."""
    turn = TurnResult(
        turn_index=1,
        content="Testing default",
        provenance=None,
    )
    assert turn.persona is None


def test_turn_result_persona_explicit_assignment() -> None:
    """Verify TurnResult carries explicit persona attribution."""
    turn = TurnResult(
        turn_index=1,
        content="Attributed turn",
        provenance=None,
        persona="champion",
    )
    assert turn.persona == "champion"


def test_base_agent_persona_configuration_and_aliases() -> None:
    """Verify BaseAgent reads persona from config or init arguments and supports property aliases."""
    cfg_with_persona = AgentConfig(
        agent_id="agt1",
        name="Agent One",
        persona="champion",
    )
    agent1 = BaseAgent(config=cfg_with_persona, llm=MockLLMConnector())
    assert agent1.persona == "champion"
    assert agent1.persona_name == "champion"

    cfg_no_persona = AgentConfig(
        agent_id="agt2",
        name="Agent Two",
    )
    agent2 = BaseAgent(config=cfg_no_persona, persona="scout", llm=MockLLMConnector())
    assert agent2.persona == "scout"
    assert agent2.persona_name == "scout"

    agent3 = BaseAgent(config=cfg_no_persona, persona_name="critic", llm=MockLLMConnector())
    assert agent3.persona == "critic"
    assert agent3.persona_name == "critic"

    # Property setters
    agent3.persona = "custom_persona"
    assert agent3.persona == "custom_persona"
    assert agent3.persona_name == "custom_persona"

    agent3.persona_name = "another_persona"
    assert agent3.persona == "another_persona"
    assert agent3.persona_name == "another_persona"


@pytest.mark.asyncio
async def test_base_agent_populates_persona_on_successful_turn() -> None:
    """Verify execute_turn populates persona in TurnResult for configured agent."""
    cfg = AgentConfig(
        agent_id="champion",
        name="Champion",
        persona="champion",
    )
    connector = MockLLMConnector(default_response="Champion response.")
    agent = BaseAgent(config=cfg, llm=connector)

    result = await agent.execute_turn("Collaborate with me")
    assert result.is_completed is True
    assert result.persona == "champion"
    assert result.content == "Champion response."


@pytest.mark.asyncio
async def test_base_agent_step_and_run_turn_aliases_populate_persona() -> None:
    """Verify step and run_turn aliases execute turn and populate persona attribution."""
    cfg = AgentConfig(
        agent_id="critic",
        name="Critic",
        persona="critic",
    )
    connector = MockLLMConnector(default_response="Critic review.")
    agent = BaseAgent(config=cfg, llm=connector)

    step_result = await agent.step("Review this design")
    assert step_result.is_completed is True
    assert step_result.persona == "critic"

    turn_result = await agent.run_turn("Review second step")
    assert turn_result.is_completed is True
    assert turn_result.persona == "critic"


@pytest.mark.asyncio
async def test_base_agent_persona_none_p6_compliance() -> None:
    """Verify an agent with unstated persona yields TurnResult.persona is None without fabricating defaults (P6)."""
    cfg = AgentConfig(
        agent_id="generic_agent",
        name="Generic Agent",
        # persona is None
    )
    connector = MockLLMConnector(default_response="Generic answer.")
    agent = BaseAgent(config=cfg, llm=connector)

    assert agent.persona is None
    result = await agent.execute_turn("Hello")
    assert result.is_completed is True
    assert result.persona is None


@pytest.mark.asyncio
async def test_base_agent_populates_persona_on_error_failover() -> None:
    """Verify failover TurnResult carries configured persona attribution on connector error."""
    cfg = AgentConfig(
        agent_id="champion",
        name="Champion",
        persona="champion",
    )

    class FailingConnector(MockLLMConnector):
        async def generate(self, request: LLMRequest) -> ModelResponse:
            raise RuntimeError("Upstream connector timeout")

    agent = BaseAgent(config=cfg, llm=FailingConnector())
    result = await agent.execute_turn("Trigger failure")
    assert result.is_completed is False
    assert result.error == "Upstream connector timeout"
    assert result.persona == "champion"


@pytest.mark.asyncio
async def test_base_agent_populates_persona_on_hook_block() -> None:
    """Verify TurnResult carries persona attribution when pre_turn hook blocks execution."""
    from uclone_x.agent.hooks import HookAction

    class BlockingHook(BaseHook):
        def __init__(self) -> None:
            super().__init__(name="blocker")

        async def on_pre_turn(self, context: HookContext) -> HookDecision:
            return HookDecision(action=HookAction.BLOCK, reason="Input violation")

    cfg = AgentConfig(
        agent_id="scout",
        name="Scout",
        persona="scout",
        hooks=(BlockingHook(),),
    )
    connector = MockLLMConnector()
    agent = BaseAgent(config=cfg, llm=connector)

    result = await agent.execute_turn("Blocked prompt")
    assert result.is_completed is False
    assert result.persona == "scout"
    assert "Turn execution blocked by hook" in result.content


class DummyEchoParams(BaseModel):
    text: str = Field(default="", description="Text to echo")


class DummyEchoTool(BaseTool[DummyEchoParams]):
    name = "dummy_echo"
    description = "Echo tool for agent unit tests"

    def run(self, params: DummyEchoParams, context: ToolContext) -> str:
        return f"Echoed: {params.text}"


class CannedToolResponseConnector(MockLLMConnector):
    """Mock connector that emits predetermined tool calls and content."""

    def __init__(
        self,
        tool_calls: tuple[ToolCallRequest, ...] = (),
        content: str | None = None,
    ) -> None:
        super().__init__()
        self._canned_tool_calls = tool_calls
        self._canned_content = content

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self._call_count += 1
        service_ref = ServiceRef(provider="mock", model="mock-model")
        # The canned calls are spent once. The agent takes agent steps until the model
        # stops asking for tools (P4, amended 2026-09-05), so a connector that repeats its
        # tool call forever describes a model that never answers — it ran to the step
        # ceiling instead of completing.
        tools_already_ran = any(m.role is MessageRole.TOOL for m in request.messages)
        return ModelResponse(
            content="Done." if tools_already_ran else self._canned_content,
            tool_calls=() if tools_already_ran else self._canned_tool_calls,
            usage=TokenUsage(
                provider="mock",
                model="mock-model",
                input_tokens=10,
                output_tokens=10,
                total_tokens=20,
            ),
            finish_reason=FinishReason.TOOL_CALLS if self._canned_tool_calls else FinishReason.STOP,
            model_name="mock-model",
            provenance=Provenance(
                path=ExecutionPath.PRIMARY,
                requested=service_ref,
                served_by=service_ref,
            ),
        )


@pytest.mark.asyncio
async def test_base_agent_tool_calls_appended_to_history_before_tool_results() -> None:
    """Verify assistant message requesting tool calls is appended to history before tool results."""
    reg = ToolRegistry()
    reg.register(DummyEchoTool())

    cfg = AgentConfig(
        agent_id="test_tool_agent",
        name="ToolAgent",
        system_prompt="",
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )

    call1 = ToolCallRequest(id="call_1", name="dummy_echo", arguments={"text": "first"})
    call2 = ToolCallRequest(id="call_2", name="dummy_echo", arguments={"text": "second"})
    connector = CannedToolResponseConnector(
        content="Requesting tool execution",
        tool_calls=(call1, call2),
    )
    agent = BaseAgent(config=cfg, llm=connector, tools=reg)

    turn_result = await agent.execute_turn("Run tools")
    assert turn_result.is_completed is True

    # USER -> ASSISTANT(tool_calls) -> TOOL -> TOOL -> ASSISTANT(answer).
    # The trailing assistant message is the point of the step loop: the model sees what
    # its tools returned and answers from them (P4, amended 2026-09-05). Before the loop
    # the turn ended at the tool results and produced no answer at all.
    history = agent.history
    assert len(history) == 5
    assert history[4].role == MessageRole.ASSISTANT
    assert history[4].content == "Done."
    assert history[4].tool_calls == ()
    assert history[0].role == MessageRole.USER
    assert history[0].content == "Run tools"

    assistant_msg = history[1]
    assert assistant_msg.role == MessageRole.ASSISTANT
    assert assistant_msg.content == "Requesting tool execution"
    assert assistant_msg.tool_calls == (call1, call2)
    assert assistant_msg in agent._history  # pyright: ignore[reportPrivateUsage]

    # Tool execution results follow immediately after the assistant message
    tool_msg_1 = history[2]
    assert tool_msg_1.role == MessageRole.TOOL
    assert tool_msg_1.tool_call_id == "call_1"
    assert tool_msg_1.name == "dummy_echo"
    assert "Echoed: first" in (tool_msg_1.content or "")

    tool_msg_2 = history[3]
    assert tool_msg_2.role == MessageRole.TOOL
    assert tool_msg_2.tool_call_id == "call_2"
    assert tool_msg_2.name == "dummy_echo"
    assert "Echoed: second" in (tool_msg_2.content or "")

    # Every tool message resolves to a tool call in the preceding assistant message
    tool_call_ids = {tc.id for tc in assistant_msg.tool_calls}
    assert tool_msg_1.tool_call_id in tool_call_ids
    assert tool_msg_2.tool_call_id in tool_call_ids


@pytest.mark.asyncio
async def test_base_agent_tool_only_turn_content_none_appends_assistant_tool_calls() -> None:
    """Verify tool-only turn with content=None appends assistant message with tool calls."""
    reg = ToolRegistry()
    reg.register(DummyEchoTool())

    cfg = AgentConfig(
        agent_id="test_silent_agent",
        name="SilentAgent",
        system_prompt="",
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )

    call_none = ToolCallRequest(id="call_none", name="dummy_echo", arguments={"text": "silent"})
    connector = CannedToolResponseConnector(
        content=None,
        tool_calls=(call_none,),
    )
    agent = BaseAgent(config=cfg, llm=connector, tools=reg)

    turn_result = await agent.execute_turn("Execute silently")
    assert turn_result.is_completed is True

    history = agent.history
    # USER -> ASSISTANT(tool_calls) -> TOOL -> ASSISTANT(answer): the loop lets the
    # model answer from what its tool returned (P4, amended 2026-09-05).
    assert len(history) == 4
    assert history[0].role == MessageRole.USER

    assistant_msg = history[1]
    assert assistant_msg.role == MessageRole.ASSISTANT
    assert assistant_msg.content is None
    assert assistant_msg.tool_calls == (call_none,)

    tool_msg = history[2]
    assert tool_msg.role == MessageRole.TOOL
    assert tool_msg.tool_call_id == "call_none"
    assert tool_msg.tool_call_id in {tc.id for tc in assistant_msg.tool_calls}


@pytest.mark.asyncio
async def test_base_agent_tool_only_turn_content_empty_string_appends_assistant_tool_calls() -> (
    None
):
    """Verify tool-only turn with empty string content appends assistant message with tool calls."""
    reg = ToolRegistry()
    reg.register(DummyEchoTool())

    cfg = AgentConfig(
        agent_id="test_empty_str_agent",
        name="EmptyStrAgent",
        system_prompt="",
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )

    call_empty = ToolCallRequest(id="call_empty", name="dummy_echo", arguments={"text": "empty"})
    connector = CannedToolResponseConnector(
        content="",
        tool_calls=(call_empty,),
    )
    agent = BaseAgent(config=cfg, llm=connector, tools=reg)

    turn_result = await agent.execute_turn("Execute empty string content")
    assert turn_result.is_completed is True

    history = agent.history
    # USER -> ASSISTANT(tool_calls) -> TOOL -> ASSISTANT(answer): the loop lets the
    # model answer from what its tool returned (P4, amended 2026-09-05).
    assert len(history) == 4
    assert history[0].role == MessageRole.USER

    assistant_msg = history[1]
    assert assistant_msg.role == MessageRole.ASSISTANT
    assert assistant_msg.content is None
    assert assistant_msg.tool_calls == (call_empty,)

    tool_msg = history[2]
    assert tool_msg.role == MessageRole.TOOL
    assert tool_msg.tool_call_id == "call_empty"
    assert tool_msg.tool_call_id in {tc.id for tc in assistant_msg.tool_calls}


@pytest.mark.asyncio
async def test_base_agent_tool_messages_resolve_to_preceding_assistant_tool_calls() -> None:
    """Verify every tool message in history resolves to a tool call in the preceding assistant message."""
    reg = ToolRegistry()
    reg.register(DummyEchoTool())

    cfg = AgentConfig(
        agent_id="test_resolution_agent",
        name="ResolutionAgent",
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )

    call_a = ToolCallRequest(id="call_a", name="dummy_echo", arguments={"text": "step 1"})
    call_b = ToolCallRequest(id="call_b", name="dummy_echo", arguments={"text": "step 2"})
    connector = CannedToolResponseConnector(
        content="Two tool calls dispatched",
        tool_calls=(call_a, call_b),
    )
    agent = BaseAgent(config=cfg, llm=connector, tools=reg)

    await agent.execute_turn("Start resolution test")

    # Traverse history and verify tool resolution invariant
    history = agent.history
    for idx, msg in enumerate(history):
        if msg.role == MessageRole.TOOL:
            # Look backwards for the preceding assistant message
            preceding_assistant: ChatMessage | None = None
            for p_idx in range(idx - 1, -1, -1):
                if history[p_idx].role == MessageRole.ASSISTANT:
                    preceding_assistant = history[p_idx]
                    break
            assert preceding_assistant is not None, (
                "Tool message orphaned without preceding assistant message"
            )
            matching_calls = [
                tc for tc in preceding_assistant.tool_calls if tc.id == msg.tool_call_id
            ]
            assert len(matching_calls) == 1, (
                f"Tool call ID {msg.tool_call_id} not found in preceding assistant"
            )
            assert matching_calls[0].name == msg.name


@pytest.mark.asyncio
async def test_max_turns_does_not_bound_a_conversation() -> None:
    """A person may keep talking. The ceiling is not measured against them.

    `max_turns` is P4's bounded-execution ceiling: it bounds a run of turns the agent
    takes *without returning to whoever asked*. It was checked against `_turn_counter`,
    a lifetime count incremented once per `execute_turn` and reset only by
    `reset_session` — so with the tool loop living inside a single call and nothing
    re-entering `execute_turn` on its own, it bounded exactly one thing: how many
    messages a person could ever send. The shipped default of 50 killed `sess_default`
    mid-conversation and told its user to reset.

    Mutations this exists to catch: check the ceiling against `_turn_counter` again, or
    hoist the step counter out of `execute_turn` so it accumulates across requests.

    (An earlier docstring here named "drop the `continuation` guard". There is no such
    guard: the step counter is a local of `execute_turn` and resets by construction. The
    line was carried over from the design that flag belonged to.)
    """
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="conversation-agent",
            name="Conversation Agent",
            max_turns=2,
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=MockLLMConnector(),
    )

    for i in range(1, 6):
        res = await agent.execute_turn(f"Turn {i}")
        assert res.is_completed is True, f"turn {i} was refused: {res.error}"
        assert res.turn_index == i
        assert agent.state == AgentState.IDLE
        # Each request's run costs exactly one step — the answer — and the next request
        # starts over. Five messages against `max_turns=2` therefore never approach the
        # ceiling. Asserting `0` here would pin nothing: `0` is also the value the
        # property had when it was assigned once in `__init__` and never updated.
        assert agent.run_turns == 1, "one answered request is one agent step, every time"
        assert agent.run_steps == 1

    assert agent.turn_counter == 5


@pytest.mark.asyncio
async def test_max_turns_bounds_a_self_driven_run() -> None:
    """The ceiling fires on the thing P4 names: steps the agent takes without returning.

    No `continuation` flag and no protocol change: the step counter is local to one
    externally-initiated request, so it resets with the request by construction. A model
    that keeps asking for tools is the self-driven run, and it terminates with an explicit
    envelope rather than spinning.

    Mutations this exists to catch: remove the ceiling; move the counter outside the loop
    so it stops counting steps; stop mirroring the count onto `run_turns`; or relax the
    check from `>` to `>=`.

    That last one used to survive. The refusal names `max_turns` in its message either
    way, and `is_completed is False` is true either way, so nothing distinguished a
    ceiling of 3 that admits 3 steps from one that admits 2 — a whole step of headroom
    silently gone. The `run_turns` assertion below is what now separates them.
    """

    class NeverSatisfiedConnector(MockLLMConnector):
        """A model that asks for the same tool forever — the runaway P4 forbids."""

        async def generate(self, request: LLMRequest) -> ModelResponse:
            return ModelResponse(
                content="calling again",
                tool_calls=(ToolCallRequest(id="c", name="dummy_echo", arguments={"text": "x"}),),
                usage=TokenUsage(
                    provider="mock", model="mock-model", input_tokens=1, output_tokens=1
                ),
                finish_reason=FinishReason.TOOL_CALLS,
                model_name="mock-model",
                provenance=Provenance.primary("mock", "mock-model"),
            )

    reg = ToolRegistry()
    reg.register(DummyEchoTool())
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="runaway",
            name="Runaway",
            max_turns=3,
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=NeverSatisfiedConnector(),
        tools=reg,
    )

    res = await agent.execute_turn("go")
    assert res.is_completed is False
    assert res.error == "Agent step budget exceeded: maximum 3 steps in a single request"
    assert agent.state == AgentState.ERROR
    # The reported count is steps *taken*, so it stops at the ceiling rather than at the
    # refused fourth attempt. `turns_remaining` therefore bottoms out at exactly zero.
    assert agent.run_turns == 3
    assert agent.run_steps == 3
    assert agent.steps_remaining == 0
    assert agent.turns_remaining == 0

    # The person can still speak: the ceiling bounded the run, not the conversation.
    agent.transition_to(AgentState.IDLE)
    agent2 = BaseAgent(
        config=AgentConfig(
            agent_id="runaway2",
            name="Runaway2",
            max_turns=3,
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=MockLLMConnector(),
    )
    for _ in range(5):
        assert (await agent2.execute_turn("hello")).is_completed is True


def test_step_budget_error_is_reachable_under_both_names() -> None:
    """The canonical name must be importable from `uclone_x.agent`, not just the alias.

    `agent/__init__.py` re-exported only the deprecated spelling, so
    `from uclone_x.agent import StepBudgetExceededError` raised ImportError.
    """
    from uclone_x.agent import StepBudgetExceededError as FromAgent
    from uclone_x.errors import StepBudgetExceededError as FromErrors

    assert FromAgent is FromErrors
    assert TurnBudgetExceededError is FromErrors

    # Canonical kwargs, canonical attributes.
    err = FromAgent("boom", max_steps=9, current_steps=9)
    assert (err.max_steps, err.current_steps) == (9, 9)
    # Deprecated kwargs land on the same attributes rather than a diverging pair.
    legacy = FromAgent("boom", max_turns=4, current_turns=3)
    assert (legacy.max_steps, legacy.current_steps) == (4, 3)
    assert (legacy.max_turns, legacy.current_turns) == (4, 3)


def test_turn_budget_exceeded_error_taxonomy() -> None:
    """Verify TurnBudgetExceededError taxonomy, attributes, and propagation."""
    err = TurnBudgetExceededError(
        "Turn budget exceeded: maximum 5 turns reached",
        max_turns=5,
        current_turns=5,
    )
    assert isinstance(err, BudgetExceededError)
    assert err.max_turns == 5
    assert err.current_turns == 5
    assert str(err) == "Turn budget exceeded: maximum 5 turns reached"


@pytest.mark.asyncio
async def test_token_ceiling_refuses_a_turn_once_the_session_budget_is_spent() -> None:
    """The token ceiling is consulted, and refuses. Pinned because it previously was not.

    `TokenBudgetManager.enforce_budget` raised from the day it was written and no turn
    ever called it, so an interactive session had no token bound at all. Reviewer
    `rev-senior-107` demonstrated the gap by deleting the whole budget block and watching
    the suite stay green: line coverage read 100% while the refusal branch had never
    executed under any test.

    Mutation this exists to catch: delete the `enforce_budget` call, or the
    `record_usage` call that makes the ceiling reachable.
    """
    budget = TokenBudgetManager()
    budget.configure_session("sess_broke", max_tokens=10)
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="broke",
            name="Broke",
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=MockLLMConnector(),
        context=AgentContext(session_id="sess_broke", agent_id="broke"),
        budget=budget,
    )

    # An open budget completes, so the refusal below is attributable to the ceiling and
    # not to something else in the turn path.
    first = await agent.execute_turn("hello")
    assert first.is_completed is True

    # Book tokens past the ceiling, then the next turn must not reach the model.
    budget.record_usage(
        "sess_broke",
        TokenUsage(
            provider="mock",
            model="mock-model",
            input_tokens=500,
            output_tokens=500,
        ),
    )
    agent.transition_to(AgentState.IDLE)
    refused = await agent.execute_turn("again")
    assert refused.is_completed is False
    assert "limit exceeded" in (refused.error or "").lower(), refused.error


@pytest.mark.asyncio
async def test_max_turns_zero_refuses_before_any_model_invocation() -> None:
    """A ceiling of zero admits no step at all, and says so.

    P4: "If it reaches `max_turns` ... execution must terminate immediately with an
    explicit error or partial result envelope — silent or unbounded spinning is strictly
    forbidden." `max_turns=0` is the degenerate end of that, and `AgentConfig` carries no
    lower bound that rules it out, so it is reachable configuration rather than a
    hypothetical. It had a test before the step loop landed and lost it in the rewrite.

    Mutation this exists to catch: guard the check as
    `if self._config.max_turns > 0 and step > self._config.max_turns`, the reading that
    treats zero as "unlimited". That is the plausible rewrite here, and it is the one no
    other test can see: every other ceiling in the suite is positive, so the added
    conjunct is true for all of them and the mutant survives everywhere else.
    """
    connector = MockLLMConnector()
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="zero-ceiling",
            name="Zero Ceiling",
            max_turns=0,
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=connector,
    )

    res = await agent.execute_turn("anything")

    assert res.is_completed is False
    assert res.error == "Agent step budget exceeded: maximum 0 steps in a single request"
    assert res.content == ""
    assert agent.state == AgentState.ERROR
    # Weak on its own — `0` is also the constructor's value, and a `max_turns=0` agent can
    # never have run a step for it to be stale from. `test_max_turns_bounds_a_self_driven_run`
    # carries the discrimination that `run_turns` stops at the ceiling.
    assert agent.run_turns == 0, "no step was taken, so none may be reported as taken"
    # The refusal is *before* the model, not after it: a ceiling that spends a call and
    # then refuses bounds nothing.
    assert connector.call_count == 0


@pytest.mark.asyncio
async def test_token_ceiling_refusal_propagates_as_itself_not_as_a_failover() -> None:
    """A budget refusal is not a provider failover and must not be reported as one.

    P6, "Error classification never authorises substitution", classification table
    (`docs/principles/details/p6-fail-fast-observability.md`):

        | Failure class                    | Retry / failover eligible? | Substitution allowed? |
        | Quota or budget ceiling exceeded | No — must propagate        | **No**                |

    `BudgetExceededError` fell into the generic handler and came back as `path=FAILOVER`,
    `served_by=agent.core/error_handler`, with an `attempts` record, a `failover.event`
    span and a `PROVIDER_FAILOVER` bus notice — describing a provider substitution attempt
    that never happened. A refusal dressed as a substitution is the silent substitution
    P6 forbids, running in the reporting direction.

    The existing ceiling test asserts only `is_completed is False` and `"limit exceeded"`,
    and both are true of the FAILOVER envelope as well — which is why it stayed green
    while the classification was wrong. This one asserts the envelope's classification.

    The transport-failure control below is what keeps that honest: it proves this agent,
    on this bus, does still produce a real failover envelope, so the absence of one for
    the budget refusal is a property of the classification and not of the fixture.
    Attribution is read from the result envelope, not from the trace.

    Mutation this exists to catch: drop the `except BudgetExceededError` clause.
    """
    bus = EventBus()

    # Control: a transport failure IS failover-shaped, on the same agent shape and bus.
    failing_llm = MagicMock(spec=LLMProviderProtocol)
    failing_llm.generate = AsyncMock(side_effect=ConnectionResetError("socket reset"))
    control = BaseAgent(
        config=AgentConfig(
            agent_id="failover-control",
            name="Failover Control",
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=cast(LLMProviderProtocol, failing_llm),
        bus=bus,
    )
    control_sub = bus.subscribe("agent.chat.*")
    control_result = await control.execute_turn("boom")
    assert control_result.is_completed is False
    assert control_result.provenance is not None
    assert control_result.provenance.path is ExecutionPath.FAILOVER
    assert control_result.provenance.served_by is not None
    assert control_result.provenance.served_by.provider == "agent.core"
    control_notice = await asyncio.wait_for(control_sub.get(), timeout=2.0)
    assert control_notice.type is EventType.PROVIDER_FAILOVER

    # Subject: the same envelope path, entered by a budget ceiling instead.
    manager = TokenBudgetManager()
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="budget-refusal",
            name="Budget Refusal",
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=MockLLMConnector(),
        context=AgentContext(session_id="sess_refusal", agent_id="budget-refusal"),
        bus=bus,
        budget=manager,
    )
    manager.set_budget(
        "sess_refusal",
        TokenBudget(
            max_tokens=1_000,
            used_input_tokens=600,
            used_output_tokens=500,
        ),
    )
    sub = bus.subscribe("agent.chat.*")

    result = await agent.execute_turn("hello")

    assert result.is_completed is False
    assert result.error == "Session token limit exceeded: 1100/1000"
    assert result.provenance is None, (
        "a budget ceiling is not failover-eligible; it must not carry a failover envelope"
    )
    assert [
        s
        for s in cast(TelemetryTracer, agent.tracer).get_completed_spans()
        if s.name == "failover.event"
    ] == []
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(sub.get(), timeout=0.2)


# --------------------------------------------------------------------------------------
# A listener must not move the ceiling (#916).
#
# A non-`None` `stream_callback` puts `_invoke_model` on the `llm.stream` branch. That is
# a choice of *endpoint*, made by whoever is watching, and the ceiling is enforced from
# what the endpoint reports. Measured at `99625f8` with the connector below: where the
# stream reports the provider's count, both runs are refused at turn 4 (150/120). Where
# the stream omits it, the unlistened run is refused at turn 4 and the listened run was
# **never refused in ten turns and booked 0 tokens** — the `len // 4` stand-in was
# discarded rather than booked, so the step was charged nothing.
# --------------------------------------------------------------------------------------

_SCRIPTED_REPLY = "The same reply, whoever happens to be listening to it."
_SCRIPTED_COUNT = TokenUsage(
    provider="ollama",
    model="scripted-model",
    input_tokens=30,
    output_tokens=20,
)


class _UsageScriptedConnector(MockLLMConnector):
    """The same reply and the same provider count from both endpoints — unless told not to.

    `stream_reports_usage=False` is the provider #916 is about: an OpenAI-compatible server
    that answers a stream and ignores `include_usage`.
    """

    def __init__(self, *, stream_reports_usage: bool) -> None:
        super().__init__()
        self._stream_reports_usage = stream_reports_usage

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        return ModelResponse(
            content=_SCRIPTED_REPLY,
            usage=_SCRIPTED_COUNT,
            finish_reason=FinishReason.STOP,
            model_name="scripted-model",
            provenance=Provenance.primary(provider="ollama", model="scripted-model"),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        for word in _SCRIPTED_REPLY.split(" "):
            yield StreamChunk(delta_content=f"{word} ")
        yield StreamChunk(
            finish_reason=FinishReason.STOP,
            usage=_SCRIPTED_COUNT if self._stream_reports_usage else None,
        )


async def _run_until_refused(
    *,
    stream_reports_usage: bool,
    listening: bool,
) -> tuple[int | None, str | None, tuple[TokenUsage, ...]]:
    """One conversation against a 120-token ceiling: (refused at, reason, ledger)."""
    manager = TokenBudgetManager()
    manager.set_budget("sess_916", TokenBudget(max_tokens=120))
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="listened",
            name="Listened",
            llm_config=AgentLLMConfig(model_name="scripted-model"),
        ),
        llm=_UsageScriptedConnector(stream_reports_usage=stream_reports_usage),
        context=AgentContext(session_id="sess_916", agent_id="listened"),
        budget=manager,
    )
    heard: list[str] = []

    async def listener(event: str, data: dict[str, Any]) -> None:
        if event == "token":
            heard.append(str(data.get("content", "")))

    for turn in range(1, 11):
        res = await agent.execute_turn(
            f"turn {turn}", stream_callback=listener if listening else None
        )
        if not res.is_completed:
            return turn, res.error, manager.get_turn_history("sess_916")
        agent.transition_to(AgentState.IDLE)
    # Not vacuous: a listened run that never streamed would be the unlistened run twice.
    assert not listening or heard, "the listener was never called, so nothing was compared"
    return None, None, manager.get_turn_history("sess_916")


@pytest.mark.asyncio
async def test_a_listener_does_not_move_the_ceiling_when_the_stream_reports_usage() -> None:
    """The same conversation is refused at the same turn, for the same reason, watched or not.

    This is the test #916 asks for, in the case where it can hold exactly: the provider
    counts on both endpoints, so both runs must book the provider's figures and nothing
    else. It passed at `99625f8`, and is kept because it is the invariant the fix is
    stated against, not a regression proof on its own.

    Ignoring the stream's usage chunk makes the listened run book an estimate (at
    `99625f8`, nothing at all), so it is refused elsewhere or never:

    Killed by: src/uclone_x/agent/base.py :: last_usage = chunk.usage
    Becomes: last_usage = None
    """
    unwatched = await _run_until_refused(stream_reports_usage=True, listening=False)
    watched = await _run_until_refused(stream_reports_usage=True, listening=True)

    assert unwatched[:2] == (4, "Session token limit exceeded: 150/120")
    assert watched == unwatched
    assert {u.count_source for u in watched[2]} == {TokenCountSource.PROVIDER}


@pytest.mark.asyncio
async def test_a_stream_without_usage_is_charged_as_a_labelled_estimate() -> None:
    """A step whose stream reported no count is charged, and the figure says it is an estimate.

    The ceiling rule (#916, recorded in the room UI design document §6.7): an estimate is
    **charged like a count**, and it is labelled on the `TokenUsage` and on any refusal it
    contributed to. Discarding it — what `99625f8` did — leaves every watched conversation
    on such a provider unbounded while the unwatched one is bounded.

    The two runs are *not* asserted to be refused at the same turn: the listened run has no
    provider count to be refused on, so equality would have to be manufactured by choosing
    a reply whose `len // 4` happens to match. What is asserted is what can be true — both
    are bounded, and every figure the listened run was refused on is marked as estimated.

    Labelling it `PROVIDER` drops the note, and dropping the note leaves a refusal that
    reads as provider-counted.

    Killed by: src/uclone_x/agent/base.py :: count_source=TokenCountSource.ESTIMATE,
    Becomes: count_source=TokenCountSource.PROVIDER,
    Killed by: src/uclone_x/llm/budget.py :: if estimated_steps
    Becomes: if False
    """
    unwatched = await _run_until_refused(stream_reports_usage=False, listening=False)
    watched = await _run_until_refused(stream_reports_usage=False, listening=True)

    # Headless, the stream is never asked, so nothing about it can matter.
    assert unwatched[:2] == (4, "Session token limit exceeded: 150/120")
    assert {u.count_source for u in unwatched[2]} == {TokenCountSource.PROVIDER}

    refused_at, reason, ledger = watched
    assert refused_at is not None, "a watched conversation ran ten turns past its ceiling"
    assert ledger, "the watched run was refused without booking anything"
    assert {u.count_source for u in ledger} == {TokenCountSource.ESTIMATE}
    booked = sum(u.total_tokens for u in ledger)
    assert reason == (
        f"Session token limit exceeded: {booked}/120 "
        f"(includes {len(ledger)} estimated step(s): the provider reported no usage)"
    )


def test_agent_config_step_budget_aliasing() -> None:
    """AgentConfig and SubAgentSpec max_steps and max_turns sync transparently."""
    from uclone_x.agent.models import SubAgentSpec

    cfg1 = AgentConfig(agent_id="a1", name="A1", max_steps=15)
    assert cfg1.max_steps == 15
    assert cfg1.max_turns == 15

    cfg2 = AgentConfig(agent_id="a2", name="A2", max_turns=25)
    assert cfg2.max_steps == 25
    assert cfg2.max_turns == 25

    spec1 = SubAgentSpec(name="s1", role="worker", system_prompt="p", max_steps=10)
    assert spec1.max_steps == 10
    assert spec1.max_turns == 10

    spec2 = SubAgentSpec(name="s2", role="worker", system_prompt="p", max_turns=12)
    assert spec2.max_steps == 12
    assert spec2.max_turns == 12


def test_dynamic_persona_definition_and_utilization() -> None:
    """A test defines a persona at runtime and asserts that subsequent agent turns/steps utilize it.

    Killed by: src/uclone_x/agent/base.py :: body = persona.system_prompt if persona is not None else config_prompt
    Becomes: body = config_prompt
    """
    from uclone_x.agent import BaseAgent
    from uclone_x.agent.models import AgentConfig, PersonaDefinition

    agent = BaseAgent(
        config=AgentConfig(
            agent_id="test_agent", name="Test Agent", system_prompt="Default system prompt"
        )
    )
    assert agent.effective_system_prompt == "Default system prompt"

    persona = PersonaDefinition(
        name="dynamic_reviewer",
        role="Code Reviewer",
        system_prompt="You are a strict code reviewer. Do not accept bad code.",
    )

    # Define it dynamically
    agent.define_persona(persona)

    # Switch agent to this persona
    agent.persona = "dynamic_reviewer"

    # The agent should now utilize the persona's system prompt
    assert (
        agent.effective_system_prompt == "You are a strict code reviewer. Do not accept bad code."
    )

    # Resetting a session should use the effective system prompt
    session = agent.reset_session("session_1")
    assert session.messages[0].content == "You are a strict code reviewer. Do not accept bad code."


#: Container attributes of `BaseAgent` that legitimately outlive `reset_session`, each with
#: the reason it is not per-session state. Anything *not* listed here is treated by
#: `test_reset_session_empties_every_per_session_accumulator` as a per-session accumulator
#: and must be empty after a reset of the only session that scenario has (#670).
#:
#: `_session_compactors` is deliberately **not** listed. It was, on the first version of
#: this test, as "a cache owned by the agent … not conversational state" — and that was
#: false: the cached `ContextCompactor` carries `_superseded_ledger_count` and
#: `_supersession_reasons` (`llm/compactor.py`) into text the model reads, across the reset.
#: `reset_session` now evicts the reset session's compactor, so the scan pins that rather
#: than excusing it.
_AGENT_LIFETIME_CONTAINERS: dict[str, str] = {
    "_sessions": "the session registry itself; it holds the freshly reset session",
    "_processing_errors": (
        "a bounded ring of absorbed event-loop errors, exposed for diagnostics; a reset "
        "of a conversation is not a reason to forget the agent's own failures"
    ),
}


def _agent_container(agent: BaseAgent, name: str) -> Sized:
    """Read one of the agent's internal containers by name, without a private attribute.

    The invariant under test is a property of the agent's *internal* state, so the test has
    to look at it. Going through `vars` keeps the read in one documented place rather than
    scattering protected-member access (and its pyright suppressions) over the assertions.
    """
    return cast(Sized, vars(agent)[name])


def _measured_size(value: object) -> int | None:
    """How full `value` is, or `None` if it is not something this scan can measure.

    `Sized` alone is too narrow *and* too wide. Too wide, because `str` and `bytes` are
    sized and a non-empty string attribute is not an accumulator. Too narrow, because the
    likeliest next shape for a pending-events queue — `asyncio.Queue`, `queue.Queue` — has
    no `__len__` at all and reports through `qsize()`. Both are measured here.
    """
    if isinstance(value, str | bytes):
        return None
    if isinstance(value, Sized):
        return len(value)
    qsize = getattr(value, "qsize", None)
    if callable(qsize):
        return int(cast(int, qsize()))
    return None


def test_reset_session_empties_every_per_session_accumulator() -> None:
    """A reset must leave nothing of the previous session behind, not merely its history.

    `evaluation/answerer.py` resets between problems precisely so a score cannot depend on
    the order the problems happen to be in. `_pending_durable_events` broke that guarantee
    while history kept it: appended to by every turn, drained only by `persist_session`,
    and so carried across the boundary a reset is supposed to be (#670).

    The assertion is deliberately the invariant and not the symptom. Checking only
    `_pending_durable_events` would pass the moment the next accumulator is added; checking
    only the history passed against the defect this pins. Every container attribute the
    agent holds is scanned, and each one that survives a reset must be justified by name in
    `_AGENT_LIFETIME_CONTAINERS`.

    **What the scan guarantees, exactly**, because the first version of this docstring
    claimed more than it delivers:

    * It enumerates `vars(agent)` — the instance `__dict__`. Class-level containers are
      invisible to it, and it would raise `TypeError` against a `__slots__` agent.
    * It measures anything with `__len__` or `qsize()` that is not a `str`/`bytes`
      (`_measured_size`). It does not recurse: state held *inside* an object the agent
      caches is out of reach, which is why `_session_compactors` had to be evicted in
      `reset_session` rather than reasoned about here.
    * It catches a new accumulator only if **this scenario fills it**. One turn on a
      tool-less agent fills `_pending_durable_events`, and `_record_loaded_skill` primes
      `_loaded_skills` (#676). One container it scans — `_stranded_event_counts` — cannot
      become non-empty here at all: exercising it needs an error-absorbed leg that is not
      built yet.

    Killed by: src/uclone_x/agent/base.py :: if event.get("session_id") != sid
    Killed by: src/uclone_x/agent/base.py :: self._session_compactors.pop(sid, None)
    Becomes: pass
    Killed by: src/uclone_x/agent/base.py :: self._loaded_skills.clear()  # reset active session skills
    Becomes: pass  # reset active session skills
    """
    agent = BaseAgent(
        config=AgentConfig(agent_id="reset_inv", name="Reset Invariant"),
        llm=MockLLMConnector(),
        context=AgentContext(session_id="sess_reset_inv", agent_id="reset_inv"),
    )
    agent._record_loaded_skill("sample_skill")  # pyright: ignore[reportPrivateUsage]
    assert "sample_skill" in agent.loaded_skills

    asyncio.run(agent.execute_turn("first problem"))
    pending_after_first_turn = len(_agent_container(agent, "_pending_durable_events"))
    assert pending_after_first_turn > 0, (
        "the durable-event queue never filled, so this test would assert nothing about a "
        "reset clearing it"
    )

    agent.reset_session()

    survivors: dict[str, int] = {}
    for name, value in vars(agent).items():
        if name in _AGENT_LIFETIME_CONTAINERS:
            continue
        size = _measured_size(value)
        if size is not None and size > 0:
            survivors[name] = size
    assert survivors == {}, (
        f"state survived reset_session: {survivors}. Either clear it in reset_session, or "
        f"add it to _AGENT_LIFETIME_CONTAINERS with the reason it is not session state."
    )
    assert agent.loaded_skills == frozenset()
    assert agent.run_steps == 0
    assert [m.role for m in agent.get_session().messages] == [MessageRole.SYSTEM]

    # The measurement from the issue: 10 -> 16 entries across two turns with a reset
    # between them. The second turn must start the queue from empty, so the count after it
    # is the count one turn produces -- not the sum of both.
    asyncio.run(agent.execute_turn("second problem"))
    assert len(_agent_container(agent, "_pending_durable_events")) == pending_after_first_turn, (
        "the second turn's durable events were appended to the first turn's, across a "
        "reset: a durable store wired to this queue would persist events from a session "
        "that was explicitly ended."
    )


def test_reset_session_clears_loaded_skills() -> None:
    """A skill loaded in one session must not leak into the next across reset_session (#676).

    Killed by: src/uclone_x/agent/base.py :: self._loaded_skills.clear()  # reset active session skills
    Becomes: pass  # reset active session skills
    """
    agent = BaseAgent(
        config=AgentConfig(agent_id="skill_reset", name="Skill Reset"),
        llm=MockLLMConnector(),
        context=AgentContext(session_id="sess_skill_reset", agent_id="skill_reset"),
    )
    agent._record_loaded_skill("sample_skill")  # pyright: ignore[reportPrivateUsage]
    assert "sample_skill" in agent.loaded_skills

    agent.reset_session()

    assert agent.loaded_skills == frozenset()


def test_delete_session_clears_loaded_skills() -> None:
    """Deleting the active session must clear loaded skills (#676).

    Killed by: src/uclone_x/agent/base.py :: self._loaded_skills.clear()  # delete active session skills
    Becomes: pass  # delete active session skills
    """
    agent = BaseAgent(
        config=AgentConfig(agent_id="skill_del", name="Skill Delete"),
        llm=MockLLMConnector(),
        context=AgentContext(session_id="sess_skill_del", agent_id="skill_del"),
    )
    agent._record_loaded_skill("sample_skill")  # pyright: ignore[reportPrivateUsage]
    assert "sample_skill" in agent.loaded_skills

    agent.delete_session()

    assert agent.loaded_skills == frozenset()


def test_reset_of_one_session_leaves_another_sessions_pending_events_alone() -> None:
    """A reset must not clear work belonging to a session the caller did not name.

    The first version of the #670 fix cleared the whole agent-wide queue and zeroed
    `_run_steps` unconditionally, ignoring the `sid` it had just resolved. `reset_session`
    explicitly supports targeting a non-active session, and `ui/app.py` calls it with a
    caller-supplied id from the clear-history path — so clearing the history of an idle
    side session destroyed the *live* session's unpersisted durable events before any
    store saw them, and dropped the step count the UI renders its budget bar from to zero
    for a run still in flight.

    Deleting an audit queue is not the conservative reading of an ambiguity: mis-filed
    events can be repaired by hand and deleted ones cannot. The fix is to attribute rather
    than to guess — every turn stamps `session_id` onto the events it produces — so this
    asserts on the attribution as well as on the survival.

    Killed by: src/uclone_x/agent/base.py :: if event.get("session_id") != sid
    Becomes: if event.get("session_id") == sid
    Killed by: src/uclone_x/agent/base.py :: reset, anchor_provenance=self._resolved_persona()
    Becomes: reset, anchor_provenance=self._resolved_persona()); self._sessions[self._context.session_id] = _LiveSession.from_state(reset, anchor_provenance=self._resolved_persona()
    Killed by: src/uclone_x/agent/base.py :: event["session_id"] = turn_session_id
    """
    agent = BaseAgent(
        config=AgentConfig(agent_id="reset_scope", name="Reset Scope"),
        llm=MockLLMConnector(),
        context=AgentContext(session_id="sA", agent_id="reset_scope"),
    )

    asyncio.run(agent.execute_turn("a question for sA"))
    pending = cast("list[dict[str, object]]", vars(agent)["_pending_durable_events"])
    assert pending, "no durable events were queued, so this test would assert nothing"
    assert {event.get("session_id") for event in pending} == {"sA"}, (
        "a turn's durable events must name the session they were produced on, or a "
        "scoped clear has nothing to scope by"
    )
    pending_for_sa = len(pending)
    steps_for_sa = agent.run_steps
    assert steps_for_sa > 0, "the run took no steps, so the step-count assertion is empty"

    # sB has never run a turn and contributed nothing to the queue.
    agent.reset_session("sB")

    assert len(cast("Sized", vars(agent)["_pending_durable_events"])) == pending_for_sa, (
        "resetting sB destroyed sA's unpersisted durable events; a store wired to this "
        "queue would never see the turn that produced them"
    )
    assert agent.run_steps == steps_for_sa, (
        "resetting sB zeroed the active session's step count, so the budget the UI shows "
        "for a run in flight drops to zero"
    )
    assert [m.role for m in agent.get_session("sA").messages] != [MessageRole.SYSTEM], (
        "resetting sB cleared sA's history"
    )


class _RecordingSessionStore:
    def __init__(self) -> None:
        self.saved_events_by_session: dict[str, list[dict[str, Any]]] = {}

    def load(self, session_id: str) -> SessionState | None:
        return None

    def save(
        self, state: SessionState, pending_events: Sequence[Any] | None = None
    ) -> SessionState:
        if pending_events is not None:
            self.saved_events_by_session.setdefault(state.session_id, []).extend(
                cast("list[dict[str, Any]]", list(pending_events))
            )
        return state

    def delete(self, session_id: str, artifacts_dir: Path | None = None) -> bool:
        self.saved_events_by_session.pop(session_id, None)
        return True

    def list_session_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.saved_events_by_session))

    def clear_event_log(self, session_id: str) -> None:
        return None

    def save_context_body(self, session_id: str, digest: str, body: str) -> None:
        return None

    def load_context_body(self, session_id: str, digest: str) -> str | None:
        return None


def test_delete_session_drops_its_queued_durable_events() -> None:
    """A deleted session must drop its pending durable events so they cannot leak into another session's record (#682).

    Killed by: src/uclone_x/agent/base.py :: e.get("session_id") != sid  # delete_session drops queued events
    Becomes: True  # delete_session drops queued events
    """
    store = _RecordingSessionStore()
    agent = BaseAgent(
        config=AgentConfig(agent_id="del_events", name="Delete Events"),
        llm=MockLLMConnector(),
        context=AgentContext(session_id="sA", agent_id="del_events"),
        store=store,
    )

    asyncio.run(agent.execute_turn("question for sA"))
    pending_a = [
        e
        for e in cast("list[dict[str, Any]]", vars(agent)["_pending_durable_events"])
        if e.get("session_id") == "sA"
    ]
    assert pending_a, "turn on sA queued no durable events"

    agent.switch_session("sB")
    asyncio.run(agent.execute_turn("question for sB"))

    agent.delete_session("sA")

    remaining_events = cast("list[dict[str, Any]]", vars(agent)["_pending_durable_events"])
    assert not any(e.get("session_id") == "sA" for e in remaining_events), (
        "delete_session('sA') left sA's durable events queued in _pending_durable_events"
    )

    agent.persist_session("sB")
    saved_sb_events = store.saved_events_by_session.get("sB", [])
    assert saved_sb_events, "persisting sB wrote no events"
    assert not any(e.get("session_id") == "sA" for e in saved_sb_events), (
        "persisting sB wrote sA's durable events into sB's record"
    )


def test_persist_session_only_persists_named_session_events() -> None:
    """persist_session must only write durable events belonging to the targeted session (#682).

    Killed by: src/uclone_x/agent/base.py :: d_event.get("session_id") == sid  # persist_session selects session events
    Becomes: True  # persist_session selects session events
    """
    store = _RecordingSessionStore()
    agent = BaseAgent(
        config=AgentConfig(agent_id="persist_filter", name="Persist Filter"),
        llm=MockLLMConnector(),
        context=AgentContext(session_id="sA", agent_id="persist_filter"),
        store=store,
    )

    asyncio.run(agent.execute_turn("question for sA"))
    agent.switch_session("sB")
    asyncio.run(agent.execute_turn("question for sB"))

    agent.persist_session("sB")

    saved_sb_events = store.saved_events_by_session.get("sB", [])
    assert saved_sb_events, "persisting sB wrote no events"
    assert all(e.get("session_id") == "sB" for e in saved_sb_events), (
        f"persisting sB mis-filed other session events: "
        f"{[e.get('session_id') for e in saved_sb_events]}"
    )


def test_persist_session_preserves_other_sessions_queued_events() -> None:
    """persist_session must retain unpersisted events belonging to other sessions (#682).

    Killed by: src/uclone_x/agent/base.py :: d_event.get("session_id") != sid  # persist_session retains other session events
    Becomes: False  # persist_session retains other session events
    """
    store = _RecordingSessionStore()
    agent = BaseAgent(
        config=AgentConfig(agent_id="persist_retain", name="Persist Retain"),
        llm=MockLLMConnector(),
        context=AgentContext(session_id="sA", agent_id="persist_retain"),
        store=store,
    )

    asyncio.run(agent.execute_turn("question for sA"))
    agent.switch_session("sB")
    asyncio.run(agent.execute_turn("question for sB"))

    agent.persist_session("sA")

    remaining_events = cast("list[dict[str, Any]]", vars(agent)["_pending_durable_events"])
    sb_pending = [e for e in remaining_events if e.get("session_id") == "sB"]
    assert sb_pending, (
        "persisting sA cleared sB's unpersisted durable events from _pending_durable_events"
    )


@pytest.mark.asyncio
async def test_agent_turn_fails_when_token_budget_exhausted_during_reasoning() -> None:
    """When a model generation finishes with FinishReason.LENGTH and empty content/tools, the turn must fail (#695).

    Killed by: src/uclone_x/agent/base.py :: and resp.finish_reason == FinishReason.LENGTH
    Becomes: and False
    """

    class _LengthExhaustedConnector(MockLLMConnector):
        async def generate(self, request: LLMRequest) -> ModelResponse:
            return ModelResponse(
                content="",
                thinking="Step 1: Deliberating for a very long time... (tokens ran out)",
                tool_calls=(),
                usage=TokenUsage(
                    provider="mock",
                    input_tokens=10,
                    output_tokens=100,
                    total_tokens=110,
                ),
                finish_reason=FinishReason.LENGTH,
                provenance=Provenance.primary(
                    provider="mock",
                    model="mock-model",
                ),
            )

    agent = BaseAgent(
        config=AgentConfig(
            agent_id="length-exhausted-agent",
            name="Length Exhausted Agent",
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=_LengthExhaustedConnector(),
    )

    result = await agent.execute_turn("Solve hard problem")
    assert result.is_completed is False
    assert result.error is not None
    assert "token budget exhausted" in result.error.lower()
    assert result.provenance is None


@pytest.mark.asyncio
async def test_a_tool_call_outside_the_allowlist_does_not_run_mid_turn() -> None:
    """Advertising is a hint; the allowlist is enforced where the call is executed.

    `execute_turn` shows the model only the permitted names, and the execution path used to
    do a bare registry lookup -- so a call the model produced for a name it was never shown
    ran anyway. Small models produce exactly that, which is why `agent.text_tool_calls`
    exists at all. Driven through a real turn rather than the private execution helper,
    because the gap was between what the turn advertises and what the turn runs.

    Killed by: src/uclone_x/agent/base.py :: if allowed_names and effective_tc.name not in allowed_names:
    Becomes: if False:
    """
    executed: list[str] = []

    class RecordingTool(BaseTool[DummyEchoParams]):
        name = "forbidden_tool"
        description = "Records that it ran."

        def run(self, params: DummyEchoParams, context: ToolContext) -> str:
            executed.append(self.name)
            return "ran"

    reg = ToolRegistry()
    reg.register(RecordingTool())
    connector = CannedToolResponseConnector(
        content="Calling a tool it was never shown",
        tool_calls=(ToolCallRequest(id="call_1", name="forbidden_tool", arguments={"text": "x"}),),
    )
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="scoped_agent",
            name="ScopedAgent",
            system_prompt="",
            allowed_tools=("some_other_tool",),
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=connector,
        tools=reg,
    )

    result = await agent.execute_turn("Go")

    assert executed == []
    record = result.tool_executions[0]
    assert record.status is ToolResultStatus.ERROR
    # "not allowed" and "not there" are different answers; a scoped registry would have
    # reported this one as the other.
    assert "allowed_tools" in (record.error or "")
    assert "not found" not in (record.error or "")


# --------------------------------------------------------------------------------------
# A stream that fails mid-turn fails the turn (#938).
#
# `_invoke_model` used to catch any exception out of `llm.stream` and call `llm.generate`
# for the same request, under the connector's `PRIMARY` provenance. Measured at `049a224`
# with the connector below (three content chunks, then a provider error): the turn
# *completed*, `generate` was called once, the listener heard the three partial deltas and
# then the whole retried reply as a fourth, the provenance said `primary`, and the ledger
# held one entry — the retry's. The partial stream the provider had already served was
# spent and never booked. The rule now (room UI design document §6.7 **[#938]**): the turn
# fails with `LLMStreamInterruptedError`, nothing is re-requested, the partial reply and any
# tool call it carried are discarded unexecuted, and the partial stream is booked — the
# provider's count if a chunk carried one, else a labelled estimate, and nothing when no
# chunk arrived at all.
# --------------------------------------------------------------------------------------

_RETRIED_REPLY = "A whole second answer that a silent retry would have delivered."


class _StreamThatFailsConnector(MockLLMConnector):
    """Streams a scripted prefix and then fails; counts every call to `generate`.

    `generate` answers, so a silent retry is observable as a completed turn, a call count
    and a second copy of the reply at the listener, rather than as an error either way.
    """

    def __init__(self, chunks: Sequence[StreamChunk]) -> None:
        super().__init__()
        self._chunks = tuple(chunks)
        self.generate_calls = 0

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.generate_calls += 1
        return ModelResponse(
            content=_RETRIED_REPLY,
            usage=_SCRIPTED_COUNT,
            finish_reason=FinishReason.STOP,
            model_name="scripted-model",
            provenance=Provenance.primary(provider="ollama", model="scripted-model"),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        from uclone_x.errors import LLMProviderError

        for chunk in self._chunks:
            yield chunk
        raise LLMProviderError("Ollama stream connection error: peer closed connection")


async def _a_turn_whose_stream_fails(
    chunks: Sequence[StreamChunk], *, tools: ToolRegistry | None = None
) -> tuple[TurnResult, _StreamThatFailsConnector, list[tuple[str, dict[str, Any]]], Any]:
    """One watched turn: (result, connector, everything the listener heard, ledger)."""
    manager = TokenBudgetManager()
    manager.set_budget("sess_938", TokenBudget(max_tokens=100_000))
    connector = _StreamThatFailsConnector(chunks)
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="listened",
            name="Listened",
            system_prompt="",
            llm_config=AgentLLMConfig(model_name="scripted-model"),
        ),
        llm=connector,
        context=AgentContext(session_id="sess_938", agent_id="listened"),
        budget=manager,
        tools=tools,
    )
    heard: list[tuple[str, dict[str, Any]]] = []

    async def listener(event: str, data: dict[str, Any]) -> None:
        heard.append((event, data))

    result = await agent.execute_turn("answer me", stream_callback=listener)
    return result, connector, heard, manager.get_turn_history("sess_938")


def _tokens(heard: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [str(data.get("content", "")) for event, data in heard if event == "token"]


@pytest.mark.asyncio
async def test_a_stream_that_fails_mid_reply_fails_the_turn_and_books_the_partial_stream() -> None:
    """Not retried, not replayed, not attributed as primary — and not free.

    The listener hears the partial reply once and nothing after it: the turn's failure is
    the signal that supersedes it (the room lands a failed row, the chat stream's `done`
    replaces the bubble). The partial stream reported no usage, so it is booked as an
    estimate of what was actually streamed, not of a reply that never arrived.

    Restoring the silent retry — `generate` re-requested and its reply replayed to the
    listener under the connector's `PRIMARY` provenance — completes the turn:

    Killed by: src/uclone_x/agent/base.py :: raise interrupted from stream_err
    Becomes: resp = await llm.generate(req); _r = stream_callback("token", {"content": resp.content}); _ = (await _r) if asyncio.iscoroutine(_r) else None

    Dropping the booking of the partial stream leaves the ledger empty:

    Killed by: src/uclone_x/agent/base.py :: self._budget.record_usage(self._context.session_id, partial_usage)
    Becomes: pass

    The estimate is the shared one over UTF-8 bytes, rounded up (#980), not the characters
    rounded down it was: `len // 4` books 4 output tokens for this 17-byte reply, not 5.

    Killed by: src/uclone_x/agent/base.py :: out_tokens = estimate_reply_tokens(content, tool_calls)
    Becomes: out_tokens = max(1, len(content) // 4)
    """
    partial = ["The ", "partial ", "reply"]
    result, connector, heard, ledger = await _a_turn_whose_stream_fails(
        [StreamChunk(delta_content=delta) for delta in partial]
    )

    assert result.is_completed is False
    assert connector.generate_calls == 0, "the failed stream was re-requested through generate"
    assert _tokens(heard) == partial, "the listener heard something other than the partial reply"
    assert result.content == ""
    assert result.error is not None
    assert "after 3 chunk(s)" in result.error
    assert "LLMProviderError: Ollama stream connection error" in result.error
    # In band, on the result (R27): the turn is not attributed to a primary answer, and the
    # attempt names what failed.
    assert result.provenance is not None
    assert result.provenance.path is not ExecutionPath.PRIMARY
    assert [a.error_class for a in result.provenance.attempts] == ["LLMStreamInterruptedError"]

    assert len(ledger) == 1, f"the partial stream was booked {len(ledger)} times"
    booked = ledger[0]
    assert booked.count_source is TokenCountSource.ESTIMATE
    assert booked.provider == "ollama"
    assert booked.output_tokens == estimate_reply_tokens("".join(partial))


@pytest.mark.asyncio
async def test_a_stream_that_fails_before_its_first_chunk_books_nothing() -> None:
    """No chunk is no evidence the provider served anything, so nothing is estimated.

    The failure still fails the turn and is still not retried; the ceiling is not charged
    for a request the provider may have refused at the door (a 429 or 503 is raised before
    the first chunk by every in-tree connector).

    Killed by: src/uclone_x/agent/base.py :: if partial_usage is None and chunks_received:
    Becomes: if partial_usage is None:
    """
    result, connector, heard, ledger = await _a_turn_whose_stream_fails([])

    assert result.is_completed is False
    assert connector.generate_calls == 0
    assert _tokens(heard) == []
    assert result.error is not None and "after 0 chunk(s)" in result.error
    assert ledger == ()


@pytest.mark.asyncio
async def test_a_stream_that_fails_after_a_tool_call_discards_the_call_unexecuted() -> None:
    """A tool call from a stream that did not finish is never run, and is not reported.

    The call may be truncated — its arguments are whatever arrived — and the step that asked
    for it has no reply, so executing it would act on half a decision. It is named in the
    error so a reader knows one was dropped. A count the provider sent before the failure is
    booked as that count, not replaced by an estimate. (The in-tree connectors send theirs
    on a late chunk, so there this is a connection that drops just after it; a count on the
    first chunk keeps the script short.)

    Treating the partial stream as a finished step executes the call:

    Killed by: src/uclone_x/agent/base.py :: raise interrupted from stream_err
    Becomes: resp = ModelResponse(content="".join(content_chunks), tool_calls=tuple(tool_calls_list), usage=partial_usage, finish_reason=FinishReason.TOOL_CALLS, model_name=model_name, provenance=Provenance.primary(provider=llm.provider_name, model=model_name))

    Estimating over a count the provider did send books a figure it never reported:

    Killed by: src/uclone_x/agent/base.py :: partial_usage = last_usage
    Becomes: partial_usage = None
    """
    executed: list[str] = []

    class RecordingTool(BaseTool[DummyEchoParams]):
        name = "recording_tool"
        description = "Records that it ran."

        def run(self, params: DummyEchoParams, context: ToolContext) -> str:
            executed.append(params.text)
            return "ran"

    reg = ToolRegistry()
    reg.register(RecordingTool())
    early_count = TokenUsage(provider="ollama", model="scripted-model", input_tokens=40)
    result, connector, heard, ledger = await _a_turn_whose_stream_fails(
        [
            StreamChunk(usage=early_count),
            StreamChunk(delta_content="Let me look that up. "),
            StreamChunk(
                tool_calls=(
                    ToolCallRequest(id="call_1", name="recording_tool", arguments={"text": "x"}),
                )
            ),
        ],
        tools=reg,
    )

    assert executed == [], "a tool call from an unfinished stream was executed"
    assert result.is_completed is False
    assert result.tool_calls == ()
    assert result.tool_executions == ()
    assert [event for event, _ in heard if event.startswith("tool")] == []
    assert result.error is not None and "1 tool call(s)" in result.error
    assert connector.generate_calls == 0
    assert ledger == (early_count,)


# A watched step and a headless step are estimated alike (#980).
#
# When the provider sends no count, `generate` completes it in the connector
# (`resolve_token_counts`) and a stream completes it in `_invoke_model`
# (`_estimate_stream_usage`). The stream's estimate was `len // 4` over characters with no
# message framing, no tool definitions and no reply tool calls: at `8683cf9`, on the cases
# below, 16 / 7 against 23 / 8 on ASCII, 16 / 7 against 49 / 8 with a tool definition,
# 10 / 3 against 29 / 10 on Hangul, and 16 / 1 against 49 / 15 for a reply that is one tool
# call. Whether a head was watching changed the figure the budget booked.
# --------------------------------------------------------------------------------------

_PARITY_TOOL = ToolDefinition(
    name="web_search",
    description="Search the web",
    parameters={"type": "object", "properties": {"q": {"type": "string"}}},
)
_PARITY_CALL = ToolCallRequest(id="call_1", name="web_search", arguments={"q": "capital of France"})
_PARITY_SECOND_CALL = ToolCallRequest(
    id="call_2", name="web_search", arguments={"q": "population of Paris"}
)
_ENGLISH = ("You are a helpful assistant.", "What is the capital of France?")
_KOREAN = ("당신은 도움이 되는 비서입니다.", "프랑스의 수도는 어디인가요?")


class _NoUsageStreamConnector(MockLLMConnector):
    """A stream that carries a reply and its tool calls and never a count, then may fail."""

    def __init__(self, reply: str, tool_calls: tuple[ToolCallRequest, ...], *, fails: bool) -> None:
        super().__init__()
        self._scripted_reply = reply
        self._scripted_calls = tool_calls
        self._scripted_failure = fails

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        from uclone_x.errors import LLMProviderError

        if self._scripted_reply:
            yield StreamChunk(delta_content=self._scripted_reply)
        if self._scripted_calls:
            yield StreamChunk(tool_calls=self._scripted_calls)
        if self._scripted_failure:
            raise LLMProviderError("Ollama stream connection error: peer closed connection")
        yield StreamChunk(
            finish_reason=FinishReason.TOOL_CALLS if self._scripted_calls else FinishReason.STOP
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_fails", [False, True], ids=["stream-ended", "stream-failed"])
@pytest.mark.parametrize(
    ("prompt", "tools", "reply", "reply_calls"),
    [
        (_ENGLISH, (), "The capital of France is Paris.", ()),
        (_ENGLISH, (_PARITY_TOOL,), "The capital of France is Paris.", ()),
        (_KOREAN, (), "프랑스의 수도는 파리입니다.", ()),
        (_ENGLISH, (_PARITY_TOOL,), "", (_PARITY_CALL,)),
        (_ENGLISH, (_PARITY_TOOL,), "", (_PARITY_CALL, _PARITY_SECOND_CALL)),
    ],
    ids=["ascii", "ascii-tool-definition", "hangul", "tool-call-only-reply", "two-tool-calls"],
)
async def test_a_watched_step_is_estimated_exactly_as_a_headless_one(
    prompt: tuple[str, str],
    tools: tuple[ToolDefinition, ...],
    reply: str,
    reply_calls: tuple[ToolCallRequest, ...],
    stream_fails: bool,
) -> None:
    """For the same request and reply, a stream without a count books what `generate` would.

    The headless figure is `resolve_token_counts` with neither count reported, which is what
    every connector's `generate` books then. The watched figure is what `_invoke_model`
    books from a stream that sent no count, both when the stream ends and when it fails
    after its first chunk (#938). Both are labelled `ESTIMATE`. A tool call that arrived on
    a failed stream is discarded unexecuted, but the provider served it, so it is counted.

    Leaving the tool definitions out of the input:

    Killed by: src/uclone_x/agent/base.py :: in_tokens = estimate_request_tokens(req)
    Becomes: in_tokens = estimate_request_tokens(req.model_copy(update={"tools": ()}))

    Restoring `len // 4` over the messages' text, with no framing:

    Killed by: src/uclone_x/agent/base.py :: estimate_request_tokens(req)
    Becomes: max(1, len(str([m.content for m in req.messages])) // 4)

    Leaving the reply's tool calls out of the output:

    Killed by: src/uclone_x/agent/base.py :: out_tokens = estimate_reply_tokens(content, tool_calls)
    Becomes: out_tokens = estimate_reply_tokens(content)

    Counting only the reply's first tool call:

    Killed by: src/uclone_x/agent/base.py :: out_tokens = estimate_reply_tokens(content, tool_calls)
    Becomes: out_tokens = estimate_reply_tokens(content, tool_calls[:1])

    Not passing the stream's tool calls, from a stream that ended or one that failed:

    Killed by: src/uclone_x/agent/base.py :: full_content, tool_calls_list
    Becomes: full_content, ()
    Killed by: src/uclone_x/agent/base.py :: known_model, "".join(content_chunks), tool_calls
    Becomes: known_model, "".join(content_chunks), ()
    """
    request = LLMRequest(
        messages=(
            ChatMessage(role=MessageRole.SYSTEM, content=prompt[0]),
            ChatMessage(role=MessageRole.USER, content=prompt[1]),
        ),
        tools=tools,
    )
    manager = TokenBudgetManager()
    manager.set_budget("sess_980", TokenBudget(max_tokens=100_000))
    connector = _NoUsageStreamConnector(reply, reply_calls, fails=stream_fails)
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="listened",
            name="Listened",
            llm_config=AgentLLMConfig(model_name="scripted-model"),
        ),
        llm=connector,
        context=AgentContext(session_id="sess_980", agent_id="listened"),
        budget=manager,
    )

    async def listener(event: str, data: dict[str, Any]) -> None:
        return None

    if stream_fails:
        with pytest.raises(LLMStreamInterruptedError):
            await agent._invoke_model(  # pyright: ignore[reportPrivateUsage]
                connector, request, stream_callback=listener
            )
    else:
        await agent._invoke_model(  # pyright: ignore[reportPrivateUsage]
            connector, request, stream_callback=listener
        )

    ledger = manager.get_turn_history("sess_980")
    assert len(ledger) == 1, f"the step was booked {len(ledger)} times"
    watched = ledger[0]
    headless_in, headless_out, headless_source = resolve_token_counts(
        request, None, None, reply=reply, tool_calls=reply_calls
    )
    assert (watched.input_tokens, watched.output_tokens) == (headless_in, headless_out)
    assert watched.count_source is TokenCountSource.ESTIMATE
    assert headless_source is TokenCountSource.ESTIMATE


class _ProviderDownConnector(MockLLMConnector):
    """A connector whose provider is unreachable: every call raises, as a transient outage does."""

    async def generate(self, request: LLMRequest) -> ModelResponse:
        from uclone_x.errors import LLMProviderError

        raise LLMProviderError("provider unavailable (scripted)")


@pytest.mark.asyncio
async def test_a_budget_refusal_is_named_by_its_stop_reason() -> None:
    """A spent budget is stated in `stop_reason`, so a head can tell it from an outage (#969).

    A retry cannot pass a ceiling the ledger has already reached -- the ledger only grows
    -- so a head offering Retry on this refusal offers a remedy that is refused again. The
    refusal carried the loop's seed, `not_started`, which is exactly what a provider
    outage carries, so the only difference a head could read was the wording of `error`.

    Killed by: src/uclone_x/agent/base.py :: stop_reason = "budget_exceeded"
    Becomes: stop_reason = stop_reason
    """
    budget = TokenBudgetManager()
    budget.configure_session("sess_spent", max_tokens=0)
    refused = await BaseAgent(
        config=AgentConfig(
            agent_id="spent",
            name="Spent",
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=MockLLMConnector(),
        context=AgentContext(session_id="sess_spent", agent_id="spent"),
        budget=budget,
    ).execute_turn("hello")

    assert refused.error is not None and "limit exceeded" in refused.error.lower()
    assert refused.stop_reason == "budget_exceeded"

    # Control: a failure a retry can get past is not reported as one.
    failed = await BaseAgent(
        config=AgentConfig(
            agent_id="unlucky",
            name="Unlucky",
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=_ProviderDownConnector(),
    ).execute_turn("hello")
    assert failed.error is not None
    assert failed.stop_reason != "budget_exceeded"


class _RecordingConnector(MockLLMConnector):
    """Answers normally, and keeps every request it was sent."""

    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses=responses)
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        return await super().generate(request)


def _prompts(request: LLMRequest) -> list[tuple[str | None, str | None]]:
    """The user messages a request showed the model, as (sender name, text)."""
    return [(m.name, m.content) for m in request.messages if m.role is MessageRole.USER]


@pytest.mark.asyncio
async def test_an_unanswered_prompt_from_another_sender_is_not_a_repeat_of_it() -> None:
    """Only a sender's own unanswered prompt is a repeat of it (#969).

    A failed turn leaves its prompt at the end of history, and an identical re-send is not
    appended again -- that is what stops a retry showing the model the same question twice.
    The rule is the *sender's* prompt, not the words: `name` is what tells two senders apart
    in a transcript that names them, and `reconstruct_history` sets it from a stored prompt.
    Without the clause, a head that names its users would silently lose one user's message
    whenever it repeated another's unanswered words.

    Killed by: src/uclone_x/agent/base.py :: and last.name == user_prompt.name
    Becomes: and True
    """
    named = _RecordingConnector(["Answered"])
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="named-senders",
            name="Named Senders",
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=named,
    )
    agent.load_history([ChatMessage(role=MessageRole.USER, content="same words", name="alice")])

    answered = await agent.execute_turn("same words")

    assert answered.is_completed is True
    assert _prompts(named.requests[-1]) == [("alice", "same words"), (None, "same words")]

    # Control: the sender's own unanswered prompt is still not repeated to the model.
    same = _RecordingConnector(["Answered"])
    own = BaseAgent(
        config=AgentConfig(
            agent_id="one-sender",
            name="One Sender",
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=same,
    )
    own.load_history([ChatMessage(role=MessageRole.USER, content="same words")])

    assert (await own.execute_turn("same words")).is_completed is True
    assert _prompts(same.requests[-1]) == [(None, "same words")]


def _system_sent(request: LLMRequest) -> str:
    """The single system message a request actually put on the wire."""
    systems = [m.content or "" for m in request.messages if m.role is MessageRole.SYSTEM]
    assert len(systems) == 1, f"expected exactly one system message, got {len(systems)}"
    return systems[0]


@pytest.mark.asyncio
async def test_hot_reloading_the_model_reframes_the_system_message_actually_sent() -> None:
    """The turn the model receives follows `hot_reload_llm`, not the anchor seeded at init (#921).

    `hot_reload_llm` moves `llm_config.model_name`, so `effective_system_prompt` answers for
    the new family at once. The wire call took `self._history[0]`, which still carried the
    family framing the session was seeded with, so the property and the message actually sent
    disagreed and nothing reported it (P6). Asserting the property alone cannot see this: it
    was already correct while the request was wrong, which is the whole defect.

    Killed by: src/uclone_x/agent/request_record.py :: messages.insert(0, ChatMessage(role=MessageRole.SYSTEM, content=system))
    Becomes: pass
    """
    wire = _RecordingConnector(["Answered"])
    agent = BaseAgent(
        config=AgentConfig(agent_id="reframe", name="Reframe"),
        llm=wire,
    )
    assert STEERABILITY_POLICY in (agent.history[0].content or "")

    agent.hot_reload_llm(model_name="hermes3:8b")
    assert (await agent.execute_turn("hello")).is_completed is True

    sent = _system_sent(wire.requests[-1])
    assert HERMES_STEERABILITY_POLICY in sent
    assert STEERABILITY_POLICY not in sent
    assert sent == agent.effective_system_prompt


@pytest.mark.asyncio
async def test_hot_reloading_back_off_a_family_reframes_the_system_message_actually_sent() -> None:
    """The reverse switch is the half a one-way substitution on the anchor cannot reach (#921).

    A session seeded on Hermes carries Hermes framing in `history[0]`. Re-framing *that text*
    for Qwen is a no-op, because there is no canonical policy left in it to substitute, so the
    outbound turn keeps the Hermes framing for the rest of the session's life. Only running
    the substitution from the canonical form fixes both directions.
    """
    wire = _RecordingConnector(["Answered"])
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="reframe-back",
            name="Reframe Back",
            llm_config=AgentLLMConfig(model_name="hermes3:8b"),
        ),
        llm=wire,
    )
    assert HERMES_STEERABILITY_POLICY in (agent.history[0].content or "")

    agent.hot_reload_llm(model_name="llama3.2")
    assert (await agent.execute_turn("hello")).is_completed is True

    sent = _system_sent(wire.requests[-1])
    assert STEERABILITY_POLICY in sent
    assert HERMES_STEERABILITY_POLICY not in sent
    assert sent == agent.effective_system_prompt

    # Hot reloading to Qwen applies QWEN_STEERABILITY_POLICY
    agent.hot_reload_llm(model_name="qwen3:8b")
    assert (await agent.execute_turn("hello again")).is_completed is True
    sent_qwen = _system_sent(wire.requests[-1])
    assert QWEN_STEERABILITY_POLICY in sent_qwen
    assert HERMES_STEERABILITY_POLICY not in sent_qwen
    assert sent_qwen == agent.effective_system_prompt

    # The stored anchor is left exactly as it was: the record of what earlier turns were
    # seeded with is not edited to claim framing those turns never carried.
    assert HERMES_STEERABILITY_POLICY in (agent.history[0].content or "")


@pytest.mark.asyncio
async def test_an_operators_own_system_prompt_survives_the_re_framing_untouched() -> None:
    """A prompt embedding none of the default components is never replaced by the resolved one.

    This is what the in-place refresh would have cost: overwriting the anchor with
    `effective_system_prompt` discards whatever a caller anchored, and the loss is
    unrecoverable once the session is persisted.
    """
    custom = "You are a microservice orchestration agent."
    wire = _RecordingConnector(["Answered"])
    agent = BaseAgent(
        config=AgentConfig(agent_id="custom-prompt", name="Custom", system_prompt=custom),
        llm=wire,
    )

    agent.hot_reload_llm(model_name="hermes3:8b")
    assert (await agent.execute_turn("hello")).is_completed is True

    assert _system_sent(wire.requests[-1]) == custom
    assert agent.history[0].content == custom


# --------------------------------------------------------------------------------------
# Adopting a persona after construction adopts all of it (#1081).
#
# `persona` and `persona_name` were a bare `self._persona = value` each, with no other
# statement in either body. `effective_system_prompt` re-resolves on every read, so the
# *prompt* followed the assignment; the *tool* restriction was resolved by a block in
# `__init__`, which an assignment does not re-run; and the anchored system turn kept
# naming the persona in force when the session was seeded. The tests below pin the tool
# restriction -- that a late persona applies one, and how it composes with an operator's
# own list -- then the system message actually sent, then that the alias setter reaches
# all of it.
# --------------------------------------------------------------------------------------


class _PermittedTool(BaseTool[DummyEchoParams]):
    name = "permitted_tool"
    writes_files = False  # read-only: the scout persona leaves enable_write_tools off (#1167)
    description = "A tool a scout persona is allowed to call."

    def run(self, params: DummyEchoParams, context: ToolContext) -> str:
        return "ran"


class _WithheldTool(BaseTool[DummyEchoParams]):
    name = "withheld_tool"
    writes_files = False  # read-only: the scout persona leaves enable_write_tools off (#1167)
    description = "A tool a scout persona is not allowed to call."

    def run(self, params: DummyEchoParams, context: ToolContext) -> str:
        return "ran"


def _scouting_persona() -> PersonaDefinition:
    """A persona whose `allowed_tools` names `permitted_tool` and nothing else."""
    return PersonaDefinition(
        name="late_scout",
        role="Scout",
        system_prompt="You scout. Report what you find.",
        allowed_tools=("permitted_tool",),
    )


def _agent_with_both_tools(config: AgentConfig) -> BaseAgent:
    registry = ToolRegistry()
    registry.register(_PermittedTool())
    registry.register(_WithheldTool())
    agent = BaseAgent(config=config, tools=registry)
    agent.define_persona(_scouting_persona())
    return agent


@pytest.mark.asyncio
async def test_a_persona_set_after_construction_restricts_tools_as_well_as_prompt() -> None:
    """Adopting a persona late must adopt its `allowed_tools`, not only its prompt (#1081).

    The prompt half was already live, which is what made this hard to see: an agent given
    a persona after construction reported the persona's prompt from
    `effective_system_prompt` while calling tools the persona forbids. Asserting the
    prompt alone cannot see it: the prompt was already right while the enforcement was
    wrong.

    Killed by: src/uclone_x/agent/base.py :: resolved = persona.granted_tools
    Becomes: pass
    """
    agent = _agent_with_both_tools(AgentConfig(agent_id="late", name="Late"))

    # Before: no persona, so nothing is restricted and the withheld tool still runs.
    assert agent.config.allowed_tools == ()
    assert (await agent.execute_tool_call("withheld_tool", {"text": "x"})).status is (
        ToolResultStatus.SUCCESS
    )

    agent.persona = "late_scout"

    assert agent.effective_system_prompt == "You scout. Report what you find."
    assert agent.config.allowed_tools == ("permitted_tool", *BASE_PERSONA_TOOLS)
    assert (await agent.execute_tool_call("permitted_tool", {"text": "x"})).status is (
        ToolResultStatus.SUCCESS
    )
    with pytest.raises(PermissionError, match="allowed_tools"):
        await agent.execute_tool_call("withheld_tool", {"text": "x"})


@pytest.mark.asyncio
async def test_an_operators_allowed_tools_outranks_a_persona_adopted_later() -> None:
    """An operator's own list wins, and a persona's list is recomputed rather than layered.

    `__init__` applied a persona's tools only when the operator supplied none
    (`not config.allowed_tools`). Extending that rule to a late persona is a behaviour
    choice, so it is pinned here rather than left to be inferred: an operator list stands
    unchanged through a persona adoption, a persona swapped for another yields the second
    persona's tools rather than the union or the first's, and clearing the persona takes
    the persona's tools away rather than leaving them in force under no persona.

    Killed by: src/uclone_x/agent/base.py :: resolved = self._operator_allowed_tools
    Becomes: resolved = self._config.allowed_tools
    """
    operator_scoped = _agent_with_both_tools(
        AgentConfig(agent_id="op", name="Op", allowed_tools=("withheld_tool",))
    )

    operator_scoped.persona = "late_scout"

    assert operator_scoped.config.allowed_tools == ("withheld_tool",)
    with pytest.raises(PermissionError, match="allowed_tools"):
        await operator_scoped.execute_tool_call("permitted_tool", {"text": "x"})

    # No operator list: the persona's applies, a second persona replaces it rather than
    # adding to it, and clearing the persona takes it away again.
    unscoped = _agent_with_both_tools(AgentConfig(agent_id="unscoped", name="Unscoped"))
    unscoped.define_persona(
        PersonaDefinition(
            name="late_auditor",
            role="Auditor",
            system_prompt="You audit.",
            allowed_tools=("withheld_tool",),
        )
    )

    unscoped.persona = "late_scout"
    assert unscoped.config.allowed_tools == ("permitted_tool", *BASE_PERSONA_TOOLS)

    unscoped.persona = "late_auditor"
    assert unscoped.config.allowed_tools == ("withheld_tool", *BASE_PERSONA_TOOLS)

    unscoped.persona = None
    assert unscoped.config.allowed_tools == ()


@pytest.mark.asyncio
async def test_a_persona_set_after_construction_changes_the_system_message_actually_sent() -> None:
    """The turn the model receives follows the persona, and history keeps the old record (#1081).

    Two properties, and the second is the one #1078 chose per-turn recomputation to keep.
    The message the model is sent must name the persona in force, *and* `history[0]` must
    go on reporting what the turns *before* the adoption were sent -- so the anchor is not
    rewritten in place, because that would edit the record of a turn that really did carry
    the old prompt. `history[0]` is not a record of what later turns sent and never was:
    from #921 on, the anchor is re-framed on the way out and the resolved text is not
    written back.

    Killed by: src/uclone_x/agent/base.py ::
        return session.anchor_provenance != self._resolved_persona()
    Becomes: return False
    """
    wire = _RecordingConnector(["Before", "After"])
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="anchored", name="Anchored", system_prompt="You are the default."
        ),
        llm=wire,
    )
    agent.define_persona(_scouting_persona())

    assert (await agent.execute_turn("first")).is_completed is True
    assert _system_sent(wire.requests[-1]) == "You are the default."

    agent.persona = "late_scout"
    assert (await agent.execute_turn("second")).is_completed is True

    sent = _system_sent(wire.requests[-1])
    assert sent == "You scout. Report what you find."
    assert sent == agent.effective_system_prompt
    # The anchor still says what the first turn -- and only the first turn -- was sent.
    assert agent.history[0].content == "You are the default."


@pytest.mark.asyncio
async def test_an_assignment_that_resolves_to_no_persona_leaves_a_hydrated_anchor_alone() -> None:
    """Two assignments that adopt nothing must not discard an operator's hydrated prompt.

    Both constructions move `_persona` and neither moves the persona the prompt axis
    resolves to: `None` clears a name that was never set, and an unresolvable name is a
    name no store or registry answers for. The first revision of this fix recorded the
    assignment rather than its effect, and each of these then replaced a system turn the
    operator had hydrated with `config.system_prompt` -- the failure mode #1081 records as
    measured and rejected for PR #937, reached by a different route.

    What holds the line here is not the comparison but *whose text the anchor is*: this
    session's anchor came in through `load_history`, so it is stamped `CALLER` and no
    resolution of the axis makes it stale. The mutation drops exactly that reading.

    Killed by: src/uclone_x/agent/base.py ::
        if isinstance(session.anchor_provenance, _AnchorWriter):
    Becomes: if False:
    """
    for label, value in (("clear", None), ("unresolvable", "no_such_persona")):
        wire = _RecordingConnector(["Answered"])
        agent = BaseAgent(
            config=AgentConfig(
                agent_id=f"hydrated-{label}",
                name="Hydrated",
                system_prompt="CONFIG PROMPT",
            ),
            llm=wire,
        )
        agent.define_persona(_scouting_persona())
        agent.load_history(
            [ChatMessage(role=MessageRole.SYSTEM, content="OPERATOR HYDRATED PROMPT")]
        )

        agent.persona = value

        assert (await agent.execute_turn("go")).is_completed is True
        assert _system_sent(wire.requests[-1]) == "OPERATOR HYDRATED PROMPT", label


@pytest.mark.asyncio
async def test_clearing_a_persona_the_anchor_was_written_under_does_move_the_wire() -> None:
    """Dropping a persona that *was* in force is a move, and the wire has to follow it.

    The companion control, and the reason the stamp records a resolution rather than
    merely "a caller wrote this". The agent is built with the persona and `reset_session`
    re-anchors under it, so the anchor is stamped with the persona itself; the clear then
    makes the stamp and the axis disagree. A `reset_session` that stamped `None` -- the
    mutation below -- would leave this turn carrying the dropped persona's prompt while
    `effective_system_prompt` reported the configured one: the same P6 divergence #1081
    exists to close, pointing the other way.

    Killed by: src/uclone_x/agent/base.py :: reset, anchor_provenance=self._resolved_persona()
    Becomes: reset, anchor_provenance=None
    """
    wire = _RecordingConnector(["One"])
    agent = BaseAgent(
        config=AgentConfig(agent_id="dropped", name="Dropped", system_prompt="CONFIG PROMPT"),
        llm=wire,
        persona="late_scout",
    )
    agent.define_persona(_scouting_persona())
    # Anchor the session under the persona: `__init__` seeded it before `define_persona`
    # could resolve the name, and `reset_session` re-seeds from `effective_system_prompt`
    # without going near `_set_persona`.
    agent.reset_session()
    assert agent.history[0].content == "You scout. Report what you find."

    agent.persona = None

    assert (await agent.execute_turn("go")).is_completed is True
    assert _system_sent(wire.requests[-1]) == "CONFIG PROMPT"
    assert _system_sent(wire.requests[-1]) == agent.effective_system_prompt


@pytest.mark.asyncio
async def test_the_persona_alias_setter_adopts_exactly_what_the_persona_setter_adopts() -> None:
    """`persona_name` is an alias, so it must have every consequence `persona` has (#1081).

    The two were independent assignments of the same attribute -- a shape that stays
    correct only while adopting a persona has no consequences beyond the name. Now that
    adoption also recomputes the tool scope, the alias is pinned to the same path rather
    than trusted to repeat it.

    Killed by: src/uclone_x/agent/base.py :: self.persona = value
    Becomes: self._persona = value
    """
    wire = _RecordingConnector(["Answered"])
    registry = ToolRegistry()
    registry.register(_PermittedTool())
    registry.register(_WithheldTool())
    agent = BaseAgent(
        config=AgentConfig(agent_id="alias", name="Alias", system_prompt="You are the default."),
        llm=wire,
        tools=registry,
    )
    agent.define_persona(_scouting_persona())

    agent.persona_name = "late_scout"

    assert agent.persona == "late_scout"
    assert agent.config.allowed_tools == ("permitted_tool", *BASE_PERSONA_TOOLS)
    with pytest.raises(PermissionError, match="allowed_tools"):
        await agent.execute_tool_call("withheld_tool", {"text": "x"})

    assert (await agent.execute_turn("go")).is_completed is True
    assert _system_sent(wire.requests[-1]) == "You scout. Report what you find."


@pytest.mark.asyncio
async def test_an_anchor_hydrated_after_a_persona_was_adopted_stays_the_callers() -> None:
    """Hydrating *after* adopting a persona still leaves the caller's own system turn (#1081).

    The second revision of this fix recorded "an assignment moved the resolved persona" on
    the agent, which is a different claim from "the anchor in front of me was composed
    under a different persona". These two constructions are where they come apart: the
    assignment really did move the axis, and then the caller supplied an anchor of its own
    that the move says nothing about. Measured on the wire, base `4e5a2844` sends
    `OPERATOR HYDRATED PROMPT` for both and that revision sent the persona's prompt.

    The round trip is the second construction because it also lands on `None` and back,
    so a rule keyed on the *last* assignment alone reads it as no movement at all.

    Killed by: src/uclone_x/agent/base.py :: state, anchor_provenance=_AnchorWriter.CALLER
    Becomes: state, anchor_provenance=None
    """
    for label, names in (
        ("adopt", ("late_scout",)),
        ("round_trip", ("late_scout", None, "late_scout")),
    ):
        wire = _RecordingConnector(["Answered"])
        agent = BaseAgent(
            config=AgentConfig(
                agent_id=f"hydrated-after-{label}",
                name="HydratedAfter",
                system_prompt="CONFIG PROMPT",
            ),
            llm=wire,
        )
        agent.define_persona(_scouting_persona())

        for name in names:
            agent.persona = name
        agent.load_history(
            [ChatMessage(role=MessageRole.SYSTEM, content="OPERATOR HYDRATED PROMPT")]
        )

        assert (await agent.execute_turn("go")).is_completed is True
        assert _system_sent(wire.requests[-1]) == "OPERATOR HYDRATED PROMPT", label


@pytest.mark.asyncio
async def test_an_operators_anchor_loaded_before_a_persona_is_adopted_stays_the_operators() -> None:
    """C11: the caller's own system turn survives a real persona adopted after it (#1081).

    `load_history` puts an operator-composed system turn in place, and then a persona that
    resolves is adopted. Base `a3fcb675` and this head both send the operator's text here,
    and #1081 requires it: replacing a system turn a caller supplied on purpose is the
    failure mode it records as measured and rejected for PR #937. `effective_system_prompt`
    reports the persona's prompt meanwhile. That is the one disagreement the fix leaves in
    place on purpose, because the agent did not compose this anchor and cannot say which
    axis position it came from.

    The C1/C2 test above cannot see the mutation below. Its assignments resolve to nothing,
    so an anchor stamped with "the persona in force when `load_history` ran" (`None`) still
    matches. Only an adoption that resolves separates "the caller wrote it" from "written
    under no persona".

    Killed by: src/uclone_x/agent/base.py ::
        state, anchor_provenance=_AnchorWriter.CALLER
    Becomes: state, anchor_provenance=self._resolved_persona()
    """
    wire = _RecordingConnector(["Answered"])
    agent = BaseAgent(
        config=AgentConfig(agent_id="loaded-then-adopt", name="C11", system_prompt="CONFIG PROMPT"),
        llm=wire,
    )
    agent.define_persona(_scouting_persona())
    agent.load_history([ChatMessage(role=MessageRole.SYSTEM, content="OPERATOR HYDRATED PROMPT")])

    agent.persona = "late_scout"

    assert (await agent.execute_turn("go")).is_completed is True
    assert _system_sent(wire.requests[-1]) == "OPERATOR HYDRATED PROMPT"
    assert agent.effective_system_prompt == "You scout. Report what you find."


@pytest.mark.asyncio
async def test_a_store_restored_agent_seeded_anchor_is_re_resolved_when_a_persona_is_adopted(
    tmp_path: Path,
) -> None:
    """A session restored from the store follows a persona adopted after it (#1152).

    This test was `test_known_gap_a_store_restored_session_keeps_its_anchor_after_a_persona_is_adopted`
    and pinned the opposite: `CONFIG PROMPT` on the wire while `effective_system_prompt`
    reported the persona's. That gap was #1081's divergence surviving a round trip, because
    the stamp closing it for agent-seeded sessions lived on `_LiveSession` and not on
    `SessionState`, so `hydrate_session` attributed every restored anchor to the caller and
    never re-resolved it. The stamp is now a field of the record, so this anchor comes back
    saying what it is -- text *this agent* composed under no persona -- and adopting one
    makes it stale exactly as it would have before the session was ever persisted.

    `history[0]` is unchanged, which is the property #1078 protects: the anchor still
    reports what the first turn really was sent, and only the turn being built is re-framed.

    Killed by: src/uclone_x/agent/base.py ::
        anchor_provenance=_persisted_anchor_provenance(self.anchor_provenance),
    Becomes: anchor_provenance=None,
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    config = AgentConfig(agent_id="restored", name="Restored", system_prompt="CONFIG PROMPT")
    first = BaseAgent(config=config, llm=_RecordingConnector(["Before"]), store=store)
    assert (await first.execute_turn("first")).is_completed is True
    first.persist_session()

    wire = _RecordingConnector(["After"])
    restored = BaseAgent(config=config, llm=wire, store=store)
    restored.define_persona(_scouting_persona())
    assert restored.hydrate_session() is not None
    assert restored.history[0].content == "CONFIG PROMPT"

    restored.persona = "late_scout"

    assert (await restored.execute_turn("go")).is_completed is True
    assert _system_sent(wire.requests[-1]) == "You scout. Report what you find."
    assert _system_sent(wire.requests[-1]) == restored.effective_system_prompt
    assert restored.history[0].content == "CONFIG PROMPT"


@pytest.mark.asyncio
async def test_a_store_restored_callers_anchor_is_still_the_callers_after_a_persona_is_adopted(
    tmp_path: Path,
) -> None:
    """The store round trip must not turn a caller's anchor into the agent's (#1152).

    This is the half that makes #1152 a persistence question rather than a one-line stamp
    change. C11 already holds in memory: an anchor supplied through `load_history` is left
    alone whatever the axis does. Persisting it and restoring it must not lose that, and
    the cheap fix for the test above -- stamping restored sessions with the axis position
    in force at restore time -- loses exactly this, re-resolving an anchor the caller wrote
    on purpose. That is the failure mode #1081 records as measured and rejected for #937.

    What holds it is that `CALLER` is a value the record carries, not an assumption made
    about every record.

    Killed by: src/uclone_x/agent/base.py :: return AnchorProvenance(author=AnchorAuthor.CALLER)
    Becomes: return AnchorProvenance(author=AnchorAuthor.AGENT, persona=None)
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    config = AgentConfig(agent_id="restored-caller", name="RestoredCaller", system_prompt="CONFIG")
    first = BaseAgent(config=config, llm=_RecordingConnector(["Before"]), store=store)
    first.load_history([ChatMessage(role=MessageRole.SYSTEM, content="OPERATOR HYDRATED PROMPT")])
    first.persist_session()

    wire = _RecordingConnector(["After"])
    restored = BaseAgent(config=config, llm=wire, store=store)
    restored.define_persona(_scouting_persona())
    assert restored.hydrate_session() is not None

    restored.persona = "late_scout"

    assert (await restored.execute_turn("go")).is_completed is True
    assert _system_sent(wire.requests[-1]) == "OPERATOR HYDRATED PROMPT"
    assert restored.effective_system_prompt == "You scout. Report what you find."


@pytest.mark.asyncio
async def test_a_record_with_no_recorded_provenance_is_left_alone_and_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A restored anchor nothing stamped is not re-resolved, and does not pass in silence (P6, #1152).

    Every record written before `anchor_provenance` existed reads this way, and there are
    only three things to do with it. Re-resolving it would discard a caller's own text on
    the strength of a provenance nobody recorded. Refusing the hydration would make every
    existing record unloadable. Leaving it alone *quietly* is the silent fallback P6
    forbids: the consequence -- a persona adopted here will not move the system turn the
    model is sent -- is invisible from the outside and looks exactly like a bug.

    So it is left alone and said out loud. The record is written through the real store and
    then stripped of the key, which is the shape a legacy record actually has.

    Killed by: src/uclone_x/agent/base.py :: if record is None:
    Becomes: if record is None and False:
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    config = AgentConfig(agent_id="legacy", name="Legacy", system_prompt="CONFIG PROMPT")
    first = BaseAgent(config=config, llm=_RecordingConnector(["Before"]), store=store)
    assert (await first.execute_turn("first")).is_completed is True
    first.persist_session()

    path = store.session_path(first.session_id)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record.pop("anchor_provenance") is not None
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    wire = _RecordingConnector(["After"])
    restored = BaseAgent(config=config, llm=wire, store=store)
    restored.define_persona(_scouting_persona())
    with caplog.at_level(logging.WARNING, logger="uclone_x.agent.base"):
        assert restored.hydrate_session() is not None
    assert "no recorded provenance" in caplog.text

    restored.persona = "late_scout"

    assert (await restored.execute_turn("go")).is_completed is True
    assert _system_sent(wire.requests[-1]) == "CONFIG PROMPT"


@pytest.mark.asyncio
async def test_a_session_restored_with_no_provenance_records_it_on_the_next_anchor(
    tmp_path: Path,
) -> None:
    """Persisting an unstamped session keeps saying "unknown", and a reset stops saying it (#1152).

    Two halves of one rule. Saving a session whose provenance was never recorded must not
    invent one -- `UNRECORDED` renders back to an absent field, so a round trip through this
    agent does not launder "nobody knows" into "the agent composed it under no persona".
    And the absence is not permanent: `reset_session` composes a fresh anchor from
    `effective_system_prompt`, so that anchor's provenance is knowable and is stamped on the
    write that creates it.

    Killed by: src/uclone_x/agent/base.py :: if provenance is _AnchorWriter.UNRECORDED:
    Becomes: if provenance is _AnchorWriter.CALLER:
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    config = AgentConfig(agent_id="relay", name="Relay", system_prompt="CONFIG PROMPT")
    seeder = BaseAgent(config=config, llm=_RecordingConnector(["Before"]), store=store)
    assert (await seeder.execute_turn("first")).is_completed is True
    seeder.persist_session()

    path = store.session_path(seeder.session_id)
    record = json.loads(path.read_text(encoding="utf-8"))
    record.pop("anchor_provenance")
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    relay = BaseAgent(config=config, llm=_RecordingConnector(["After"]), store=store)
    assert relay.hydrate_session() is not None
    assert relay.persist_session().anchor_provenance is None

    relay.define_persona(_scouting_persona())
    relay.persona = "late_scout"
    stamped = relay.reset_session().anchor_provenance
    assert stamped is not None
    assert stamped.author is AnchorAuthor.AGENT
    assert stamped.persona == _scouting_persona()


def test_a_caller_composed_anchor_cannot_be_recorded_under_a_persona() -> None:
    """`AnchorProvenance` refuses the one pairing that would read as a claim nobody can make.

    "The caller wrote this anchor under persona X" is not knowable: the agent did not
    compose the text and has no axis position to attribute it to. A record carrying that
    pair would be read by `_restored_anchor_provenance` as `CALLER` and the persona
    silently dropped -- a field written and never read. Refused at the constructor, so it
    cannot reach the store (P6).

    Killed by: src/uclone_x/agent/session.py ::
        if self.author is AnchorAuthor.CALLER and self.persona is not None:
    Becomes: if False:
    """
    assert AnchorProvenance(author=AnchorAuthor.CALLER).persona is None
    with pytest.raises(ValidationError):
        AnchorProvenance(author=AnchorAuthor.CALLER, persona=_scouting_persona())


@pytest.mark.asyncio
async def test_a_definition_registered_after_the_anchor_moves_the_wire_with_it() -> None:
    """`define_persona` moves the axis without any assignment, and the wire follows (#1081).

    The name is set at construction and resolves to nothing at that moment, so the session
    is anchored under `config.system_prompt`; the definition arrives afterwards and the
    axis resolves to the persona. Nothing was assigned, so a rule watching the setter sees
    no movement -- base `4e5a2844` and both earlier revisions of this fix were measured
    sending `CONFIG PROMPT` here while `effective_system_prompt` reported the persona's,
    which is the divergence #1081 is about, reached without touching `persona` at all.

    Comparing the anchor's stamped resolution against the current one covers it because
    the stamp is a `PersonaDefinition`, not a name: registering a definition for a name
    that had none changes what the name resolves to, and that is the comparison.

    Killed by: src/uclone_x/agent/base.py :: anchor_provenance=self._resolved_persona(),
    Becomes: anchor_provenance=_AnchorWriter.CALLER,
    """
    wire = _RecordingConnector(["Answered"])
    agent = BaseAgent(
        config=AgentConfig(agent_id="late-def", name="LateDef", system_prompt="CONFIG PROMPT"),
        llm=wire,
        persona="late_scout",
    )
    assert agent.history[0].content == "CONFIG PROMPT"

    agent.define_persona(_scouting_persona())

    assert (await agent.execute_turn("go")).is_completed is True
    assert _system_sent(wire.requests[-1]) == "You scout. Report what you find."
    assert _system_sent(wire.requests[-1]) == agent.effective_system_prompt


# --------------------------------------------------------------------------------------
# A session with no system anchor is sent what `effective_system_prompt` reports (#1091).
#
# Every test above starts from a `SYSTEM` turn at `history[0]`. The turn builder's other
# branch -- no anchor at all -- sent the sections alone, or no system turn whatsoever,
# while `effective_system_prompt` named the persona's or the configured prompt. Measured on
# `4a3d94b6` for every shape below, with and without a plan section: the wire carried
# `[]` or the plan section only, and the property reported the prompt.
# --------------------------------------------------------------------------------------


def _anchorless_agent(shape: str, wire: _RecordingConnector, ontology: Any = None) -> BaseAgent:
    """An agent whose active session holds no `SYSTEM` turn, reached by `shape`."""
    user_row = ChatMessage(role=MessageRole.USER, content="earlier question")
    config_prompt = "" if shape == "seeded_empty_then_persona" else "CONFIG PROMPT"
    agent = BaseAgent(
        config=AgentConfig(
            agent_id=f"anchorless-{shape}",
            name="A",
            system_prompt=config_prompt,
        ),
        llm=wire,
        ontology=ontology,
    )
    agent.define_persona(_scouting_persona())
    if shape in ("user_first_persona", "user_first_no_persona"):
        agent.load_history([user_row])
    elif shape == "empty_hydrated_persona":
        agent.load_history([])
    if shape != "user_first_no_persona":
        agent.persona = "late_scout"
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shape",
    [
        "user_first_persona",
        "user_first_no_persona",
        "empty_hydrated_persona",
        "seeded_empty_then_persona",
    ],
)
async def test_a_session_with_no_system_anchor_is_sent_the_prompt_it_reports(shape: str) -> None:
    """With no anchor to re-frame, the turn carries `effective_system_prompt` (#1091).

    `user_first_persona` is the issue's construction. The other three reach the same branch
    by other doors: no persona at all (so this is not a persona defect), `load_history([])`,
    and a session seeded with no prompt that a persona then arrives on. The history the
    caller loaded is left as loaded -- the synthesised turn is built on the way out and not
    written back, so the record still says no system turn was stored (#1078).

    Killed by: src/uclone_x/agent/base.py :: base_sys = self.effective_system_prompt
    Becomes: base_sys = ""
    """
    wire = _RecordingConnector(["Answered"])
    agent = _anchorless_agent(shape, wire)

    assert (await agent.execute_turn("go")).is_completed is True

    assert agent.effective_system_prompt != ""
    assert _system_sent(wire.requests[-1]) == agent.effective_system_prompt
    assert all(m.role is not MessageRole.SYSTEM for m in agent.history)


def _one_asserted_invariant() -> MagicMock:
    """An ontology that answers every turn with one asserted invariant, `ridge_is_mapped`."""
    invariant = MagicMock()
    invariant.tier.value = "asserted"
    invariant.name = "ridge_is_mapped"
    invariant.rule_expression = "ridge == mapped"
    ontology = MagicMock()
    ontology.get_active_invariants = MagicMock(return_value=[invariant])
    return ontology


@pytest.mark.asyncio
async def test_a_synthesised_system_turn_carries_the_sections_as_an_anchored_one_does() -> None:
    """The sections join the synthesised prompt, in the anchored path's order (#1091).

    Before the fix the sections *were* the whole system turn on this branch, so a section
    in force displaced the persona's prompt instead of joining it. The twin agent is seeded
    and so anchored; both must put the same text on the wire, prompt first. An asserted
    invariant stands in for the system-turn sections: the plan, which this test used
    before, no longer travels in the system turn at all.

    Killed by: src/uclone_x/agent/request_record.py :: .strip() if slow_context else identity
    Becomes: .strip() if False else identity
    """
    wires = {"anchorless": _RecordingConnector(["A"]), "anchored": _RecordingConnector(["B"])}
    anchorless = _anchorless_agent(
        "user_first_persona", wires["anchorless"], _one_asserted_invariant()
    )
    anchored = BaseAgent(
        config=AgentConfig(
            agent_id="anchored-twin",
            name="T",
            system_prompt="CONFIG PROMPT",
        ),
        llm=wires["anchored"],
        ontology=_one_asserted_invariant(),
    )
    anchored.define_persona(_scouting_persona())
    anchored.persona = "late_scout"
    for agent in (anchorless, anchored):
        assert (await agent.execute_turn("go")).is_completed is True

    sent = _system_sent(wires["anchorless"].requests[-1])
    assert sent.startswith(
        f"{anchorless.effective_system_prompt}\n\n[Active Domain Ontology Invariants]"
    )
    assert sent == _system_sent(wires["anchored"].requests[-1])


@pytest.mark.asyncio
async def test_an_anchorless_session_is_sent_the_prompt_framed_for_its_model() -> None:
    """The synthesised turn is the *effective* prompt, model-family framing included (#921).

    The default prompt embeds the canonical steerability policy, which
    `effective_system_prompt` re-frames for Hermes. Synthesising from the unframed base
    would put the canonical framing on the wire while the property reports Hermes's.

    Killed by: src/uclone_x/agent/base.py :: base_sys = self.effective_system_prompt
    Becomes: base_sys = self._system_prompt_base()
    """
    wire = _RecordingConnector(["Answered"])
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="anchorless-hermes",
            name="Hermes",
            llm_config=AgentLLMConfig(model_name="hermes3:8b"),
        ),
        llm=wire,
    )
    agent.load_history([ChatMessage(role=MessageRole.USER, content="earlier question")])

    assert (await agent.execute_turn("go")).is_completed is True

    sent = _system_sent(wire.requests[-1])
    assert HERMES_STEERABILITY_POLICY in sent
    assert sent == agent.effective_system_prompt


@pytest.mark.asyncio
async def test_an_anchorless_session_with_nothing_to_send_gets_no_system_turn() -> None:
    """An empty resolution and no sections synthesise nothing, not an empty `SYSTEM` turn.

    `SessionState.seed` writes a system turn if and only if the prompt is non-empty; the
    turn builder keeps the same rule, so an agent configured with no prompt sends none.

    Killed by: src/uclone_x/agent/base.py :: system_message=anchored_turn or bool(resolved),
    Becomes: system_message=True,
    """
    wire = _RecordingConnector(["Answered"])
    agent = BaseAgent(
        config=AgentConfig(agent_id="promptless", name="P", system_prompt=""), llm=wire
    )
    agent.load_history([ChatMessage(role=MessageRole.USER, content="earlier question")])

    assert (await agent.execute_turn("go")).is_completed is True

    assert agent.effective_system_prompt == ""
    assert [m for m in wire.requests[-1].messages if m.role is MessageRole.SYSTEM] == []


# --------------------------------------------------------------------------------------
# A definition that arrives for the persona in force applies its tools as well (#1153).
#
# The setter half of this defect was #1081: a persona adopted by assignment moved the
# prompt and not the tool restriction. `define_persona` is the other way to move what the
# name in force resolves to, with no assignment at all, and it had the same gap. The
# prompt followed the definition because `effective_system_prompt` re-resolves on every
# read, and since #1087 the wire did too (the C10 test above). The tool scope is stored,
# not re-resolved, so it stayed at whatever the name resolved to before the definition.
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_definition_for_the_persona_in_force_restricts_tools_as_well_as_prompt() -> None:
    """The issue's reproduction: name set first, definition registered afterwards (#1153).

    Before the definition the name resolves to nothing, so the agent is unrestricted and
    that is correct. After it, the agent sends the scout's prompt, so it must also be held
    to the scout's tools. It must not be offered the withheld tool either, and it must be
    refused when it calls that tool directly. Asserting the prompt alone could not see
    this, because the prompt was already right.

    Killed by: src/uclone_x/agent/base.py :: self._apply_persona_tool_scope()  # a registration can move the persona in force
    Becomes:
    """
    registry = ToolRegistry()
    registry.register(_PermittedTool())
    registry.register(_WithheldTool())
    wire = _RecordingConnector(["Answered"])
    agent = BaseAgent(
        config=AgentConfig(agent_id="n", name="N", system_prompt="CONFIG PROMPT"),
        llm=wire,
        tools=registry,
        persona="late_scout",
    )

    # Before: `late_scout` resolves to nothing, so there is no restriction to apply.
    assert agent.config.allowed_tools == ()
    assert (await agent.execute_tool_call("withheld_tool", {"text": "x"})).status is (
        ToolResultStatus.SUCCESS
    )

    agent.define_persona(_scouting_persona())

    assert (await agent.execute_turn("go")).is_completed is True
    assert _system_sent(wire.requests[-1]) == "You scout. Report what you find."
    assert agent.config.allowed_tools == ("permitted_tool", *BASE_PERSONA_TOOLS)
    assert [t.name for t in wire.requests[-1].tools] == ["permitted_tool"]
    assert (await agent.execute_tool_call("permitted_tool", {"text": "x"})).status is (
        ToolResultStatus.SUCCESS
    )
    with pytest.raises(PermissionError, match="allowed_tools"):
        await agent.execute_tool_call("withheld_tool", {"text": "x"})


@pytest.mark.asyncio
async def test_a_definition_recomputes_the_tool_scope_by_the_setters_rule() -> None:
    """A definition applies the rule `_apply_persona_tool_scope` documents, nothing else (#1153).

    Three cases, each of which a different wrong fix would get wrong:

    * an operator's own `allowed_tools` still wins outright over a definition that arrives
      for the persona in force;
    * redefining the persona in force with other tools replaces its tools rather than
      keeping the old definition's, or adding the two together;
    * registering a name that is not in force leaves the scope as it was.

    Killed by: src/uclone_x/agent/base.py :: self._apply_persona_tool_scope()  # a registration can move the persona in force
    Becomes:
    """
    operator_scoped = _agent_with_both_tools(
        AgentConfig(agent_id="op", name="Op", allowed_tools=("withheld_tool",))
    )
    operator_scoped.persona = "late_scout"
    operator_scoped.define_persona(_scouting_persona())
    assert operator_scoped.config.allowed_tools == ("withheld_tool",)

    agent = _agent_with_both_tools(AgentConfig(agent_id="redef", name="Redef"))
    agent.persona = "late_scout"
    assert agent.config.allowed_tools == ("permitted_tool", *BASE_PERSONA_TOOLS)

    agent.define_persona(
        PersonaDefinition(
            name="late_scout",
            role="Scout",
            system_prompt="You scout differently.",
            allowed_tools=("withheld_tool",),
        )
    )
    assert agent.effective_system_prompt == "You scout differently."
    assert agent.config.allowed_tools == ("withheld_tool", *BASE_PERSONA_TOOLS)
    with pytest.raises(PermissionError, match="allowed_tools"):
        await agent.execute_tool_call("permitted_tool", {"text": "x"})

    agent.define_persona(
        PersonaDefinition(
            name="bystander",
            role="Bystander",
            system_prompt="You are not in force.",
            allowed_tools=("permitted_tool",),
        )
    )
    assert agent.config.allowed_tools == ("withheld_tool", *BASE_PERSONA_TOOLS)


# --------------------------------------------------------------------------------------
# A stable system turn: volatile state rides at the tail of the request.
#
# Every provider cache (vLLM/Ollama prefix KV reuse, Anthropic and OpenAI prompt caching)
# keys on the token prefix, so a byte that moves in the system turn invalidates everything
# after it. The plan, the memory facts and the tool-scoping notice change from turn to
# turn; they travel in a `[Turn Context]` block on the last user turn instead, which is
# never written to history.
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_plan_step_completed_between_turns_leaves_the_system_turn_byte_identical() -> None:
    """Ticking a plan step changes the tail of the next request, not its system turn.

    Before, the plan section was composed into the system turn, so the prefix every
    provider caches changed on each `update_step_status` and the whole conversation was
    re-prefilled. The first turn's prompt is also sent back as the user wrote it: the
    block is attached on the way out and never persisted, so history stays a pure prefix.

    Killed by: src/uclone_x/agent/base.py :: turn_sections.append(plan_section)
    Becomes: sections.append(plan_section)
    """
    wire = _RecordingConnector(["one", "two"])
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="prefix-plan",
            name="P",
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=wire,
    )
    agent.create_plan("Survey", ["map the ridge", "report"])
    assert (await agent.execute_turn("start")).is_completed is True
    agent.update_step_status(1, completed=True)
    assert (await agent.execute_turn("continue")).is_completed is True

    first, second = wire.requests
    assert _system_sent(first) == _system_sent(second)
    assert "map the ridge" not in _system_sent(second)
    tail = second.messages[-1]
    assert tail.role is MessageRole.USER
    assert tail.content is not None
    assert tail.content.startswith(f"continue\n\n{TURN_CONTEXT_HEADER}")
    assert "[x]" in tail.content and "map the ridge" in tail.content
    assert [m.content for m in second.messages[1:3]] == ["start", "one"]


@pytest.mark.asyncio
async def test_the_turn_context_joins_the_user_prompt_rather_than_following_it() -> None:
    """On the first step the block is merged into the prompt: no two user turns in a row.

    Strict chat templates (gemma, mistral under vLLM) refuse consecutive user turns, so a
    separate trailing message would fail every turn that carried a plan.

    Killed by: src/uclone_x/agent/request_record.py :: and last.role is MessageRole.USER and
    Becomes: and False and
    """
    wire = _RecordingConnector(["one"])
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="prefix-merge",
            name="P",
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=wire,
    )
    agent.create_plan("Survey", ["map the ridge"])
    assert (await agent.execute_turn("start")).is_completed is True

    roles = [m.role for m in wire.requests[-1].messages]
    assert roles == [MessageRole.SYSTEM, MessageRole.USER]
    assert (wire.requests[-1].messages[-1].content or "").startswith(
        f"start\n\n{TURN_CONTEXT_HEADER}"
    )


class _RecordingToolConnector(CannedToolResponseConnector):
    """Calls its canned tools once, answers after, and keeps every request it was sent."""

    def __init__(self, tool_calls: tuple[ToolCallRequest, ...]) -> None:
        super().__init__(tool_calls=tool_calls, content=None)
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        return await super().generate(request)


@pytest.mark.asyncio
async def test_the_step_after_a_tool_result_still_says_which_tools_were_withheld() -> None:
    """The scoping notice travels on every step, as its own user turn after a tool result.

    The scoped tool list is sent on every step of a turn, but the step rebuild dropped the
    notice: from the second step on the model saw fewer tools with no word that any were
    withheld (P6). After a TOOL message the block cannot be merged, so it follows it.

    Killed by: src/uclone_x/agent/base.py :: step_messages = assemble_request_messages(req_layers)
    Becomes: step_messages = assemble_request_messages(self._prepare_turn_layers())
    """
    registry = ToolRegistry()
    registry.register(DummyEchoTool())
    registry.register(_WithheldTool())
    wire = _RecordingToolConnector(
        (ToolCallRequest(id="call_1", name="dummy_echo", arguments={"text": "hi"}),)
    )
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="prefix-scoped",
            name="P",
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=wire,
        tools=registry,
        tool_scoper=LexicalToolScoper(top_k=1),
    )

    assert (await agent.execute_turn("use dummy_echo on this")).is_completed is True

    assert len(wire.requests) == 2
    for request in wire.requests:
        assert all(TURN_CONTEXT_HEADER not in (m.content or "") for m in request.messages[:-1])
        assert "withheld" in (request.messages[-1].content or "")
    step_two = wire.requests[1].messages
    assert [m.role for m in step_two[-2:]] == [MessageRole.TOOL, MessageRole.USER]
    assert (step_two[-1].content or "").startswith(TURN_CONTEXT_HEADER)


# --------------------------------------------------------------------------------------
# An errored turn reports the tools that ran before it failed (#1366)
# --------------------------------------------------------------------------------------


class _NoteParams(BaseModel):
    text: str = Field(default="")


class _NoteWriter(BaseTool[_NoteParams]):
    """Writes `note.md` and says so, as `file_write` does."""

    name = "note_writer"
    description = "Writes note.md."
    writes_files = True

    def __init__(self, workspace: Path) -> None:
        super().__init__()
        self.workspace = workspace

    def run(self, params: _NoteParams, context: ToolContext) -> dict[str, Any]:
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.workspace / "note.md").write_text("hello", encoding="utf-8")
        return {"path": "note.md"}


class _WriteThenRaise(MockLLMConnector):
    """Step 1 asks for the writer; step 2 raises `failure` at the provider."""

    def __init__(self, failure: Exception) -> None:
        super().__init__(
            default_response="ok",
            tool_calls=(ToolCallRequest(id="c1", name="note_writer", arguments={"text": "x"}),),
        )
        self.failure = failure
        self.calls = 0

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.calls += 1
        if self.calls >= 2:
            raise self.failure
        return await super().generate(request)


def _writing_agent(tmp_path: Path, failure: Exception) -> tuple[BaseAgent, _WriteThenRaise]:
    registry = ToolRegistry()
    registry.register(cast(Any, _NoteWriter(tmp_path / "ws")))
    llm = _WriteThenRaise(failure)
    agent = BaseAgent(config=AgentConfig(agent_id="scribe", name="Scribe"), tools=registry, llm=llm)
    return agent, llm


class _WriteThenRaiseTool(_NoteWriter):
    """Writes `note.md`, then raises out of `execute` itself: the file stays.

    `BaseTool.execute` turns a raise inside `run` into an error result, so this overrides
    `execute`, as a tool with its own transport can, to reach the agent's handler.
    """

    name = "note_writer"

    async def execute(
        self,
        params: dict[str, Any] | ToolContext | None = None,
        context: ToolContext | None = None,
        **kwargs: Any,
    ) -> Any:
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.workspace / "note.md").write_text("hello", encoding="utf-8")
        raise OSError("disk went away after the write")


class TestAToolThatRaisedKeepsItsDeclarations:
    """A tool that raised had already started; its record still says it can write (#1366)."""

    @pytest.mark.asyncio
    async def test_a_writer_that_raised_is_recorded_as_a_writer(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/agent/base.py :: writes_files=declared_writes,  # raised partway
        Becomes: writes_files=False,  # raised partway
        """
        registry = ToolRegistry()
        registry.register(cast(Any, _WriteThenRaiseTool(tmp_path / "ws")))
        llm = MockLLMConnector(
            default_response="ok",
            tool_calls=(ToolCallRequest(id="c1", name="note_writer", arguments={"text": "x"}),),
        )
        agent = BaseAgent(
            config=AgentConfig(agent_id="scribe", name="Scribe"), tools=registry, llm=llm
        )
        result = await agent.execute_turn("write it")

        assert (tmp_path / "ws" / "note.md").exists()
        (record,) = [e for e in result.tool_executions if e.tool_call_id == "c1"]
        assert record.status == ToolResultStatus.ERROR
        assert record.writes_files is True


class TestAnErroredTurnKeepsItsTools:
    """`execute_turn`'s two error handlers returned `tool_executions=()` (#1366).

    The step-budget refusal already passed the accumulated list. These two did not, so a
    turn whose step 1 wrote a file and whose step 2 met a provider error or a budget
    reported that it ran no tools -- and every reader of the result, a room above all,
    took that as "none".
    """

    @pytest.mark.asyncio
    async def test_a_provider_error_after_a_tool_step_keeps_the_step(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/agent/base.py :: tool_executions=tuple(tool_executions),  # ran before the failure
        Becomes: tool_executions=(),  # ran before the failure
        """
        agent, llm = _writing_agent(tmp_path, RuntimeError("provider went away"))
        result = await agent.execute_turn("write it")

        assert llm.calls == 2
        assert result.error is not None
        assert (tmp_path / "ws" / "note.md").exists()
        assert [e.tool_name for e in result.tool_executions] == ["note_writer"]
        assert result.tool_executions[0].writes_files is True
        assert [c.name for c in result.tool_calls] == ["note_writer"]
        assert result.tool_executions_complete is True

    @pytest.mark.asyncio
    async def test_a_budget_ceiling_after_a_tool_step_keeps_the_step(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/agent/base.py :: tool_executions=tuple(tool_executions),  # ran before the ceiling
        Becomes: tool_executions=(),  # ran before the ceiling
        """
        agent, llm = _writing_agent(tmp_path, BudgetExceededError("ceiling reached"))
        result = await agent.execute_turn("write it")

        assert llm.calls == 2
        assert result.stop_reason == "budget_exceeded"
        assert [e.tool_name for e in result.tool_executions] == ["note_writer"]
        assert result.tool_executions_complete is True

    @pytest.mark.asyncio
    async def test_a_failure_while_tools_run_says_the_list_is_incomplete(
        self, tmp_path: Path
    ) -> None:
        """Tools may have run in the failing step with no record: a lower bound, said so.

        Killed by: src/uclone_x/agent/base.py :: tools_unreported = True
        Becomes: tools_unreported = False
        Killed by: src/uclone_x/agent/base.py :: tool_executions_complete=not tools_unreported,  # a failure mid-step
        Becomes: tool_executions_complete=True,  # a failure mid-step
        """
        agent, _ = _writing_agent(tmp_path, RuntimeError("unused"))

        async def _raise_mid_step(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("a tool step failed partway")

        cast(Any, agent)._execute_tools = _raise_mid_step
        result = await agent.execute_turn("write it")

        assert result.error is not None
        assert result.tool_executions == ()
        assert result.tool_executions_complete is False

    @pytest.mark.asyncio
    async def test_a_ceiling_while_tools_run_says_the_list_is_incomplete(
        self, tmp_path: Path
    ) -> None:
        """The budget handler's own flag: a ceiling met mid-step leaves no record either.

        Killed by: src/uclone_x/agent/base.py :: tool_executions_complete=not tools_unreported,  # a ceiling mid-step
        Becomes: tool_executions_complete=True,  # a ceiling mid-step
        """
        agent, _ = _writing_agent(tmp_path, RuntimeError("unused"))

        async def _ceiling_mid_step(*_args: Any, **_kwargs: Any) -> Any:
            raise BudgetExceededError("ceiling reached while a tool ran")

        cast(Any, agent)._execute_tools = _ceiling_mid_step
        result = await agent.execute_turn("write it")

        assert result.stop_reason == "budget_exceeded"
        assert result.tool_executions == ()
        assert result.tool_executions_complete is False
