"""Unit tests for BaseAgent reactive state machine and lifecycle loop (FR-1, P1, P4, P6, P8)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from tests.conftest import finish_after_tools
from uclone_x.a2a.wire import agent_event_from_wire_json, agent_event_to_wire_json
from uclone_x.agent import BaseAgent
from uclone_x.agent.models import (
    AgentConfig,
    AgentLLMConfig,
    AgentState,
    TurnResult,
)
from uclone_x.agent.session import CompactionResult
from uclone_x.core.provenance import (
    AttemptRecord,
    ExecutionPath,
    Provenance,
    ServiceRef,
)
from uclone_x.engine.event_bus import (
    AgentEvent,
    EventBus,
    EventBusError,
    EventSource,
    EventType,
)
from uclone_x.errors import (
    InvalidStateTransitionError,
    LLMConnectorNotConfiguredError,
    MissingProvenanceError,
)
from uclone_x.llm import context_window as context_window_module
from uclone_x.llm.compactor import estimate_message_tokens, resolve_model_context_limit
from uclone_x.llm.connectors.ollama import OllamaConnector
from uclone_x.llm.context_window import DEFAULT_OLLAMA_NUM_CTX, OllamaContextWindows
from uclone_x.llm.models import (
    ChatMessage,
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.llm.protocols import LLMProviderProtocol
from uclone_x.sandbox.models import IsolationLevel
from uclone_x.telemetry import SpanStatus, TelemetryTracer
from uclone_x.tools.models import ToolResult
from uclone_x.tools.protocols import ToolProtocol, ToolRegistryProtocol
from uclone_x.ui.app import AgentSessionManager, create_ui_app


def _sample_config(agent_id: str = "agent-001") -> AgentConfig:
    return AgentConfig(
        agent_id=agent_id,
        name="Test Agent",
        role="Unit Tester",
        system_prompt="You are a helpful test assistant.",
        llm_config=AgentLLMConfig(model_name="mock-model", temperature=0.5),
    )


@pytest.mark.asyncio
async def test_agent_initialization() -> None:
    config = _sample_config("agent-init")
    agent = BaseAgent(config=config)

    assert agent.agent_id == "agent-init"
    assert agent.state == AgentState.IDLE
    assert agent.config == config
    assert agent.context.agent_id == "agent-init"
    assert agent.context.current_state == AgentState.IDLE


@pytest.mark.asyncio
async def test_valid_and_invalid_state_transitions() -> None:
    config = _sample_config("agent-transitions")
    agent = BaseAgent(config=config)

    # Valid transitions
    agent.transition_to(AgentState.INGESTING)
    assert agent.state == AgentState.INGESTING
    assert agent.context.current_state == AgentState.INGESTING

    agent.transition_to(AgentState.REASONING)
    assert agent.state == AgentState.REASONING

    agent.transition_to(AgentState.CALLING_TOOL)
    assert agent.state == AgentState.CALLING_TOOL

    agent.transition_to(AgentState.AWAITING_INPUT)
    assert agent.state == AgentState.AWAITING_INPUT

    agent.transition_to(AgentState.REASONING)
    assert agent.state == AgentState.REASONING

    agent.transition_to(AgentState.EMITTING_RESPONSE)
    assert agent.state == AgentState.EMITTING_RESPONSE

    agent.transition_to(AgentState.IDLE)
    assert agent.state == AgentState.IDLE

    # Invalid transition (IDLE -> CALLING_TOOL)
    with pytest.raises(InvalidStateTransitionError):
        agent.transition_to(AgentState.CALLING_TOOL)


@pytest.mark.asyncio
async def test_agent_lifecycle_start_and_stop() -> None:
    bus = EventBus()
    await bus.start()
    try:
        config = _sample_config("agent-lifecycle")
        agent = BaseAgent(config=config, bus=bus)

        await agent.start()
        # Idempotent start
        await agent.start()
        assert agent.state == AgentState.IDLE

        # Publish an event to the agent
        event = AgentEvent(
            type=EventType.USER_INPUT,
            topic="agent.agent-lifecycle",
            recipient_id="agent-lifecycle",
            payload={"message": "Hello Agent!"},
        )
        await bus.publish(event)

        # Allow background loop task to process
        await asyncio.sleep(0.05)

        await agent.stop()
        assert agent.state == AgentState.TERMINATED
    finally:
        await bus.stop()


@pytest.mark.asyncio
async def test_execute_turn_without_a_connector_refuses_instead_of_echoing() -> None:
    """P6: no connector wired means no result, not a canned one (issue #136a).

    `execute_turn` used to answer `f"Ack: {input}"` with `is_completed=True` and
    `provenance` naming `agent.core/BaseAgent` as both `requested` and `served_by`, so
    `degraded` computed `False` and a caller could not tell the literal from a model's
    answer. That is P6's "hardcoded default ... returned as if it were a success", and
    its Classification Procedure reaches *forbidden* at question 1 — no `Provenance`
    value rescues it, so the turn produces no value at all.
    """
    config = _sample_config("agent-echo")
    agent = BaseAgent(config=config)

    with pytest.raises(LLMConnectorNotConfiguredError) as excinfo:
        await agent.execute_turn("Test input message")

    # The configured model the caller believed it was talking to is named.
    assert "mock-model" in str(excinfo.value)
    # Refusing is a precondition, so the agent is left exactly as it was found: no
    # state transition, no turn consumed, and the input never entered the history.
    assert agent.state == AgentState.IDLE
    assert agent.context.current_state == AgentState.IDLE
    assert all("Test input message" not in (m.content or "") for m in agent._history)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_refusal_without_a_connector_does_not_depend_on_a_configured_model() -> None:
    """An agent with no model *and* no connector refuses too (issue #136a).

    `AgentLLMConfig.model_name` defaults to `None`, so a rule that only refused when a
    model was configured would have left the default configuration answering `"Ack: …"`
    with `is_completed=True` — the same undetectable substitution, reached by the
    commoner path.
    """
    config = AgentConfig(agent_id="agent-echo-nomodel", name="No Model")
    assert config.llm_config.model_name is None
    agent = BaseAgent(config=config)

    with pytest.raises(LLMConnectorNotConfiguredError):
        await agent.execute_turn("Test input message")


@pytest.mark.asyncio
async def test_execute_turn_with_mock_llm() -> None:
    config = _sample_config("agent-llm")
    mock_llm = MagicMock(spec=LLMProviderProtocol)

    prov = Provenance(
        path=ExecutionPath.PRIMARY,
        requested=ServiceRef(provider="mock", model="mock-model"),
        served_by=ServiceRef(provider="mock", model="mock-model"),
    )
    mock_resp = ModelResponse(
        content="Generated LLM response",
        usage=TokenUsage(provider="mock", model="mock-model", input_tokens=10, output_tokens=20),
        finish_reason=FinishReason.STOP,
        model_name="mock-model",
        provenance=prov,
    )
    mock_llm.generate = AsyncMock(return_value=mock_resp)

    finish_after_tools(mock_llm)

    agent = BaseAgent(config=config, llm=mock_llm)
    result = await agent.execute_turn("Generate something")

    assert result.is_completed is True
    assert result.content == "Generated LLM response"
    assert result.provenance == prov
    assert agent.state == AgentState.IDLE
    mock_llm.generate.assert_called_once()


@pytest.mark.asyncio
async def test_execute_turn_with_unconfigured_model_name_passes_none_to_llm() -> None:
    """Agent with model_name unset passes model=None in LLMRequest rather than 'default' (#279)."""
    config = AgentConfig(
        agent_id="agent-unconfigured-model",
        name="Unconfigured Model Agent",
        llm_config=AgentLLMConfig(model_name=None),
    )
    mock_llm = MagicMock(spec=LLMProviderProtocol)
    prov = Provenance.primary("mock", "mock-model")
    mock_resp = ModelResponse(
        content="Generated response",
        usage=TokenUsage(provider="mock", model="mock-model", input_tokens=10, output_tokens=20),
        finish_reason=FinishReason.STOP,
        model_name="mock-model",
        provenance=prov,
    )
    mock_llm.generate = AsyncMock(return_value=mock_resp)

    finish_after_tools(mock_llm)

    agent = BaseAgent(config=config, llm=mock_llm)
    result = await agent.execute_turn("Hello agent")

    assert result.is_completed is True
    mock_llm.generate.assert_called_once()
    called_req: LLMRequest = mock_llm.generate.call_args[0][0]
    assert called_req.model is None


@pytest.mark.asyncio
async def test_execute_turn_with_tools() -> None:
    config = _sample_config("agent-tools")
    mock_llm = MagicMock(spec=LLMProviderProtocol)
    mock_tools = MagicMock(spec=ToolRegistryProtocol)

    tool_call = ToolCallRequest(
        id="call_1",
        name="calculator",
        arguments={"a": 2, "b": 3},
    )
    mock_resp = ModelResponse(
        content="Calculated result",
        tool_calls=(tool_call,),
        usage=TokenUsage(provider="mock", model="mock-model", input_tokens=15, output_tokens=25),
        finish_reason=FinishReason.TOOL_CALLS,
        model_name="mock-model",
        provenance=None,
    )
    mock_llm.generate = AsyncMock(return_value=mock_resp)

    mock_tool = MagicMock(spec=ToolProtocol)
    mock_tool.name = "calculator"
    mock_tool.description = "Calculator tool"
    mock_tool.parameters_schema = {}

    tool_res = ToolResult(
        output={"result": 5},
        success=True,
        execution_time_ms=5.0,
        isolation_level=IsolationLevel.WORKSPACE,
        provenance=Provenance(
            path=ExecutionPath.PRIMARY,
            requested=ServiceRef(provider="tools", model="calculator"),
            served_by=ServiceRef(provider="tools", model="calculator"),
        ),
    )
    mock_tool.execute = AsyncMock(return_value=tool_res)
    mock_tools.get = MagicMock(return_value=mock_tool)
    mock_tools.list_tools = MagicMock(return_value=[mock_tool])

    finish_after_tools(mock_llm)

    agent = BaseAgent(config=config, llm=mock_llm, tools=mock_tools)
    result = await agent.execute_turn("Compute 2 + 3")

    assert result.is_completed is True
    # The turn's content is the model's FINAL answer, after it has seen its tool
    # results (P4, amended 2026-09-05). It was the first step's text, which a
    # tool-using turn leaves as a fragment or empty.
    assert result.content == "Done."
    assert len(result.tool_calls) == 1, "the calls that ran are still reported"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "calculator"
    assert len(result.tool_executions) == 1
    assert result.tool_executions[0].tool_name == "calculator"
    assert result.tool_executions[0].arguments == {"a": 2, "b": 3}
    assert result.tool_executions[0].output == {"result": 5}
    assert result.tool_executions[0].status == "success"
    assert result.tool_executions[0].error is None
    assert result.tool_executions[0].duration_ms == 5.0
    mock_tool.execute.assert_called_once()


@pytest.mark.asyncio
async def test_execute_turn_error_handling() -> None:
    config = _sample_config("agent-err")
    mock_llm = MagicMock(spec=LLMProviderProtocol)
    mock_llm.generate = AsyncMock(side_effect=RuntimeError("Provider exploded"))

    finish_after_tools(mock_llm)

    agent = BaseAgent(config=config, llm=mock_llm)
    result = await agent.execute_turn("Cause an error")

    assert result.is_completed is False
    assert result.error is not None
    assert "Provider exploded" in result.error
    assert agent.state == AgentState.ERROR
    # P6: A failed turn must carry degraded provenance, not clean primary
    assert result.provenance is not None
    assert result.provenance.degraded is True
    assert result.provenance.path is ExecutionPath.FAILOVER
    assert result.provenance.served_by.provider == "agent.core"
    assert len(result.provenance.attempts) == 1
    assert result.provenance.attempts[0].error_class == "RuntimeError"


@pytest.mark.asyncio
async def test_consecutive_turns_against_a_failing_provider_both_return_results() -> None:
    """A failed turn must not leave the agent unable to take the next one (issue #136b).

    `VALID_TRANSITIONS[ERROR]` is `{IDLE, TERMINATED}`, and `transition_to(INGESTING)`
    sits outside `execute_turn`'s `try`. So turn 2 used to raise
    `InvalidStateTransitionError` *out of* `execute_turn`, where its own handler could
    not convert it into an error `TurnResult`. Driven through the bus this meant the
    agent answered every other event and dropped the alternates.
    """
    config = _sample_config("agent-err-twice")
    mock_llm = MagicMock(spec=LLMProviderProtocol)
    mock_llm.generate = AsyncMock(side_effect=RuntimeError("provider down"))
    finish_after_tools(mock_llm)
    agent = BaseAgent(config=config, llm=mock_llm)

    first = await agent.execute_turn("one")
    # ERROR stays observable between turns rather than being cleared on the way out.
    assert agent.state is AgentState.ERROR
    second = await agent.execute_turn("two")

    for result in (first, second):
        assert isinstance(result, TurnResult)
        assert result.is_completed is False
        assert result.error is not None
        assert "provider down" in result.error
        assert result.provenance is not None
        assert result.provenance.degraded is True
        assert result.provenance.path is ExecutionPath.FAILOVER
    assert (first.turn_index, second.turn_index) == (1, 2)
    assert mock_llm.generate.await_count == 2


@pytest.mark.asyncio
async def test_event_loop_answers_every_event_from_a_failing_provider() -> None:
    """The bus-level shape of issue #136b: no dropped alternate events.

    Each failed turn publishes an `AGENT_REPLY` naming the failure rather than the loop
    absorbing an `InvalidStateTransitionError` for every second event.
    """
    config = _sample_config("agent-err-loop")
    bus = EventBus()
    await bus.start()
    mock_llm = MagicMock(spec=LLMProviderProtocol)
    mock_llm.generate = AsyncMock(side_effect=RuntimeError("provider down"))
    agent = BaseAgent(config=config, bus=bus, llm=mock_llm)
    await agent.start()
    caller = bus.register_publisher(sender_id="caller_1", source=EventSource.USER)
    replies = bus.subscribe(f"session.sess_{config.agent_id}")

    try:
        for i in range(3):
            await caller.publish(
                AgentEvent(
                    type=EventType.USER_INPUT,
                    recipient_id=agent.agent_id,
                    topic=f"agent.{agent.agent_id}",
                    payload={"message": f"question {i}"},
                )
            )
            await bus.wait_until_idle()

        received = [await asyncio.wait_for(replies.get(), timeout=1.0) for _ in range(3)]
        assert [e.payload["turn_index"] for e in received] == [1, 2, 3]
        for reply in received:
            assert reply.payload["is_completed"] == "False"
            # The cause travels with the reply; it is not log-only (P6).
            assert "provider down" in str(reply.payload["error"])
            assert reply.provenance is not None
            assert reply.provenance.degraded is True
            assert reply.provenance.path is ExecutionPath.FAILOVER
        assert agent.processing_errors == ()
    finally:
        replies.close()
        await agent.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_failed_turn_publishes_degraded_provenance_and_error_payload() -> None:
    """A failed turn must emit degraded provenance, not clean primary (Issue #150, P6)."""
    config = _sample_config("agent-failed-turn-prov")
    bus = EventBus()
    await bus.start()
    mock_llm = MagicMock(spec=LLMProviderProtocol)
    mock_llm.generate = AsyncMock(side_effect=RuntimeError("Provider connection timeout"))
    agent = BaseAgent(config=config, bus=bus, llm=mock_llm)
    await agent.start()
    caller = bus.register_publisher(sender_id="caller_1", source=EventSource.USER)
    replies = bus.subscribe(f"session.sess_{config.agent_id}")

    try:
        await caller.publish(
            AgentEvent(
                type=EventType.USER_INPUT,
                recipient_id=agent.agent_id,
                topic=f"agent.{agent.agent_id}",
                payload={"message": "Do something"},
            )
        )
        await bus.wait_until_idle()
        reply = await asyncio.wait_for(replies.get(), timeout=1.0)

        assert reply.type is EventType.AGENT_REPLY
        assert reply.payload["is_completed"] == "False"
        assert reply.payload["error"] == "Provider connection timeout"
        assert reply.provenance is not None
        assert reply.provenance.degraded is True
        assert reply.provenance.path is ExecutionPath.FAILOVER
        assert reply.provenance.served_by.provider == "agent.core"
        assert len(reply.provenance.attempts) == 1
        assert reply.provenance.attempts[0].error_class == "RuntimeError"
    finally:
        replies.close()
        await agent.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_agent_state_error_cleared_on_entry() -> None:
    """AgentState.ERROR is preserved between turns and cleared upon new turn entry (Issue #150)."""
    config = _sample_config("agent-clear-error-entry")
    mock_llm = MagicMock(spec=LLMProviderProtocol)
    # First turn fails, second turn succeeds
    mock_llm.generate = AsyncMock(
        side_effect=[
            RuntimeError("Provider outage"),
            ModelResponse(
                content="Recovered response",
                tool_calls=(),
                finish_reason=FinishReason.STOP,
                usage=TokenUsage(provider="mock", input_tokens=5, output_tokens=5),
                provenance=Provenance.primary("mock", "mock-model"),
            ),
        ]
    )

    finish_after_tools(mock_llm)

    agent = BaseAgent(config=config, llm=mock_llm)

    # Turn 1: Fails
    res1 = await agent.execute_turn("Prompt 1")
    assert res1.is_completed is False
    assert res1.provenance is not None
    assert res1.provenance.degraded is True
    # ERROR state is preserved and observable between turns
    assert agent.state is AgentState.ERROR
    assert agent.context.current_state is AgentState.ERROR

    # Turn 2: Succeeds - ERROR cleared on entry to IDLE -> INGESTING -> REASONING -> IDLE
    res2 = await agent.execute_turn("Prompt 2")
    assert res2.is_completed is True
    assert res2.content == "Recovered response"
    assert res2.provenance is not None
    assert res2.provenance.degraded is False
    assert agent.state is AgentState.IDLE
    assert agent.context.current_state is AgentState.IDLE


@pytest.mark.asyncio
async def test_process_event_filtering_and_interrupt() -> None:
    config = _sample_config("agent-filter")
    agent = BaseAgent(config=config)

    # Event for different recipient is ignored
    other_event = AgentEvent(
        type=EventType.USER_INPUT,
        recipient_id="other-agent",
        payload={"message": "Not for you"},
    )
    handled = await agent.process_event(other_event)
    assert handled is False

    # Interrupt event resets state to IDLE
    agent.transition_to(AgentState.INGESTING)
    interrupt_event = AgentEvent(
        type=EventType.INTERRUPT,
        recipient_id="agent-filter",
    )
    handled = await agent.process_event(interrupt_event)
    assert handled is True
    assert agent.state == AgentState.IDLE


@pytest.mark.asyncio
async def test_turns_are_serialized_and_correlations_preserved() -> None:
    """Two concurrent turns are serialized without history interleaving and carry correlation_id (Issue #60)."""
    config = _sample_config("agent-interleaving")
    bus = EventBus()

    call_order: list[str] = []

    mock_llm = MagicMock(spec=LLMProviderProtocol)

    turn_counter = 0

    async def mock_generate(req: object) -> ModelResponse:
        nonlocal turn_counter
        turn_counter += 1
        t_num = turn_counter
        call_order.append(f"start_turn_{t_num}")
        await asyncio.sleep(0.05)  # Simulate latency
        call_order.append(f"end_turn_{t_num}")
        return ModelResponse(
            content=f"Response to turn {t_num}",
            tool_calls=(),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(provider="mock", input_tokens=10, output_tokens=5, total_tokens=15),
            provenance=Provenance(
                path=ExecutionPath.PRIMARY,
                requested=ServiceRef(provider="agent.core", model="mock"),
                served_by=ServiceRef(provider="agent.core", model="mock"),
            ),
        )

    mock_llm.generate = AsyncMock(side_effect=mock_generate)

    agent = BaseAgent(config=config, bus=bus, llm=mock_llm)

    event1 = AgentEvent(
        event_id="evt_input_1",
        type=EventType.USER_INPUT,
        sender_id="caller_1",
        recipient_id="agent-interleaving",
        payload={"message": "First question"},
    )
    event2 = AgentEvent(
        event_id="evt_input_2",
        type=EventType.USER_INPUT,
        sender_id="caller_2",
        recipient_id="agent-interleaving",
        payload={"message": "Second question"},
    )

    sub = bus.subscribe("session.*")

    # Run two turns concurrently
    task1 = asyncio.create_task(agent.process_event(event1))
    task2 = asyncio.create_task(agent.process_event(event2))

    res1, res2 = await asyncio.gather(task1, task2)
    assert res1 is True
    assert res2 is True

    # Check execution order was strictly serialized (start 1 -> end 1 -> start 2 -> end 2)
    assert call_order == ["start_turn_1", "end_turn_1", "start_turn_2", "end_turn_2"]

    # Verify AGENT_REPLY events published to bus carry correct correlation_id
    reply1 = await sub.get()
    assert reply1.type == EventType.AGENT_REPLY
    assert reply1.correlation_id == "evt_input_1"
    assert reply1.recipient_id == "caller_1"
    assert reply1.payload["correlation_id"] == "evt_input_1"
    assert reply1.payload["turn_index"] == 1

    reply2 = await sub.get()
    assert reply2.type == EventType.AGENT_REPLY
    assert reply2.correlation_id == "evt_input_2"
    assert reply2.recipient_id == "caller_2"
    assert reply2.payload["correlation_id"] == "evt_input_2"
    assert reply2.payload["turn_index"] == 2


def _failover_provenance() -> Provenance:
    """A provenance that is *not* the unremarkable case, so a round trip can be told apart."""
    return Provenance(
        path=ExecutionPath.FAILOVER,
        requested=ServiceRef(provider="primary_llm", model="p-1"),
        served_by=ServiceRef(provider="secondary_llm", model="s-1"),
        attempts=(
            AttemptRecord(
                provider="primary_llm",
                model="p-1",
                error_class="RateLimitError",
                status_code=429,
            ),
        ),
    )


@pytest.mark.asyncio
async def test_published_reply_provenance_survives_both_transports() -> None:
    """A published result's P6 provenance survives the in-memory fastpath and the A2A wire (#117).

    The two transports P2 names are exercised against the same published event:

    * the in-process fastpath, which fans the frozen `AgentEvent` out **by reference** —
      every subscriber receives the identical object, so the `Provenance` instance is
      carried with no serialisation at all;
    * the A2A JSON wire, which serialises through `a2a/wire.py` and re-validates.

    Reading it back on either side is ordinary attribute access and ordinary validation:
    no payload-key lookup, no un-freezing, no JSON-mode dance at the call site.
    """
    config = _sample_config("agent-provenance")
    bus = EventBus()
    served = _failover_provenance()

    mock_llm = MagicMock(spec=LLMProviderProtocol)
    mock_llm.generate = AsyncMock(
        return_value=ModelResponse(
            content="Answered by the failover provider",
            tool_calls=(),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(provider="secondary_llm", input_tokens=4, output_tokens=6),
            provenance=served,
        )
    )

    agent = BaseAgent(config=config, bus=bus, llm=mock_llm)
    sub_a = bus.subscribe("session.*")
    sub_b = bus.subscribe("session.*")

    assert await agent.process_event(
        AgentEvent(
            event_id="evt_prov_1",
            type=EventType.USER_INPUT,
            sender_id="caller_1",
            recipient_id="agent-provenance",
            payload={"message": "Who answered this?"},
        )
    )

    # --- Transport 1: the in-memory fastpath -------------------------------------
    reply = await sub_a.get()
    assert reply.type == EventType.AGENT_REPLY

    # Typed, on the envelope, read without decoding anything.
    provenance = reply.provenance
    assert isinstance(provenance, Provenance)
    assert provenance == served
    assert provenance.path is ExecutionPath.FAILOVER
    assert provenance.served_by.provider == "secondary_llm"
    assert provenance.degraded is True
    assert provenance.attempts[0].error_class == "RateLimitError"

    # The interim `payload["provenance"]` convention is retired, not merely superseded.
    assert "provenance" not in reply.payload

    # Zero-copy fan-out: the second subscriber holds the very same frozen envelope, so
    # the fastpath carries the `Provenance` object itself rather than a copy of it.
    reply_b = await sub_b.get()
    assert reply_b is reply
    assert reply_b.provenance is provenance

    # --- Transport 2: the A2A JSON wire ------------------------------------------
    wire_json = agent_event_to_wire_json(reply)
    # `degraded` is derived, so it is not put on the wire for a peer to trust.
    assert "degraded" not in wire_json

    restored = agent_event_from_wire_json(wire_json)
    restored_provenance = restored.provenance
    assert isinstance(restored_provenance, Provenance)
    assert restored_provenance == served
    assert restored_provenance.path is ExecutionPath.FAILOVER
    assert isinstance(restored_provenance.attempts, tuple)
    assert restored_provenance.attempts[0].status_code == 429
    # Recomputed by the receiving model from `requested` != `served_by`.
    assert restored_provenance.degraded is True


def _llm_returning(provenance: Provenance | None) -> MagicMock:
    """A connector whose `ModelResponse` states exactly `provenance` — `None` included."""
    llm = MagicMock(spec=LLMProviderProtocol)
    llm.generate = AsyncMock(
        return_value=ModelResponse(
            content="an answer nobody claims",
            tool_calls=(),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(provider="mock", input_tokens=1, output_tokens=1),
            provenance=provenance,
        )
    )
    return llm


@pytest.mark.asyncio
async def test_connector_provenance_is_propagated_not_defaulted() -> None:
    """A connector that states nothing yields `TurnResult.provenance is None` (P6).

    `execute_turn` used to substitute `Provenance.primary("agent.core", "BaseAgent")`
    for a missing connector attribution. That named `BaseAgent` as the server of a value
    an LLM produced, turning "not stated" into a positively asserted clean primary — the
    default-masquerading-as-a-real-answer P6 forbids. Absence is now carried, not
    repaired.
    """
    agent = BaseAgent(
        config=_sample_config("agent-propagates"),
        llm=_llm_returning(None),
    )

    result = await agent.execute_turn("who answered this?")

    assert result.is_completed is True
    assert result.provenance is None


@pytest.mark.asyncio
async def test_unattributable_turn_result_is_never_published() -> None:
    """P6 producer-side: `require_provenance` stops an unattributable reply at the publisher.

    Reached through a real connector rather than by patching `execute_turn`: the guard
    has to be on a path production traffic can take, or it is decoration.
    """
    config = _sample_config("agent-no-provenance")
    bus = EventBus()
    agent = BaseAgent(config=config, bus=bus, llm=_llm_returning(None))
    sub = bus.subscribe("session.*")

    with pytest.raises(MissingProvenanceError):
        await agent.process_event(
            AgentEvent(
                event_id="evt_prov_2",
                type=EventType.USER_INPUT,
                sender_id="caller_1",
                recipient_id="agent-no-provenance",
                payload={"message": "hello"},
            )
        )

    assert sub.empty()


@pytest.mark.asyncio
async def test_event_loop_survives_a_failing_event_and_keeps_consuming() -> None:
    """P1: one unhandleable event must not kill the reactive loop (and P6: not silently).

    `process_event` raising `MissingProvenanceError` inside `_event_loop`'s task used to
    leave `_running` True and the state reporting `IDLE` while every later event was
    dropped forever, surfacing only as a GC-time "Task exception was never retrieved".
    """
    config = _sample_config("agent-loop-survives")
    bus = EventBus()
    await bus.start()
    agent = BaseAgent(config=config, bus=bus, llm=_llm_returning(None))
    await agent.start()
    caller = bus.register_publisher(sender_id="caller_1", source=EventSource.USER)
    replies = bus.subscribe(f"session.sess_{config.agent_id}")

    try:
        await caller.publish(
            AgentEvent(
                type=EventType.USER_INPUT,
                recipient_id=agent.agent_id,
                topic=f"agent.{agent.agent_id}",
                payload={"message": "first, unattributable"},
            )
        )
        await bus.wait_until_idle()
        await asyncio.sleep(0)

        # The failure was absorbed, recorded, and did not kill the task.
        assert agent._loop_task is not None  # pyright: ignore[reportPrivateUsage]
        assert not agent._loop_task.done()  # pyright: ignore[reportPrivateUsage]
        assert [type(e) for e in agent.processing_errors] == [MissingProvenanceError]
        assert agent.state is AgentState.IDLE
        assert replies.empty()

        # ...and the loop still answers the next event, this time attributable.
        agent._llm = _llm_returning(Provenance.primary("mock", "m-1"))  # pyright: ignore[reportPrivateUsage]
        await caller.publish(
            AgentEvent(
                type=EventType.USER_INPUT,
                recipient_id=agent.agent_id,
                topic=f"agent.{agent.agent_id}",
                payload={"message": "second, attributable"},
            )
        )
        await bus.wait_until_idle()
        await asyncio.sleep(0)

        reply = await asyncio.wait_for(replies.get(), timeout=1.0)
        assert reply.type is EventType.AGENT_REPLY
        assert reply.provenance == Provenance.primary("mock", "m-1")
        assert len(agent.processing_errors) == 1
    finally:
        await agent.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_base_agent_failover_emits_telemetry_span_and_correlates_span_id() -> None:
    """BaseAgent.execute_turn natively emits failover.event span and correlates span_id in Provenance (P6, P8, #175)."""
    tracer = TelemetryTracer()
    config = _sample_config("agent-failover-tracer")
    mock_llm = MagicMock(spec=LLMProviderProtocol)
    mock_llm.generate = AsyncMock(side_effect=RuntimeError("Connection refused to provider"))

    finish_after_tools(mock_llm)

    agent = BaseAgent(config=config, llm=mock_llm, tracer=tracer)
    assert agent.tracer is tracer

    result = await agent.execute_turn("Test failover span emission")

    assert result.is_completed is False
    assert result.error == "Connection refused to provider"
    assert agent.state is AgentState.ERROR

    # Completed telemetry spans check (P6 Check 5)
    spans = tracer.get_completed_spans()
    failover_spans = [s for s in spans if s.name == "failover.event"]
    assert len(failover_spans) == 1
    span = failover_spans[0]
    assert span.status is SpanStatus.ERROR
    assert span.attributes["requested_provider"] == "agent-failover-tracer"
    assert span.attributes["served_provider"] == "agent.core"
    assert span.attributes["served_model"] == "error_handler"
    assert span.attributes["error_class"] == "RuntimeError"

    # Span ID threaded into in-band provenance attempts
    assert result.provenance is not None
    assert result.provenance.path is ExecutionPath.FAILOVER
    assert result.provenance.degraded is True
    assert result.provenance.requested.provider == "agent-failover-tracer"
    assert result.provenance.served_by.provider == "agent.core"
    assert len(result.provenance.attempts) == 1
    attempt = result.provenance.attempts[0]
    assert attempt.span_id == span.span_id
    assert attempt.error_class == "RuntimeError"


@pytest.mark.asyncio
async def test_base_agent_failover_publishes_provider_failover_ordered_before_agent_reply() -> None:
    """BaseAgent.execute_turn natively publishes PROVIDER_FAILOVER notice strictly before AGENT_REPLY (P6, P8, #175)."""
    bus = EventBus()
    tracer = TelemetryTracer()
    config = _sample_config("agent-failover-bus")
    mock_llm = MagicMock(spec=LLMProviderProtocol)
    mock_llm.generate = AsyncMock(side_effect=RuntimeError("Provider outage"))

    agent = BaseAgent(config=config, bus=bus, llm=mock_llm, tracer=tracer)
    all_events = bus.subscribe("*")

    input_event = AgentEvent(
        type=EventType.USER_INPUT,
        sender_id="user_caller",
        recipient_id=agent.agent_id,
        topic=f"agent.{agent.agent_id}",
        payload={"message": "Trigger failover"},
    )

    handled = await agent.process_event(input_event)
    assert handled is True

    evt_failover = await asyncio.wait_for(all_events.get(), timeout=2.0)
    evt_reply = await asyncio.wait_for(all_events.get(), timeout=2.0)

    # 1. Verify Event Types
    assert evt_failover.type is EventType.PROVIDER_FAILOVER
    assert evt_reply.type is EventType.AGENT_REPLY

    # 2. Strict total order: PROVIDER_FAILOVER strictly ordered before AGENT_REPLY
    assert evt_failover < evt_reply
    assert evt_failover.sequence < evt_reply.sequence
    assert evt_failover.priority <= evt_reply.priority

    # 3. Payload and Provenance Truth
    assert evt_failover.payload["requested_provider"] == "agent-failover-bus"
    assert evt_failover.payload["served_provider"] == "agent.core"
    assert evt_failover.payload["error_class"] == "RuntimeError"
    assert evt_failover.provenance is not None
    assert evt_failover.provenance.path is ExecutionPath.FAILOVER
    assert evt_failover.provenance.degraded is True

    # 4. Span ID correlation
    spans = tracer.get_completed_spans()
    failover_spans = [s for s in spans if s.name == "failover.event"]
    assert len(failover_spans) == 1
    assert evt_failover.provenance.attempts[0].span_id == failover_spans[0].span_id
    assert evt_reply.provenance is not None
    assert evt_reply.provenance.attempts[0].span_id == failover_spans[0].span_id


def test_base_agent_should_compact_session_uses_llm_predicate() -> None:
    """BaseAgent._should_compact_session delegates trigger decision strictly to ContextCompactor.should_compact_at (P5, #226)."""
    config = AgentConfig(
        agent_id="agent-compact-predicate",
        name="Compactor Tester",
        llm_config=AgentLLMConfig(
            model_name="mock-model",
            auto_compact=True,
            compaction_threshold_tokens=500,
        ),
    )
    agent = BaseAgent(config=config)

    # Empty messages
    assert agent._should_compact_session("sess_1", []) is False  # pyright: ignore[reportPrivateUsage]

    # Short messages under 500 tokens
    short_messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="System instructions"),
        ChatMessage(role=MessageRole.USER, content="Hello"),
    ]
    assert agent._should_compact_session("sess_1", short_messages) is False  # pyright: ignore[reportPrivateUsage]

    # Long messages exceeding 500 tokens
    long_messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="System instructions"),
        ChatMessage(role=MessageRole.USER, content="A" * 2400),  # ~600 tokens
    ]
    assert agent._should_compact_session("sess_1", long_messages) is True  # pyright: ignore[reportPrivateUsage]

    # auto_compact=False disables compaction predicate
    disabled_config = AgentConfig(
        agent_id="agent-compact-disabled",
        name="Compactor Disabled Tester",
        llm_config=AgentLLMConfig(
            model_name="mock-model",
            auto_compact=False,
            compaction_threshold_tokens=500,
        ),
    )
    disabled_agent = BaseAgent(config=disabled_config)
    assert disabled_agent._should_compact_session("sess_1", long_messages) is False  # pyright: ignore[reportPrivateUsage]

    # compaction_threshold_tokens <= 0 disables compaction predicate
    zero_config = AgentConfig(
        agent_id="agent-compact-zero",
        name="Compactor Zero Tester",
        llm_config=AgentLLMConfig(
            model_name="mock-model",
            auto_compact=True,
            compaction_threshold_tokens=0,
        ),
    )
    zero_agent = BaseAgent(config=zero_config)
    assert zero_agent._should_compact_session("sess_1", long_messages) is False  # pyright: ignore[reportPrivateUsage]

    # Dynamic model context window awareness (70% threshold):
    # Gemini Flash (1,000,000 tokens): 70% is 700,000 tokens, so ~600 tokens does NOT trigger compaction.
    gemini_config = AgentConfig(
        agent_id="agent-gemini",
        name="Gemini Agent",
        llm_config=AgentLLMConfig(
            model_name="gemini-1.5-flash",
            auto_compact=True,
        ),
    )
    gemini_agent = BaseAgent(config=gemini_config)
    assert gemini_agent._should_compact_session("sess_1", long_messages) is False  # pyright: ignore[reportPrivateUsage]

    # Explicit context_limit configured (e.g. 800 tokens -> 70% is 560 tokens):
    # ~600 tokens exceeds 560 tokens, so it triggers compaction.
    small_context_config = AgentConfig(
        agent_id="agent-small-ctx",
        name="Small Context Agent",
        llm_config=AgentLLMConfig(
            model_name="custom-model",
            context_limit=800,
            auto_compact=True,
        ),
    )
    small_agent = BaseAgent(config=small_context_config)
    assert small_agent._should_compact_session("sess_1", long_messages) is True  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_compact_session_records_absorbed_publish_failure(tmp_path: Path) -> None:
    """A committed compaction whose CONTEXT_COMPACTED publish fails absorbs the error into processing_errors (P6, #227)."""
    bus = EventBus()
    await bus.start()
    config = _sample_config("agent-compaction-fail")
    agent = BaseAgent(config=config, bus=bus)

    # Populate dialogue history so compaction has messages to compress
    messages = [ChatMessage(role=MessageRole.SYSTEM, content="System prompt")]
    for i in range(10):
        messages.append(ChatMessage(role=MessageRole.USER, content=f"q{i} " + "x" * 200))
        messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=f"a{i} " + "y" * 200))
    agent.load_history(messages, turn_counter=10)
    messages_before = len(agent.history)

    # Ensure publisher is registered and mock publish to fail with EventBusError
    assert agent._publisher is not None  # pyright: ignore[reportPrivateUsage]
    agent._publisher.publish = AsyncMock(side_effect=EventBusError("bus is full"))  # pyright: ignore[reportPrivateUsage]

    # 1. Assert agent.compact_session() completes and returns the valid CompactionResult
    result: CompactionResult = await agent.compact_session(reason="manual_test")
    assert isinstance(result, CompactionResult)
    assert result.session_id == f"sess_{agent.agent_id}"
    assert result.reason == "manual_test"
    assert result.messages_before == messages_before
    assert result.messages_after < messages_before
    assert len(agent.history) == result.messages_after

    # 2. Assert that agent.processing_errors contains the recorded EventBusError
    assert len(agent.processing_errors) == 1
    err = agent.processing_errors[0]
    assert isinstance(err, EventBusError)
    assert "bus is full" in str(err)

    # 3. Assert that /api/diagnostics and /api/health report the agent_processing_errors
    mgr = AgentSessionManager(bus=bus)
    mgr._agents[result.session_id] = agent  # pyright: ignore[reportPrivateUsage]
    app = create_ui_app(static_dir=tmp_path, bus=bus, session_manager=mgr)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        health_resp = await client.get("/api/health")
        assert health_resp.status_code == 200
        health_data: dict[str, Any] = health_resp.json()
        assert health_data["status"] == "degraded"
        agent_health_errs = health_data["absorbed_failures"]["agent_processing_errors"]
        assert agent_health_errs["count"] == 1
        assert agent.agent_id in agent_health_errs["agents"]
        assert any("bus is full" in msg for msg in agent_health_errs["agents"][agent.agent_id])

        diag_resp = await client.get("/api/diagnostics")
        assert diag_resp.status_code == 200
        diag_data: dict[str, Any] = diag_resp.json()
        agent_diag_errs = diag_data["absorbed_failures"]["agent_processing_errors"]
        assert agent_diag_errs["count"] == 1
        assert agent.agent_id in agent_diag_errs["agents"]
        assert any("bus is full" in msg for msg in agent_diag_errs["agents"][agent.agent_id])

    await bus.stop()


# --- #1372: a local model is compacted against the window its daemon serves ------------

_OLLAMA_BASE = "http://served-window.test:11434"
_LLAMA = "llama3.2:1b"
#: The table figure the issue reports for `llama3*`, and a served one below it. The
#: connector now always sends `num_ctx` (`DEFAULT_OLLAMA_NUM_CTX` when nothing is
#: configured), so a served figure differs from the sent one only when the daemon clamps
#: it to a smaller trained window -- which the fake daemon models by default here.
_TABLE_WINDOW = 128_000
_SERVED_WINDOW = 8_192


class _FakeOllamaDaemon:
    """Answers `/api/chat` and `/api/ps` the way the daemon does for window purposes.

    A chat request loads the model at the `num_ctx` it carries, clamped to the trained
    window, or at the daemon's own default when it carries none; `/api/ps` then reports
    that. `loaded` may be seeded to model a daemon that already has the model loaded.
    """

    def __init__(self, *, default_ctx: int = 4_096, trained_ctx: int = _SERVED_WINDOW) -> None:
        self.default_ctx = default_ctx
        self.trained_ctx = trained_ctx
        self.loaded: dict[str, int] = {}
        self.chat_bodies: list[dict[str, Any]] = []
        self.ps_reads = 0

    @staticmethod
    def _full_name(model: str) -> str:
        """The daemon loads an untagged name as `:latest` and reports it that way."""
        return model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            self.ps_reads += 1
            models = [{"model": m, "context_length": n} for m, n in self.loaded.items()]
            return httpx.Response(200, json={"models": models})
        body = json.loads(request.content)
        self.chat_bodies.append(body)
        requested = body.get("options", {}).get("num_ctx")
        ctx = self.default_ctx if requested is None else min(requested, self.trained_ctx)
        self.loaded[self._full_name(body["model"])] = ctx
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "message": {"role": "assistant", "content": "ok"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 5,
                "eval_count": 1,
            },
        )


@pytest.fixture
def served_windows(monkeypatch: pytest.MonkeyPatch) -> OllamaContextWindows:
    """A window store of this test's own, in place of the process-wide one."""
    store = OllamaContextWindows()
    monkeypatch.setattr(context_window_module, "OLLAMA_CONTEXT_WINDOWS", store)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_INDEPTH_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_FAST_MODEL", raising=False)
    return store


def _ollama_agent(
    store: OllamaContextWindows,
    daemon: _FakeOllamaDaemon | None = None,
    **llm_config: Any,
) -> BaseAgent:
    daemon = daemon or _FakeOllamaDaemon()
    connector = OllamaConnector(
        base_url=_OLLAMA_BASE,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(daemon.handler)),
        context_windows=store,
    )
    return BaseAgent(
        config=AgentConfig(
            agent_id="local",
            name="Local",
            llm_config=AgentLLMConfig(**llm_config),
        ),
        llm=connector,
    )


def _history_between_the_two_thresholds() -> list[ChatMessage]:
    """About 25,000 tokens: over 70% of the served window, under 70% of the table's."""
    messages = [ChatMessage(role=MessageRole.USER, content="x" * 100_000)]
    tokens = estimate_message_tokens(messages)
    assert int(_SERVED_WINDOW * 0.7) < tokens < int(_TABLE_WINDOW * 0.7)
    return messages


def test_an_ollama_model_is_compacted_against_the_served_window_not_the_table(
    served_windows: OllamaContextWindows,
) -> None:
    """The issue's case: `llama3.2:1b` served at 32,768 while the table says 128,000.

    Between 70% of the one and 70% of the other the daemon cut every turn from the front
    and no compaction ran. The limit is now the served figure, at all three sites that
    read it.

    Killed by: src/uclone_x/llm/context_window.py :: SERVED_WINDOW_PROVIDERS: frozenset[str] = frozenset({"ollama"})
    Becomes: SERVED_WINDOW_PROVIDERS: frozenset[str] = frozenset({"vllm"})
    """
    assert resolve_model_context_limit(_LLAMA) == _TABLE_WINDOW
    served_windows.remember(_OLLAMA_BASE, _LLAMA, _SERVED_WINDOW)
    agent = _ollama_agent(served_windows, model_name=_LLAMA)
    history = _history_between_the_two_thresholds()

    assert agent._context_window() == _SERVED_WINDOW  # pyright: ignore[reportPrivateUsage]
    assert agent._should_compact_session("s", history) is True  # pyright: ignore[reportPrivateUsage]
    request = LLMRequest(messages=tuple(history))
    assert agent._should_compact_session("s", history, request=request) is True  # pyright: ignore[reportPrivateUsage]


def test_an_ollama_window_the_daemon_has_not_reported_is_the_one_sent_not_the_table(
    served_windows: OllamaContextWindows,
) -> None:
    """Before the daemon reports a figure, the window is the `num_ctx` the connector
    sends -- never the table's, which is a claim about another machine and the one that
    let the daemon truncate silently (P6).

    Killed by: src/uclone_x/llm/context_window.py :: sent = configured_tokens if configured_tokens is not None else default_ollama_num_ctx()
    Becomes: sent = configured_tokens if configured_tokens is not None else resolve_model_context_limit(model)
    """
    agent = _ollama_agent(served_windows, model_name=_LLAMA)

    assert agent._context_window() == DEFAULT_OLLAMA_NUM_CTX  # pyright: ignore[reportPrivateUsage]


def test_the_compaction_window_follows_the_daemons_context_length_like_the_request(
    served_windows: OllamaContextWindows, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trigger counts against what the connector sends, and with
    `OLLAMA_CONTEXT_LENGTH` set that is the variable's window, not the default.

    Killed by: src/uclone_x/llm/context_window.py :: sent = configured_tokens if configured_tokens is not None else default_ollama_num_ctx()
    Becomes: sent = configured_tokens if configured_tokens is not None else DEFAULT_OLLAMA_NUM_CTX
    """
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "8192")
    agent = _ollama_agent(served_windows, model_name=_LLAMA)

    assert agent._context_window() == 8_192  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_with_nothing_configured_the_default_window_is_sent_served_and_counted(
    served_windows: OllamaContextWindows,
) -> None:
    """Fresh-machine E2E: a daemon left to choose served 4096 tokens on a 16 GB GPU and a
    turn failed after its first tool result. With no `context_limit`, the default is sent,
    the daemon loads the model at it, and the trigger counts against the same figure.

    Killed by: src/uclone_x/llm/connectors/ollama.py :: options["num_ctx"] = num_ctx
    Becomes: options["num_ctx_unused"] = num_ctx
    """
    daemon = _FakeOllamaDaemon(default_ctx=4_096, trained_ctx=40_960)
    agent = _ollama_agent(served_windows, daemon, model_name="qwen3:8b")

    result = await agent.execute_turn("hi")
    await agent._observe_context_window()  # pyright: ignore[reportPrivateUsage]

    assert result.error is None
    assert [b["options"].get("num_ctx") for b in daemon.chat_bodies] == [DEFAULT_OLLAMA_NUM_CTX]
    assert daemon.loaded == {"qwen3:8b": DEFAULT_OLLAMA_NUM_CTX}
    assert agent._context_window() == DEFAULT_OLLAMA_NUM_CTX  # pyright: ignore[reportPrivateUsage]


def test_the_window_of_an_unnamed_model_is_the_one_the_connector_sends_it_to(
    served_windows: OllamaContextWindows, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A room seat names no model; `OLLAMA_MODEL` does, and its window is the one to use.

    Killed by: src/uclone_x/agent/base.py :: if getattr(self._llm, "provider_name", None) in SERVED_WINDOW_PROVIDERS:
    Becomes: if getattr(self._llm, "provider_name", None) in ():
    """
    monkeypatch.setenv("OLLAMA_MODEL", _LLAMA)
    served_windows.remember(_OLLAMA_BASE, _LLAMA, _SERVED_WINDOW)
    agent = _ollama_agent(served_windows)

    assert agent._context_window() == _SERVED_WINDOW  # pyright: ignore[reportPrivateUsage]


def test_an_explicit_threshold_at_or_over_the_window_gives_way_to_the_window_ratio(
    served_windows: OllamaContextWindows,
) -> None:
    """A 100,000-token threshold against a 32,768 window would never fire before the
    daemon cut the request, so the window's ratio applies instead.

    Killed by: src/uclone_x/agent/base.py :: if threshold == 60_000 or threshold <= 0 or threshold >= window:
    Becomes: if threshold == 60_000 or threshold <= 0:
    """
    served_windows.remember(_OLLAMA_BASE, _LLAMA, _SERVED_WINDOW)
    agent = _ollama_agent(served_windows, model_name=_LLAMA, compaction_threshold_tokens=100_000)
    history = _history_between_the_two_thresholds()

    assert agent._should_compact_session("s", history) is True  # pyright: ignore[reportPrivateUsage]
    request = LLMRequest(messages=tuple(history))
    assert agent._should_compact_session("s", history, request=request) is True  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_a_configured_context_limit_is_sent_as_num_ctx_and_is_the_limit(
    served_windows: OllamaContextWindows,
) -> None:
    """`context_limit` is what the request asks the daemon for and what the trigger
    counts against, so the two agree by construction.

    Killed by: src/uclone_x/agent/base.py :: if (self._config.llm_config.context_limit or 0) > 0
    Becomes: if (self._config.llm_config.context_limit or 0) > 10**9
    """
    daemon = _FakeOllamaDaemon(trained_ctx=131_072)
    # Not `DEFAULT_OLLAMA_NUM_CTX`: a configured limit equal to the default would read the
    # same whether or not the configuration reached the request.
    agent = _ollama_agent(served_windows, daemon, model_name=_LLAMA, context_limit=32_768)

    result = await agent.execute_turn("hi")
    await agent._observe_context_window()  # pyright: ignore[reportPrivateUsage]

    assert result.error is None
    assert [b["options"].get("num_ctx") for b in daemon.chat_bodies] == [32_768]
    assert daemon.loaded == {_LLAMA: 32_768}
    assert agent._context_window() == 32_768  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_a_num_ctx_the_daemon_clamps_is_read_back_and_the_smaller_figure_wins(
    served_windows: OllamaContextWindows,
) -> None:
    """Ollama clamps `num_ctx` to the trained window. The model was already loaded at the
    daemon's default, so the store holds a figure from before the request; the reload
    must be read again, and the clamped figure -- not the configured one -- is the limit.

    Killed by: src/uclone_x/llm/context_window.py :: if served is not None and served < sent:
    Becomes: if served is not None and served < 0:
    Killed by: src/uclone_x/llm/connectors/ollama.py :: if self._windows.get(base, name) is None or name in self._unconfirmed:
    Becomes: if self._windows.get(base, name) is None:
    """
    daemon = _FakeOllamaDaemon(trained_ctx=131_072)
    daemon.loaded[_LLAMA] = _SERVED_WINDOW
    agent = _ollama_agent(served_windows, daemon, model_name=_LLAMA, context_limit=262_144)

    result = await agent.execute_turn("hi")
    await agent._observe_context_window()  # pyright: ignore[reportPrivateUsage]

    assert result.error is None
    assert daemon.chat_bodies[0]["options"]["num_ctx"] == 262_144
    assert daemon.loaded == {_LLAMA: 131_072}
    assert agent._context_window() == 131_072  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_an_untagged_model_finds_the_window_the_daemon_reports_under_latest(
    served_windows: OllamaContextWindows,
) -> None:
    """`llama3.2` is loaded, and listed in `/api/ps`, as `llama3.2:latest`. The window
    must be found under the name as configured, and once found it is not asked for again
    on every check.

    Killed by: src/uclone_x/llm/context_window.py :: return name + ":latest"
    Becomes: return name
    """
    daemon = _FakeOllamaDaemon()
    agent = _ollama_agent(served_windows, daemon, model_name="llama3.2")

    result = await agent.execute_turn("hi")
    await agent._observe_context_window()  # pyright: ignore[reportPrivateUsage]
    reads_once_known = daemon.ps_reads
    await agent._observe_context_window()  # pyright: ignore[reportPrivateUsage]

    assert result.error is None
    assert daemon.loaded == {"llama3.2:latest": _SERVED_WINDOW}
    assert agent._context_window() == _SERVED_WINDOW  # pyright: ignore[reportPrivateUsage]
    assert daemon.ps_reads == reads_once_known


@pytest.mark.asyncio
async def test_an_untagged_model_sent_a_num_ctx_is_read_back_once_and_the_clamp_wins(
    served_windows: OllamaContextWindows,
) -> None:
    """With `context_limit` set, the sent `num_ctx` and the daemon's reading of it must be
    matched under one name, so the clamp is noticed and the probe stops once it is.

    Killed by: src/uclone_x/llm/connectors/ollama.py :: window_key = ollama_model_key(model)
    Becomes: window_key = model
    Killed by: src/uclone_x/llm/connectors/ollama.py :: name = ollama_model_key(resolve_ollama_model(model))
    Becomes: name = resolve_ollama_model(model)
    """
    daemon = _FakeOllamaDaemon(trained_ctx=131_072)
    daemon.loaded["llama3.2:latest"] = _SERVED_WINDOW
    agent = _ollama_agent(served_windows, daemon, model_name="llama3.2", context_limit=262_144)

    result = await agent.execute_turn("hi")
    await agent._observe_context_window()  # pyright: ignore[reportPrivateUsage]
    reads_once_confirmed = daemon.ps_reads
    await agent._observe_context_window()  # pyright: ignore[reportPrivateUsage]

    assert result.error is None
    assert daemon.chat_bodies[0]["options"]["num_ctx"] == 262_144
    assert daemon.loaded == {"llama3.2:latest": 131_072}
    assert agent._context_window() == 131_072  # pyright: ignore[reportPrivateUsage]
    assert daemon.ps_reads == reads_once_confirmed


@pytest.mark.asyncio
async def test_the_agent_reads_the_window_store_its_connector_writes(
    served_windows: OllamaContextWindows,
) -> None:
    """A connector given a store of its own records the daemon's figure there, and the
    compaction limit must be read from that store, not from the process-wide one.

    Killed by: src/uclone_x/agent/base.py :: store=store if isinstance(store, OllamaContextWindows) else None,
    Becomes: store=None,
    """
    own_store = OllamaContextWindows()
    daemon = _FakeOllamaDaemon()
    daemon.loaded[_LLAMA] = _SERVED_WINDOW
    agent = _ollama_agent(own_store, daemon, model_name=_LLAMA)

    await agent._observe_context_window()  # pyright: ignore[reportPrivateUsage]

    assert served_windows.get(_OLLAMA_BASE, _LLAMA) is None
    assert own_store.get(_OLLAMA_BASE, _LLAMA) == _SERVED_WINDOW
    assert agent._context_window() == _SERVED_WINDOW  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_the_compaction_check_reads_the_served_window_before_counting(
    served_windows: OllamaContextWindows,
) -> None:
    """A daemon that already has the model loaded is asked before the first check, so a
    long resumed session is counted against the served window on its first turn.

    Killed by: src/uclone_x/agent/base.py :: await self._observe_context_window()
    Becomes: pass
    """
    daemon = _FakeOllamaDaemon()
    daemon.loaded[_LLAMA] = _SERVED_WINDOW
    agent = _ollama_agent(served_windows, daemon, model_name=_LLAMA)

    await agent._auto_compact_if_needed()  # pyright: ignore[reportPrivateUsage]

    assert daemon.ps_reads == 1
    assert agent._context_window() == _SERVED_WINDOW  # pyright: ignore[reportPrivateUsage]
