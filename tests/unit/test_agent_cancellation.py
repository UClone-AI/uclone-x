"""Unit tests for agent turn cancellation and graceful interruption (Issue #828).

Covers:
- Task cancellation during `BaseAgent.execute_turn` releases the `_turn_lock`.
- Agent state resets to `AgentState.IDLE` upon cancellation.
- A subsequent turn executes successfully without `InvalidStateTransitionError`.
- Durable events record `outcome="cancelled"` and `stop_reason="cancelled"`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest

from uclone_x.agent import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig, AgentState
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    ModelResponse,
    StreamChunk,
    TokenUsage,
)
from uclone_x.llm.protocols import LLMProviderProtocol


class _PausingLLMConnector:
    """LLM connector that pauses on request until cancelled or released."""

    def __init__(self, response_text: str = "Recovered turn response") -> None:
        self.provider_name = "mock"
        self._default_model = "mock-model"
        self.response_text = response_text
        self.started_event = asyncio.Event()
        self.release_event = asyncio.Event()

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.started_event.set()
        await self.release_event.wait()
        service_ref = ServiceRef(provider="mock", model="mock-model")
        return ModelResponse(
            content=self.response_text,
            thinking=None,
            tool_calls=(),
            usage=TokenUsage(
                provider="mock",
                model="mock-model",
                input_tokens=10,
                output_tokens=10,
                total_tokens=20,
            ),
            finish_reason=FinishReason.STOP,
            model_name="mock-model",
            provenance=Provenance(
                path=ExecutionPath.PRIMARY,
                requested=service_ref,
                served_by=service_ref,
            ),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        self.started_event.set()
        await self.release_event.wait()
        yield StreamChunk(
            delta_content=self.response_text,
            delta_thinking=None,
            tool_calls=(),
            usage=None,
            finish_reason=FinishReason.STOP,
        )


def _make_test_agent(agent_id: str, llm: LLMProviderProtocol, bus: EventBus) -> BaseAgent:
    config = AgentConfig(
        agent_id=agent_id,
        name=f"Agent {agent_id}",
        role="Cancellation Tester",
        system_prompt="Test agent prompt",
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )
    return BaseAgent(config=config, bus=bus, llm=llm)


@pytest.mark.asyncio
async def test_execute_turn_cancellation_releases_lock_and_resets_idle() -> None:
    """Test that cancelling an execute_turn task resets state to IDLE and releases _turn_lock."""
    bus = EventBus()
    await bus.start()
    try:
        pausing_llm = _PausingLLMConnector()
        agent = _make_test_agent("cancel-agent-1", pausing_llm, bus)
        await agent.start()

        # Launch turn in background task
        turn_task = asyncio.create_task(agent.execute_turn("Pause please"))

        # Wait until LLM execution starts
        await pausing_llm.started_event.wait()
        assert agent.state in (AgentState.INGESTING, AgentState.REASONING)
        assert agent._turn_lock.locked() is True  # pyright: ignore[reportPrivateUsage]

        # Cancel the task while in flight
        turn_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn_task

        # Verify state is clean and lock is released
        assert agent.state == AgentState.IDLE
        assert agent._turn_lock.locked() is False  # pyright: ignore[reportPrivateUsage]

        # Verify subsequent turn executes normally without InvalidStateTransitionError
        normal_llm = _PausingLLMConnector(response_text="Clean second turn")
        normal_llm.release_event.set()
        agent._llm = normal_llm  # pyright: ignore[reportPrivateUsage]

        res = await agent.execute_turn("Next prompt")
        assert res.is_completed is True
        assert res.content == "Clean second turn"
        assert agent.state == AgentState.IDLE
        assert agent._turn_lock.locked() is False  # pyright: ignore[reportPrivateUsage]
    finally:
        await bus.stop()


@pytest.mark.asyncio
async def test_execute_turn_cancellation_records_cancelled_durable_event() -> None:
    """Test that cancelling an execute_turn turn records a TURN_END event with outcome=cancelled.

    Killed by: src/uclone_x/agent/base.py :: outcome = "cancelled"
    Becomes: outcome = "completed"
    """
    bus = EventBus()
    await bus.start()
    try:
        pausing_llm = _PausingLLMConnector()
        agent = _make_test_agent("cancel-agent-durable", pausing_llm, bus)
        await agent.start()

        turn_task = asyncio.create_task(agent.execute_turn("Pause durable"))
        await pausing_llm.started_event.wait()
        turn_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await turn_task

        pending_events = agent._pending_durable_events  # pyright: ignore[reportPrivateUsage]
        turn_ends = [e for e in pending_events if e.get("type") == "TURN_END"]
        assert len(turn_ends) == 1
        turn_end = turn_ends[0]
        assert turn_end.get("outcome") == "cancelled"
        assert turn_end.get("stop_reason") == "cancelled"
    finally:
        await bus.stop()


@pytest.mark.asyncio
async def test_ollama_connector_cancellation_propagates() -> None:
    """Test that cancelling an Ollama connector call cleanly raises CancelledError and closes client."""
    from uclone_x.llm.connectors.ollama import OllamaConnector
    from uclone_x.llm.models import ChatMessage, MessageRole

    started_event = asyncio.Event()

    async def slow_handler(request: httpx.Request) -> httpx.Response:
        started_event.set()
        await asyncio.sleep(10)
        return httpx.Response(200, json={"message": {"content": "Late"}})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(slow_handler))
    connector = OllamaConnector(
        base_url="http://localhost:11434",
        http_client=mock_client,
    )

    req = LLMRequest(
        messages=(ChatMessage(role=MessageRole.USER, content="Hello Ollama"),),
        model="llama3",
    )

    task = asyncio.create_task(connector.generate(req))
    await started_event.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    await mock_client.aclose()
