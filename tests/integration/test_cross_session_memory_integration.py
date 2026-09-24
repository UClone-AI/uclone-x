"""L2 Integration test for cross-session memory across distinct agent sessions (Issue #476, #479)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.agent.session import SessionStore
from uclone_x.core.provenance import Provenance
from uclone_x.engine.event_bus import AgentEvent, EventBus, EventType
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.memory.store import CrossSessionMemory


class _CapturingLLM(MockLLMConnector):
    """Mock connector that records requests sent to the LLM."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        super().__init__()
        self._responses_list = list(responses)
        self.captured_requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.captured_requests.append(request)
        if self._responses_list:
            return self._responses_list.pop(0)
        return await super().generate(request)


@pytest.mark.asyncio
async def test_cross_session_memory_survives_across_distinct_sessions_and_retraction_removes_it(
    tmp_path: Path,
) -> None:
    """L2 integration test:
    1. Agent in Session 1 records a fact via record_memory_fact tool.
    2. Distinct Session 2 starts; system prompt demonstrably includes the fact recorded in Session 1.
    3. Session 2 retracts the fact via retract_memory_fact tool.
    4. Distinct Session 3 starts; system prompt confirms the retracted fact is omitted.

    Killed by: src/uclone_x/memory/store.py :: self._facts[fact_id] = retracted
    Becomes: pass
    """
    bus = EventBus(maxsize=100)
    await bus.start()

    sessions_dir = tmp_path / "sessions"
    mem_file = tmp_path / "memory_store.json"

    memory = CrossSessionMemory(storage_path=mem_file)
    store = SessionStore(storage_dir=sessions_dir)

    # -------------------------------------------------------------
    # 1. Session 1: record a fact using tool
    # -------------------------------------------------------------
    tc_record = ToolCallRequest(
        id="call_record_fact_1",
        name="record_memory_fact",
        arguments={
            "subject": "ci_pipeline",
            "predicate": "gate_command",
            "object_value": "./ucx test check --fast",
            "confidence": 0.98,
        },
    )
    llm1 = _CapturingLLM(
        responses=[
            ModelResponse(
                content="Recording CI gate command to memory.",
                tool_calls=(tc_record,),
                finish_reason=FinishReason.TOOL_CALLS,
                usage=TokenUsage(provider="mock"),
                provenance=Provenance.primary("mock"),
            ),
            ModelResponse(
                content="Recorded fact successfully.",
                finish_reason=FinishReason.STOP,
                usage=TokenUsage(provider="mock"),
                provenance=Provenance.primary("mock"),
            ),
        ]
    )

    agent1 = BaseAgent(
        config=AgentConfig(
            agent_id="agent_worker_1",
            name="Worker1",
            llm_config=AgentLLMConfig(model_name="mock", auto_compact=False),
        ),
        bus=bus,
        llm=llm1,
        store=store,
        memory=memory,
    )
    await agent1.start()

    sub1 = bus.subscribe({f"session.{agent1.session_id}"}, recipient_id="tester_1")
    try:
        await bus.publish(
            AgentEvent(
                type=EventType.USER_INPUT,
                topic=f"session.{agent1.session_id}",
                sender_id="tester_1",
                recipient_id=agent1.agent_id,
                payload={"message": "Please record our CI gate command."},
            )
        )

        reply1 = None
        for _ in range(50):
            if not sub1.empty():
                evt = await sub1.get()
                if evt.type == EventType.AGENT_REPLY:
                    reply1 = evt
                    break
            await asyncio.sleep(0.02)
        assert reply1 is not None, "Agent 1 failed to reply"

        facts = memory.list_facts(subject="ci_pipeline")
        assert len(facts) == 1
        fact_id = facts[0].fact_id
        assert facts[0].object_value == "./ucx test check --fast"
        assert facts[0].source_session_id == agent1.session_id
    finally:
        await agent1.stop()
        sub1.close()

    # -------------------------------------------------------------
    # 2. Session 2: distinct session inherits memory from Session 1
    # -------------------------------------------------------------
    memory2 = CrossSessionMemory(storage_path=mem_file)
    llm2 = _CapturingLLM(
        responses=[
            ModelResponse(
                content="I see the CI gate command from memory.",
                finish_reason=FinishReason.STOP,
                usage=TokenUsage(provider="mock"),
                provenance=Provenance.primary("mock"),
            )
        ]
    )

    agent2 = BaseAgent(
        config=AgentConfig(
            agent_id="agent_worker_2",
            name="Worker2",
            llm_config=AgentLLMConfig(model_name="mock", auto_compact=False),
        ),
        bus=bus,
        llm=llm2,
        store=store,
        memory=memory2,
    )
    assert agent2.session_id != agent1.session_id
    await agent2.start()

    sub2 = bus.subscribe({f"session.{agent2.session_id}"}, recipient_id="tester_2")
    try:
        await bus.publish(
            AgentEvent(
                type=EventType.USER_INPUT,
                topic=f"session.{agent2.session_id}",
                sender_id="tester_2",
                recipient_id=agent2.agent_id,
                payload={"message": "What is the CI gate command?"},
            )
        )

        reply2 = None
        for _ in range(50):
            if not sub2.empty():
                evt = await sub2.get()
                if evt.type == EventType.AGENT_REPLY:
                    reply2 = evt
                    break
            await asyncio.sleep(0.02)
        assert reply2 is not None

        # Check prompt injected into LLM2
        assert len(llm2.captured_requests) > 0
        req = llm2.captured_requests[0]
        sys_msg = req.messages[0]
        assert sys_msg.role == MessageRole.SYSTEM
        assert "[Cross-Session Memory Facts]" not in (sys_msg.content or "")
        tail = req.messages[-1]
        assert tail.role == MessageRole.USER
        assert tail.content is not None
        assert "[Cross-Session Memory Facts]" in tail.content
        assert "ci_pipeline: gate_command -> ./ucx test check --fast" in tail.content

        # Retract fact in Session 2
        tc_retract = ToolCallRequest(
            id="call_retract_1",
            name="retract_memory_fact",
            arguments={
                "fact_id": fact_id,
                "reason": "Switching to bazel test",
            },
        )
        llm2_retract = _CapturingLLM(
            responses=[
                ModelResponse(
                    content="Retracting old CI command.",
                    tool_calls=(tc_retract,),
                    finish_reason=FinishReason.TOOL_CALLS,
                    usage=TokenUsage(provider="mock"),
                    provenance=Provenance.primary("mock"),
                ),
                ModelResponse(
                    content="Fact retracted.",
                    finish_reason=FinishReason.STOP,
                    usage=TokenUsage(provider="mock"),
                    provenance=Provenance.primary("mock"),
                ),
            ]
        )
        agent2._llm = llm2_retract  # pyright: ignore[reportPrivateUsage]

        await bus.publish(
            AgentEvent(
                type=EventType.USER_INPUT,
                topic=f"session.{agent2.session_id}",
                sender_id="tester_2",
                recipient_id=agent2.agent_id,
                payload={"message": "Retract the CI gate command."},
            )
        )

        reply2_retract = None
        for _ in range(50):
            if not sub2.empty():
                evt = await sub2.get()
                if evt.type == EventType.AGENT_REPLY:
                    reply2_retract = evt
                    break
            await asyncio.sleep(0.02)
        assert reply2_retract is not None

        stored_fact = memory2.get_fact(fact_id)
        assert stored_fact is not None
        assert stored_fact.retracted is True
    finally:
        await agent2.stop()
        sub2.close()

    # -------------------------------------------------------------
    # 3. Session 3: distinct session verifies fact is no longer in prompt
    # -------------------------------------------------------------
    memory3 = CrossSessionMemory(storage_path=mem_file)
    llm3 = _CapturingLLM(
        responses=[
            ModelResponse(
                content="No active CI command found.",
                finish_reason=FinishReason.STOP,
                usage=TokenUsage(provider="mock"),
                provenance=Provenance.primary("mock"),
            )
        ]
    )

    agent3 = BaseAgent(
        config=AgentConfig(
            agent_id="agent_worker_3",
            name="Worker3",
            llm_config=AgentLLMConfig(model_name="mock", auto_compact=False),
        ),
        bus=bus,
        llm=llm3,
        store=store,
        memory=memory3,
    )
    await agent3.start()

    sub3 = bus.subscribe({f"session.{agent3.session_id}"}, recipient_id="tester_3")
    try:
        await bus.publish(
            AgentEvent(
                type=EventType.USER_INPUT,
                topic=f"session.{agent3.session_id}",
                sender_id="tester_3",
                recipient_id=agent3.agent_id,
                payload={"message": "Check memory please."},
            )
        )

        reply3 = None
        for _ in range(50):
            if not sub3.empty():
                evt = await sub3.get()
                if evt.type == EventType.AGENT_REPLY:
                    reply3 = evt
                    break
            await asyncio.sleep(0.02)
        assert reply3 is not None

        assert len(llm3.captured_requests) > 0
        req3 = llm3.captured_requests[0]
        sys_msg3 = req3.messages[0]
        content = sys_msg3.content or ""
        assert "ci_pipeline: gate_command" not in content
    finally:
        await agent3.stop()
        sub3.close()
        await bus.stop()
