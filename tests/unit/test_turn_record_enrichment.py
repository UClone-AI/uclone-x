"""Unit tests for turn record enrichment: caller_turn_id link, MODEL_RESPONSE, timestamps, usage (#1489).

Design reference: turn inspection design §4.2 (#1489).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig
from uclone_x.agent.session import EVENT_LOG_SUBDIR, SessionStore
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.errors import LLMProviderError
from uclone_x.llm.budget import TokenBudgetManager
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    ModelResponse,
    StreamChunk,
    TokenCountSource,
    TokenUsage,
    ToolCallRequest,
    aggregate_token_usages,
)
from uclone_x.log.reader import TURN_LOOP_EVENT_TYPES, read_session_log
from uclone_x.room.models import (
    Participant,
    ParticipantKind,
    RoomPolicy,
    RoomState,
    SelectionVerdict,
    SpeakerDecision,
    SpeakerRequest,
)
from uclone_x.room.orchestrator import RoomOrchestrator
from uclone_x.room.selectors import MentionSelector
from uclone_x.room.service import RoomService
from uclone_x.room.store import RoomStore
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry

_PROV = Provenance(
    path=ExecutionPath.PRIMARY,
    requested=ServiceRef(provider="scripted", model="scripted"),
    served_by=ServiceRef(provider="scripted", model="scripted"),
    attempts=(),
)
_USAGE_1 = TokenUsage(
    provider="scripted",
    model="scripted-v1",
    input_tokens=10,
    output_tokens=5,
    count_source=TokenCountSource.PROVIDER,
)
_USAGE_2 = TokenUsage(
    provider="scripted",
    model="scripted-v1",
    input_tokens=20,
    output_tokens=8,
    count_source=TokenCountSource.PROVIDER,
)


class _ScriptedLLM(BaseLLMConnector):
    """Calls `dummy` `tool_steps` times, then answers."""

    def __init__(self, tool_steps: int) -> None:
        super().__init__()
        self._tool_steps = tool_steps
        self.calls: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return "scripted"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.calls.append(request)
        n = len(self.calls)
        if n <= self._tool_steps:
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                content=None,
                tool_calls=(
                    ToolCallRequest(id=f"tc_{n}", name="dummy_tool", arguments={"val": n}),
                ),
                usage=_USAGE_1,
                provenance=_PROV,
                model_name="scripted-v1",
            )
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            content="answer done",
            tool_calls=(),
            usage=_USAGE_2,
            provenance=_PROV,
            model_name="scripted-v1",
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(delta_content="unused")


#: What a stream had reported when it stopped: the provider's own count, on a chunk.
_PARTIAL_USAGE = TokenUsage(
    provider="stream_provider",
    model="stream-model",
    input_tokens=100,
    output_tokens=7,
    count_source=TokenCountSource.PROVIDER,
)


class _DummyParams(BaseModel):
    val: int = Field(default=0)


class _DummyTool(BaseTool[_DummyParams]):
    name = "dummy_tool"
    description = "Dummy tool for turn tests"

    def run(self, params: _DummyParams, context: ToolContext) -> dict[str, Any]:
        return {"val": params.val, "echo": "ok"}


def _only_log(store: SessionStore) -> Path:
    logs = sorted((store.storage_dir / EVENT_LOG_SUBDIR).glob("*.jsonl"))
    assert len(logs) == 1, logs
    return logs[0]


class TestTurnRecordEnrichment:
    """Covers #1489 acceptance criteria for agent and room turn recording."""

    @pytest.mark.asyncio
    async def test_two_step_tool_turn_records_caller_turn_id_and_model_responses(
        self, tmp_path: Path
    ) -> None:
        """A two-step tool turn persisted to disk carries caller_turn_id, MODEL_RESPONSE, timestamps.

        Each step's recorded response equals the scripted one: tool-call arguments, usage and
        finish reason exactly, and each tool result's `duration_ms` is the measured duration
        on the turn's own `ToolExecutionRecord` -- not merely a number (#1489 acceptance).

        Killed by: src/uclone_x/agent/base.py :: "arguments": unwrap_immutable(tc.arguments),
        Becomes: "arguments": {},
        Killed by: src/uclone_x/agent/base.py :: "usage": usage.model_dump(mode="json") if usage is not None else None,
        Becomes: "usage": {"x": 1},
        Killed by: src/uclone_x/agent/base.py :: "duration_ms": tr.duration_ms,
        Becomes: "duration_ms": 0.0,
        """
        store = SessionStore(storage_dir=tmp_path / "sessions")
        llm = _ScriptedLLM(tool_steps=2)
        registry = ToolRegistry()
        registry.register(_DummyTool())
        config = AgentConfig(
            agent_id="test_agent",
            name="TestAgent",
            llm_config=AgentLLMConfig(model_name="scripted"),
            max_steps=10,
        )
        agent = BaseAgent(config=config, llm=llm, tools=registry, store=store)
        await agent.start()

        caller_id = "test-caller-turn-uuid-42"
        result = await agent.execute_turn("hello", caller_turn_id=caller_id)
        assert result.is_completed, result.error
        assert result.usage is not None
        # 2 tool steps + 1 final answer step = 3 model calls
        # 2 * 10 + 20 = 40 input tokens, 2 * 5 + 8 = 18 output tokens
        assert result.usage.input_tokens == 40
        assert result.usage.output_tokens == 18
        assert result.usage.total_tokens == 58
        assert result.usage.count_source == TokenCountSource.PROVIDER

        agent.persist_session()
        log_file = _only_log(store)
        events = list(read_session_log(log_file))

        # Check TURN_START
        start_events = [e for e in events if e.get("type") == "TURN_START"]
        assert len(start_events) == 1
        start_evt = start_events[0]
        assert start_evt.get("caller_turn_id") == caller_id
        assert "at" in start_evt
        datetime.fromisoformat(start_evt["at"])  # valid ISO-8601

        # Check MODEL_RESPONSE
        model_responses = [e for e in events if e.get("type") == "MODEL_RESPONSE"]
        assert len(model_responses) == 3

        for i, mr in enumerate(model_responses, start=1):
            assert mr["step"] == i
            assert mr["turn_index"] == 1
            assert mr["started_at"] <= mr["ended_at"]
            datetime.fromisoformat(mr["started_at"])
            datetime.fromisoformat(mr["ended_at"])
            assert mr["error"] is None
            assert mr["model_name"] == "scripted-v1"
            assert mr["streamed"] is False
            assert mr["thinking"] is None

        # Steps 1 and 2 are the scripted tool calls, exactly.
        for n in (1, 2):
            mr = model_responses[n - 1]
            assert mr["content"] == ""
            assert mr["finish_reason"] == "tool_calls"
            assert mr["tool_calls"] == [
                {"id": f"tc_{n}", "name": "dummy_tool", "arguments": {"val": n}}
            ]
            assert mr["usage"] == _USAGE_1.model_dump(mode="json")

        # Step 3 is the scripted final answer, exactly.
        assert model_responses[2]["finish_reason"] == "stop"
        assert model_responses[2]["content"] == "answer done"
        assert model_responses[2]["tool_calls"] == []
        assert model_responses[2]["usage"] == _USAGE_2.model_dump(mode="json")

        # Check TOOL_RESULT
        tool_results = [e for e in events if e.get("type") == "TOOL_RESULT"]
        assert len(tool_results) == 2
        measured = {r.tool_call_id: r.duration_ms for r in result.tool_executions}
        assert set(measured) == {"tc_1", "tc_2"}
        for tr in tool_results:
            assert "at" in tr
            datetime.fromisoformat(tr["at"])
            # The measured duration, not a stand-in: a real call takes a nonzero time.
            assert measured[tr["tool_call_id"]] > 0
            assert tr["duration_ms"] == measured[tr["tool_call_id"]]

        # Check TURN_END
        end_events = [e for e in events if e.get("type") == "TURN_END"]
        assert len(end_events) == 1
        end_evt = end_events[0]
        assert "at" in end_evt
        datetime.fromisoformat(end_evt["at"])

    def test_read_session_log_requires_model_response_registered(self, tmp_path: Path) -> None:
        """read_session_log accepts MODEL_RESPONSE; fails closed if removed from TURN_LOOP_EVENT_TYPES."""
        log_path = tmp_path / "test.jsonl"
        lines = [
            '{"schema": "uclone_x.log.v0", "version": "0.1.0-dev", "session_id": "s1"}\n',
            '{"offset": 1, "session_id": "s1", "type": "TURN_START", "turn_index": 1, "at": "2026-09-23T10:00:00Z"}\n',
            '{"offset": 2, "session_id": "s1", "type": "MODEL_RESPONSE", "turn_index": 1, "step": 1, "started_at": "2026-09-23T10:00:01Z", "ended_at": "2026-09-23T10:00:02Z", "content": "hi", "thinking": null, "tool_calls": [], "finish_reason": "stop", "model_name": "m", "usage": null, "error": null}\n',
            '{"offset": 3, "session_id": "s1", "type": "TURN_END", "turn_index": 1, "outcome": "completed", "steps": 1, "tool_executions": 0, "stop_reason": "model_stopped", "at": "2026-09-23T10:00:03Z"}\n',
        ]
        log_path.write_text("".join(lines), encoding="utf-8")

        # Must succeed with current registry
        events = list(read_session_log(log_path))
        assert len(events) == 3
        assert events[1]["type"] == "MODEL_RESPONSE"

        # If MODEL_RESPONSE is removed from registry, it must fail closed
        assert "MODEL_RESPONSE" in TURN_LOOP_EVENT_TYPES

    @pytest.mark.asyncio
    async def test_streaming_thinking_accumulated(self, tmp_path: Path) -> None:
        """StreamChunk.delta_thinking is accumulated and populated on ModelResponse and MODEL_RESPONSE (D3 fix)."""
        store = SessionStore(storage_dir=tmp_path / "sessions")

        class _ThinkingLLM(BaseLLMConnector):
            @property
            def provider_name(self) -> str:
                return "thinking_provider"

            async def generate(self, request: LLMRequest) -> ModelResponse:
                raise NotImplementedError()

            async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
                yield StreamChunk(delta_thinking="I am ", model="thinking-model")
                yield StreamChunk(delta_thinking="reasoning deeply.", model="thinking-model")
                yield StreamChunk(delta_content="Here is ", model="thinking-model")
                yield StreamChunk(delta_content="the answer.", model="thinking-model")

        llm = _ThinkingLLM()
        config = AgentConfig(
            agent_id="thinker",
            name="Thinker",
            llm_config=AgentLLMConfig(model_name="thinking-model"),
        )
        agent = BaseAgent(config=config, llm=llm, store=store)
        await agent.start()

        received_stream_events: list[tuple[str, dict[str, Any]]] = []

        def callback(evt: str, data: dict[str, Any]) -> None:
            received_stream_events.append((evt, data))

        result = await agent.execute_turn("question", stream_callback=callback)
        assert result.is_completed
        assert result.content == "Here is the answer."

        agent.persist_session()
        log_file = _only_log(store)
        events = list(read_session_log(log_file))

        mr = [e for e in events if e.get("type") == "MODEL_RESPONSE"][0]
        assert mr["thinking"] == "I am reasoning deeply."
        assert mr["content"] == "Here is the answer."
        assert mr["streamed"] is True

    @pytest.mark.asyncio
    async def test_mid_stream_interrupted_error_records_model_response(
        self, tmp_path: Path
    ) -> None:
        """Mid-stream interruption emits MODEL_RESPONSE with error and partial_content before re-raising.

        The usage the budget was charged for the partial stream is the usage the record
        states and the turn's `usage` sums: the three figures agree (P6), and the thinking
        that streamed before the drop is kept.

        Killed by: src/uclone_x/agent/base.py :: step_usages.append(progress.usage)
        Becomes: pass
        Killed by: src/uclone_x/agent/base.py :: progress.usage = self._book_partial_stream(
        Becomes: _ = self._book_partial_stream(
        """
        store = SessionStore(storage_dir=tmp_path / "sessions")

        class _FailingStreamLLM(BaseLLMConnector):
            @property
            def provider_name(self) -> str:
                return "fail_provider"

            async def generate(self, request: LLMRequest) -> ModelResponse:
                raise NotImplementedError()

            async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
                yield StreamChunk(delta_thinking="considering", model="stream-model")
                yield StreamChunk(delta_content="First partial ", model="stream-model")
                yield StreamChunk(
                    delta_content="second partial. ", model="stream-model", usage=_PARTIAL_USAGE
                )
                raise ConnectionResetError("network dropped mid-stream")

        llm = _FailingStreamLLM()
        config = AgentConfig(
            agent_id="interrupted_agent",
            name="InterruptedAgent",
            llm_config=AgentLLMConfig(model_name="stream-model"),
        )
        budget = TokenBudgetManager()
        agent = BaseAgent(
            config=config,
            llm=llm,
            store=store,
            budget=budget,
            context=AgentContext(session_id="sess_interrupted", agent_id="interrupted_agent"),
        )
        await agent.start()

        def callback(evt: str, data: dict[str, Any]) -> None:
            pass

        result = await agent.execute_turn("prompt", stream_callback=callback)
        assert not result.is_completed
        assert result.error is not None and "interrupted" in result.error

        agent.persist_session()
        log_file = _only_log(store)
        events = list(read_session_log(log_file))

        mr_events = [e for e in events if e.get("type") == "MODEL_RESPONSE"]
        assert len(mr_events) == 1
        mr = mr_events[0]
        assert mr["error"] is not None
        assert mr["error"]["type"] == "LLMStreamInterruptedError"
        assert mr["error"]["partial_content"] == "First partial second partial. "
        assert mr["content"] == "First partial second partial. "
        assert mr["thinking"] == "considering"
        assert mr["model_name"] == "stream-model"

        # Charged, recorded and summed: one figure in three places.
        assert budget.get_turn_history("sess_interrupted") == (_PARTIAL_USAGE,)
        assert mr["usage"] == _PARTIAL_USAGE.model_dump(mode="json")
        assert result.usage is not None
        assert (result.usage.input_tokens, result.usage.output_tokens) == (100, 7)
        assert result.usage.total_tokens == 107
        assert result.usage.count_source == TokenCountSource.PROVIDER

    @pytest.mark.asyncio
    async def test_cancellation_preserves_model_response_and_reraises(self, tmp_path: Path) -> None:
        """A cancelled turn re-raises CancelledError and preserves MODEL_RESPONSE in durable events."""
        store = SessionStore(storage_dir=tmp_path / "sessions")

        class _CancellingLLM(BaseLLMConnector):
            @property
            def provider_name(self) -> str:
                return "cancel_provider"

            async def generate(self, request: LLMRequest) -> ModelResponse:
                raise asyncio.CancelledError("task cancelled by caller")

            async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
                yield StreamChunk(delta_content="before cancel")
                raise asyncio.CancelledError("stream cancelled")

        llm = _CancellingLLM()
        config = AgentConfig(
            agent_id="cancelling_agent",
            name="CancellingAgent",
            llm_config=AgentLLMConfig(model_name="cancel-model"),
        )
        agent = BaseAgent(config=config, llm=llm, store=store)
        await agent.start()

        with pytest.raises(asyncio.CancelledError):
            await agent.execute_turn("prompt")

        agent.persist_session()
        log_file = _only_log(store)
        events = list(read_session_log(log_file))

        mr_events = [e for e in events if e.get("type") == "MODEL_RESPONSE"]
        assert len(mr_events) == 1
        mr = mr_events[0]
        assert mr["error"] is not None
        assert mr["error"]["type"] == "CancelledError"

        end_events = [e for e in events if e.get("type") == "TURN_END"]
        assert len(end_events) == 1
        assert end_events[0]["outcome"] == "cancelled"

    @pytest.mark.asyncio
    async def test_stopping_a_streamed_turn_records_what_had_streamed(self, tmp_path: Path) -> None:
        """Stop during a stream: the record keeps how far it got, and cancellation propagates.

        Room turns are always streamed, so this is the ordinary Stop path. The task is
        cancelled while the stream is waiting for its next chunk, after thinking, content
        and a provider count arrived. `MODEL_RESPONSE` keeps all three and names the
        `CancelledError`; the task still ends cancelled, not with a turn error.

        Killed by: src/uclone_x/agent/base.py :: partial_content = progress.content if progress.content_chunks else None
        Becomes: partial_content = None
        Killed by: src/uclone_x/agent/base.py :: "thinking": progress.thinking,
        Becomes: "thinking": None,
        Killed by: src/uclone_x/agent/base.py :: progress.usage = cancelled_usage
        Becomes: progress.usage = None
        """
        store = SessionStore(storage_dir=tmp_path / "sessions")

        class _HangingStreamLLM(BaseLLMConnector):
            @property
            def provider_name(self) -> str:
                return "stream_provider"

            async def generate(self, request: LLMRequest) -> ModelResponse:
                raise NotImplementedError()

            async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
                yield StreamChunk(delta_thinking="weighing it", model="stream-model")
                yield StreamChunk(delta_content="Partial ", model="stream-model")
                yield StreamChunk(
                    delta_content="answer", model="stream-model", usage=_PARTIAL_USAGE
                )
                await asyncio.Event().wait()  # the next chunk never comes
                yield StreamChunk(delta_content="never sent")

        budget = TokenBudgetManager()
        agent = BaseAgent(
            config=AgentConfig(
                agent_id="stopped_agent",
                name="StoppedAgent",
                llm_config=AgentLLMConfig(model_name="stream-model"),
            ),
            llm=_HangingStreamLLM(),
            store=store,
            budget=budget,
            context=AgentContext(session_id="sess_stopped", agent_id="stopped_agent"),
        )
        await agent.start()

        streamed_all = asyncio.Event()

        def callback(evt: str, data: dict[str, Any]) -> None:
            if evt == "token" and data.get("content") == "answer":
                streamed_all.set()

        task = asyncio.create_task(agent.execute_turn("prompt", stream_callback=callback))
        await asyncio.wait_for(streamed_all.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()

        agent.persist_session()
        events = list(read_session_log(_only_log(store)))
        mr_events = [e for e in events if e.get("type") == "MODEL_RESPONSE"]
        assert len(mr_events) == 1
        mr = mr_events[0]
        assert mr["streamed"] is True
        assert mr["content"] == "Partial answer"
        assert mr["thinking"] == "weighing it"
        assert mr["model_name"] == "stream-model"
        assert mr["error"] == {
            "type": "CancelledError",
            "message": "",
            "partial_content": "Partial answer",
        }
        assert mr["usage"] == _PARTIAL_USAGE.model_dump(mode="json")
        assert budget.get_turn_history("sess_stopped") == (_PARTIAL_USAGE,)

        end_events = [e for e in events if e.get("type") == "TURN_END"]
        assert [e["outcome"] for e in end_events] == ["cancelled"]

    @pytest.mark.asyncio
    async def test_a_rolled_back_turn_keeps_its_model_responses(self, tmp_path: Path) -> None:
        """A room turn that fails after a tool step is rolled back, and its responses stay.

        Design §4.2.2 and §5: the failed turn is the one a person most wants to inspect,
        so #1487's rollback takes its messages out of the conversation but must leave its
        `MODEL_RESPONSE` events in the log -- the tool call that ran and the failure that
        ended it.

        Killed by: src/uclone_x/agent/base.py :: self._pending_durable_events.append(
        Becomes: self._pending_durable_events = [e for e in self._pending_durable_events if e.get("type") != "MODEL_RESPONSE"]; self._pending_durable_events.append(
        """
        sessions = SessionStore(tmp_path / "sessions")
        seat = Participant(
            id="scout",
            kind=ParticipantKind.AGENT,
            display_name="Scout",
            session_id="sess_room__r1__scout",
        )
        alice = Participant(id="alice", kind=ParticipantKind.HUMAN, display_name="Alice")

        class _ToolThenFail(BaseLLMConnector):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            @property
            def provider_name(self) -> str:
                return "scripted"

            async def generate(self, request: LLMRequest) -> ModelResponse:
                self.calls += 1
                if self.calls == 1:
                    return ModelResponse(
                        finish_reason=FinishReason.TOOL_CALLS,
                        content=None,
                        tool_calls=(
                            ToolCallRequest(id="c1", name="dummy_tool", arguments={"val": 7}),
                        ),
                        usage=_USAGE_1,
                        provenance=_PROV,
                        model_name="scripted-v1",
                    )
                raise LLMProviderError("provider unavailable (scripted)")

            async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
                yield StreamChunk(delta_content="unused")

        registry = ToolRegistry()
        registry.register(_DummyTool())
        agent = BaseAgent(
            config=AgentConfig(
                agent_id="scout",
                name="scout",
                llm_config=AgentLLMConfig(model_name="scripted"),
            ),
            llm=_ToolThenFail(),
            tools=registry,
            context=AgentContext(session_id="sess_room__r1__scout", agent_id="scout"),
            store=sessions,
        )

        class _Seat:
            async def resolve(self, participant: Participant) -> BaseAgent:
                return agent

        class _ScoutSpeaks:
            @property
            def name(self) -> str:
                return "scout-speaks"

            async def select(self, request: SpeakerRequest) -> SpeakerDecision:
                return SpeakerDecision(
                    verdict=SelectionVerdict.SPEAK,
                    speaker_id="scout",
                    selector=self.name,
                    confidence=1.0,
                )

        rooms = RoomStore(tmp_path / "rooms")
        rooms.save(
            RoomState(
                room_id="r1",
                participants=(alice, seat),
                policy=RoomPolicy(max_agent_turns_per_human_message=1),
            )
        )
        orch = RoomOrchestrator(store=rooms, selectors=[_ScoutSpeaks()], resolver=_Seat())

        state = await orch.post("r1", "alice", "discuss the cache")
        assert state.transcript[-1].error is not None

        agent.persist_session()
        events = list(read_session_log(_only_log(sessions)))
        rolled_back = [e for e in events if e["type"] == "TURN_ROLLED_BACK"]
        assert [e["turn_index"] for e in rolled_back] == [1], [e["type"] for e in events]
        responses = [e for e in events if e["type"] == "MODEL_RESPONSE"]
        assert [(r["turn_index"], r["step"]) for r in responses] == [(1, 1), (1, 2)]
        assert responses[0]["tool_calls"] == [
            {"id": "c1", "name": "dummy_tool", "arguments": {"val": 7}}
        ]
        assert responses[0]["error"] is None
        assert responses[1]["error"]["type"] == "LLMProviderError"

    @pytest.mark.asyncio
    async def test_room_orchestrator_populates_room_message_usage(self, tmp_path: Path) -> None:
        """RoomOrchestrator copies result.usage onto RoomMessage.usage and passes caller_turn_id."""
        store = SessionStore(storage_dir=tmp_path / "sessions")
        room_store = RoomStore(storage_dir=tmp_path / "rooms")
        llm = _ScriptedLLM(tool_steps=1)
        registry = ToolRegistry()
        registry.register(_DummyTool())

        config = AgentConfig(
            agent_id="bot1",
            name="Bot1",
            llm_config=AgentLLMConfig(model_name="scripted"),
        )
        agent = BaseAgent(config=config, llm=llm, tools=registry, store=store)
        await agent.start()

        room_service = RoomService(store=room_store)
        room = room_service.create("Room 1")
        room_service.add_participant(
            room.room_id,
            participant_id="alice",
            kind=ParticipantKind.HUMAN,
            display_name="Alice",
        )
        room_service.add_participant(
            room.room_id,
            participant_id="bot1",
            kind=ParticipantKind.AGENT,
            display_name="Bot 1",
            persona="bot1",
        )

        class _SimpleResolver:
            async def resolve(self, participant: Any) -> Any:
                return agent

        orchestrator = RoomOrchestrator(
            store=room_store,
            selectors=[MentionSelector()],
            resolver=_SimpleResolver(),
        )

        state = await orchestrator.post(room.room_id, "alice", "@bot1 please do something")
        bot_msg = state.transcript[-1]
        assert bot_msg.sender_id == "bot1"
        assert bot_msg.turn_id is not None
        assert bot_msg.usage is not None

        # 1 tool step + 1 stop step = 2 model calls
        # 10 + 20 = 30 input tokens, 5 + 8 = 13 output tokens
        assert bot_msg.usage.input_tokens == 30
        assert bot_msg.usage.output_tokens == 13
        assert bot_msg.usage.total_tokens == 43
        assert bot_msg.usage.count_source == TokenCountSource.PROVIDER

        # Verify caller_turn_id was persisted
        agent.persist_session()
        log_file = _only_log(store)
        events = list(read_session_log(log_file))
        start_evt = [e for e in events if e.get("type") == "TURN_START"][0]
        assert start_evt["caller_turn_id"] == bot_msg.turn_id

    def test_aggregate_token_usages_mixed_sources(self) -> None:
        """aggregate_token_usages labels with ESTIMATE if any step is not PROVIDER."""
        u1 = TokenUsage(
            provider="p",
            model="m",
            input_tokens=10,
            output_tokens=5,
            count_source=TokenCountSource.PROVIDER,
        )
        u2 = TokenUsage(
            provider="p",
            model="m",
            input_tokens=20,
            output_tokens=10,
            count_source=TokenCountSource.ESTIMATE,
        )
        combined = aggregate_token_usages([u1, u2])
        assert combined is not None
        assert combined.input_tokens == 30
        assert combined.output_tokens == 15
        assert combined.total_tokens == 45
        assert combined.count_source == TokenCountSource.ESTIMATE
