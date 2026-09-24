from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, cast

import pytest
from pydantic import BaseModel, Field

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.agent.session import SessionState
from uclone_x.core.log_writer import LogWriterProtocol
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


class DummyLogWriter(LogWriterProtocol):
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def write_entry(self, entry: str | Mapping[str, Any]) -> str:
        if isinstance(entry, Mapping):
            self.entries.append(dict(entry))
        else:
            self.entries.append({"raw": entry})
        return ""

    def write_line(self, line: str) -> str:
        return ""


class DummySessionStore:
    def __init__(self, writer: DummyLogWriter) -> None:
        self.writer = writer
        self.states: list[SessionState] = []

    def save(
        self, state: SessionState, pending_events: Sequence[Any] | None = None
    ) -> SessionState:
        self.states.append(state)
        if pending_events:
            for ev in pending_events:
                if isinstance(ev, Mapping):
                    mapping_entry = cast("Mapping[str, Any]", ev)
                    self.writer.write_entry(mapping_entry)
        return state

    def load(self, session_id: str) -> SessionState | None:
        return None

    def delete(self, session_id: str, artifacts_dir: Any = None) -> bool:
        return True

    def list_session_ids(self) -> tuple[str, ...]:
        return ()

    def clear_event_log(self, session_id: str) -> None:
        return None

    def save_context_body(self, session_id: str, digest: str, body: str) -> None:
        return None

    def load_context_body(self, session_id: str, digest: str) -> str | None:
        return None


class DummyLLM(BaseLLMConnector):
    def __init__(self, responses: list[ModelResponse]) -> None:
        super().__init__()
        self.responses = responses
        self.idx = 0
        self.calls: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return "dummy"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.calls.append(request)
        resp = self.responses[self.idx]
        self.idx += 1
        return resp

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(delta_content="test")


class DummyToolParams(BaseModel):
    x: int = Field(default=0)


class DummyEchoTool(BaseTool[DummyToolParams]):
    name = "my_tool"
    description = "Echo tool"

    def run(self, params: DummyToolParams, context: ToolContext) -> dict[str, Any]:
        return {"result": params.x + 1}


@pytest.mark.asyncio
async def test_durable_events_pairing() -> None:
    """Emission preserves assistant/tool pairing and records requests and attempts.

    Killed by: src/uclone_x/agent/base.py :: self._pending_durable_events.extend(durable_events)
    """
    writer = DummyLogWriter()
    store = DummySessionStore(writer)

    prov = Provenance(
        path=ExecutionPath.PRIMARY,
        requested=ServiceRef(provider="agent", model="dummy"),
        served_by=ServiceRef(provider="agent", model="dummy"),
        attempts=(),
    )
    usage = TokenUsage(provider="dummy", model="dummy", input_tokens=0, output_tokens=0)

    llm = DummyLLM(
        [
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                content=None,
                tool_calls=(ToolCallRequest(id="tc_1", name="my_tool", arguments={"x": 1}),),
                usage=usage,
                provenance=prov,
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                content="Tool is done",
                tool_calls=(),
                usage=usage,
                provenance=prov,
            ),
        ]
    )

    config = AgentConfig(
        agent_id="agent_1",
        name="Agent",
        llm_config=AgentLLMConfig(model_name="dummy"),
    )

    registry = ToolRegistry()
    registry.register(DummyEchoTool())

    agent = BaseAgent(config=config, llm=llm, tools=registry, store=store)  # type: ignore[reportArgumentType]
    await agent.start()

    res = await agent.execute_turn("hello")
    assert res.is_completed is True
    agent.persist_session()

    # Assert events
    events: list[dict[str, Any]] = writer.entries
    assert len(events) > 0, "No events emitted to the log!"

    types: list[str] = [str(e["type"]) for e in events if "type" in e]
    assert "TURN_START" in types, "TURN_START missing"
    assert "USER_MESSAGE" in types, "USER_MESSAGE missing"
    assert "ASSISTANT_ATTEMPT" in types, "ASSISTANT_ATTEMPT missing"
    assert "REQUEST_CONTEXT" in types, "REQUEST_CONTEXT missing"
    assert "TOOL_CALL" in types, "TOOL_CALL missing"
    assert "TOOL_RESULT" in types, "TOOL_RESULT missing"
    assert "TURN_END" in types, "TURN_END missing"

    # pairing invariant: no recorded span can begin with an unpaired TOOL
    tool_calls: list[dict[str, Any]] = [e for e in events if e.get("type") == "TOOL_CALL"]
    tool_results: list[dict[str, Any]] = [e for e in events if e.get("type") == "TOOL_RESULT"]
    assert len(tool_calls) == len(tool_results)
    assert tool_calls[0]["tool_call_id"] == tool_results[0]["tool_call_id"]
