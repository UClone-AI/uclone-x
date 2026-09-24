"""Unit tests for turn trace reader and step detail reconstruction (#1490).

Tests verify that requests and responses can be reassembled exactly from session
logs and context bodies, handling mixed logs, rolled-back turns, and errors.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig
from uclone_x.agent.request_record import (
    RequestRecordError,
    rebuild_requests,
)
from uclone_x.agent.session import (
    ContextSnapshot,
    SessionState,
    SessionStore,
    content_digest,
)
from uclone_x.agent.turn_trace import (
    NO_REQUEST_RECORDED,
    SUBAGENT_UNREADABLE,
    StepNotFoundError,
    TurnNotLinkedError,
    trace_step,
    trace_turn,
)
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    ModelResponse,
    StreamChunk,
    TokenCountSource,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.log.reader import read_session_log
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


class _DummyParams(BaseModel):
    val: int = Field(default=1)


class _DummyTool(BaseTool[_DummyParams]):
    name = "dummy_tool"
    description = "A dummy tool"

    def run(self, params: _DummyParams, context: ToolContext) -> dict[str, Any]:
        return {"result": f"ran with val={params.val}"}


class _ScriptedTraceLLM(BaseLLMConnector):
    def __init__(self, tool_steps: int = 1) -> None:
        super().__init__()
        self._tool_steps = tool_steps
        self.calls: list[LLMRequest] = []
        self.responses: list[ModelResponse] = []

    @property
    def provider_name(self) -> str:
        return "scripted"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.calls.append(request)
        n = len(self.calls)
        if n <= self._tool_steps:
            resp = ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                content=None,
                tool_calls=(
                    ToolCallRequest(id=f"tc_{n}", name="dummy_tool", arguments={"val": n}),
                ),
                usage=_USAGE_1,
                provenance=_PROV,
                model_name="scripted-v1",
            )
        else:
            resp = ModelResponse(
                finish_reason=FinishReason.STOP,
                content="final answer",
                tool_calls=(),
                usage=_USAGE_2,
                provenance=_PROV,
                model_name="scripted-v1",
            )
        self.responses.append(resp)
        return resp

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(delta_content="unused")


def _make_snapshot(
    *,
    turn_index: int = 1,
    tools_digest: str,
    identity_digest: str,
    slow_context_digest: str,
    turn_context_digest: str,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int | None = 1000,
    system_message: bool = True,
    auto_compact: bool = False,
    compaction_threshold_tokens: int = 50000,
) -> ContextSnapshot:
    return ContextSnapshot(
        turn_index=turn_index,
        tools_digest=tools_digest,
        identity_digest=identity_digest,
        slow_context_digest=slow_context_digest,
        system_message=system_message,
        turn_context_digest=turn_context_digest,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        auto_compact=auto_compact,
        compaction_threshold_tokens=compaction_threshold_tokens,
    )


@pytest.mark.asyncio
async def test_trace_turn_round_trip_real_loop(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/turn_trace.py :: for s in sorted(step_numbers):
    Becomes: for s in []:
    """
    connector = _ScriptedTraceLLM(tool_steps=1)
    registry = ToolRegistry()
    registry.register(_DummyTool())

    store = SessionStore(storage_dir=tmp_path / "sessions")
    config = AgentConfig(
        agent_id="test_agent",
        name="TestAgent",
        llm_config=AgentLLMConfig(model_name="scripted"),
        max_steps=10,
    )
    agent = BaseAgent(config=config, llm=connector, tools=registry, store=store)
    await agent.start()

    turn_id = "turn_uuid_101"
    result = await agent.execute_turn("hello round trip", caller_turn_id=turn_id)
    assert result.error is None
    agent.persist_session()

    log_path = store.event_log_path(agent.session_id)
    assert log_path is not None and log_path.exists()
    events = read_session_log(log_path)
    loaded_state = store.load(agent.session_id)
    assert loaded_state is not None

    trace = trace_turn(store, loaded_state, events, caller_turn_id=turn_id)

    assert trace.session_id == agent.session_id
    assert trace.turn_index == 1
    assert trace.started_at is not None
    assert trace.ended_at is not None
    assert trace.rolled_back is False
    assert len(trace.steps) == 2

    # Step 1: tool call
    step1 = trace.steps[0]
    assert step1.step == 1
    assert step1.request_status == "ok"
    assert step1.verified is True
    assert step1.message_count == len(connector.calls[0].messages)
    assert step1.response_status == "ok"
    assert step1.response is not None
    assert step1.response.content == ""
    assert len(step1.response.tool_calls) == 1
    assert step1.response.tool_calls[0]["name"] == "dummy_tool"
    assert len(step1.tool_results) == 1
    assert step1.tool_results[0].tool_call_id == "tc_1"
    assert json.loads(step1.tool_results[0].output) == {"result": "ran with val=1"}

    # Step 2: final answer
    step2 = trace.steps[1]
    assert step2.step == 2
    assert step2.request_status == "ok"
    assert step2.verified is True
    assert step2.message_count == len(connector.calls[1].messages)
    assert step2.response_status == "ok"
    assert step2.response is not None
    assert step2.response.content == "final answer"
    assert step2.response.tool_calls == []
    assert step2.tool_results == []


def test_rebuild_requests_d1_pre_1472_error(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/request_record.py :: detail="request recorded before request capture (#1421)"
    Becomes: detail="other error"
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    state = SessionState(session_id="sess_d1", agent_id="test_agent")

    # A pre-#1472 REQUEST_CONTEXT event has no "snapshot"
    events = [
        {
            "type": "REQUEST_CONTEXT",
            "step": 1,
            "message_count": 2,
            "kept_message_count": 0,
            "appended_messages": [{"role": "user", "content": "hi"}],
            "digest": "abc",
        }
    ]

    with pytest.raises(RequestRecordError) as exc_info:
        rebuild_requests(store, state, events)
    assert exc_info.value.detail == "request recorded before request capture (#1421)"


def test_rebuild_requests_select(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/request_record.py :: if id(event) not in chosen_ids:
    Becomes: if False:
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")

    identity = "Identity"
    slow = "Slow"
    id_d = content_digest(identity)
    slow_d = content_digest(slow)
    turn_d = content_digest("")
    tools_d = content_digest("[]")

    sid = "sess_select"
    store.save_context_body(sid, id_d, identity)
    store.save_context_body(sid, slow_d, slow)
    store.save_context_body(sid, turn_d, "")
    store.save_context_body(sid, tools_d, "[]")

    snap = _make_snapshot(
        turn_index=1,
        tools_digest=tools_d,
        identity_digest=id_d,
        slow_context_digest=slow_d,
        turn_context_digest=turn_d,
        model="gpt-4o",
    )
    state = SessionState(session_id=sid, agent_id="test_agent", context_snapshots=(snap,))

    events = [
        {
            "type": "REQUEST_CONTEXT",
            "step": 1,
            "snapshot": snap.snapshot_id,
            "request": 1,
            "base_request": None,
            "message_count": 1,
            "kept_message_count": 0,
            "appended_messages": [{"role": "user", "content": "msg 1"}],
            "digest": "d1",
        },
        {
            "type": "REQUEST_CONTEXT",
            "step": 2,
            "snapshot": snap.snapshot_id,
            "request": 2,
            "base_request": 1,
            "message_count": 2,
            "kept_message_count": 1,
            "appended_messages": [{"role": "user", "content": "msg 2"}],
            "digest": "d2",
        },
    ]

    rebuilt = rebuild_requests(store, state, events, select=lambda ev: int(ev.get("step", 0)) == 2)
    assert len(rebuilt) == 1
    assert rebuilt[0].step == 2
    req_messages = rebuilt[0].request.messages
    assert any(m.content == "msg 1" for m in req_messages)
    assert any(m.content == "msg 2" for m in req_messages)


def test_trace_step_detail(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/turn_trace.py :: "tools_count": len(rebuilt.request.tools),
    Becomes: "tools_count": 0,
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")

    identity = "Agent identity"
    slow = "Slow instructions"
    turn_ctx = "Turn context block"
    tools = json.dumps([{"name": "test_tool", "description": "tool desc"}])

    id_d = content_digest(identity)
    slow_d = content_digest(slow)
    turn_d = content_digest(turn_ctx)
    tools_d = content_digest(tools)

    sid = "sess_step_detail"
    store.save_context_body(sid, id_d, identity)
    store.save_context_body(sid, slow_d, slow)
    store.save_context_body(sid, turn_d, turn_ctx)
    store.save_context_body(sid, tools_d, tools)

    snap = _make_snapshot(
        turn_index=1,
        tools_digest=tools_d,
        identity_digest=id_d,
        slow_context_digest=slow_d,
        turn_context_digest=turn_d,
        model="claude-3-5",
        temperature=0.7,
        max_tokens=2048,
    )
    state = SessionState(session_id=sid, agent_id="test_agent", context_snapshots=(snap,))

    events: list[dict[str, Any]] = [
        {"type": "TURN_START", "turn_index": 1, "caller_turn_id": "turn_sd"},
        {
            "type": "REQUEST_CONTEXT",
            "step": 1,
            "snapshot": snap.snapshot_id,
            "request": 1,
            "base_request": None,
            "message_count": 2,
            "kept_message_count": 0,
            "appended_messages": [{"role": "user", "content": "query"}],
            "digest": "digest_val",
        },
        {
            "type": "MODEL_RESPONSE",
            "turn_index": 1,
            "step": 1,
            "content": "step answer",
            "thinking": "thought process",
            "tool_calls": [],
        },
        {"type": "TURN_END", "turn_index": 1, "outcome": "completed"},
    ]

    detail = trace_step(store, state, events, caller_turn_id="turn_sd", step=1)
    assert detail.step == 1
    assert detail.request is not None
    assert detail.request["model"] == "claude-3-5"
    assert detail.layers is not None
    assert detail.layers["identity"] == identity
    assert detail.layers["slow_context"] == slow
    assert detail.layers["turn_context"] == turn_ctx
    assert detail.layers["tools_count"] == 1
    assert detail.layers["system_message"] is True
    assert detail.response is not None
    assert detail.response.content == "step answer"
    assert detail.response.thinking == "thought process"

    # Step not found raises StepNotFoundError
    with pytest.raises(StepNotFoundError):
        trace_step(store, state, events, caller_turn_id="turn_sd", step=99)


# --------------------------------------------------------------------------------------
# Logs written by the real agent loop (review of PR #1512): every event order and field
# shape below is the one `BaseAgent` writes, never one assembled by hand.
# --------------------------------------------------------------------------------------

#: One model call: a reply, a tool call, or a whole response.
_Step = str | ToolCallRequest | ModelResponse


class _Script(BaseLLMConnector):
    """Answers each call with the next step, and keeps every request it was sent."""

    def __init__(self, steps: Sequence[_Step]) -> None:
        super().__init__()
        self._steps = list(steps)
        self.requests: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return "scripted"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        step = self._steps.pop(0)
        if isinstance(step, ModelResponse):
            return step
        if isinstance(step, ToolCallRequest):
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                content=None,
                tool_calls=(step,),
                usage=_USAGE_1,
                provenance=_PROV,
            )
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            content=step,
            tool_calls=(),
            usage=_USAGE_2,
            provenance=_PROV,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(delta_content="unused")


class _NoteParams(BaseModel):
    text: str = Field(default="")


class _NoteTool(BaseTool[_NoteParams]):
    name = "note"
    description = "Takes a note"

    def run(self, params: _NoteParams, context: ToolContext) -> dict[str, Any]:
        return {"noted": params.text}


class _HeldTool(BaseTool[_NoteParams]):
    """Runs until released, so a turn can be stopped while its tool is running."""

    name = "held"
    description = "Waits until released"

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, params: _NoteParams, context: ToolContext) -> dict[str, Any]:
        self.started.set()
        await self.release.wait()
        return {"done": True}


_SID = "sess_trace_real"


def _call(call_id: str, name: str = "note") -> ToolCallRequest:
    return ToolCallRequest(id=call_id, name=name, arguments={"text": "x"})


def _seat(
    llm: _Script,
    store: SessionStore,
    *tools: BaseTool[Any],
    require_evidence: bool = False,
) -> BaseAgent:
    registry = ToolRegistry()
    for tool in tools or (_NoteTool(),):
        registry.register(tool)
    return BaseAgent(
        config=AgentConfig(
            agent_id="scout",
            name="scout",
            llm_config=AgentLLMConfig(model_name="scripted"),
            require_evidence_before_answer=require_evidence,
        ),
        llm=llm,
        tools=registry,
        context=AgentContext(session_id=_SID, agent_id="scout"),
        store=store,
    )


def _log_lines(store: SessionStore) -> tuple[Path, list[str]]:
    path = store.event_log_path(_SID)
    assert path is not None and path.exists()
    return path, path.read_text(encoding="utf-8").splitlines()


def _events(store: SessionStore) -> list[dict[str, Any]]:
    path, _ = _log_lines(store)
    return [dict(e) for e in read_session_log(path)]


def _without(events: list[dict[str, Any]], drop: dict[str, Any]) -> list[dict[str, Any]]:
    """`events` less the one event whose fields include all of `drop`."""
    matches = [e for e in events if all(e.get(k) == v for k, v in drop.items())]
    assert len(matches) == 1, (drop, matches)
    return [e for e in events if e is not matches[0]]


def _request_of(events: list[dict[str, Any]], request: int) -> dict[str, Any]:
    return next(e for e in events if e["type"] == "REQUEST_CONTEXT" and e["request"] == request)


async def _turns(store: SessionStore, llm: _Script, *prompts: str) -> BaseAgent:
    agent = _seat(llm, store)
    await agent.start()
    for n, prompt in enumerate(prompts, start=1):
        assert (await agent.execute_turn(prompt, caller_turn_id=f"t{n}")).is_completed
    agent.persist_session()
    return agent


@pytest.mark.asyncio
async def test_a_stopped_and_rolled_back_turn_reads_its_marks_after_turn_end(
    tmp_path: Path,
) -> None:
    """Blocker 1: the rollback and the dropped step are written after `TURN_END`.

    The turn is stopped while its second tool runs, so the agent drops the unanswered step
    (`TOOL_STEP_DROPPED`, after `TURN_END`) and the caller rolls the turn back
    (`TURN_ROLLED_BACK`, after that). Both belong to the stopped turn, by `turn_index`,
    and to no other.

    Killed by: src/uclone_x/agent/turn_trace.py :: marks = [*turn_events, *own_after_end]
    Becomes: marks = list(turn_events)
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    held = _HeldTool()
    llm = _Script([_call("c1"), _call("c2", "held"), "second answer"])
    agent = _seat(llm, store, _NoteTool(), held)
    await agent.start()
    checkpoint = agent.checkpoint_turn()
    stopped = asyncio.create_task(agent.execute_turn("first", caller_turn_id="t1"))
    await asyncio.wait_for(held.started.wait(), 2.0)
    stopped.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stopped
    agent.roll_back_turn(checkpoint, reason="stopped")
    agent.persist_session()
    assert (await agent.execute_turn("again", caller_turn_id="t2")).is_completed
    agent.persist_session()

    events = _events(store)
    types = [e["type"] for e in events]
    end = types.index("TURN_END")
    assert types[end + 1 : end + 3] == ["TOOL_STEP_DROPPED", "TURN_ROLLED_BACK"]
    state = store.load(_SID)
    assert state is not None

    first = trace_turn(store, state, events, caller_turn_id="t1")
    assert first.turn_index == 1
    assert first.rolled_back is True
    assert [d["unanswered_tool_call_ids"] for d in first.dropped_tool_steps] == [["c2"]]
    assert first.turn_end is not None and first.turn_end["outcome"] == "cancelled"
    assert [s.step for s in first.steps] == [1, 2]
    assert [s.verified for s in first.steps] == [True, True]
    assert [r.tool_call_id for r in first.steps[0].tool_results] == ["c1"]
    assert first.steps[1].tool_results == []

    second = trace_turn(store, state, events, caller_turn_id="t2")
    assert second.rolled_back is False
    assert second.dropped_tool_steps == []
    assert [s.verified for s in second.steps] == [True]
    assert second.steps[0].response is not None
    assert second.steps[0].response.content == "second answer"


@pytest.mark.asyncio
async def test_a_turn_with_no_turn_end_stops_at_the_next_turn_start(tmp_path: Path) -> None:
    """A process that died mid-turn wrote no `TURN_END`; the next turn is not absorbed.

    Killed by: src/uclone_x/agent/turn_trace.py :: if ev.get("type") == "TURN_START":
    Becomes: if False:
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    await _turns(store, _Script([_call("c1"), "one", "two"]), "first", "second")
    state = store.load(_SID)
    assert state is not None
    events = _without(_events(store), {"type": "TURN_END", "turn_index": 1})

    trace = trace_turn(store, state, events, caller_turn_id="t1")

    assert trace.turn_end is None and trace.ended_at is None
    assert [s.step for s in trace.steps] == [1, 2]
    assert [s.response.content if s.response else None for s in trace.steps] == ["", "one"]
    assert [s.message_count for s in trace.steps] == [2, 4]


async def _restarted_log(tmp_path: Path) -> tuple[SessionStore, _Script]:
    """Turn 1 before request capture, a restart, then turn 2 with it: a real mixed log.

    Turn 1's `REQUEST_CONTEXT` is written by the real loop and then stripped of the three
    fields request capture (#1472) added -- exactly the line an older build wrote.
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    await _turns(store, _Script(["one"]), "first")
    after = _Script([_call("c1"), "two"])
    restarted = _seat(after, store)
    assert restarted.hydrate_session() is not None
    await restarted.start()
    assert (await restarted.execute_turn("second", caller_turn_id="t2")).is_completed
    restarted.persist_session()

    path, lines = _log_lines(store)
    rewritten: list[str] = []
    stripped = False
    for line in lines:
        record = json.loads(line)
        if record.get("type") == "REQUEST_CONTEXT" and not stripped:
            # Turn 1's only request. The restart numbers requests from 1 again.
            for field in ("snapshot", "request", "base_request"):
                record.pop(field)
            line = json.dumps(record)
            stripped = True
        rewritten.append(line)
    path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return store, after


@pytest.mark.asyncio
async def test_a_mixed_log_verifies_the_turns_recorded_with_request_capture(
    tmp_path: Path,
) -> None:
    """Should-fix (d): real digests from the real capture path verify after a restart.

    Killed by: src/uclone_x/agent/turn_trace.py :: "verified": rebuilt.verified,
    Becomes: "verified": None,
    """
    store, after = await _restarted_log(tmp_path)
    state = store.load(_SID)
    assert state is not None
    events = _events(store)

    old = trace_turn(store, state, events, caller_turn_id="t1")
    assert [s.request_status for s in old.steps] == ["unavailable"]
    assert old.steps[0].request_reason == "request recorded before request capture (#1421)"
    assert old.steps[0].verified is None
    assert old.steps[0].response is not None and old.steps[0].response.content == "one"

    new = trace_turn(store, state, events, caller_turn_id="t2")
    assert [s.request_status for s in new.steps] == ["ok", "ok"]
    assert [s.verified for s in new.steps] == [True, True]
    assert [s.message_count for s in new.steps] == [len(r.messages) for r in after.requests]


@pytest.mark.asyncio
async def test_step_detail_states_why_a_request_is_missing_and_verifies_one_that_is_not(
    tmp_path: Path,
) -> None:
    """Should-fix (e): `trace_step`'s `request_reason` and `verified` on real records.

    Killed by: src/uclone_x/agent/turn_trace.py :: request_reason = step_errors[step].detail
    Becomes: request_reason = None
    Killed by: src/uclone_x/agent/turn_trace.py :: verified = rebuilt.verified
    Becomes: verified = None
    Killed by: src/uclone_x/agent/turn_trace.py :: request_reason = NO_REQUEST_RECORDED
    Becomes: request_reason = None
    """
    store, after = await _restarted_log(tmp_path)
    state = store.load(_SID)
    assert state is not None
    events = _events(store)

    kept = trace_step(store, state, events, caller_turn_id="t2", step=1)
    assert kept.verified is True
    assert kept.request is not None and kept.request_reason is None
    assert kept.request["messages"] == [
        m.model_dump(mode="json") for m in after.requests[0].messages
    ]
    assert kept.layers is not None and kept.layers["tools_count"] == 1

    old = trace_step(store, state, events, caller_turn_id="t1", step=1)
    assert old.request is None and old.verified is None and old.layers is None
    assert old.request_reason == "request recorded before request capture (#1421)"
    assert old.response is not None and old.response.content == "one"

    unrecorded = _without(events, {"type": "REQUEST_CONTEXT", "request": 2})
    missing = trace_step(store, state, unrecorded, caller_turn_id="t2", step=2)
    assert missing.request is None and missing.request_reason == NO_REQUEST_RECORDED
    assert missing.response is not None and missing.response.content == "two"


@pytest.mark.asyncio
async def test_a_gap_makes_only_the_requests_built_on_it_unavailable(tmp_path: Path) -> None:
    """Blocker 2: a request missing from the log costs the requests folded on it, only.

    A gap before the traced turn: its request is reported with the gap, the trace is
    still returned. A gap after it: request 4 extends the missing request 3 and is
    step 1 of its turn, as the traced request is of turn 2; neither touches turn 2.

    Killed by: src/uclone_x/agent/turn_trace.py :: errors[step] = err
    Becomes: pass
    Also killed by one compound mutant that restores the #1490 defect (a later gap
    aborting the fold), in `src/uclone_x/agent/request_record.py`: drop the truncation
    `requests = requests[: chosen[-1] + 1]`, and make the unselected-request `continue`
    under `if id(event) not in chosen_ids:` raise `broken` when it is set.
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    await _turns(
        store, _Script(["one", "two", "three", "four"]), "first", "second", "third", "fourth"
    )
    state = store.load(_SID)
    assert state is not None
    events = _events(store)
    assert _request_of(events, 2)["kept_message_count"] > 0

    gap_before = _without(events, {"type": "REQUEST_CONTEXT", "request": 1})
    trace = trace_turn(store, state, gap_before, caller_turn_id="t2")
    assert [s.request_status for s in trace.steps] == ["unavailable"]
    assert trace.steps[0].request_reason == "request 2 extends 1, last seen None"
    assert trace.steps[0].response is not None and trace.steps[0].response.content == "two"

    gap_after = _without(events, {"type": "REQUEST_CONTEXT", "request": 3})
    assert _request_of(gap_after, 4)["base_request"] == 3
    trace = trace_turn(store, state, gap_after, caller_turn_id="t2")
    assert [s.verified for s in trace.steps] == [True]


class _Watched(dict[str, Any]):
    """An event that records which of its fields were read."""

    def __init__(self, event: dict[str, Any], reads: list[str]) -> None:
        super().__init__(event)
        self._reads = reads

    def get(self, key: str, default: Any = None) -> Any:
        self._reads.append(key)
        return super().get(key, default)

    def __getitem__(self, key: str) -> Any:
        self._reads.append(key)
        return super().__getitem__(key)


@pytest.mark.asyncio
async def test_the_fold_reads_nothing_after_the_last_selected_request(tmp_path: Path) -> None:
    """Blocker 2: "the fold stops after the last selected request" (§4.4, Revision 2).

    The requests after turn 1's are watched. Finding the requests reads each one's
    `type`, and `select` here reads nothing; the fold must read no other field of them.

    Killed by: src/uclone_x/agent/request_record.py :: requests = requests[: chosen[-1] + 1]
    Becomes: pass
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    await _turns(store, _Script(["one", "two", "three"]), "first", "second", "third")
    state = store.load(_SID)
    assert state is not None
    reads: list[str] = []
    events: list[dict[str, Any]] = [
        _Watched(e, reads) if e["type"] == "REQUEST_CONTEXT" and e["request"] > 1 else e
        for e in _events(store)
    ]
    first = _request_of(events, 1)
    assert sum(isinstance(e, _Watched) for e in events) == 2 and reads == []

    rebuilt = rebuild_requests(store, state, events, select=lambda ev: ev is first)
    assert [r.step for r in rebuilt] == [1] and rebuilt[0].verified is True
    assert [key for key in reads if key != "type"] == []


@pytest.mark.asyncio
async def test_a_request_that_keeps_nothing_rebuilds_past_a_gap(tmp_path: Path) -> None:
    """The first request after a restart restates its conversation, so a gap before it
    does not reach it.

    Killed by: src/uclone_x/agent/request_record.py :: broken = None
    Becomes: pass
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    await _turns(store, _Script(["one", "two"]), "first", "second")
    restarted = _seat(_Script(["three"]), store)
    assert restarted.hydrate_session() is not None
    await restarted.start()
    assert (await restarted.execute_turn("third", caller_turn_id="t3")).is_completed
    restarted.persist_session()
    state = store.load(_SID)
    assert state is not None
    # The restart numbers requests from 1 again: drop the first turn's, keep the third's.
    events = _events(store)
    first_request = next(e for e in events if e["type"] == "REQUEST_CONTEXT")
    events = [e for e in events if e is not first_request]
    restated = [e for e in events if e["type"] == "REQUEST_CONTEXT"][-1]
    assert restated["base_request"] is None and restated["kept_message_count"] == 0

    reasons: list[tuple[int, str | None]] = []
    rebuilt = rebuild_requests(
        store, state, events, on_error=lambda step, err: reasons.append((step, err.detail))
    )

    assert reasons == [(1, "request 2 extends 1, last seen None")]
    assert [(r.step, r.verified) for r in rebuilt] == [(1, True)]


@pytest.mark.asyncio
async def test_nudges_are_read_from_the_turn(tmp_path: Path) -> None:
    """Should-fix (e): the evidence nudge the real loop writes is in the trace.

    Killed by: src/uclone_x/agent/turn_trace.py :: nudges = [dict(ev) for ev in turn_events if ev.get("type") in _NUDGE_TYPES]
    Becomes: nudges = []
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    agent = _seat(_Script(["no", "still no, having checked"]), store, require_evidence=True)
    await agent.start()
    assert (await agent.execute_turn("does it build?", caller_turn_id="t1")).is_completed
    agent.persist_session()
    state = store.load(_SID)
    assert state is not None

    trace = trace_turn(store, state, _events(store), caller_turn_id="t1")

    assert [(n["type"], n["step"]) for n in trace.nudges] == [
        ("EVIDENCE_NUDGE", 1),
        ("EVIDENCE_NUDGE_DECLINED", 2),
    ]
    assert [s.step for s in trace.steps] == [1, 2]


@pytest.mark.asyncio
async def test_helpers_are_read_from_the_canonical_text_of_a_tool_result(
    tmp_path: Path,
) -> None:
    """Should-fix (e): `TOOL_RESULT.output` is the result's canonical text, a string.

    A delegation's result names its helper in that text; one that names a helper and
    does not parse is stated on the trace, not skipped.

    Killed by: src/uclone_x/agent/turn_trace.py :: parsed = json.loads(output)
    Becomes: parsed = output
    Killed by: src/uclone_x/agent/turn_trace.py :: reason = SUBAGENT_UNREADABLE
    Becomes: pass
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    await _turns(store, _Script([_call("c1"), "done"]), "first")
    state = store.load(_SID)
    assert state is not None
    events = _events(store)
    result = next(e for e in events if e["type"] == "TOOL_RESULT")
    assert isinstance(result["output"], str)

    result["output"] = json.dumps(
        {"response": "found it", "subagent_id": "sub_child_1"}, separators=(",", ":")
    )
    trace = trace_turn(store, state, events, caller_turn_id="t1")
    assert trace.subagents == ["sub_child_1"]
    assert trace.subagents_reason is None

    result["output"] = '{"response":"found it","subagent_id":"sub_chi'
    trace = trace_turn(store, state, events, caller_turn_id="t1")
    assert trace.subagents == []
    assert trace.subagents_reason == SUBAGENT_UNREADABLE


@pytest.mark.asyncio
async def test_tool_results_pair_by_call_id_and_by_position_before_1489(
    tmp_path: Path,
) -> None:
    """Should-fix (f): a result goes to the step whose response asked for its call.

    A log from before #1489 has no `MODEL_RESPONSE`, so no step names a call, and the
    results go to the step they follow.

    Killed by: src/uclone_x/agent/turn_trace.py :: target = call_step.get(call_id, current_step if current_step is not None else 1)
    Becomes: target = 1
    """
    both = ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        content=None,
        tool_calls=(_call("c1"), _call("c2")),
        usage=_USAGE_1,
        provenance=_PROV,
    )
    store = SessionStore(storage_dir=tmp_path / "sessions")
    await _turns(store, _Script([_call("c0"), both, "done"]), "first")
    state = store.load(_SID)
    assert state is not None
    events = _events(store)

    def paired(log: list[dict[str, Any]]) -> list[list[str]]:
        trace = trace_turn(store, state, log, caller_turn_id="t1")
        return [[r.tool_call_id for r in s.tool_results] for s in trace.steps]

    assert paired(events) == [["c0"], ["c1", "c2"], []]
    before_1489 = [e for e in events if e["type"] != "MODEL_RESPONSE"]
    assert paired(before_1489) == [["c0"], ["c1", "c2"], []]


def test_trace_turn_not_linked(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/turn_trace.py :: raise TurnNotLinkedError(f"no turn found with caller_turn_id {caller_turn_id!r}")
    Becomes: return events[0], 0, [], []
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    state = SessionState(session_id="sess_links", agent_id="test_agent")
    events = [
        {"type": "TURN_START", "turn_index": 3, "agent_id": "a", "caller_turn_id": "turn_known"},
        {"type": "TURN_END", "turn_index": 3, "outcome": "completed"},
    ]

    with pytest.raises(TurnNotLinkedError):
        trace_turn(store, state, events, caller_turn_id="turn_unknown")
    with pytest.raises(TurnNotLinkedError):
        trace_step(store, state, events, caller_turn_id="turn_unknown", step=1)


def test_trace_turn_synthetic_1000_turns_benchmark(tmp_path: Path) -> None:
    """A 1,000-turn log traces the one turn asked for, and prints how long it took.

    No time bound is asserted: a wall-clock limit fails on a loaded machine and passes on
    a slow implementation run alone. To measure, run
    `PYTHONPATH=src pytest tests/unit/test_turn_trace.py -k 1000_turns -s`.

    Killed by: src/uclone_x/agent/turn_trace.py :: if ev.get("type") == "TURN_START" and ev.get("caller_turn_id") == caller_turn_id:
    Becomes: if ev.get("type") == "TURN_START":
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    sid = "sess_benchmark_1000"

    identity = "Identity string"
    slow = "Slow string"
    id_d = content_digest(identity)
    slow_d = content_digest(slow)
    turn_d = content_digest("")
    tools_d = content_digest("[]")

    store.save_context_body(sid, id_d, identity)
    store.save_context_body(sid, slow_d, slow)
    store.save_context_body(sid, turn_d, "")
    store.save_context_body(sid, tools_d, "[]")

    snap = _make_snapshot(
        turn_index=1,
        tools_digest=tools_d,
        identity_digest=id_d,
        slow_context_digest=slow_d,
        turn_context_digest=turn_d,
        model="gpt-4o",
    )
    state = SessionState(session_id=sid, agent_id="test_agent", context_snapshots=(snap,))

    events: list[dict[str, Any]] = []
    prev_req: int | None = None
    target_turn_id = "target_turn_750"

    for t in range(1, 1001):
        cid = target_turn_id if t == 750 else f"turn_{t}"
        events.append(
            {
                "type": "TURN_START",
                "turn_index": t,
                "caller_turn_id": cid,
                "at": "2026-09-23T12:00:00Z",
            }
        )
        req_num = t
        events.append(
            {
                "type": "REQUEST_CONTEXT",
                "step": 1,
                "snapshot": snap.snapshot_id,
                "request": req_num,
                "base_request": prev_req,
                "message_count": t + 1,
                "kept_message_count": t - 1 if t > 1 else 0,
                "appended_messages": [{"role": "user", "content": f"msg_{t}"}],
                "digest": f"digest_{t}",
            }
        )
        prev_req = req_num
        events.append(
            {
                "type": "MODEL_RESPONSE",
                "turn_index": t,
                "step": 1,
                "content": f"response_{t}",
                "tool_calls": [],
            }
        )
        events.append(
            {
                "type": "TURN_END",
                "turn_index": t,
                "outcome": "completed",
                "at": "2026-09-23T12:00:01Z",
            }
        )

    t0 = time.perf_counter()
    trace = trace_turn(store, state, events, caller_turn_id=target_turn_id)
    duration_ms = (time.perf_counter() - t0) * 1000.0

    assert trace.turn_index == 750
    assert len(trace.steps) == 1
    assert trace.steps[0].request_status == "ok"
    print(f"\nSynthetic 1000-turn trace benchmark: {duration_ms:.2f} ms")
