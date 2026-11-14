"""A usage-limit stop is told as the user's paid-model limit, not the session budget.

`llm-token-gateway.md` §4.3 and §4.6: the gate refuses a paid call with
`UsageLimitReachedError`; the turn names that as `usage_limit`, the room as
`RoomTurnRefusal.USAGE_LIMIT`, and the session ledger no longer refuses on a ceiling
nobody set.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import BaseModel

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig
from uclone_x.errors import UsageLimitReachedError
from uclone_x.llm import MockLLMConnector
from uclone_x.llm.budget import TokenBudgetManager
from uclone_x.llm.models import (
    ChatMessage,
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.llm.usage.gate import (
    UsageGate,
    enter_admitted_turn,
    gate_if_paid,
    leave_admitted_turn,
)
from uclone_x.llm.usage.limits import UsageLimits, UsageWindow, limit_reached_message
from uclone_x.llm.usage.store import MemoryUsageStore, UsageEntry
from uclone_x.room.models import RoomTurnRefusal, turn_refusal
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


class _PaidMock(MockLLMConnector):
    """A hosted-looking connector that counts the calls that reach it."""

    def __init__(self) -> None:
        super().__init__(responses=["answered"])
        self.reached = 0

    @property
    def paid(self) -> bool:
        return True

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.reached += 1
        return await super().generate(request)

    def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        self.reached += 1
        return super().stream(request)


def _agent(
    llm: MockLLMConnector,
    budget: TokenBudgetManager,
    session_id: str,
    tools: ToolRegistry | None = None,
) -> BaseAgent:
    return BaseAgent(
        tools=tools,
        config=AgentConfig(
            agent_id="spender",
            name="Spender",
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=llm,
        context=AgentContext(session_id=session_id, agent_id="spender"),
        budget=budget,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["generate", "streamed"])
async def test_a_reached_usage_limit_ends_the_turn_as_usage_limit_and_a_room_refusal(
    streamed: bool,
) -> None:
    """Killed by: src/uclone_x/agent/turn_executor.py :: if isinstance(exc, UsageLimitReachedError):
    Becomes: if False:

    The streamed case is the room's path: a stream callback makes the turn stream. Both are
    refused at the turn's start, before the first call (#2431).
    Killed by: src/uclone_x/llm/usage/admission.py :: admit()  # the one check a turn gets
    Becomes: pass
    """
    store = MemoryUsageStore()
    store.add(UsageEntry(at=NOW - timedelta(hours=1), tokens=100, provider="Anthropic"))
    connector = _PaidMock()
    gate = UsageGate(store=store, limits=lambda: UsageLimits(per_5_hours=100), clock=lambda: NOW)
    gated = gate_if_paid(connector, gate)
    assert gated is connector  # wrapped in place, as the factory does it

    async def on_event(_kind: str, _data: object) -> None:
        return None

    agent = _agent(connector, TokenBudgetManager(), "sess_limit")
    if streamed:
        result = await agent.execute_turn("hi", stream_callback=on_event)
    else:
        result = await agent.execute_turn("hi")

    assert connector.reached == 0  # refused before the provider was called
    assert result.is_completed is False
    assert result.stop_reason == "usage_limit"
    assert result.error == limit_reached_message(
        UsageWindow.PER_5_HOURS, NOW + timedelta(hours=4), NOW
    )
    assert "new conversation" not in (result.error or "")
    assert turn_refusal(result.stop_reason) is RoomTurnRefusal.USAGE_LIMIT


def test_the_session_budget_ceiling_is_still_its_own_refusal() -> None:
    assert turn_refusal("budget_exceeded") is RoomTurnRefusal.BUDGET_EXCEEDED
    assert turn_refusal("usage_limit") is not RoomTurnRefusal.BUDGET_EXCEEDED


@pytest.mark.asyncio
async def test_a_session_with_no_explicit_ceiling_never_refuses() -> None:
    """Killed by: src/uclone_x/llm/budget.py :: default_max_tokens: int | None = None,
    Becomes: default_max_tokens: int | None = 1_000_000,
    """
    budget = TokenBudgetManager()
    budget.record_usage(
        "sess_open",
        TokenUsage(provider="mock", model="mock-model", input_tokens=5_000_000, output_tokens=1),
    )

    decision = budget.check_budget("sess_open", provider="mock")
    assert (decision.allowed, decision.remaining_tokens) == (True, None)

    result = await _agent(MockLLMConnector(responses=["fine"]), budget, "sess_open").execute_turn(
        "hi"
    )
    assert result.is_completed is True, result.error
    assert result.stop_reason != "budget_exceeded"

    session = budget.get_summary("sess_open")["session_budget"]
    assert (session["max_tokens"], session["remaining_tokens"], session["budget_used_pct"]) == (
        None,
        None,
        None,
    )
    assert session["total_used_tokens"] >= 5_000_001


def test_an_explicit_ceiling_is_still_enforced_and_summarised() -> None:
    budget = TokenBudgetManager()
    budget.configure_session("capped", max_tokens=100)
    budget.configure_session("open")
    budget.record_usage(
        "capped", TokenUsage(provider="mock", model="m", input_tokens=60, output_tokens=40)
    )

    assert budget.check_budget("capped").allowed is False
    assert budget.check_budget("open").allowed is True
    capped = budget.get_summary("capped")["session_budget"]
    assert (capped["max_tokens"], capped["remaining_tokens"], capped["budget_used_pct"]) == (
        100,
        0,
        100.0,
    )
    # Across sessions, a total ceiling exists only when every session has one.
    assert budget.get_summary()["session_budget"]["max_tokens"] is None


# --- #2431: the limit is checked when a turn starts; an admitted turn finishes ---------------


class _PaidToolMock(_PaidMock):
    """Two steps a turn: a tool call, then the reply, each booked by the gate."""

    def __init__(self) -> None:
        MockLLMConnector.__init__(
            self,
            responses=["calling", "done"],
            tool_calls=[ToolCallRequest(id="call_1", name="echo", arguments={})],
        )
        self.reached = 0


class _NoParams(BaseModel):
    pass


class _EchoTool(BaseTool[_NoParams]):
    name = "echo"
    description = "answers"

    def run(self, params: _NoParams, context: ToolContext) -> dict[str, Any]:
        del params, context
        return {"ok": True}


def _tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_EchoTool())
    return registry


def _just_under(limit: int) -> tuple[MemoryUsageStore, UsageGate]:
    store = MemoryUsageStore()
    store.add(UsageEntry(at=NOW - timedelta(minutes=1), tokens=limit - 1, provider="fakepaid"))
    gate = UsageGate(
        store=store, limits=lambda: UsageLimits(per_10_minutes=limit), clock=lambda: NOW
    )
    return store, gate


@pytest.mark.asyncio
async def test_an_admitted_turn_finishes_past_the_limit_and_books_every_call() -> None:
    """Owner ruling 2026-10-30 (#2431): a turn admitted just under the limit finishes, though
    its first step crosses it; both calls are booked. Stopping it at step two wasted step
    one and left its tools half done.

    Killed by: src/uclone_x/llm/usage/gate.py :: if not _ADMITTED.get():  # a call outside any admitted turn (#2431)
    Becomes: if True:  # a call outside any admitted turn (#2431)
    Killed by: src/uclone_x/agent/base.py :: leave_admitted_turn(before)
    Becomes: pass
    """
    store, gate = _just_under(1000)
    connector = _PaidToolMock()
    gate_if_paid(connector, gate)

    result = await _agent(connector, TokenBudgetManager(), "sess_finish", _tools()).execute_turn(
        "hi"
    )

    assert result.stop_reason != "usage_limit", result.error
    assert result.is_completed is True
    assert connector.reached == 2
    assert len(store.entries_since(NOW - timedelta(hours=1))) == 3  # the row before, and both calls
    # The turn's mark does not outlive it: a call after it, outside any turn, is checked.
    with pytest.raises(UsageLimitReachedError):
        await connector.generate(
            LLMRequest(model="m", messages=(ChatMessage(role=MessageRole.USER, content="x"),))
        )


@pytest.mark.asyncio
async def test_the_next_turn_is_refused_at_its_start() -> None:
    """Once the limit is past, the next turn is refused before any call.

    Killed by: src/uclone_x/agent/turn_executor.py :: enter_admitted_turn(llm, continuing=self._continuing)
    Becomes: enter_admitted_turn(None, continuing=self._continuing)
    """
    _, gate = _just_under(1000)
    connector = _PaidToolMock()
    gate_if_paid(connector, gate)
    agent = _agent(connector, TokenBudgetManager(), "sess_next", _tools())
    await agent.execute_turn("hi")
    reached = connector.reached

    second = await agent.execute_turn("again")

    assert second.stop_reason == "usage_limit"
    assert connector.reached == reached


@pytest.mark.asyncio
async def test_a_call_outside_a_turn_is_still_checked_per_call() -> None:
    """Persona drafts, the room selector and the compactor call outside a turn: each call is
    still checked."""
    store = MemoryUsageStore()
    store.add(UsageEntry(at=NOW - timedelta(minutes=1), tokens=1000, provider="fakepaid"))
    connector = _PaidMock()
    gate_if_paid(
        connector,
        UsageGate(store=store, limits=lambda: UsageLimits(per_10_minutes=1000), clock=lambda: NOW),
    )

    with pytest.raises(UsageLimitReachedError):
        await connector.generate(
            LLMRequest(model="m", messages=(ChatMessage(role=MessageRole.USER, content="x"),))
        )
    assert connector.reached == 0


@pytest.mark.asyncio
async def test_no_call_inside_an_admitted_turn_is_refused() -> None:
    """Inside an admitted turn the gate books and never refuses, streamed or not."""
    store = MemoryUsageStore()
    store.add(UsageEntry(at=NOW - timedelta(minutes=1), tokens=1000, provider="fakepaid"))
    connector = _PaidMock()
    gate_if_paid(
        connector,
        UsageGate(store=store, limits=lambda: UsageLimits(per_10_minutes=1000), clock=lambda: NOW),
    )
    request = LLMRequest(model="m", messages=(ChatMessage(role=MessageRole.USER, content="x"),))

    token = enter_admitted_turn(None)  # a turn with nothing to check at its start
    try:
        await connector.generate(request)
        [chunk async for chunk in connector.stream(request)]
    finally:
        leave_admitted_turn(token)

    # Both answered and booked, past a limit that was already reached. The row before, the
    # generate, and the stream (this mock's stream books its own inner generate too).
    assert len(store.entries_since(NOW - timedelta(hours=1))) >= 3


# --- #2431 amendment: a runaway turn stops at its own ceiling, and is continued in place --------


class _Runaway(MockLLMConnector):
    """A paid model that asks for a tool `steps` times and then answers; every call is 600
    tokens, as the provider counts them."""

    def __init__(self, steps: int) -> None:
        super().__init__(responses=["done"])
        self.steps = steps
        self.requests: list[LLMRequest] = []

    @property
    def paid(self) -> bool:
        return True

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        n = len(self.requests)
        reply = await super().generate(request)
        calls = (
            (ToolCallRequest(id=f"call_{n}", name="echo", arguments={}),) if n <= self.steps else ()
        )
        return reply.model_copy(
            update={
                "content": "" if calls else "done",
                "tool_calls": calls,
                "finish_reason": FinishReason.TOOL_CALLS if calls else FinishReason.STOP,
                "usage": TokenUsage(
                    provider="fakepaid", model="m", input_tokens=500, output_tokens=100
                ),
            }
        )


def _users(request: LLMRequest) -> list[str]:
    return [m.content or "" for m in request.messages if m.role is MessageRole.USER]


@pytest.mark.asyncio
async def test_a_runaway_turn_stops_at_a_step_boundary_once_its_own_spend_passes_the_ceiling() -> (
    None
):
    """Owner amendment 2026-10-30 (#2431): "폭주 상한 방지 한도는 두자". A turn admitted under a
    10-minute limit of 1000 spends 600 a call: the second call takes it past 1000, and it
    stops before the third, never midway through a call. It is a pause, not a failure: no
    `error`, and the two finished steps stand.

    Killed by: src/uclone_x/llm/usage/gate.py :: check_turn_ceiling()  # an admitted turn past its own ceiling stops here
    Becomes: pass  # an admitted turn past its own ceiling stops here
    Killed by: src/uclone_x/llm/usage/gate.py :: book_turn_spend(tokens)  # the running turn's own spend, for its ceiling (#2431)
    Becomes: pass  # the running turn's own spend, for its ceiling (#2431)
    """
    store = MemoryUsageStore()
    connector = _Runaway(steps=5)
    gate_if_paid(
        connector,
        UsageGate(store=store, limits=lambda: UsageLimits(per_10_minutes=1000), clock=lambda: NOW),
    )
    agent = _agent(connector, TokenBudgetManager(), "sess_runaway", _tools())

    result = await agent.execute_turn("hi")

    assert result.stop_reason == "usage_ceiling"
    assert result.error is None
    assert result.is_completed is False
    assert len(connector.requests) == 2  # the third call was never sent
    assert [e.tool_name for e in result.tool_executions] == ["echo", "echo"]
    assert len(store.entries_since(NOW - timedelta(hours=1))) == 2  # both calls booked


@pytest.mark.asyncio
async def test_continue_resumes_the_stopped_turn_without_resending_the_message() -> None:
    """[Continue] goes on from the last finished step: the person's message is in the
    request once, and the turn's tool results are there. It is the same turn, so it is not
    refused at its start although the window is now past its limit, and it may spend one
    more ceiling's worth. A new turn, by contrast, is refused.

    Killed by: src/uclone_x/agent/turn_executor.py :: return not self._continuing and not _repeats_unanswered_prompt(self._history, user_prompt)
    Becomes: return not _repeats_unanswered_prompt(self._history, user_prompt)
    Killed by: src/uclone_x/llm/usage/admission.py :: if callable(admit) and not continuing:
    Becomes: if callable(admit):
    """
    store = MemoryUsageStore()
    connector = _Runaway(steps=2)
    gate_if_paid(
        connector,
        UsageGate(store=store, limits=lambda: UsageLimits(per_10_minutes=1000), clock=lambda: NOW),
    )
    agent = _agent(connector, TokenBudgetManager(), "sess_continue", _tools())
    stopped = await agent.execute_turn("hi")
    assert stopped.stop_reason == "usage_ceiling"

    resumed = await agent.continue_turn()

    assert resumed.error is None, resumed.error
    assert resumed.is_completed is True
    assert resumed.content == "done"
    sent = connector.requests[-1]
    assert _users(sent) == ["hi"]
    assert {m.tool_call_id for m in sent.messages if m.role is MessageRole.TOOL} == {
        "call_1",
        "call_2",
    }
    # A new message is a new turn, and the window it starts in is past its limit.
    assert (await agent.execute_turn("again")).stop_reason == "usage_limit"


@pytest.mark.asyncio
async def test_with_no_limit_a_long_turn_is_never_stopped() -> None:
    """With No limit for the 10-minute window there is no ceiling: a turn of many paid
    steps runs to its answer.

    Killed by: src/uclone_x/llm/usage/gate.py :: return self._limits().per_10_minutes
    Becomes: return self._limits().per_10_minutes or 1000
    """
    store = MemoryUsageStore()
    connector = _Runaway(steps=4)
    gate_if_paid(connector, UsageGate(store=store, limits=UsageLimits, clock=lambda: NOW))

    result = await _agent(connector, TokenBudgetManager(), "sess_open", _tools()).execute_turn("hi")

    assert result.stop_reason != "usage_ceiling"
    assert result.is_completed is True, result.error
    assert len(connector.requests) == 5
