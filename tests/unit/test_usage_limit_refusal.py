"""A usage-limit stop is told as the user's paid-model limit, not the session budget.

`llm-token-gateway.md` §4.3 and §4.6: the gate refuses a paid call with
`UsageLimitReachedError`; the turn names that as `usage_limit`, the room as
`RoomTurnRefusal.USAGE_LIMIT`, and the session ledger no longer refuses on a ceiling
nobody set.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig
from uclone_x.llm import MockLLMConnector
from uclone_x.llm.budget import TokenBudgetManager
from uclone_x.llm.models import LLMRequest, ModelResponse, StreamChunk, TokenUsage
from uclone_x.llm.usage.gate import UsageGate, gate_if_paid
from uclone_x.llm.usage.limits import UsageLimits, UsageWindow, limit_reached_message
from uclone_x.llm.usage.store import MemoryUsageStore, UsageEntry
from uclone_x.room.models import RoomTurnRefusal, turn_refusal

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


def _agent(llm: MockLLMConnector, budget: TokenBudgetManager, session_id: str) -> BaseAgent:
    return BaseAgent(
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

    The streamed case is the room's path: a stream callback makes the turn stream, where the
    gate refuses on the first chunk, inside the agent's stream handler.
    Killed by: src/uclone_x/agent/turn_executor.py :: except UsageLimitReachedError:
    Becomes: except LookupError:
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
