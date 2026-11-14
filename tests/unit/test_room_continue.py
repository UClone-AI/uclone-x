"""Continuing, in place, a room turn its own usage ceiling stopped (#2431).

Owner amendment 2026-10-30: a runaway turn stops at a step boundary once its own paid spend
passes the person's 10-minute limit, and [Continue] resumes the same turn with no resend
and no new conversation. What these pin, at the room:

* **The stopped turn is not a failure.** Its row carries no `error`, says it stopped at its
  ceiling, and commits: its finished steps stay in the seat.
* **Continue goes on from there.** The seat is asked to continue, not handed the prompt
  again, and the continuation lands as the next row.
* **Only such a turn continues.** Anything else is refused, and nothing runs.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.unit.test_room_retry import (
    ALICE,
    SCOUT,
    FakeAgent,
    FakeResolver,
    ScriptedSelector,
    speak,
)
from uclone_x.agent.models import ToolExecutionRecord, TurnResult
from uclone_x.errors import NothingToContinueError
from uclone_x.room.models import RoomPolicy, RoomState


class _StoppingAgent(FakeAgent):
    """Stops its first turn at its usage ceiling after one tool step; continues to an answer."""

    def __init__(self) -> None:
        super().__init__("scout", reply="unused")
        self.continued = 0
        self.rolled_back = 0

    async def execute_turn(self, prompt: str, *, stream_callback: Any = None, **kwargs: Any) -> Any:
        self.prompts.append(prompt)
        return TurnResult(
            turn_index=len(self.prompts),
            content="",
            tool_executions=(ToolExecutionRecord(tool_name="echo", tool_call_id="call_1"),),
            is_completed=False,
            stop_reason="usage_ceiling",
            provenance=None,
        )

    async def continue_turn(self, *, stream_callback: Any = None, **kwargs: Any) -> Any:
        self.continued += 1
        return TurnResult(turn_index=2, content="Here is the melody.", provenance=None)

    def roll_back_turn(self, checkpoint: Any, *, reason: str) -> int:
        self.rolled_back += 1
        return 0


@pytest.fixture
def stopped(tmp_path: Any) -> Any:
    from uclone_x.room.orchestrator import RoomOrchestrator
    from uclone_x.room.store import RoomStore

    store = RoomStore(tmp_path)
    store.save(
        RoomState(
            room_id="r1",
            participants=(ALICE, SCOUT),
            policy=RoomPolicy(max_agent_turns_per_human_message=1),
        )
    )
    agent = _StoppingAgent()
    orch = RoomOrchestrator(
        store=store,
        selectors=[ScriptedSelector("s", [speak("scout")])],
        resolver=FakeResolver({"scout": agent}),
    )
    return store, orch, agent


@pytest.mark.asyncio
async def test_a_turn_its_ceiling_stopped_lands_as_stopped_not_failed(stopped: Any) -> None:
    """Killed by: src/uclone_x/room/orchestrator.py :: stopped_at_ceiling = result.stop_reason == "usage_ceiling"
    Becomes: stopped_at_ceiling = False
    """
    _, orch, agent = stopped

    state = await orch.post("r1", "alice", "write me a melody")

    row = state.transcript[-1]
    assert row.sender_id == "scout"
    assert row.stopped_at_ceiling is True
    assert row.error is None
    assert agent.rolled_back == 0  # its finished steps stay in the seat


@pytest.mark.asyncio
async def test_continue_resumes_the_stopped_turn_with_no_resend(stopped: Any) -> None:
    """Killed by: src/uclone_x/room/orchestrator.py :: agent.continue_turn(**turn_kwargs)
    Becomes: agent.execute_turn(prompt, **turn_kwargs)
    Killed by: src/uclone_x/room/orchestrator.py :: return await self._run_again(state, speaker, decision, continuing=True)
    Becomes: return state
    """
    store, orch, agent = stopped
    await orch.post("r1", "alice", "write me a melody")

    state = await orch.continue_turn("r1")

    assert agent.continued == 1
    assert len(agent.prompts) == 1  # the prompt was not handed over again
    assert state.transcript[-1].content == "Here is the melody."
    assert state.transcript[-1].error is None
    assert [m.sender_id for m in state.transcript] == ["alice", "scout", "scout"]
    assert store.load("r1").transcript[-2].stopped_at_ceiling is True  # the stopped row stays


@pytest.mark.asyncio
async def test_only_a_turn_its_ceiling_stopped_can_be_continued(stopped: Any) -> None:
    """Killed by: src/uclone_x/room/orchestrator.py :: if stopped is None or not stopped.stopped_at_ceiling:
    Becomes: if stopped is None:
    """
    _, orch, agent = stopped
    await orch.post("r1", "alice", "write me a melody")
    await orch.continue_turn("r1")

    with pytest.raises(NothingToContinueError):
        await orch.continue_turn("r1")
    assert agent.continued == 1
