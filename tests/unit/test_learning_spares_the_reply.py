"""#1851: learning never spends the person's next reply, and never splits a cascade.

**Budget.** Learning's model call is the clone's own and is charged to it (design §3.3 item
4, P5): to the session's ceiling, when one is set, and to the person's paid-model usage
limit. Charged after a turn, it could take what the person's next reply needed, and that
reply was refused. Now learning is skipped when what would remain after it is less than a
reply's worth (option (b) of #1851): the learning call's own worth (the request as
estimated, plus its reply cap) and a reply's (the largest call the session has booked)
must both fit. A skipped lesson is not a failure the person is shown; the reply is what
they asked for.

**Cascade split.** A learner still busy with an earlier cascade's lesson, when the next
cascade starts, used to wake between that cascade's turns and take the lesson queued so far,
so one clone's turns in one cascade went to two extraction calls. Now learning takes a
lesson only while no cascade is running in the room.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.unit import test_knowledge_extractor as K
from tests.unit.test_knowledge_extractor import (
    _EXTRACTION_MARK,  # pyright: ignore[reportPrivateUsage]
    _PROV,  # pyright: ignore[reportPrivateUsage]
    ALICE,
    SCOUT,
    _fact,  # pyright: ignore[reportPrivateUsage]
    _memory,  # pyright: ignore[reportPrivateUsage]
    _Model,  # pyright: ignore[reportPrivateUsage]
    _room,  # pyright: ignore[reportPrivateUsage]
    _seat,  # pyright: ignore[reportPrivateUsage]
    _Seats,  # pyright: ignore[reportPrivateUsage]
)
from uclone_x.agent.base import BaseAgent
from uclone_x.llm.budget import TokenBudgetManager
from uclone_x.llm.models import FinishReason, LLMRequest, ModelResponse, TokenUsage
from uclone_x.llm.usage.gate import UsageGate, gate_if_paid
from uclone_x.llm.usage.limits import UsageLimits
from uclone_x.llm.usage.store import SqliteUsageStore
from uclone_x.room.models import RoomState


@pytest.fixture(autouse=True)
def _sessions_under_tmp(tmp_path: Path) -> None:  # pyright: ignore[reportUnusedFunction]
    """The seats keep their sessions under this test's `tmp_path`, not the working tree."""
    K._SESSIONS[0] = tmp_path / "sessions"  # pyright: ignore[reportPrivateUsage]


REPLY_TOKENS = 400
LEARNING_TOKENS = 2400


class _Costly(_Model):
    """`_Model`, with every call booking tokens: a reply `REPLY_TOKENS`, learning
    `LEARNING_TOKENS`."""

    async def generate(self, request: LLMRequest) -> ModelResponse:
        response = await super().generate(request)
        system = str(request.messages[0].content) if request.messages else ""
        tokens = LEARNING_TOKENS if system.startswith(_EXTRACTION_MARK) else REPLY_TOKENS
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            content=response.content,
            tool_calls=(),
            usage=TokenUsage(
                provider="scripted", model="scripted", input_tokens=tokens, output_tokens=0
            ),
            provenance=_PROV,
        )


class _PaidCostly(_Costly):
    """`_Costly`, on a paid model: every call passes the person's usage limit."""

    @property
    def paid(self) -> bool:
        return True


def _scout_rows(state: RoomState) -> list[tuple[str, str | None]]:
    return [(m.content, m.error) for m in state.transcript if m.sender_id == SCOUT.id]


async def _two_messages(seat: BaseAgent, llm: _Model, tmp_path: Path) -> RoomState:
    orch = _room(tmp_path, _Seats((SCOUT, seat)), SCOUT.id, None, SCOUT.id)
    await orch.post("r1", ALICE.id, "I like jazz.")
    await orch.wait_for_learning("r1")
    await orch.post("r1", ALICE.id, "And tea.")
    await orch.wait_for_learning("r1")
    return orch._require_room("r1")  # pyright: ignore[reportPrivateUsage]


class TestTheNextReplyIsNeverRefusedForLearning:
    async def test_a_session_ceiling_learning_would_reach_skips_learning(
        self, tmp_path: Path
    ) -> None:
        """Room for the reply and less than learning and a reply: learning waits, the reply
        is answered.

        Killed by: src/uclone_x/agent/base.py :: if not self._room_to_learn(req):
        Becomes: if False:
        Killed by: src/uclone_x/memory/extractor.py :: return replace(outcome, skipped=True)
        Becomes: return outcome
        Killed by: src/uclone_x/room/orchestrator.py :: if outcome.added or outcome.error is not None or outcome.skipped:
        Becomes: if outcome.added or outcome.error is not None:
        """
        llm = _Costly(reply="Noted.", extractions=[[_fact("user", "likes", "jazz")]])
        budget = TokenBudgetManager()
        seat = _seat(SCOUT, llm, _memory(tmp_path))
        seat._budget = budget  # pyright: ignore[reportPrivateUsage]
        budget.configure_session(SCOUT.session_id or "", max_tokens=REPLY_TOKENS * 3)

        state = await _two_messages(seat, llm, tmp_path)

        assert _scout_rows(state) == [("Noted.", None), ("Noted.", None)]
        assert llm.extraction_requests == []
        assert all(m.knowledge_extract_error is None for m in state.transcript)
        # Said on each row, so the person sees why nothing was learned (#2328).
        assert [m.knowledge_skipped for m in state.transcript if m.sender_id == SCOUT.id] == [
            True,
            True,
        ]

    async def test_with_room_for_both_the_clone_still_learns(self, tmp_path: Path) -> None:
        """
        Killed by: src/uclone_x/agent/base.py :: return all(left >= worth + reply for left in remaining)
        Becomes: return False
        """
        llm = _Costly(reply="Noted.", extractions=[[_fact("user", "likes", "jazz")], []])
        memory = _memory(tmp_path)
        budget = TokenBudgetManager()
        seat = _seat(SCOUT, llm, memory)
        seat._budget = budget  # pyright: ignore[reportPrivateUsage]
        budget.configure_session(SCOUT.session_id or "", max_tokens=100_000)

        state = await _two_messages(seat, llm, tmp_path)

        assert _scout_rows(state) == [("Noted.", None), ("Noted.", None)]
        assert len(llm.extraction_requests) == 2
        assert [f.object_value for f in memory.list_facts()] == ["jazz"]

    async def test_a_paid_usage_limit_learning_would_reach_skips_learning(
        self, tmp_path: Path
    ) -> None:
        """The person's own paid-model limit, which every paid call passes, is held the same.

        Killed by: src/uclone_x/llm/usage/gate.py :: return max(0, min(left))
        Becomes: return None
        """
        llm = _PaidCostly(reply="Noted.", extractions=[[_fact("user", "likes", "jazz")]])
        gate = UsageGate(
            store=SqliteUsageStore(tmp_path / "usage.db"),
            limits=lambda: UsageLimits(per_week=REPLY_TOKENS * 3),
            clock=lambda: datetime.now(UTC),
        )
        gate_if_paid(llm, gate)
        seat = _seat(SCOUT, llm, _memory(tmp_path))

        state = await _two_messages(seat, llm, tmp_path)

        assert _scout_rows(state) == [("Noted.", None), ("Noted.", None)]
        assert llm.extraction_requests == []


class TestOneCascadeIsOneLesson:
    async def test_a_learner_still_busy_does_not_split_the_next_cascade(
        self, tmp_path: Path
    ) -> None:
        """The first cascade's extraction is held until the second cascade's first turn
        is in; both of that cascade's turns still go to one extraction call.

        The guard is written twice in `RoomOrchestrator._learn`: it waits until no cascade
        runs, and after waiting for the floor it looks again. Either alone keeps this case
        whole; turning both off (`await self._quiet(room_id).wait()` -> `pass` and
        `if self._cascades.get(room_id):` -> `if False:`, one two-site mutation, which the
        declaration form cannot state) splits it, and this test fails.
        """
        memory = _memory(tmp_path)
        llm = _Model(reply="Noted.", extractions=[[], [], []])
        # Message 1: Scout once. Message 2: Scout twice in one cascade.
        orch = _room(
            tmp_path,
            _Seats((SCOUT, _seat(SCOUT, llm, memory))),
            SCOUT.id,
            None,
            SCOUT.id,
            SCOUT.id,
        )
        llm.hold = asyncio.Event()
        replies = iter(["First.", "Second, one.", "Second, two."])
        hold = llm.hold

        async def reply_in_turn(request: LLMRequest) -> ModelResponse:
            system = str(request.messages[0].content) if request.messages else ""
            if not system.startswith(_EXTRACTION_MARK):
                llm.reply = next(replies)
                if llm.reply == "Second, one.":
                    hold.set()  # the first lesson's call returns mid-cascade
                    for _ in range(20):
                        await asyncio.sleep(0)
            return await _Model.generate(llm, request)

        llm.generate = reply_in_turn  # type: ignore[method-assign]

        await orch.post("r1", ALICE.id, "Hello.")
        learning = asyncio.create_task(orch.wait_for_learning("r1"))
        for _ in range(50):
            if llm.extraction_requests:
                break
            await asyncio.sleep(0)
        assert len(llm.extraction_requests) == 1
        await orch.post("r1", ALICE.id, "Go on.")
        await learning
        await orch.wait_for_learning("r1")

        spans = [str(r.messages[1].content) for r in llm.extraction_requests]
        assert len(spans) == 2, spans
        assert "First." in spans[0]
        assert "Second, one." in spans[1] and "Second, two." in spans[1]


def _booked(*inputs: int, output: int = 100) -> tuple[TokenUsage, ...]:
    return tuple(
        TokenUsage(provider="scripted", model="scripted", input_tokens=i, output_tokens=output)
        for i in inputs
    )


def test_the_next_reply_is_held_to_the_largest_call_plus_one_turns_growth() -> None:
    """The next reply carries the whole history and a new message, so it can outgrow every
    call booked so far; the room kept for it adds the largest step the history took (#2328).

    Killed by: src/uclone_x/agent/base.py :: return largest + growth
    Becomes: return largest
    Killed by: src/uclone_x/agent/base.py :: (max(0, later.input_tokens - earlier.input_tokens) for earlier, later in pairwise(booked)),
    Becomes: (max(0, earlier.input_tokens - later.input_tokens) for earlier, later in pairwise(booked)),
    """
    from uclone_x.agent.base import next_reply_worth

    assert next_reply_worth((), default=777) == 777
    # Requests of 1000, 1300 and 1450 tokens: the largest call is 1550 with its reply, and
    # the largest step between two calls is 300.
    assert next_reply_worth(_booked(1000, 1300, 1450), default=0) == 1550 + 300
    # A smaller call after a larger one is no growth.
    assert next_reply_worth(_booked(2000, 500), default=0) == 2100
