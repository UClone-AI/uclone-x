"""`agent/compaction_driver.py` on its own: compaction behind `BaseAgent` (#1736).

When a session is compacted, and committing and announcing each pass, moved out of
`agent/base.py` unchanged; the behavioural tests that drive it through `BaseAgent` stay
where they were (`test_agent_session_multitenancy.py`, `test_tool_result_ingest.py`,
`test_tool_binder.py`, ...). This file pins what the move itself introduced: the driver
holds no copy of agent state, and the agent methods it calls back are the agent's current
ones, so one replaced on the instance is the one a compaction runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig
from uclone_x.agent.session import CompactionResult
from uclone_x.llm import MockLLMConnector
from uclone_x.llm.compactor import ContextCompactor
from uclone_x.llm.models import ChatMessage, LLMRequest

#: What a stubbed `_compact_session` hands back; never inspected, only compared by identity.
_SENTINEL = cast(CompactionResult, object())


def _agent() -> BaseAgent:
    return BaseAgent(
        config=AgentConfig(agent_id="compaction", name="Compaction"),
        llm=MockLLMConnector(),
    )


@pytest.mark.asyncio
async def test_the_public_compaction_runs_the_agents_own_unguarded_pass() -> None:
    """`compact_session` calls `_compact_session` back through the agent.

    Tests replace `agent._compact_session` on the instance to observe or stub a pass. A
    driver that called its own method directly would bypass the replacement and run a
    real compaction instead.

    Killed by: src/uclone_x/agent/base.py :: compact_session=lambda: self._compact_session,
    Becomes: compact_session=lambda: self._compaction_driver.compact_session_unguarded,
    """
    agent = _agent()
    calls: list[tuple[str, str]] = []

    async def stubbed(
        sid: str,
        reason: str,
        *,
        reader_offered: bool | None = None,
    ) -> CompactionResult:
        calls.append((sid, reason))
        return _SENTINEL

    agent._compact_session = stubbed  # type: ignore[method-assign]

    assert await agent.compact_session() is _SENTINEL
    assert calls == [(agent.session_id, "manual_on_demand")]


@pytest.mark.asyncio
async def test_the_auto_compaction_asks_the_agents_own_threshold_check() -> None:
    """`_auto_compact_if_needed` consults `_should_compact_session` through the agent.

    `test_tool_binder.py` forces a compaction at a turn start by replacing that check on
    the instance. Were the driver to hold its own, the forced pass would never run and
    those tests would pass on a turn that compacted nothing.

    Killed by: src/uclone_x/agent/base.py :: should_compact_session=lambda: self._should_compact_session,
    Becomes: should_compact_session=lambda: self._compaction_driver.should_compact_session,
    """
    agent = _agent()
    reasons: list[str] = []

    def always(
        session_id: str,
        messages: Sequence[ChatMessage],
        *,
        request: LLMRequest | None = None,
    ) -> bool:
        return True

    async def stubbed(
        sid: str,
        reason: str,
        *,
        reader_offered: bool | None = None,
    ) -> CompactionResult:
        reasons.append(reason)
        return _SENTINEL

    agent._should_compact_session = always  # type: ignore[method-assign]
    agent._compact_session = stubbed  # type: ignore[method-assign]

    result = await agent._auto_compact_if_needed()  # pyright: ignore[reportPrivateUsage]

    assert result is _SENTINEL
    assert reasons == ["auto_threshold"]


def test_a_compactor_injected_after_construction_is_the_one_a_session_gets() -> None:
    """The driver reads `_injected_compactor` when a session asks, not at construction.

    A scope that captured the value would keep handing out per-session compactors after
    the agent was given a shared one, and two sessions would stop sharing its state.

    Killed by: src/uclone_x/agent/base.py :: injected_compactor=lambda: self._injected_compactor,
    Becomes: injected_compactor=lambda _c=self._injected_compactor: _c,
    """
    agent = _agent()
    shared = ContextCompactor()

    agent._injected_compactor = shared  # pyright: ignore[reportPrivateUsage]

    assert agent._session_compactor("sess_a") is shared  # pyright: ignore[reportPrivateUsage]
    assert agent._session_compactor("sess_b") is shared  # pyright: ignore[reportPrivateUsage]
