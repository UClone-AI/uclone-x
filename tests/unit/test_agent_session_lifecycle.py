"""`agent/session_lifecycle.py` on its own: the sessions behind `BaseAgent` (#1736).

Switching, loading, resetting, saving and restoring sessions moved out of `agent/base.py`
unchanged, and the behavioural tests that drive them through `BaseAgent` stay where they
were (`test_agent_session_multitenancy.py`, `test_agent_base.py`, ...). This file pins
what the move itself introduced: the lifecycle holds no copy of agent state, what it
writes lands on the agent, and the agent methods it calls back are the agent's current
ones.
"""

from __future__ import annotations

from pathlib import Path

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig
from uclone_x.agent.session import SessionState, SessionStore
from uclone_x.llm import MockLLMConnector
from uclone_x.llm.models import ChatMessage, MessageRole


def _agent(store: SessionStore | None = None) -> BaseAgent:
    return BaseAgent(
        config=AgentConfig(agent_id="lifecycle", name="Lifecycle"),
        llm=MockLLMConnector(),
        store=store,
    )


def test_a_switch_lands_the_new_active_session_on_the_agent() -> None:
    """The switch's `self._context = ...` goes through the scope's setter to the agent.

    The lifecycle has no context of its own. A setter that dropped the write would leave
    the agent reporting the session it was switched away from, and every later turn
    would file its messages there.

    Killed by: src/uclone_x/agent/base.py :: set_context=lambda value: setattr(self, "_context", value),
    Becomes: set_context=lambda value: None,
    """
    agent = _agent()
    before = agent.session_id

    agent.switch_session("sess_other")

    assert agent.session_id == "sess_other"
    assert set(agent.session_ids) == {before, "sess_other"}


def test_a_store_swapped_after_construction_is_the_one_a_hydrate_reads(tmp_path: Path) -> None:
    """The lifecycle reads `_store` when it runs, not the first time it looked.

    Callers assign agent attributes after construction (the harness ladder does). A
    lifecycle that kept the store it first saw would restore from a store nobody
    configured any more, and report "no record" for a session that has one.

    Killed by: src/uclone_x/agent/session_lifecycle.py :: return self._scope.store()
    Becomes: return self.__dict__.setdefault("_first_store", self._scope.store())
    """
    first = SessionStore(storage_dir=tmp_path / "first")
    second = SessionStore(storage_dir=tmp_path / "second")
    writer = _agent(second)
    writer.load_history([ChatMessage(role=MessageRole.USER, content="kept in the second")])
    writer.persist_session()

    agent = _agent(first)
    assert agent.hydrate_session() is None
    agent._store = second  # pyright: ignore[reportPrivateUsage]

    restored = agent.hydrate_session()

    assert restored is not None
    assert [m.content for m in agent.history] == ["kept in the second"]


def test_a_persist_writes_the_snapshot_the_agents_own_get_session_returns(
    tmp_path: Path,
) -> None:
    """`persist_session` asks the agent for the snapshot, so a replaced one is what is saved.

    `get_session` is both a moved method and a name kept on the agent. The moved
    `persist_session` reaches it back through the agent, so an override on the instance
    or a subclass decides what the record holds, as it did before the move.

    Killed by: src/uclone_x/agent/base.py :: get_session=lambda: self.get_session,
    Becomes: get_session=lambda: self._session_lifecycle.get_session,
    """
    store = SessionStore(storage_dir=tmp_path)
    agent = _agent(store)
    original = agent.get_session

    def stamped(session_id: str | None = None) -> SessionState:
        return original(session_id).model_copy(update={"turn_counter": 41})

    agent.get_session = stamped  # type: ignore[method-assign]
    agent.persist_session()

    saved = store.load(agent.session_id)
    assert saved is not None
    assert saved.turn_counter == 41
