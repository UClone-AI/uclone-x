"""`ToolInvoker` on its own: the tool catalog `BaseAgent` delegates to (#1736).

The agent-level behaviour -- binding, F12, agent-bound memory and skill tools -- is covered
through `BaseAgent` in `test_tool_binder.py`, `test_memory_composition.py` and their
neighbours. These pin what is new with the collaborator: that it reads the agent's state
through its `ToolScope` on every call rather than from a copy, and that what it is handed
at construction (local instances, agent-bound types, the binder) is what it acts on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from uclone_x.agent.tool_invoker import BoundToolsSession, ToolInvoker, ToolScope
from uclone_x.llm.models import ChatMessage
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.protocols import ToolProtocol
from uclone_x.tools.registry import ToolRegistry
from uclone_x.tools.tool_binder import SEARCH_TOOLS_NAME, ToolBinder


class _Params(BaseModel):
    text: str = Field(default="")


class _Plain(BaseTool[_Params]):
    name = "plain"
    description = "A tool any agent may share."

    def run(self, params: _Params, context: ToolContext) -> str:
        return "plain ran"


class _Other(BaseTool[_Params]):
    name = "other"
    description = "Another shared tool."

    def run(self, params: _Params, context: ToolContext) -> str:
        return "other ran"


class _RoomOnly(BaseTool[_Params]):
    name = "room_only"
    description = "Offered only inside a conversation."
    needs_room = True

    def run(self, params: _Params, context: ToolContext) -> str:
        return "room_only ran"


class _Bound(BaseTool[_Params]):
    """Stands in for a memory or skill tool: its instance belongs to one agent."""

    name = "bound"
    description = "Bound to one agent's own state."

    def __init__(self, owner: str) -> None:
        super().__init__()
        self.owner = owner

    def run(self, params: _Params, context: ToolContext) -> str:
        return f"{self.owner} ran"


@dataclass
class _Session:
    recorded: list[ChatMessage] = field(default_factory=list[ChatMessage])
    bound_tools: list[str] = field(default_factory=list[str])
    tools_pin_all: bool = False


@dataclass
class _Agent:
    """The agent-side state a scope reads, mutable so a test can change it mid-way."""

    registry: ToolRegistry | None = None
    allowed: tuple[str, ...] = ()
    room: str | None = None
    session: _Session = field(default_factory=_Session)

    def scope(self) -> ToolScope:
        def live_session(session_id: str) -> BoundToolsSession:
            return self.session

        return ToolScope(
            registry=lambda: self.registry,
            allowed_tools=lambda: self.allowed,
            room_id=lambda: self.room,
            capability_refusal=lambda tool: None,
            may_call_peers=lambda: False,
            live_session=live_session,
        )


class _Embedder:
    """Never asked to embed here: constructing a binder does not embed."""

    @property
    def model_name(self) -> str:
        return "fake-embed"

    @property
    def dimensions(self) -> int:
        return 2

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        raise AssertionError("no test here embeds")


def _registry(*tools: ToolProtocol) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _names(tools: Sequence[ToolProtocol]) -> list[str]:
    return [t.name for t in tools]


def test_the_tool_range_is_read_from_the_scope_on_every_call() -> None:
    """A persona change reaches the invoker without rebuilding it: no copy of the list.

    Killed by: src/uclone_x/agent/tool_invoker.py :: if not allowed or name in allowed:
    Becomes: if True:
    """
    agent = _Agent(_registry(_Plain(), _Other()), allowed=("plain",))
    invoker = ToolInvoker(agent.scope(), agent_bound_types=())
    assert invoker.in_tool_range("plain")
    assert not invoker.in_tool_range("other")
    agent.allowed = ("other",)
    assert not invoker.in_tool_range("plain")
    assert invoker.in_tool_range("other")
    assert _names(invoker.held_tools()) == ["other"]
    # The registry is read live too: one replaced after construction is the one listed.
    agent.registry = _registry(_Other())
    agent.allowed = ()
    assert _names(invoker.held_tools()) == ["other"]


def test_a_local_instance_resolves_before_the_registry_one_of_the_same_name() -> None:
    """The instance handed in as local is the one this agent runs, not the shared one.

    Killed by: src/uclone_x/agent/tool_invoker.py :: if local is not None:
    Becomes: if False:
    """
    shared, mine = _Bound("first agent"), _Bound("this agent")
    invoker = ToolInvoker(
        _Agent(_registry(shared)).scope(),
        local_tools={mine.name: mine},
        agent_bound_types=(),
    )
    assert invoker.resolve("bound") is mine


def test_an_agent_bound_instance_this_agent_did_not_register_is_neither_held_nor_run() -> None:
    """The types handed in are the ones refused from the registry, and only those.

    Killed by: src/uclone_x/agent/tool_invoker.py :: self._agent_bound_types = agent_bound_types
    Becomes: self._agent_bound_types = ()
    """
    registry = _registry(_Bound("first agent"), _Plain())
    invoker = ToolInvoker(_Agent(registry).scope(), agent_bound_types=(_Bound,))
    assert invoker.resolve("bound") is None
    assert invoker.resolve("plain") is not None
    assert _names(invoker.held_tools()) == ["plain"]


def test_a_room_only_tool_is_offered_once_the_turn_has_a_room() -> None:
    """Held either way; offered, and advertised in name order, only inside a room.

    Killed by: src/uclone_x/agent/tool_invoker.py :: if self._scope.room_id() is None and tool_needs_room(t):
    Becomes: if tool_needs_room(t):
    """
    agent = _Agent(_registry(_RoomOnly(), _Plain()))
    invoker = ToolInvoker(agent.scope(), agent_bound_types=())
    assert _names(invoker.held_tools()) == ["room_only", "plain"]
    assert _names(invoker.available_tools()) == ["plain"]
    agent.room = "room-1"
    assert [d.name for d in invoker.advertised_tool_definitions()] == ["plain", "room_only"]


def test_search_tools_is_agent_local_and_exists_only_with_a_binder() -> None:
    """With a binder, `search_tools` is this invoker's own, in range though no list names it.

    Killed by: src/uclone_x/agent/tool_invoker.py :: self._agent_local_tools[self._search_tool.name] = self._search_tool
    Becomes: pass
    """
    registry = _registry(_Plain())
    agent = _Agent(registry, allowed=("plain",))
    unbound = ToolInvoker(agent.scope(), agent_bound_types=())
    assert unbound.binder is None
    assert unbound.resolve(SEARCH_TOOLS_NAME) is None
    assert not unbound.in_tool_range(SEARCH_TOOLS_NAME)

    binder = ToolBinder(_Embedder())
    bound = ToolInvoker(agent.scope(), agent_bound_types=(), binder=binder)
    assert bound.binder is binder
    assert bound.in_tool_range(SEARCH_TOOLS_NAME)
    search = bound.resolve(SEARCH_TOOLS_NAME)
    assert search is not None and search is bound.local_tools[SEARCH_TOOLS_NAME]
    assert registry.get(SEARCH_TOOLS_NAME) is None
