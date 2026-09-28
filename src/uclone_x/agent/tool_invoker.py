"""Which tools an agent holds, advertises, binds and resolves (#1736).

`BaseAgent` is the facade; this is the collaborator that owns its tool catalog. It holds
the agent-local tool instances, the host binder and the agent's `search_tools`, and it
answers every question about the tools layer (design §5.1): what the agent holds, what a
turn offers, what host binding pins and appends, and which instance a call resolves to.

It owns no session and no persona. What it reads from the agent it serves -- the tool
range, the room of the running turn, the persona flags, whether it has peers, and the live
session a search appends to -- comes through a `ToolScope`, read afresh on every call, so a
persona change or a new turn reaches it without a second copy of that state.

Running a tool call (hooks, approvals, the range and persona-flag refusals, outcome
classification) is `ToolCallExecutor.execute_single_tool` in `agent/tool_execution.py`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from uclone_x.agent.models import BASE_PERSONA_TOOLS, ToolExecutionRecord
from uclone_x.llm.models import ChatMessage, LLMRequest, ToolDefinition
from uclone_x.tools.base import drop_shadowed_aliases, tool_needs_room
from uclone_x.tools.models import ToolResultStatus
from uclone_x.tools.protocols import ToolProtocol, ToolRegistryProtocol
from uclone_x.tools.schema import advertised_tool_parameters
from uclone_x.tools.tool_binder import (
    SEARCH_TOOLS_NAME,
    SearchToolsTool,
    ToolBinder,
    search_unavailable,
)

__all__ = [
    "BoundToolsSession",
    "ToolInvoker",
    "ToolScope",
    "unadvertised_tool_message",
]

#: What a binding tools layer pins (design §5.1): the base set every persona is given, and
#: the search over the rest. Every other held tool is the catalog (`base_tool_names`).
_PINNED_BASE_TOOLS = frozenset((*BASE_PERSONA_TOOLS, SEARCH_TOOLS_NAME))


def _first_line(text: str) -> str:
    """A description's first line, which is what `search_tools` lists for a tool."""
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def unadvertised_tool_message(name: str) -> str:
    """The tool result for a call to a name its request did not declare (F12).

    Plain words for the model that made the call: what happened and what to do instead.
    No agent id, no permission list and no exception text, since none of that helps the
    model choose its next call and all of it can reach the person's screen.
    """
    return (
        f"The tool '{name}' is not available this turn, so it was not run. "
        "Use only the tools listed in this request."
    )


class BoundToolsSession(Protocol):
    """The part of a live session the tools layer reads and grows (design §5.1).

    `bound_tools` is the session's grow-only bound set, in the order it was bound;
    `tools_pin_all` is set once binding fails, and pins every held tool until compaction.
    """

    messages: list[ChatMessage]
    bound_tools: list[str]
    tools_pin_all: bool


@dataclass(frozen=True, slots=True)
class ToolScope:
    """What a `ToolInvoker` reads from the agent it serves, each value read when needed.

    Callables rather than values, because every one of them changes under the invoker:
    a persona change re-resolves `allowed_tools` and the flags, each turn sets its room,
    and sessions come and go. The registry too: it is the agent's attribute, and a caller
    that replaces it after construction (the harness ladder's oracle arm does) must be
    what the invoker then lists and resolves from.
    """

    #: The agent's (possibly shared) `ToolRegistry`, or `None` for an agent with no tools.
    registry: Callable[[], ToolRegistryProtocol | None]
    #: The agent's resolved `config.allowed_tools`: its range; empty permits everything.
    allowed_tools: Callable[[], tuple[str, ...]]
    #: The conversation of the running turn, or `None` outside a room.
    room_id: Callable[[], str | None]
    #: Why the persona flags withhold a tool, or `None` when they do not.
    capability_refusal: Callable[[object], str | None]
    #: Whether `a2a_call` could reach anyone.
    may_call_peers: Callable[[], bool]
    #: The live session a `search_tools` call appends its hits to.
    live_session: Callable[[str], BoundToolsSession]


class ToolInvoker:
    """One agent's tool catalog: held, advertised, bound and resolved tools.

    The registry comes through the scope; `local_tools` are the instances bound to this
    agent's own state, consulted before it. `agent_bound_types` are
    the tool classes whose registry instance belongs to some other agent, so neither held
    nor resolved unless this agent registered its own (`AGENT_BOUND_TOOL_TYPES` in
    `agent/base.py`, which is where those adapter classes may be imported). With a `binder`
    the agent's tools layer binds, and the invoker offers its own agent-local `search_tools`.
    """

    def __init__(
        self,
        scope: ToolScope,
        *,
        local_tools: Mapping[str, ToolProtocol] | None = None,
        agent_bound_types: tuple[type[ToolProtocol], ...],
        binder: ToolBinder | None = None,
    ) -> None:
        self._scope = scope
        self._agent_local_tools: dict[str, ToolProtocol] = dict(local_tools or {})
        self._agent_bound_types = agent_bound_types
        # Host binding (design §5.1). `None` pins every held tool; see `tools_for_turn`.
        self._tool_binder = binder
        # `search_tools` is agent-local and never registered in the shared registry: it
        # searches this agent's catalog and appends to this agent's sessions, and only an
        # agent whose tools layer binds offers it (`session_tools_layer`).
        self._search_tool: SearchToolsTool | None = None
        if binder is not None:
            self._search_tool = SearchToolsTool(self.search_catalog)
            self._agent_local_tools[self._search_tool.name] = self._search_tool

    @property
    def _registry(self) -> ToolRegistryProtocol | None:
        return self._scope.registry()

    @property
    def binder(self) -> ToolBinder | None:
        """The host binder, or `None` when every held tool is pinned."""
        return self._tool_binder

    @property
    def local_tools(self) -> dict[str, ToolProtocol]:
        """The instances bound to this agent's own state, by name; resolved first."""
        return self._agent_local_tools

    def would_run_if_advertised(self, name: str) -> bool:
        """Whether a call to `name` would pass every check but the advertisement one.

        Only such a call is refused by the F12 check; any other keeps the refusal of the
        check it fails (`allowed_tools`, the registry, the persona flags) and its words.
        """
        if not self.in_tool_range(name):
            return False
        tool = self.resolve(name)
        return tool is not None and self._scope.capability_refusal(tool) is None

    def in_tool_range(self, name: str) -> bool:
        """Whether `config.allowed_tools` lets this agent run `name` (an empty list: all).

        `search_tools` passes whenever this agent offers it, though no list names it: it is
        agent-local, searches only this agent's own range and binds only from it, so it
        reaches nothing the list withholds (design §5.1).
        """
        allowed = self._scope.allowed_tools()
        if not allowed or name in allowed:
            return True
        return name == SEARCH_TOOLS_NAME and self._search_tool is not None

    def base_tool_names(self) -> frozenset[str]:
        """The names pinned when the tools layer binds; every other held tool is catalog.

        **The rule (design §5.1, Revision 3).** An `allowed_tools` list is the agent's
        *range* -- what it may use -- not what it sends. The pinned set is always
        `BASE_PERSONA_TOOLS` plus `search_tools`, less whatever the agent does not hold;
        every other held tool, so every other name in the range, is its binding catalog.
        An agent with no list holds the whole registry, and the rule is the same. There is
        one rule for a persona's list and an operator's: by the time the list reaches here
        they are one `config.allowed_tools`, and an operator list is often a persona's
        passed on (a sub-agent's, an `a2a_call` callee's). A held tool is always in range
        (`held_tools` filters by the list), and dispatch still refuses one outside it.
        """
        return _PINNED_BASE_TOOLS

    def reseed_bound_tools_from_history(self, live: BoundToolsSession) -> None:
        """Rebuild a restored session's bound set from the catalog tools its history called.

        `bound_tools` lives only in memory, so a restarted process restored a session whose
        history calls tools its next request no longer declared; on Ollama a repeat of such
        a call ("do it again") is dropped without an error. The bound set is therefore the
        catalog tools this agent still holds that the history called, in first-call order.
        Base tools and anything no longer held are skipped. Nothing happens without a
        binder: every held tool is pinned then anyway.
        """
        if self._tool_binder is None:
            return
        held = [t.name for t in self.held_tools()]
        base_names = self.base_tool_names()
        catalog = set(held) - base_names
        bound: list[str] = []
        for message in live.messages:
            for call in message.tool_calls:
                if call.name in catalog and call.name not in bound:
                    bound.append(call.name)
        live.bound_tools[:] = bound

    async def tools_for_turn(self, message: str, live: BoundToolsSession) -> list[ToolDefinition]:
        """The tools layer for the turn answering `message` (design §5.1).

        With no binder, or once binding has failed in this session, every held tool is
        pinned in canonical order. Otherwise the base set is pinned in canonical order and
        followed by the session's bound tools in the order they were bound; this message
        may append new ones, sorted by name among themselves. Nothing is ever removed
        before compaction, which clears `live.bound_tools` and `live.tools_pin_all`.
        """
        defs = self.advertised_tool_definitions()
        if self._tool_binder is None or live.tools_pin_all:
            return defs
        catalog = self.binding_catalog(defs)
        if not catalog:
            return defs
        hits = await self._tool_binder.bind(message, catalog)
        if hits is None:
            live.tools_pin_all = True
            return defs
        live.bound_tools.extend(sorted(set(hits) - set(live.bound_tools)))
        return self.session_tools_layer(live, defs)

    def tools_after_step(
        self,
        req: LLMRequest,
        messages: Sequence[ChatMessage],
        live: BoundToolsSession,
        tool_defs: list[ToolDefinition],
        shown: set[str],
        executions: Sequence[ToolExecutionRecord],
    ) -> LLMRequest:
        """The next step's request: `messages`, and the tools layer after this step ran.

        A `search_tools` hit is appended to the session's bound set, and is declared from
        the next request of this turn on (design §5.1): the one place the tools layer
        grows mid-turn, and only by appending. `tool_defs` and `shown` (what F12 lets run)
        grow in place. One helper for both, because `execute_turn` sits at pyright's
        flow-analysis limit and one more statement there exceeds it.
        """
        if any(
            rec.tool_name == SEARCH_TOOLS_NAME and rec.status is ToolResultStatus.SUCCESS
            for rec in executions
        ):
            tool_defs[len(tool_defs) :] = self.session_tools_layer(live)[len(tool_defs) :]
            shown.update(d.name for d in tool_defs)
        return req.model_copy(update={"messages": tuple(messages), "tools": tuple(tool_defs)})

    def binding_catalog(self, defs: Sequence[ToolDefinition]) -> list[ToolDefinition]:
        """The part of `defs` host binding and `search_tools` choose from."""
        base_names = self.base_tool_names()
        return [d for d in defs if d.name not in base_names]

    def session_tools_layer(
        self, live: BoundToolsSession, defs: Sequence[ToolDefinition] | None = None
    ) -> list[ToolDefinition]:
        """The tools layer `live` declares now, without binding anything new.

        Pinned (no binder, a failed bind, or an empty catalog): every held tool in
        canonical order. Binding: the base set plus `search_tools` in canonical order,
        followed by the bound tools in the order they were bound. `search_tools` is offered
        only here, so an agent that pins everything -- one whose range is all base tools,
        or one with no binder -- never pays for it, and a pinned session does not carry a
        search it has no use for.
        """
        held = list(self.advertised_tool_definitions() if defs is None else defs)
        if self._tool_binder is None or self._search_tool is None or live.tools_pin_all:
            return held
        catalog = self.binding_catalog(held)
        if not catalog:
            return held
        catalog_names = {d.name for d in catalog}
        search = ToolDefinition(
            name=self._search_tool.name,
            description=self._search_tool.description,
            parameters=advertised_tool_parameters(self._search_tool),
        )
        base = sorted(
            [*(d for d in held if d.name not in catalog_names), search], key=lambda d: d.name
        )
        by_name = {d.name: d for d in catalog}
        return [*base, *(by_name[name] for name in live.bound_tools if name in by_name)]

    async def search_catalog(self, query: str, session_id: str) -> str:
        """Run `search_tools(query)` for `session_id` (design §5.1).

        Ranks this agent's catalog with the binder, appends the hits to the session's
        grow-only bound set, and answers with each hit's name and one-line description,
        never a schema. The hits are declared from the next request on. A search that
        cannot run -- no binder, nothing to search, or a failed embedder -- is refused in
        plain words (`SEARCH_UNAVAILABLE_MESSAGE`), with no exception text.
        """
        binder = self._tool_binder
        catalog = self.binding_catalog(self.advertised_tool_definitions())
        hits = await binder.search(query, catalog) if binder is not None and catalog else None
        if hits is None:
            raise search_unavailable()
        if not hits:
            return "No tool matched that. Try other words, or use the tools you have."
        live = self._scope.live_session(session_id)
        newly_found = sorted(set(hits).difference(live.bound_tools))
        live.bound_tools.extend(newly_found)
        by_name = {d.name: d for d in catalog}
        lines = [f"- {name}: {_first_line(by_name[name].description)}" for name in hits]
        return "Found, and callable from your next step:\n" + "\n".join(lines)

    def advertised_tool_definitions(self) -> list[ToolDefinition]:
        """`available_tools` as the model is sent them, before host binding.

        Each schema goes through `advertised_tool_parameters` (#1542, #1543), so every
        connector sends the same compacted bytes; `parameters_schema` itself stays raw.
        Sorted by name (design §5.1): the registry lists tools in registration order, and
        MCP servers register in whichever order `asyncio.gather` finishes them, so that
        order must not reach the request prefix.
        """
        # The canonical order covers the held set. A tool host binding appends
        # mid-session goes after the base set, not sorted into it (`tools_for_turn`).
        return [
            ToolDefinition(
                name=t.name,
                description=t.description,
                parameters=advertised_tool_parameters(t),
            )
            for t in sorted(self.available_tools(), key=lambda tool: tool.name)
        ]

    def available_tools(self) -> list[ToolProtocol]:
        """What a turn offers the model, before host binding: `held_tools`, fitted to the turn.

        A turn outside a conversation (a room) is not offered the tools that declare
        `needs_room` (see `tool_needs_room`): nothing it could use from them is worth their
        schemas' room in the window. Host binding may pin only part of it (`tools_for_turn`).
        """
        offered: list[ToolProtocol] = []
        for t in self.held_tools():
            # A tool that declares `needs_room` (the story tools) is kept out of a turn
            # outside a room, so its schema does not cost that request (#1556, #1576).
            if self._scope.room_id() is None and tool_needs_room(t):
                continue
            offered.append(t)
        return offered

    def held_tools(self) -> list[ToolProtocol]:
        """The tools this agent actually has, whatever conversation its next turn is in.

        `config.allowed_tools` is a list of permissions, not of tools. A name in it may have
        nothing registered behind it -- a memory tool on an agent built without a store, a
        file tool on a registry that has none -- and an empty list permits everything. This
        is the registry filtered by that list, less what this agent cannot run. It does not
        depend on the room of the last turn, so what a dashboard lists for an agent that
        has not yet run in a room still includes the tools it would get there (#1576);
        `tool_needs_room` says which those are.
        """
        if self._registry is None:
            return []
        from uclone_x.tools.builtin.a2a import A2A_CALL_TOOL_NAME as a2a_call_name

        allowed = self._scope.allowed_tools() if self._scope.allowed_tools() else None
        held: list[ToolProtocol] = []
        for t in drop_shadowed_aliases(self._registry.list_tools(filter_names=allowed)):
            # Advertising an agent-bound tool this agent cannot resolve would offer a
            # capability whose every call answers "not found".
            if isinstance(t, self._agent_bound_types) and t.name not in self._agent_local_tools:
                continue
            # A tool the persona flags withhold is not offered. This is the hint; the
            # refusals in `ToolCallExecutor.execute_single_tool` and `execute_tool_call` are the
            # enforcement (#1167).
            if self._scope.capability_refusal(t) is not None:
                continue
            # `a2a_call` is offered only to an agent that has someone to call (#1558). The
            # tool refuses the rest itself; this keeps its schema out of every other turn.
            if t.name == a2a_call_name and not self._scope.may_call_peers():
                continue
            held.append(t)
        return held

    def resolve(self, name: str) -> ToolProtocol | None:
        """The instance this agent executes for `name`, agent-local binding first.

        A tool held in `_agent_local_tools` is bound to state only this agent may act on --
        its memory store, and its skill registry with the `_loaded_skills` set that records
        what it loaded. Resolving those from the registry instead would hand an agent
        whichever instance was registered first, which in the UI is another agent's.
        """
        local = self._agent_local_tools.get(name)
        if local is not None:
            return local
        candidate = self._registry.get(name) if self._registry is not None else None
        # An agent-bound *instance* this agent did not register is *another* agent's, and
        # the shared registry will happily hand it over. An agent composed without a
        # memory store then records into whichever store was composed first; an agent
        # composed without a skill registry loads from another agent's approval list.
        # "No such tool" is the honest answer; a working call into someone else's state is
        # not (P6). A tool that merely shares the name is not one of those instances and
        # resolves normally.
        if isinstance(candidate, self._agent_bound_types):
            return None
        return candidate
