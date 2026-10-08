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

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from uclone_x.agent.k_act import text_tools_section, tool_reference_line, visible_reply_text
from uclone_x.agent.models import BASE_PERSONA_TOOLS, ToolExecutionRecord
from uclone_x.agent.tools_module import (
    DEFAULT_TOOLS_MODULE,
    ToolsModuleName,
    bound_layer,
    grow_bound,
    pinned_layer,
)
from uclone_x.llm.models import (
    ChatMessage,
    LLMRequest,
    MessageRole,
    ToolDefinition,
)
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
from uclone_x.tools.tool_ranking import closest_tool_names, unknown_tool_message

__all__ = [
    "BoundToolsSession",
    "ToolInvoker",
    "ToolScope",
    "is_short_follow_up",
    "unadvertised_tool_message",
]

_FOLLOW_UP_SHORT_PATTERNS = re.compile(
    r"^(다시|다시\s*해봐|다시\s*해줘|다시\s*그려봐|다시\s*그려줘|계속|계속해|계속해줘|재시도|한번\s*더|한\s*번\s*더|이어서|"
    r"redo|retry|again|continue|more|try\s+again|do\s+it\s+again|one\s+more\s+time)"
    r"(\s*[\.!\?~]*)$",
    re.IGNORECASE,
)


def is_short_follow_up(message: str) -> bool:
    """Whether `message` is a short retry, repetition, or continuation directive (#2168)."""
    cleaned = message.strip()
    return len(cleaned) <= 30 and bool(_FOLLOW_UP_SHORT_PATTERNS.match(cleaned))


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

    @property
    def recorded(self) -> Sequence[ChatMessage]:
        """The session's history as its log holds it, forms unrendered (#1848).

        Read, never assigned. Rendering a form reads a kept result, which a restore must
        not depend on: only the tool calls are read here, and no form carries one.
        """
        ...

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

    `tools_module` is how the tools layer is built (`agent/tools_module.py`, #2188):
    ``native`` as described; ``pinned`` declares every held tool and never binds, though a
    binder is given; ``bound`` binds as ``native`` does but offers no `search_tools`;
    ``k_act`` builds the layer as ``bound`` does but declares none of it
    (`declared_tools`), describing it in text instead.
    """

    def __init__(
        self,
        scope: ToolScope,
        *,
        local_tools: Mapping[str, ToolProtocol] | None = None,
        agent_bound_types: tuple[type[ToolProtocol], ...],
        binder: ToolBinder | None = None,
        tools_module: ToolsModuleName = DEFAULT_TOOLS_MODULE,
    ) -> None:
        self._scope = scope
        self._tools_module: ToolsModuleName = tools_module
        self._agent_local_tools: dict[str, ToolProtocol] = dict(local_tools or {})
        self._agent_bound_types = agent_bound_types
        # Host binding (design §5.1). `None` pins every held tool; see `tools_for_turn`.
        # The binder is kept whatever the module, since a sub-agent's host takes it
        # (`binder`); `_binding` is the one this agent's own tools layer binds with.
        self._tool_binder = binder
        self._binding = binder if tools_module != "pinned" else None
        # `search_tools` is agent-local and never registered in the shared registry: it
        # searches this agent's catalog and appends to this agent's sessions, and only an
        # agent whose tools layer binds under the native module offers it
        # (`session_tools_layer`).
        self._search_tool: SearchToolsTool | None = None
        if self._binding is not None and tools_module == "native":
            self._search_tool = SearchToolsTool(self.search_catalog)
            self._agent_local_tools[self._search_tool.name] = self._search_tool

    @property
    def _registry(self) -> ToolRegistryProtocol | None:
        return self._scope.registry()

    @property
    def binder(self) -> ToolBinder | None:
        """The host's binder, or `None` when the host has none (whatever the module)."""
        return self._tool_binder

    @property
    def tools_module(self) -> ToolsModuleName:
        """How this agent builds its tools layer (#2188)."""
        return self._tools_module

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
        if self._binding is None:
            return
        held = [t.name for t in self.held_tools()]
        base_names = self.base_tool_names()
        catalog = set(held) - base_names
        bound: list[str] = []
        for message in live.recorded:
            for call in message.tool_calls:
                if call.name in catalog and call.name not in bound:
                    bound.append(call.name)
        live.bound_tools[:] = bound

    def reseed_bound_tools_post_compaction(
        self, live: BoundToolsSession, *, keep_recent_turns: int = 5
    ) -> None:
        """Reseed bound tools after compaction, keeping catalog tools called in recent turns (#2168).

        Unconditionally clearing bound tools on compaction caused subsequent turns with short
        directives ("다시 해봐", "continue") or low semantic similarity to lose tools like
        `generate_image` that the model was actively using. Reseeds tools called in the
        last `keep_recent_turns` user turns while discarding older, unused tools.
        """
        if self._binding is None:
            return
        active_tools = [t.name for t in self.held_tools()]
        base_names = self.base_tool_names()
        valid_catalog = set(active_tools) - base_names

        turn_count = 0
        recent_messages: list[ChatMessage] = []
        for msg in reversed(live.recorded):
            if msg.role is MessageRole.USER:
                turn_count += 1
                if turn_count > keep_recent_turns:
                    break
            recent_messages.append(msg)
        recent_messages.reverse()

        retained: list[str] = []
        for message in recent_messages:
            for call in message.tool_calls:
                if call.name in valid_catalog and call.name not in retained:
                    retained.append(call.name)

        last_calls = getattr(live, "last_turn_tool_calls", None) or ()
        for call in last_calls:
            call_name = getattr(call, "name", None)
            if call_name in valid_catalog and call_name not in retained:
                retained.append(call_name)

        live.bound_tools[:] = retained

    async def tools_for_turn(self, message: str, live: BoundToolsSession) -> list[ToolDefinition]:
        """The tools layer for the turn answering `message` (design §5.1).

        With no binder, or once binding has failed in this session, every held tool is
        pinned in canonical order. Otherwise the base set is pinned in canonical order and
        followed by the session's bound tools in the order they were bound; this message
        may append new ones, sorted by name among themselves. Nothing is ever removed
        before compaction, which clears `live.bound_tools` and `live.tools_pin_all`.
        """
        defs = self.advertised_tool_definitions()
        if self._binding is None or live.tools_pin_all:
            return pinned_layer(defs)
        catalog = self.binding_catalog(defs)
        if not catalog:
            return defs

        # Short follow-ups ("다시 해봐", "retry") repeat or continue the previous action.
        # Preserve tools called in the last turn so they are not dropped by embedding score (#2168).
        # In room turns, checkpoint_turn clears last_turn_tool_calls before the turn runs,
        # so fall back to the most recent assistant message in live.messages.
        if is_short_follow_up(message):
            last_calls = getattr(live, "last_turn_tool_calls", None) or ()
            if not last_calls:
                for m in reversed(getattr(live, "recorded", ())):
                    if m.role == MessageRole.ASSISTANT and m.tool_calls:
                        last_calls = m.tool_calls
                        break
            catalog_names = {t.name for t in catalog}
            for call in last_calls:
                call_name = getattr(call, "name", None)
                if call_name in catalog_names and call_name not in live.bound_tools:
                    live.bound_tools.append(call_name)

        hits = await self._binding.bind(message, catalog)
        if hits is None:
            live.tools_pin_all = True
            return defs
        grow_bound(live.bound_tools, hits)
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
        the next request of this turn on (design §5.1), as is a tool this step bound on the
        call that named it (`bind_on_call`): the two places the tools layer grows mid-turn,
        and only by appending. A step the loop guard cleared (`tool_defs` empty) stays
        cleared. `tool_defs` and `shown` (what F12 lets run)
        grow in place. One helper for both, because `execute_turn` sits at pyright's
        flow-analysis limit and one more statement there exceeds it.
        """
        searched = any(
            rec.tool_name == SEARCH_TOOLS_NAME and rec.status is ToolResultStatus.SUCCESS
            for rec in executions
        )
        # R3: a tool this step bound on the call that named it is declared from now on.
        bound_on_call = bool(tool_defs) and any(
            rec.tool_name in live.bound_tools and rec.tool_name not in shown for rec in executions
        )
        if searched or bound_on_call:
            tool_defs[len(tool_defs) :] = self.session_tools_layer(live)[len(tool_defs) :]
            shown.update(d.name for d in tool_defs)
        tools = self.declared_tools(tool_defs)
        return req.model_copy(update={"messages": tuple(messages), "tools": tools})

    def binding_catalog(self, defs: Sequence[ToolDefinition]) -> list[ToolDefinition]:
        """The part of `defs` host binding and `search_tools` choose from."""
        base_names = self.base_tool_names()
        return [d for d in defs if d.name not in base_names]

    def session_tools_layer(
        self, live: BoundToolsSession, defs: Sequence[ToolDefinition] | None = None
    ) -> list[ToolDefinition]:
        """The tools layer `live` declares now, without binding anything new.

        Pinned (no binder, the ``pinned`` module, a failed bind, or an empty catalog):
        every held tool in canonical order. Binding: the base set, plus `search_tools`
        under the ``native`` module, in canonical order, followed by the bound tools in the
        order they were bound. `search_tools` is offered only here, so an agent that pins
        everything -- one whose range is all base tools, or one with no binder -- never
        pays for it, and a pinned session does not carry a search it has no use for.
        """
        held = list(self.advertised_tool_definitions() if defs is None else defs)
        if self._binding is None or live.tools_pin_all:
            return pinned_layer(held)
        catalog = self.binding_catalog(held)
        if not catalog:
            return held
        catalog_names = {d.name for d in catalog}
        base = [d for d in held if d.name not in catalog_names]
        if self._search_tool is not None:
            search = ToolDefinition(
                name=self._search_tool.name,
                description=self._search_tool.description,
                parameters=advertised_tool_parameters(self._search_tool),
            )
            base = sorted([*base, search], key=lambda d: d.name)
        return bound_layer(base, catalog, live.bound_tools)

    def declared_tools(self, defs: Sequence[ToolDefinition]) -> tuple[ToolDefinition, ...]:
        """What a request declares to the provider for the tools layer `defs`.

        ``k_act`` declares nothing: its tools are text (`text_tools_section`,
        `text_tools_to_announce`). Every other module declares the layer as it is. `defs`
        stays the layer a call is checked against (F12) either way.
        """
        return () if self._tools_module == "k_act" else tuple(defs)

    def visible_reply(self, content: str) -> str:
        """A reply's text as a person is shown it: under ``k_act``, without its calls."""
        return visible_reply_text(content) if self._tools_module == "k_act" else content

    def _text_base(self) -> tuple[list[ToolDefinition], bool]:
        """``k_act``'s base set, described in the system text, and whether more can bind.

        The pinned part of the layer: with no binder or no catalog every held tool, else
        the held tools outside the binding catalog (no `search_tools`: the module has none).
        """
        defs = self.advertised_tool_definitions()
        if self._binding is None:
            return defs, False
        catalog = {d.name for d in self.binding_catalog(defs)}
        if not catalog:
            return defs, False
        return [d for d in defs if d.name not in catalog], True

    def text_tools_section(self) -> str | None:
        """The system text that offers ``k_act``'s base set; None under another module,
        or with no tool at all."""
        if self._tools_module != "k_act":
            return None
        base, more = self._text_base()
        if not base and not more:
            return None
        return text_tools_section(base, more=more)

    def text_tools_to_announce(
        self, layer: Sequence[ToolDefinition], history: Sequence[ChatMessage]
    ) -> list[ToolDefinition]:
        """The tools of `layer` that ``k_act`` has not described yet, in layer order.

        A tool is described once: in the system text (the base set), or on the user
        message or tool result where it first applied, which stays in history. So what is
        described is read off the history itself, not kept beside it: a restart, which
        reseeds the bound set from the calls in history, describes nothing twice, and a
        tool whose description a compaction folded away is described again. Empty under
        any other module.
        """
        if self._tools_module != "k_act":
            return []
        base = {d.name for d in self._text_base()[0]}
        shown = "\n".join(m.content for m in history if m.content)
        return [d for d in layer if d.name not in base and tool_reference_line(d) not in shown]

    def bind_on_call(self, name: str, session_id: str) -> bool:
        """Bind `name` on the call that named it (R3, #2190); whether it may now run.

        A call to a held catalog tool its request did not declare -- one binding missed --
        binds that tool and runs, instead of being refused (F12). Only where this agent's
        tools layer binds (a binder, a session not pinned) and only for a tool of this
        turn's binding catalog: a name outside `allowed_tools`, withheld by the persona
        flags or by the room, or a base tool, is not in it and keeps its refusal. The tool
        is appended to the session's grow-only bound set, so from the next request it is
        declared after every tool bound before it; nothing is inserted earlier.
        """
        live = self._scope.live_session(session_id)
        if live.tools_pin_all or self._binding is None:
            return False
        # The binding catalog is held tools only: in range, not withheld by the persona
        # flags or the room, and not base. Anything else is not bound here.
        catalog = self.binding_catalog(self.advertised_tool_definitions())
        if name not in {d.name for d in catalog}:
            return False
        grow_bound(live.bound_tools, [name])
        return True

    def knows_tool(self, name: str) -> bool:
        """Whether any tool answers to `name`: this agent's own, or one in the registry,
        whatever this agent may use. A name nothing answers to gets `unknown_tool_result`."""
        if name in self._agent_local_tools:
            return True
        return self._registry is not None and self._registry.get(name) is not None

    def unknown_tool_result(self, name: str, arguments: Mapping[str, object]) -> str:
        """R4 (#2190): the plain tool result for a call to `name`, which no tool has.

        It lists the closest tools this agent may use this turn (`closest_tool_names`, by
        the call's name and argument names), so a misspelled or invented name finds the
        real one; a catalog tool named next is bound on that call (`bind_on_call`).
        """
        candidates = [
            (
                d.name,
                d.description,
                [
                    str(key)
                    for key in cast("Mapping[str, object]", d.parameters.get("properties") or {})
                ],
            )
            for d in self.advertised_tool_definitions()
        ]
        return unknown_tool_message(name, closest_tool_names(name, list(arguments), candidates))

    async def search_catalog(self, query: str, session_id: str) -> str:
        """Run `search_tools(query)` for `session_id` (design §5.1).

        Ranks this agent's catalog with the binder, appends the hits to the session's
        grow-only bound set, and answers with each hit's name and one-line description,
        never a schema. The hits are declared from the next request on. A search that
        cannot run -- no binder, nothing to search, or a failed embedder -- is refused in
        plain words (`SEARCH_UNAVAILABLE_MESSAGE`), with no exception text.
        """
        binder = self._binding
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
