"""The one place a clone is assembled: its host and its config (llm-request-layering §5.9).

A clone builds its LLM request one way, in a 1:1 chat, in a room seat and from the CLI
(owner ruling, 2026-09-27). Before this module each head assembled the clone itself, at
eight `HostDependencies(` sites, and the copies had drifted: only the room bound tools,
the 1:1 chat named the agent differently and set no workspace, and each CLI command
opened its own memory store. Now a head says what the app has (`AppScope`) and asks for
a clone (`build_clone`). A room seat passes only what a room adds: its framing, its
display name and the room's A2A transport (§5.9.3). A clone's rules engine is its own,
one per clone id like its memory (clone-knowledge-graph §3.1, step 6); no seat has one.

A shell module, as `room/resolver.py` is: it composes agents, so it may import adapters.
Nothing in the kernel imports it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TypeAlias

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.bootstrap import agent_config_for_persona
from uclone_x.agent.composition import HostDependencies, compose_agent
from uclone_x.agent.models import (
    DEFAULT_SYSTEM_PROMPT,
    AgentConfig,
    AgentContext,
    AgentLLMConfig,
    PersonaDefinition,
)
from uclone_x.agent.persona_registry import PersonaRegistry, get_default_persona_registry
from uclone_x.memory.store import CrossSessionMemory, default_cross_session_memory
from uclone_x.ontology.engine import OntologyEngine
from uclone_x.ontology.protocols import OntologyEngineProtocol
from uclone_x.story import StoryLifecycleHook
from uclone_x.tools.tool_binder import ToolBinder, tool_binder_for

if TYPE_CHECKING:
    from uclone_x.a2a.protocols import A2ATransportProtocol
    from uclone_x.llm.protocols import LLMProviderProtocol
    from uclone_x.tools.protocols import ToolRegistryProtocol

__all__ = [
    "APP_ONTOLOGY",
    "AppScope",
    "BuiltClone",
    "CLONE_ONTOLOGY_ROOT",
    "GlobalModels",
    "OntologyChoice",
    "build_clone",
    "clone_namespace",
    "clone_ontology",
    "connector_tool_binder",
    "follow_global_models",
    "local_app_scope",
    "memory_map",
    "named_clone_prompt",
    "ontology_map",
    "provider_tool_binder",
    "saved_models",
    "with_app_lifecycle_hooks",
]


GlobalModels = Callable[[], tuple[str | None, str | None]]
"""Returns ``(deep, fast)`` as Settings holds them at the moment of the call."""


def follow_global_models(own: AgentLLMConfig, deep: str | None, fast: str | None) -> AgentLLMConfig:
    """``own`` with each empty model slot filled from the global ``deep`` and ``fast``.

    The one rule for which model a clone runs on, in 1:1 chat and in a room alike: a model
    the persona names is kept, and a slot it leaves empty follows Settings. An empty global
    fast model means the deep one.
    """
    updates: dict[str, str] = {}
    if not own.model_name and deep:
        updates["model_name"] = deep
    fast_or_deep = fast or deep
    if not own.fast_model and fast_or_deep:
        updates["fast_model"] = fast_or_deep
    return own.model_copy(update=updates) if updates else own


def provider_tool_binder(provider: str, base_url: str) -> ToolBinder | None:
    """The host binder for `provider` (design §5.1), or `None` where every tool is pinned.

    One binder per app scope, not per clone: each tool description is then embedded once
    for every clone that app serves. The grow-only bound set itself is per seat session.
    """
    from uclone_x.llm.connectors.ollama_embedder import OllamaEmbedder

    return tool_binder_for(provider, base_url, lambda url: OllamaEmbedder(base_url=url))


def connector_tool_binder(llm: object) -> ToolBinder | None:
    """The host binder for the connector a CLI command built, read off the connector.

    The CLI has no Settings to read the provider from; the connector it resolved names
    its own (`provider_name`) and the endpoint it speaks to (`base_url`).
    """
    provider = getattr(llm, "provider_name", "")
    base_url = getattr(llm, "base_url", "")
    if not isinstance(provider, str) or not isinstance(base_url, str):
        return None
    return provider_tool_binder(provider, base_url)


def memory_map(
    opener: Callable[[str], CrossSessionMemory] = default_cross_session_memory,
) -> Callable[[str], CrossSessionMemory]:
    """A get-or-create map from clone id to that clone's one memory store.

    `CrossSessionMemory.save()` rewrites the whole document, so two stores over one file
    lose each other's writes (P6). A process keeps one map and hands it to every build.
    """
    stores: dict[str, CrossSessionMemory] = {}

    def memory_for(clone_id: str) -> CrossSessionMemory:
        existing = stores.get(clone_id)
        if existing is None:
            existing = opener(clone_id)
            stores[clone_id] = existing
        return existing

    return memory_for


#: Where every clone's rules engine is named: `<root>/<clone id>` (§3.1, Q6 #1654).
CLONE_ONTOLOGY_ROOT: Final = "https://uclone-x.ai/ontology/clones"


def clone_namespace(clone_id: str) -> str:
    """The namespace of clone `clone_id`'s rules engine: one per clone, in every conversation."""
    return f"{CLONE_ONTOLOGY_ROOT}/{clone_id}"


def clone_ontology(clone_id: str) -> OntologyEngineProtocol:
    """A new, empty rules engine for clone `clone_id`, in the clone's own namespace.

    It holds the rules a person gives the clone and nothing learned in a conversation: what
    the clone learned is its facts, in its memory, and what follows from them under these
    rules is worked out on read and never kept (clone-knowledge-graph §3.1).
    """
    return OntologyEngine(agent_id=clone_id, namespace_iri=clone_namespace(clone_id))


def ontology_map(
    opener: Callable[[str], OntologyEngineProtocol] = clone_ontology,
) -> Callable[[str], OntologyEngineProtocol]:
    """A get-or-create map from clone id to that clone's one rules engine, as `memory_map`.

    One engine per clone and not per seat: a clone seated in two conversations reasons
    under one set of rules, and two clones never share an engine (§3.1, step 6).
    """
    engines: dict[str, OntologyEngineProtocol] = {}

    def ontology_for(clone_id: str) -> OntologyEngineProtocol:
        existing = engines.get(clone_id)
        if existing is None:
            existing = opener(clone_id)
            engines[clone_id] = existing
        return existing

    return ontology_for


def local_app_scope(
    *,
    workspace_root: Path,
    llm: LLMProviderProtocol,
    tools: ToolRegistryProtocol,
    memory_for: Callable[[str], CrossSessionMemory] | None = None,
    ontology_for: Callable[[str], OntologyEngineProtocol] | None = None,
    persona_registry: PersonaRegistry | None = None,
    llm_override: AgentLLMConfig | None = None,
    global_models: GlobalModels | None = None,
    **host_parts: Any,
) -> AppScope:
    """The app scope of a one-process head (a CLI command): the same clone the app builds.

    Its personas are the installation's; its memory is one store and its rules one engine
    per clone id for the process (`memory_for` / `ontology_for`, else a fresh
    `memory_map` / `ontology_map`); it binds tools
    where its connector is local (§5.1). `global_models` is the command's saved model
    choice, which fills only the slots a persona leaves empty, as Settings does in the app.
    """
    return AppScope.create(
        workspace_root=workspace_root,
        persona_registry=(
            persona_registry
            if persona_registry is not None
            # Not checked against `tools`: a command's registry holds the built-in tools
            # only, so a persona naming an MCP tool would refuse to load, and with it the
            # command -- for a persona it may not even answer as.
            else get_default_persona_registry(workspace_root)
        ),
        memory_for=memory_for if memory_for is not None else memory_map(),
        ontology_for=ontology_for if ontology_for is not None else ontology_map(),
        llm_override=llm_override,
        global_models=global_models,
        llm=llm,
        tools=tools,
        tool_binder=connector_tool_binder(llm),
        **host_parts,
    )


def saved_models(model: str | None) -> GlobalModels | None:
    """The `global_models` of a command whose saved choice names `model`, else `None`."""
    if model is None:
        return None
    return lambda: (model, None)


def _no_read_roots() -> tuple[Path, ...]:
    return ()


@dataclass(frozen=True)
class AppScope:
    """What every clone an app serves shares (§5.9.2, App scope), plus where clone-scope
    data is looked up (the persona registry, the memory map and the rules-engine map).

    Built once by a head and handed to every `build_clone`. `host` holds app-scope parts
    only -- no persona, no memory, no transport -- and is what a peer-call agent starts
    from. A head builds it with `AppScope.create`, so no head constructs a host itself.
    """

    host: HostDependencies
    workspace_root: Path
    persona_registry: PersonaRegistry
    #: Clone id -> that clone's one memory store; `None` builds clones with no memory.
    memory_for: Callable[[str], CrossSessionMemory] | None = None
    #: Clone id -> that clone's one rules engine; `None` builds clones with no engine.
    ontology_for: Callable[[str], OntologyEngineProtocol] | None = None
    global_models: GlobalModels | None = None
    #: Asked on every build, so a folder added in Settings reaches the next clone.
    read_roots: Callable[[], tuple[Path, ...]] = field(default=_no_read_roots)
    #: An installation-wide model override (`--model`), winning over each persona's own.
    llm_override: AgentLLMConfig | None = None
    #: The host a peer-call agent is built from, read per call; `None` means `host`. A head
    #: whose connector can change under a live clone passes one, so the callee follows.
    live_host: Callable[[], HostDependencies] | None = None

    @classmethod
    def create(
        cls,
        *,
        workspace_root: Path,
        persona_registry: PersonaRegistry,
        memory_for: Callable[[str], CrossSessionMemory] | None = None,
        ontology_for: Callable[[str], OntologyEngineProtocol] | None = None,
        global_models: GlobalModels | None = None,
        read_roots: Callable[[], tuple[Path, ...]] | None = None,
        llm_override: AgentLLMConfig | None = None,
        live_host: Callable[[], HostDependencies] | None = None,
        **host_parts: Any,
    ) -> AppScope:
        """A scope over a host holding `host_parts` (bus, llm, tools, tracer, store, ...).

        Refuses the clone-scope host fields: those are `build_clone`'s to set, per clone.
        """
        clone_parts = _CLONE_SCOPE_HOST_FIELDS & host_parts.keys()
        if clone_parts:
            raise TypeError(
                f"{sorted(clone_parts)} are set per clone by build_clone, not on the app scope"
            )
        return cls(
            host=HostDependencies(**host_parts),
            workspace_root=workspace_root,
            persona_registry=persona_registry,
            memory_for=memory_for,
            ontology_for=ontology_for,
            global_models=global_models,
            read_roots=read_roots or _no_read_roots,
            llm_override=llm_override,
            live_host=live_host,
        )

    def app_host(self) -> HostDependencies:
        """The app-scope host, read at the moment of the call (a `host_factory`)."""
        return self.live_host() if self.live_host is not None else self.host

    def with_llm(self, llm: LLMProviderProtocol | None) -> AppScope:
        """This scope with the connector Settings just installed (#1446)."""
        return dataclasses.replace(self, host=dataclasses.replace(self.host, llm=llm))


#: Set per clone by `build_clone`; an app scope carrying one would hand it to every clone.
#: The rules engine is one of them (step 6): one engine on the app scope was the manager's
#: shared engine, which every clone the app built reasoned in.
_CLONE_SCOPE_HOST_FIELDS: Final = frozenset(
    {"memory", "ontology", "persona", "persona_name", "persona_definitions", "a2a_transport"}
)


class _AppOntology:
    """Sentinel: the clone takes its own engine from the app scope (`ontology_for`)."""


APP_ONTOLOGY: Final = _AppOntology()

#: What a build is told to give the clone: an engine, none, or its own from the app scope.
OntologyChoice: TypeAlias = OntologyEngineProtocol | None | _AppOntology


@dataclass(frozen=True)
class BuiltClone:
    """A composed clone, and what its head needs to keep about it."""

    agent: BaseAgent
    persona: PersonaDefinition | None
    #: Whether its (deep, fast) model follows Settings, because nothing named one. Only
    #: those move when Settings changes.
    follows: tuple[bool, bool]


def named_clone_prompt(clone_id: str) -> str:
    """The prompt a clone without a persona speaks as: its own name, then the default."""
    return (
        f"You are {clone_id}, a specialized UClone-X autonomous agent assistant. "
        f"You collaborate with the user, execute tools, and maintain rigorous accuracy."
        f"\n\n{DEFAULT_SYSTEM_PROMPT}"
    )


def with_app_lifecycle_hooks(host: HostDependencies) -> HostDependencies:
    """`host` with the turn lifecycle hooks every UClone-X clone runs with (#1732).

    The agent imports no domain; the story a turn has open moves between its steps only
    because this puts `StoryLifecycleHook` in. Idempotent, so a host that already has it
    is returned as is, and a host's own hooks are kept ahead of it.
    """
    if any(isinstance(hook, StoryLifecycleHook) for hook in host.lifecycle_hooks):
        return host
    return dataclasses.replace(host, lifecycle_hooks=(*host.lifecycle_hooks, StoryLifecycleHook()))


def build_clone(
    app: AppScope,
    *,
    clone_id: str,
    session_id: str,
    persona: str | None = None,
    display_name: str | None = None,
    seat_framing: str = "",
    model_name: str | None = None,
    ontology: OntologyChoice = APP_ONTOLOGY,
    a2a_transport: A2ATransportProtocol | None = None,
    fallback_prompt: str | None = None,
    fallback_llm: AgentLLMConfig | None = None,
    config_update: Mapping[str, Any] | None = None,
) -> BuiltClone:
    """Compose clone `clone_id` in session `session_id`, the same way in every head.

    Shared (§5.9.3): the persona is looked up by `persona` (else by `clone_id`) and
    registered on the agent; its config comes from `agent_config_for_persona`; its model
    is `model_name`, else the app override, else the persona's, with empty slots following
    Settings; its memory is the app's one store for `clone_id`; host binding applies
    whenever the app has a binder; and a persona with `a2a_peers` reaches them.

    Its rules engine is the app's one engine for `clone_id` (`ontology_for`), unless
    `ontology` names one (a test's, or `None` for none). Room-only, passed by the room:
    `seat_framing`, `display_name` and `a2a_transport` (the room's). A clone without a persona speaks as
    `fallback_prompt` (default: one naming the clone, `named_clone_prompt`) on
    `fallback_llm`. `config_update` carries what a CLI command sets
    and no persona holds (isolation, a step budget).
    """
    persona_def = app.persona_registry.get_persona(persona or clone_id)
    host = with_app_lifecycle_hooks(app.host)
    if not isinstance(ontology, _AppOntology):
        host = dataclasses.replace(host, ontology=ontology)
    elif app.ontology_for is not None:
        host = dataclasses.replace(host, ontology=app.ontology_for(clone_id))
    if app.memory_for is not None:
        host = dataclasses.replace(host, memory=app.memory_for(clone_id))
    if persona_def is not None:
        # Registered before the agent seeds its session, so the anchor is composed with
        # this definition in force whichever registry the agent would read on its own.
        host = dataclasses.replace(
            host,
            persona=persona_def.name,
            persona_name=persona_def.name,
            persona_definitions=(persona_def,),
        )

    if app.llm_override is not None:
        llm_config = app.llm_override
    elif persona_def is not None:
        llm_config = persona_def.llm_config
    else:
        llm_config = fallback_llm if fallback_llm is not None else AgentLLMConfig()
    if model_name:
        llm_config = llm_config.model_copy(update={"model_name": model_name})
    follows = (not llm_config.model_name, not llm_config.fast_model)
    if app.global_models is not None:
        deep, fast = app.global_models()
        llm_config = follow_global_models(llm_config, deep, fast)

    name = display_name or (persona_def.name if persona_def is not None else clone_id)
    read_roots = app.read_roots()
    if persona_def is not None:
        config = agent_config_for_persona(
            persona_def,
            agent_id=clone_id,
            name=name,
            seat_framing=seat_framing,
            llm_config=llm_config,
            workspace_dir=app.workspace_root,
            read_roots=read_roots,
        )
    else:
        config = AgentConfig(
            agent_id=clone_id,
            name=name,
            system_prompt=fallback_prompt
            if fallback_prompt is not None
            else named_clone_prompt(clone_id),
            seat_framing=seat_framing,
            llm_config=llm_config,
            workspace_dir=app.workspace_root,
            read_roots=read_roots,
        )
    if config_update:
        config = config.model_copy(update=dict(config_update))

    if (
        a2a_transport is None
        and host.a2a_transport is None
        and persona_def is not None
        and persona_def.a2a_peers
    ):
        a2a_transport = _peer_transport(app, persona_def)
    if a2a_transport is not None:
        host = dataclasses.replace(host, a2a_transport=a2a_transport)

    agent = compose_agent(
        config=config,
        host=host,
        context=AgentContext(
            session_id=session_id,
            agent_id=clone_id,
            workspace_root=app.workspace_root,
        ),
    )
    return BuiltClone(agent=agent, persona=persona_def, follows=follows)


def _peer_transport(app: AppScope, persona: PersonaDefinition) -> A2ATransportProtocol:
    """An in-process transport answering `persona`'s `a2a_peers` (#1659).

    Outside a room this is how a clone reaches its peers; a room passes its own. Each peer
    is answered by a one-off agent from the app-scope host, read per call so a connector
    replaced in Settings reaches the callee too.
    """
    from uclone_x.a2a.in_memory import A2AInMemoryTransport
    from uclone_x.room.a2a_handlers import register_persona_handlers

    transport = A2AInMemoryTransport()
    register_persona_handlers(
        transport,
        host_factory=app.app_host,
        persona_registry=app.persona_registry,
        workspace_root=app.workspace_root,
        llm_config=app.llm_override,
        read_roots=app.read_roots,
        personas=persona.a2a_peers,
        global_models=app.global_models,
    )
    return transport
