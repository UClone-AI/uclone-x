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
import hashlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TypeAlias

import yaml
from pydantic import ValidationError

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.bootstrap import agent_config_for_persona
from uclone_x.agent.clone_store import (
    CloneStoreReport,
    append_report,
    read_imports,
    record_import,
)
from uclone_x.agent.composition import HostDependencies, compose_agent
from uclone_x.agent.models import (
    DEFAULT_SYSTEM_PROMPT,
    AgentConfig,
    AgentContext,
    AgentLLMConfig,
    PersonaDefinition,
)
from uclone_x.agent.persona_avatar import PersonaAvatarStore
from uclone_x.agent.persona_registry import PersonaRegistry, get_default_persona_registry
from uclone_x.agent.persona_store import DEFAULT_PERSONA_NAME
from uclone_x.agent.tools_module import select_tools_module
from uclone_x.browser.turn import BrowserTurnHook
from uclone_x.core.agent_home import (
    ONTOLOGY_FILE_NAME,
    AgentHome,
    AgentHomeError,
    clone_handles,
    clone_root_lock,
    is_agent_id,
    peer_handles,
    seat_id_for,
)
from uclone_x.extensions import extension_lifecycle_hooks
from uclone_x.memory.store import CrossSessionMemory, default_cross_session_memory
from uclone_x.ontology.engine import OntologyEngine
from uclone_x.ontology.protocols import OntologyEngineProtocol
from uclone_x.tools.tool_binder import ToolBinder, tool_binder_for

if TYPE_CHECKING:
    from uclone_x.a2a.protocols import A2ATransportProtocol
    from uclone_x.agent.protocols import TurnLifecycleHookProtocol
    from uclone_x.llm.gateway import ModelGateway
    from uclone_x.llm.protocols import LLMProviderProtocol
    from uclone_x.tools.protocols import ToolRegistryProtocol

logger = logging.getLogger(__name__)

__all__ = [
    "APP_ONTOLOGY",
    "AppScope",
    "BuiltClone",
    "check_tools_module",
    "CLONE_ONTOLOGY_ROOT",
    "GlobalModels",
    "OntologyChoice",
    "build_clone",
    "clone_namespace",
    "clone_ontology",
    "connector_tool_binder",
    "follow_global_models",
    "import_repository_ontologies",
    "local_app_scope",
    "memory_map",
    "named_clone_prompt",
    "ontology_map",
    "provider_tool_binder",
    "command_gateway",
    "saved_models",
    "with_app_lifecycle_hooks",
]


GlobalModels = Callable[[], tuple[str | None, str | None]]
"""Returns ``(deep, fast)`` bare model ids for one connector, at the moment of the call.

For a head with one connector and no model gateway (a test's host). A head with a gateway
(`AppScope.gateway`) resolves refs through it instead (model-gateway §3.3).
"""


def follow_global_models(own: AgentLLMConfig, deep: str | None, fast: str | None) -> AgentLLMConfig:
    """``own`` with each empty model slot filled from the global ``deep`` and ``fast``.

    The rule for one connector with no gateway: a model the persona names is kept, and a
    slot it leaves empty follows the given ones. An empty fast model means the deep one.
    With a gateway, `ModelGateway.bind` applies the same rule to refs.
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

    Two stores over one file no longer lose each other's writes (each write is one sqlite
    transaction), but a store holds the clone's embedding index, so a process keeps one map
    and hands it to every build rather than build an index per caller.
    """
    stores: dict[str, CrossSessionMemory] = {}

    def memory_for(clone_id: str) -> CrossSessionMemory:
        # Keyed by id: a handle is resolved first, so one clone never has two stores.
        key = seat_id_for(clone_id)
        existing = stores.get(key)
        if existing is None:
            existing = opener(key)
            stores[key] = existing
        return existing

    return memory_for


#: Where every clone's rules engine is named: `<root>/<clone id>` (§3.1, Q6 #1654).
CLONE_ONTOLOGY_ROOT: Final = "https://uclone-x.ai/ontology/clones"

#: In a repository: the rules `ucx ontology` kept per handle before they moved into the
#: clone's own directory (clone-data-scopes §3.8 step 5). Read only by the import.
REPOSITORY_ONTOLOGY_DIR: Final = "ontology"


def clone_namespace(clone_id: str) -> str:
    """The namespace of clone `clone_id`'s rules engine: one per clone, in every conversation."""
    return f"{CLONE_ONTOLOGY_ROOT}/{clone_id}"


def clone_ontology(clone_id: str) -> OntologyEngineProtocol:
    """The rules engine of clone `clone_id`, in the clone's own namespace, loaded from its
    `ontology.yaml` (which `ucx ontology` writes) when it has one.

    It holds the rules a person gives the clone and nothing learned in a conversation: what
    the clone learned is its facts, in its memory, and what follows from them under these
    rules is worked out on read and never kept (clone-knowledge-graph §3.1). A key no clone
    directory backs (`seat_id_for` keeps a name no clone carries) gets an empty engine.
    A file that cannot be read or parsed is logged and the clone starts with an empty
    engine, as the import treats the same file: a turn never fails on it (#2134).
    """
    engine = OntologyEngine(agent_id=clone_id, namespace_iri=clone_namespace(clone_id))
    if is_agent_id(clone_id):
        path = AgentHome.for_clone(clone_id).ontology_path
        try:
            engine.load_from_yaml(path)
        except (OSError, UnicodeDecodeError, yaml.YAMLError, ValidationError) as exc:
            # The type only, as the import reports it: the message quotes the file (#2136).
            logger.warning(
                "%s was not loaded, so the clone has no rules: %s", path, type(exc).__name__
            )
            return OntologyEngine(agent_id=clone_id, namespace_iri=clone_namespace(clone_id))
    return engine


def _repository_ontologies(directory: Path | None) -> dict[str, Path]:
    """`<directory>/ontology/<handle>.yaml` by handle; the old `default.yaml` is `clone`'s."""
    folder = directory / REPOSITORY_ONTOLOGY_DIR if directory is not None else None
    if folder is None or not folder.is_dir():
        return {}
    files: dict[str, Path] = {}
    for source in sorted(folder.glob("*.yaml")):
        if source.stem == "default":  # the CLI's default agent until 2026-09-27
            files.setdefault(DEFAULT_PERSONA_NAME, source)
        else:
            files[source.stem] = source  # `clone.yaml` wins over `default.yaml`
    return files


def _merge_rules(held: OntologyEngine, offered: OntologyEngine) -> tuple[int, list[str]]:
    """Add to `held` what `offered` has and it lacks; name what both have but differently."""
    added = 0
    conflicts: list[str] = []
    for concept in offered.list_concepts():
        mine = held.get_concept(concept.name)
        if mine is None:
            held.register_entity(concept)
            added += 1
        elif mine != concept:
            conflicts.append(f"concept {concept.name}")
    for axiom in offered.list_axioms():
        own_axiom = held.get_axiom(axiom.name)
        if own_axiom is None:
            held.register_axiom(axiom)
            added += 1
        elif own_axiom != axiom:
            conflicts.append(f"axiom {axiom.name}")
    for relation in offered.list_relations():
        key = (relation.source_entity, relation.predicate, relation.target_entity)
        same = [
            r
            for r in held.list_relations()
            if (r.source_entity, r.predicate, r.target_entity) == key
            and r.is_directed == relation.is_directed
        ]
        if not same:
            held.register_relation(relation)
            added += 1
        elif same[0] != relation:
            conflicts.append(f"relation {' -> '.join(key)}")
    return added, conflicts


def import_repository_ontologies(
    directory: Path | None,
    *,
    root: Path | None = None,
    report: CloneStoreReport | None = None,
) -> CloneStoreReport:
    """§3.8 step 5: merge `<directory>/ontology/<handle>.yaml` into that clone's rules.

    Once per (file, digest), recorded in the clone's directory as a workspace persona is:
    a file that changes is merged again, and what the clone already holds is skipped. An
    element the clone holds differently is reported and not applied. A file naming no
    clone, or one two clones claim, is left alone.
    """
    report = report if report is not None else CloneStoreReport()
    files = _repository_ontologies(directory)
    if not files:
        return report
    try:
        with clone_root_lock(root) as base:
            claims = clone_handles(base)
            for handle, source in files.items():
                owners = claims.get(handle, ())
                if len(owners) != 1:
                    continue
                folder = base / owners[0]
                source_key = str(source.resolve())
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                if read_imports(folder).get(source_key) == digest:
                    continue
                target = folder / ONTOLOGY_FILE_NAME
                offered = OntologyEngine()
                held = OntologyEngine(agent_id=owners[0], namespace_iri=clone_namespace(owners[0]))
                try:
                    offered.load_from_yaml(source)
                    held.load_from_yaml(target)
                except (OSError, yaml.YAMLError, ValidationError) as exc:
                    report.add(
                        f"{source} was not imported into clone '{handle}': {type(exc).__name__}."
                    )
                    continue
                added, conflicts = _merge_rules(held, offered)
                if added:
                    held.save_to_yaml(target)
                record_import(folder, Path(source_key), digest)
                line = f"imported {added} rule(s) from {source} into clone '{handle}' (agents/{folder.name})."
                if conflicts:
                    line += f" Kept the clone's own {', '.join(conflicts)}; the file differs."
                report.add(line)
            if report.lines:
                append_report(base, report)
    except (AgentHomeError, OSError) as exc:  # as `ensure_clone_store`: a start goes on
        report.add(f"the repository's ontology files were not imported: {exc}")
    return report


def ontology_map(
    opener: Callable[[str], OntologyEngineProtocol] = clone_ontology,
) -> Callable[[str], OntologyEngineProtocol]:
    """A get-or-create map from clone id to that clone's one rules engine, as `memory_map`.

    One engine per clone and not per seat: a clone seated in two conversations reasons
    under one set of rules, and two clones never share an engine (§3.1, step 6).
    """
    engines: dict[str, OntologyEngineProtocol] = {}

    def ontology_for(clone_id: str) -> OntologyEngineProtocol:
        key = seat_id_for(clone_id)  # by id, as `memory_map`
        existing = engines.get(key)
        if existing is None:
            existing = opener(key)
            engines[key] = existing
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
    gateway: ModelGateway | None = None,
    **host_parts: Any,
) -> AppScope:
    """The app scope of a one-process head (a CLI command): the same clone the app builds.

    Its personas are the installation's; its memory is one store and its rules one engine
    per clone id for the process (`memory_for` / `ontology_for`, else a fresh
    `memory_map` / `ontology_map`); it binds tools
    where its connector is local (§5.1). `global_models` is the command's saved model
    choice, which fills only the slots a persona leaves empty, as Settings does in the app.
    `gateway` serves a persona that names its own model ref; the command's connector serves
    every slot that follows the default (`DefaultBinding`).
    """
    registry = (
        persona_registry
        if persona_registry is not None
        # Not checked against `tools`: a command's registry holds the built-in tools
        # only, so a persona naming an MCP tool would refuse to load, and with it the
        # command -- for a persona it may not even answer as.
        else get_default_persona_registry(workspace_root)
    )
    # A CLI start in a repository with the old `ontology/<handle>.yaml` (§3.8 step 5).
    import_repository_ontologies(workspace_root)
    return AppScope.create(
        workspace_root=workspace_root,
        persona_registry=registry,
        memory_for=memory_for if memory_for is not None else memory_map(),
        ontology_for=ontology_for if ontology_for is not None else ontology_map(),
        llm_override=llm_override,
        global_models=global_models,
        gateway=gateway,
        llm=llm,
        tools=tools,
        tool_binder=connector_tool_binder(llm),
        **host_parts,
    )


def command_gateway(llm: LLMProviderProtocol, model: str | None) -> ModelGateway:
    """The gateway of a terminal command: its own connector answers the default slots.

    The command resolved `llm` and `model` by its own precedence (a flag, a variable, the
    saved default); a persona that names its own model ref is served from that ref's
    connection in the session root's settings file (model-gateway §3.4).
    """
    from uclone_x.llm.gateway import DefaultBinding, ModelGateway

    return ModelGateway(default_binding=DefaultBinding(llm, model))


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
    #: Resolves each clone's model ref to a connector (model-gateway §3.3). With one, every
    #: clone gets its own connector from its resolved ref and `global_models` is not read.
    gateway: ModelGateway | None = None

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
        gateway: ModelGateway | None = None,
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
            gateway=gateway,
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
    {
        "memory",
        "ontology",
        "persona",
        "persona_name",
        "persona_definitions",
        "a2a_transport",
        "avatar_present",
    }
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
    #: Its model config before any default was filled in: what a gateway re-binds when the
    #: connections or defaults change.
    own_llm_config: AgentLLMConfig | None = None
    #: Its own deep model ref, when it names one (`None` follows the system default).
    pinned_ref: str | None = None


def named_clone_prompt(clone_id: str) -> str:
    """The prompt a clone without a persona speaks as: its own name, then the default."""
    return (
        f"You are {clone_id}, a specialized UClone-X autonomous agent assistant. "
        f"You collaborate with the user, execute tools, and maintain rigorous accuracy."
        f"\n\n{DEFAULT_SYSTEM_PROMPT}"
    )


def with_app_lifecycle_hooks(host: HostDependencies) -> HostDependencies:
    """`host` with the turn lifecycle hooks every UClone-X clone runs with (#1732).

    The agent imports no domain, and neither does this: the extensions' hooks come first
    (`extensions.extension_lifecycle_hooks`, #2205). The story extension's are why the
    story a turn has open moves between its steps, and why a Writer turn gets its next
    scene in its context and reply lines at its end (#1732, #1808). Then the core's own.
    Idempotent: a hook of a type the host already has is not added again, and a host's own
    hooks are kept ahead of them.
    """
    held = host.lifecycle_hooks
    # Typed as the protocol the agent calls, so a hook missing `after_tool_step` fails
    # the type check here rather than every tool step of every turn (#2159).
    added: list[TurnLifecycleHookProtocol] = []
    for hook in extension_lifecycle_hooks():
        if not any(isinstance(have, type(hook)) for have in (*held, *added)):
            added.append(hook)
    if not any(isinstance(hook, BrowserTurnHook) for hook in held):
        added.append(BrowserTurnHook())
    if not added:
        return host
    return dataclasses.replace(host, lifecycle_hooks=(*held, *added))


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
            avatar_present=_avatar_lookup(app.persona_registry, persona_def.name),
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
    own_llm_config = llm_config
    pinned_ref: str | None = None
    if app.gateway is not None:
        # The clone's own ref, else the default, each on its own connector (§3.4). A ref
        # that cannot be served gets a connector that refuses in plain words (§3.6).
        seat = app.gateway.bind(llm_config)
        llm_config = seat.llm_config
        pinned_ref = seat.pinned_ref
        if seat.llm is not None:
            host = dataclasses.replace(host, llm=seat.llm)
    elif app.global_models is not None:
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
    return BuiltClone(
        agent=agent,
        persona=persona_def,
        follows=follows,
        own_llm_config=own_llm_config,
        pinned_ref=pinned_ref,
    )


def check_tools_module(app: AppScope, clone_id: str, persona: str | None = None) -> None:
    """Refuse, before a head serves, a clone whose tools module setting is unknown (#2188).

    The same rule `BaseAgent` applies when a seat is built (`select_tools_module` over the
    clone's setting and the provider), asked early, so a server that builds its seats per
    connection says so at start, in the refusal's own plain words.

    Raises:
        UnknownToolsModuleError: The clone's setting names a module this build lacks.
    """
    persona_def = app.persona_registry.get_persona(persona or clone_id)
    select_tools_module(
        persona_def.tools_module if persona_def is not None else None,
        str(getattr(app.host.llm, "provider_name", "") or ""),
    )


def _avatar_lookup(registry: PersonaRegistry, persona: str) -> Callable[[], bool]:
    """Whether `persona` has a picture, asked at the moment `show_self` reports it."""
    return lambda: PersonaAvatarStore(registry).find(persona) is not None


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
        # Stored as clone ids, answered by handle (clone-data-scopes §3.3).
        personas=peer_handles(persona.a2a_peers),
        global_models=app.global_models,
        gateway=app.gateway,
    )
    return transport
