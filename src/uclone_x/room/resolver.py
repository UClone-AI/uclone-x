"""The concrete `RoomAgentResolverProtocol`: one live agent per participant, kept apart.

The protocol states the obligation — for two distinct participants, return agents sharing
no session and no ontology — and an obligation stated on an interface is discharged by
whoever implements it. This is the implementation the room ships with, and it discharges
it *structurally*: an agent is built against the ids already stamped on its `Participant`
by `RoomService`, and a session id that two participants somehow share is refused here
rather than colliding later inside `SessionStore.save`, where the error names a revision
precondition and not the roster edit that caused it. The ontology half is the clone's: a
seat's rules engine is its clone's one engine (`AppScope.ontology_for`, keyed by clone id),
so two clones never share one and no seat has one of its own (clone-knowledge-graph step 6).

Lives in the Core, not in the CLI. Composing an agent from a participant is the room's
business logic; `ucx room say` is the shell over it (P8).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.clone_builder import (
    AppScope,
    GlobalModels,
    build_clone,
    follow_global_models,
)
from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.models import AgentLLMConfig, PersonaDefinition
from uclone_x.agent.persona_registry import PersonaRegistry, get_default_persona_registry
from uclone_x.agent.protocols import BaseAgentProtocol
from uclone_x.errors import ParticipantNotResolvableError
from uclone_x.llm.protocols import LLMProviderProtocol
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.ontology.protocols import OntologyEngineProtocol
from uclone_x.room.models import Participant, ParticipantKind

if TYPE_CHECKING:
    from uclone_x.a2a.protocols import A2ATransportProtocol

__all__ = [
    "GlobalModels",
    "RoomAgentResolver",
    "follow_global_models",
    "room_participant_system_prompt",
]


def room_participant_system_prompt(participant: Participant) -> str:
    """The system prompt an agent gets for its seat in a room.

    States that the conversation is shared and that other voices in the span are other
    participants — without which an agent handed `[critic]: TTL is wrong` reads it as the
    user's own words and answers the wrong interlocutor.
    """
    purpose = participant.persona_summary or "contribute your own perspective"
    return (
        f"You are {participant.display_name} ({participant.id}), one participant in a "
        f"shared multi-agent conversation. Your role: {purpose}.\n"
        "Lines you are shown are prefixed with the speaker's id in square brackets; they "
        "are other participants, not all of them addressed to you. Reply with your own "
        "contribution only — do not narrate the conversation, impersonate another "
        "participant, or prefix your reply with your own name."
    )


class RoomAgentResolver:
    """Builds and caches one agent per participant of one room.

    Cached because a room is a conversation: constructing a fresh agent per turn would
    discard the in-memory session between turns and reload it from disk every time, which
    is both slower and a second writer's worth of opportunity to lose an update.

    Cached **by session id**, not by participant id. A participant id is unique within one
    room and this cache is not scoped to one — nothing in the constructor names a room — so
    keying by id made `scout` in room B resolve to room A's live agent, writing one room's
    conversation into the other's session. The session id is already unique per
    `(room, participant)` pair, which is what `RoomService` derives it to be.
    """

    _app: AppScope
    _a2a_transport: A2ATransportProtocol | None
    _agents: dict[str, BaseAgentProtocol]
    _follows: dict[str, tuple[bool, bool]]
    _sessions: dict[str, str]
    _edited_personas: dict[str, tuple[PersonaDefinition, PersonaDefinition | None]]

    def __init__(
        self,
        host: AppScope | HostDependencies,
        *,
        llm_config: AgentLLMConfig | None = None,
        ontology_for: Callable[[str], OntologyEngineProtocol] | None = None,
        memory_factory: Callable[[str], CrossSessionMemory] | None = None,
        persona_registry: PersonaRegistry | None = None,
        workspace_root: Path | None = None,
        read_roots: Callable[[], tuple[Path, ...]] | None = None,
        global_models: GlobalModels | None = None,
        a2a_transport: A2ATransportProtocol | None = None,
    ) -> None:
        """Build seats from `host`: the app scope every clone is built from (§5.9).

        A bare `HostDependencies` is accepted too, with the scope's other parts given as
        keywords. Given an `AppScope`, those keywords are the scope's and are refused here,
        so a room cannot build its seats differently from the chat that shares the scope.
        """
        if isinstance(host, AppScope):
            given = [
                name
                for name, value in (
                    ("llm_config", llm_config),
                    ("memory_factory", memory_factory),
                    ("ontology_for", ontology_for),
                    ("persona_registry", persona_registry),
                    ("workspace_root", workspace_root),
                    ("read_roots", read_roots),
                    ("global_models", global_models),
                )
                if value is not None
            ]
            if given:
                raise TypeError(f"{given} belong to the AppScope, not to the resolver")
            self._app = host
        else:
            if host.ontology is not None:
                # Every seat would be handed that one engine: the shared engine step 6
                # retired. A clone's engine is its own, looked up by its id.
                raise TypeError(
                    "A host carrying an ontology would give every seat one shared engine; "
                    "pass `ontology_for`, which gives each clone its own"
                )
            root = (
                workspace_root.resolve()
                if workspace_root is not None
                else (
                    host.workspace.root.resolve()
                    if host.workspace is not None
                    else Path(os.getenv("UCLONE_WORKSPACE_DIR", os.getcwd())).resolve()
                )
            )
            self._app = AppScope(
                host=host,
                workspace_root=root,
                persona_registry=(
                    persona_registry
                    if persona_registry is not None
                    else get_default_persona_registry(root)
                ),
                #: participant id -> that participant's own cross-session memory store.
                #: Per participant and not per room, the same shape as `ontology_for`:
                #: one store behind two seats makes one agent's recollection readable as
                #: another's. Given none, room agents are composed with no memory at all --
                #: which since #1098 means the tools are neither resolved nor advertised.
                memory_for=memory_factory,
                #: clone id -> that clone's one rules engine, the same shape as memory.
                ontology_for=ontology_for,
                global_models=global_models,
                read_roots=read_roots or (lambda: ()),
                llm_override=llm_config,
            )
        #: The room's transport, handed to every seat so a seat reaches the personas the
        #: room answers (#1558). Room scope, not app scope: a chat clone has none of it.
        self._a2a_transport = a2a_transport
        #: session id -> the live agent built against it. Keyed by the *session* and not
        #: by the participant id, because an id is unique within one room and this cache
        #: is not: `scout` in two rooms is two participants with two derived sessions, and
        #: keying by id handed the second one the first room's live agent — still bound to
        #: the first room's session, so one room's conversation was written into the
        #: other's. Nothing downstream could see it: only one session was ever opened, so
        #: neither the claim below nor the session store's revision precondition fires.
        self._agents: dict[str, BaseAgentProtocol] = {}
        #: session id -> whether that seat's (deep, fast) model follows Settings, because
        #: its persona named none. Only those move when Settings changes.
        self._follows: dict[str, tuple[bool, bool]] = {}
        #: session id -> the seat's model config before defaults were filled in, which the
        #: gateway re-binds when the connections or defaults change (model-gateway §3.3).
        self._own_llm: dict[str, AgentLLMConfig] = {}
        #: participant id -> (the name it is shown by, its own deep model ref), for a seat
        #: whose clone names its own model: a failure of that model names both (§3.6).
        self._pinned: dict[str, tuple[str, str]] = {}
        #: session id -> the participant id it was issued to. The check that turns the
        #: protocol's isolation obligation into a refusal.
        self._sessions: dict[str, str] = {}
        #: persona name -> (its definition as last saved while this resolver was open,
        #: what this resolver's registry gave for it then). A seat built later takes the
        #: saved one only while the registry still gives what it gave at the save (#1904).
        self._edited_personas: dict[str, tuple[PersonaDefinition, PersonaDefinition | None]] = {}

    @property
    def host(self) -> HostDependencies:
        """The host every seat is built with, including a connector replaced since (#1446)."""
        return self._app.host

    @property
    def app(self) -> AppScope:
        """The scope every seat is built from."""
        return self._app

    @property
    def persona_registry(self) -> PersonaRegistry:
        """Where this resolver looks a seat's persona up."""
        return self._app.persona_registry

    @property
    def workspace_root(self) -> Path:
        return self._app.workspace_root

    @property
    def llm_config(self) -> AgentLLMConfig | None:
        """The installation-wide model override, or `None` for each persona's own."""
        return self._app.llm_override

    @property
    def read_roots(self) -> Callable[[], tuple[Path, ...]]:
        return self._app.read_roots

    async def resolve(
        self, participant: Participant, *, one_seat: bool = False
    ) -> BaseAgentProtocol:
        """Return the live agent for `participant`, constructing it on first use.

        `one_seat` drops the multi-agent seat framing (§5.9.3): a clone alone with its
        people is framed as a 1:1 chat is. A built agent whose roster crossed between one
        clone and two is re-framed here, before its next turn.

        Raises:
            ParticipantNotResolvableError: The participant is a human (which has no agent
                behind it), carries no session id, or carries a session id already issued
                to a different participant.
        """
        if participant.kind is not ParticipantKind.AGENT:
            raise ParticipantNotResolvableError(
                f"{participant.id!r} is a {participant.kind.value} participant and has no "
                f"agent behind it; only an agent can be given the floor"
            )
        if not participant.session_id:
            raise ParticipantNotResolvableError(
                f"Participant {participant.id!r} carries no session id. Each agent keeps "
                f"its own session, and an agent with none would share whichever session "
                f"the host's default names."
            )
        claimed_by = self._sessions.get(participant.session_id)
        if claimed_by is not None and claimed_by != participant.id:
            raise ParticipantNotResolvableError(
                f"Participants {claimed_by!r} and {participant.id!r} both claim session "
                f"{participant.session_id!r}. Two writers on one session lose one side's "
                f"turns; the roster, not the resolver, is what needs fixing."
            )

        # One name for the cache key, used by both the read and the write, so that what
        # the cache is keyed *by* is a single decision rather than two lines that have to
        # keep agreeing.
        cache_key = participant.session_id
        framing = "" if one_seat else room_participant_system_prompt(participant)
        cached = self._agents.get(cache_key)
        if cached is not None:
            if isinstance(cached, BaseAgent):
                cached.set_read_roots(self._app.read_roots())
                cached.set_seat_framing(framing)
            return cached

        # Everything else is built as a 1:1 chat builds it (owner ruling 2026-09-27): the
        # room adds only what a room has -- the seat's framing (none in a one-seat room,
        # §5.9.3), its display name and the room's transport. Its rules engine, like its
        # memory, is its clone's own. The framing goes to the agent as its
        # own field and the agent composes the prompt (`compose_identity_prompt`).
        built = build_clone(
            self._app,
            clone_id=participant.id,
            session_id=participant.session_id,
            display_name=participant.display_name or participant.id,
            seat_framing=framing,
            a2a_transport=self._a2a_transport,
        )
        agent = built.agent
        follows = built.follows
        # A seat built from a registry read before the edit still takes it.
        self._take_persona_edit(agent)
        # Resume before the first turn: the room outlives the process that drives it, so an
        # agent that did not hydrate would answer a continuing conversation from a blank
        # history and then persist that over the record.
        agent.hydrate_session()

        self._agents[cache_key] = agent
        self._follows[cache_key] = follows
        if built.own_llm_config is not None:
            self._own_llm[cache_key] = built.own_llm_config
        if built.pinned_ref is not None:
            self._pinned[participant.id] = (
                participant.display_name or participant.id,
                built.pinned_ref,
            )
        else:
            self._pinned.pop(participant.id, None)
        self._sessions[participant.session_id] = participant.id
        return agent

    def pinned_model(self, participant_id: str) -> tuple[str, str] | None:
        """``(shown name, own model ref)`` when this seat's clone names its own model."""
        return self._pinned.get(participant_id)

    def models_changed(self) -> None:
        """Re-bind every built seat after the connections or default models changed.

        The gateway resolves each seat's own config again (its own ref, else the default
        now in Settings) and the seat takes the connector and model ids from it at once, so
        a model chosen in Settings reaches an open conversation (#1446). A seat whose clone
        names its own model keeps it.
        """
        gateway = self._app.gateway
        if gateway is None:
            return
        for cache_key, agent in self._agents.items():
            own = self._own_llm.get(cache_key)
            # Every agent cached here came from `compose_agent`; the protocol it is held as
            # does not declare the reload, and a fake that is not a `BaseAgent` has no
            # connector to swap.
            if own is None or not isinstance(agent, BaseAgent):
                continue
            seat = gateway.bind(own)
            # Exactly what the gateway resolved now, `None` included: a cleared default or one
            # moved to another connection must not leave the old connector or fast id behind.
            agent.rebind_llm(
                seat.llm,
                model_name=seat.llm_config.model_name,
                fast_model=seat.llm_config.fast_model,
            )

    def replace_llm(self, llm: LLMProviderProtocol | None) -> None:
        """Answer with `llm` from now on, for a resolver with no gateway (a test's host).

        A seat whose persona names its own model keeps it; a seat that follows the given
        global models takes them with the connector.
        """
        self._app = self._app.with_llm(llm)
        global_models = self._app.global_models
        deep, fast = global_models() if global_models is not None else (None, None)
        for cache_key, agent in self._agents.items():
            if isinstance(agent, BaseAgent):
                deep_follows, fast_follows = self._follows.get(cache_key, (False, False))
                agent.hot_reload_llm(
                    llm,
                    model_name=deep if deep_follows else None,
                    fast_model=(fast or deep) if fast_follows else None,
                )

    def persona_edited(self, persona: PersonaDefinition) -> int:
        """Hold `persona` as saved; return how many built seats speak as it.

        Nothing is changed on a live agent here, nor at `resolve`: a seat may be in the
        middle of a turn, and compaction resolves a seat while another seat's turn runs
        (#1899 review). Each built seat speaking as `persona` is handed the edit to hold
        (`BaseAgent.stage_persona_edit`), and takes it at its own next turn start, under
        its turn lock, where the turn opens a `persona_edited` epoch for it.
        """
        self._edited_personas[persona.name] = (
            persona,
            self._app.persona_registry.get_persona(persona.name),
        )
        seats = [
            agent
            for agent in self._agents.values()
            if isinstance(agent, BaseAgent) and agent.persona == persona.name
        ]
        for agent in seats:
            agent.stage_persona_edit(persona)
        return len(seats)

    def _take_persona_edit(self, agent: BaseAgent) -> None:
        """Put the last saved definition of a *newly built* seat's persona in force.

        The seat was built from a registry read before the edit, and has run no turn, so
        there is no turn to interrupt and no epoch to mark. `define_persona` recomputes the
        tool scope; the prompt is composed from the definition on every turn. What is fixed
        at construction -- the persona's model -- follows the definition the build read.

        The held copy is used only while this resolver's registry still gives what it gave
        when the edit was saved, i.e. while the registry lags the save. Once the registry
        gives something else, it changed after the save by some other path, and the build
        already read that newer definition: the held copy is dropped rather than put over
        it (#1904).
        """
        name = agent.persona
        held = self._edited_personas.get(name) if name else None
        if held is None:
            return
        edited, registry_at_save = held
        if self._app.persona_registry.get_persona(edited.name) != registry_at_save:
            del self._edited_personas[edited.name]
            return
        if agent.persona_definition != edited:
            agent.define_persona(edited)

    def live_agents(self) -> tuple[BaseAgent, ...]:
        """Every seat agent this resolver has built and still holds, for the health reads."""
        return tuple(agent for agent in self._agents.values() if isinstance(agent, BaseAgent))

    def seated_agent_ids(self) -> frozenset[str]:
        """The ids of the agents this resolver has built and still holds.

        A *seat that is filled*, not a roster entry: a participant listed in the room but
        never given the floor has no agent behind it and is not here. That is the
        distinction the caller needs, because "is this clone running" is a question about
        instances and a roster answers it `yes` for a room nobody has spoken in.

        Read from `_sessions` rather than from `_agents`, whose key is the session id.
        Both are written on the same resolve, so they hold the same seats; this one is
        already keyed the way the answer is shaped.
        """
        return frozenset(self._sessions.values())

    def live_agent(self, session_id: str) -> BaseAgentProtocol | None:
        """The agent already built for this session, or `None` if none has been.

        A *read*, deliberately separate from `resolve`: the callers are the conversation's
        history controls, and asking "how full is this seat's context" or "cut this seat
        back" must not be the act that constructs an agent. `resolve` loads a session from
        disk, builds an ontology engine and a memory store, and claims the session id — work
        a reader has no business causing, and which would report a freshly built seat as
        holding nothing rather than reporting that nobody has spoken.

        It exists at all because a seated agent is cached *here* and never written into
        `AgentSessionManager`'s own map, so `AgentSessionManager.get_agent` answers `None`
        for every room seat. A history control that believed it took the no-agent branch:
        writing a cleared session straight to the store while a live agent still held the
        old messages, which that agent then persisted back over the write.
        """
        return self._agents.get(session_id)
