"""The concrete `RoomAgentResolverProtocol`: one live agent per participant, kept apart.

The protocol states the obligation — for two distinct participants, return agents sharing
no session and no ontology — and an obligation stated on an interface is discharged by
whoever implements it. This is the implementation the room ships with, and it discharges
it *structurally*: an agent is built against the ids already stamped on its `Participant`
by `RoomService`, and a session id — or an ontology namespace — that two participants
somehow share is refused here rather than colliding later inside `SessionStore.save`, where
the error names a revision precondition and not the roster edit that caused it, or inside a
knowledge graph, where nothing names it at all.

Lives in the Core, not in the CLI. Composing an agent from a participant is the room's
business logic; `ucx room say` is the shell over it (P8).
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from pathlib import Path

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.bootstrap import agent_config_for_persona
from uclone_x.agent.composition import HostDependencies, compose_agent
from uclone_x.agent.models import (
    DEFAULT_SYSTEM_PROMPT,
    AgentConfig,
    AgentContext,
    AgentLLMConfig,
)
from uclone_x.agent.persona_registry import PersonaRegistry, get_default_persona_registry
from uclone_x.agent.protocols import BaseAgentProtocol
from uclone_x.errors import ParticipantNotResolvableError, SeatKnowledgeUnreadableError
from uclone_x.llm.protocols import LLMProviderProtocol
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.ontology.protocols import OntologyEngineProtocol
from uclone_x.room.knowledge import KnowledgeLoad, SeatKnowledgeProtocol
from uclone_x.room.models import Participant, ParticipantKind

__all__ = ["RoomAgentResolver", "room_participant_system_prompt"]


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

    _host: HostDependencies
    _llm_config: AgentLLMConfig | None
    _ontology_factory: Callable[[str], OntologyEngineProtocol] | None
    _memory_factory: Callable[[str], CrossSessionMemory] | None
    _knowledge: SeatKnowledgeProtocol | None
    _workspace_root: Path
    _persona_registry: PersonaRegistry
    _agents: dict[str, BaseAgentProtocol]
    _sessions: dict[str, str]
    _namespaces: dict[str, str]

    def __init__(
        self,
        host: HostDependencies,
        *,
        llm_config: AgentLLMConfig | None = None,
        ontology_factory: Callable[[str], OntologyEngineProtocol] | None = None,
        memory_factory: Callable[[str], CrossSessionMemory] | None = None,
        persona_registry: PersonaRegistry | None = None,
        workspace_root: Path | None = None,
        read_roots: Callable[[], tuple[Path, ...]] | None = None,
        knowledge: SeatKnowledgeProtocol | None = None,
    ) -> None:
        self._host = host
        #: Where each seat's knowledge was saved after its turns (#1367). Loaded into the
        #: engine `ontology_factory` builds, before the seat's first turn, so a seat resumed
        #: after a restart continues from what it knew rather than from an empty graph --
        #: the knowledge half of what `hydrate_session` does for the session.
        self._knowledge = knowledge
        #: Asked on every resolve rather than once, so a folder added in Settings reaches
        #: a seat that is already built.
        self._read_roots: Callable[[], tuple[Path, ...]] = read_roots or (lambda: ())
        self._llm_config = llm_config
        self._ontology_factory = ontology_factory
        #: participant id -> that participant's own cross-session memory store. Per
        #: participant and not per room, the same shape as `ontology_factory`: a store is
        #: what `record_memory_fact` writes into, and one store behind two seats makes one
        #: agent's recollection readable as another's. Given none, room agents are composed
        #: with no memory at all -- which since #1098 means the tools are neither resolved
        #: nor advertised, rather than silently reaching the first store composed anywhere.
        self._memory_factory = memory_factory
        self._workspace_root = (
            workspace_root.resolve()
            if workspace_root is not None
            else (
                self._host.workspace.root.resolve()
                if self._host.workspace is not None
                else Path(os.getenv("UCLONE_WORKSPACE_DIR", os.getcwd())).resolve()
            )
        )
        self._persona_registry = (
            persona_registry
            if persona_registry is not None
            else get_default_persona_registry(self._workspace_root)
        )
        #: session id -> the live agent built against it. Keyed by the *session* and not
        #: by the participant id, because an id is unique within one room and this cache
        #: is not: `scout` in two rooms is two participants with two derived sessions, and
        #: keying by id handed the second one the first room's live agent — still bound to
        #: the first room's session, so one room's conversation was written into the
        #: other's. Nothing downstream could see it: only one session was ever opened, so
        #: neither the claim below nor the session store's revision precondition fires.
        self._agents: dict[str, BaseAgentProtocol] = {}
        #: session id -> the participant id it was issued to. The check that turns the
        #: protocol's isolation obligation into a refusal.
        self._sessions: dict[str, str] = {}
        #: ontology namespace -> the participant id it was issued to. The same check for
        #: the other half of the obligation: G4 is a property of the *namespace*, not of
        #: the engine object, so two engines built over one IRI still merge the concepts
        #: P7 keeps apart and `is not` proves nothing about them.
        self._namespaces: dict[str, str] = {}

    async def resolve(self, participant: Participant) -> BaseAgentProtocol:
        """Return the live agent for `participant`, constructing it on first use.

        Raises:
            ParticipantNotResolvableError: The participant is a human (which has no agent
                behind it), carries no session id, carries a session id or an ontology
                namespace already issued to a different participant, or carries a namespace
                this resolver cannot honour because it has no ontology factory and the host
                would hand every agent its one shared engine.
            SeatKnowledgeUnreadableError: The seat's saved knowledge is there and cannot be
                read (a `ParticipantNotResolvableError`).
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
        cached = self._agents.get(cache_key)
        if cached is not None:
            if isinstance(cached, BaseAgent):
                cached.set_read_roots(self._read_roots())
            return cached

        host = self._host
        if participant.ontology_namespace:
            namespace_owner = self._namespaces.get(participant.ontology_namespace)
            if namespace_owner is not None and namespace_owner != participant.id:
                raise ParticipantNotResolvableError(
                    f"Participants {namespace_owner!r} and {participant.id!r} both claim "
                    f"ontology namespace {participant.ontology_namespace!r}. One namespace "
                    f"for two agents merges what each induced from its own experience, "
                    f"which is the per-agent grounding P7 requires and not a room feature."
                )
            if self._ontology_factory is None and host.ontology is not None:
                raise ParticipantNotResolvableError(
                    f"Participant {participant.id!r} carries ontology namespace "
                    f"{participant.ontology_namespace!r}, but this resolver was built with "
                    f"no ontology factory and the host already carries an engine. Every "
                    f"participant would be handed that one engine and induce into one "
                    f"graph — the merge the per-participant namespace exists to prevent. "
                    f"Give the resolver an `ontology_factory`, or a host without an "
                    f"ontology if the room's agents are to run without one."
                )
            if self._ontology_factory is not None:
                engine = self._ontology_factory(participant.ontology_namespace)
                if self._knowledge is not None:
                    try:
                        loaded = self._knowledge.load_into(participant.session_id, engine)
                    except SeatKnowledgeUnreadableError as exc:
                        # Unreadable and not set aside: refused rather than built over an
                        # empty engine, which would be saved over the record after the turn.
                        # The message is shown in the conversation, so it names the clone
                        # and nothing else; the path and cause stay on the exception.
                        raise SeatKnowledgeUnreadableError(
                            f"{participant.display_name}'s knowledge record for this "
                            f"conversation could not be read, and could not be set aside.",
                            path=exc.path,
                            cause=exc.cause,
                        ) from exc
                    if loaded is KnowledgeLoad.SET_ASIDE:
                        # The record is kept under another name; the engine it failed in may
                        # hold part of it, and the seat starts over from nothing instead.
                        fresh = self._ontology_factory
                        engine = fresh(participant.ontology_namespace)
                host = self._replace_ontology(host, engine)

        if self._memory_factory is not None:
            host = dataclasses.replace(host, memory=self._memory_factory(participant.id))

        # Check for persona definition from registry (matching participant.persona or participant.id)
        persona_lookup = participant.persona or participant.id
        persona_def = self._persona_registry.get_persona(persona_lookup)

        # The seat's framing goes to the agent as its own field and the agent composes the
        # prompt (`compose_identity_prompt`). Composing it here and handing it in after
        # construction left the stored anchor without the framing the turn then sent.
        seat_framing = room_participant_system_prompt(participant)

        if persona_def is not None:
            # Registered on the agent before it seeds the session, so the anchor is
            # composed with this definition in force; the registry here may not be the
            # one the agent would read on its own.
            host = dataclasses.replace(
                host,
                persona=persona_def.name,
                persona_name=persona_def.name,
                persona_definitions=(persona_def,),
            )

        if self._llm_config is not None:
            resolved_llm_config = self._llm_config
        elif persona_def is not None:
            resolved_llm_config = persona_def.llm_config
        else:
            resolved_llm_config = AgentLLMConfig()

        display_name = participant.display_name or participant.id
        read_roots = self._read_roots()
        if persona_def is not None:
            # Built where the chat head builds it, so a seat is given its tools by the same
            # rule as a 1:1 chat: the persona's list is resolved by the agent -- from the
            # definition registered on it through `persona_definitions` above -- and not
            # copied in as the operator's list, which would win over every later edit
            # (#1448). No registry proxy either: one fixed here would hide a tool an edit
            # adds, and the agent's own scope already filters what it offers and refuses
            # what it withholds.
            config = agent_config_for_persona(
                persona_def,
                agent_id=participant.id,
                name=display_name,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                seat_framing=seat_framing,
                llm_config=resolved_llm_config,
                workspace_dir=self._workspace_root,
                read_roots=read_roots,
            )
        else:
            config = AgentConfig(
                agent_id=participant.id,
                name=display_name,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                seat_framing=seat_framing,
                llm_config=resolved_llm_config,
                workspace_dir=self._workspace_root,
                read_roots=read_roots,
            )
        agent = compose_agent(
            config=config,
            host=host,
            context=AgentContext(
                session_id=participant.session_id,
                agent_id=participant.id,
                workspace_root=self._workspace_root,
            ),
        )
        # Resume before the first turn: the room outlives the process that drives it, so an
        # agent that did not hydrate would answer a continuing conversation from a blank
        # history and then persist that over the record.
        agent.hydrate_session()

        self._agents[cache_key] = agent
        self._sessions[participant.session_id] = participant.id
        if participant.ontology_namespace:
            self._namespaces[participant.ontology_namespace] = participant.id
        return agent

    def replace_llm(self, llm: LLMProviderProtocol | None) -> None:
        """Answer with `llm` from now on: in every seat already built, and in any built later.

        The host is held for the resolver's lifetime, so a connector replaced in Settings
        reached neither half until the room was reopened after a restart (#1446). A seat's
        model name is its persona's, so only the connector is swapped here.
        """
        self._host = dataclasses.replace(self._host, llm=llm)
        for agent in self._agents.values():
            # Every agent cached here came from `compose_agent`; the protocol it is held as
            # does not declare the reload, and a fake that is not a `BaseAgent` has no
            # connector to swap.
            if isinstance(agent, BaseAgent):
                agent.hot_reload_llm(llm)

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

    @staticmethod
    def _replace_ontology(
        host: HostDependencies, ontology: OntologyEngineProtocol
    ) -> HostDependencies:
        """A copy of `host` carrying this participant's own ontology engine (P7, G4)."""
        return dataclasses.replace(host, ontology=ontology)
