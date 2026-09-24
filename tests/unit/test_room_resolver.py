"""Tests for `RoomAgentResolver` — the runtime half of the room's isolation obligation.

`RoomAgentResolverProtocol` states it plainly: for two distinct participants the resolver
must return agents that **share no session and no ontology**. Design §7.3 recorded that
half as still to verify — the orchestrator's tests pin that it resolves each participant by
its own ids, and nothing pinned what the resolver then does with them. This file is that
missing half, and it exists because `src/uclone_x/room/resolver.py` shipped with no test
file at all.

What it pins, in order of what it would cost to get wrong:

* **A session is never handed to two agents.** Not by a second participant claiming it (the
  resolver already refused that), and not by the cache handing one room's agent back for a
  participant of a *different* room that happens to share an id. The second is the one that
  was silent: two live agents writing one `SessionState`, refused later by the session
  store's revision precondition with a message about revisions rather than about the room.
* **A namespace is never handed to two agents.** G4 is not a property of the engine
  *object*; it is a property of the namespace an engine induces into. Two engines built
  over one IRI merge exactly the concepts P7 keeps apart.
* **An agent that carries a namespace is never given the host's shared ontology.** Building
  the resolver without an `ontology_factory` used to mean every participant in the room got
  the *same* engine — the one on `HostDependencies` — with its own namespace ignored.
* **A human has no agent behind it**, and neither does an agent carrying no session.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.models import AgentLLMConfig, PersonaDefinition
from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.session import SessionStore
from uclone_x.engine.event_bus import EventBus
from uclone_x.errors import ParticipantNotResolvableError
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.ontology.engine import OntologyEngine
from uclone_x.ontology.protocols import OntologyEngineProtocol
from uclone_x.room.models import Participant, ParticipantKind
from uclone_x.room.resolver import RoomAgentResolver, room_participant_system_prompt
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.registry import ToolRegistry


@pytest.fixture
def host_factory(tmp_path: Path) -> Callable[..., HostDependencies]:
    """Build a host whose every record lands under `tmp_path` (R10)."""

    def build(ontology: OntologyEngineProtocol | None = None) -> HostDependencies:
        return HostDependencies(
            bus=EventBus(),
            llm=MockLLMConnector(),
            tools=ToolRegistry(),
            tracer=TelemetryTracer(),
            store=SessionStore(tmp_path / "sessions"),
            ontology=ontology,
        )

    return build


@pytest.fixture
def host(host_factory: Callable[..., HostDependencies]) -> HostDependencies:
    return host_factory()


def agent_participant(
    participant_id: str, session_id: str, namespace: str = "", persona: str = ""
) -> Participant:
    """An agent participant stamped with the ids `RoomService` would have derived."""
    return Participant(
        id=participant_id,
        kind=ParticipantKind.AGENT,
        display_name=participant_id,
        persona_summary=persona,
        session_id=session_id,
        ontology_namespace=namespace,
    )


# --------------------------------------------------------------------------------------
# Who gets an agent at all
# --------------------------------------------------------------------------------------


class TestWhoGetsAnAgent:
    @pytest.mark.asyncio
    async def test_a_human_participant_has_no_agent_behind_it(self, host: HostDependencies) -> None:
        """The floor is given to agents; a human speaks for itself.

        Killed by: src/uclone_x/room/resolver.py :: if participant.kind is not ParticipantKind.AGENT:
        Becomes: if False:
        """
        # Given a session id on purpose. A human normally has none, so without one this
        # test passes whether the kind check exists or not — the *next* guard, "no session
        # to claim", refuses it too, and the test cannot tell which one spoke. The
        # declared mutation did not kill for exactly that reason, and the fitness check
        # re-ran it and said so. A human carrying a session id is refusable only by the
        # guard this test is named for.
        alice = Participant(
            id="alice",
            kind=ParticipantKind.HUMAN,
            display_name="Alice",
            session_id="sess_room__r1__alice",
        )

        with pytest.raises(ParticipantNotResolvableError, match="alice"):
            await RoomAgentResolver(host).resolve(alice)

    @pytest.mark.asyncio
    async def test_an_agent_carrying_no_session_is_refused(self, host: HostDependencies) -> None:
        """An agent with no session id would silently take the host's default one.

        Which is the shared session the whole per-participant derivation exists to avoid,
        so it is refused here rather than discovered when two agents' turns interleave.

        Killed by: src/uclone_x/room/resolver.py :: if not participant.session_id:
        Becomes: if False:
        """
        with pytest.raises(ParticipantNotResolvableError, match="session id"):
            await RoomAgentResolver(host).resolve(agent_participant("scout", ""))

    @pytest.mark.asyncio
    async def test_two_participants_claiming_one_session_are_refused(
        self, host: HostDependencies
    ) -> None:
        """The roster is what needs fixing, and the refusal has to say so.

        Left to the session store, the same collision surfaces at the end of a turn as a
        revision precondition — an error that names a revision and not the roster edit that
        caused it.

        Killed by: src/uclone_x/room/resolver.py :: if claimed_by is not None and claimed_by != participant.id:
        Becomes: if False:
        """
        resolver = RoomAgentResolver(host)
        await resolver.resolve(agent_participant("scout", "sess_room__r1__shared"))

        with pytest.raises(ParticipantNotResolvableError) as caught:
            await resolver.resolve(agent_participant("critic", "sess_room__r1__shared"))

        message = str(caught.value)
        assert "scout" in message and "critic" in message, "name both claimants"
        assert "roster" in message

    @pytest.mark.asyncio
    async def test_resolving_one_participant_twice_returns_the_same_live_agent(
        self, host: HostDependencies
    ) -> None:
        """A room is a conversation, so the agent is kept between turns.

        Rebuilding per turn would discard the in-memory session and reload it from disk
        every time — slower, and a second writer's worth of opportunity to lose an update.

        Killed by: src/uclone_x/room/resolver.py :: return cached
        Becomes: pass
        """
        resolver = RoomAgentResolver(host)
        scout = agent_participant("scout", "sess_room__r1__scout")

        first = await resolver.resolve(scout)
        second = await resolver.resolve(scout)

        assert first is second
        assert first.context.session_id == "sess_room__r1__scout"


# --------------------------------------------------------------------------------------
# G3: no two participants on one session
# --------------------------------------------------------------------------------------


class TestSessionIsolation:
    @pytest.mark.asyncio
    async def test_the_same_id_in_another_room_does_not_get_the_first_rooms_agent(
        self, host: HostDependencies
    ) -> None:
        """The cache is keyed by the session, because the id is not unique across rooms.

        `scout` in room A and `scout` in room B are two participants with two derived
        sessions. Keyed by id alone, the second resolve returned the *first* room's live
        agent — still bound to `sess_room__a__scout` — so room B's conversation was written into
        room A's session. Neither the claim check nor the store's precondition could see
        it: only one session was ever opened, and the collision was the cache's own.

        Killed by: src/uclone_x/room/resolver.py :: cache_key = participant.session_id
        Becomes: cache_key = participant.id
        """
        resolver = RoomAgentResolver(host)

        in_room_a = await resolver.resolve(agent_participant("scout", "sess_room__a__scout"))
        in_room_b = await resolver.resolve(agent_participant("scout", "sess_room__b__scout"))

        assert in_room_a is not in_room_b, "one agent for two rooms is one session for two"
        assert in_room_a.context.session_id == "sess_room__a__scout"
        assert in_room_b.context.session_id == "sess_room__b__scout"


# --------------------------------------------------------------------------------------
# G4: no two participants on one ontology
# --------------------------------------------------------------------------------------


class TestOntologyIsolation:
    @pytest.mark.asyncio
    async def test_each_participant_induces_into_its_own_namespace(
        self, host: HostDependencies
    ) -> None:
        """The factory is called with the participant's own namespace, once each."""
        asked: list[str] = []

        def factory(namespace: str) -> OntologyEngineProtocol:
            asked.append(namespace)
            return OntologyEngine(namespace_iri=namespace)

        resolver = RoomAgentResolver(host, ontology_factory=factory)
        scout = await resolver.resolve(
            agent_participant("scout", "sess_room__r1__scout", "https://n/r1/scout")
        )
        critic = await resolver.resolve(
            agent_participant("critic", "sess_room__r1__critic", "https://n/r1/critic")
        )

        assert asked == ["https://n/r1/scout", "https://n/r1/critic"]
        assert scout.ontology is not critic.ontology

    @pytest.mark.asyncio
    async def test_two_participants_sharing_one_namespace_are_refused(
        self, host: HostDependencies
    ) -> None:
        """G4 is a property of the namespace, not of the engine object.

        Two engines built over one IRI are two objects that induce into one graph, so
        `scout is not critic` proves nothing: the concepts merge anyway, and the per-agent
        grounding P7 requires is gone. The session half of this obligation was refused from
        the first commit; the ontology half was not checked at all.

        Killed by: src/uclone_x/room/resolver.py :: if namespace_owner is not None and namespace_owner != participant.id:
        Becomes: if False:
        """
        resolver = RoomAgentResolver(
            host, ontology_factory=lambda ns: OntologyEngine(namespace_iri=ns)
        )
        await resolver.resolve(
            agent_participant("scout", "sess_room__r1__scout", "https://n/shared")
        )

        with pytest.raises(ParticipantNotResolvableError) as caught:
            await resolver.resolve(
                agent_participant("critic", "sess_room__r1__critic", "https://n/shared")
            )

        message = str(caught.value)
        assert "scout" in message and "critic" in message
        assert "https://n/shared" in message

    @pytest.mark.asyncio
    async def test_a_namespaced_participant_is_refused_the_hosts_shared_ontology(
        self, host_factory: Callable[..., HostDependencies]
    ) -> None:
        """Without a factory, every participant used to get the *one* host engine.

        Silently: the participant carried its own namespace, the resolver ignored it
        because it had nothing to build an engine with, and `compose_agent` handed each
        agent `host.ontology`. Every room agent then induced into one graph — the exact
        merge G4 forbids, produced by an omission at construction rather than by any
        roster edit.

        Killed by: src/uclone_x/room/resolver.py :: if self._ontology_factory is None and host.ontology is not None:
        Becomes: if False:
        """
        shared = OntologyEngine(namespace_iri="https://n/host")
        resolver = RoomAgentResolver(host_factory(shared))

        with pytest.raises(ParticipantNotResolvableError) as caught:
            await resolver.resolve(
                agent_participant("scout", "sess_room__r1__scout", "https://n/r1/scout")
            )

        assert "ontology" in str(caught.value)

    @pytest.mark.asyncio
    async def test_an_agent_with_no_namespace_runs_without_an_ontology(
        self, host: HostDependencies
    ) -> None:
        """A host with no ontology and a participant with no namespace share nothing."""
        agent = await RoomAgentResolver(host).resolve(
            agent_participant("scout", "sess_room__r1__scout")
        )

        assert agent.ontology is None


# --------------------------------------------------------------------------------------
# The seat's prompt
# --------------------------------------------------------------------------------------


class TestSystemPrompt:
    def test_the_prompt_names_the_seat_and_explains_the_prefixes(self) -> None:
        """An agent handed `[critic]: TTL is wrong` must not read it as the user's words.

        Killed by: src/uclone_x/room/resolver.py :: "Lines you are shown are prefixed with the speaker's id in square brackets; they "
        Becomes: ""
        """
        prompt = room_participant_system_prompt(
            agent_participant("critic", "sess_room__r1__critic", persona="finds the flaw")
        )

        assert "critic" in prompt
        assert "finds the flaw" in prompt
        assert "square brackets" in prompt
        assert "impersonate" in prompt

    def test_a_participant_with_no_persona_still_gets_a_role(self) -> None:
        """An empty summary must not render as an empty role clause.

        Killed by: src/uclone_x/room/resolver.py :: purpose = participant.persona_summary or "contribute your own perspective"
        Becomes: purpose = participant.persona_summary
        """
        prompt = room_participant_system_prompt(agent_participant("scout", "sess_room__r1__scout"))

        assert "Your role: ." not in prompt
        assert "contribute your own perspective" in prompt


# --------------------------------------------------------------------------------------
# What a seated agent remembers
# --------------------------------------------------------------------------------------


class TestPerParticipantMemory:
    """Memory is per seat, the same way the ontology is.

    `HostDependencies.memory` shipped with a comment claiming it was "wired by every head".
    Neither room head wired it: `ui/rooms.py` and `cli/commands/room.py` both built their
    host with no `memory=`, so every agent seated in a room was composed without a store
    and — since #1098 closed the registry fallback — had `record_memory_fact` neither
    advertised to it nor resolvable. A room agent could not remember anything.

    The fix is deliberately not a store on the host. One store there is handed to every
    participant, and each would read back the others' recollections as its own: the exact
    failure `ontology_factory` already exists to prevent for the knowledge graph.

    Ids that name no persona are used on purpose, so the wiring under test is the only
    thing that can fail here. Whether a persona's allowlist lets the memory tools through
    is a separate question -- every persona is given them since #1402 -- and is pinned in
    `test_persona_base_tools.py`.
    """

    @pytest.mark.asyncio
    async def test_each_participant_is_composed_with_its_own_store(
        self, host: HostDependencies, tmp_path: Path
    ) -> None:
        """Two seats, two stores, and a fact recorded in one is absent from the other.

        Killed by: src/uclone_x/room/resolver.py :: host = dataclasses.replace(host, memory=self._memory_factory(participant.id))
        Becomes: host = host
        """
        stores: dict[str, CrossSessionMemory] = {}

        def memory_for(agent_id: str) -> CrossSessionMemory:
            store = CrossSessionMemory(storage_path=tmp_path / f"{agent_id}.json")
            stores[agent_id] = store
            return store

        resolver = RoomAgentResolver(host, memory_factory=memory_for)

        alpha = cast(
            "BaseAgent", await resolver.resolve(agent_participant("alpha", "sess_room__r1__alpha"))
        )
        beta = cast(
            "BaseAgent", await resolver.resolve(agent_participant("beta", "sess_room__r1__beta"))
        )

        assert alpha.memory is stores["alpha"]
        assert beta.memory is stores["beta"]
        assert alpha.memory is not beta.memory

        result = await alpha.execute_tool_call(
            "record_memory_fact",
            {"subject": "the room", "predicate": "seats", "object_value": "two agents"},
        )
        assert result.status == "success", result.error
        assert [fact.subject for fact in stores["alpha"].list_facts()] == ["the room"]
        assert stores["beta"].list_facts() == []

    @pytest.mark.asyncio
    async def test_a_resolver_given_no_factory_seats_agents_without_memory(
        self, host: HostDependencies
    ) -> None:
        """The documented fallback, pinned so it stays a decision rather than an accident.

        A resolver with no `memory_factory` composes agents with no store, and #1098 means
        that is an honest absence: the tool is not there rather than reaching whichever
        store some other agent registered first.
        """
        resolver = RoomAgentResolver(host)

        alpha = cast(
            "BaseAgent", await resolver.resolve(agent_participant("alpha", "sess_room__r1__alpha"))
        )

        assert alpha.memory is None
        with pytest.raises(KeyError, match="not registered"):
            await alpha.execute_tool_call(
                "record_memory_fact",
                {"subject": "a", "predicate": "b", "object_value": "c"},
            )


# --------------------------------------------------------------------------------------
# Which model a seated agent is served by
# --------------------------------------------------------------------------------------


class TestPersonaModelSelection:
    """The persona's configured model must reach the agent the room composes.

    Pinned here because #1208 made this path the *only* one. Until that PR a model
    reached the wire two ways: this one, and `#1138`'s per-message override through
    `POST /api/chat/stream`. The override was the tested one, and it is the one the
    retirement deletes — so the whole claim that the deletion costs no user-visible
    capability now rests on the untested path, which is the asymmetry this closes.

    What it guards against is silent: an installation-wide `llm_config=` added to the
    `RoomAgentResolver` construction in `src/uclone_x/ui/rooms.py` would make the first
    term of the `or` chain truthy, every persona would be served by the installation
    default, and `PersonaEditor`'s model field would become a control that does nothing
    while reporting that it saved. Nothing else in the repository fails on that.
    """

    @pytest.mark.asyncio
    async def test_a_personas_configured_model_survives_resolution(
        self, host: HostDependencies, tmp_path: Path
    ) -> None:
        """What `PersonaEditor` saved is what the seated agent is configured to call.

        Killed by: src/uclone_x/room/resolver.py :: resolved_llm_config = persona_def.llm_config
        Becomes: resolved_llm_config = None
        """
        registry = PersonaRegistry(workspace_root=tmp_path, include_defaults=False)
        registry.register_persona(
            PersonaDefinition(
                name="novelist",
                role="Creative Fiction Writer",
                description="Specialist in narrative prose.",
                system_prompt="You are a novelist.",
                # Deliberately not a model any default could coincide with: a real
                # default would let this pass while resolution silently ignored the
                # persona, which is the failure the test exists to catch.
                llm_config=AgentLLMConfig(model_name="qwen3:32b-a-very-specific-tag"),
            )
        )
        resolver = RoomAgentResolver(host, persona_registry=registry)
        participant = Participant(
            id="author",
            kind=ParticipantKind.AGENT,
            display_name="Author",
            persona="novelist",
            session_id="sess_room__r1__author",
        )

        agent = await resolver.resolve(participant)

        assert agent.config.llm_config.model_name == "qwen3:32b-a-very-specific-tag", (
            "the seat was composed against some other model than the one its persona configures"
        )

    @pytest.mark.asyncio
    async def test_an_installation_wide_config_is_what_would_take_the_persona_away(
        self, host: HostDependencies, tmp_path: Path
    ) -> None:
        """The precedence, stated rather than left to be discovered by whoever changes it.

        `self._llm_config` wins over the persona by construction, and the head relies on
        never passing one (`src/uclone_x/ui/rooms.py`). Pinning the losing case makes that
        reliance visible at the resolver instead of only in the caller's absence of an
        argument, so a future caller that adds one is choosing it knowingly.
        """
        registry = PersonaRegistry(workspace_root=tmp_path, include_defaults=False)
        registry.register_persona(
            PersonaDefinition(
                name="novelist",
                role="Creative Fiction Writer",
                description="Specialist in narrative prose.",
                system_prompt="You are a novelist.",
                llm_config=AgentLLMConfig(model_name="qwen3:32b-a-very-specific-tag"),
            )
        )
        resolver = RoomAgentResolver(
            host,
            llm_config=AgentLLMConfig(model_name="an-installation-wide-default"),
            persona_registry=registry,
        )
        participant = Participant(
            id="author",
            kind=ParticipantKind.AGENT,
            display_name="Author",
            persona="novelist",
            session_id="sess_room__r1__author",
        )

        agent = await resolver.resolve(participant)

        assert agent.config.llm_config.model_name == "an-installation-wide-default"

    @pytest.mark.asyncio
    async def test_room_agent_receives_workspace_root(
        self, host: HostDependencies, tmp_path: Path
    ) -> None:
        """Verify that room agents are composed with a non-None workspace_root."""
        ws_dir = tmp_path / "custom_workspace"
        ws_dir.mkdir(parents=True, exist_ok=True)
        resolver = RoomAgentResolver(host, workspace_root=ws_dir)
        participant = Participant(
            id="scout",
            kind=ParticipantKind.AGENT,
            display_name="Scout",
            session_id="sess_room__r1__scout",
        )
        agent = cast(BaseAgent, await resolver.resolve(participant))
        assert agent.context.workspace_root == ws_dir.resolve()
        assert agent.config.workspace_dir == ws_dir.resolve()

    @pytest.mark.asyncio
    async def test_room_agent_preserves_room_instructions_with_persona(
        self, host: HostDependencies, tmp_path: Path
    ) -> None:
        """Verify that adopting a persona in a room preserves room participation instructions."""
        registry = PersonaRegistry(workspace_root=tmp_path, include_defaults=False)
        registry.register_persona(
            PersonaDefinition(
                name="novelist",
                role="Creative Fiction Writer",
                description="Specialist in narrative prose.",
                system_prompt="Evocative prose only.",
            )
        )
        resolver = RoomAgentResolver(host, persona_registry=registry)
        participant = Participant(
            id="story_writer",
            kind=ParticipantKind.AGENT,
            display_name="Story Writer",
            persona="novelist",
            session_id="sess_room__r1__story_writer",
        )
        agent = cast(BaseAgent, await resolver.resolve(participant))
        prompt = agent.effective_system_prompt
        assert "shared multi-agent conversation" in prompt
        assert "Evocative prose only." in prompt

    @pytest.mark.asyncio
    async def test_room_agent_without_persona_includes_default_system_prompt(
        self, host: HostDependencies
    ) -> None:
        """Verify that room agents without personas receive default system prompt instructions."""
        from uclone_x.agent.prompts import IMAGE_GENERATION

        resolver = RoomAgentResolver(host)
        participant = Participant(
            id="helper",
            kind=ParticipantKind.AGENT,
            display_name="Helper",
            session_id="sess_room__r1__helper",
        )
        agent = cast(BaseAgent, await resolver.resolve(participant))
        prompt = agent.effective_system_prompt
        assert "shared multi-agent conversation" in prompt
        assert IMAGE_GENERATION in prompt
