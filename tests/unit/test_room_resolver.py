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
* **Two clones never share a rules engine, and one clone keeps one.** Since
  clone-knowledge-graph step 6 a seat's engine is its clone's (`AppScope.ontology_for`,
  keyed by clone id): the same clone seated in two rooms reads one engine, two clones two.
* **A host carrying one engine is refused.** Handing it to the resolver would give every
  seat that one shared engine -- the manager engine step 6 retired.
* **A human has no agent behind it**, and neither does an agent carrying no session.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.clone_builder import ontology_map
from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.models import AgentLLMConfig, PersonaDefinition
from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.session import SessionStore
from uclone_x.engine.event_bus import EventBus
from uclone_x.errors import ParticipantNotResolvableError
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.ontology.engine import OntologyEngine
from uclone_x.ontology.models import OntologyRelation
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


def agent_participant(participant_id: str, session_id: str, persona: str = "") -> Participant:
    """An agent participant stamped with the ids `RoomService` would have derived."""
    return Participant(
        id=participant_id,
        kind=ParticipantKind.AGENT,
        display_name=participant_id,
        persona_summary=persona,
        session_id=session_id,
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
# G4: one rules engine per clone (clone-knowledge-graph step 6)
# --------------------------------------------------------------------------------------


class TestOntologyIsClonesOwn:
    @pytest.mark.asyncio
    async def test_two_clones_never_share_an_engine_and_one_clone_keeps_one(
        self, host: HostDependencies
    ) -> None:
        """A seat's engine is its clone's, whichever room it is seated in.

        Killed by: src/uclone_x/agent/clone_builder.py :: host = dataclasses.replace(host, ontology=app.ontology_for(clone_id))
        Becomes: host = dataclasses.replace(host, ontology=app.ontology_for("clone"))
        Killed by: src/uclone_x/agent/clone_builder.py :: engines[key] = existing
        Becomes: pass
        """
        resolver = RoomAgentResolver(host, ontology_for=ontology_map())
        scout_a = await resolver.resolve(agent_participant("scout", "sess_room__a__scout"))
        scout_b = await resolver.resolve(agent_participant("scout", "sess_room__b__scout"))
        critic = await resolver.resolve(agent_participant("critic", "sess_room__a__critic"))

        assert scout_a.ontology is not None and critic.ontology is not None
        assert scout_a.ontology is not critic.ontology, "two clones on one engine"
        assert scout_a.ontology is scout_b.ontology, "one clone, two engines in two rooms"
        assert scout_a is not scout_b, "one agent for two rooms is one session for two"

    @pytest.mark.asyncio
    async def test_what_one_clone_is_taught_the_other_does_not_hold(
        self, host: HostDependencies
    ) -> None:
        """Not just two objects: a rule registered on one clone's engine is not the other's."""
        resolver = RoomAgentResolver(host, ontology_for=ontology_map())
        scout = await resolver.resolve(agent_participant("scout", "sess_room__a__scout"))
        critic = await resolver.resolve(agent_participant("critic", "sess_room__a__critic"))
        assert isinstance(scout.ontology, OntologyEngine)
        assert isinstance(critic.ontology, OntologyEngine)

        scout.ontology.register_relation(
            OntologyRelation(source_entity="tide", predicate="part_of", target_entity="sea")
        )

        assert [r.source_entity for r in scout.ontology.list_relations()] == ["tide"]
        assert critic.ontology.list_relations() == []

    def test_a_host_carrying_one_engine_is_refused(
        self, host_factory: Callable[..., HostDependencies]
    ) -> None:
        """One engine on the host would be every seat's: the shared engine step 6 retired.

        Killed by: src/uclone_x/room/resolver.py :: if host.ontology is not None:
        Becomes: if False:
        """
        shared = OntologyEngine(namespace_iri="https://n/host")

        with pytest.raises(TypeError) as caught:
            RoomAgentResolver(host_factory(shared))

        assert "ontology_for" in str(caught.value)

    @pytest.mark.asyncio
    async def test_with_no_engines_to_hand_out_a_seat_runs_without_one(
        self, host: HostDependencies
    ) -> None:
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
    failure `ontology_for` prevents for the knowledge graph, one engine per clone.

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

        Killed by: src/uclone_x/agent/clone_builder.py :: host = dataclasses.replace(host, memory=app.memory_for(clone_id))
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

        Killed by: src/uclone_x/agent/clone_builder.py :: llm_config = persona_def.llm_config
        Becomes: llm_config = AgentLLMConfig()
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
                llm_config=AgentLLMConfig(model_name="ollama/qwen3:32b-a-very-specific-tag"),
            )
        )
        resolver = RoomAgentResolver(host, persona_registry=registry)
        participant = Participant(
            id="novelist",
            kind=ParticipantKind.AGENT,
            display_name="Author",
            session_id="sess_room__r1__novelist",
        )

        agent = await resolver.resolve(participant)

        assert agent.config.llm_config.model_name == "ollama/qwen3:32b-a-very-specific-tag", (
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
                llm_config=AgentLLMConfig(model_name="ollama/qwen3:32b-a-very-specific-tag"),
            )
        )
        resolver = RoomAgentResolver(
            host,
            llm_config=AgentLLMConfig(model_name="an-installation-wide-default"),
            persona_registry=registry,
        )
        participant = Participant(
            id="novelist",
            kind=ParticipantKind.AGENT,
            display_name="Author",
            session_id="sess_room__r1__novelist",
        )

        agent = await resolver.resolve(participant)

        assert agent.config.llm_config.model_name == "an-installation-wide-default"

    @pytest.mark.asyncio
    async def test_a_seat_follows_the_settings_models_only_where_its_persona_names_none(
        self, host: HostDependencies, tmp_path: Path
    ) -> None:
        """Deep and fast each follow Settings in the slot the persona left empty.

        The seat that follows is the one that sent no model before, so its connector filled
        in a model id written in source. The seat whose persona names a model keeps it,
        and keeps it through a Settings change.

        Killed by: src/uclone_x/agent/clone_builder.py :: llm_config = follow_global_models(llm_config, deep, fast)
        Becomes: pass
        """
        registry = PersonaRegistry(workspace_root=tmp_path, include_defaults=False)
        registry.register_persona(
            PersonaDefinition(
                name="plain",
                role="Generalist",
                description="Names no model.",
                system_prompt="You help.",
            )
        )
        registry.register_persona(
            PersonaDefinition(
                name="novelist",
                role="Creative Fiction Writer",
                description="Specialist in narrative prose.",
                system_prompt="You are a novelist.",
                llm_config=AgentLLMConfig(model_name="box/own-deep", fast_model="box/own-fast"),
            )
        )
        models: dict[str, str | None] = {"deep": "settings-deep", "fast": None}
        resolver = RoomAgentResolver(
            host,
            persona_registry=registry,
            global_models=lambda: (models["deep"], models["fast"]),
        )

        def seat(pid: str) -> Participant:
            # A seat is its clone: its persona is found by its id.
            return Participant(
                id=pid,
                kind=ParticipantKind.AGENT,
                display_name=pid,
                session_id=f"sess_room__r1__{pid}",
            )

        follower = cast(BaseAgent, await resolver.resolve(seat("plain")))
        owner = cast(BaseAgent, await resolver.resolve(seat("novelist")))

        # Fast left empty in Settings means deep.
        assert follower.config.llm_config.model_name == "settings-deep"
        assert follower.config.llm_config.fast_model == "settings-deep"
        assert owner.config.llm_config.model_name == "box/own-deep"
        assert owner.config.llm_config.fast_model == "box/own-fast"

        models.update(deep="settings-deep-2", fast="settings-fast-2")
        replacement = MockLLMConnector()
        resolver.replace_llm(replacement)

        assert follower.llm is replacement
        assert follower.config.llm_config.model_name == "settings-deep-2"
        assert follower.config.llm_config.fast_model == "settings-fast-2"
        assert owner.llm is replacement
        assert owner.config.llm_config.model_name == "box/own-deep"
        assert owner.config.llm_config.fast_model == "box/own-fast"

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
            id="novelist",
            kind=ParticipantKind.AGENT,
            display_name="Story Writer",
            session_id="sess_room__r1__novelist",
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
