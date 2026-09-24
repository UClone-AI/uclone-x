"""Tests for the room speaker selectors.

The distinction these pin hardest is the three-valued verdict. A selector that abstains,
one that decides on silence, and one that fails are three different things, and the prior
art this design replaces collapsed all three into one `None` — so each test below that
asserts `ABSTAIN` rather than `SILENCE` (or a raise rather than either) is pinning that
separation, not merely an enum value.
"""

from __future__ import annotations

from typing import Final

import pytest

from uclone_x.errors import SpeakerSelectionError, UnknownRoomParticipantError
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import LLMRequest, ModelResponse
from uclone_x.room.models import (
    Participant,
    ParticipantKind,
    RoomMessage,
    RoomPolicy,
    SelectionVerdict,
    SpeakerDecision,
    SpeakerRequest,
    TurnState,
)
from uclone_x.room.protocols import SpeakerSelectorProtocol
from uclone_x.room.selectors import (
    DefaultResponderSelector,
    LLMSpeakerSelector,
    MentionSelector,
    SoleAgentSelector,
    build_selector_chain,
)

ALICE = Participant(id="alice", kind=ParticipantKind.HUMAN, display_name="Alice")
SCOUT = Participant(
    id="scout",
    kind=ParticipantKind.AGENT,
    display_name="Scout",
    persona_summary="Explores the codebase and reports findings.",
    session_id="sess_room__r1__scout",
    ontology_namespace="https://uclone-x.ai/ontology/scout",
)
CRITIC = Participant(
    id="critic-1",
    kind=ParticipantKind.AGENT,
    display_name="Critic",
    persona_summary="Adversarially reviews proposals.",
    aliases=("critic", "reviewer"),
    session_id="sess_room__r1__critic-1",
    ontology_namespace="https://uclone-x.ai/ontology/critic-1",
)


def _request(
    *participants: Participant,
    utterances: tuple[tuple[str, str], ...] = (("alice", "hello"),),
    turn_state: TurnState | None = None,
    policy: RoomPolicy | None = None,
) -> SpeakerRequest:
    """Build a `SpeakerRequest` from `(sender_id, content)` pairs."""
    transcript = tuple(
        RoomMessage(seq=i, sender_id=sender, content=content)
        for i, (sender, content) in enumerate(utterances, start=1)
    )
    return SpeakerRequest(
        room_id="r1",
        participants=participants,
        transcript=transcript,
        turn_state=turn_state or TurnState(),
        policy=policy or RoomPolicy(),
    )


class TestSoleAgentSelector:
    @pytest.mark.asyncio
    async def test_routes_to_the_only_agent(self) -> None:
        decision = await SoleAgentSelector().select(_request(ALICE, SCOUT))
        assert decision.verdict is SelectionVerdict.SPEAK
        assert decision.speaker_id == "scout"
        assert decision.selector == "sole_agent"

    @pytest.mark.asyncio
    async def test_abstains_once_a_second_agent_joins(self) -> None:
        decision = await SoleAgentSelector().select(_request(ALICE, SCOUT, CRITIC))
        # ABSTAIN, not SILENCE: this rule has no opinion about a two-agent room, and
        # reporting silence here would end the chain before the LLM selector ran.
        assert decision.verdict is SelectionVerdict.ABSTAIN
        assert decision.speaker_id is None

    @pytest.mark.asyncio
    async def test_abstains_in_a_room_with_no_agents(self) -> None:
        decision = await SoleAgentSelector().select(_request(ALICE))
        assert decision.verdict is SelectionVerdict.ABSTAIN


class TestMentionSelector:
    @pytest.mark.asyncio
    async def test_routes_on_an_exact_id(self) -> None:
        request = _request(ALICE, SCOUT, CRITIC, utterances=(("alice", "@scout take a look"),))
        decision = await MentionSelector().select(request)
        assert decision.verdict is SelectionVerdict.SPEAK
        assert decision.speaker_id == "scout"

    @pytest.mark.asyncio
    async def test_routes_on_an_alias(self) -> None:
        request = _request(ALICE, SCOUT, CRITIC, utterances=(("alice", "@critic your turn"),))
        decision = await MentionSelector().select(request)
        assert decision.speaker_id == "critic-1"

    @pytest.mark.asyncio
    async def test_id_beats_another_participants_alias(self) -> None:
        """An alias must not hijack mail addressed to a real id."""
        shadow = Participant(
            id="scout",
            kind=ParticipantKind.AGENT,
            display_name="Scout",
            session_id="sess_room__r1__scout",
        )
        impostor = Participant(
            id="other",
            kind=ParticipantKind.AGENT,
            display_name="Other",
            aliases=("scout",),
            session_id="sess_room__r1__other",
        )
        request = _request(ALICE, impostor, shadow, utterances=(("alice", "@scout hi"),))
        decision = await MentionSelector().select(request)
        assert decision.speaker_id == "scout"

    @pytest.mark.asyncio
    async def test_first_mentioned_agent_speaks(self) -> None:
        request = _request(
            ALICE, SCOUT, CRITIC, utterances=(("alice", "@critic @scout compare notes"),)
        )
        decision = await MentionSelector().select(request)
        assert decision.speaker_id == "critic-1"

    @pytest.mark.asyncio
    async def test_abstains_with_no_mention(self) -> None:
        request = _request(ALICE, SCOUT, CRITIC, utterances=(("alice", "what do you think?"),))
        decision = await MentionSelector().select(request)
        assert decision.verdict is SelectionVerdict.ABSTAIN

    @pytest.mark.asyncio
    async def test_email_is_not_a_mention(self) -> None:
        request = _request(
            ALICE, SCOUT, CRITIC, utterances=(("alice", "mail me at me@scout.example"),)
        )
        decision = await MentionSelector().select(request)
        assert decision.verdict is SelectionVerdict.ABSTAIN

    @pytest.mark.asyncio
    async def test_unknown_mention_refuses_instead_of_guessing(self) -> None:
        request = _request(ALICE, SCOUT, CRITIC, utterances=(("alice", "@phantm help"),))
        with pytest.raises(UnknownRoomParticipantError) as excinfo:
            await MentionSelector().select(request)
        # The roster travels in the message: the remedy is for the human to retype.
        assert "@scout" in str(excinfo.value)
        assert "phantm" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_an_agents_own_mention_does_not_dispatch(self) -> None:
        """Only the human's words address anyone.

        An agent writing `@critic` in its reply must not thereby summon the critic: that
        is a prompt-injection route into the room's control plane, and an agent-to-agent
        loop with no human in it.
        """
        request = _request(
            ALICE,
            SCOUT,
            CRITIC,
            utterances=(("alice", "what do you think?"), ("scout", "good question, @critic?")),
        )
        decision = await MentionSelector().select(request)
        assert decision.verdict is SelectionVerdict.ABSTAIN

    @pytest.mark.asyncio
    async def test_abstains_when_a_human_addresses_a_human(self) -> None:
        bob = Participant(id="bob", kind=ParticipantKind.HUMAN, display_name="Bob")
        request = _request(ALICE, bob, SCOUT, CRITIC, utterances=(("alice", "@bob thoughts?"),))
        decision = await MentionSelector().select(request)
        assert decision.verdict is SelectionVerdict.ABSTAIN


class TestLLMSpeakerSelector:
    @pytest.mark.asyncio
    async def test_speaks_the_named_agent_with_provenance(self) -> None:
        llm = MockLLMConnector(
            responses=['{"speaker_id": "critic-1", "confidence": 0.8, "reasoning": "review fits"}']
        )
        decision = await LLMSpeakerSelector(llm).select(_request(ALICE, SCOUT, CRITIC))
        assert decision.verdict is SelectionVerdict.SPEAK
        assert decision.speaker_id == "critic-1"
        assert decision.confidence == pytest.approx(0.8)
        # P6: a decision a model produced carries the attribution of the model.
        assert decision.provenance is not None

    @pytest.mark.asyncio
    async def test_null_speaker_is_silence_not_abstain(self) -> None:
        """A decided quiet room ends the chain; it is not a pass to the next selector."""
        llm = MockLLMConnector(
            responses=['{"speaker_id": null, "confidence": 0.9, "reasoning": "exchange concluded"}']
        )
        decision = await LLMSpeakerSelector(llm).select(_request(ALICE, SCOUT, CRITIC))
        assert decision.verdict is SelectionVerdict.SILENCE
        assert decision.speaker_id is None
        assert decision.reasoning == "exchange concluded"

    @pytest.mark.asyncio
    async def test_tolerates_a_fenced_reply(self) -> None:
        llm = MockLLMConnector(
            responses=['```json\n{"speaker_id": "scout", "confidence": 1.0}\n```']
        )
        decision = await LLMSpeakerSelector(llm).select(_request(ALICE, SCOUT, CRITIC))
        assert decision.speaker_id == "scout"

    @pytest.mark.asyncio
    async def test_ignores_an_unexpected_extra_key(self) -> None:
        llm = MockLLMConnector(
            responses=['{"speaker_id": "scout", "confidence": 1.0, "aside": "chatty"}']
        )
        decision = await LLMSpeakerSelector(llm).select(_request(ALICE, SCOUT, CRITIC))
        assert decision.speaker_id == "scout"

    @pytest.mark.asyncio
    async def test_provider_failure_raises_and_is_never_silence(self) -> None:
        class BrokenProvider(MockLLMConnector):
            async def generate(self, request: object) -> object:  # type: ignore[override]
                raise RuntimeError("provider exploded")

        with pytest.raises(SpeakerSelectionError, match="provider exploded"):
            await LLMSpeakerSelector(BrokenProvider()).select(_request(ALICE, SCOUT, CRITIC))

    @pytest.mark.asyncio
    async def test_unparseable_reply_raises(self) -> None:
        llm = MockLLMConnector(responses=["I think the critic should go next, personally."])
        with pytest.raises(SpeakerSelectionError, match="no JSON object"):
            await LLMSpeakerSelector(llm).select(_request(ALICE, SCOUT, CRITIC))

    @pytest.mark.asyncio
    async def test_empty_reply_raises(self) -> None:
        llm = MockLLMConnector(responses=["   "])
        with pytest.raises(SpeakerSelectionError, match="empty reply"):
            await LLMSpeakerSelector(llm).select(_request(ALICE, SCOUT, CRITIC))

    @pytest.mark.asyncio
    async def test_hallucinated_speaker_raises(self) -> None:
        llm = MockLLMConnector(responses=['{"speaker_id": "nobody-here", "confidence": 1.0}'])
        with pytest.raises(SpeakerSelectionError, match="not an agent in room"):
            await LLMSpeakerSelector(llm).select(_request(ALICE, SCOUT, CRITIC))

    @pytest.mark.asyncio
    async def test_human_named_by_the_model_raises(self) -> None:
        """The model may not hand the floor back to a person."""
        llm = MockLLMConnector(responses=['{"speaker_id": "alice", "confidence": 1.0}'])
        with pytest.raises(SpeakerSelectionError, match="not an agent in room"):
            await LLMSpeakerSelector(llm).select(_request(ALICE, SCOUT, CRITIC))

    @pytest.mark.asyncio
    async def test_abstains_without_spending_a_call_when_no_agents(self) -> None:
        llm = MockLLMConnector(responses=['{"speaker_id": "scout"}'])
        decision = await LLMSpeakerSelector(llm).select(_request(ALICE))
        assert decision.verdict is SelectionVerdict.ABSTAIN
        assert llm.call_count == 0

    @pytest.mark.asyncio
    async def test_prompt_carries_roster_purposes_and_floor_state(self) -> None:
        """What the selector is told is a routing input, so it is pinned."""
        captured: list[str] = []

        class CapturingProvider(MockLLMConnector):
            async def generate(self, request):  # type: ignore[no-untyped-def,override]
                captured.append(request.messages[-1].content or "")
                return await super().generate(request)

        llm = CapturingProvider(responses=['{"speaker_id": "scout", "confidence": 1.0}'])
        await LLMSpeakerSelector(llm).select(
            _request(
                ALICE,
                SCOUT,
                CRITIC,
                utterances=(("alice", "where is the cache?"),),
                turn_state=TurnState(agent_turns_since_human=2, last_speaker_id="critic-1"),
            )
        )
        prompt = captured[0]
        assert "Explores the codebase" in prompt
        assert "Adversarially reviews" in prompt
        assert "AGENT TURNS SINCE THE LAST HUMAN MESSAGE: 2" in prompt
        assert "LAST SPEAKER: critic-1" in prompt
        assert "[alice]: where is the cache?" in prompt

    @pytest.mark.asyncio
    async def test_string_null_or_none_is_silence_not_hallucinated_speaker(self) -> None:
        """Models outputting 'null' or 'none' as strings mean silence, not a phantom agent."""
        for token in ("null", "NULL", "none", "None", "nobody", "nil"):
            llm = MockLLMConnector(
                responses=[f'{{"speaker_id": "{token}", "confidence": 0.95, "reasoning": "quiet"}}']
            )
            decision = await LLMSpeakerSelector(llm).select(_request(ALICE, SCOUT, CRITIC))
            assert decision.verdict is SelectionVerdict.SILENCE
            assert decision.speaker_id is None

    @pytest.mark.asyncio
    async def test_default_max_tokens_accommodates_reasoning_models(self) -> None:
        from uclone_x.room.selectors import DEFAULT_SELECTOR_MAX_TOKENS

        assert DEFAULT_SELECTOR_MAX_TOKENS >= 1024
        captured_requests: list[LLMRequest] = []

        class RequestCapturingProvider(MockLLMConnector):
            async def generate(self, request: LLMRequest):  # type: ignore[no-untyped-def,override]
                captured_requests.append(request)
                return await super().generate(request)

        llm = RequestCapturingProvider(responses=['{"speaker_id": "scout"}'])
        await LLMSpeakerSelector(llm).select(_request(ALICE, SCOUT, CRITIC))
        assert captured_requests[0].max_tokens == DEFAULT_SELECTOR_MAX_TOKENS

    @pytest.mark.asyncio
    async def test_fallback_to_thinking_when_content_is_empty(self) -> None:
        """Reasoning models that put JSON in thinking or truncate at boundary are handled."""

        class ThinkingModelProvider(MockLLMConnector):
            async def generate(self, request):  # type: ignore[no-untyped-def,override]
                base_resp = await super().generate(request)
                return base_resp.model_copy(
                    update={
                        "content": "",
                        "thinking": 'Thinking about who speaks next: {"speaker_id": "scout", "confidence": 1.0}',
                    }
                )

        selector = LLMSpeakerSelector(ThinkingModelProvider(responses=["placeholder"]))
        decision = await selector.select(_request(ALICE, SCOUT, CRITIC))
        assert decision.verdict is SelectionVerdict.SPEAK
        assert decision.speaker_id == "scout"


class TestMentionTokenEdges:
    """Regressions from the PR #690 review: punctuation around a mention."""

    @pytest.mark.asyncio
    async def test_a_trailing_period_does_not_break_the_address(self) -> None:
        """`@scout.` is a sentence, not an unknown participant."""
        request = _request(ALICE, SCOUT, CRITIC, utterances=(("alice", "please help @scout."),))
        decision = await MentionSelector().select(request)
        assert decision.speaker_id == "scout"

    @pytest.mark.asyncio
    async def test_trailing_punctuation_of_several_kinds_is_trimmed(self) -> None:
        for text in ("@scout,", "@scout!", "@scout?", "@scout...", "@scout-", "(@scout)"):
            request = _request(ALICE, SCOUT, CRITIC, utterances=(("alice", f"hey {text} ok"),))
            decision = await MentionSelector().select(request)
            assert decision.speaker_id == "scout", text

    @pytest.mark.asyncio
    async def test_a_bracketed_or_emphasised_mention_is_still_an_address(self) -> None:
        for text in ("(@scout) look", "**@scout** look", "- @scout look"):
            request = _request(ALICE, SCOUT, CRITIC, utterances=(("alice", text),))
            decision = await MentionSelector().select(request)
            assert decision.speaker_id == "scout", text

    @pytest.mark.asyncio
    async def test_resolution_is_case_insensitive(self) -> None:
        """Every id and mention elsewhere in this file is lowercase, so nothing pinned this."""
        mixed = Participant(
            id="Scout-Prime",
            kind=ParticipantKind.AGENT,
            display_name="Scout Prime",
            aliases=("Recon",),
            session_id="sess_room__r1__scout-prime",
        )
        for text in ("@scout-prime hi", "@SCOUT-PRIME hi", "@recon hi", "@RECON hi"):
            request = _request(ALICE, mixed, utterances=(("alice", text),))
            decision = await MentionSelector().select(request)
            assert decision.speaker_id == "Scout-Prime", text

    @pytest.mark.asyncio
    async def test_an_interior_dot_still_resolves_a_dotted_id(self) -> None:
        dotted = Participant(
            id="scout.v2",
            kind=ParticipantKind.AGENT,
            display_name="Scout v2",
            session_id="sess_room__r1__scout.v2",
        )
        request = _request(ALICE, dotted, utterances=(("alice", "@scout.v2 hi"),))
        decision = await MentionSelector().select(request)
        assert decision.speaker_id == "scout.v2"


class TestSelectorReplyIsStrict:
    """A reply that names nobody must not be read as a decision to be quiet."""

    @pytest.mark.asyncio
    async def test_a_reply_with_no_speaker_id_key_raises(self) -> None:
        llm = MockLLMConnector(responses=['{"reasoning": "critic should go"}'])
        with pytest.raises(SpeakerSelectionError):
            await LLMSpeakerSelector(llm).select(_request(ALICE, SCOUT, CRITIC))

    @pytest.mark.asyncio
    async def test_a_misspelled_speaker_key_raises_rather_than_falling_silent(self) -> None:
        llm = MockLLMConnector(responses=['{"speaker": "critic-1", "confidence": 0.9}'])
        with pytest.raises(SpeakerSelectionError):
            await LLMSpeakerSelector(llm).select(_request(ALICE, SCOUT, CRITIC))

    @pytest.mark.asyncio
    async def test_an_empty_object_raises(self) -> None:
        llm = MockLLMConnector(responses=["{}"])
        with pytest.raises(SpeakerSelectionError):
            await LLMSpeakerSelector(llm).select(_request(ALICE, SCOUT, CRITIC))

    @pytest.mark.asyncio
    async def test_an_explicit_null_is_still_a_decided_silence(self) -> None:
        llm = MockLLMConnector(responses=['{"speaker_id": null, "reasoning": "done"}'])
        decision = await LLMSpeakerSelector(llm).select(_request(ALICE, SCOUT, CRITIC))
        assert decision.verdict is SelectionVerdict.SILENCE


class TestMembershipLinesAreNotAddresses:
    """A join or leave row shares the transcript with speech; selectors read only speech.

    Both helpers that scan a transcript — the one that finds the last human utterance and
    the one that works out which addresses are still outstanding — were written when a
    transcript held nothing but utterances.
    """

    @pytest.mark.asyncio
    async def test_a_mention_inside_a_membership_line_addresses_nobody(self) -> None:
        """`@scout` in the text of a join line is a rendering, not an address.

        The line's content is composed by the room, and reading it as a human's words
        would let the room dispatch agents on its own prose.

        Killed by: src/uclone_x/room/selectors.py :: if message.is_utterance and message.sender_id in humans:
        Becomes: if message.sender_id in humans:
        """
        from uclone_x.room.models import RoomMessageKind

        transcript = (
            RoomMessage(seq=1, sender_id="alice", content="anything to add @critic?"),
            RoomMessage(
                seq=2,
                sender_id="alice",
                content="alice joined the room alongside @scout",
                kind=RoomMessageKind.JOIN,
            ),
        )
        request = SpeakerRequest(
            room_id="r1",
            participants=(ALICE, SCOUT, CRITIC),
            transcript=transcript,
            turn_state=TurnState(),
            policy=RoomPolicy(),
        )

        decision = await MentionSelector().select(request)

        assert decision.verdict is SelectionVerdict.SPEAK
        assert decision.speaker_id == "critic-1", (
            "the human addressed critic; the join line's @scout is the room's own text"
        )

    @pytest.mark.asyncio
    async def test_an_agents_join_line_does_not_count_as_having_spoken(self) -> None:
        """An agent that joined after being addressed still owes an answer.

        `_outstanding_addresses` treats any non-human row since the last human message as
        that participant having spoken. A join line would retire the address without a
        word being said.
        """
        from uclone_x.room.models import RoomMessageKind

        transcript = (
            RoomMessage(seq=1, sender_id="alice", content="@scout take a look"),
            RoomMessage(
                seq=2, sender_id="scout", content="scout joined the room", kind=RoomMessageKind.JOIN
            ),
        )
        request = SpeakerRequest(
            room_id="r1",
            participants=(ALICE, SCOUT, CRITIC),
            transcript=transcript,
            turn_state=TurnState(),
            policy=RoomPolicy(),
        )

        decision = await MentionSelector().select(request)

        assert decision.verdict is SelectionVerdict.SPEAK
        assert decision.speaker_id == "scout"

    @pytest.mark.asyncio
    async def test_a_trailing_join_line_does_not_re_open_an_answered_address(self) -> None:
        """The rule guard asks for the last *utterance*, not the last row.

        `_answers_a_human_message` decides whether an address is still outstanding. With a
        join line as the final row, reading `transcript[-1]` sees a human's id and reports
        an outstanding address, so `SoleAgentSelector` re-elects the same speaker — the
        re-firing defect of §3.4.1, reintroduced through the new row type.

        Killed by: src/uclone_x/room/selectors.py :: last = next((m for m in reversed(request.transcript) if m.is_utterance), None)
        Becomes: last = next((m for m in reversed(request.transcript)), None)
        """
        from uclone_x.room.models import RoomMessageKind

        transcript = (
            RoomMessage(seq=1, sender_id="alice", content="hello"),
            RoomMessage(seq=2, sender_id="scout", content="hi back"),
            RoomMessage(
                seq=3, sender_id="alice", content="alice joined the room", kind=RoomMessageKind.JOIN
            ),
        )
        request = SpeakerRequest(
            room_id="r1",
            participants=(ALICE, SCOUT),
            transcript=transcript,
            turn_state=TurnState(),
            policy=RoomPolicy(),
        )

        decision = await SoleAgentSelector().select(request)

        assert decision.verdict is SelectionVerdict.ABSTAIN


async def _consult(
    chain: tuple[SpeakerSelectorProtocol, ...], request: SpeakerRequest
) -> SpeakerDecision:
    """Run a selector chain the way `RoomOrchestrator` does: first non-abstention wins."""
    decision = SpeakerDecision(verdict=SelectionVerdict.ABSTAIN, selector="none")
    for selector in chain:
        decision = await selector.select(request)
        if decision.verdict is not SelectionVerdict.ABSTAIN:
            return decision
    return decision


class TestTheChainAgreesAboutAnAddress:
    """The two rule guards must not let one selector answer mail another would refuse.

    `SoleAgentSelector` and `DefaultResponderSelector` ask `_answers_a_human_message`;
    `MentionSelector` asks `_spoke_since_the_last_human`. That split is principled — a
    mention may legitimately fire more than once per human message, and the blunt guard is
    too coarse for it. What neither guard says anything about is *ordering*, and ordering
    is where the two disagreed: a selector that never looks at an address ran in front of
    the only one that does.
    """

    @pytest.mark.asyncio
    async def test_an_unresolvable_mention_is_refused_in_a_one_agent_room(self) -> None:
        r"""§3.5, and Revision 1 §4.1.3: an unrecognised address is never rerouted.

        In the room shape §3.4.1 calls the most common — one human, one agent —
        `SoleAgentSelector` stood in front of `MentionSelector` and answered without
        reading the address at all, so `@phantm help` was served by `scout`. That is
        exactly the routing of an unrecognised address to a default agent that §3.5
        refuses, with the roster refusal sitting unreachable behind it.

        The recorded mutation is the single-line form the fitness check can replay: the
        chain's address-reading link displaced by the rule that ignores addresses.

        Killed by: src/uclone_x/room/selectors.py :: MentionSelector(),
        Becomes: SoleAgentSelector(),
        """
        chain = build_selector_chain(RoomPolicy())
        request = _request(ALICE, SCOUT, utterances=(("alice", "@phantm help"),))
        with pytest.raises(UnknownRoomParticipantError):
            await _consult(chain, request)

    @pytest.mark.asyncio
    async def test_a_one_agent_room_still_answers_a_message_with_no_address(self) -> None:
        """The ordering fix must not cost the cheap case its answer."""
        decision = await _consult(build_selector_chain(RoomPolicy()), _request(ALICE, SCOUT))
        assert decision.verdict is SelectionVerdict.SPEAK
        assert decision.speaker_id == "scout"

    @pytest.mark.asyncio
    async def test_a_default_responder_does_not_answer_an_unresolvable_address(self) -> None:
        """The same shadowing, reached through link 3 rather than link 1.

        `DefaultResponderSelector` already runs behind `MentionSelector`, so this passes on
        the unfixed chain too. It is here because the ordering above is the only thing
        holding it, and a later reshuffle that puts the designated responder first would
        re-open the defect in the room shape where it is least visible.
        """
        policy = RoomPolicy(default_responder_id="scout")
        request = SpeakerRequest(
            room_id="r1",
            participants=(ALICE, SCOUT, CRITIC),
            transcript=(RoomMessage(seq=1, sender_id="alice", content="@phantm help"),),
            turn_state=TurnState(),
            policy=policy,
        )
        with pytest.raises(UnknownRoomParticipantError):
            await _consult(build_selector_chain(policy), request)


class TestAutoRoutingUclone2Parity:
    """uclone2 parity: autonomous routing by persona matching when no mention is present."""

    @pytest.mark.asyncio
    async def test_auto_routing_appends_llm_selector_when_provider_given(self) -> None:
        llm = MockLLMConnector()
        chain = build_selector_chain(RoomPolicy(), provider=llm)
        assert any(isinstance(s, LLMSpeakerSelector) for s in chain)

    @pytest.mark.asyncio
    async def test_auto_routing_disabled_excludes_llm_selector(self) -> None:
        llm = MockLLMConnector()
        chain = build_selector_chain(RoomPolicy(auto_routing=False), provider=llm)
        assert not any(isinstance(s, LLMSpeakerSelector) for s in chain)

    @pytest.mark.asyncio
    async def test_uclone2_field_aliases_in_selector_reply(self) -> None:
        """uclone2's SlowRouter JSON output schema (selected_bot_id, probability) is accepted."""
        llm = MockLLMConnector(
            responses=[
                '{"selected_bot_id": "critic-1", "probability": 0.85, "reasoning": "uclone2 match"}'
            ]
        )
        decision = await LLMSpeakerSelector(llm).select(_request(ALICE, SCOUT, CRITIC))
        assert decision.speaker_id == "critic-1"
        assert decision.confidence == 0.85
        assert decision.reasoning == "uclone2 match"

    @pytest.mark.asyncio
    async def test_multi_agent_unaddressed_message_routes_to_best_persona(self) -> None:
        llm = MockLLMConnector(
            responses=[
                '{"speaker_id": "scout", "confidence": 0.95, "reasoning": "scout fits research"}'
            ]
        )
        chain = build_selector_chain(RoomPolicy(), provider=llm)
        decision = await _consult(
            chain, _request(ALICE, SCOUT, CRITIC, utterances=(("alice", "investigate the index"),))
        )
        assert decision.verdict is SelectionVerdict.SPEAK
        assert decision.speaker_id == "scout"
        assert decision.reasoning == "scout fits research"


class TestAMentionIsNotEveryAtSign:
    """A token the human never meant as an address must not abort their message.

    An unresolvable mention raises and refuses the whole utterance (§3.5), which makes
    every false positive of `MENTION_PATTERN` a hard failure rather than a misroute — the
    same shape as the trailing-period defect that section already had to fix once.
    """

    @pytest.mark.asyncio
    async def test_a_handle_inside_a_url_is_not_an_address(self) -> None:
        r"""Pasting a link must not refuse the message that carries it.

        Killed by: src/uclone_x/room/selectors.py :: (?<![\w@/])@
        Becomes: (?<![\w@])@
        """
        for text in (
            "see https://github.com/@octocat for the fix",
            "https://medium.com/@phantom/post",
        ):
            request = _request(ALICE, SCOUT, CRITIC, utterances=(("alice", text),))
            decision = await MentionSelector().select(request)
            assert decision.verdict is SelectionVerdict.ABSTAIN, text

    @pytest.mark.asyncio
    async def test_punctuation_after_a_bare_at_sign_is_not_an_address(self) -> None:
        r"""`@.` and `@-` resolved to nobody, and so refused the whole utterance.

        Killed by: src/uclone_x/room/selectors.py :: @(\w[\w.\-]*)
        Becomes: @([\w.\-]+)
        """
        for text in ("priced at 5 @. each", "a range 3 @- 4", "see @... below"):
            request = _request(ALICE, SCOUT, CRITIC, utterances=(("alice", text),))
            decision = await MentionSelector().select(request)
            assert decision.verdict is SelectionVerdict.ABSTAIN, text

    @pytest.mark.asyncio
    async def test_the_forms_the_design_calls_addresses_still_are(self) -> None:
        """The two guards above must not take the bracketed and emphasised forms with them."""
        for text in ("(@scout)", "**@scout**", "- @scout", "@scout's idea", "@scout."):
            request = _request(ALICE, SCOUT, CRITIC, utterances=(("alice", f"x {text} y"),))
            decision = await MentionSelector().select(request)
            assert decision.speaker_id == "scout", text

    @pytest.mark.asyncio
    async def test_an_email_address_is_still_not_a_mention(self) -> None:
        for text in ("me@scout.example", "@@scout", "10@scout"):
            request = _request(ALICE, SCOUT, CRITIC, utterances=(("alice", text),))
            decision = await MentionSelector().select(request)
            assert decision.verdict is SelectionVerdict.ABSTAIN, text


class TestAnAddressIsReadWhole:
    """A participant id is not ASCII by construction, and a prefix of one is not it."""

    @pytest.mark.asyncio
    async def test_a_non_ascii_id_can_be_addressed(self) -> None:
        r"""An ASCII-only token class made a non-ASCII participant unreachable, in silence.

        Nothing constrains `Participant.id` to ASCII, so a room can hold an agent no
        mention can name: the pattern matches no token, `MentionSelector` abstains, and
        the address is dropped without a word — the failure mode §3.5 spends its length
        refusing everywhere else.

        Killed by: src/uclone_x/room/selectors.py :: @(\w[\w.\-]*)
        Becomes: @([A-Za-z0-9][A-Za-z0-9_.\-]*)
        """
        jeong = Participant(
            id="정찰",
            kind=ParticipantKind.AGENT,
            display_name="Jeongchal",
            session_id="sess_room__r1__jeong",
        )
        request = _request(ALICE, jeong, CRITIC, utterances=(("alice", "@정찰 봐줘"),))
        decision = await MentionSelector().select(request)
        assert decision.verdict is SelectionVerdict.SPEAK
        assert decision.speaker_id == "정찰"

    @pytest.mark.asyncio
    async def test_an_address_is_not_truncated_into_a_different_participant(self) -> None:
        r"""`@scouté` is not `@scout`, and answering as scout answers mail addressed elsewhere.

        The ASCII class stopped at the accent and handed the truncated prefix to
        `_resolve`, which found a real participant — a misroute the human cannot see,
        where §3.5 requires a refusal naming the roster.

        Killed by: src/uclone_x/room/selectors.py :: @(\w[\w.\-]*)
        Becomes: @([A-Za-z0-9][A-Za-z0-9_.\-]*)
        """
        request = _request(ALICE, SCOUT, CRITIC, utterances=(("alice", "@scouté please"),))
        with pytest.raises(UnknownRoomParticipantError) as excinfo:
            await MentionSelector().select(request)
        assert "scouté" in str(excinfo.value)


class TestTranscriptHelpersAtTheEdges:
    """`_last_human_utterance` and `_spoke_since_the_last_human` over degenerate windows."""

    @pytest.mark.asyncio
    async def test_no_rule_selector_decides_on_an_empty_transcript(self) -> None:
        """A room with nothing said in it has no address to answer, and must not raise."""
        policy = RoomPolicy(default_responder_id="scout")
        request = SpeakerRequest(
            room_id="r1",
            participants=(ALICE, SCOUT),
            transcript=(),
            turn_state=TurnState(),
            policy=policy,
        )
        for selector in (SoleAgentSelector(), MentionSelector(), DefaultResponderSelector()):
            decision = await selector.select(request)
            assert decision.verdict is SelectionVerdict.ABSTAIN, selector.name

    @pytest.mark.asyncio
    async def test_a_window_holding_no_human_message_decides_nothing(self) -> None:
        """Every rule reads from a human's words; a window with none has nothing to read."""
        request = _request(
            ALICE,
            SCOUT,
            CRITIC,
            utterances=(("scout", "@critic still there?"), ("critic-1", "yes")),
        )
        for selector in (SoleAgentSelector(), MentionSelector(), DefaultResponderSelector()):
            decision = await selector.select(request)
            assert decision.verdict is SelectionVerdict.ABSTAIN, selector.name

    @pytest.mark.asyncio
    async def test_a_failed_turn_retires_the_address_it_failed_to_answer(self) -> None:
        """A turn that errored counts as having held the floor. Deliberate, and pinned here.

        `RoomOrchestrator._run_turn` records a failure as an utterance with empty content
        and `error` set, so `_spoke_since_the_last_human` sees the speaker. Reading that
        row instead as "has not spoken" would re-elect a persistently failing agent for
        every remaining turn of the budget — the loop §3.4.1 exists to prevent — and
        §3.13's `retry` is the way back, not another selection pass.

        Killed by: src/uclone_x/room/selectors.py :: if not message.is_utterance:
        Becomes: if not message.is_utterance or message.error:
        """
        transcript = (
            RoomMessage(seq=1, sender_id="alice", content="@scout @critic compare"),
            RoomMessage(seq=2, sender_id="scout", content="", error="RuntimeError: boom"),
        )
        request = SpeakerRequest(
            room_id="r1",
            participants=(ALICE, SCOUT, CRITIC),
            transcript=transcript,
            turn_state=TurnState(),
            policy=RoomPolicy(),
        )
        decision = await MentionSelector().select(request)
        assert decision.speaker_id == "critic-1"

    @pytest.mark.asyncio
    async def test_the_same_agent_named_twice_answers_once(self) -> None:
        """A repeated address is one address; it must not buy a second turn."""
        first = _request(ALICE, SCOUT, CRITIC, utterances=(("alice", "@scout @scout go"),))
        assert (await MentionSelector().select(first)).speaker_id == "scout"

        after = _request(
            ALICE,
            SCOUT,
            CRITIC,
            utterances=(("alice", "@scout @scout go"), ("scout", "done")),
        )
        assert (await MentionSelector().select(after)).verdict is SelectionVerdict.ABSTAIN


_MALFORMED_REPLIES: Final = (
    ('{"decision": {"speaker_id": "scout"}}', "raise"),
    ('{"speaker_id": "scout"} {"speaker_id": "critic-1"}', "raise"),
    ('{"speaker_id": 7}', "raise"),
    ('{"speaker_id": ["scout"]}', "raise"),
    ('{"speaker_id": true}', "raise"),
    ('{"speaker_id": "scout", "confidence": 5}', "raise"),
    ('{"speaker_id": "scout", "confidence": -1}', "raise"),
    ('{"speaker_id": "scout", "confidence": "high"}', "raise"),
    ('{"speaker_id": "scout", "confidence": null}', "raise"),
    ('{"speaker_id": "r1"}', "raise"),
    ('{"speaker_id": "critic"}', "raise"),
    ('{"speaker_id": "@scout"}', "raise"),
    ('I think {like this}: {"speaker_id": "scout"}', "raise"),
    ('{"speaker_id": "scout"', "raise"),
    ("null", "raise"),
    ('[{"speaker_id": "scout", "confidence": 1.0}]', "speak"),
    ('{"speaker_id": "  scout  "}', "speak"),
    ('{"speaker_id": "scout", "reasoning": "use {} here"}', "speak"),
    ('{"speaker_id": "scout", "reasoning": "' + "x" * 100_000 + '"}', "speak"),
)


class TestNoMalformedReplyBecomesASilentSilence:
    """The P6 line, held across every reply shape worth constructing.

    `SILENCE` is a decision the room records and renders (§3.14). A reply the selector
    cannot read is not that decision, and the prior art this design replaces is exactly
    where the two were one value — so what this table asserts is the *absence* of
    `SILENCE`, not merely the presence of an error. `"critic"` is in it because an alias
    resolves a human's mention and must not resolve a model's answer: the model is given
    ids and is held to them.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("reply", "expected"), _MALFORMED_REPLIES)
    async def test_shape(self, reply: str, expected: str) -> None:
        """One reply shape, held to raising or to a named speaker.

        Killed by: src/uclone_x/room/selectors.py :: confidence: float = Field(default=1.0, ge=0.0, le=1.0)
        Becomes: confidence: float = Field(default=1.0)
        """
        llm = MockLLMConnector(responses=[reply])
        request = _request(ALICE, SCOUT, CRITIC)
        if expected == "raise":
            with pytest.raises(SpeakerSelectionError):
                await LLMSpeakerSelector(llm).select(request)
            return
        decision = await LLMSpeakerSelector(llm).select(request)
        assert decision.verdict is SelectionVerdict.SPEAK
        assert decision.speaker_id == "scout"


class TestLLMSelectorAutonomousAndThinking:
    """Verify autonomous mode rendering and thinking=False parameter on LLMRequest."""

    @pytest.mark.asyncio
    async def test_llm_selector_requests_no_thinking_tokens(self) -> None:
        recorded: list[LLMRequest] = []

        class RecordingMock(MockLLMConnector):
            async def generate(self, request: LLMRequest) -> ModelResponse:
                recorded.append(request)
                return await super().generate(request)

        llm = RecordingMock(responses=['{"speaker_id": "scout"}'])
        request = _request(ALICE, SCOUT, CRITIC)
        selector = LLMSpeakerSelector(llm)
        await selector.select(request)
        assert len(recorded) == 1
        assert recorded[0].thinking is False

    def test_llm_selector_renders_autonomous_mode(self) -> None:
        request_normal = _request(ALICE, SCOUT, CRITIC, policy=RoomPolicy(autonomous=False))
        rendered_normal = LLMSpeakerSelector._render(request_normal)  # pyright: ignore[reportPrivateUsage]
        assert "AUTONOMOUS DISCUSSION MODE: ENABLED" not in rendered_normal

        request_auto = _request(ALICE, SCOUT, CRITIC, policy=RoomPolicy(autonomous=True))
        rendered_auto = LLMSpeakerSelector._render(request_auto)  # pyright: ignore[reportPrivateUsage]
        assert "AUTONOMOUS DISCUSSION MODE: ENABLED" in rendered_auto

    @pytest.mark.asyncio
    async def test_llm_selector_falls_back_to_thinking_for_json(self) -> None:
        class ThinkingMockLLM(MockLLMConnector):
            async def generate(self, request: LLMRequest) -> ModelResponse:
                resp = await super().generate(request)
                return resp.model_copy(
                    update={
                        "content": "I think scout should speak next.",
                        "thinking": '{"speaker_id": "scout", "confidence": 1.0, "reasoning": "scout is ready"}',
                    }
                )

        llm = ThinkingMockLLM()
        request = _request(ALICE, SCOUT, CRITIC)
        decision = await LLMSpeakerSelector(llm).select(request)
        assert decision.verdict is SelectionVerdict.SPEAK
        assert decision.speaker_id == "scout"
