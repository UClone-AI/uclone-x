"""The selectors that decide who speaks next, cheapest first.

Three implementations of `SpeakerSelectorProtocol`, meant to be chained in this order:

1. `SoleAgentSelector` — a room with one agent has nothing to decide. No model call.
2. `MentionSelector` — the human named someone. No model call.
3. `LLMSpeakerSelector` — nobody was named and several agents could answer. One model call.

The ordering is the point. A selector that answers without a model answers in
microseconds and cannot hallucinate a speaker, so every rule that can be stated as a rule
is stated as one and runs before the model is reached. Only the genuinely open case — an
unaddressed message in a room of several agents — spends a call.

Each selector abstains rather than guesses, and raises rather than abstains when it is
itself broken. `SelectionVerdict` documents why those three outcomes are distinct.
"""

from __future__ import annotations

import json
import re
from typing import Final, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from uclone_x.errors import SpeakerSelectionError, UnknownRoomParticipantError
from uclone_x.llm.models import ChatMessage, LLMRequest, MessageRole
from uclone_x.llm.protocols import LLMProviderProtocol
from uclone_x.room.models import (
    Participant,
    ParticipantKind,
    RoomPolicy,
    SelectionVerdict,
    SpeakerDecision,
    SpeakerRequest,
)
from uclone_x.room.protocols import SpeakerSelectorProtocol

__all__ = [
    "DEFAULT_SELECTOR_MAX_TOKENS",
    "DefaultResponderSelector",
    "LLMSpeakerSelector",
    "MENTION_PATTERN",
    "build_selector_chain",
    "MentionSelector",
    "SoleAgentSelector",
]

MENTION_PATTERN: Final = re.compile(r"(?<![\w@/])@(\w[\w.\-]*)")
r"""Mention tokens in an utterance.

A negative lookbehind rather than a whitespace anchor. The anchor form rejected the same
addresses a reader would call obvious — `(@scout)`, `**@scout**`, `- @scout` — and silently,
since an unmatched mention just falls through to the next selector. The lookbehind still
refuses `me@scout.example`, because the character before the `@` there is a word character,
which is the case the guard exists for; `@@scout` is refused for the same reason.

**Every false positive here is a hard failure, not a misroute**, because §3.5 makes an
unresolvable mention refuse the whole utterance. That asymmetry is what the rest of this
pattern is shaped by:

* `/` joins `\w` and `@` in the lookbehind. `https://github.com/@octocat` is a link a
  human pastes, and reading `@octocat` out of it raised `UnknownRoomParticipantError` and
  threw the message away — the trailing-period defect of §3.5 arriving through a URL
  instead of a full stop. A query parameter (`?ref=@scout`) is still read as an address;
  the common paste is the path form, and full URL parsing in a mention scanner buys less
  than it costs.
* The token must **start** with a word character, so `@.`, `@-` and `@...` in ordinary
  prose are not tokens at all. `.` and `-` stay legal *inside* an id (`@scout.v2`).
* `\w` rather than `[A-Za-z0-9_]`, because nothing constrains `Participant.id` to ASCII.
  The ASCII class had two failures and both were silent: a participant id such as `정찰`
  matched no token and was unreachable, and `@scouté` matched the prefix `scout`, which
  `_resolve` then found — answering in a different participant's voice. Under `\w` the
  first resolves and the second refuses with the roster, which is the answer §3.5 asks
  for. The cost is that a mention run together with non-ASCII text (`@scout봐줘`) now
  refuses rather than dispatching; a refusal naming the roster is the outcome this design
  prefers to a guess.

Kept module-level and exported because the head highlights the same tokens it routes on,
and two regexes drift.
"""

_MENTION_TRAILING: Final = ".-"
"""Characters stripped from the end of a mention token before a second resolution attempt.

`.` and `-` are legal *inside* an id — `@scout.v2` must resolve — and are also how a
sentence ends. Resolution therefore tries the token as written first, so a dotted id keeps
its dot, and only then tries it trimmed. Without this, `please help @scout.` raised
`UnknownRoomParticipantError` and aborted the whole message: §3.5 made an unresolvable
mention refuse, which turned ordinary punctuation into a hard failure.
"""


def _agents(participants: tuple[Participant, ...]) -> tuple[Participant, ...]:
    """The agent participants, in roster order."""
    return tuple(p for p in participants if p.kind is ParticipantKind.AGENT)


def _answers_a_human_message(request: SpeakerRequest) -> bool:
    """True when the most recent utterance is a human's.

    **The guard that keeps a rule from firing twice for one message**, used by the two rule
    selectors that can produce at most one turn per human message. A rule selector is a
    stateless function of the room, so nothing about it changes between the turn it decides
    and the next round of the loop: `SoleAgentSelector` still sees one agent. Without this,
    such a rule re-elected the same speaker on every remaining turn, and the speaker was
    handed an empty span — its own reply having already advanced its high-water mark — so an
    ordinary one-agent room spent its whole turn budget and called the model twice with
    nothing to answer.

    **`MentionSelector` deliberately does not use it.** One address may name several agents
    and must produce a turn for each, so "the last utterance is a human's" is too coarse
    there; it asks `_spoke_since_the_last_human` which addresses are still outstanding
    instead. The two guards are not interchangeable, and a new selector must pick by asking
    whether it can legitimately fire more than once per human message.

    A rule answers *an address*. Once it has been answered, there is no address outstanding,
    and continuing the conversation is a judgement — which is `LLMSpeakerSelector`'s job, and
    why that selector deliberately does not consult this.

    The last **utterance**, not the last row: a membership row carries a participant's id as
    its sender, so `transcript[-1]` sees a human's id when a human is merely seated and
    reports an outstanding address that nobody made — re-electing the same speaker, which is
    the defect this guard exists to prevent, arriving through the new row type.
    """
    humans = {p.id for p in request.participants if p.kind is ParticipantKind.HUMAN}
    last = next((m for m in reversed(request.transcript) if m.is_utterance), None)
    return last is not None and last.sender_id in humans


def _spoke_since_the_last_human(request: SpeakerRequest) -> frozenset[str]:
    """Ids that have spoken after the most recent human utterance.

    **What keeps `MentionSelector` from re-firing, now that it may fire more than once.**
    `@scout @critic compare` names two agents and must produce two turns, so the blunt
    guard — decide only when the last utterance is a human's — is too coarse here: it
    answered the first address and dropped the second silently, which is what a user
    reported as "I asked two and one replied". The sharper question is which addresses are
    still outstanding, and the transcript answers it without the selector holding any
    state of its own.

    Membership rows are skipped rather than counted: an agent's join line is not that agent
    having spoken, and treating it as such retires an address the agent still owes an answer
    to — the quiet drop this helper was written to end.
    """
    spoken: set[str] = set()
    humans = {p.id for p in request.participants if p.kind is ParticipantKind.HUMAN}
    for message in reversed(request.transcript):
        if not message.is_utterance:
            continue
        if message.sender_id in humans:
            break
        spoken.add(message.sender_id)
    return frozenset(spoken)


def _last_human_utterance(request: SpeakerRequest) -> str | None:
    """Text of the most recent human message in the window, or `None` if there is none.

    Mentions are read from the human's own words only. An agent that emits `@critic` in
    its reply must not thereby dispatch another agent: that is a prompt-injection route
    into the room's control plane, and an agent-to-agent loop with no human in it.
    """
    humans = {p.id for p in request.participants if p.kind is ParticipantKind.HUMAN}
    for message in reversed(request.transcript):
        if message.is_utterance and message.sender_id in humans:
            return message.content
    return None


class SoleAgentSelector:
    """Routes to the only agent in the room.

    The cheapest rule there is, and the one that matters most in practice: the ordinary
    one-human-one-agent conversation is a room with nothing to route, and it must not pay
    a model call to discover that. Abstains as soon as a second agent joins.
    """

    @property
    def name(self) -> str:
        return "sole_agent"

    async def select(self, request: SpeakerRequest) -> SpeakerDecision:
        """Name the single agent participant, or abstain."""
        if not _answers_a_human_message(request):
            return SpeakerDecision(
                verdict=SelectionVerdict.ABSTAIN,
                selector=self.name,
                reasoning="the last utterance was not a human's; this rule answers addresses",
            )
        agents = _agents(request.participants)
        if len(agents) != 1:
            return SpeakerDecision(
                verdict=SelectionVerdict.ABSTAIN,
                selector=self.name,
                reasoning=f"{len(agents)} agents in the room; nothing to decide by this rule",
            )
        return SpeakerDecision(
            verdict=SelectionVerdict.SPEAK,
            speaker_id=agents[0].id,
            selector=self.name,
            reasoning="the room's only agent",
        )


class MentionSelector:
    """Routes to the participant the human addressed by name.

    Resolution is `id` first, then `aliases`, both case-insensitively. When several
    participants are named, **every addressed agent answers once, in mention order**: the
    outstanding addresses are read off the transcript rather than held in the selector, so
    it fires once per address and abstains when the list empties. An unresolvable name
    refuses the whole address before any of it is served.
    """

    @property
    def name(self) -> str:
        return "mention"

    async def select(self, request: SpeakerRequest) -> SpeakerDecision:
        """Name the next unanswered addressed agent, abstain when none is outstanding.

        Raises:
            UnknownRoomParticipantError: A mention names nobody in the room. Refusing is
                deliberate: routing an unrecognised address to a default agent answers in
                a voice the human did not ask for, and does so invisibly.
        """
        utterance = _last_human_utterance(request)
        if utterance is None:
            return SpeakerDecision(
                verdict=SelectionVerdict.ABSTAIN,
                selector=self.name,
                reasoning="no human utterance in the window to read an address from",
            )

        tokens = MENTION_PATTERN.findall(utterance)
        if not tokens:
            return SpeakerDecision(
                verdict=SelectionVerdict.ABSTAIN,
                selector=self.name,
                reasoning="no mention in the last human utterance",
            )

        # **Every token is resolved before any is chosen.** Refusing lazily, as the loop
        # that picked and returned in one pass did, made the refusal depend on where the
        # bad name sat: `@phantm @scout` raised on the first pass and wrote nothing, while
        # `@scout @phantm` gave scout the floor and only raised on the pass after — so the
        # caller got an exception *and* lost the state for a turn that had really happened.
        # One utterance class, two incompatible outcomes. An address is decided whole.
        addressed: list[tuple[str, Participant]] = []
        for token in tokens:
            resolved = self._resolve(token, request.participants)
            if resolved is None:
                roster = ", ".join(
                    f"@{p.id}" + (f" ({'/'.join(p.aliases)})" if p.aliases else "")
                    for p in request.participants
                )
                raise UnknownRoomParticipantError(
                    f"Room {request.room_id!r} has no participant matching @{token}. "
                    f"Participants: {roster or '(none)'}"
                )
            addressed.append((token, resolved))

        answered = _spoke_since_the_last_human(request)
        for token, resolved in addressed:
            if resolved.kind is ParticipantKind.AGENT and resolved.id not in answered:
                return SpeakerDecision(
                    verdict=SelectionVerdict.SPEAK,
                    speaker_id=resolved.id,
                    selector=self.name,
                    reasoning=f"addressed directly as @{token}",
                )

        # Every address has been answered, or all of them named humans. A human addressing
        # another human is a real utterance in a shared room and not an instruction to any
        # agent, so this abstains and lets a later selector decide whether anyone joins in.
        return SpeakerDecision(
            verdict=SelectionVerdict.ABSTAIN,
            selector=self.name,
            reasoning=(
                "every addressed agent has answered"
                if answered
                else "the addressed participants are human"
            ),
        )

    @classmethod
    def _resolve(cls, token: str, participants: tuple[Participant, ...]) -> Participant | None:
        """Match `token` against ids, then aliases; retry once with trailing punctuation off.

        Ids win over aliases so an alias cannot hijack mail addressed to another
        participant's real id, and the untrimmed token is tried first so a dotted id keeps
        its dot.
        """
        for candidate in cls._candidates(token):
            lowered = candidate.lower()
            for participant in participants:
                if participant.id.lower() == lowered:
                    return participant
            for participant in participants:
                if any(alias.lower() == lowered for alias in participant.aliases):
                    return participant
        return None

    @staticmethod
    def _candidates(token: str) -> tuple[str, ...]:
        """The token as written, then the same token with trailing `.`/`-` removed."""
        trimmed = token.rstrip(_MENTION_TRAILING)
        if trimmed and trimmed != token:
            return (token, trimmed)
        return (token,)


class _SelectorReply(BaseModel):
    """The selector's wire reply, validated rather than hand-parsed.

    `extra="ignore"` and not `"forbid"`, uniquely in this package: a model that adds a
    stray commentary key has still answered the question, and rejecting the whole reply
    for it would convert formatting noise into a routing failure. Every field a decision
    depends on is still strictly typed.
    """

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _coerce_uclone2_fields(cls, data: object) -> object:
        if isinstance(data, dict):
            raw_dict = cast(dict[object, object], data)
            coerced: dict[str, object] = {str(k): v for k, v in raw_dict.items()}
            if "speaker_id" not in coerced:
                if "selected_bot_id" in coerced:
                    coerced["speaker_id"] = coerced["selected_bot_id"]
                elif "bot_id" in coerced:
                    coerced["speaker_id"] = coerced["bot_id"]
            if "confidence" not in coerced and "probability" in coerced:
                coerced["confidence"] = coerced["probability"]
            return coerced
        return data

    speaker_id: str | None = Field(
        description="Required, with no default. An explicit `null` is how the model says "
        "nobody should speak; a *missing* key is a malformed reply and must raise. With a "
        "default of `None` the two were the same value, so a reply that named a speaker "
        'under a slightly wrong key — `{"speaker": "critic"}` — was read as a decided '
        "silence. That is the exact conflation `SelectionVerdict` exists to prevent, "
        "reintroduced one layer down."
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: str = ""


class DefaultResponderSelector:
    """Routes an unaddressed message to the room's designated responder.

    Third in the chain and the reason the chain rarely reaches a model: in a room whose
    operator picked the participants, the common unaddressed message has an obvious
    recipient, and naming it in policy answers without a call.

    **Nothing is defaulted.** With no `default_responder_id` set this abstains, rather than
    picking the first agent or a conventional name. A responder nobody chose answers in a
    voice the operator did not pick, and the human cannot see that their message was
    routed by a guess — the defect this selector was written to replace.
    """

    @property
    def name(self) -> str:
        return "default_responder"

    async def select(self, request: SpeakerRequest) -> SpeakerDecision:
        """Name the configured responder, or abstain if none is configured.

        Raises:
            SpeakerSelectionError: A responder is configured but is not an agent of this
                room — a policy that names a participant who left, or names a human. Not
                an abstention: the operator asked for something the room cannot honour,
                and falling through would hide a misconfiguration behind whichever
                selector runs next.
        """
        if not _answers_a_human_message(request):
            return SpeakerDecision(
                verdict=SelectionVerdict.ABSTAIN,
                selector=self.name,
                reasoning="the last utterance was not a human's; this rule answers addresses",
            )
        responder_id = request.policy.default_responder_id
        if not responder_id:
            return SpeakerDecision(
                verdict=SelectionVerdict.ABSTAIN,
                selector=self.name,
                reasoning="no default responder configured for this room",
            )

        match = next((p for p in request.participants if p.id == responder_id), None)
        if match is None or match.kind is not ParticipantKind.AGENT:
            roster = ", ".join(p.id for p in _agents(request.participants))
            raise SpeakerSelectionError(
                f"Room {request.room_id!r} names {responder_id!r} as its default "
                f"responder, but it is not an agent of the room (agents: {roster or 'none'})"
            )

        return SpeakerDecision(
            verdict=SelectionVerdict.SPEAK,
            speaker_id=match.id,
            selector=self.name,
            reasoning="the room's designated responder for unaddressed messages",
        )


DEFAULT_SELECTOR_MAX_TOKENS: Final[int] = 2048
"""Default max output tokens for speaker selection classification.

Reasoning and thinking models (such as Qwen3, DeepSeek-R1) generate internal thinking tokens
before emitting the final JSON payload. A tight token ceiling (e.g. 256 or 1024) causes thinking tokens
to exhaust the entire budget before the JSON payload is emitted, leading to empty replies and
selection crashes. 2048 tokens provides ample room for reasoning followed by the compact JSON.
"""

_SYSTEM_PROMPT: Final = """\
You allocate the floor in a multi-agent chat room. Read the roster and the recent \
conversation, then name the ONE participant who should speak next, or decide that nobody \
should.

Rules:
- Choose the agent whose stated purpose best fits what was last said.
- If the human user greets, asks a question, or addresses the room (e.g. '안녕', '누구 있니?', '아무도 없니?'), \
an agent MUST respond. If no specific specialist is indicated, pick the primary personal clone/partner (e.g. clone) \
or the first available agent.
- If the human user tells agents to converse among themselves (e.g. '니들끼리 이야기 해봐', '대화 나눠봐', '토론해봐', 'talk among yourselves', 'free discussion'), \
an agent MUST respond to initiate or continue the discussion.
- In AUTONOMOUS DISCUSSION MODE (when enabled), agents are expected to maintain an active, collaborative dialogue with each other. \
Select an appropriate agent to respond, build upon the previous speaker's ideas, ask follow-up questions, or offer alternative perspectives. \
Never choose nobody/silence merely because the human is silent or the last speaker was an agent.
- If the previous turn's speaker was an agent who asked a question, proposed a topic/idea, or invited collaboration \
(e.g. '어떤 이야기를 시작할까요?', '어떻게 생각하세요?', '이 아이디어는 어때?'), you MUST select another agent \
to answer and continue the collaboration. Never choose nobody/silence when an open question or cooperative proposal \
between agents remains unanswered.
- Choose nobody only when the exchange has reached a natural resting point where all questions have been answered, \
when the agents are repeating each other, or when the last message needs no answer. Never choose nobody if a human \
message or an agent's open collaborative question is still unanswered.
- Never choose a human participant, and never choose the agent that spoke last unless it was asked a direct follow-up question.
- Do not answer the conversation yourself. Emit only the JSON object.

Reply with exactly this JSON object and nothing else:
{"speaker_id": "<participant id, or null for nobody>", "confidence": <0.0-1.0>, \
"reasoning": "<one short sentence>"}"""


class LLMSpeakerSelector:
    """Asks a model who should speak next.

    Last in the chain and the only selector that costs a call, so it runs on the one case
    the rules cannot settle. Its model is configured separately from any participant's
    (`RoomPolicy.selector_llm`): this is a short classification over a bounded window, and
    the room's reasoning models are the wrong instrument for it.

    **Every failure raises.** A provider error, an unparseable reply, a reply naming
    somebody who is not in the room — each is a `SpeakerSelectionError`, never a verdict
    of silence. This is the P6 line the prior art crossed: there, one `None` served as
    "nobody should speak", "ask the next router" and "the provider raised", so a dead
    selector and a quiet room were indistinguishable and the room went silent either way.
    """

    def __init__(
        self,
        provider: LLMProviderProtocol,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_SELECTOR_MAX_TOKENS,
    ) -> None:
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return "llm"

    async def select(self, request: SpeakerRequest) -> SpeakerDecision:
        """Ask the model, and hold its answer to the roster.

        Raises:
            SpeakerSelectionError: The provider failed, the reply would not parse, or it
                named a participant the room does not have.
        """
        agents = _agents(request.participants)
        if not agents:
            return SpeakerDecision(
                verdict=SelectionVerdict.ABSTAIN,
                selector=self.name,
                reasoning="no agents in the room",
            )
        if len(agents) == 1 and request.turn_state.last_speaker_id == agents[0].id:
            return SpeakerDecision(
                verdict=SelectionVerdict.ABSTAIN,
                selector=self.name,
                reasoning="the room's only agent has already spoken",
            )

        llm_request = LLMRequest(
            model=self._model,
            messages=(
                ChatMessage(role=MessageRole.SYSTEM, content=_SYSTEM_PROMPT),
                ChatMessage(role=MessageRole.USER, content=self._render(request)),
            ),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            thinking=False,
        )

        try:
            response = await self._provider.generate(llm_request)
        except Exception as exc:
            raise SpeakerSelectionError(
                f"Selector {self.name!r} could not reach a decision for room "
                f"{request.room_id!r}: provider {self._provider.provider_name!r} failed: {exc}"
            ) from exc

        speaker_source = response.content
        if (
            (speaker_source is None or "{" not in speaker_source)
            and response.thinking
            and "{" in response.thinking
        ):
            speaker_source = response.thinking
        elif speaker_source is None or not speaker_source.strip():
            if response.thinking:
                speaker_source = response.thinking

        speaker_id, confidence, reasoning = self._parse(
            speaker_source, request.room_id, response.model_name
        )

        if speaker_id is None:
            return SpeakerDecision(
                verdict=SelectionVerdict.SILENCE,
                confidence=confidence,
                selector=self.name,
                reasoning=reasoning,
                provenance=response.provenance,
            )

        if not any(agent.id == speaker_id for agent in agents):
            raise SpeakerSelectionError(
                f"Selector {self.name!r} named {speaker_id!r}, which is not an agent in "
                f"room {request.room_id!r} (agents: {', '.join(a.id for a in agents)}). "
                f"Model: {response.model_name}"
            )

        return SpeakerDecision(
            verdict=SelectionVerdict.SPEAK,
            speaker_id=speaker_id,
            confidence=confidence,
            selector=self.name,
            reasoning=reasoning,
            provenance=response.provenance,
        )

    @staticmethod
    def _render(request: SpeakerRequest) -> str:
        """Render the roster and window as the selector's user message."""
        lines = ["ROSTER:"]
        for participant in request.participants:
            kind = participant.kind.value
            purpose = participant.persona_summary or "(no stated purpose)"
            lines.append(f"- {participant.id} [{kind}] {participant.display_name}: {purpose}")

        last_speaker = request.turn_state.last_speaker_id
        lines.append("")
        if request.policy.autonomous:
            lines.append("AUTONOMOUS DISCUSSION MODE: ENABLED")
        lines.append(
            f"AGENT TURNS SINCE THE LAST HUMAN MESSAGE: {request.turn_state.agent_turns_since_human}"
        )
        lines.append(f"LAST SPEAKER: {last_speaker or '(none)'}")
        lines.append("")
        lines.append("CONVERSATION:")
        for message in request.transcript:
            lines.append(f"[{message.sender_id}]: {message.content}")
        return "\n".join(lines)

    def _parse(
        self, content: str | None, room_id: str, model_name: str
    ) -> tuple[str | None, float, str]:
        """Extract `(speaker_id, confidence, reasoning)` from the model's reply.

        A fenced or prose-wrapped JSON object is tolerated — the brace span is taken —
        because that is formatting noise rather than a different answer. Anything past
        that raises: a reply this cannot read is not a decision, and treating it as
        silence would invent one.
        """
        if content is None or not content.strip():
            raise SpeakerSelectionError(
                f"Selector {self.name!r} received an empty reply for room {room_id!r} "
                f"from model {model_name!r}"
            )

        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end <= start:
            raise SpeakerSelectionError(
                f"Selector {self.name!r} received a reply with no JSON object for room "
                f"{room_id!r} from model {model_name!r}: {content[:200]!r}"
            )

        try:
            parsed = _SelectorReply.model_validate_json(content[start : end + 1])
        except (ValidationError, json.JSONDecodeError) as exc:
            raise SpeakerSelectionError(
                f"Selector {self.name!r} received an unparseable reply for room "
                f"{room_id!r} from model {model_name!r}: {content[start : end + 1][:200]!r}"
            ) from exc

        if parsed.speaker_id is not None and not parsed.speaker_id.strip():
            # Blank is neither a name nor the explicit `null` that means silence.
            raise SpeakerSelectionError(
                f"Selector {self.name!r} received a blank speaker_id for room {room_id!r} "
                f"from model {model_name!r}; use null to decide on silence"
            )
        speaker = parsed.speaker_id.strip() if parsed.speaker_id else None
        if speaker is not None and speaker.lower() in ("null", "none", "nil", "nobody"):
            speaker = None
        reasoning = parsed.reasoning or f"selected by {model_name}"
        return speaker, parsed.confidence, reasoning


def build_selector_chain(
    policy: RoomPolicy,
    provider: LLMProviderProtocol | None = None,
    max_tokens: int = DEFAULT_SELECTOR_MAX_TOKENS,
) -> tuple[SpeakerSelectorProtocol, ...]:
    """The standard chain for a room, cheapest link first.

    Assembled in the Core rather than by each surface, so a head and a CLI cannot end up
    routing the same room differently — which is the divergence a chain order expressed
    twice eventually produces.

    **`MentionSelector` runs first, ahead of the cheaper `SoleAgentSelector`.** Links 1-3 are
    all free, so "cheapest first" ranks nothing between them, and ordering them by cost
    instead of by specificity put a selector that never reads an address in front of the only
    one that does. In the room shape §3.4.1 calls the most common — one human, one agent —
    `@phantm help` was therefore answered by the room's single agent: an unrecognised address
    rerouted to a default responder, invisibly, which is the behaviour §3.5 and Revision 1
    §4.1.3 both refuse and which `MentionSelector` alone raises on. A rule that reads the
    address decides before a rule that does not.

    The model link is active whenever autonomous routing is enabled (the default,
    `policy.auto_routing=True`) or `policy.selector_llm` is specified, *and* a provider is
    available. With links 1-3 in front of it an addressed message costs zero model calls,
    matching uclone2's FastRouter + SlowRouter design. When no provider is supplied, the
    room uses the rule chain and settles on silence when rules cannot decide.
    """
    chain: list[SpeakerSelectorProtocol] = [
        MentionSelector(),
        SoleAgentSelector(),
        DefaultResponderSelector(),
    ]
    if policy.auto_routing and provider is not None:
        model = policy.selector_llm.model_name if policy.selector_llm is not None else None
        temperature = policy.selector_llm.temperature if policy.selector_llm is not None else 0.0
        chain.append(
            LLMSpeakerSelector(
                provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
    return tuple(chain)
