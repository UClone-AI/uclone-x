"""Types for a multi-agent chat room: who is in it, what was said, and who speaks next.

**A room is a transcript, not a session.** The transcript holds *final utterances only* —
what a reader of the chat window sees. Every participating agent keeps its **own**
`SessionState` (`session_id` on `Participant`), its own tool traces, its own compaction
and its own ontology namespace. Nothing in this module is shared agent state; the room is
the one place the separate agents become mutually visible, and it is deliberately the
thinnest such place.

That split is the whole design, so it is worth naming what it rejects. The alternative —
one session that every agent reads and writes — was considered and dropped: it puts two
writers on one `SessionState`, which `SessionStore.save`'s revision precondition refuses,
and it would leak one agent's tool output and compaction ledger into every other agent's
context. Per-agent sessions cost a copy of the conversation per participant. That cost is
accepted and is not hidden: see `RoomPolicy.transcript_window`, which bounds it.

**Selection is a value, never an action.** A selector returns a `SpeakerDecision` and
notifies no one; the orchestrator acts on it. This keeps every selector a pure function of
the room, which is what makes the routing rules testable without a live agent.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from uclone_x.agent.models import AgentLLMConfig
from uclone_x.core.immutable import ImmutableStrMapping
from uclone_x.core.provenance import Provenance
from uclone_x.llm.models import TokenUsage

__all__ = [
    "Participant",
    "ParticipantKind",
    "RoomMessage",
    "RoomMessageKind",
    "RoomFileRecord",
    "RoomPolicy",
    "RoomState",
    "RoomToolUse",
    "RoomWrittenFile",
    "SelectionVerdict",
    "SpeakerDecision",
    "SpeakerRequest",
    "TurnState",
    "turn_refusal",
]


def _now_iso() -> str:
    """Current UTC instant as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


class ParticipantKind(StrEnum):
    """What kind of speaker a participant is.

    The distinction drives turn policy rather than presentation: a human utterance resets
    the bot-turn budget, an agent utterance spends it.
    """

    HUMAN = "human"
    AGENT = "agent"


class Participant(BaseModel):
    """One member of a room, and the per-member resources the room must keep apart.

    `session_id` and `ontology_namespace` are declared here rather than resolved later so
    that "each agent keeps its own session and its own knowledge graph" is a property of
    the room's *data*, checkable by reading a `RoomState`, instead of a convention each
    resolver is trusted to follow. `RoomAgentResolverProtocol` states the matching
    obligation on the runtime side.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: str = Field(description="Agent id, or the human's user id.")
    kind: ParticipantKind
    display_name: str
    persona_summary: str = Field(
        default="",
        description="One line describing what this participant is for. This is the only "
        "thing a selector is told about an agent, so it is a routing input and not "
        "decoration: an empty summary makes the participant effectively unroutable by "
        "any selector that reasons about fit.",
    )
    aliases: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Extra names this participant answers to in a mention, beyond `id` "
        "— a role word such as `critic`, so a human need not remember a generated agent "
        "id. Matched case-insensitively, after `id` and never before it: an alias that "
        "collides with another participant's id must not win, or renaming one "
        "participant's alias would silently redirect another's mail.",
    )
    session_id: str = Field(
        description="The participant's own session. Distinct per participant — two "
        "participants sharing one id would put two writers on one `SessionState`, which "
        "`SessionStore.save` refuses on its revision precondition. Empty for a human, "
        "who has no agent session.",
        default="",
    )
    ontology_namespace: str = Field(
        default="",
        description="Namespace IRI for this participant's own ontology engine (P7). "
        "Separate per agent: a shared namespace would merge one agent's induced concepts "
        "into another's, which is not a room feature but a loss of the per-agent "
        "grounding P7 requires. Empty for a human, and for an agent running without an "
        "ontology.",
    )
    persona: str = Field(
        default="",
        description="Name of registered persona blueprint to hydrate this participant with.",
    )


class RoomMessageKind(StrEnum):
    """What kind of row this is — speech, or a change to who is in the room.

    **A membership change had to be recorded somewhere, and every other place was worse.**
    The question a reader asks when an agent goes quiet is "why did critic stop replying?",
    and it is asked *of the conversation*, so the answer has to be in the conversation. A
    side-channel — a separate audit log, or a field on `RoomState` holding the last roster
    change — answers it only for a reader who knows to go looking, and holds one event
    where the interesting case is the room whose roster moved three times.

    The cost of putting it in the transcript is that the transcript stops being uniformly
    speech, and three consumers read it *as* speech: the span an agent is handed
    (`RoomOrchestrator._render_span`), the interjection check (`_interjected`), and the
    mention scan in `selectors.py`. A membership row that slipped past any of them would
    put the room's own prose into a participant's mouth, or make the orchestrator believe
    a human had cut in when one merely joined. So the kind is a field rather than a
    convention over `content`, and `RoomMessage.is_utterance` is the single predicate all
    three filter on — one place to get right, and one place to change.

    The alternative shape, a separate `RoomEvent` tuple alongside `transcript`, was
    rejected for the reason `seq` exists: two sequences cannot be interleaved after the
    fact, and "critic left *before* it was addressed" is exactly the ordering a reader
    needs.
    """

    UTTERANCE = "utterance"
    JOIN = "join"
    LEAVE = "leave"


class RoomTurnRefusal(StrEnum):
    """Why a failed turn will fail the same way if it is retried (#969).

    Set on a failed row only when a retry cannot succeed until something outside the
    conversation changes. A head offers Retry on every other failure, and on this one
    states the remedy instead: a Retry that can only be refused again is a remedy that
    does not work. Decided from the turn's structured `TurnResult.stop_reason`, never
    from the wording of `error`.
    """

    #: A token ceiling refused the turn. The ledger only grows, so it refuses a
    #: retry too; a new conversation starts with budgets of its own.
    BUDGET_EXCEEDED = "budget_exceeded"

    #: The speaker's model cannot use tools, which every clone turn sends. A retry on the
    #: same model is refused the same way; choosing another model is the remedy.
    MODEL_WITHOUT_TOOLS = "model_without_tools"


def turn_refusal(stop_reason: str | None) -> RoomTurnRefusal | None:
    """The refusal a failed turn's `TurnResult.stop_reason` states, if it states one (#969).

    The one mapping, shared by the room orchestrator and the chat head's `/api/chat`, so
    the two heads cannot disagree about which failures a retry is refused on. The token
    ceiling and a model without tool support qualify: `step_budget_exceeded` is not a
    refusal, since `run_steps` resets per turn and a retry starts with the whole step budget.
    """
    if stop_reason == "budget_exceeded":
        return RoomTurnRefusal.BUDGET_EXCEEDED
    if stop_reason == "model_without_tools":
        return RoomTurnRefusal.MODEL_WITHOUT_TOOLS
    return None


class RoomMessage(BaseModel):
    """One row of the transcript — what the chat window shows.

    Almost always a final utterance. Deliberately *not* here: tool calls, tool results,
    intermediate assistant turns and compaction ledgers. Those belong to the speaker's own
    session. A reader of the room sees the conversation; a reader of a session sees how one
    participant produced its half of it.

    The exception is a membership row (`RoomMessageKind`), which records a join or a leave
    in the same ordered record so the conversation can explain its own gaps. Check
    `is_utterance` — never `kind is UTTERANCE` inline, and never `content` — before
    treating a row as something somebody said.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    seq: int = Field(description="Position in the transcript, 1-based and gap-free.")
    sender_id: str = Field(
        description="Who spoke — or, for a membership row, who joined or left. It is the "
        "subject of the row and not its author: the room composes a membership row's "
        "text, which is why `is_utterance` and not `sender_id` decides whether a row is "
        "somebody's word."
    )
    content: str
    kind: RoomMessageKind = Field(
        default=RoomMessageKind.UTTERANCE,
        description="Speech, or a roster change. Defaults to speech so that transcripts "
        "written before membership rows existed still load — `extra='forbid'` with "
        "`strict` would otherwise reject every stored room.",
    )
    created_at: str = Field(default_factory=_now_iso)
    decision: SpeakerDecision | None = Field(
        default=None,
        description="The decision that caused this utterance, for an agent message. "
        "Carried on the message so that 'why did this one answer' is answerable from the "
        "transcript alone rather than only from logs.",
    )
    provenance: Provenance | None = Field(
        default=None,
        description="P6 attribution for an agent utterance, carried over from the "
        "`TurnResult` that produced it.",
    )
    usage: TokenUsage | None = Field(
        default=None,
        description="Tokens this utterance consumed, when the producing turn reported "
        "them. Separate from `provenance` because the two answer different questions: "
        "`provenance` names *which model* served the turn and carries no quantity, and "
        "deriving a token count from it would be the plausible substituted value P6 "
        "forbids.\n\n"
        "**Nothing populates it on the orchestrated path yet.** `TurnResult` drops the "
        "`TokenUsage` its `ModelResponse` carried, so the orchestrator has nothing to copy "
        "here. A reader that sums these rows must not read a missing figure as zero.",
    )
    error: str | None = Field(
        default=None,
        description="Set when the selected agent's turn failed. The failure is recorded "
        "in the transcript rather than dropped: a room that silently skips a failed "
        "speaker presents as an agent that chose not to answer, which is the "
        "indistinguishable-failure P6 forbids.",
    )
    refusal: RoomTurnRefusal | None = Field(
        default=None,
        description="Set, beside `error`, when the failure is one a retry would meet "
        "again (#969). `None` on every other row, and on rows stored before it existed.",
    )
    completed: bool = Field(
        default=True,
        description="Whether this utterance completed normally. False when the turn "
        "was interrupted mid-execution.",
    )
    persist_error: str | None = Field(
        default=None,
        description="Set when the speaker's own session could not be written after this "
        "turn, naming the error. The row still stands -- the reply is real -- but what the "
        "turn added to that agent's context will not be there after a restart, and saying "
        "nothing would present it as durable (P6). Kept apart from `error` because `error` "
        "means the turn failed, which is what `retry` and `last_seen_seq` act on. `None` "
        "on a human's row, a membership row, and a row stored before it existed.",
    )
    knowledge_persist_error: str | None = Field(
        default=None,
        description="Set when what the speaker has learned could not be saved after this "
        "turn, naming the error (#1367). The same contract as `persist_error`, for the "
        "seat's knowledge rather than its session: the reply stands, but what the seat "
        "knows will not be there after a restart, and saying nothing would present it as "
        "kept (P6). `None` when it was saved, when the seat has no knowledge store, on a "
        "human's row, a membership row, and a row stored before it existed.",
    )
    knowledge_set_aside: bool = Field(
        default=False,
        description="Whether the speaker's knowledge record for this conversation could not "
        "be read before this turn and was set aside -- renamed beside where it was, kept, "
        "never deleted (#1367). It concerns that record only; the clone's saved memory "
        "facts are a different file and are not touched. Said on the row so a record that "
        "could not be read is not presented as one that never was (P6). A flag, "
        "not text: where the file went and why it was unreadable are in the log.",
    )
    memory_facts_tried: int = Field(
        default=0,
        ge=0,
        description="How many distinct facts the turn asked to save to memory (#1375). "
        "Counted as `room/orchestrator.py::_memory_save_outcome` states. `0` when none was "
        "attempted, on a human's row, a membership row, and a row stored before it existed.",
    )
    memory_facts_unsaved: int = Field(
        default=0,
        ge=0,
        description="How many of `memory_facts_tried` were never saved in the turn. The "
        "reply is the model's and is left as written -- including when it says the save "
        "worked, which is the case this exists for (#1375): the failures were visible only "
        "in the seat's tool history, and the conversation showed the claim alone. A count "
        "and not the error: the tool's error is written for the model and carries argument "
        "dumps and class names, so it stays in the tool history and never reaches the "
        "conversation as copy.",
    )
    rendered_through: int = Field(
        default=0,
        description="The last seq the prompt this utterance answers actually included -- "
        "the span the turn saw, not the message its selection was made for. The human "
        "message that gave a speaker the floor can be superseded by a newer one before "
        "the reply lands (design doc §3.8); comparing this against the latest human "
        "message's `seq` is what tells a real answer from a late reply that never saw it "
        "(#945). 0 on a human's own row and on a membership row, neither of which answers "
        "anything, and on a message stored before this field existed, which carries no "
        "evidence of what it rendered.",
    )

    turn_id: str | None = Field(
        default=None,
        description="The id the orchestrator minted for the turn that produced this row, "
        "the same one its `AGENT_REPLY` and `TOOL_CALL` events carry. What joins a row to "
        "`RoomState.tool_uses`: a row number cannot, because a rewind reuses it (#1353). "
        "`None` on a human's row, a membership row, and a row stored before it existed.",
    )
    tools_recorded: bool = Field(
        default=False,
        description="True when the tools this turn ran were written to "
        "`RoomState.tool_uses` -- including when it ran none. False on a turn that raised "
        "or was interrupted, which returns no `TurnResult` and so no account of its tools, "
        "and on every row stored before the room recorded tools. A reader must not present "
        "an empty tool list for a row where this is False as 'no tools were used' (P6).",
    )

    @property
    def is_utterance(self) -> bool:
        """True when this row is something a participant said.

        The one predicate every transcript consumer filters on. It exists as a property
        rather than as an inline comparison so that a fourth row kind — a topic change, a
        policy edit — is admitted by editing this line, instead of by finding every reader
        that spelled the check itself and hoping none was missed.
        """
        return self.kind is RoomMessageKind.UTTERANCE


class SelectionVerdict(StrEnum):
    """The three outcomes of asking a selector who speaks next.

    `SILENCE` and `ABSTAIN` are separated on purpose, and the separation is the main thing
    this interface adds over the routing it is modelled on. There, one `None` return meant
    all three of "no one should answer", "I have no opinion, ask the next router" and "my
    provider call raised" — so a broken selector and a deliberately quiet room were the
    same value, and the room went silent either way with nothing to distinguish them.

    Here: `SILENCE` is a decision and ends selection. `ABSTAIN` declines to decide and
    passes to the next selector — a chain that abstains all the way through ends in
    `SILENCE` recorded by the orchestrator, never by accident. A selector that *fails*
    raises; it never returns a verdict.
    """

    SPEAK = "speak"
    SILENCE = "silence"
    ABSTAIN = "abstain"


class SpeakerDecision(BaseModel):
    """Who speaks next, decided by one selector.

    Frozen and self-describing: `selector` and `reasoning` travel with the verdict so the
    orchestrator can record *why* in the transcript without asking the selector again.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    verdict: SelectionVerdict
    speaker_id: str | None = Field(
        default=None,
        description="Required when `verdict` is SPEAK, and must be an agent participant "
        "of the room; forbidden otherwise. Validated by the orchestrator against the "
        "live roster, because a selector naming a participant that left is a real case "
        "and not a programming error.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How sure this selector is. A rule-based selector answers 1.0; an "
        "LLM selector reports what it was asked for. Consumed by `RoomPolicy."
        "hesitation_seconds`, and meaningless when `verdict` is ABSTAIN.",
    )
    selector: str = Field(description="Name of the selector that produced this decision.")
    reasoning: str = Field(
        default="",
        description="Why. Shown in the room's decision trail; required by no validator "
        "and worth writing anyway, since it is what makes a misroute diagnosable.",
    )
    provenance: Provenance | None = Field(
        default=None,
        description="P6 attribution when a model made this choice. A selector that calls "
        "an LLM must set it: the decision is a value a model produced, and which model "
        "chose the speaker is exactly what an operator needs when routing degrades.",
    )


class TurnState(BaseModel):
    """The room's floor state — what the policy needs in order to stop a runaway.

    Kept on the room rather than in an external store: the single-machine path admits no
    out-of-process broker (P3), and this is precisely the state that lived in Redis in the
    design this is modelled on.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    agent_turns_since_human: int = Field(
        default=0,
        ge=0,
        description="Reset by every human utterance, incremented by every agent one. "
        "Bounded by `RoomPolicy.max_agent_turns_per_human_message`.",
    )
    last_speaker_id: str | None = None
    last_activity_ts: float = Field(
        default=0.0,
        description="Monotonic timestamp of the last human activity, including typing. "
        "A hesitating orchestrator compares this across its wait to learn whether a human "
        "took the floor while it waited.",
    )
    races_forgiven: int = Field(
        default=0,
        ge=0,
        description="How many departure races were handled and forgiven. Incremented "
        "when a selected speaker was in the room when the selector answered but left "
        "before the floor could be given (P6 observability, #755).",
    )


class RoomPolicy(BaseModel):
    """The knobs that bound a room's cost and pace."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    max_agent_turns_per_human_message: int = Field(
        default=3,
        ge=0,
        description="Hard ceiling on agent utterances between two human ones. This is the "
        "room's P4 step budget and the only structural defence against an agent-to-agent "
        "loop; a selector's judgement is a probabilistic one and does not substitute.",
    )
    max_span_messages: int = Field(
        default=40,
        ge=1,
        description="Ceiling on how many transcript messages one turn hands its speaker. "
        "Distinct from `transcript_window`, which bounds what a *selector* reads: this "
        "bounds what an *agent* is given, and nothing bounded it before. An agent "
        "addressed for the first time in a long room was handed the whole backlog, so a "
        "room's per-turn cost grew with its length without bound. A *count* ceiling is not a "
        "context guarantee — forty long messages still overrun a window, and bounding "
        "characters is the separate job of compaction inside each agent's own session — "
        "but it removes the unbounded growth, which is what made a long room's cost "
        "unpredictable. When the span is longer, the most recent are kept and the drop is "
        "stated in the prompt — conversation the agent has never seen is a real loss, and "
        "a silent one would leave it answering confidently from a gap it cannot see.",
    )
    transcript_window: int = Field(
        default=15,
        ge=1,
        description="How many trailing utterances a selector sees, and the bound on what "
        "the per-participant copying costs. Windowing counts utterances rather than raw rows "
        "so that intervening membership events cannot push an address out of view before its "
        "turns finish.",
    )
    hesitation_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Ceiling on the pause before a low-confidence speaker takes the "
        "floor, scaled by `1 - confidence`, during which human activity cancels the turn "
        "and yields the floor. Defaults to 0 — disabled: on a local desktop head a "
        "multi-second silence reads as a hung application, and the mechanism only earns "
        "its delay where the head reports human typing.",
    )
    default_responder_id: str = Field(
        default="",
        description="Agent that answers an unaddressed message when no rule has named "
        "anyone. Empty means there is no designated responder, and an unaddressed message "
        "falls through to whatever selector runs next — an LLM selector if one is "
        "configured, and otherwise recorded silence. Deliberately not defaulted to a "
        "participant: a default responder that nobody chose answers in a voice the "
        "operator did not pick, which is the defect this replaces.",
    )
    auto_routing: bool = Field(
        default=True,
        description="Whether an unaddressed message in a multi-agent room falls back to "
        "LLM-based autonomous routing by persona matching when no mention is present "
        "(uclone2 parity). Defaults to True.",
    )
    selector_llm: AgentLLMConfig | None = Field(
        default=None,
        description="Model for an LLM selector, separate from any participant's. "
        "Selection runs once per human message and is a short classification, so it is "
        "the one place in a room where a smaller model is the right default.",
    )
    autonomous: bool = Field(
        default=False,
        description="Whether agents discuss and collaborate autonomously as long as the "
        "user is actively viewing the room, bounded by a 20-turn safety circuit breaker.",
    )

    @model_validator(mode="after")
    def _window_must_outlast_the_turn_ceiling(self) -> RoomPolicy:
        """Refuse a pair of knobs that would starve an address without saying so.

        `MentionSelector` reads the outstanding addresses off the transcript window it is
        given. Once enough agent turns push the human utterance out of that window, the
        selector can no longer see who was addressed, abstains, and the remaining agents
        are simply never called — silently, which is the failure mode this room design
        spends most of its effort refusing elsewhere. The relation the fan-out depends on
        is therefore checked where it is set, rather than left as a coupling between two
        independently tunable numbers that nothing states.
        """
        if self.transcript_window <= self.max_agent_turns_per_human_message:
            raise ValueError(
                f"transcript_window ({self.transcript_window}) must exceed "
                f"max_agent_turns_per_human_message ({self.max_agent_turns_per_human_message}): "
                f"a window that the turns themselves fill pushes the addressing message out "
                f"of view, and the addresses still outstanding are then dropped unannounced"
            )
        return self


class RoomToolUse(BaseModel):
    """One tool call a seat made during a room turn, as the room recorded it (#1353).

    **Metadata and bounded previews, not the trace.** The call and its result in full stay
    in the seat's own session (G1, G2), which is what the model reads back. This is the
    room's account of *that a call happened*, kept on the room so the dock can answer
    "what did this seat do" for a seat that is not running and after its context was
    compacted. Nothing here is ever rendered into another participant's prompt: prompts
    are built from `transcript` alone.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    turn_id: str = Field(description="The turn that made the call; see `RoomMessage.turn_id`.")
    participant_id: str = Field(description="The seat that made the call.")
    tool_name: str
    tool_call_id: str | None = Field(
        default=None,
        description="The model's id for the call, which also keys it in the seat's session.",
    )
    status: str = Field(description="`ToolResultStatus` value: success, error, timeout...")
    error: str | None = None
    duration_ms: float = 0.0
    arguments_preview: str = Field(
        default="",
        description="The call's arguments as JSON, cut to a bounded length. A write's "
        "arguments carry the whole file, which does not belong in the room's record.",
    )
    output_preview: str = Field(
        default="", description="The result as text, cut to the same bound."
    )
    truncated: bool = Field(default=False, description="True when either preview was cut to fit.")
    written_path: str | None = Field(
        default=None,
        description="The workspace-relative path the call wrote, when the tool declares "
        "that it writes, succeeded, and named a `path` in its output.",
    )
    wrote_unnamed: bool = Field(
        default=False,
        description="True when the call *may* have written files the room cannot name: a "
        "call that reached a tool declaring `writes_files` and did not both succeed and name "
        "a path (a shell, including one that wrote and then failed), or any call to a tool "
        "that starts a sub-agent, whose own calls never reach this record (#1366). A "
        "possibility, not an observed write: a reader says 'may have written', never "
        "'wrote'. The name is kept because rooms already store it.",
    )
    subagent_id: str | None = Field(
        default=None,
        description="The sub-agent the call started, when the tool declares that it spawns "
        "one (P4) and named it.",
    )
    recorded_at: str = Field(default_factory=_now_iso)


class RoomWrittenFile(BaseModel):
    """A file a seat wrote through a tool while in this room (#1354, P8).

    Kept apart from `tool_uses` because the two outlive different things. A rewind or a
    clear takes back what was *said*, so the tool calls of the removed turns go with them;
    it does not take back the file, which is still on disk and still this room's output.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    path: str = Field(description="Workspace-relative path, as the writing tool reported it.")
    participant_id: str
    tool_name: str
    turn_id: str
    tool_call_id: str | None = None
    written_at: str = Field(default_factory=_now_iso)


class RoomFileRecord(BaseModel):
    """The known reasons `RoomState.written_files` may be missing a write (#1366).

    `written_files` lists the writes the room saw a tool name. It cannot list a write it
    never saw: a turn that ended before reporting its tools, a tool that writes without
    naming a path (a shell), a helper a seat started, or anything written before the room
    kept the list. And the rows that show such a gap -- the transcript, `tool_uses` -- are
    removed by a clear or a rewind, while the files stay on disk, so the gaps are counted
    here, in a record that only ever grows.

    **It never proves an absence.** No count here being zero means that no file was
    written: any tool -- a shell, an MCP server, a helper -- can write without the room
    seeing it. The dock lists the recorded files and names these gaps; it never says that
    nothing was written (P6).

    **Monotonic.** Every field is set or incremented by the act it counts and nothing
    resets it: not a clear, not a rewind. The default describes a room stored before this
    record existed -- its earlier turns were never counted, which is itself a named gap --
    and only `RoomService.create` writes `kept_since_creation=True`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    kept_since_creation: bool = Field(
        default=False,
        description="True only for a room created by a build that kept this record from "
        "its first row. False on a room loaded without it, whose earlier turns may have "
        "written files nobody counted.",
    )
    unrecorded_turns: int = Field(
        default=0,
        ge=0,
        description="Turns that ended (raised or were stopped) before reporting their tools.",
    )
    unattributed_writes: int = Field(
        default=0,
        ge=0,
        description="Calls that may have written files without naming them: see "
        "`RoomToolUse.wrote_unnamed`. A count of possibilities, never of observed writes.",
    )
    turns_started: int = Field(
        default=0,
        ge=0,
        description="Turns the orchestrator began, each counted in a save of its own before "
        "the turn ran any tool. A turn whose landing save never happened -- a lost "
        "compare-and-swap, a disk error, the process dying mid-turn -- leaves this ahead "
        "of `turns_landed`, which is how that loss stays visible (#1366). A turn another "
        "process is running on the same room directory is ahead by the same count until "
        "it lands: which turn is in progress is known only to the process running it, so "
        "a reader names both causes (#1388).",
    )
    turns_landed: int = Field(
        default=0,
        ge=0,
        description="Turns whose row, tools and written files reached the store, counted in "
        "that same save. Never ahead of `turns_started` for a room created with both; a "
        "room where it is ahead began before turns were counted as they started.",
    )
    clears: int = Field(default=0, ge=0, description="Times the history was cleared.")
    rewinds: int = Field(
        default=0, ge=0, description="Times the history was rewound and lost at least one row."
    )


class RoomState(BaseModel):
    """One room: its roster, its transcript, and its floor state.

    Frozen, and versioned by `revision` on the same compare-and-swap contract as
    `SessionState` — a room has one writer (the orchestrator) by design, and the
    precondition is what turns a second writer into a refused write rather than a lost
    utterance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    room_id: str = Field(default_factory=lambda: f"room_{uuid.uuid4().hex[:12]}")
    title: str = Field(
        default="",
        description="What a person calls this room. A plain title and not a structured "
        "label, because the thing it replaces is `RoomStore.list_room_ids()` returning "
        "`room_9f2c1ab40e7d` — 'reopen yesterday's room' meant recognising a hex string, "
        "and no amount of structure fixes that better than a sentence the operator wrote. "
        "Kept on the room rather than derived from the first utterance: a derived title "
        "changes when the conversation is compacted or the opening message is edited, and "
        "an identifier that moves is not one.\n\n"
        "**Empty is a legacy value, not a supported one.** The default exists so that a "
        "room stored before this field did still loads under `extra='forbid'`; "
        "`RoomService.create` refuses a blank title, so an untitled room means one written "
        "by an older build, and that is what a reader should infer from seeing the id "
        "rendered in its place.",
    )
    participants: tuple[Participant, ...] = Field(default_factory=tuple)
    transcript: tuple[RoomMessage, ...] = Field(default_factory=tuple)
    turn_state: TurnState = Field(default_factory=TurnState)
    policy: RoomPolicy = Field(default_factory=RoomPolicy)
    last_seen_seq: ImmutableStrMapping = Field(
        default_factory=dict,
        description="Per-participant high-water mark over `RoomMessage.seq`, as strings "
        "so the mapping stays JSON-shaped on disk. What an agent has already been shown "
        "is not shown again: its own session already holds it, and re-sending would "
        "duplicate the conversation inside that session. Advanced only after a turn "
        "succeeds, so a failed turn is retried against the same unseen span.",
    )
    last_decision: SpeakerDecision | None = Field(
        default=None,
        description="The decision that ended the most recent turn loop. Carries the "
        "`SILENCE` that closed a room nobody chose to speak in — a verdict with no "
        "utterance to hang it on, and therefore the one decision that would otherwise "
        "vanish. A reader must be able to tell a room that decided to be quiet from one "
        "whose selectors failed (G8).",
    )
    tool_uses: tuple[RoomToolUse, ...] = Field(
        default_factory=tuple,
        description="Every tool call a seat made in a recorded turn, in order (#1353). "
        "Written in the same save as the turn's row, so the two cannot disagree. Rows "
        "whose `tools_recorded` is False have no entries here, and that is not the same "
        "as having used none.",
    )
    written_files: tuple[RoomWrittenFile, ...] = Field(
        default_factory=tuple,
        description="Every file a seat wrote through a tool in this room, in write order, "
        "one entry per write (#1354). Survives a rewind and a clear: see `RoomWrittenFile`.",
    )
    file_record: RoomFileRecord = Field(
        default_factory=RoomFileRecord,
        description="The known reasons `written_files` may be missing a write (#1366). "
        "Never a proof that it is complete: see `RoomFileRecord`.",
    )
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)

    revision: int = Field(
        default=0, description="Monotonic write counter; 0 means never persisted."
    )

    @model_validator(mode="after")
    def _a_room_seats_one_human(self) -> RoomState:
        """Refuse the roster the runtime cannot serve, and say what it could not serve.

        The room is a local surface over one operator's Core, and its turn loop is written
        for one floor: `post()` reads, appends and saves under the store's compare-and-swap,
        so two humans posting at once means the later write is refused as stale and that
        message is lost to whoever sent it. `last_seen_seq` is an agent's mark, so a second
        human would have no record of what they had read either.

        None of that was visible from the model, which is the actual defect: a roster that
        admits two humans advertises a shared room, and the next reader has to run the turn
        loop to find out it is not one. The bound is therefore stated here, on the record,
        rather than only in the service that writes it.

        `model_copy(update=...)` does **not** re-run validators, so this fires on
        construction and on load — the two paths a stored or hand-assembled room takes —
        while the orchestrator's own in-flight copies stay free of the cost. The guard on
        the write path is `RoomService.add_participant`, which refuses the seat with the
        same reason before a record is ever built.
        """
        humans = [p.id for p in self.participants if p.kind is ParticipantKind.HUMAN]
        if len(humans) > 1:
            raise ValueError(
                f"A room seats one human, and this roster has {len(humans)} "
                f"({', '.join(humans)}). Two humans posting at once collide on the "
                f"transcript's single-writer precondition, and the later message is "
                f"refused rather than merged; a shared room needs a serialisation story "
                f"that this room does not have. Open one room per person."
            )
        return self

    @property
    def last_utterance(self) -> RoomMessage | None:
        """The last thing somebody *said*, skipping membership rows. `None` in a new room.

        The last *row* is a different thing and the difference matters wherever a failed
        turn is being looked for: `remove_participant` appends a leave row, so a room whose
        agent failed and then left has a leave as its final row and the failure one place
        further back. A reader that took the final row would report that nothing failed,
        for a room whose failure is still sitting there unanswered.

        Defined here rather than in each caller for the reason `RoomMessage.is_utterance`
        is: the orchestrator asks this question to decide whether a turn can be re-run, and
        the CLI asks it to decide whether to offer the retry, and the two must not be able
        to drift into disagreeing about which turn is the one in question.
        """
        return next((m for m in reversed(self.transcript) if m.is_utterance), None)


class SpeakerRequest(BaseModel):
    """Everything a selector is given, and nothing else.

    A request rather than loose arguments, so that adding an input later does not change
    every selector's signature — and so that a selector provably cannot reach a
    participant's session, tools or ontology. Selection reasons over the room's public
    surface only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    room_id: str
    participants: tuple[Participant, ...]
    transcript: tuple[RoomMessage, ...] = Field(
        description=(
            "The trailing window bounded by `RoomPolicy.transcript_window` utterances, "
            "preserving interleaved membership context, oldest first."
        )
    )
    turn_state: TurnState
    policy: RoomPolicy
