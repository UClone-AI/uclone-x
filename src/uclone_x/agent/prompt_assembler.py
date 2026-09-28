"""What a turn's request says besides the conversation: identity, context sections, anchors.

Moved out of `agent/base.py` unchanged (#1736, stage 2). `BaseAgent` stays the facade: it
keeps `_prepare_turn_layers`, `_prepare_turn_messages` and `_nudged_retry`, which call into
the `PromptAssembler` it builds, and it keeps the decisions the assembly reads but does not
make -- `_anchor_is_stale`, `_system_prompt_base`, `effective_system_prompt` -- because
those read the persona axis and the live session, which are the agent's.

The assembler holds no state of its own. Everything it reads comes through a
`PromptScope` of callables, evaluated on every call, so an agent whose `_config`,
`_tools`, `_context` or live session is swapped after construction is read as it now is
(`evals/harness_ladder/runner.py` swaps `agent._tools`, and tests patch the rest).

The module-level helpers are the pure half: the identity layer (`compose_identity_prompt`),
the `[Turn Context]` block, the anchor-provenance stamp a session carries and its persisted
form, the request-context delta, and the undone-attempt statement.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol

from uclone_x.agent.models import AgentConfig, PersonaDefinition, PlanState, ToolExecutionRecord
from uclone_x.agent.prompts import adapt_system_prompt
from uclone_x.agent.request_record import (
    RequestLayers,
    compose_system_message,
    messages_digest,
    serialize_tools,
)
from uclone_x.agent.session import (
    AnchorAuthor,
    AnchorProvenance,
    ContextSnapshot,
    content_digest,
)
from uclone_x.core.context_state import (
    ContextEntry,
    ContextEpoch,
    ContextForm,
    recorded_forms,
    recorded_renderings,
    render_entries,
    shown_entries,
)
from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.secrets import redact_credentials
from uclone_x.llm.models import ChatMessage, LLMRequest, MessageRole, ToolCallRequest
from uclone_x.memory.recall import recall_prompt_section
from uclone_x.ontology.protocols import OntologyEngineProtocol
from uclone_x.skills.models import SkillStatus, missing_required_tools
from uclone_x.skills.protocols import SkillRegistryProtocol
from uclone_x.tools.protocols import ToolRegistryProtocol

if TYPE_CHECKING:
    from uclone_x.memory import CrossSessionMemory

__all__ = [
    "FILE_TOOL_NAMES",
    "TURN_CONTEXT_HEADER",
    "AnchorWriter",
    "LiveAnchorProvenance",
    "PromptAssembler",
    "PromptScope",
    "SnapshotSession",
    "compose_identity_prompt",
    "drop_once_drawn",
    "has_anchor",
    "persisted_anchor_provenance",
    "present_sections",
    "request_context_delta",
    "restored_anchor_provenance",
    "turn_context_block",
    "undone_attempt_section",
]


#: The tools whose paths resolve against the workspace; holding any one of them is what
#: makes the `[Workspace]` prompt section worth its tokens.
FILE_TOOL_NAMES = frozenset(
    {"file_read", "file_write", "file_edit", "file_search", "directory_list"}
)


class AnchorWriter(Enum):
    """The anchor provenances that are not a persona resolution.

    Two members, because there are exactly two things a persona resolution cannot
    express: "this agent did not compose the anchored system turn", and "nothing on
    record says what composed it". They are **not** the same claim and collapsing them
    would be a silent fallback (P6): the first is knowledge, the second is its absence,
    and only the second is a condition worth reporting. Both are treated alike by
    `_anchor_is_stale` — an anchor with no position behind it has nothing to compare —
    and that shared answer is a consequence, not the reason they are one type.

    An `Enum` rather than bare `object()`s so the type checker can see them in a union.
    """

    CALLER = "caller"
    UNRECORDED = "unrecorded"


#: What a session's anchored system turn was composed from. A `PersonaDefinition` — or
#: `None` for "the axis resolved to no persona" — is *this agent's own* resolution at the
#: moment it wrote that anchor, which `_anchor_is_stale` compares against the axis as it
#: stands when a turn is built. `AnchorWriter.CALLER` is text that arrived from outside
#: (`load_history`): the agent did not compose it and has no axis position to attribute it
#: to, so it is not re-resolved. `AnchorWriter.UNRECORDED` is a session restored from a
#: record that carries no `anchor_provenance` — written before the field existed, or built
#: by a door with no answer to give (#1152).
LiveAnchorProvenance = PersonaDefinition | None | AnchorWriter


def has_anchor(messages: Sequence[ChatMessage]) -> bool:
    """Whether `messages` opens with a `SYSTEM` turn — the anchor the turn builder re-frames.

    One expression for the two places that ask (#1174, #1152): the turn builder, which
    sends what `effective_system_prompt` reports when there is no anchor, and
    `hydrate_session`, which only reports missing provenance for a record that actually
    has an anchor for the provenance to be missing *about*. A record of user-first rows
    has nothing to re-resolve, so warning about it would be noise.
    """
    return bool(messages) and messages[0].role == MessageRole.SYSTEM


def compose_identity_prompt(
    *,
    config_prompt: str,
    persona: PersonaDefinition | None,
    seat_framing: str = "",
) -> str:
    """The identity layer: the one function that builds the prompt an agent speaks as.

    The anchor a session stores and the system message a turn sends both come from here,
    through `BaseAgent._system_prompt_base`. The room used to assemble a seat's prompt
    itself and hand the result in after the agent had already seeded its session. The
    stored anchor then lacked the seat's framing while the turn sent it, so the record
    and the wire disagreed.

    * No framing: the persona's prompt when one is in force, else `config_prompt`, as
      before.
    * Framing and a persona with instructions: the framing, then the persona's
      instructions under a labelled header.
    * Framing and nothing else to say: the framing, then `config_prompt`.
    """
    body = persona.system_prompt if persona is not None else config_prompt
    if not seat_framing:
        return body
    if persona is not None and persona.system_prompt:
        return f"{seat_framing}\n\n[Persona Instructions: {persona.role}]\n{persona.system_prompt}"
    return f"{seat_framing}\n\n{config_prompt}"


def composed_seat_framing(
    identity: str, *, config_prompt: str, persona: PersonaDefinition | None
) -> str | None:
    """The seat framing `identity` was composed with, or `None` if it is not this identity.

    The inverse of `compose_identity_prompt` for one persona and configured prompt: `""`
    when `identity` is the unframed composition, the framing text when it is the framed
    one, and `None` when it is neither -- a prompt a caller wrote, or one composed from
    something else. `None` claims nothing, so a reader treats it as "cannot tell".

    Exists because a seat's framing can change under an anchor that was already written:
    a one-seat room seeds its clone without the multi-agent framing (§5.9.3), and a second
    clone joining puts it back. The anchor records the persona that composed it, not the
    framing, so the framing is read back from the text. Compare canonical forms
    (`adapt_system_prompt(..., None)`) on both sides, so a model-family re-framing is not
    mistaken for a framing change.
    """
    bare = compose_identity_prompt(config_prompt=config_prompt, persona=persona)
    if identity == bare:
        return ""
    # What follows the framing in a framed composition: compose under a one-character
    # framing and drop that character, so this cannot drift from the composer's layout.
    tail = compose_identity_prompt(config_prompt=config_prompt, persona=persona, seat_framing="\0")[
        1:
    ]
    if len(identity) > len(tail) and identity.endswith(tail):
        return identity[: -len(tail)]
    return None


#: Opens the block of state that changes during a conversation. It travels at the tail of
#: the request, never in the system turn: see `turn_context_block`.
TURN_CONTEXT_HEADER = (
    "[Turn Context]\n"
    "Supplied by the runtime with every request, not written by the user. It states the "
    "current value of state that changes during the conversation; where it disagrees "
    "with an earlier turn, this is the current one."
)


def turn_context_block(sections: Sequence[str]) -> str:
    """The `[Turn Context]` block for `sections`, placed at the tail of the request.

    Empty when there are none. `place_turn_context` puts it at the tail, never in the
    system turn.

    **The system turn is the head of every prefix a provider or a local server can reuse.**
    One changed byte there re-prefills the whole conversation behind it -- on a local
    model the latency the conversation waits on, on a hosted one the full input price.
    The plan (a box ticked per completed step), cross-session memory (a fact recorded
    mid-conversation, inserted by confidence rather than appended) and the tool-scoping
    notice (recomputed from each prompt) all change inside a conversation, so while they
    lived in the system turn every such change discarded the entire cached prefix.

    At the tail a change costs only the block itself. The block is built per request and
    never written to history, so the next request's prefix -- system turn plus history --
    is byte-identical to this one's up to where this block began.

    A `USER` role, not `SYSTEM`: the Anthropic and Gemini connectors hoist every `SYSTEM`
    message, wherever it stands, into the one top-level system field, which would put the
    block straight back at the head. When the request already ends with the user's
    message (a turn's first step) the block joins that message rather than following it
    as a second consecutive user turn, which chat templates that require alternating
    roles refuse; after a tool result there is no user message to join, and it follows.
    """
    if not sections:
        return ""
    return "\n\n".join((TURN_CONTEXT_HEADER, *sections))


def persisted_anchor_provenance(provenance: LiveAnchorProvenance) -> AnchorProvenance | None:
    """Render a live stamp into the shape `SessionState` persists (#1152).

    `UNRECORDED` renders as `None` — "the record still does not say" — rather than as
    anything the reader could mistake for a stamp. A session restored from a legacy
    record and persisted again is honest about the gap instead of inventing a filling for
    it, and the *next* anchor this agent composes stamps itself properly.
    """
    if provenance is AnchorWriter.UNRECORDED:
        return None
    if provenance is AnchorWriter.CALLER:
        return AnchorProvenance(author=AnchorAuthor.CALLER)
    return AnchorProvenance(author=AnchorAuthor.AGENT, persona=provenance)


def restored_anchor_provenance(record: AnchorProvenance | None) -> LiveAnchorProvenance:
    """Read a persisted stamp back into the live form, or report that there is none.

    The inverse of `persisted_anchor_provenance`, and the whole of what #1152 asked for:
    with the stamp on the record, a restored session can say which persona composed its
    anchor instead of every restored anchor reading as the caller's and never being
    re-resolved.

    A record with no stamp reads as `UNRECORDED`, never as `None`. `None` here means "the
    agent composed this under no persona", which is a claim — and a false one would make
    the very next persona adoption overwrite an anchor the caller may have supplied.
    """
    if record is None:
        return AnchorWriter.UNRECORDED
    if record.author is AnchorAuthor.CALLER:
        return AnchorWriter.CALLER
    return record.persona


def request_context_delta(
    previous: Sequence[dict[str, Any]], current: Sequence[dict[str, Any]]
) -> tuple[int, list[dict[str, Any]]]:
    """What a request's conversation adds to the previous one's: `(kept, appended)` (#1442).

    `kept` is the length of the prefix `current` shares with `previous`, and `appended` is
    the rest of `current`. The durable `REQUEST_CONTEXT` event records these instead of the
    whole conversation, because the whole of it was re-recorded on every step: an N-step
    turn wrote N copies of a history that grows each step, so the event log grew
    quadratically in N. A request's conversation is the previous one plus what the step
    added, so `appended` is that delta and the log grows linearly. The prefix is compared,
    not assumed, so a conversation that rewrites an earlier message -- compaction, a
    history edit -- records everything from the first difference, and the full
    conversation is always `previous[:kept] + appended`.
    """
    kept = 0
    for before, now in zip(previous, current, strict=False):
        if before != now:
            break
        kept += 1
    return kept, list(current[kept:])


#: Longest rendering of one argument value, and of one call's whole argument list, in the
#: undone-attempt statement. A summary says which call it was; the call itself is in the log.
_UNDONE_ARG_VALUE_CHARS: Final = 40


_UNDONE_ARGS_CHARS: Final = 120


#: Calls listed before the rest are counted rather than named.
_UNDONE_CALLS_LISTED: Final = 20


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _summarize_call(call: ToolCallRequest) -> str:
    """`name(key=value, ...)`, each value and the whole list clipped; never a result."""
    parts: list[str] = []
    for key, value in call.arguments.items():
        rendered = json.dumps(unwrap_immutable(value), ensure_ascii=False, default=str)
        parts.append(f"{key}={_clip(rendered, _UNDONE_ARG_VALUE_CHARS)}")
    return redact_credentials(f"{call.name}({_clip(', '.join(parts), _UNDONE_ARGS_CHARS)})")


def present_sections(*sections: str | None) -> tuple[str, ...]:
    """The non-empty turn-context sections, in order."""
    return tuple(section for section in sections if section)


def drop_once_drawn(
    sections: list[str], plan: str | None, executions: Sequence[ToolExecutionRecord]
) -> None:
    """Remove the image-set plan note from `sections` once a step has run `generate_image`.

    A step that only read a character sheet leaves the plan standing; once the images
    were drawn, the note would ask for them a second time. In place, so the turn loop
    carries no reassigned section tuple (pyright's flow analysis of `execute_turn`).
    """
    if plan in sections and any(e.tool_name == "generate_image" for e in executions):
        sections.remove(plan)


def undone_attempt_section(calls: Sequence[ToolCallRequest]) -> str:
    """The turn-context statement of what an undone attempt called, or `""` (#1495).

    A rolled-back attempt leaves the conversation, but not the world: a tool it ran may
    have saved a memory or written a file. Without this, the retry is shown the same
    history the attempt started from and may run a non-idempotent call a second time.
    A step refused for the context window is withheld from history after its tools ran,
    so its calls are stated here too (#1509).

    **A runtime statement in the turn context, not a `TOOL` message.** A `TOOL` message
    is a tool's own result; one the runtime composed would be the substituted result P6
    forbids, and it would enter history permanently. The statement is layer 5 of the
    request-layering design (§5): built per request, never written to history, recorded
    verbatim in the turn's context snapshot. It names each call and a clipped summary of
    its arguments, never an output, and says only what is known: the call was made, and
    undoing the attempt did not undo it.
    """
    if not calls:
        return ""
    listed = [f"- {_summarize_call(call)}" for call in calls[:_UNDONE_CALLS_LISTED]]
    if len(calls) > _UNDONE_CALLS_LISTED:
        listed.append(f"- and {len(calls) - _UNDONE_CALLS_LISTED} more")
    return (
        "[Undone Attempt]\n"
        "An earlier attempt was stopped and undone, so its messages are not in this "
        "conversation. It called the tools below. Undoing the attempt did not undo what "
        "they did, and a call may have run even where no result was recorded.\n" + "\n".join(listed)
    )


class SnapshotSession(Protocol):
    """The part of a live session the context snapshot and the request delta write to."""

    stored_bodies: set[str]
    pending_bodies: dict[str, str]
    context_snapshots: list[ContextSnapshot]
    last_conversation: list[dict[str, Any]]
    last_request: int | None
    recalled_memory: str | None
    #: Per log entry of the history a compaction left, the entry it shows and its form
    #: (`compacted_entries`, #1848), until the next request records them.
    compacted_entries: dict[str, ContextEntry]

    @property
    def context_epochs(self) -> Sequence[ContextEpoch]:
        """What each earlier request showed, per epoch (#1443)."""
        ...

    def logged_history(self) -> list[tuple[str, ChatMessage]]:
        """Each history message's log entry and the message its logged body decodes to,
        logging any not yet logged (#1848)."""
        ...

    def record_shown(self, shown: list[ContextEntry], *, step: int) -> ContextEpoch:
        """Record what a request's conversation showed in the context state (#1443)."""
        ...


@dataclass(frozen=True, slots=True)
class PromptScope:
    """The agent state the assembler reads, as callables so it is read live on every call."""

    ontology: Callable[[], OntologyEngineProtocol | None]
    skills: Callable[[], SkillRegistryProtocol | None]
    config: Callable[[], AgentConfig]
    tools: Callable[[], ToolRegistryProtocol | None]
    memory: Callable[[], CrossSessionMemory | None]
    workspace_root: Callable[[], Path | None]
    current_plan: Callable[[], PlanState | None]
    history: Callable[[], Sequence[ChatMessage]]
    active_session: Callable[[], SnapshotSession]
    turn_counter: Callable[[], int]
    #: Whether the active session's anchor was composed under another persona-axis
    #: position than the one in force now (`BaseAgent._anchor_is_stale`).
    anchor_is_stale: Callable[[], bool]
    system_prompt_base: Callable[[], str]
    effective_system_prompt: Callable[[], str]


class PromptAssembler:
    """Builds a turn's request layers and records what each request carried.

    The accessors below carry the names the agent's own attributes have, so the methods
    read exactly as they did on `BaseAgent`; each one reads the scope, never a copy.
    """

    def __init__(self, scope: PromptScope) -> None:
        self._scope = scope
        #: `recorded_forms` and `recorded_renderings` of the epochs they were computed
        #: from, which are held so a reused `id()` never matches (#1875, item 5). Several
        #: prepares of one request read the same epochs; a request that records a new one
        #: replaces the sequence.
        self._forms_of: tuple[Sequence[ContextEpoch], ContextEpoch | None] | None = None
        self._forms: dict[str, ContextForm] = {}
        self._renderings: dict[str, ContextEntry] = {}

    def _recorded(
        self, epochs: Sequence[ContextEpoch]
    ) -> tuple[dict[str, ContextForm], dict[str, ContextEntry]]:
        """`recorded_forms(epochs)` and `recorded_renderings(epochs)`, reused while
        `epochs` is the sequence they were read from.

        The sequence and its last epoch must both be the same objects: `advance` only ever
        appends an epoch or replaces the last, so an epoch added or extended in place is
        read afresh.
        """
        last = epochs[-1] if epochs else None
        cached = self._forms_of
        if cached is None or cached[0] is not epochs or cached[1] is not last:
            self._forms = recorded_forms(epochs)
            self._renderings = recorded_renderings(epochs)
            self._forms_of = (epochs, last)
        return self._forms, self._renderings

    @property
    def _ontology(self) -> OntologyEngineProtocol | None:
        return self._scope.ontology()

    @property
    def _skills(self) -> SkillRegistryProtocol | None:
        return self._scope.skills()

    @property
    def _config(self) -> AgentConfig:
        return self._scope.config()

    @property
    def _tools(self) -> ToolRegistryProtocol | None:
        return self._scope.tools()

    @property
    def _memory(self) -> CrossSessionMemory | None:
        return self._scope.memory()

    @property
    def _history(self) -> Sequence[ChatMessage]:
        return self._scope.history()

    @property
    def _active_session(self) -> SnapshotSession:
        return self._scope.active_session()

    @property
    def _turn_counter(self) -> int:
        return self._scope.turn_counter()

    @property
    def current_plan(self) -> PlanState | None:
        return self._scope.current_plan()

    @property
    def effective_system_prompt(self) -> str:
        return self._scope.effective_system_prompt()

    def _resolve_workspace_root(self) -> Path | None:
        return self._scope.workspace_root()

    def _system_prompt_base(self) -> str:
        return self._scope.system_prompt_base()

    def _anchor_is_stale(self) -> bool:
        return self._scope.anchor_is_stale()

    async def recall_memory(self, message: str) -> str | None:
        """The memory section recalled for `message`, or `None` with no memory store.

        Ranked against the message (clone-knowledge-graph §3.5), so it changes per turn and
        is sent only in the turn-context tail; `prepare_turn_layers` reads it back from the
        live session, where the turn executor holds it for the turn.
        """
        memory = self._memory
        return None if memory is None else await recall_prompt_section(memory, message)

    def get_active_invariants_prompt_section(
        self,
        domain: str | None = None,
        tier_filter: Literal["asserted", "candidate", "all"] = "asserted",
    ) -> str:
        """Query active invariants using tier_filter='asserted' by default (P7, P8)."""
        if self._ontology is None:
            return ""
        invariants = self._ontology.get_active_invariants(domain=domain, tier_filter=tier_filter)
        if not invariants:
            return ""
        lines = ["[Active Domain Ontology Invariants]:"]
        for inv in invariants:
            rule_detail = (
                inv.rule_expression
                or (
                    f"{inv.predicate} == {inv.object_value}"
                    if inv.predicate and inv.object_value
                    else inv.predicate
                )
                or inv.description
                or inv.name
            )
            lines.append(f"- Rule ({inv.tier.value}): {inv.name} -> {rule_detail}")
        return "\n".join(lines)

    def get_active_skills_prompt_section(self) -> str:
        """Return progressive disclosure prompt section listing approved skills (P9)."""
        if self._skills is None:
            return ""
        # A skill whose `requires_tools` the agent's tool scope does not grant is left out
        # (#1826): offering it would teach the model a procedure it cannot carry out. The
        # scope is the declared one, so the listing is stable across a session's turns.
        scope = self._config.allowed_tools
        active_skills = [
            s
            for s in self._skills.list_skills()
            if s.manifest.status == SkillStatus.ACTIVE
            and not missing_required_tools(s.manifest, scope)
        ]
        if not active_skills:
            return ""
        active_skills.sort(key=lambda s: s.manifest.name)
        lines = [
            "[Available Approved Skills]",
            "The following modular skills are approved and available for on-demand use. "
            "To view full procedural instructions for any skill, invoke the 'load_skill' tool.",
        ]
        for s in active_skills:
            desc = (s.manifest.description or "").strip() or "No description provided."
            lines.append(f"- {s.manifest.name}: {desc}")
        return "\n".join(lines)

    def get_workspace_prompt_section(self) -> str | None:
        """Tell the model where its file tools resolve paths, when it holds any.

        Without it the model knows only that it works "in the user's workspace", so it
        invents a relative path for a folder it was told about by name and cannot tell a
        folder that is absent from one that is out of bounds.
        """
        allowed = self._config.allowed_tools
        # A name in the list is a permission, not a tool: every persona is now permitted
        # the read-only file tools (#1402), including on a registry that has none. The
        # section is only true when one of them is actually there to be called.
        if allowed and not any(
            name in FILE_TOOL_NAMES and self._tools is not None and self._tools.get(name)
            for name in allowed
        ):
            return None
        workspace = self._resolve_workspace_root()
        if workspace is None:
            return None
        lines = [
            "[Workspace]",
            f"File tools resolve relative paths against the workspace folder: {workspace}",
        ]
        read_roots = self._config.read_roots
        if read_roots:
            lines.append(
                "The file tools may also read (never write) these folders outside it, by "
                "absolute path:"
            )
            lines.extend(f"- {root.resolve()}" for root in read_roots)
        lines.append(
            "The file tools refuse any other path. When a folder or file is not found, "
            "list the folder that should contain it before telling the user it is missing."
        )
        return "\n".join(lines)

    def get_active_plan_prompt_section(self) -> str | None:
        """Format the current interactive execution plan for injection into the system prompt."""
        plan = self.current_plan
        if not plan:
            return None

        lines = [f"### Active Execution Plan: {plan.title}", f"Status: {plan.status.upper()}", ""]

        for step in plan.steps:
            box = "[x]" if step.completed else "[ ]"
            lines.append(f"{step.index}. {box} {step.description}")
            if step.verification:
                lines.append(f"   Verification: {step.verification}")

        return "\n".join(lines)

    def record_context_snapshot(self, req: LLMRequest, layers: RequestLayers) -> str:
        """Record what `req` carries besides the conversation; return the snapshot's id.

        Appends a `ContextSnapshot` to the active session unless the last one already says
        the same, so a turn adds one, and a step adds another only when its tools, identity,
        slow context, turn context or model settings differ. The large bodies wait on
        the session until the next save writes them, once each, ahead of the record.
        """
        session = self._active_session
        tools_body = serialize_tools(req.tools)
        bodies = {
            content_digest(tools_body): tools_body,
            content_digest(layers.identity): layers.identity,
            content_digest(layers.slow_context): layers.slow_context,
            content_digest(layers.turn_context): layers.turn_context,
        }
        candidate = ContextSnapshot(
            turn_index=self._turn_counter,
            tools_digest=content_digest(tools_body),
            identity_digest=content_digest(layers.identity),
            slow_context_digest=content_digest(layers.slow_context),
            system_message=layers.system_message,
            turn_context_digest=content_digest(layers.turn_context),
            model=req.model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            auto_compact=req.auto_compact,
            compaction_threshold_tokens=req.compaction_threshold_tokens,
        )
        for digest, body in bodies.items():
            if digest not in session.stored_bodies:
                session.pending_bodies[digest] = body
        if not session.context_snapshots or session.context_snapshots[-1] != candidate:
            session.context_snapshots.append(candidate)
        return candidate.snapshot_id

    def request_context_fields(
        self, step: int, req: LLMRequest, layers: RequestLayers
    ) -> dict[str, Any]:
        """The `REQUEST_CONTEXT` event's fields for `req`: its snapshot, and what it added.

        The event names the snapshot that holds the tools, identity, slow context, turn
        context and model settings, and records the conversation as a delta on the
        previous request of this session -- the previous step, or the last step of the
        previous turn. A turn's first step used to record its whole request, so every turn
        re-recorded the conversation so far (#1421). `rebuild_requests` reverses this.

        `request` numbers the requests of this working copy and `base_request` names the
        one this extends, so a reader can tell a gap in the log from a real extension.
        `digest` is over the whole request as sent, before redaction.

        What the conversation showed -- each message's log entry and form -- goes to the
        session's context state (`record_shown`, #1443), which extends the current epoch
        or opens a new one.
        """
        session = self._active_session
        snapshot_id = self.record_context_snapshot(req, layers)
        session.record_shown(list(layers.shown), step=step)
        conversation = [m.model_dump() for m in layers.conversation]
        kept, appended = request_context_delta(session.last_conversation, conversation)
        base_request = session.last_request
        request_number = 1 if base_request is None else base_request + 1
        session.last_conversation = conversation
        session.last_request = request_number
        return {
            "step": step,
            "snapshot": snapshot_id,
            "request": request_number,
            "base_request": base_request,
            "message_count": len(req.messages),
            "kept_message_count": kept,
            "appended_messages": appended,
            "digest": messages_digest([m.model_dump() for m in req.messages]),
        }

    def prepare_turn_layers(self, extra_sections: Sequence[str] = ()) -> RequestLayers:
        """Construct turn layers: invariants and skills in the system turn, volatile state at the tail (P7, P8, P9).

        Returns the layers apart, so the turn can record them in a `ContextSnapshot`;
        `assemble_request_messages` puts them together, for the request and for a rebuild
        of it from the record alike (#1421).

        **Only sections that hold still across a conversation join the system turn** --
        the asserted invariants and the approved skills. The plan, cross-session memory
        and `extra_sections` change inside a conversation and travel at the tail of the
        request instead, so a change to them does not discard the cached prefix; see
        `turn_context_block`.

        The anchored system turn is **re-framed for the configured model family on every
        turn** rather than trusted as `self._history[0]` (#921). `hot_reload_llm` moves
        `llm_config.model_name`, so `effective_system_prompt` starts answering for the
        new family immediately while the seeded anchor still carries the old family's
        steerability framing; the wire call took the anchor, so the property and the
        message actually sent disagreed, silently (P6).

        Resolved here rather than by rewriting the anchor in place, for two reasons that
        the in-place rewrite cannot reach:

        * **Sessions this agent has not materialised yet.** `_live_session` seeds on
          first touch and `load_history` hydrates from the store, both of which can
          happen *after* a `hot_reload_llm` call. A refresh that walks `self._sessions`
          at reload time reframes only what is live at that instant; a session restored
          from a record persisted under the old model stays stale for the rest of its
          life. Resolving on the turn covers every session on the only path that matters.
        * **The stored history keeps saying what was actually anchored.** The anchor is
          persisted, so overwriting it would edit the record of earlier turns to claim
          framing those turns never carried, and it is unrecoverable once saved.

        `adapt_system_prompt` is a substitution *from* the canonical policy, so an
        operator's own prompt — one that embeds no framing this codebase knows — comes
        back byte-identical and is never replaced by the resolved default.

        **The persona axis is resolved here too, against the axis position the anchor was
        composed under** (#1081). Adopting a persona changes which prompt the agent is
        supposed to be sending, and a session anchored before that adoption carries the
        previous one, so `effective_system_prompt` and the wire described different
        personas for the same agent. It is resolved on the turn for the same two reasons
        the model family is, and `_anchor_is_stale` decides it per session by comparing
        that session's `anchor_provenance` against what the axis resolves to now.

        Reading it off the *session* rather than off the agent is not a detail. The eleven
        constructions in the PR body were measured against two agent-wide flags before
        this one, and the four regressions between them — C1, C2, C8, C9 — are all the
        same shape: one flag answering for a session whose anchor a caller composed and a
        session the agent seeded, which are different answers. Per session it also reaches
        a session seeded on first touch, or one hydrated after the persona was adopted,
        which a refresh walking `self._sessions` at set time structurally cannot.

        Clearing an adopted persona is resolved by the same comparison rather than by a
        case of its own: `_system_prompt_base` falls back to `config.system_prompt`, so an
        anchor composed under the persona just dropped is recomputed from configuration
        rather than salvaged.
        """
        sections: list[str] = []
        invariants_section = self.get_active_invariants_prompt_section(tier_filter="asserted")
        if invariants_section:
            sections.append(invariants_section)
        skills_section = self.get_active_skills_prompt_section()
        if skills_section:
            sections.append(skills_section)

        workspace_section = self.get_workspace_prompt_section()
        if workspace_section:
            sections.append(workspace_section)

        turn_sections: list[str] = [section for section in extra_sections if section]
        plan_section = self.get_active_plan_prompt_section()
        if plan_section:
            turn_sections.append(plan_section)

        if self._memory is not None:
            # The running turn's recall when a turn set one; the message-free section
            # otherwise (a request built outside a turn has no message to rank against).
            memory_section = self._active_session.recalled_memory
            if memory_section is None:
                memory_section = self._memory.format_prompt_section()
            if memory_section:
                turn_sections.append(memory_section)

        messages = list(self._history)
        combined_section = "\n\n".join(sections)
        turn_context = turn_context_block(turn_sections)

        anchored_turn = has_anchor(messages)
        if anchored_turn:
            # The anchor is re-framed whether or not there is a section to inject: the
            # framing has to track the model either way, and returning `self._history`
            # untouched on the no-sections path is how #921 stayed live for an agent with
            # no ontology, skills, plan or memory wired up.
            anchored = messages[0].content or ""
            base_sys = adapt_system_prompt(
                self._system_prompt_base() if self._anchor_is_stale() else anchored,
                self._config.llm_config.model_name,
            )
        else:
            # **No anchor at all: the turn sends what `effective_system_prompt` reports**
            # (#1091). A session hydrated through `load_history` from rows whose first is
            # not a `SYSTEM` turn, `load_history([])`, and a session seeded under an empty
            # `config.system_prompt` before a persona was adopted all reach here. This
            # branch used to send the sections alone, or no system turn at all, while
            # `effective_system_prompt` named the persona's or the configured prompt.
            #
            # Resolved here, on the turn, rather than by the other two answers #1091
            # names. *Seeding an anchor at hydration* would write into the record a system
            # turn the caller did not supply and earlier turns were never sent (#1078), and
            # cannot reach a seeded session that simply had no prompt until a persona
            # arrived. *Refusing the hydration* breaks every caller that loads a
            # user-first transcript today. Synthesising is also not #1081's hazard of
            # discarding a caller's anchor: there is no caller-composed system turn to
            # discard, so nothing a caller supplied is replaced, and history is left as
            # loaded. An empty resolution synthesises nothing, exactly as
            # `SessionState.seed` writes no empty `SYSTEM` turn.
            base_sys = self.effective_system_prompt

        # One composition for both branches, so a synthesised turn carries the sections in
        # exactly the order and spacing an anchored one does.
        resolved = compose_system_message(base_sys, combined_section)
        history = messages[1:] if anchored_turn else messages
        # The context state (#1443): which log entry each message shows, in which form,
        # and the entry an already-shown result refers back to (§5.8, Rule 2). The history
        # names the log bodies and their order; each entry is rendered in its form from
        # the log (#1848) -- its own body, or the rendering a compaction dropped it to --
        # so a rebuild that reads only the log renders the same request.
        # A body an epoch recorded as a rendering shows the entry it renders; any other
        # keeps the form the epochs before recorded for it, where its message carries none
        # (`shown_form`, #1866). After a compaction, the entries it derived from the
        # history before it are the new epoch's (`compacted_entries`, #1848).
        session = self._active_session
        logged = session.logged_history()[len(messages) - len(history) :]
        prior, renderings = self._recorded(session.context_epochs)
        shown = shown_entries(logged, prior, {**renderings, **session.compacted_entries})
        by_entry = dict(logged)
        conversation = render_entries(shown, by_entry.__getitem__)
        return RequestLayers(
            identity=base_sys,
            slow_context=combined_section,
            system_message=anchored_turn or bool(resolved),
            conversation=tuple(conversation),
            turn_context=turn_context,
            shown=shown,
        )
