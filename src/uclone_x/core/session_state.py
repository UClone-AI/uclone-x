"""The session record's model: `SessionState` and what it carries (#1734).

Lowered out of `agent/session.py` so that `core/session_store.py` can type
`SessionStoreProtocol` over `SessionState` without importing the single-agent package.
`SessionStore`, the one persistence boundary, stays in `agent/session.py`, which
re-exports every name below under its old spelling -- each is the one class object, not a
copy. "This module's docstring" in the methods below is `agent/session.py`'s: the
paragraph "Validation is a property of the constructor, not of the type" lives there,
because the family it names spans both modules.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from uclone_x.core.context_state import ContextEntry, ContextEpoch
from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.models import PersonaDefinition, PlanState
from uclone_x.core.secrets import redact_credentials, redact_log_payload
from uclone_x.core.session_log import (
    SessionLogEntry,
    history_entry_ids,
    is_kept_text,
    logged_message,
)
from uclone_x.llm.models import ChatMessage, MessageRole, ToolCallRequest

__all__ = [
    "AnchorAuthor",
    "AnchorProvenance",
    "ContextSnapshot",
    "SessionState",
    "content_digest",
    "history_entries_not_their_messages",
    "log_position",
    "recorded_before_history_entries",
    "redact_message",
]

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def log_position(entry: str) -> int | None:
    """The log position a `e<n>` entry id names, or `None` for anything else."""
    digits = entry.removeprefix("e")
    if digits == entry or not digits.isdigit():
        return None
    return int(digits)


_log_position = log_position


def _logged_entries(
    log: Sequence[SessionLogEntry], messages: Sequence[ChatMessage]
) -> tuple[str, ...]:
    """The log entry of each of `messages`, by digest, or none unless `log` holds all (#1985).

    A message the log cannot keep -- a form that records no source -- names none here;
    `SessionState` refuses it in its own words.
    """
    if not log or not messages:
        return ()
    try:
        digests = [logged_message(redact_message(message)).digest for message in messages]
    except ValueError:
        return ()
    known = history_entry_ids(log, digests)
    if None in known:
        return ()
    return tuple(entry for entry in known if entry is not None)


def _now_iso() -> str:
    """Current UTC instant as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def redact_message(message: ChatMessage) -> ChatMessage:
    """Return `message` with any credential shapes in content or tool calls redacted on write.

    Option A mitigation from #569: credentials pasted into chat or returned from tools
    must not enter session state or durable persistence unmasked.

    Known Limitations of Pattern-Based Redaction (Option A):
    Pattern-based redaction is a heuristic mitigation, not an absolute guarantee.
    It catches known credential shapes (e.g. OpenAI `sk-...`, GitHub `ghp_...`, Anthropic
    `sk-ant-...`, AWS access keys, Bearer tokens, and explicit secret assignments), but cannot
    detect arbitrary high-entropy strings, bespoke tokens without prefixes, or obfuscated
    secrets without high false-positive rates. Per Principle 3 (P3) and threat model T3,
    redaction on write reduces retention risk, but does not replace process-level isolation
    or host egress boundaries.
    """
    if isinstance(message, dict):
        raw_obj: object = message
        clean_dict: dict[str, object] = dict(cast(dict[str, object], raw_obj))
        content_val = clean_dict.get("content")
        if isinstance(content_val, str):
            clean_dict["content"] = redact_credentials(content_val)
        tool_calls_val = clean_dict.get("tool_calls")
        if isinstance(tool_calls_val, (list, tuple)):
            clean_tcs: list[object] = []
            for item in cast("list[object] | tuple[object, ...]", tool_calls_val):
                if isinstance(item, dict):
                    tc_dict = dict(cast(dict[str, object], item))
                    args_val = tc_dict.get("arguments")
                    if isinstance(args_val, dict):
                        unwrapped = unwrap_immutable(cast(dict[str, object], args_val))
                        tc_dict["arguments"] = redact_log_payload(unwrapped)
                    clean_tcs.append(tc_dict)
                else:
                    clean_tcs.append(item)
            clean_dict["tool_calls"] = clean_tcs
        try:
            return ChatMessage.model_validate(clean_dict)
        except Exception:
            return cast(ChatMessage, clean_dict)

    new_content = message.content
    if message.content is not None:
        new_content = redact_credentials(message.content)

    new_tool_calls: list[ToolCallRequest] = []
    tool_calls_modified = False
    for tc in message.tool_calls:
        if tc.arguments:
            unwrapped = cast(dict[str, Any], unwrap_immutable(tc.arguments))
            sanitized_args = cast(dict[str, Any], redact_log_payload(unwrapped))
            if sanitized_args != unwrapped:
                tool_calls_modified = True
                new_tool_calls.append(
                    ToolCallRequest(
                        id=tc.id,
                        name=tc.name,
                        arguments=sanitized_args,
                    )
                )
                continue
        new_tool_calls.append(tc)

    if new_content == message.content and not tool_calls_modified:
        return message

    return ChatMessage(
        role=message.role,
        content=new_content,
        name=message.name,
        tool_call_id=message.tool_call_id,
        tool_calls=tuple(new_tool_calls),
        compaction_ledger=message.compaction_ledger,
        form=message.form,
        # A picture is not text and is not redacted; dropping it here lost a screenshot
        # whose tool result happened to quote a credential shape (#2107).
        images=message.images,
    )


class AnchorAuthor(StrEnum):
    """Who composed a session's anchored `SYSTEM` turn.

    Two members, because the turn builder asks exactly one question of a restored anchor:
    is it text this agent composed from its own persona axis, or text that arrived from
    outside it? Only the first is the agent's to re-resolve.
    """

    AGENT = "agent"
    CALLER = "caller"


class AnchorProvenance(BaseModel):
    """What composed a session's anchored `SYSTEM` turn, in the shape the store writes (#1152).

    The agent's own working copy carries this as a `PersonaDefinition`, a `None` meaning
    "the axis resolved to no persona", or a marker meaning "the caller wrote it". That
    union is a Python type and does not survive a JSON record, so it is spelled here as
    an author plus the persona resolution the author had — which is what makes a restored
    session able to say whether its anchor is re-resolvable, instead of every restored
    anchor reading as the caller's.

    `persona` is the resolution in force when the agent composed the anchor, and `None`
    under `AGENT` is a real answer — "composed under no persona" — not a missing one. That
    is why the *absence of this whole object* is what records "unknown": a record written
    before this field existed carries no `anchor_provenance` at all, and collapsing that
    into `AGENT`/`None` would claim a provenance nobody stamped and make every legacy
    anchor re-resolvable, discarding a caller's text (P6, and the mode #1081 records as
    measured and rejected for PR #937).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    author: AnchorAuthor = Field(description="Whether the agent or the caller composed the anchor")
    persona: PersonaDefinition | None = Field(
        default=None,
        description="The persona resolution the agent composed the anchor under; always "
        "absent for a caller-composed anchor.",
    )

    @model_validator(mode="after")
    def _caller_anchors_carry_no_persona(self) -> AnchorProvenance:
        """Refuse the one combination that would be read as a claim nobody can make.

        A caller-composed anchor has no axis position behind it by definition. A record
        pairing `CALLER` with a persona would either be ignored — a field written and not
        read — or be taken as "the caller wrote it under this persona", which is not a
        thing the agent can know. Refused at the constructor so it cannot reach the store.
        """
        if self.author is AnchorAuthor.CALLER and self.persona is not None:
            raise ValueError(
                "A caller-composed anchor carries no persona: the agent did not compose "
                "that text and has no axis position to attribute it to. Use "
                "AnchorProvenance(author=AnchorAuthor.AGENT, persona=...) for an anchor "
                "the agent composed."
            )
        return self


def content_digest(text: str) -> str:
    """SHA-256 hex of `text` as UTF-8: the address a layer body is stored under."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ContextSnapshot(BaseModel):
    """What one request carried besides the conversation: layers 1-3, turn context, model.

    Appended to the session by the agent on every turn, and again within a turn when one of
    these changes (a nudge adds to the turn context). A `REQUEST_CONTEXT` event names the
    snapshot its request was built from, by `snapshot_id`, and records only the messages
    the conversation gained since the previous request. The two together rebuild the
    request exactly: see `rebuild_requests` in `agent/request_record.py`.

    Every layer is stored by hash here and as a body in the session store, once per
    distinct text: the tool schemas, the identity prompt, the slow context and the turn
    context. The record grows by a few hashes per turn rather than by the prompt, the tool
    list or the turn context, which would otherwise be copied into every snapshot and
    rewritten with the whole record on every save. The bodies are redacted on disk; see
    `SessionStore.save_context_body`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    turn_index: int = Field(description="The session's turn counter when this was taken.")
    tools_digest: str = Field(description="SHA-256 of the tool schemas, as sent, in order.")
    identity_digest: str = Field(description="SHA-256 of the identity prompt.")
    slow_context_digest: str = Field(
        description="SHA-256 of the slow context: invariants, skills and workspace."
    )
    system_message: bool = Field(
        description="Whether the request opened with a system message built from the "
        "identity and slow context. False only when both were empty and there was no anchor."
    )
    turn_context_digest: str = Field(
        description="SHA-256 of the `[Turn Context]` block, or of the empty text when there "
        "was none."
    )
    model: str | None
    temperature: float
    max_tokens: int | None
    auto_compact: bool
    compaction_threshold_tokens: int
    tools_module: str | None = Field(
        default=None,
        description="The tools module the request's tools layer was built under (#2188): "
        "`pinned` or `bound`. `None` is `native`, the default, and is left out of the "
        "record, so a default snapshot and its id are what they were before modules. Read "
        "through `recorded_tools_module`, which refuses a name this build does not know.",
    )

    @model_serializer(mode="wrap")
    def _omit_native_module(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """Leave `tools_module` out when unset, so a default snapshot is written, and
        hashed into its `snapshot_id`, in the shape a build from before the field reads
        (#1844)."""
        data: dict[str, object] = handler(self)
        if data.get("tools_module") is None:
            data.pop("tools_module", None)
        return data

    @field_validator(
        "tools_digest", "identity_digest", "slow_context_digest", "turn_context_digest"
    )
    @classmethod
    def _is_sha256(cls, value: str) -> str:
        if not _SHA256_HEX.fullmatch(value):
            raise ValueError("A layer digest is 64 lowercase hex characters (SHA-256).")
        return value

    @property
    def snapshot_id(self) -> str:
        """SHA-256 of this snapshot's canonical JSON: the name events refer to it by."""
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return content_digest(canonical)


class SessionState(BaseModel):
    """One conversation session held by the Core Engine.

    Frozen: a turn produces a new state rather than mutating the old one, so a state
    handed to a caller cannot be changed underneath it. `turn_counter` travels with the
    messages because the two are only meaningful together — the CLI `/reset` defect was
    precisely that it replaced the messages and left the counter running.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    session_id: str
    agent_id: str
    messages: tuple[ChatMessage, ...] = Field(default_factory=tuple)

    @field_validator("messages", mode="after")
    @classmethod
    def _redact_messages(cls, messages: tuple[ChatMessage, ...]) -> tuple[ChatMessage, ...]:
        """Redact known credential shapes on write across all messages (#569).

        A tool result shown in a smaller form is held as what it records -- its form and
        the kept result it was cut from (`ChatMessage.rendered_from`) -- and never as text:
        its text is rendered from that result when it is shown, so a record cannot hold
        text other than what was sent (#1848). A record written before that holds a form's
        text, or no source for it, and is refused here rather than shown as whatever text
        it held; the store treats a record that fails validation as absent and sets it
        aside (#1844).
        """
        for position, message in enumerate(messages):
            if message.form is not None and message.rendered_from is None:
                raise ValueError(
                    f"messages[{position}] is a {message.form} that records no result it "
                    "was rendered from (#1848)"
                )
            if message.form is not None and message.content is not None:
                raise ValueError(
                    f"messages[{position}] is a {message.form} that holds text; a form is "
                    "recorded, not written (#1848)"
                )
        return tuple(redact_message(m) for m in messages)

    plan: PlanState | None = Field(
        default=None, description="Active execution plan state for the session"
    )
    turn_counter: int = 0
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
    revision: int = Field(
        default=0,
        description="Monotonic write counter advanced by `SessionStore.save`. A state "
        "carries the revision of the record it was read from, and `save` refuses it if "
        "the record has moved on since — see `SessionStore.save` for the whole contract, "
        "including the bound on what the precondition guarantees, and #219 for what it "
        "replaces. `0` means 'never persisted', which is also what a legacy record with "
        "no `revision` key reads back as. **Not tamper-proof**: no *snapshot* door "
        "advances it, but a directly constructed or `model_copy`-ed `SessionState` can "
        "carry any value, which `save` cannot distinguish from a genuine read. That is "
        "accepted deliberately and the alternatives are rejected by name — see the #248 "
        "decision in `SessionStore.save`, and do not read this field as a security "
        "control: `save` is not the only route to the record.",
    )
    anchor_provenance: AnchorProvenance | None = Field(
        default=None,
        description="What composed `messages[0]` when it is a `SYSTEM` turn (#1152). "
        "`None` means the record does not say — either it predates this field, or it was "
        "built by a door that has no answer to give. It does **not** mean 'nobody' and it "
        "does not mean 'no persona': a restored anchor with no recorded provenance is "
        "left alone and reported, never re-resolved as though it had been stamped. Only "
        "`BaseAgent`'s live session knows the answer, so only its snapshot writes this.",
    )
    context_snapshots: tuple[ContextSnapshot, ...] = Field(
        default_factory=tuple,
        description="What each turn's requests carried besides the conversation, oldest "
        "first (#1421). A record written before this field reads back with none, and a "
        "reset clears them together with the event log that refers to them.",
    )
    session_log: tuple[SessionLogEntry, ...] = Field(
        default=(),
        description="Every message that entered `messages`, oldest first, append-only "
        "(#1443); see `core/session_log.py`. Each entry names its message's body in the "
        "context body store. A reset clears it with the history it describes.",
    )
    context_epochs: tuple[ContextEpoch, ...] = Field(
        default=(),
        description="What each request showed of `session_log`, per epoch: an ordered "
        "list of (entry id, form) that only grows within an epoch (#1443); see "
        "`core/context_state.py`. A record written before this field reads back with "
        "none, and the next request opens the first epoch. A reset clears it.",
    )
    compacted_entries: tuple[ContextEntry, ...] = Field(
        default=(),
        description="What a compaction derived for the request that opens the next epoch, "
        "saved until that request records it (#1848): per logged message of the compacted "
        "history, the entry it shows and its form, keyed by the log entry whose body it is "
        "(`ContextEntry.body`). Left out of the record when empty, so a record with none "
        "is written as before (#1844). A record that has it is not readable by a build "
        "from before this field, which sets it aside (#1844).",
    )
    epoch_causes: tuple[str, ...] = Field(
        default=(),
        description="Why the next request opens a new epoch, as declared since the last "
        "request and saved until that request records it (#1848): `compaction`, "
        "`rollback`, and the other causes `ContextEpoch.opened_by` names. `restored` is "
        "not saved, because loading the record declares it again. Left out of the record "
        "when empty, so a record with none is written as before (#1844). A record that "
        "has it is not readable by a build from before this field, which sets it aside "
        "(#1844).",
    )

    history_entries: tuple[str, ...] = Field(
        default=(),
        description="The log entry each of `messages` is, in order (#1848): the history as "
        "the live session derives it from its last epoch and the log, so a load adopts "
        "the same entries by id rather than matching bodies. Every record a session "
        "writes carries it; a stored record that has a log and messages but no entries "
        "was written before it and is refused (`recorded_before_history_entries`), as is one "
        "whose message is not the body of the entry it names "
        "(`history_entries_not_their_messages`, #1985). Empty on a state built in-process "
        "from messages alone, whose messages are then matched to the log by digest; "
        "`with_messages` names them when its log holds every message. Left out of the "
        "record when empty.",
    )

    @model_serializer(mode="wrap")
    def _omit_empty_pending_fields(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        """Leave `compacted_entries`, `epoch_causes` and `history_entries` out when empty, so
        a record without them is written in the shape a build from before the fields reads
        (#1844)."""
        data: dict[str, object] = handler(self)
        if not data.get("compacted_entries"):
            data.pop("compacted_entries", None)
        if not data.get("epoch_causes"):
            data.pop("epoch_causes", None)
        if not data.get("history_entries"):
            data.pop("history_entries", None)
        if data.get("history_entries") and data.get("session_log"):
            data.pop("messages", None)
        return data

    @model_validator(mode="after")
    def _history_entries_are_logged_messages(self) -> SessionState:
        """Refuse `history_entries` that do not name one logged message per message (#1848).

        Checked by structure alone -- counts and ids, no body is read or hashed -- so a
        record build costs nothing per message here (#1974, item 11). Whether each entry's
        body is its message is checked once, where a record is adopted (`from_state`).
        """
        entries = self.history_entries
        if not entries:
            return self
        if self.messages and len(entries) != len(self.messages):
            raise ValueError(
                f"history_entries names {len(entries)} log entries for "
                f"{len(self.messages)} messages"
            )
        for position, entry in enumerate(entries):
            index = _log_position(entry)
            if index is None or index >= len(self.session_log):
                raise ValueError(f"history_entries[{position}] is {entry!r}, not in the log")
            if is_kept_text(self.session_log[index]):
                raise ValueError(
                    f"history_entries[{position}] is {entry!r}, a kept text, not a message"
                )
        return self

    @field_validator("context_epochs", mode="after")
    @classmethod
    def _epoch_numbers_are_positions(
        cls, epochs: tuple[ContextEpoch, ...]
    ) -> tuple[ContextEpoch, ...]:
        """Refuse epochs whose numbers are not `0..n-1`: one was dropped or reordered."""
        for position, epoch in enumerate(epochs):
            if epoch.number != position:
                raise ValueError(
                    f"context_epochs[{position}] is numbered {epoch.number}; epochs are "
                    "only appended, so epoch n is numbered n."
                )
        return epochs

    @field_validator("session_log", mode="after")
    @classmethod
    def _log_ids_are_positions(
        cls, log: tuple[SessionLogEntry, ...]
    ) -> tuple[SessionLogEntry, ...]:
        """Refuse a log whose ids are not `e0..e(n-1)`: an entry was dropped or reordered."""
        for position, entry in enumerate(log):
            if entry.id != f"e{position}":
                raise ValueError(
                    f"session_log entry {position} has id {entry.id!r}; the log is "
                    "append-only, so entry n is `e<n>`."
                )
        return log

    @classmethod
    def seed(cls, session_id: str, agent_id: str, system_prompt: str = "") -> SessionState:
        """Create a fresh session seeded exactly as `BaseAgent.__init__` seeds history.

        A `SYSTEM` message if and only if `system_prompt` is non-empty. The `or ""`
        spelling the CLI used produced an empty `SYSTEM` message instead, which is a
        state no other path in the tree can construct.
        """
        messages: tuple[ChatMessage, ...] = ()
        if system_prompt:
            messages = (ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),)
        return cls(session_id=session_id, agent_id=agent_id, messages=messages, plan=None)

    def reset(
        self, system_prompt: str = "", anchor_provenance: AnchorProvenance | None = None
    ) -> SessionState:
        """Return this session purged back to its seeded state.

        `anchor_provenance` travels beside `system_prompt` and not through `self` because
        a reset *composes a new anchor*: whatever composed the old one has been discarded
        along with it, so carrying the old stamp over would describe text that is no
        longer there. The caller supplying the prompt is the only one that knows what
        composed it; omitting it records "this record does not say", which is the honest
        answer for a reset performed outside an agent.

        `created_at` is carried over — a reset session is the same session, and losing
        its creation time would make the store unable to say how long it has existed.

        `revision` is carried over for a sharper reason: a reset is a *write to the same
        record*, so it has to satisfy the same compare-and-swap as any other write. A
        reset that zeroed the revision alongside the turn counter would be refused by
        `SessionStore.save` on every already-persisted session — the reset would become
        the one operation that could never be saved.

        Constructed rather than `model_copy`-ed, for the reason in
        "Validation is a property of the constructor, not of the type" in this module's
        docstring, which also names the other door in this family.
        """
        seeded = SessionState.seed(
            session_id=self.session_id,
            agent_id=self.agent_id,
            system_prompt=system_prompt,
        )
        # Constructed rather than `model_copy`-ed: `model_copy` does not validate, and
        # every state this module hands back must be one `SessionStore.load` can read.
        return SessionState(
            session_id=seeded.session_id,
            agent_id=seeded.agent_id,
            messages=seeded.messages,
            plan=seeded.plan,
            turn_counter=0,
            created_at=self.created_at,
            updated_at=_now_iso(),
            revision=self.revision,
            anchor_provenance=anchor_provenance,
        )

    def with_messages(
        self,
        messages: Sequence[ChatMessage],
        turn_counter: int | None = None,
        updated_at: str | None = None,
    ) -> SessionState:
        """Return this session carrying a new message sequence.

        Built through the constructor, **not** `model_copy`. `model_copy(update=...)`
        performs no validation, so `with_messages(msgs, turn_counter="9")` used to be
        accepted here, persist `"9"` to disk, and then read back as `None` from
        `SessionStore.load` — a write that reported success and a record that
        subsequently claimed no session existed. That is the silent-wipe shape, arriving
        through this module's own public API, and `strict=True` cannot catch it unless
        the value actually passes through validation.

        See "Validation is a property of the constructor, not of the type" in this
        module's docstring for the other door in this family and why closing one is
        not enough.

        `revision` is carried through unchanged and takes no parameter, which is what
        makes the compare-and-swap in `SessionStore.save` work across a turn: the state a
        caller builds from what it read still claims the revision it read, so a write
        built on a stale read is still recognisably stale, and no caller can set it to
        something else.

        `anchor_provenance` is kept **only while the anchor itself is unchanged**. This
        method replaces the whole sequence, so it can replace `messages[0]` — and a stamp
        describing text that is no longer there is worse than no stamp, because it is the
        one input `BaseAgent._anchor_is_stale` trusts. Compaction, which keeps the anchor
        and drops turns behind it, therefore keeps its provenance; a wholesale
        `load_history` does not.

        `updated_at` is the contrast that explains why it is not the concurrency token.
        It defaults to now here, and #223 made it *preservable* by parameter — so a
        caller can carry one across a `with_messages`, which it could not before. That
        still does not make it able to arbitrate, for three reasons that `revision`
        avoids by construction: preserving it is **opt-in**, so a precondition built on it
        would be silently unenforced for every caller that did not pass it; it is
        **caller-settable**, so a stale writer can simply supply the value that will
        match; and it is a wall-clock string rather than a counter, so two writes inside
        one clock tick are indistinguishable. `SessionStore.save` stamps it again on the
        way to disk in any case.

        `history_entries` names the log entry of each new message when the log holds every
        one of them, matched by digest as a load matches a state that names none
        (`history_entry_ids`); otherwise none, and the live session that adopts the state
        logs what the log lacks. Dropped whole, a state with a log would be saved naming
        none, and the store would set it aside on the next load as a record from before
        #1848 (#1985).
        """
        replacement = tuple(messages)
        return SessionState(
            session_id=self.session_id,
            agent_id=self.agent_id,
            messages=replacement,
            plan=self.plan,
            turn_counter=self.turn_counter if turn_counter is None else turn_counter,
            created_at=self.created_at,
            updated_at=_now_iso() if updated_at is None else updated_at,
            revision=self.revision,
            anchor_provenance=(
                self.anchor_provenance if replacement[:1] == self.messages[:1] else None
            ),
            context_snapshots=self.context_snapshots,
            session_log=self.session_log,
            context_epochs=self.context_epochs,
            history_entries=_logged_entries(self.session_log, replacement),
        )

    def with_plan(self, plan: PlanState | None) -> SessionState:
        """Return this session carrying a new plan state."""
        return SessionState(
            session_id=self.session_id,
            agent_id=self.agent_id,
            messages=self.messages,
            plan=plan,
            turn_counter=self.turn_counter,
            created_at=self.created_at,
            updated_at=_now_iso(),
            revision=self.revision,
            anchor_provenance=self.anchor_provenance,
            context_snapshots=self.context_snapshots,
            session_log=self.session_log,
            context_epochs=self.context_epochs,
            compacted_entries=self.compacted_entries,
            epoch_causes=self.epoch_causes,
            history_entries=self.history_entries,
        )


def recorded_before_history_entries(state: SessionState) -> str | None:
    """Why a stored record is from before its history was read from the log, or `None`.

    Every record a session writes names the log entry of each message it holds
    (`history_entries`, #1848). One with a log but no entries was written by an earlier
    build: its messages can only be matched to the log by their text, and one from before
    #1854 holds a compaction's stub as a whole message, which would be labelled `full`.
    It is not converted -- there are no users to convert for -- so the store reads it as
    unreadable and sets it aside (#1844, #1974 item 5). A record with no log is read as
    before, its messages logged when it is adopted: with no log it keeps no full result
    for any message to be a form of, so each message is all there is of it, and `full`
    says so. Decided by structure alone; no message text is read, so a tool's output
    that quotes a stub's header never trips it.
    """
    if state.messages and state.session_log and not state.history_entries:
        return (
            "its messages name no log entries: a record from before the history was "
            "read from the log (#1848)"
        )
    return None


def history_entries_not_their_messages(state: SessionState) -> str | None:
    """Why a stored record's messages are not the log entries it names, or `None` (#1985).

    The model refuses entries by structure alone -- a count that does not match, an id
    not in the log, a kept text -- and reads no message. This is the rest: each message
    is the body of the entry the record names for it, by digest. A record that fails it
    cannot be adopted (`_LiveSession.from_state` refuses it on every resume), so the
    store reads it as unreadable and sets it aside like any record this build cannot
    read (#1844), instead of failing each time the session is opened.

    Each message is hashed as the record holds it, not redacted again: `SessionState`
    redacts its messages as it is built, and redacting is most of the cost of this read,
    which runs whenever the store reads a record (#1974 item 11).
    """
    for entry_id, message in zip(state.history_entries, state.messages, strict=False):
        index = _log_position(entry_id)
        if index is None or state.session_log[index].digest != logged_message(message).digest:
            return f"the message recorded for log entry {entry_id} is not its body (#1985)"
    return None
