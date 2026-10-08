"""Typed error taxonomy for UClone-X.

Principle 6 (Fail-Fast & Zero Silent Fallbacks) requires failures to propagate as
explicit, attributable errors rather than as substituted values. That is only possible
if there is a taxonomy to raise from, so this module is the single root for every
runtime error the framework raises.

The A2A section mirrors the A2A protocol specification's error table one type at a
time, because the A2A specification requires the in-process fastpath to raise the same
typed error a remote caller would have received over the JSON-RPC binding. The canonical
code mappings ride on the class so a binding can translate without a lookup table of its
own.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    # Import-time cycle, annotation-time not: `uclone_x.agent.session` imports this
    # module, so the real import must never run. `from __future__ import annotations`
    # makes every annotation below a string, so the forward reference resolves for
    # Pyright and for `typing.get_type_hints` without the runtime import existing.
    from uclone_x.agent.session import SessionState

__all__ = [
    "A2AError",
    "ADKConversionError",
    "ADKMalformedContentError",
    "ADKUnmappedRoleError",
    "ADKUnrepresentableMessageError",
    "AgentStateError",
    "BudgetExceededError",
    "CapabilityUnavailableError",
    "ComfyUIError",
    "ComfyUIExecutionError",
    "ContentTypeNotSupportedError",
    "DashboardNotIdentifiedError",
    "DashboardStopUnconfirmedError",
    "EmbedderNotConfiguredError",
    "EmbeddingDimensionError",
    "EmbeddingError",
    "ExtendedAgentCardNotConfiguredError",
    "ExtensionSupportRequiredError",
    "FormTextMismatchError",
    "FrontendBuildFailedError",
    "HeadRoomWriteError",
    "InvalidAgentResponseError",
    "InvalidStateTransitionError",
    "LLMConnectorNotConfiguredError",
    "LLMCredentialsNotConfiguredError",
    "LLMError",
    "LLMModelNotConfiguredError",
    "LLMProviderError",
    "LLMProviderNotConfiguredError",
    "LLMStreamInterruptedError",
    "LLMTimeoutError",
    "ListeningProcessLookupError",
    "LogHeaderError",
    "LogOffsetContiguityError",
    "LogOffsetCorruptCursorError",
    "LogOffsetError",
    "LogOffsetSessionMismatchError",
    "LogVersionError",
    "MalformedToolCallArgumentsError",
    "MemoryStoreUnreadableError",
    "MissingDependencyError",
    "MissingProvenanceError",
    "ModelLacksToolSupportError",
    "ModelNotAvailableError",
    "NothingToRetryError",
    "OntologyContradictionError",
    "OntologyError",
    "OntologyHashMismatchError",
    "OntologyPromotionError",
    "OntologyRetractionBlockedError",
    "OntologyViolationError",
    "ParticipantNotResolvableError",
    "PathTraversalError",
    "PlainRefusalError",
    "PlanGenerationError",
    "PromotionCriteriaNotMetError",
    "ProvenanceError",
    "ProviderAuthError",
    "ProviderFailureError",
    "ProviderFailureKind",
    "ProviderOutageError",
    "ProviderQuotaError",
    "ProviderResponseError",
    "ProviderUnreachableError",
    "PushNotificationNotSupportedError",
    "RoomAlreadyExistsError",
    "RoomError",
    "RoomWorkspaceRefusedError",
    "WorkspaceRefusalCode",
    "RoomIdError",
    "RoomNotFoundError",
    "SandboxViolationError",
    "SecondHumanInRoomError",
    "SessionEventLogNotConfiguredError",
    "SessionIdCollisionError",
    "SessionMutationDuringTurnError",
    "SessionRecordUnreadableError",
    "SessionStoreNotConfiguredError",
    "SessionSwitchWhileRunningError",
    "SkillAuditError",
    "SkillNotApprovedError",
    "SpeakerSelectionError",
    "StaleRoomWriteError",
    "StaleSessionWriteError",
    "StepBudgetExceededError",
    "StructuredOutputUnsupportedError",
    "TaskNotCancelableError",
    "TaskNotFoundError",
    "TelemetryError",
    "TelemetryExportError",
    "TokenBudgetExhaustedError",
    "ToolError",
    "TurnNotLandedError",
    "TurnNotStartedError",
    "UCloneXError",
    "UnknownLogEventError",
    "UnknownRoomParticipantError",
    "UnmappableChatMessageError",
    "UnparseableDirectiveError",
    "UnreadableRoomRecordError",
    "UnsupportedOperationError",
    "VersionNotSupportedError",
]


class UCloneXError(Exception):
    """Root of every error raised by the UClone-X runtime."""


class PlainRefusalError(UCloneXError):
    """An operation refused for a reason already written for a person.

    Its message is shown as it is: a tool that raises one fails with exactly this text,
    not with the exception's class name and a prefix around it, so the refusal a
    conversation shows is plain words (#1555).

    `reason_code`, when given, is a short stable name for *why* (`no_image_engine`): the
    tool's result carries it as `{"reason_code": ...}` in its output, so a caller can
    branch on the reason without parsing the sentence.
    """

    def __init__(self, message: str, *, reason_code: str | None = None) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class MemoryStoreUnreadableError(UCloneXError):
    """A clone's saved memory document is on disk and cannot be read (#1401).

    Raised by `memory.store.read_saved_facts`, which changes nothing: unlike loading a
    store to write to it, a read never moves the document aside. The message is plain and
    names neither the file nor the parser; `path` and `cause` carry both for the log.
    """

    def __init__(self, message: str, *, path: Path | None = None, cause: str | None = None) -> None:
        super().__init__(message)
        self.path = path
        self.cause = cause


class MissingDependencyError(UCloneXError, ImportError):
    """An optional framework or dependency required for a shell or adapter is missing.

    Principle 6 (Fail-Fast & Zero Silent Fallbacks) requires that optional framework
    dependencies fail at composition time with an explicit, attributable error rather
    than at turn time with silent fallbacks. Core-shell architecture §1 goal 3.
    """

    def __init__(
        self,
        extra: str,
        package: str,
        feature: str | None = None,
    ) -> None:
        self.extra = extra
        self.package = package
        self.feature = feature
        msg = f"Optional dependency '{package}' is required"
        if feature:
            msg += f" for {feature}"
        msg += f". Install it with: pip install 'uclone-x[{extra}]'"
        super().__init__(msg)


# --------------------------------------------------------------------------------------
# Agent lifecycle (Principle 1, FR-1)
# --------------------------------------------------------------------------------------


class AgentStateError(UCloneXError):
    """Base error for agent lifecycle and state machine violations."""


class InvalidStateTransitionError(AgentStateError):
    """Attempted an illegal transition between agent lifecycle states."""


class PlanGenerationError(AgentStateError):
    """A plan was requested and no genuine plan could be produced.

    Principle 6 forbids "substituted mock, empty or default results" unconditionally, so
    a planner that cannot decompose its input must raise rather than return a filler
    checklist. `uclone_x.agent.planner.PlanGenerator` raises this when the query carries
    no structure its declared heuristic can decompose.
    """


class SessionStoreNotConfiguredError(AgentStateError):
    """A session persistence operation was requested with no `SessionStore` wired.

    Follows `LLMConnectorNotConfiguredError`: nothing was attempted, so there is nothing
    to retry or fail over to, and Principle 6 forbids reporting a write that did not
    happen as a success. Silently returning would leave the caller believing a session
    was persisted, which is the failure mode that makes an in-memory session look durable.
    """


class SessionEventLogNotConfiguredError(AgentStateError):
    """A `SessionStore` was handed durable turn events it has no log to write to.

    Raised before anything is written, so the record, the events and the caller's queue
    are all left as they were. The alternative -- writing the record and dropping the
    events -- is what #1442 found every production store doing, silently (P6).
    """


class SessionMutationDuringTurnError(AgentStateError):
    """A session was reset or switched while a reasoning turn was still in flight.

    `execute_turn` holds `_turn_lock` across the whole turn and appends to the *active*
    session as it goes. Resetting that session mid-turn discards the user message the
    turn is answering, so the turn completes into the reset session and yields
    `[SYSTEM, ASSISTANT]` — an answer with no question — reported as
    `TurnResult(is_completed=True)` with a turn index the reset had already zeroed.
    Switching mid-turn is the same defect one step over: the assistant message lands in
    whichever conversation became active.

    Neither is recoverable after the fact and neither is visible to the caller, so per
    P6 the mutation is refused rather than allowed to corrupt the transcript. Await the
    turn, then reset.
    """


class SessionSwitchWhileRunningError(AgentStateError):
    """A running agent's session switch could not be carried onto the event plane (#225).

    **This no longer means "switching while running is forbidden".** It was that until
    #225: `BaseAgent.start` bound its subscription to `session.{session_id}` and there was
    no route off it, so every switch on a started agent raised. `switch_session` now
    repoints the live subscription instead, and the ordinary case succeeds.

    What remains is the case where the repointing itself is refused — the bus's topic
    allowlist or wildcard-capability rules reject `session.{new_session_id}`, so the agent
    would end up reporting a session whose events it cannot receive. That is the same
    mis-routing hazard the error was named for, reached from the other side, and per P6 it
    fails fast: the subscription, the active session id and the hosted-session map are
    left exactly as they were rather than the agent proceeding half-switched.
    """


class SessionIdCollisionError(AgentStateError):
    """A session record was reached under an id that is not the id it holds (#256, #219 defect 1).

    **The filesystem, not the guard, is what folds the two ids together.** A session id is
    interpolated into a filename, and APFS and NTFS are both case-insensitive and
    normalization-insensitive, so `"SessA"` and `"sessa"` — and NFC `"séance"` against its
    NFD spelling — name **one file** on a default macOS or Windows install while remaining
    two distinct `str`s to Python. Both are legal single path components, so
    `validate_session_id` and `resolve_session_path` pass them correctly: the P3
    containment control is not implicated and there is nothing for it to refuse.

    **This is raised on the read, because that is where the defect happens.** Measured on
    `e8b3e2f`: `save(SessionState(session_id="SessA", turn_counter=7))` followed by
    `load("SESSA")` returned `SessA`'s entire conversation, and the record it returned
    *identified itself as* `SessA`. So a caller addressing a variant was handed another
    session's conversation and adopted it — cross-session information disclosure with
    nothing concurrent happening, on a shared `~/.uclone/sessions`.

    The write damage follows from that read rather than standing beside it. The caller asks
    for `"SESSA"`, receives a state whose `session_id` is `"SessA"`, appends a turn and
    persists what it was handed; `save` writes by the *state's* id, and because the caller
    read first it holds a current `revision`, so `StaleSessionWriteError`'s precondition
    **legitimately accepts** the write. `ucx run` and the dashboard both hydrate before
    writing, so that read-modify-write arm is the ordinary path. This is why refusing only
    at write time cannot close it: by then every check available — correct revision,
    correct id, single writer — passes, and the disclosure has already occurred.

    **Refusal rather than a merge or a rename, and the reason is the migration.** Encoding
    the id into a collision-free filename (percent-encoding, or a hash) makes new records
    safe and makes *every existing filename wrong*; a record whose name becomes invalid is
    not merely unreadable but invisible to `SessionStore.list_session_ids`, and `load`
    reports unparseable as absent, so a botched upgrade would look exactly like every
    session vanishing. Verifying the id **inside** the record instead needs no migration at
    all: every filename on disk stays valid, the id stays readable in `ls`, and the check
    is independent of how a given filesystem folds names, which matters because a variant
    pair created on ext4 is legal there and becomes a collision only when the tree is read
    on macOS. Readable in `ls` is not the same as *authoritative*, though: a filename and
    its record can disagree, so `SessionStore.list_session_ids` reads ids out of records
    rather than off stems. See `uclone_x.agent.session.verify_record_identity` for the single
    implementation and the rejected options in full.

    Not recoverable by rebasing, unlike `StaleSessionWriteError`: there is no shared record
    to re-apply an intent onto. The two ids are irreconcilable on this filesystem, so the
    caller must choose a different session id — which is why `asked_session_id` and
    `record_session_id` are both carried, and why the `path` they share is named.
    """

    def __init__(
        self,
        message: str,
        *,
        asked_session_id: str,
        record_session_id: str,
        path: Path,
    ) -> None:
        super().__init__(message)
        # The id the caller addressed, and the id the record on disk claims. Kept as two
        # separate attributes rather than one "session_id" because a caller handling this
        # needs to know which of the two is its own — the whole defect is that they are
        # indistinguishable to the filesystem and distinguishable to nothing else.
        self.asked_session_id = asked_session_id
        self.record_session_id = record_session_id
        # The one path both ids resolve to. Named so a report can point at the file rather
        # than leave the user to work out which of their session records is involved.
        self.path = path


class SessionRecordUnreadableError(AgentStateError):
    """A session record this build cannot read could not be moved aside, so it was not replaced (#1844).

    `SessionStore.save` and `SessionStore.delete` move a record that will not load to
    `<name>.unreadable-<UTC time>` before writing or removing anything at its name: it is
    most often a conversation a newer build wrote, and replacing it would lose it. When
    that rename fails the write is refused instead, and the record stays where it was.
    `path` is where it is; `cause` is why it could not be read. Neither belongs in text a
    person is shown.
    """

    def __init__(self, message: str, *, path: Path, cause: str) -> None:
        super().__init__(message)
        self.path = path
        self.cause = cause


class FormTextMismatchError(PlainRefusalError):
    """A shortened tool result was handed in as text other than what it records (#1848).

    A form -- an excerpt or a stub -- is held as the kept result it was cut from and the
    share it keeps, and its text is rendered from those. Text that differs from that
    rendering would be sent as something the record does not say, so it is refused. The
    message is plain; `detail` names the form and its handle, for the log only.
    """

    def __init__(self, message: str, *, detail: str) -> None:
        super().__init__(message, reason_code="form_text_mismatch")
        self.detail = detail


class StaleSessionWriteError(AgentStateError):
    """A session write was refused because the record moved on since it was read (#219).

    The optimistic-concurrency refusal. `SessionStore.save` compares the `revision` the
    caller is holding against the one on disk; a mismatch means another writer committed
    in between, so applying this write would discard that writer's update. Last-write-wins
    is what this replaces, and it was not a theoretical gap: the CLI and the UI share one
    Core record per session by design (P8's single store) and both persist every turn, so
    a `ucx run` and a dashboard on one session id lost each other's turns as a matter of
    course. Measured on the compaction path it was worse than a lost turn — the writer
    whose compaction was overwritten kept the compacted context in memory while the record
    held the pre-compaction sequence and the compaction ledger was gone from both, leaving
    no participant holding the truth.

    **Recoverable by construction, which is why it is an error and not a merge.** The
    store cannot merge two message sequences: it holds two finished states, not the
    operations that produced them, and it has no way to tell a concurrent append from a
    compaction. Concatenating them in the compaction case would resurrect exactly the
    messages the compaction deliberately dropped, which is a worse outcome than the
    refusal. So the store refuses and hands back everything needed to rebase without a
    second read: `current` is the on-disk state, `actual_revision` is where the record
    now is, and `expected_revision` is where the caller thought it was. Re-apply the
    caller's intent onto `current` and save again — `BaseAgent.hydrate_session` is the
    ready-made form of that when the caller is willing to adopt the record wholesale.

    Refusal rather than a traceback at the top of a REPL: both persistence call sites in
    the tree already catch around `persist_session` (`cli.commands.run` warns the user
    inline and continues the session; `ui.app` logs and returns the reply), so this
    surfaces as a reported non-durable turn rather than as a crash three turns into a
    conversation.
    """

    def __init__(
        self,
        message: str,
        *,
        session_id: str,
        expected_revision: int,
        actual_revision: int,
        current: SessionState,
    ) -> None:
        super().__init__(message)
        self.session_id = session_id
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision
        # The record as it now stands on disk — what to rebase onto. Typed through a
        # `TYPE_CHECKING`-only import because `uclone_x.agent.session` imports this
        # module, so a runtime import would be a cycle; the annotation is a string under
        # `from __future__ import annotations`, so the type is available to Pyright and
        # to a caller's `isinstance` without one.
        self.current: SessionState = current


# --------------------------------------------------------------------------------------
# Provenance (Principle 6, In-Band Provenance)
# --------------------------------------------------------------------------------------


class ProvenanceError(UCloneXError):
    """Base for failures in the in-band attribution chain."""


class MissingProvenanceError(ProvenanceError):
    """A result envelope crossed a component boundary without provenance.

    Principle 6: "Absence is a violation, not a default." A consumer must raise rather
    than assume the value came from the primary path.
    """


# --------------------------------------------------------------------------------------
# Budget and quota (Principle 5)
# --------------------------------------------------------------------------------------


class BudgetExceededError(UCloneXError):
    """A token ceiling was reached.

    Principle 6 classifies a quota ceiling as never eligible for retry, failover or
    substitution: it must propagate.
    """


class UsageLimitReachedError(BudgetExceededError):
    """The user's own limit on paid-model tokens is reached (`llm-token-gateway.md` §4.3).

    Raised by the usage gate before a paid call is sent, so no request reaches the
    provider. The message is written for a person: which window, when it lifts, and the
    two remedies. It carries no token figure, code or class name.

    Distinct from `ProviderQuotaError`, which is the provider's limit on the user's
    account. This one is the limit the user set in UClone-X. It is never retried and never
    answered by switching to another provider or a local model (P6): the choice is the
    user's.
    """

    def __init__(self, message: str, *, window: str, available_again_at: datetime) -> None:
        super().__init__(message)
        #: The window whose limit is reached: `per_10_minutes`, `per_5_hours` or `per_week`.
        self.window = window
        #: When usage in that window falls back under its limit, timezone-aware.
        self.available_again_at = available_again_at


class UsageLimitsUnreadableError(PlainRefusalError):
    """A usage limit is set to something that is not a number of tokens.

    Refused rather than ignored: an unreadable limit silently treated as "no limit" would
    spend past a ceiling the user believes is set (P6). The message names where the value
    came from, so it can be corrected.
    """


class StepBudgetExceededError(BudgetExceededError):
    """An agent step budget ceiling was reached (Principle 4, Principle 6).

    Principle 4 requires bounded execution with an explicit step budget
    (`AgentConfig.max_steps`). Principle 6 forbids silent continuation past declared
    bounds. When the agent exhausts the steps allowed inside a single interaction turn,
    execution halts rather than silently proceeding past the ceiling.

    The quantity bounded here is the **agent step** — one model invocation and its tool
    round, taken without returning to the caller — never the interaction turn. Bounding
    interaction turns terminates long human conversations, which is the defect recorded
    in issue 2026-09-05-001. See the agent-runtime terminology guide.
    """

    def __init__(
        self,
        message: str,
        *,
        max_steps: int | None = None,
        current_steps: int | None = None,
    ) -> None:
        super().__init__(message)
        self.max_steps = max_steps
        self.current_steps = current_steps


class CapabilityUnavailableError(UCloneXError):
    """A tool or session requires a capability the host does not provide.

    **Nothing raises this yet.** It is declared with the contracts in `uclone_x.core` so the
    refusal has a name before the code that performs it exists; the rules below are what a
    raiser is to satisfy, not a description of current behaviour.

    It is to be raised **at registration or when a session opens**, never on the turn that
    first calls the tool. That placement is the point: a tool the host cannot honour should
    not have a schema the model can see, and a failure arriving at turn time names the call
    rather than the composition that should have refused it (P6,
    the core/shell architecture note §1 goal 4 and §6.6).

    A raiser should name both the tool and the missing capabilities, so the repair does not
    require deriving which requirement went unmet. This class adds no `__init__`, so that
    obligation is the raiser's and is stated here rather than enforced.
    """


class ListeningProcessLookupError(UCloneXError):
    """Whether anything is listening on a port could not be determined.

    Distinct from finding nothing. `ucx ui stop` used to read a missing `lsof`, a query
    `lsof` rejected, and a port number outside the TCP range all as an empty result, and
    so reported "no server found" about a port it had never inspected — a substituted
    empty result, which P6 forbids (#881).
    """

    def __init__(self, port: int, reason: str) -> None:
        self.port = port
        self.reason = reason
        super().__init__(f"cannot tell whether anything is listening on port {port}: {reason}")


class DashboardNotIdentifiedError(UCloneXError):
    """Something holds a port that `ucx ui stop` cannot identify as a UClone-X dashboard.

    `stop` used to signal whatever listened on the port it was given, so a mistyped
    `--port 5432` would have stopped a local Postgres (#927). It now signals only the
    launcher a dashboard record names, while `ps` still reports that PID as the same
    process; anything else is this refusal, with the reason and a remedy, rather than a
    guess in either direction (P6).
    """

    def __init__(self, port: int, reason: str) -> None:
        self.port = port
        self.reason = reason
        super().__init__(f"refusing to signal anything on port {port}: {reason}")


class FrontendBuildFailedError(UCloneXError):
    """The dashboard's frontend had to be rebuilt before serving, and the build failed.

    Raised rather than warned about (P6). `_ensure_frontend_built` used to catch the
    failure, print a yellow line and serve the **stale** committed bundle: the dashboard
    came up looking fine while showing code the user is not running, which is the
    substituted result P6 forbids (#1075). The launcher only reaches a build at all when
    the committed bundle does *not* match the source, so there is no bundle here worth
    falling back to.
    """

    def __init__(self, frontend_dir: Path, reason: BaseException) -> None:
        self.frontend_dir = frontend_dir
        self.reason = reason
        detail = ""
        stderr = getattr(reason, "stderr", None)
        if isinstance(stderr, bytes):
            detail = stderr.decode("utf-8", "replace")
        elif isinstance(stderr, str):
            detail = stderr
        tail = f": {detail.strip()[-2000:]}" if detail.strip() else ""
        super().__init__(
            f"the dashboard bundle is out of date and `npm run build` in {frontend_dir} "
            f"failed ({type(reason).__name__}), so the stale bundle is not being served"
            f"{tail}"
        )


class DashboardStopUnconfirmedError(UCloneXError):
    """`ucx ui stop` signalled the dashboard but could not confirm what followed.

    Distinct from `DashboardNotIdentifiedError`, whose message says nothing was signalled:
    here SIGTERM was sent, so saying so would be false (review of #951).
    """

    def __init__(self, port: int, reason: str) -> None:
        self.port = port
        self.reason = reason
        super().__init__(f"stopping the dashboard on port {port} is unconfirmed: {reason}")


class LogOffsetError(UCloneXError):
    """Base error for the session log's ordering."""


class LogOffsetSessionMismatchError(LogOffsetError):
    """A batch names a session other than the log it is being appended to.

    Checked **before** contiguity, and the order is the requirement. #256 records the cost of
    the reverse: a stale-revision error naming a writer and a revision for a session the user
    had never heard of, when the actual problem was that the id did not match. An id collision
    must report as one, and only a real ordering problem reports as an ordering problem.
    """


class LogOffsetCorruptCursorError(LogOffsetError):
    """A stored cursor record cannot be read as one.

    Corruption is not a contiguity problem and not an identity problem, so it is neither of
    those errors. It is refused rather than treated as an absent log: absent hands back the
    first offset and lets a writer overwrite a log that exists.
    """


class LogOffsetContiguityError(LogOffsetError):
    """A batch does not continue the stored log from its cursor.

    A gap in an append-only log is, later, indistinguishable from an event that was never
    written — and the record model exists so that "what happened" is answerable. Refused
    rather than accepted, with the expected offset named so the repair does not require
    deriving it.
    """


class LogVersionError(UCloneXError):
    """Base error for session log versioning and compatibility failures."""


class LogHeaderError(LogVersionError):
    """A session log artefact carries an invalid, unreadable, or unsupported header.

    A reader must know the format generation before parsing events. An absent header,
    corrupted header, or unsupported log format version is refused immediately.
    """

    def __init__(
        self,
        message: str,
        *,
        path: Path | None = None,
        found_version: str | None = None,
        supported_versions: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.path = path
        self.found_version = found_version
        self.supported_versions = supported_versions


class UnknownLogEventError(LogVersionError):
    """A session log contains an unknown event type that was not marked ignorable.

    Principle 6 (Fail-Fast & Zero Silent Fallbacks): silently skipping an unknown required
    event reconstructs a session that never happened and reports success doing it. Refused
    by default unless the writer explicitly declared the event ignorable (`ignorable=True`).
    """

    def __init__(
        self,
        message: str,
        *,
        event_type: str,
        path: Path | None = None,
        event_data: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.event_type = event_type
        self.path = path
        self.event_data = event_data


class LLMError(UCloneXError):
    """Base error for LLM subsystem failures."""


class TokenBudgetExhaustedError(LLMError):
    """The model exhausted its token budget before producing an answer or tool calls (#695).

    Raised when a model terminates with `FinishReason.LENGTH` without emitting any
    `content` or `tool_calls` (for example, when a reasoning model expends its entire token
    budget inside the reasoning/thinking channel).
    """


class LLMProviderError(LLMError):
    """An LLM provider returned an error or was unreachable."""


class LLMTimeoutError(LLMProviderError):
    """A call to a provider ran past the ceiling the caller set for it (#1233).

    A subclass of `LLMProviderError`, because a call that never answered *is* a
    provider failure and every existing `except LLMProviderError` must keep
    catching it. It is a distinct type because it is a distinct fact, and the two
    were previously indistinguishable: `pull_model` reported a read timeout with
    the same `Failed to connect to Ollama: ...` wording it uses for a refused
    connection, so a pull that ran past its ceiling read as a daemon that was
    never there. P6 — the deadline is the caller's own decision, and a surface
    that cannot tell it apart from the daemon's refusal cannot name the remedy.

    `seconds` is the ceiling that expired, so a caller can say which one it was
    rather than making the reader guess at the number.
    """

    def __init__(self, message: str, *, seconds: float) -> None:
        super().__init__(message)
        self.seconds = seconds


class ModelLacksToolSupportError(LLMProviderError):
    """The chosen model cannot take tool definitions, so no clone turn can run on it.

    Ollama refuses such a request with a 400 whose body says the model "does not support
    tools" (a reasoning model such as `deepseek-r1:14b`, for one). Every clone turn sends
    tools, so a retry meets the same refusal until the model changes. The message is
    written for the person who picked the model: it names the model and what to pick
    instead, and carries no status code, response body or class name, so a surface may
    show `str()` as is. *Where* to pick it is the head's to say (P8) -- `--model` on the
    command line, Settings on the dashboard and in an ACP client -- so it is not here.
    """

    def __init__(self, model: str) -> None:
        super().__init__(
            f"The model {model} can't use tools, which UClone-X clones need. Pick a "
            "model that supports tools, for example qwen3:8b."
        )
        self.model = model


class StructuredOutputUnsupportedError(LLMProviderError):
    """The connector cannot constrain a reply to a JSON schema (`LLMRequest.response_schema`).

    Raised before any request is sent. A connector that dropped the schema and answered
    anyway would hand the caller free text it asked to be structured, with nothing on the
    result saying so (P6), so the ones that have no faithful mapping refuse instead. A
    caller that can do without structure -- the image-set planner -- catches this and
    skips the structured call.
    """

    def __init__(self, provider: str) -> None:
        super().__init__(f"The {provider} connector cannot constrain a reply to a JSON schema.")
        self.provider = provider


class ProviderFailureKind(StrEnum):
    """What went wrong on a hosted provider's side of a model call (#1630).

    One member per remedy, because telling them apart is only worth doing when each asks
    something different of the person using the app: a retired model needs another model,
    a rejected key a new key, a spent quota time or a plan, an unreachable host a
    connection, an outage patience. A value is also the `TurnStopReason` of a turn that
    failed on it, so a consumer branches on a field and not on a provider's wording.
    """

    MODEL_UNAVAILABLE = "model_unavailable"
    PROVIDER_AUTH = "provider_auth"
    PROVIDER_QUOTA = "provider_quota"
    PROVIDER_UNREACHABLE = "provider_unreachable"
    PROVIDER_OUTAGE = "provider_outage"
    PROVIDER_ERROR = "provider_error"


class ProviderFailureError(LLMProviderError):
    """A hosted provider's failure, told as what stopped and whose side it is on (#1630).

    The message is written for someone new to model providers: it names the provider and,
    where it matters, the model, and carries no status code, response body, URL or class
    name, so a surface may show `str()` as is. The raw response is logged where the failure
    is classified (`uclone_x.llm.connectors.failures`). *Where* to act -- Settings, a flag,
    an environment variable -- is the head's to say (P8), as it is for
    `ModelLacksToolSupportError`.

    `retryable` is whether the same request can succeed later with nothing changed. A
    retired model and a rejected key cannot, so a head offers no Retry for them.
    """

    kind: ClassVar[ProviderFailureKind]
    retryable: ClassVar[bool] = True

    def __init__(self, message: str, *, provider: str, model: str) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model


class ModelNotAvailableError(ProviderFailureError):
    """The provider does not serve the requested model: retired, or misspelled."""

    kind = ProviderFailureKind.MODEL_UNAVAILABLE
    retryable = False

    def __init__(self, *, provider: str, model: str) -> None:
        super().__init__(
            f"The model {model} is not available from {provider}. It may have been "
            "retired, or the name may be misspelled.",
            provider=provider,
            model=model,
        )


class ProviderAuthError(ProviderFailureError):
    """The provider refused the API key: wrong, revoked, or without access."""

    kind = ProviderFailureKind.PROVIDER_AUTH
    retryable = False

    def __init__(self, *, provider: str, model: str) -> None:
        super().__init__(f"{provider} did not accept the API key.", provider=provider, model=model)


class ProviderQuotaError(ProviderFailureError):
    """The key has run into the provider's rate limit, quota or credit balance."""

    kind = ProviderFailureKind.PROVIDER_QUOTA

    def __init__(self, *, provider: str, model: str) -> None:
        super().__init__(
            f"You've reached {provider}'s usage limit for this API key. Wait a minute and "
            "try again, or check your plan with them.",
            provider=provider,
            model=model,
        )


class ProviderUnreachableError(ProviderFailureError):
    """The request never got an answer: no connection, DNS, TLS, or a timeout."""

    kind = ProviderFailureKind.PROVIDER_UNREACHABLE

    def __init__(self, *, provider: str, model: str) -> None:
        super().__init__(
            f"Couldn't get an answer from {provider}. Check your internet connection and "
            "try again.",
            provider=provider,
            model=model,
        )


class ProviderOutageError(ProviderFailureError):
    """The provider answered with a server-side failure, or said it was overloaded."""

    kind = ProviderFailureKind.PROVIDER_OUTAGE

    def __init__(self, *, provider: str, model: str) -> None:
        super().__init__(
            f"{provider}'s service is having problems right now. This isn't caused by "
            "your settings. Try again in a few minutes.",
            provider=provider,
            model=model,
        )


class ProviderResponseError(ProviderFailureError):
    """The provider answered with something none of the other kinds recognises.

    Not a guess at one of them: a response this code cannot place is said to be exactly
    that (P6). The message does not send the user to a log: no head configures one, so there
    is nowhere to look. It names the two settings a beginner can check instead.
    """

    kind = ProviderFailureKind.PROVIDER_ERROR

    def __init__(self, *, provider: str, model: str) -> None:
        super().__init__(
            f"{provider} returned an error UClone-X didn't recognise. If it happens again, "
            "check the model name and any custom endpoint.",
            provider=provider,
            model=model,
        )


class ImageNotReturnedError(ProviderFailureError):
    """A provider asked for a picture answered without one.

    Gemini's image models can answer a `generateContent` that asked for `IMAGE` output with
    text only, or with no parts at all when they decline the description. That is not a
    picture, and no placeholder stands in for it (P6). Not retryable as it is: the same
    description is likely to be declined again.
    """

    kind = ProviderFailureKind.PROVIDER_ERROR
    retryable = False

    def __init__(self, *, provider: str, model: str) -> None:
        super().__init__(
            f"{provider} answered without a picture. It may have declined the description; "
            "try describing the picture differently.",
            provider=provider,
            model=model,
        )


class LLMStreamInterruptedError(LLMError):
    """A streamed model step stopped before it finished, so the turn fails (#938).

    Raised by `TurnExecutor.invoke_model` (`agent/turn_executor.py`) when iterating
    `llm.stream` raises after the call was made -- before the first chunk or after some. The cause is chained (`__cause__`) and
    named in the message, and is usually the provider's (`LLMProviderError`); a listener that
    raises while a delta is delivered interrupts the stream the same way.

    It is not retried. The step used to be re-requested through `llm.generate` under
    `PRIMARY` provenance, from an implicit `except` branch: paid twice, the reply replayed to
    the listener after the partial one, and nothing in band saying so. P6 permits a retry
    only when it is declared and attributed, and a declared one would also need a restart
    signal no head understands; failing the turn needs neither. The partial reply and any
    tool call the stream carried are discarded unexecuted, and the partial stream's usage is
    booked before this is raised (room UI design document §6.7 [#938]).

    Not an `LLMProviderError`, because the interruption is not always the provider's -- the
    chained cause says whose it was.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str | None,
        chunks_received: int,
        discarded_tool_calls: int,
        partial_content: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.chunks_received = chunks_received
        self.discarded_tool_calls = discarded_tool_calls
        self.partial_content = partial_content


class MalformedToolCallArgumentsError(LLMError):
    """An LLM response contained tool-call arguments that could not be parsed as valid JSON."""


class UnmappableChatMessageError(LLMError):
    """A `ChatMessage` has no faithful representation in a provider's request shape.

    Raised while building an outbound payload, before any network call, when a field
    the provider requires is absent or when the model permits a value the wire format
    cannot carry. It exists so that such a message fails here rather than being
    defaulted: substituting a placeholder name attributes a result to a tool that never
    ran, and substituting `""` for `None` makes "no recorded value" and "the empty
    value" arrive at the provider as the same bytes. Both are the P6 silent
    substitution, and both are unobservable downstream once the request has been sent.

    It is not an `LLMProviderError`: no provider was reached and none returned an
    error. The offending value is named in the message so the caller can identify the
    message it built without re-deriving it from the payload.
    """


class LLMCredentialsNotConfiguredError(LLMError):
    """A keyed provider connector was constructed with no API key available.

    Raised from the connector's `__init__`, not from `generate`/`stream`, and that
    placement is the whole point. Substituting `""` for an absent key — as
    `api_key or os.getenv(...) or ""` did, then `{"x-api-key": self.api_key or ""}`
    again at send time — produces a syntactically valid request carrying an empty
    credential, so a **configuration** defect arrives as a provider-side `401` on the
    first billed call. That moves the error away from its cause and dresses it as a
    transport fault a caller's retry or failover path may legitimately re-attempt
    (P6, #385).

    It is not an `LLMProviderError`: no provider was reached, nothing is retryable, and
    the repair is to supply a key rather than to try again. The environment variable
    that would have supplied it is named in the message.
    """


class LLMProviderNotConfiguredError(LLMError):
    """Nothing in the environment named an LLM provider, so no connector was built.

    Raised by `create_llm_connector` when no provider argument is given, `LLM_PROVIDER` is
    unset, no credential variable is present, and no Ollama endpoint variable is set.

    The value of raising here is that the alternative is worse than an error. Building a
    default `OllamaConnector` against `http://localhost:11434` succeeds — that connector has
    no credential to validate — so an **unconfigured installation** surfaces later as a
    refused TCP connection on the first turn. That message is a true statement about a socket
    and a false account of what is wrong: the problem is that no provider was chosen, not
    that a particular port is closed. Under P6 the substituted default is the defect; under
    P0 *Complete Default Composition* the person least able to translate "connection refused
    to localhost:11434" into "configure a provider" is exactly the first-time user.

    It is not an `LLMProviderError`: no provider was reached, none returned an error, and
    nothing is retryable. It is a sibling of `LLMCredentialsNotConfiguredError`, which covers
    the adjacent case where a provider *was* named and its key was missing. The message names
    the ways to configure one so the repair does not require reading this file.
    """


class LLMModelNotConfiguredError(LLMError):
    """A request reached a connector that names no model, and the connector was given none.

    Raised before any network call. A connector used to fill in a model id written in its
    source (``gemini-1.5-pro``, ``gpt-4o``, ``claude-3-5-sonnet``) when neither the request
    nor anything else named one, and those ids went stale: a person who had chosen Gemini
    Flash in Settings had a retired model called on their behalf, and was told only that
    the model was unavailable. A model id in source is a substituted value (P6); the model
    comes from the request, the connector's configured model, or the settings file, and
    when none of them names one the request is refused here, in plain words.

    It is not an `LLMProviderError`: no provider was reached, and nothing is retryable.
    ``provider`` is the provider's display name, for a head to say which one needs a model.
    The sentence says what is missing and leaves where to choose one to the head, which
    adds its own remedy: a flag on the command line, Settings in the dashboard and for an
    ACP client.
    """

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(f"No model is chosen for {provider}.")


class LLMConnectorNotConfiguredError(LLMError):
    """A reasoning turn was requested on a component with no LLM connector wired.

    Principle 6 forbids answering with a substituted value, and this failure has no
    declared recovery: nothing was attempted, so there is nothing to retry or fail over
    to. It therefore propagates. It is a configuration defect rather than a runtime
    fault, so it is raised as a precondition, before any state transition or history
    mutation, leaving the component exactly as it was found.
    """


class EmbeddingError(LLMError):
    """An embedding provider returned an error, was unreachable, or answered unusably.

    Declared so that the one thing a retrieval stack must never do has a name. The prior
    art this port draws from answered a failed embedding call with a zero vector of the
    right length; every cosine similarity against it is then exactly 0, so the failure
    reaches the caller as "nothing in memory matched" -- a sentence about the corpus,
    produced by a transport error. Principle 6 forbids substituting a value the provider
    did not produce, and a zero vector is such a value. Embedding failures raise.
    """


class EmbeddingDimensionError(EmbeddingError):
    """An embedding came back with a width other than the one the embedder declares.

    A vector store indexes on a fixed width, so a short or long vector is not a degraded
    result that ranks poorly -- it is either a crash inside the similarity loop or, worse,
    a silent truncation. The mismatch is reported where it can still name the model that
    produced it.
    """

    def __init__(self, model: str, expected: int, actual: int) -> None:
        super().__init__(
            f"Embedder '{model}' declares {expected} dimensions but returned a vector of "
            f"width {actual}. The declared width is what the vector store indexes on, so "
            f"this is refused rather than padded or truncated."
        )
        self.model = model
        self.expected = expected
        self.actual = actual


class EmbedderNotConfiguredError(LLMError):
    """A semantic operation was requested with no embedder wired.

    Follows `LLMConnectorNotConfiguredError`: nothing was attempted, so there is nothing
    to retry or fail over to. Callers that can degrade to a non-semantic method must do so
    by *asking* whether an embedder is present and saying which method they used, never by
    catching this and presenting a lexical ranking as a semantic one.
    """


# --------------------------------------------------------------------------------------
# Tools & External Connectors (Principle 6)
# --------------------------------------------------------------------------------------


class ToolError(UCloneXError, RuntimeError):
    """Base error for tool and external connector failures."""


class ComfyUIError(ToolError):
    """Raised when a ComfyUI server interaction or workflow execution fails."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        node_errors: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.node_errors = node_errors


class ComfyUIExecutionError(ComfyUIError):
    """A queued workflow reached ComfyUI and failed while executing.

    Distinct from workflow rejection at submission: the workflow was accepted by the server,
    but either failed during execution or completed without saving any image outputs.
    """


# --------------------------------------------------------------------------------------
# Execution isolation (Principle 3)
# --------------------------------------------------------------------------------------


class SandboxViolationError(UCloneXError):
    """An execution request breached its isolation contract."""


class PathTraversalError(SandboxViolationError):
    """A path resolved outside the workspace boundary."""


# --------------------------------------------------------------------------------------
# Skills (Principle 9)
# --------------------------------------------------------------------------------------


class SkillAuditError(UCloneXError):
    """A skill could not be audited."""


class SkillNotApprovedError(SkillAuditError):
    """Registration was attempted for a skill that no passing audit covers."""


# --------------------------------------------------------------------------------------
# Ontology (Principle 7)
# --------------------------------------------------------------------------------------


class OntologyError(UCloneXError):
    """Base error for ontology subsystem failures."""


class OntologyViolationError(OntologyError):
    """A payload failed compiled ontology validation."""


class OntologyContradictionError(OntologyViolationError):
    """An asserted or induced term contradicts existing ontology invariants."""


class OntologyRetractionBlockedError(OntologyViolationError):
    """Retraction of a term is blocked because asserted dependents exist."""


class OntologyPromotionError(OntologyViolationError):
    """Candidate term does not satisfy promotion criteria (observations/contradictions)."""


class PromotionCriteriaNotMetError(OntologyPromotionError):
    """Candidate term does not satisfy promotion criteria (observations/sessions/contradictions)."""


class OntologyHashMismatchError(OntologyViolationError):
    """Validation attempted against a mismatched content_hash snapshot."""


class UnparseableDirectiveError(OntologyError):
    """A natural language or structured directive could not be parsed into an ontology element.

    Principle 6 (Fail-Fast & Zero Silent Fallbacks) requires uninterpretable directives
    to raise an explicit error rather than silently degrading into junk concepts.
    """

    def __init__(
        self,
        text: str,
        message: str | None = None,
        expected_forms: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        self.text = text
        self.expected_forms = tuple(
            expected_forms
            or (
                "Concept: 'concept <Name> [extends <Parent>] [with attributes <k:v, ...>] [requires <f1, f2>]'",
                "Concept definition: '<Name> is a <Parent>' or 'define concept <Name>'",
                "Relation triplet: '<Source> -> <predicate> -> <Target>'",
                "Invariant rule: 'all <X> must <Y>', 'whenever <X>, <Y>', '<X> requires <Y>'",
                "Axiom rule: 'rule <Name>: <subject> <predicate> == <value>'",
            )
        )
        if message is None:
            forms_str = "\n  - " + "\n  - ".join(self.expected_forms)
            message = (
                f"Cannot parse directive '{text}' into a structured ontology element. "
                f"Supported patterns include:{forms_str}"
            )
        super().__init__(message)


# --------------------------------------------------------------------------------------
# A2A (the A2A protocol specification's error table)
# --------------------------------------------------------------------------------------


class A2AError(UCloneXError):
    """Base for the A2A error types, carrying their canonical code mappings.

    Subclasses set the three class attributes so any binding — JSON-RPC, gRPC,
    HTTP+JSON or the in-process fastpath — can map a raised error without maintaining
    its own table.
    """

    jsonrpc_code: int = -32603
    grpc_status: str = "INTERNAL"
    http_status: int = 500


class TaskNotFoundError(A2AError):
    """The referenced task does not exist."""

    jsonrpc_code = -32001
    grpc_status = "NOT_FOUND"
    http_status = 404


class TaskNotCancelableError(A2AError):
    """The task is in a state that cannot be canceled."""

    jsonrpc_code = -32002
    grpc_status = "FAILED_PRECONDITION"
    http_status = 400


class PushNotificationNotSupportedError(A2AError):
    """Push notification configuration is not offered by this agent."""

    jsonrpc_code = -32003
    grpc_status = "FAILED_PRECONDITION"
    http_status = 400


class UnsupportedOperationError(A2AError):
    """The requested operation is not supported, or not in the task's current state."""

    jsonrpc_code = -32004
    grpc_status = "FAILED_PRECONDITION"
    http_status = 400


class ContentTypeNotSupportedError(A2AError):
    """No requested media type is supported."""

    jsonrpc_code = -32005
    grpc_status = "INVALID_ARGUMENT"
    http_status = 400


class InvalidAgentResponseError(A2AError):
    """A peer returned a response that does not conform to the protocol."""

    jsonrpc_code = -32006
    grpc_status = "INTERNAL"
    http_status = 500


class ExtendedAgentCardNotConfiguredError(A2AError):
    """No extended Agent Card is configured."""

    jsonrpc_code = -32007
    grpc_status = "FAILED_PRECONDITION"
    http_status = 400


class ExtensionSupportRequiredError(A2AError):
    """A required protocol extension is not supported by the caller."""

    jsonrpc_code = -32008
    grpc_status = "FAILED_PRECONDITION"
    http_status = 400


class VersionNotSupportedError(A2AError):
    """The requested protocol version is not supported."""

    jsonrpc_code = -32009
    grpc_status = "FAILED_PRECONDITION"
    http_status = 400


# --------------------------------------------------------------------------------------
# Telemetry (Principle 6, FR-10.3)
# --------------------------------------------------------------------------------------


class TelemetryError(UCloneXError, RuntimeError):
    """Base error for telemetry subsystem failures."""


class TelemetryExportError(TelemetryError):
    """Telemetry span or metric export to remote collector failed."""


# --------------------------------------------------------------------------------------
# uclone2 / Google ADK integration adapters (Principle 5, Principle 6, Issue #367)
# --------------------------------------------------------------------------------------


class ADKConversionError(UCloneXError):
    """A `Content` <-> `ChatMessage` conversion had no faithful result.

    Raised by `uclone_x.adapters.uclone2.adk_content` instead of substituting a default.
    The two shapes are not isomorphic, so some inputs genuinely have no counterpart; P6
    makes that a raise rather than a coerced approximation, and every message below
    carries the offending value so a caller can see what could not be mapped.
    """


class ADKUnmappedRoleError(ADKConversionError):
    """A role has no counterpart on the other side of the conversion.

    Either an ADK `Content.role` outside `{"user", "model"}` (including `None`), or
    `MessageRole.SYSTEM`, which ADK carries out of band in
    `GenerateContentConfig.system_instruction` rather than as a `Content` role.
    """


class ADKMalformedContentError(ADKConversionError):
    """An ADK `Content` has no single-`ChatMessage` representation.

    A part mix that cannot be expressed in one message (a tool result beside text, more
    than one tool result, a `function_call` under `"user"`), or a `FunctionCall` with no
    `id` — which `ToolCallRequest` requires and which cannot be synthesised without
    breaking correlation with the matching `FunctionResponse`.
    """


class ADKUnrepresentableMessageError(ADKConversionError):
    """A populated `ChatMessage` field has no ADK slot in the role being emitted.

    `name` or `tool_call_id` outside a `FunctionResponse`, or `tool_calls` on a role
    other than `ASSISTANT`. The field is not silently dropped.
    """


class RoomError(UCloneXError):
    """Base class for multi-agent room failures."""


class SpeakerSelectionError(RoomError):
    """A speaker selector could not reach a judgement.

    The P6 boundary of `SpeakerSelectorProtocol.select`. A selector has three legitimate
    answers — name a speaker, decide on silence, abstain to the next selector — and a
    failure is none of them. Returning silence would claim a judgement the selector did
    not make; returning abstain would hide a dead selector behind whichever one runs next,
    and in both cases the room simply goes quiet, which is what a working quiet room also
    looks like. So a provider failure, an unparseable response or missing configuration
    raises here and the orchestrator records it in the transcript.
    """


class UnknownRoomParticipantError(RoomError):
    """A message addressed a participant the room does not have.

    A user input fault, not a selector fault, and kept distinct from
    `SpeakerSelectionError` for that reason: the remedy is for the human to retype the
    name, so the roster belongs in the message. It is an error rather than a fallback
    because the alternative — routing `@phantm` to whichever agent is default — answers
    in the voice of an agent the user did not address, and the user has no way to see
    that the address was dropped.
    """


class ParticipantNotResolvableError(RoomError):
    """No live agent can be produced for a participant.

    An unknown agent id, or a human participant, which has no agent behind it.
    """


class RoomIdError(RoomError, PathTraversalError):
    """A room id cannot be used to address a record.

    The room-facing half of the path guard `RoomStore.room_path` reuses. The check stays
    shared — one traversal control with two callers is the point — but its wording is not:
    `resolve_session_path` speaks of a *session*, which is true of the control and
    meaningless to someone running `ucx room`, who has no session in view and was asking
    about a room.

    **Both bases, deliberately.** It is a room error, so a room's caller can catch it with
    the rest; and it is still a `PathTraversalError`, because it is still a containment
    refusal and anything treating that class as a security boundary must keep catching it.
    Narrowing it to `RoomError` alone would have made a traversal attempt invisible to
    every such handler, which is a worse outcome than the wording this fixes.
    """


class RoomNotFoundError(RoomError):
    """No room exists under the given id.

    Distinct from a store that cannot be reached: a backend able to tell "no record" from
    "storage unavailable" must raise its own transport error for the second, because
    treating an outage as an absent room seeds an empty one over a live conversation.
    """


class RoomAlreadyExistsError(RoomError):
    """A room was created under an id the store already holds.

    Refused rather than merged or overwritten. A `create` that found an existing record
    and returned it would make "the room I just made is empty" and "the room I just
    reopened is not" the same call, and an overwrite would destroy a conversation because
    two people chose the same name.
    """


class SecondHumanInRoomError(RoomError):
    """A second human was offered a seat in a room that serves one.

    Not a capacity limit and not a licence: it is the boundary of what the turn loop can
    actually serialise. `RoomOrchestrator.post` loads the room, appends and saves under the
    store's compare-and-swap, so two humans posting at once means the later write is
    refused as stale and that person's message is simply lost to them — a correct refusal
    with no outcome a chat surface can render. Nothing else in the room supplies the parts
    a shared room needs either: `RoomState.last_seen_seq` is an agent's mark, so a second
    human has no notion of what they have read, and the interjection check answers "has the
    human spoken?" for one floor rather than arbitrating between two.

    Refused at the roster, where the reason can still be given, rather than at the post that
    would collide with it several steps later. A room whose human has left can seat another.
    """


class NothingToRetryError(RoomError):
    """A retry was asked for in a room whose last utterance did not fail.

    Distinct from `RoomNotFoundError` and from `UnknownRoomParticipantError` because the
    remedy is different in kind: there is nothing wrong with the room, and nothing for the
    caller to fix — the turn it wanted re-run either succeeded or was superseded by
    somebody speaking since. Raised rather than returned as an unchanged `RoomState`,
    which would make "retried and it worked" and "there was nothing to retry"
    indistinguishable to the caller, in the one command a user reaches for precisely
    because they cannot tell what the room did.
    """


class HeadRoomWriteError(RoomError):
    """A post or retry was sent to a room a head keeps (#1885).

    A head's room (`ucx run`, `ucx loop`, ACP, A2A) has one writer, the head: it runs the
    clone on the seat's session and records each turn. A second surface driving a turn
    there would be a second writer of that session. The message is plain, because it
    reaches the person through the app and `ucx room say`.
    """


#: Why a conversation's workspace was not changed: the head words it from this.
WorkspaceRefusalCode = Literal["not_a_folder", "app_state", "story_open"]


class RoomWorkspaceRefusedError(RoomError):
    """A conversation's workspace was not changed (clone-data-scopes §3.6).

    The message is the English sentence; `code` says which refusal it is, so a head words
    it in the reader's language: the path is not an existing folder (`not_a_folder`), it
    holds the app's own state (`app_state`), or a story is open (`story_open`).
    """

    def __init__(self, message: str, code: WorkspaceRefusalCode) -> None:
        super().__init__(message)
        self.code: WorkspaceRefusalCode = code


class TurnNotStartedError(RoomError):
    """A turn was refused because the room could not record that it was starting (#1366).

    The room counts a turn as started in a save of its own before the turn runs any tool,
    so a turn whose result is later lost still shows as a gap rather than as nothing. When
    that first save fails the turn is not run at all: running it unrecorded is exactly the
    loss the count exists to expose. Nothing ran, so there is nothing to account for.
    """


class TurnNotLandedError(RoomError):
    """A turn ran, and the room could not save it (#1495).

    The seat has already been put back to where it was before the turn, so the seat and
    the transcript agree that the turn did not happen. The message is written for the
    person reading the conversation: it names the seat and that the reply could not be saved,
    pointing to the server log for the reason (#1503). The store's own failure -- an
    `OSError` carrying a path, say -- is chained on `__cause__` and logged, never written
    into the message.
    """


class UnreadableRoomRecordError(UCloneXError):
    """A stored room record exists, and this build cannot load it (#1411).

    Raised by `RoomStore.load` in place of the parser's own `ValidationError`, which
    carries the offending field paths or the JSON decoder's complaint. Kept apart from that
    class because a caller cannot tell the two apart otherwise: `ui/rooms._http_error`
    answers a `ValidationError` as the caller's malformed request, with its field dump as
    the reason, and a record the store wrote earlier is not the caller's request. It is a
    fault in the store, so the route logs it and answers in words of its own.

    **Not a `RoomError`, deliberately.** A `RoomError`'s message is a refusal written for
    the person who caused it, and both the route and `reader_facing_reason` pass it through
    as copy. Nobody caused this one, and the parser's cause is chained on `__cause__` for
    the log, not written into the message.
    """

    def __init__(self, room_id: str) -> None:
        super().__init__(f"Room {room_id!r} is stored in a shape this build will not load.")
        self.room_id = room_id


class StaleRoomWriteError(RoomError):
    """A room write was refused because the record moved on since it was read.

    The same optimistic-concurrency refusal as `StaleSessionWriteError`, applied to the
    transcript. A room has one writer by design — its orchestrator — so this firing means
    two orchestrators are driving one room, which would interleave utterances and lose
    one side's turns rather than merely conflicting on a field.
    """
