"""The session log: every message that entered a session's history, typed and append-only (#1443).

`SessionState.messages` is what the next request carries. It shrinks: a retry replaces a
rejected answer, a refused step is withheld, a rollback restores a checkpoint, compaction
folds turns into a ledger. The log is the record none of those touch. Each entry names
one message that was in the history at some point, by the SHA-256 of its canonical JSON,
and the message itself is stored once as a body in the session's context body store
(#1421) under that digest -- so the record grows by a few fields per message, not by the
message.

A tool result whose history form is an excerpt of a stored result (#1422) also names the
`tr_` handle of the full body in the tool-result blob store, so the log holds every tool
output in full even where the history keeps only its head and tail.

Kinds (#1849). A tool result is logged by what produced it: `subagent` for a delegated
sub-agent's result, `retrieval` for a search tool's hits (`SUBAGENT_TOOLS`,
`RETRIEVAL_TOOLS`), `tool_result` for any other. The memory facts a turn recalled are
not a message: they are sent in that turn's `[Turn Context]` tail, never in the history.
They are logged once per turn as a `memory` entry whose body is the section's text
(`logged_text`), with the `tr_` handle `tool_result_read` reads it by. Logging them
changes no request.

The writer is `_LiveSession` in `agent/session_lifecycle.py`, the one place a session's
history is held (`log_history` for messages, `log_entry` for the rest); a loaded record
is reconciled against its messages there too.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from uclone_x.core.secrets import redact_credentials
from uclone_x.core.tool_results import handle_in
from uclone_x.llm.models import ChatMessage, MessageRole

__all__ = [
    "RETRIEVAL_TOOLS",
    "SUBAGENT_TOOLS",
    "LoggedMessage",
    "SessionLogEntry",
    "SessionLogKind",
    "SessionLogProvenance",
    "history_entry_ids",
    "log_kind",
    "logged_message",
    "logged_text",
    "new_entry",
]


class SessionLogKind(StrEnum):
    """What a log entry records.

    `RETRIEVAL` and `SUBAGENT` are tool results, told apart by the tool that produced them
    (`log_kind`). `MEMORY` is the memory section a turn recalled, which is not a message
    (`logged_text`, #1849).
    """

    SYSTEM = "system"
    UTTERANCE = "utterance"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SUMMARY = "summary"
    RETRIEVAL = "retrieval"
    MEMORY = "memory"
    SUBAGENT = "subagent"


class SessionLogProvenance(StrEnum):
    """How an entry came to be in the log.

    `RECORDED`: written when the message entered the history. `MIGRATED`: backfilled when a
    record written before the log existed was loaded -- the message is real, but when it
    entered and what it displaced are not known, so its `turn` is `None`.
    """

    RECORDED = "recorded"
    MIGRATED = "migrated"


class SessionLogEntry(BaseModel):
    """One message that entered a session's history."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: str = Field(description="`e<n>`, the entry's position in the log.")
    kind: SessionLogKind
    turn: int | None = Field(
        description="The session's turn counter when the message entered the history; "
        "`None` for a migrated entry."
    )
    digest: str = Field(
        description="SHA-256 of the message's canonical JSON, the name its body is stored "
        "under in the session's context body store."
    )
    size: int = Field(description="Bytes of the stored body (UTF-8).")
    provenance: SessionLogProvenance
    blob: str | None = Field(
        default=None,
        description="For a tool result held as an excerpt: the `tr_` handle of the full "
        "result in the tool-result blob store.",
    )

    @field_validator("digest")
    @classmethod
    def _is_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("A log entry's digest is 64 lowercase hex characters (SHA-256).")
        return value


#: The tool whose result is a delegated sub-agent's answer (`tools/builtin/subagent.py`).
#: Its entry's handle, when the result was over the cap, reads the whole answer back.
SUBAGENT_TOOLS: Final = frozenset({"delegate_subagent"})

#: The tools whose results are retrieval hits: ranked or matched items of a store the
#: clone searched (`memory/tools.py`, `tools/builtin/web.py`, `tools/builtin/filesystem.py`).
#: Named here rather than read from the tools so this kernel module depends on none of
#: them; `tests/unit/test_context_state.py` checks the names against the tools.
RETRIEVAL_TOOLS: Final = frozenset({"query_memory_facts", "web_search", "file_search"})


def log_kind(message: ChatMessage) -> SessionLogKind:
    """The kind of entry `message` is logged as.

    A tool result is `subagent` or `retrieval` by the tool that produced it (#1849), and
    `tool_result` otherwise. The kind is a label on the entry: a request renders every
    tool result the same way, whatever its kind.
    """
    if message.role == MessageRole.SYSTEM:
        return SessionLogKind.SUMMARY if message.compaction_ledger else SessionLogKind.SYSTEM
    if message.role == MessageRole.TOOL:
        if message.name in SUBAGENT_TOOLS:
            return SessionLogKind.SUBAGENT
        if message.name in RETRIEVAL_TOOLS:
            return SessionLogKind.RETRIEVAL
        return SessionLogKind.TOOL_RESULT
    if message.role == MessageRole.ASSISTANT and message.tool_calls:
        return SessionLogKind.TOOL_CALL
    return SessionLogKind.UTTERANCE


@dataclass(frozen=True, slots=True)
class LoggedMessage:
    """A message rendered for the log: its body, the body's digest, and its entry fields."""

    body: str
    digest: str
    kind: SessionLogKind
    blob: str | None


def logged_message(message: ChatMessage) -> LoggedMessage:
    """Render `message` as the body the log stores and the digest that names it.

    The body is the message's canonical JSON -- every field, keys sorted -- so it is the
    message, not a rendering of it, and `ChatMessage.model_validate_json(body)` gives it
    back. Callers pass messages that are already redacted, as everything in a history is.
    """
    body = json.dumps(
        message.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    blob = handle_in(message.content) if message.role == MessageRole.TOOL else None
    return LoggedMessage(
        body=body,
        digest=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        kind=log_kind(message),
        blob=blob,
    )


def logged_text(kind: SessionLogKind, text: str, *, blob: str | None) -> LoggedMessage:
    """Render `text`, which is not a message, as the body the log stores (#1849).

    For what a request sent outside the history, such as the recalled memory section of
    a turn's `[Turn Context]`. The body is the text itself, redacted, so its digest is
    never that of a message's canonical JSON and `history_entry_ids` never matches it to
    a history message. `blob` is the `tr_` handle the same text is stored under in the
    tool-result store, when it was.
    """
    body = redact_credentials(text)
    return LoggedMessage(
        body=body,
        digest=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        kind=kind,
        blob=blob,
    )


def new_entry(
    position: int, rendered: LoggedMessage, *, turn: int | None, provenance: SessionLogProvenance
) -> SessionLogEntry:
    """The entry at `position` for `rendered`."""
    return SessionLogEntry(
        id=f"e{position}",
        kind=rendered.kind,
        turn=turn,
        digest=rendered.digest,
        size=len(rendered.body.encode("utf-8")),
        provenance=provenance,
        blob=rendered.blob,
    )


def history_entry_ids(log: Sequence[SessionLogEntry], digests: Sequence[str]) -> list[str | None]:
    """The log entry each of a history's messages is, by digest, or `None` for one it lacks.

    For a history read back with its log: the `n` occurrences of a digest in the history
    are the last `n` entries of that digest in the log, in order, since the history holds
    what entered it most recently. An occurrence beyond what the log has is `None` -- it
    has not been logged. A multiset match: two identical messages are two entries.
    """
    by_digest: dict[str, list[str]] = {}
    for entry in log:
        by_digest.setdefault(entry.digest, []).append(entry.id)
    wanted = Counter(digests)
    pools = {digest: by_digest.get(digest, [])[-count:][::-1] for digest, count in wanted.items()}
    return [pools[digest].pop() if pools[digest] else None for digest in digests]
